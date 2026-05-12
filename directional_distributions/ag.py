"""Angular Gaussian family: IAG, ESAG, and GAG distributions on S².

This module collects the three members of the angular Gaussian family,
obtained by projecting a trivariate normal N(mu, V) onto the sphere
via z -> z/||z||.  They form a nested hierarchy:

    IAG (3 params)  <  ESAG (5 params)  <  GAG (8 params)

References
----------
Paine et al. (2018), "An elliptically symmetric angular Gaussian
distribution", Stat Comput 28:689-697.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import (
    BaseDistribution,
    _apply_reduction,
    _construct_orthonormal_basis,
    _build_cholesky,
    make_grid,
)


_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# AG-family math utility
# ---------------------------------------------------------------------------

def _log_M2(alpha: Tensor) -> Tensor:
    """Compute log(M_2(alpha)) numerically stably.

    M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)

    where Phi is the standard normal CDF and phi is the standard normal PDF.

    For large negative alpha, direct computation suffers from catastrophic
    cancellation.  The negative-tail branch rewrites ``Phi`` with ``erfcx`` and
    computes the sensitive subtraction in float64:

        M_2(-x) = exp(-x^2/2) *
                  [0.5(1 + x^2) erfcx(x / sqrt(2)) - x / sqrt(2pi)]

    This avoids the slow ``torch.func`` batching fallback for
    ``torch.special.log_ndtr`` while retaining stable tail values.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (4).
    """
    # Direct computation (accurate for moderate alpha)
    alpha_direct = torch.clamp(alpha, min=-3.5)
    log_phi = -0.5 * alpha_direct ** 2 - 0.5 * _LOG_2PI
    phi = torch.exp(log_phi)
    Phi = 0.5 * (1.0 + torch.erf(alpha_direct / _SQRT_2))
    M2_direct = (1.0 + alpha_direct ** 2) * Phi + alpha_direct * phi

    alpha_hi = alpha.to(torch.float64)
    x_hi = torch.clamp(-alpha_hi, min=0.0)
    inner_hi = (
        0.5 * (1.0 + x_hi ** 2) * torch.special.erfcx(x_hi / _SQRT_2)
        - x_hi / _SQRT_2PI
    )
    M2_tail = (
        -0.5 * x_hi ** 2
        + torch.log(torch.clamp(inner_hi, min=1e-300))
    ).to(alpha.dtype)

    # Use direct for alpha >= -3.5, stable form for alpha < -3.5.
    # The direct branch suffers catastrophic cancellation in M2_direct for
    # alpha < ~-3.8 (float32), while the tail branch (computed in float64)
    # is accurate for all alpha < 0.
    return torch.where(
        alpha >= -3.5,
        torch.log(torch.clamp(M2_direct, min=1e-40)),
        M2_tail,
    )


def _ag_log_kernel(Q: Tensor, S: Tensor, T: Tensor) -> Tensor:
    """Angular Gaussian log-density without the ``-log(2pi)`` constant."""
    Q_safe = torch.clamp(Q, min=1e-8)
    return -1.5 * torch.log(Q_safe) - 0.5 * (S - T ** 2) + _log_M2(T)


def _safe_unit_vector(vec: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize vectors, using +z when the direction is undefined."""
    norm = vec.norm(p=2, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vec)
    fallback[..., 2] = 1.0
    return torch.where(norm > eps, vec / norm.clamp_min(eps), fallback)


def _sphere_retract(y: Tensor, e1: Tensor, e2: Tensor, eta: Tensor) -> Tensor:
    """Exponential-map retraction from tangent coordinates."""
    tangent = eta[:, 0:1] * e1 + eta[:, 1:2] * e2
    theta = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / theta.clamp_min(1e-30)
    y_next = torch.cos(theta) * y + torch.sin(theta) * direction
    return _safe_unit_vector(y_next)


# ---------------------------------------------------------------------------
# Mode-finding on S^2 (Riemannian Newton with backtracking line search)
# ---------------------------------------------------------------------------

def _multistart_mode(
    y_inits: Tensor,
    log_p_one,
    params: tuple,
    num_iters: int = 20,
    trust_radius: float = 0.5,
    ls_iters: int = 10,
) -> Tensor:
    """Run Newton-on-sphere from multiple initial points; keep the per-batch best.

    Args:
        y_inits: [K, B, 3] candidate starting directions (each unit-norm).
        log_p_one, params, num_iters, trust_radius, ls_iters: see
            :func:`_newton_mode_on_sphere`.

    Returns:
        [B, 3] best mode found across the K starts.
    """
    from torch.func import vmap

    K, B, _ = y_inits.shape
    flat_init = y_inits.reshape(K * B, 3)
    flat_params = tuple(
        p.unsqueeze(0).expand((K,) + p.shape).reshape((K * B,) + p.shape[1:])
        for p in params
    )

    flat_modes = _newton_mode_on_sphere(
        flat_init, log_p_one, flat_params,
        num_iters=num_iters, trust_radius=trust_radius, ls_iters=ls_iters,
    )

    log_p_fn = vmap(log_p_one, in_dims=(0,) + (0,) * len(params))
    flat_log_p = log_p_fn(flat_modes, *flat_params)        # [K * B]
    log_p = flat_log_p.reshape(K, B)
    modes = flat_modes.reshape(K, B, 3)
    best = log_p.argmax(dim=0)                              # [B]
    return modes[best, torch.arange(B, device=modes.device)]


def _newton_mode_on_sphere(
    y_init: Tensor,
    log_p_one,
    params: tuple,
    num_iters: int = 20,
    trust_radius: float = 0.5,
    ls_iters: int = 10,
    grad_tol: float = 1e-5,
) -> Tensor:
    """Maximize a per-sample log-density on S^2 by Newton's method.

    Args:
        y_init: [B, 3] starting points (assumed unit-norm).
        log_p_one: callable mapping a single sample's (y, *params) to a scalar
            log-density. Must be twice-differentiable through ``torch.func``.
        params: tuple of [B, ...] tensors carrying per-sample parameters.
        num_iters: Maximum Newton iterations.
        trust_radius: Maximum angular step per iteration, in radians.
        ls_iters: Maximum backtracking halvings per Newton step.

    Returns:
        [B, 3] unit-norm MAP directions (detached).
    """
    from torch.func import grad as _grad, hessian, vmap

    in_dims = (0,) + (0,) * len(params)
    log_p_fn = vmap(log_p_one, in_dims=in_dims)
    grad_fn = vmap(_grad(log_p_one), in_dims=in_dims)
    hess_fn = vmap(hessian(log_p_one), in_dims=in_dims)

    y = _safe_unit_vector(y_init.detach().clone())
    eye2 = torch.eye(2, dtype=y.dtype, device=y.device)

    # Per-sample early stopping: shrink the working batch each iteration by
    # dropping samples that are already stationary (g≈0 with neg-def Hessian)
    # or "stuck" (LS found no improving step). The autograd grad/hess passes,
    # which dominate cost, then run only over the still-active subset.
    active = torch.arange(y.shape[0], device=y.device)
    y_a = y
    params_a = params

    for _ in range(num_iters):
        if active.numel() == 0:
            break

        g = grad_fn(y_a, *params_a)                                # [B_a, 3]
        H_eu = hess_fn(y_a, *params_a)                             # [B_a, 3, 3]

        e1, e2 = _construct_orthonormal_basis(y_a)                 # [B_a, 3] each
        E = torch.stack([e1, e2], dim=-1)                          # [B_a, 3, 2]

        # Riemannian Hessian: H_R[j,k] = e_j^T H_eu e_k - (g . y) delta_jk
        g_R = torch.einsum('bi,bik->bk', g, E)                     # [B_a, 2]
        H_R = torch.einsum('bim,bij,bjn->bmn', E, H_eu, E)         # [B_a, 2, 2]
        g_dot_y = (g * y_a).sum(dim=-1)
        H_R = H_R - g_dot_y[:, None, None] * eye2

        # 2x2 spectrum of H_R: tr = lam_+ + lam_-, det = lam_+ * lam_-.
        a_h = H_R[:, 0, 0]
        b_h = H_R[:, 0, 1]
        c_h = H_R[:, 1, 1]
        tr_H = a_h + c_h
        det_H = a_h * c_h - b_h * b_h
        disc = (tr_H * tr_H - 4.0 * det_H).clamp_min(0.0).sqrt()
        lam_max = 0.5 * (tr_H + disc)
        neg_def = (det_H > 1e-12) & (tr_H < 0)
        g_norm = g_R.norm(dim=-1)

        # First-order stationary and not a saddle => keep the current y.
        stationary = (g_norm <= grad_tol) & (lam_max <= 1e-8)
        if stationary.all():
            # y is already up to date for the active samples (either via the
            # initial copy from y_init or via the post-step commit below from
            # the previous iteration), so no write-back is needed here -- and
            # attempting one on iter 0 would alias y[active] with y_a.
            break

        # Newton step (uphill when H_R is neg-def).  Solve only the safe
        # negative-definite blocks so flat/uniform cases cannot raise here.
        eta_newton = torch.zeros_like(g_R)
        if neg_def.any():
            eta_newton[neg_def] = torch.linalg.solve(
                H_R[neg_def], -g_R[neg_def, :, None]
            ).squeeze(-1)

        # Eigenvector v_+ for lam_max (used to escape saddles).
        v_a = torch.stack([b_h, lam_max - a_h], dim=-1)
        v_b = torch.stack([lam_max - c_h, b_h], dim=-1)
        v_pos = torch.where(v_a.norm(dim=-1, keepdim=True) > 1e-10, v_a, v_b)
        v_pos = F.normalize(v_pos, p=2, dim=-1)
        fallback_2d = torch.zeros_like(v_pos)
        fallback_2d[..., 0] = 1.0
        v_pos = torch.where(v_pos.norm(dim=-1, keepdim=True) > 1e-12, v_pos, fallback_2d)

        # Step strategy:
        #   - neg-def Hessian: Newton step
        #   - indefinite (lam_max > 0): walk along v_+ (the positive-curvature
        #     direction) at the trust radius -- this escapes saddles even
        #     where the gradient vanishes exactly
        #   - otherwise (e.g. positive semi-def): plain gradient ascent
        eta_indef = trust_radius * v_pos
        eta = torch.where(
            neg_def[:, None],
            eta_newton,
            torch.where((lam_max > 1e-8)[:, None], eta_indef, g_R),
        )
        eta = torch.where(stationary[:, None], torch.zeros_like(eta), eta)

        eta_norm = eta.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        eta = eta * torch.clamp(trust_radius / eta_norm, max=1.0)

        log_p_curr = log_p_fn(y_a, *params_a)

        # Probe both signs only for curvature-based saddle escapes.  Newton and
        # gradient steps already carry a meaningful ascent orientation.
        sign_probe = (~neg_def) & (lam_max > 1e-8)
        if sign_probe.any():
            y_plus = _sphere_retract(y_a, e1, e2, eta)
            y_minus = _sphere_retract(y_a, e1, e2, -eta)
            lp_plus = log_p_fn(y_plus, *params_a)
            lp_minus = log_p_fn(y_minus, *params_a)
            flip = sign_probe & (lp_minus > lp_plus)
            eta = torch.where(flip[:, None], -eta, eta)

        # Per-batch backtracking line search.
        B_a = y_a.shape[0]
        alpha = torch.ones(B_a, 1, dtype=y_a.dtype, device=y_a.device)
        y_next = y_a.clone()
        done = torch.zeros(B_a, dtype=torch.bool, device=y_a.device)

        for _ls in range(ls_iters):
            step = alpha * eta
            y_try = _sphere_retract(y_a, e1, e2, step)
            log_p_try = log_p_fn(y_try, *params_a)
            accept = (log_p_try > log_p_curr + 1e-12) & ~done
            y_next = torch.where(accept[:, None], y_try, y_next)
            done = done | accept
            if done.all():
                break
            alpha = torch.where(done[:, None], alpha, alpha * 0.5)

        y_a = y_next
        # Scatter back into the full result buffer. ``y_next`` was built by
        # ``y_a.clone()`` inside the LS loop, so it does not alias ``y``.
        y[active] = y_a

        # Drop samples that converged this iter, OR that the line search could
        # not improve. The latter ("stuck") cannot make progress on the next
        # iter either -- the gradient/Hessian and so eta are unchanged when y
        # doesn't move -- so keeping them in the active set only burns autograd.
        keep = (~stationary) & done
        if not keep.all():
            active = active[keep]
            y_a = y_a[keep]
            params_a = tuple(p[keep] for p in params_a)

    return y


# ---------------------------------------------------------------------------
# Isotropic Angular Gaussian (IAG)
# ---------------------------------------------------------------------------

def iag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """
    Isotropic Angular Gaussian (IAG) negative log-likelihood loss.

    The IAG distribution is the angular Gaussian with V = I (identity covariance),
    making it rotationally symmetric about the mean direction. It is a special
    case of ESAG with gamma = (0, 0).

    The density is:
        f_IAG(y) = (1/2pi) * exp[0.5 * ((y.mu)^2 - ||mu||^2)] * M_2(y.mu)

    where M_2(alpha) = (1 + alpha^2)Phi(alpha) + alpha*phi(alpha), with phi and Phi being the standard
    normal PDF and CDF respectively.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 3] predicted mean vectors mu. The magnitude ||mu|| controls
              concentration (higher = more peaked), and mu/||mu|| is the mean direction.
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred  # [B, 3]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Compute terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||mu||^2 [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y.mu [B]

    # NLL = log(2pi) - angular-Gaussian log kernel with Q = 1
    nll = _LOG_2PI - _ag_log_kernel(torch.ones_like(y_dot_mu), mu_norm_sq, y_dot_mu)

    return _apply_reduction(nll, reduction)


class IAG(BaseDistribution):
    """Isotropic Angular Gaussian distribution on S^2.

    The IAG distribution is rotationally symmetric about the mean direction,
    with concentration controlled by ||mu||.

    The density is:
        f_IAG(y) = (1/2pi) exp[0.5((y.mu)^2 - ||mu||^2)] M_2(y.mu)

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 3] raw network output (the mean vector mu).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred)

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B]. Higher = more peaked."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_IAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred  # [B, 3]
        mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

        y_dot_mu = points @ mu.T  # [N, B]
        # log f = -log(2pi) + angular-Gaussian log kernel with Q = 1
        log_p = (
            -_LOG_2PI
            + _ag_log_kernel(torch.ones_like(y_dot_mu), mu_norm_sq[None, :], y_dot_mu)
        )

        return log_p.T  # [B, N]

    def mode(self) -> Tensor:
        """MAP direction on S^2.

        IAG is rotationally symmetric about mu, so the mode is exactly
        ``mu / ||mu||``.

        Returns:
            [B, 3] unit-norm MAP directions.
        """
        return self.mean_direction.detach()


# ---------------------------------------------------------------------------
# Elliptically Symmetric Angular Gaussian (ESAG)
# ---------------------------------------------------------------------------

def esag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """
    Elliptically Symmetric Angular Gaussian (ESAG) negative log-likelihood loss.

    The ESAG distribution has ellipse-like contours on the sphere, enabling
    modeling of anisotropic directional uncertainty. It generalizes IAG by
    adding shape parameters gamma = (gamma_1, gamma_2) that control the ellipticity.

    The density is:
        f_ESAG(y) = C_3 / (y'V^{-1}y)^(3/2) * exp[0.5*((y.mu)^2/(y'V^{-1}y) - ||mu||^2)]
                    * M_2(y.mu / sqrt(y'V^{-1}y))

    where V^{-1} is constructed from mu and gamma per Equation (18) of the paper.

    Reference: Paine et al. (2018), "An elliptically symmetric angular Gaussian
    distribution", Stat Comput 28:689-697.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = mu (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = gamma = (gamma_1, gamma_2) (shape parameters for ellipticity)
              Setting gamma = (0, 0) recovers the IAG distribution.
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred[:, :3]      # [B, 3]
    gamma1 = pred[:, 3]   # [B]
    gamma2 = pred[:, 4]   # [B]

    # Normalize y_true to ensure unit vectors
    y = F.normalize(y_true, p=2, dim=1)  # [B, 3]

    # Basic terms
    mu_norm_sq = (mu ** 2).sum(dim=1)  # ||mu||^2 [B]
    y_dot_mu = (y * mu).sum(dim=1)     # y.mu [B]

    # Construct orthonormal basis {xi_1, xi_2} perpendicular to mu
    xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

    # Projections of y onto the basis vectors
    a = (y * xi1).sum(dim=1)  # y.xi_1 [B]
    b = (y * xi2).sum(dim=1)  # y.xi_2 [B]

    # Compute y'V^{-1}y using Equation (18):
    # y'V^{-1}y = 1 + gamma_1(a^2 - b^2) + 2*gamma_2*a*b + (sqrt(1 + gamma_1^2 + gamma_2^2) - 1)(a^2 + b^2)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    y_Vinv_y = (1.0
                + gamma1 * (a ** 2 - b ** 2)
                + 2.0 * gamma2 * a * b
                + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    # Clamp for numerical stability (V^{-1} is positive definite, so this should be > 0)
    y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)

    # NLL = log(2pi) - angular-Gaussian log kernel
    alpha = y_dot_mu / torch.sqrt(y_Vinv_y)
    nll = _LOG_2PI - _ag_log_kernel(y_Vinv_y, mu_norm_sq, alpha)

    return _apply_reduction(nll, reduction)


class ESAG(BaseDistribution):
    """Elliptically Symmetric Angular Gaussian distribution on S^2.

    The ESAG distribution generalizes IAG with ellipse-like contours on the
    sphere, controlled by shape parameters gamma = (gamma_1, gamma_2). Setting gamma = (0, 0)
    recovers the IAG distribution.

    Reference: Paine et al. (2018), Stat Comput 28:689-697.

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is mu and
            pred[:, 3:5] is gamma = (gamma_1, gamma_2).
    """

    n_params = 5

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred[:, :3])

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B]. Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (gamma_1, gamma_2) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_ESAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]       # [B, 3]
        gamma1 = self._pred[:, 3]    # [B]
        gamma2 = self._pred[:, 4]    # [B]

        mu_norm_sq = (mu ** 2).sum(dim=1)  # [B]

        # Construct orthonormal basis perpendicular to mu
        xi1, xi2 = _construct_orthonormal_basis(mu)  # [B, 3] each

        # y.mu, y.xi_1, y.xi_2 for all (point, sample) pairs
        y_dot_mu = points @ mu.T    # [N, B]
        a = points @ xi1.T          # [N, B]  (y.xi_1)
        b = points @ xi2.T          # [N, B]  (y.xi_2)

        # y'V^{-1}y (Equation 18)
        gamma_sq = gamma1 ** 2 + gamma2 ** 2        # [B]
        sqrt_term = torch.sqrt(1.0 + gamma_sq)      # [B]
        a_sq_plus_b_sq = a ** 2 + b ** 2             # [N, B]

        y_Vinv_y = (1.0
                    + gamma1[None, :] * (a ** 2 - b ** 2)
                    + 2.0 * gamma2[None, :] * a * b
                    + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

        y_Vinv_y = torch.clamp(y_Vinv_y, min=1e-8)  # [N, B]

        # log f = -log(2pi) + angular-Gaussian log kernel
        alpha = y_dot_mu / torch.sqrt(y_Vinv_y)
        log_p = -_LOG_2PI + _ag_log_kernel(y_Vinv_y, mu_norm_sq[None, :], alpha)

        return log_p.T  # [B, N]

    def mode(
        self,
        num_iters: int = 20,
        trust_radius: float = 0.5,
        ls_iters: int = 10,
    ) -> Tensor:
        """MAP direction on S^2 via Riemannian Newton iteration.

        For moderate ``gamma`` the mode coincides with ``mu / ||mu||`` by
        symmetry of the elliptical contours, but for large enough
        ``gamma_1`` the ``(a^2 - b^2)`` term can flip a Hessian eigenvalue
        and turn the mean direction into a saddle point. This method runs
        Newton's method in the tangent space of ``S^2`` (with backtracking
        line search and fall-back to a gradient step when the Hessian is
        not negative-definite), starting from ``mean_direction``. For the
        common case of moderate gamma, the method exits in one step.

        Args:
            num_iters: Maximum Newton iterations.
            trust_radius: Maximum angular step per iteration, in radians.
            ls_iters: Maximum backtracking halvings per Newton step.

        Returns:
            [B, 3] unit-norm MAP directions (detached from the autograd graph).
        """
        pred = self._pred.detach()
        mu = pred[:, :3]
        gamma1 = pred[:, 3]
        gamma2 = pred[:, 4]
        xi1, xi2 = _construct_orthonormal_basis(mu)
        mu_norm_sq = (mu ** 2).sum(dim=1)

        def _esag_log_p_one(y_b, mu_b, xi1_b, xi2_b, g1_b, g2_b, mns_b):
            a = (y_b * xi1_b).sum()
            b = (y_b * xi2_b).sum()
            y_dot_mu = (y_b * mu_b).sum()
            g_sq = g1_b * g1_b + g2_b * g2_b
            sqrt_term = torch.sqrt(1.0 + g_sq)
            r2 = a * a + b * b
            y_Vinv_y = (1.0
                        + g1_b * (a * a - b * b)
                        + 2.0 * g2_b * a * b
                        + (sqrt_term - 1.0) * r2).clamp_min(1e-8)
            T = y_dot_mu / torch.sqrt(y_Vinv_y)
            return _ag_log_kernel(y_Vinv_y, mns_b, T)

        # ESAG's V^{-1} acts in the (xi1, xi2) plane; its 2x2 block is
        # [[1 + g1 + sqrt(1+g^2) - 1, g2], [g2, 1 - g1 + sqrt(1+g^2) - 1]]
        # = [[g1 + s, g2], [g2, -g1 + s]] with s = sqrt(1 + g1^2 + g2^2).
        # Its smaller eigenvalue corresponds to V's largest spread; the
        # eigenvector lifted to 3D gives an extra start for Newton.
        s = torch.sqrt(1.0 + gamma1 ** 2 + gamma2 ** 2)
        block_tr = 2.0 * s
        block_det = (gamma1 + s) * (-gamma1 + s) - gamma2 * gamma2
        disc = (block_tr * block_tr - 4.0 * block_det).clamp_min(0.0).sqrt()
        lam_min_2d = 0.5 * (block_tr - disc)              # smallest eigenvalue
        # Eigenvector in 2D for lam_min: (a - lam) x + b y = 0
        ev_x = gamma2
        ev_y = lam_min_2d - (gamma1 + s)
        ev = torch.stack([ev_x, ev_y], dim=-1)
        ev = F.normalize(ev, p=2, dim=-1)
        fallback_2d = torch.zeros_like(ev)
        fallback_2d[..., 0] = 1.0
        ev = torch.where(ev.norm(dim=-1, keepdim=True) > 1e-12, ev, fallback_2d)
        v_spread = ev[:, 0:1] * xi1 + ev[:, 1:2] * xi2    # [B, 3]

        md = self.mean_direction
        y_inits = torch.stack([md, v_spread, -v_spread], dim=0)  # [3, B, 3]

        mode = _multistart_mode(
            y_inits,
            _esag_log_p_one,
            (mu, xi1, xi2, gamma1, gamma2, mu_norm_sq),
            num_iters=num_iters,
            trust_radius=trust_radius,
            ls_iters=ls_iters,
        )
        lp_mode = self.log_pdf(mode).diagonal()
        lp_md = self.log_pdf(md).diagonal()
        return torch.where((lp_mode <= lp_md + 1e-5)[:, None], md, mode)


# ---------------------------------------------------------------------------
# General Angular Gaussian (GAG)
# ---------------------------------------------------------------------------

def gag_nll_loss(pred: Tensor, y_true: Tensor, reduction: str = "mean") -> Tensor:
    """General Angular Gaussian (GAG) negative log-likelihood loss.

    The GAG is the full 8-parameter angular Gaussian on S^2, with density:

        f_AG(y; mu, V) = (2pi)^{-1} |V|^{-1/2} (y^T V^{-1} y)^{-3/2}
                        * exp[1/2((y^T V^{-1} mu)^2 / (y^T V^{-1} y) - mu^T V^{-1} mu)]
                        * M_2(y^T V^{-1} mu / sqrt(y^T V^{-1} y))

    V^{-1} is parameterised as LL^T via a log-Cholesky factor with det(L) = 1,
    so |V| = 1 and the log-determinant term vanishes.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (2).

    Args:
        pred: [B, 9] predictions where:
              - pred[:, :3]  = mu  (mean vector, unconstrained)
              - pred[:, 3:6] = raw log-diagonal of Cholesky factor L
              - pred[:, 6:9] = off-diagonal entries (L_21, L_31, L_32)
        y_true: [B, 3] true unit direction vectors on S^2.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Reduced NLL loss (scalar for ``"mean"``/``"sum"``, [B] for ``"none"``).
    """
    mu = pred[:, :3]                           # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)        # [B, 3]
    L = _build_cholesky(pred)                  # [B, 3, 3]

    # Transformed vectors: z_y = L^T y,  z_mu = L^T mu
    # (batch matrix-vector multiply via einsum)
    z_y = torch.einsum('bji,bj->bi', L, y)    # [B, 3]
    z_mu = torch.einsum('bji,bj->bi', L, mu)  # [B, 3]

    Q = (z_y ** 2).sum(dim=1)                  # ||z_y||^2 = y^T V^{-1} y  [B]
    S = (z_mu ** 2).sum(dim=1)                 # ||z_mu||^2 = mu^T V^{-1} mu  [B]
    z_dot = (z_y * z_mu).sum(dim=1)            # z_y.z_mu = y^T V^{-1} mu  [B]

    Q_safe = torch.clamp(Q, min=1e-8)
    T = z_dot / torch.sqrt(Q_safe)             # [B]

    # NLL = log(2pi) - angular-Gaussian log kernel
    nll = _LOG_2PI - _ag_log_kernel(Q_safe, S, T)

    return _apply_reduction(nll, reduction)


class GAG(BaseDistribution):
    """General Angular Gaussian distribution on S^2.

    The GAG is the full 8-parameter angular Gaussian, generalising ESAG by
    allowing the covariance eigenvectors to be independent of mu.  This
    enables asymmetric, non-elliptical contours on the sphere.

    V^{-1} is parameterised via a log-Cholesky factor L with det(L) = 1.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (2).

    Args:
        pred: [B, 9] raw network output where pred[:, :3] is mu,
            pred[:, 3:6] is the raw log-diagonal of L, and
            pred[:, 6:9] is the off-diagonal (L_21, L_31, L_32).
    """

    n_params = 9

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return _safe_unit_vector(self._pred[:, :3])

    @property
    def concentration(self) -> Tensor:
        """Concentration ||mu|| [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def cholesky_factor(self) -> Tensor:
        """Normalised lower-triangular Cholesky factor L [B, 3, 3].

        V^{-1} = LL^T with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GAG(y) at points on S^2.

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                          # [B, 3]
        L = _build_cholesky(self._pred)                 # [B, 3, 3]

        mu_Vinv_mu = (torch.einsum('bji,bj->bi', L, mu) ** 2).sum(dim=1)  # S [B]

        # Transform all grid points through each batch element's L^T:
        # z_y[b, n, i] = L^T[b, i, j] . points[n, j] = L[b, j, i] . points[n, j]
        z_y = torch.einsum('bji,nj->bni', L, points)   # [B, N, 3]

        # Also transform mu: z_mu[b, i] = L^T[b, i, j] . mu[b, j]
        z_mu = torch.einsum('bji,bj->bi', L, mu)       # [B, 3]

        Q = (z_y ** 2).sum(dim=2)                       # [B, N]
        z_dot = torch.einsum('bni,bi->bn', z_y, z_mu)   # [B, N]

        Q_safe = torch.clamp(Q, min=1e-8)
        T = z_dot / torch.sqrt(Q_safe)                  # [B, N]

        # log f = -log(2pi) + angular-Gaussian log kernel
        log_p = -_LOG_2PI + _ag_log_kernel(Q_safe, mu_Vinv_mu[:, None], T)

        return log_p  # [B, N]

    def mode(
        self,
        n_lat: int = 91,
        n_lon: int = 180,
        num_iters: int = 5,
        trust_radius: float = 0.25,
        ls_iters: int = 10,
        grid_chunk: int = 8192,
    ) -> Tensor:
        """MAP direction on S^2 via dense grid argmax with a Newton polish.

        The GAG log-density is analytic and easily batchable, so the global
        MAP can be found by evaluating ``log_pdf`` on a uniform lat/lon grid
        and taking the per-sample argmax.  A small number of Newton iterations
        from that single seed then refines the cell-quantised argmax to
        float precision.

        This is much cheaper than multi-start Newton because each Newton
        iteration requires a 3x3 Hessian via autograd over the same scalar
        log-density that one grid eval already computes -- so 20 iterations
        of 15 starts cost roughly 300x what a single grid pass costs, with
        nearly identical accuracy in practice (the grid argmax is already
        within the basin of the global mode by construction, which also
        handles the rare bimodal GAG regime correctly without any explicit
        multi-start machinery).

        Args:
            n_lat, n_lon: lat/lon grid resolution. Default 91 x 180 gives
                ~2 deg cell spacing, well below the typical mode-vs-mean
                offset; the polish brings the result to float precision.
            num_iters: Newton polishing iterations from the grid seed.
                Default 5 -- the seed is already within ~2 deg of the true
                MAP, so a few quadratic-converging Newton steps suffice.
                Pass ``0`` to skip the polish and return the raw argmax.
            trust_radius: max angular step per iteration, in radians.
            ls_iters: max backtracking halvings per Newton step.
            grid_chunk: batch chunk size for the grid log_pdf eval. The
                intermediate ``[B, n_lat*n_lon]`` log-density tensor is the
                memory hot-spot; chunk to keep peak usage bounded.

        Returns:
            [B, 3] unit-norm MAP directions (detached from the autograd graph).
        """
        from torch.func import vmap

        pred = self._pred.detach()
        mu = pred[:, :3]
        L = _build_cholesky(pred)
        z_mu = torch.einsum('bji,bj->bi', L, mu)
        S = (z_mu ** 2).sum(dim=1)
        device, dtype = pred.device, pred.dtype
        B = pred.shape[0]

        def _gag_log_p_one(y_b, L_b, z_mu_b, S_b):
            z_y = L_b.T @ y_b
            Q = (z_y * z_y).sum().clamp_min(1e-8)
            T = (z_y * z_mu_b).sum() / torch.sqrt(Q)
            return _ag_log_kernel(Q, S_b, T)

        # 1) Grid argmax: evaluate log_pdf on a uniform lat/lon grid and pick
        # the highest-density cell per sample. Chunked over the batch axis to
        # bound the [B, N] intermediate.
        coarse = make_grid(n_lat=n_lat, n_lon=n_lon, device=device).points.to(dtype)
        grid_seed = torch.empty(B, 3, device=device, dtype=dtype)
        for s in range(0, B, grid_chunk):
            e = min(s + grid_chunk, B)
            lp = GAG(pred[s:e]).log_pdf(coarse)               # [b, N]
            grid_seed[s:e] = coarse[lp.argmax(dim=-1)]

        # 2) Additional candidate seeds at ~zero cost: mean_direction (closed-
        # form MAP for IAG) and the eigendirections of V (also of V^{-1}) and
        # their antipodes. The latter act as cheap insurance for extreme
        # anisotropy where the global mode sits in a basin narrower than the
        # grid cell.
        md = _safe_unit_vector(mu)
        Vinv = L @ L.transpose(1, 2)
        _, eigvecs = torch.linalg.eigh(Vinv)                  # [B, 3, 3]
        eig_axes = eigvecs.transpose(1, 2)                    # [B, 3, 3] rows

        candidates = torch.cat([
            grid_seed[:, None, :],     # [B, 1, 3]
            md[:, None, :],            # [B, 1, 3]
            eig_axes,                  # [B, 3, 3]
            -eig_axes,                 # [B, 3, 3]
        ], dim=1)                      # [B, 8, 3]

        # Score every candidate via the scalar log-density. Double-vmap: inner
        # over the batch axis (params + candidate aligned), outer over the K
        # candidate axis (params broadcast).
        lp_batch = vmap(_gag_log_p_one, in_dims=(0, 0, 0, 0))
        lp_all = vmap(lp_batch, in_dims=(1, None, None, None))(
            candidates, L, z_mu, S
        ).T                                                   # [B, K]
        best = lp_all.argmax(dim=-1)                          # [B]
        seed = candidates[torch.arange(B, device=device), best]

        if num_iters <= 0:
            return seed

        # 3) Newton polish from the single best seed.
        return _newton_mode_on_sphere(
            seed, _gag_log_p_one, (L, z_mu, S),
            num_iters=num_iters, trust_radius=trust_radius, ls_iters=ls_iters,
        )
