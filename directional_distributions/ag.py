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

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import (
    BaseDistribution,
    _apply_reduction,
    _construct_orthonormal_basis,
    _build_cholesky,
)


# ---------------------------------------------------------------------------
# AG-family math utility
# ---------------------------------------------------------------------------

def _log_M2(alpha: Tensor) -> Tensor:
    """Compute log(M_2(alpha)) numerically stably.

    M_2(alpha) = (1 + alpha^2) * Phi(alpha) + alpha * phi(alpha)

    where Phi is the standard normal CDF and phi is the standard normal PDF.

    For large negative alpha, direct computation suffers from catastrophic
    cancellation. We use torch.special.log_ndtr for the tail, which provides
    correct values and gradients even for alpha << 0.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (4).
    """
    # Direct computation (accurate for moderate alpha)
    log_phi = -0.5 * alpha ** 2 - 0.5 * np.log(2 * np.pi)
    phi = torch.exp(log_phi)
    Phi = 0.5 * (1.0 + torch.erf(alpha / np.sqrt(2)))
    M2_direct = (1.0 + alpha ** 2) * Phi + alpha * phi

    # Stable computation for large negative alpha:
    #   M_2(alpha) = Phi(alpha) * [(1+alpha^2) + alpha * phi(alpha)/Phi(alpha)]
    #   log M_2 = log Phi(alpha) + log[(1+alpha^2) + alpha * exp(log phi(alpha) - log Phi(alpha))]
    #
    # The inner term (1+alpha^2) + alpha*phi/Phi ~ 2/alpha^2 for large |alpha|, computed as the
    # difference of two ~alpha^2-sized quantities. Float32 loses all significant
    # digits around |alpha| > 26, so we upcast to float64 for this subtraction.
    #
    # log_ndtr is not implemented for half/bfloat16 on CPU, so upcast if needed.
    compute_dtype = torch.float64 if alpha.dtype != torch.float64 else torch.float64
    alpha_hi = alpha.to(compute_dtype)
    log_phi_hi = -0.5 * alpha_hi ** 2 - 0.5 * np.log(2 * np.pi)
    log_Phi_hi = torch.special.log_ndtr(alpha_hi)
    ratio_hi = alpha_hi * torch.exp(log_phi_hi - log_Phi_hi)
    inner_hi = (1.0 + alpha_hi ** 2) + ratio_hi
    M2_stable = (log_Phi_hi + torch.log(torch.clamp(inner_hi, min=1e-300))).to(alpha.dtype)

    # Use direct for alpha >= -3.5, stable form for alpha < -3.5.
    # The direct branch suffers catastrophic cancellation in M2_direct for
    # alpha < ~-3.8 (float32), while the stable branch (computed in float64)
    # is accurate for all alpha < 0.
    return torch.where(alpha >= -3.5, torch.log(torch.clamp(M2_direct, min=1e-40)), M2_stable)


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

    # log(M_2(y.mu))
    log_M2 = _log_M2(y_dot_mu)

    # NLL = log(2pi) + 0.5*(||mu||^2 - (y.mu)^2) - log(M_2(y.mu))
    nll = np.log(2 * np.pi) + 0.5 * (mu_norm_sq - y_dot_mu ** 2) - log_M2

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
        return F.normalize(self._pred, p=2, dim=1)

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
        log_M2 = _log_M2(y_dot_mu)  # [N, B]

        # log f = -log(2pi) + 0.5*((y.mu)^2 - ||mu||^2) + log(M_2(y.mu))
        log_p = -np.log(2 * np.pi) + 0.5 * (y_dot_mu ** 2 - mu_norm_sq[None, :]) + log_M2

        return log_p.T  # [B, N]


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

    # Argument for M_2
    sqrt_y_Vinv_y = torch.sqrt(y_Vinv_y)
    alpha = y_dot_mu / sqrt_y_Vinv_y

    # log(M_2(alpha))
    log_M2 = _log_M2(alpha)

    # NLL = log(2pi) + 1.5*log(y'V^{-1}y) + 0.5*(||mu||^2 - (y.mu)^2/(y'V^{-1}y)) - log(M_2(alpha))
    nll = (np.log(2 * np.pi)
           + 1.5 * torch.log(y_Vinv_y)
           + 0.5 * (mu_norm_sq - y_dot_mu ** 2 / y_Vinv_y)
           - log_M2)

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
        return F.normalize(self._pred[:, :3], p=2, dim=1)

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

        # M_2 argument
        sqrt_y_Vinv_y = torch.sqrt(y_Vinv_y)
        alpha = y_dot_mu / sqrt_y_Vinv_y

        log_M2 = _log_M2(alpha)  # [N, B]

        # log f = -log(2pi) - 1.5*log(y'V^{-1}y) + 0.5*((y.mu)^2/(y'V^{-1}y) - ||mu||^2) + log(M_2)
        log_p = (-np.log(2 * np.pi)
                 - 1.5 * torch.log(y_Vinv_y)
                 + 0.5 * (y_dot_mu ** 2 / y_Vinv_y - mu_norm_sq[None, :])
                 + log_M2)

        return log_p.T  # [B, N]


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

    log_M2 = _log_M2(T)

    # NLL = log(2pi) + 1.5*log(Q) + 0.5*(S - T^2) - log M_2(T)
    nll = (np.log(2 * np.pi)
           + 1.5 * torch.log(Q_safe)
           + 0.5 * (S - T ** 2)
           - log_M2)

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
        return F.normalize(self._pred[:, :3], p=2, dim=1)

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

        log_M2 = _log_M2(T)                             # [B, N]

        # log f = -log(2pi) - 1.5*log Q - 0.5*(S - T^2) + log M_2(T)
        log_p = (-np.log(2 * np.pi)
                 - 1.5 * torch.log(Q_safe)
                 - 0.5 * (mu_Vinv_mu[:, None] - T ** 2)
                 + log_M2)

        return log_p  # [B, N]
