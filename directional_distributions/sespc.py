"""Spherical Elliptically Symmetric Projected Cauchy distribution: loss function and evaluation.

The SESPC arises from projecting a trivariate Cauchy distribution with
location-constrained scatter matrix (Σμ = μ, |Σ| = 1) onto the sphere S².
It produces ellipse-like contours analogous to the ESAG distribution in the
angular Gaussian family.

The Σ⁻¹ construction follows Paine et al. (2018), identical to ESAG.

Reference
---------
Tsagris & Alzeley (2024), "Circular and Spherical Projected Cauchy
Distributions", arXiv:2302.02468v4, Equation (22).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _construct_orthonormal_basis, _sc_log_density


def sespc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Spherical Elliptically Symmetric Projected Cauchy (SESPC) NLL loss.

    The SESPC generalises SIPC with ellipse-like contours on the sphere,
    controlled by shape parameters γ = (γ₁, γ₂).  It is the Cauchy analog
    of ESAG.

    Args:
        pred: [B, 5] predictions where:
              - pred[:, :3] = μ (mean vector, magnitude controls concentration)
              - pred[:, 3:5] = γ = (γ₁, γ₂) (shape parameters for ellipticity)
              Setting γ = (0, 0) recovers the SIPC distribution.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]                                  # [B, 3]
    gamma1 = pred[:, 3]                               # [B]
    gamma2 = pred[:, 4]                               # [B]

    y = F.normalize(y_true, p=2, dim=1)               # [B, 3]

    # Since Σμ = μ: A = y·μ, Γ² = ‖μ‖²
    A = (y * mu).sum(dim=1)                            # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                    # [B]

    # Construct orthonormal basis perpendicular to μ
    xi1, xi2 = _construct_orthonormal_basis(mu)        # [B, 3] each

    # Projections
    a = (y * xi1).sum(dim=1)                           # [B]
    b = (y * xi2).sum(dim=1)                           # [B]

    # B = y⊤Σ⁻¹y (Paine et al. 2018, Eq. 18)
    gamma_sq = gamma1 ** 2 + gamma2 ** 2
    sqrt_term = torch.sqrt(1.0 + gamma_sq)
    a_sq_plus_b_sq = a ** 2 + b ** 2

    B = (1.0
         + gamma1 * (a ** 2 - b ** 2)
         + 2.0 * gamma2 * a * b
         + (sqrt_term - 1.0) * a_sq_plus_b_sq)

    B = torch.clamp(B, min=1e-8)                      # [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class SESPC(BaseDistribution):
    """Spherical Elliptically Symmetric Projected Cauchy distribution on S².

    The SESPC generalises SIPC with ellipse-like contours on the sphere,
    controlled by shape parameters γ = (γ₁, γ₂).  Setting γ = (0, 0)
    recovers the SIPC distribution.  It is the Cauchy analog of ESAG.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (22).

    Args:
        pred: [B, 5] raw network output where pred[:, :3] is μ and
            pred[:, 3:5] is γ = (γ₁, γ₂).
    """

    n_params = 5

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def gamma(self) -> Tensor:
        """Ellipticity parameters (γ₁, γ₂) [B, 2]."""
        return self._pred[:, 3:5]

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_SESPC(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                         # [B, 3]
        gamma1 = self._pred[:, 3]                      # [B]
        gamma2 = self._pred[:, 4]                      # [B]
        Gamma_sq = (mu ** 2).sum(dim=1)                # [B]

        xi1, xi2 = _construct_orthonormal_basis(mu)    # [B, 3] each

        # [N, B] intermediates
        A = points @ mu.T                              # [N, B]
        a = points @ xi1.T                             # [N, B]
        b = points @ xi2.T                             # [N, B]

        gamma_sq = gamma1 ** 2 + gamma2 ** 2           # [B]
        sqrt_term = torch.sqrt(1.0 + gamma_sq)         # [B]
        a_sq_plus_b_sq = a ** 2 + b ** 2               # [N, B]

        B = (1.0
             + gamma1[None, :] * (a ** 2 - b ** 2)
             + 2.0 * gamma2[None, :] * a * b
             + (sqrt_term - 1.0)[None, :] * a_sq_plus_b_sq)

        B = torch.clamp(B, min=1e-8)                  # [N, B]

        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)  # [N, B]

        return _sc_log_density(A, B, Gamma_sq_exp).T   # [B, N]
