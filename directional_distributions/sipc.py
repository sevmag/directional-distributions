"""Spherical Isotropic Projected Cauchy distribution: loss function and evaluation.

The SIPC arises from projecting a trivariate Cauchy distribution with
Σ = I onto the sphere S².  It is rotationally symmetric about the mean
direction, analogous to the IAG distribution in the angular Gaussian family.

Reference
---------
Tsagris & Alzeley (2024), "Circular and Spherical Projected Cauchy
Distributions", arXiv:2302.02468v4, Equation (20).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _sc_log_density


def sipc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Spherical Isotropic Projected Cauchy (SIPC) negative log-likelihood loss.

    The SIPC is the projected Cauchy with Σ = I, making it rotationally
    symmetric about the mean direction.  It is the Cauchy analog of IAG.

    The density is:

        f(y; μ) ∝ [B(Γ²+1)√Δ·Ω + 2AΔ] / Δ²

    where A = y·μ, B = 1, Γ² = ‖μ‖², Δ = Γ²+1−A².

    Args:
        pred: [B, 3] predicted mean vectors μ.  The magnitude ‖μ‖ controls
              concentration (higher = more peaked), and μ/‖μ‖ is the mean
              direction.
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred                                       # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)             # [B, 3]

    # With Σ = I: A = y·μ, B = ‖y‖² = 1, Γ² = ‖μ‖²
    A = (y * mu).sum(dim=1)                          # [B]
    B = torch.ones_like(A)                           # [B]
    Gamma_sq = (mu ** 2).sum(dim=1)                  # [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class SIPC(BaseDistribution):
    """Spherical Isotropic Projected Cauchy distribution on S².

    The SIPC is rotationally symmetric about the mean direction,
    with concentration controlled by ‖μ‖.  It is the Cauchy analog of IAG.

    The density is derived by projecting a trivariate Cauchy C(μ, I)
    onto the sphere via z ↦ z/‖z‖.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (20).

    Args:
        pred: [B, 3] raw network output (the mean vector μ).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred, p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_SIPC(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred                              # [B, 3]
        Gamma_sq = (mu ** 2).sum(dim=1)              # [B]

        A = points @ mu.T                            # [N, B]
        B = torch.ones_like(A)                       # [N, B]
        Gamma_sq_exp = Gamma_sq[None, :].expand_as(A)  # [N, B]

        return _sc_log_density(A, B, Gamma_sq_exp).T  # [B, N]
