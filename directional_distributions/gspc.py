"""General Spherical Projected Cauchy distribution: loss function and evaluation.

The General Spherical Projected Cauchy (GSPC) is the full 8-parameter member
of the spherical projected Cauchy family on S², obtained by projecting a
trivariate Cauchy C(μ, Σ) onto the sphere via z ↦ z/‖z‖.  It generalises
the SESPC (5 params) by relaxing the eigenvector constraint Σμ = μ,
analogous to how GAG generalises ESAG.

Parameterisation
----------------
Identical to GAG: Σ⁻¹ = LLᵀ via log-Cholesky factor with det(L) = 1.
The network outputs 9 raw values (3 for μ, 6 for L), with 8 effective
free parameters after centering the log-diagonal.

Reference
---------
Tsagris & Alzeley (2024), "Circular and Spherical Projected Cauchy
Distributions", arXiv:2302.02468v4, Equation (18).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _build_cholesky, _sc_log_density


def gspc_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """General Spherical Projected Cauchy (GSPC) negative log-likelihood loss.

    The GSPC is the full 8-parameter projected Cauchy on S², with density
    given by Eq. (18) of the reference with |Σ| = 1.

    Σ⁻¹ is parameterised as LLᵀ via a log-Cholesky factor with det(L) = 1,
    identical to the GAG parameterisation.

    Args:
        pred: [B, 9] predictions where:
              - pred[:, :3]  = μ  (mean vector, unconstrained)
              - pred[:, 3:6] = raw log-diagonal of Cholesky factor L
              - pred[:, 6:9] = off-diagonal entries (L₂₁, L₃₁, L₃₂)
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]                                   # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)                # [B, 3]
    L = _build_cholesky(pred)                          # [B, 3, 3]

    # Transformed vectors: z_y = L⊤y,  z_mu = L⊤μ
    z_y = torch.einsum('bji,bj->bi', L, y)            # [B, 3]
    z_mu = torch.einsum('bji,bj->bi', L, mu)          # [B, 3]

    B = (z_y ** 2).sum(dim=1)                          # ‖z_y‖² = y⊤Σ⁻¹y  [B]
    Gamma_sq = (z_mu ** 2).sum(dim=1)                  # ‖z_μ‖² = μ⊤Σ⁻¹μ  [B]
    A = (z_y * z_mu).sum(dim=1)                        # z_y·z_μ = y⊤Σ⁻¹μ  [B]

    return -_sc_log_density(A, B, Gamma_sq).mean()


class GSPC(BaseDistribution):
    """General Spherical Projected Cauchy distribution on S².

    The GSPC is the full 8-parameter projected Cauchy, generalising SESPC
    by allowing the scatter matrix eigenvectors to be independent of μ.
    This enables asymmetric, non-elliptical contours on the sphere.

    Σ⁻¹ is parameterised via a log-Cholesky factor L with det(L) = 1.

    Reference: Tsagris & Alzeley (2024), arXiv:2302.02468v4, Eq. (18).

    Args:
        pred: [B, 9] raw network output where pred[:, :3] is μ,
            pred[:, 3:6] is the raw log-diagonal of L, and
            pred[:, 6:9] is the off-diagonal (L₂₁, L₃₁, L₃₂).
    """

    n_params = 9

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred[:, :3], p=2, dim=1)

    @property
    def concentration(self) -> Tensor:
        """Concentration ‖μ‖ [B].  Higher = more peaked."""
        return self._pred[:, :3].norm(p=2, dim=1)

    @property
    def cholesky_factor(self) -> Tensor:
        """Normalised lower-triangular Cholesky factor L [B, 3, 3].

        Σ⁻¹ = LLᵀ with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GSPC(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                          # [B, 3]
        L = _build_cholesky(self._pred)                 # [B, 3, 3]

        # Transform all grid points and μ through L⊤
        z_y = torch.einsum('bji,nj->bni', L, points)   # [B, N, 3]
        z_mu = torch.einsum('bji,bj->bi', L, mu)       # [B, 3]

        B = (z_y ** 2).sum(dim=2)                       # [B, N]
        Gamma_sq = (z_mu ** 2).sum(dim=1)               # [B]
        A = torch.einsum('bni,bi->bn', z_y, z_mu)      # [B, N]

        Gamma_sq_exp = Gamma_sq[:, None].expand_as(A)   # [B, N]

        return _sc_log_density(A, B, Gamma_sq_exp)      # [B, N]
