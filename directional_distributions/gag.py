"""General Angular Gaussian distribution: loss function and evaluation.

The General Angular Gaussian (GAG) is the full 8-parameter member of the
angular Gaussian family on S², obtained by projecting a trivariate normal
N(μ, V) onto the sphere via z ↦ z/‖z‖.  It generalises the ESAG (5 params)
by relaxing the eigenvector constraint Vμ = μ, allowing asymmetric and
non-elliptical contour shapes.

Parameterisation
----------------
We parameterise V⁻¹ = LLᵀ via its **log-Cholesky factor**: the network
outputs 9 raw values (3 for μ, 6 for L), and the diagonal of L is
exponentiated after centering in log-space so that det(L) = 1, enforcing
|V| = 1.  This eliminates the scale redundancy and gives 8 effective free
parameters.

All NLL terms are computed through the transformed vectors z_y = Lᵀy and
z_μ = Lᵀμ, so no matrix inversion is ever needed.

Reference
---------
Paine et al. (2018), "An elliptically symmetric angular Gaussian
distribution", Stat Comput 28:689-697, Equation (2).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _build_cholesky, _log_M2


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def gag_nll_loss(pred: Tensor, y_true: Tensor) -> Tensor:
    """General Angular Gaussian (GAG) negative log-likelihood loss.

    The GAG is the full 8-parameter angular Gaussian on S², with density:

        f_AG(y; μ, V) = (2π)⁻¹ |V|⁻¹ᐟ² (yᵀV⁻¹y)⁻³ᐟ²
                        × exp[½((yᵀV⁻¹μ)² / (yᵀV⁻¹y) − μᵀV⁻¹μ)]
                        × M₂(yᵀV⁻¹μ / √(yᵀV⁻¹y))

    V⁻¹ is parameterised as LLᵀ via a log-Cholesky factor with det(L) = 1,
    so |V| = 1 and the log-determinant term vanishes.

    Args:
        pred: [B, 9] predictions where:
              - pred[:, :3]  = μ  (mean vector, unconstrained)
              - pred[:, 3:6] = raw log-diagonal of Cholesky factor L
              - pred[:, 6:9] = off-diagonal entries (L₂₁, L₃₁, L₃₂)
        y_true: [B, 3] true unit direction vectors on S².

    Returns:
        Scalar mean NLL loss over the batch.
    """
    mu = pred[:, :3]                           # [B, 3]
    y = F.normalize(y_true, p=2, dim=1)        # [B, 3]
    L = _build_cholesky(pred)                  # [B, 3, 3]

    # Transformed vectors: z_y = Lᵀ y,  z_mu = Lᵀ μ
    # (batch matrix-vector multiply via einsum)
    z_y = torch.einsum('bji,bj->bi', L, y)    # [B, 3]
    z_mu = torch.einsum('bji,bj->bi', L, mu)  # [B, 3]

    Q = (z_y ** 2).sum(dim=1)                  # ‖z_y‖² = yᵀV⁻¹y  [B]
    S = (z_mu ** 2).sum(dim=1)                 # ‖z_μ‖² = μᵀV⁻¹μ  [B]
    z_dot = (z_y * z_mu).sum(dim=1)            # z_y·z_μ = yᵀV⁻¹μ  [B]

    Q_safe = torch.clamp(Q, min=1e-8)
    T = z_dot / torch.sqrt(Q_safe)             # [B]

    log_M2 = _log_M2(T)

    # NLL = log(2π) + 1.5·log(Q) + 0.5·(S − T²) − log M₂(T)
    nll = (np.log(2 * np.pi)
           + 1.5 * torch.log(Q_safe)
           + 0.5 * (S - T ** 2)
           - log_M2)

    return nll.mean()


# ---------------------------------------------------------------------------
# Distribution class
# ---------------------------------------------------------------------------

class GAG(BaseDistribution):
    """General Angular Gaussian distribution on S².

    The GAG is the full 8-parameter angular Gaussian, generalising ESAG by
    allowing the covariance eigenvectors to be independent of μ.  This
    enables asymmetric, non-elliptical contours on the sphere.

    V⁻¹ is parameterised via a log-Cholesky factor L with det(L) = 1.

    Reference: Paine et al. (2018), Stat Comput 28:689-697, Equation (2).

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

        V⁻¹ = LLᵀ with det(L) = 1.
        """
        return _build_cholesky(self._pred)

    def log_pdf(self, points: Tensor) -> Tensor:
        """Evaluate log f_GAG(y) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred[:, :3]                          # [B, 3]
        L = _build_cholesky(self._pred)                 # [B, 3, 3]

        mu_Vinv_mu = (torch.einsum('bji,bj->bi', L, mu) ** 2).sum(dim=1)  # S [B]

        # Transform all grid points through each batch element's Lᵀ:
        # z_y[b, n, i] = Lᵀ[b, i, j] · points[n, j] = L[b, j, i] · points[n, j]
        z_y = torch.einsum('bji,nj->bni', L, points)   # [B, N, 3]

        # Also transform μ: z_mu[b, i] = Lᵀ[b, i, j] · μ[b, j]
        z_mu = torch.einsum('bji,bj->bi', L, mu)       # [B, 3]

        Q = (z_y ** 2).sum(dim=2)                       # [B, N]
        z_dot = torch.einsum('bni,bi->bn', z_y, z_mu)   # [B, N]

        Q_safe = torch.clamp(Q, min=1e-8)
        T = z_dot / torch.sqrt(Q_safe)                  # [B, N]

        log_M2 = _log_M2(T)                             # [B, N]

        # log f = −log(2π) − 1.5·log Q − 0.5·(S − T²) + log M₂(T)
        log_p = (-np.log(2 * np.pi)
                 - 1.5 * torch.log(Q_safe)
                 - 0.5 * (mu_Vinv_mu[:, None] - T ** 2)
                 + log_M2)

        return log_p  # [B, N]
