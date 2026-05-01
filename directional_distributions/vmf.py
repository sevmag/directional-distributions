"""Von Mises-Fisher distribution: loss function and evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ._base import BaseDistribution, _apply_reduction


def von_mises_fisher_loss(
    n_pred: Tensor,
    n_true: Tensor,
    kappa_reg: float = 0.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """
    von Mises-Fisher loss with coupled direction and κ.

    Expects n_pred [B,3]: direction = normalize(n_pred), κ = ||n_pred||.

    Args:
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """
    direction = F.normalize(n_pred, p=2, dim=1)
    kappa = n_pred.norm(p=2, dim=1)
    cos_sim = (direction * n_true).sum(dim=1)
    log_C = -kappa + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
    nll = -(kappa * cos_sim + log_C) + kappa_reg * kappa
    return _apply_reduction(nll, reduction)


class VMF(BaseDistribution):
    """Von Mises-Fisher distribution on S².

    The vMF distribution has density:

        f(y | μ, κ) = C(κ) exp(κ μ·y)

    where C(κ) = κ / (4π sinh(κ)) is the normalization constant.

    Direction and concentration are coupled: the network outputs μ ∈ ℝ³,
    with direction = μ/||μ|| and κ = ||μ||.

    Args:
        pred: [B, 3] raw network output (the mean vector μ).
    """

    n_params = 3

    @property
    def mean_direction(self) -> Tensor:
        """Unit mean direction [B, 3]."""
        return F.normalize(self._pred, p=2, dim=1)

    @property
    def kappa(self) -> Tensor:
        """Concentration parameter κ = ||μ|| [B]."""
        return self._pred.norm(p=2, dim=1)

    def log_pdf(self, points: Tensor, eps: float = 1e-8) -> Tensor:
        """Evaluate log f(y | μ, κ) at points on S².

        Args:
            points: [N, 3] unit vectors on the sphere.
            eps: Small constant for numerical stability.

        Returns:
            [B, N] log-probability density.
        """
        mu = self._pred          # [B, 3]
        kappa = self.kappa        # [B]

        # κ μ̂·y = ||μ|| (μ/||μ||)·y = μ·y
        dot = points @ mu.T  # [N, B]

        # log C(κ) = log(κ / (4π sinh(κ)))
        #          = log(κ) - log(2π) - κ - log(1 - exp(-2κ))
        log_C = (
            -kappa
            + torch.log((kappa + eps) / (1 - torch.exp(-2 * kappa) + 2 * eps))
            - np.log(2 * np.pi)
        )

        return (log_C[None, :] + dot).T  # [B, N]
