"""Directional distribution loss functions and evaluation utilities for PyTorch."""

from .vmf import von_mises_fisher_loss, VMF
from .iag import iag_nll_loss, IAG
from .esag import esag_nll_loss, ESAG
from .gag import gag_nll_loss, GAG
from .sipc import sipc_nll_loss, SIPC
from .sespc import sespc_nll_loss, SESPC
from .gspc import gspc_nll_loss, GSPC
from ._base import SphereGrid, make_grid
from ._plotting import plot_mollweide, set_style, COLOR_CYCLE

__all__ = [
    # Loss functions (Angular Gaussian family)
    "von_mises_fisher_loss",
    "iag_nll_loss",
    "esag_nll_loss",
    "gag_nll_loss",
    # Loss functions (Spherical Projected Cauchy family)
    "sipc_nll_loss",
    "sespc_nll_loss",
    "gspc_nll_loss",
    # Distribution classes (Angular Gaussian family)
    "VMF",
    "IAG",
    "ESAG",
    "GAG",
    # Distribution classes (Spherical Projected Cauchy family)
    "SIPC",
    "SESPC",
    "GSPC",
    # Grid utilities
    "SphereGrid",
    "make_grid",
    # Plotting
    "plot_mollweide",
    "set_style",
    "COLOR_CYCLE",
]
