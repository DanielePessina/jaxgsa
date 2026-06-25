"""Defines the EFASTResult dataclass for eFAST sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax.problem import Problem


@dataclass
class EFASTResult:
    """Extended FAST sensitivity analysis results.

    Stores first-order (S1) and total-order (ST) Sobol indices computed
    via Fourier amplitude decomposition. eFAST does not produce
    second-order interaction indices.

    Args:
        S1: (D,) first-order indices.
        ST: (D,) total-order indices.
        problem: Problem definition used for the analysis.
        omega_0: The primary frequency used in the analysis.
        M: The interference factor (number of harmonics summed).
    """

    S1: Array
    ST: Array
    problem: Problem
    omega_0: int = 0
    M: int = 4

    def to_dataset(self) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset."""
        param_names = list(self.problem.names)
        coords: dict = {"param": param_names}
        data_vars: dict = {
            "S1": ("param", np.asarray(self.S1)),
            "ST": ("param", np.asarray(self.ST)),
        }
        return xr.Dataset(data_vars, coords=coords)
