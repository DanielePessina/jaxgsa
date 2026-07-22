"""Defines the EFASTResult dataclass for eFAST sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class EFASTResult:
    """Extended FAST sensitivity analysis results.

    Stores first-order (S1) and total-order (ST) Sobol indices computed
    via Fourier amplitude decomposition. eFAST does not produce
    second-order interaction indices.

    Shapes follow the convention ``(T, K, D)`` for time-resolved analyses,
    ``(K, D)`` when the time dimension is squeezed, or ``(D,)`` for scalar
    output, where *K* is the number of outputs and *D* the number of
    parameters.

    Attributes:
        S1: First-order indices — ``(D,)``, ``(K, D)``, or ``(T, K, D)``.
        ST: Total-order indices — same shape as S1.
        problem: Problem definition used for the analysis.
        omega_0: The primary frequency used in the analysis.
        M: The interference factor (number of harmonics summed).
    """

    S1: Array
    ST: Array
    problem: Problem
    omega_0: int = 0
    M: int = 4

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {"S1": self.S1.shape, "ST": self.ST.shape}
        return f"EFASTResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``S1.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``S1`` and ``ST``.
        """
        dims_s1, coords = _dims_and_coords(self.S1.ndim, self.S1.shape, self.problem, time_coords)

        data_vars: dict = {
            "S1": (dims_s1, np.asarray(self.S1)),
            "ST": (dims_s1, np.asarray(self.ST)),
        }

        return xr.Dataset(data_vars, coords=coords)
