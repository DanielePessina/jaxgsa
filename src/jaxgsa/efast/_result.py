"""Defines the EFASTResult dataclass for eFAST sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class EFASTResult:
    """Extended FAST sensitivity analysis results.

    Stores first-order (S1) and total-order (ST) Sobol indices computed from a
    Fourier amplitude decomposition. eFAST produces no second-order
    interaction indices.

    Index shapes mirror the shape of the analyzed output Y: ``(D,)`` for a
    scalar output, ``(K, D)`` when the time dimension is squeezed, or
    ``(T, K, D)`` for time-resolved analyses. *D* is the number of parameters,
    *K* the number of outputs, and *T* the number of time steps.

    Attributes:
        S1: First-order indices, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        ST: Total-order indices, same shape as ``S1``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. A search curve can never be dropped, so the report is
            only ever informational here. See
            :class:`jaxgsa._core.invalid.InvalidReport`.
        omega_0: Primary frequency used in the analysis.
        M: Interference factor, the number of harmonics summed.
    """

    S1: Array
    ST: Array
    problem: Problem
    invalid: InvalidReport
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
