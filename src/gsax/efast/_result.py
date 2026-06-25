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

    Shapes follow the convention ``(T, K, D)`` for time-resolved analyses
    or ``(K, D)`` when the time dimension is squeezed, where *K* is the
    number of outputs and *D* the number of parameters.

    Args:
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
        param_names = list(self.problem.names)
        output_names = self.problem.output_names
        ndim = self.S1.ndim

        if ndim == 1:
            dims_s1 = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            K = self.S1.shape[0]
            if output_names is not None and len(output_names) != K:
                msg = f"output_names length {len(output_names)} != K={K}"
                raise ValueError(msg)
            onames = list(output_names) if output_names else [f"y{i}" for i in range(K)]
            dims_s1 = ("output", "param")
            coords = {"param": param_names, "output": onames}
        elif ndim == 3:
            T, K = self.S1.shape[0], self.S1.shape[1]
            if output_names is not None and len(output_names) != K:
                msg = f"output_names length {len(output_names)} != K={K}"
                raise ValueError(msg)
            onames = list(output_names) if output_names else [f"y{i}" for i in range(K)]
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims_s1 = ("time", "output", "param")
            coords = {"param": param_names, "output": onames, "time": tcoords}
        else:
            msg = f"Unexpected S1.ndim={ndim}"
            raise ValueError(msg)

        data_vars: dict = {
            "S1": (dims_s1, np.asarray(self.S1)),
            "ST": (dims_s1, np.asarray(self.ST)),
        }

        return xr.Dataset(data_vars, coords=coords)
