"""Defines the PAWNResult dataclass for PAWN sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class PAWNResult:
    """PAWN sensitivity analysis results.

    Stores the PAWN index (median/max/mean KS distance across bins)
    and optional bootstrap confidence intervals.

    Attributes:
        pawn: PAWN sensitivity index per parameter.
        pawn_conf: Bootstrap confidence interval ``[lower, upper]``,
            or ``None`` when ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
    """

    pawn: Array
    pawn_conf: Array | None
    problem: Problem

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variable ``pawn`` and optionally
            ``pawn_lower`` / ``pawn_upper``.
        """
        param_names = list(self.problem.names)
        ndim = self.pawn.ndim

        if ndim == 1:
            dims = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            K = self.pawn.shape[0]
            onames = _default_output_names(K, self.problem)
            dims = ("output", "param")
            coords = {"output": onames, "param": param_names}
        elif ndim == 3:
            T = self.pawn.shape[0]
            K = self.pawn.shape[1]
            onames = _default_output_names(K, self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims = ("time", "output", "param")
            coords = {"time": tcoords, "output": onames, "param": param_names}
        else:
            raise ValueError(f"Unexpected pawn.ndim={ndim}")

        data_vars: dict = {
            "pawn": (dims, np.asarray(self.pawn)),
        }

        if self.pawn_conf is not None:
            data_vars["pawn_lower"] = (dims, np.asarray(self.pawn_conf[0]))
            data_vars["pawn_upper"] = (dims, np.asarray(self.pawn_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
