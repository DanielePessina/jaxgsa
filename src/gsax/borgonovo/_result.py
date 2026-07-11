"""Defines the DeltaResult dataclass for Borgonovo delta sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class DeltaResult:
    """Borgonovo delta sensitivity analysis results.

    Stores the moment-independent delta index and the given-data
    first-order Sobol index computed from the same class partition,
    with optional bootstrap confidence intervals.

    Attributes:
        delta: Borgonovo delta index per parameter (bias-corrected when
            the analysis ran with ``bias_correct=True`` and
            ``n_bootstrap > 0``).
        delta_conf: Percentile bootstrap confidence interval
            ``[lower, upper]``, or ``None`` when ``n_bootstrap=0``.
        S1: Given-data first-order Sobol index per parameter.
        S1_conf: Percentile bootstrap confidence interval
            ``[lower, upper]``, or ``None`` when ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
    """

    delta: Array
    delta_conf: Array | None
    S1: Array
    S1_conf: Array | None
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
            An ``xr.Dataset`` with variables ``delta`` and ``S1`` and
            optionally ``delta_lower`` / ``delta_upper`` /
            ``S1_lower`` / ``S1_upper``.
        """
        param_names = list(self.problem.names)
        ndim = self.delta.ndim

        if ndim == 1:
            dims = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            K = self.delta.shape[0]
            onames = _default_output_names(K, self.problem)
            dims = ("output", "param")
            coords = {"output": onames, "param": param_names}
        elif ndim == 3:
            T = self.delta.shape[0]
            K = self.delta.shape[1]
            onames = _default_output_names(K, self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims = ("time", "output", "param")
            coords = {"time": tcoords, "output": onames, "param": param_names}
        else:
            raise ValueError(f"Unexpected delta.ndim={ndim}")

        data_vars: dict = {
            "delta": (dims, np.asarray(self.delta)),
            "S1": (dims, np.asarray(self.S1)),
        }

        if self.delta_conf is not None:
            data_vars["delta_lower"] = (dims, np.asarray(self.delta_conf[0]))
            data_vars["delta_upper"] = (dims, np.asarray(self.delta_conf[1]))
        if self.S1_conf is not None:
            data_vars["S1_lower"] = (dims, np.asarray(self.S1_conf[0]))
            data_vars["S1_upper"] = (dims, np.asarray(self.S1_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
