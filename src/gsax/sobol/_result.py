"""Defines the SAResult dataclass for storing sensitivity analysis results."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _dims_and_coords
from gsax.problem import Problem


@dataclass
class SAResult:
    """Sobol sensitivity analysis results.

    Stores first-order (S1), total-order (ST), and optionally second-order (S2)
    Sobol indices, with optional bootstrap confidence intervals.

    Shapes follow the convention ``(T, K, D)`` for time-resolved analyses or
    ``(K, D)`` when the time dimension is squeezed, where *K* is the number of
    outputs and *D* the number of parameters.

    ``S2`` is stored as a symmetric ``(..., D, D)`` matrix. Only the upper
    triangle is estimated directly; the lower triangle mirrors it for
    convenience, and the diagonal is undefined and therefore set to ``NaN``.

    Confidence interval arrays (``*_conf``) have an extra leading dimension of
    size 2 representing ``[lower, upper]`` bounds. ``S2_conf`` follows the same
    symmetric-with-``NaN``-diagonal contract as ``S2``.
    """

    S1: Array  # (T, K, D) or (K, D) if time squeezed
    ST: Array
    S2: Array | None  # (..., D, D), symmetric, diagonal NaN, None if not computed
    problem: Problem
    S1_conf: Array | None = None  # (2, T, K, D) or squeezed; [lower, upper]
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    nan_counts: dict[str, int] | None = None

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "S1": self.S1.shape,
            "ST": self.ST.shape,
            "S2": self.S2.shape if self.S2 is not None else None,
        }
        if self.S1_conf is not None:
            shapes["S1_conf"] = self.S1_conf.shape
        if self.ST_conf is not None:
            shapes["ST_conf"] = self.ST_conf.shape
        if self.S2_conf is not None:
            shapes["S2_conf"] = self.S2_conf.shape
        return f"SAResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``S1.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``S1``, ``ST``, and optionally
            ``S2``, ``S1_lower/upper``, ``ST_lower/upper``, ``S2_lower/upper``.
        """
        dims_s1, coords = _dims_and_coords(self.S1.ndim, self.S1.shape, self.problem, time_coords)
        param_names = list(self.problem.names)

        data_vars: dict = {
            "S1": (dims_s1, np.asarray(self.S1)),
            "ST": (dims_s1, np.asarray(self.ST)),
        }

        # S2 has two parameter axes (interaction between param_i and param_j),
        # so it uses separate coordinate names to avoid an xarray dimension clash.
        if self.S2 is not None:
            dims_s2 = (*dims_s1[:-1], "param_i", "param_j")
            data_vars["S2"] = (dims_s2, np.asarray(self.S2))
            coords["param_i"] = param_names
            coords["param_j"] = param_names

        # Split the (2, ...) confidence arrays into *_lower and *_upper
        # variables so users can select bounds without integer indexing.
        for name, arr in [
            ("S1", self.S1_conf),
            ("ST", self.ST_conf),
        ]:
            if arr is not None:
                data_vars[f"{name}_lower"] = (dims_s1, np.asarray(arr[0]))
                data_vars[f"{name}_upper"] = (dims_s1, np.asarray(arr[1]))

        if self.S2_conf is not None and self.S2 is not None:
            dims_s2 = (*dims_s1[:-1], "param_i", "param_j")
            data_vars["S2_lower"] = (dims_s2, np.asarray(self.S2_conf[0]))
            data_vars["S2_upper"] = (dims_s2, np.asarray(self.S2_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
