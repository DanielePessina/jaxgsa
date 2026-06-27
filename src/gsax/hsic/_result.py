"""Defines the HSICResult dataclass for kernel-based sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class HSICResult:
    """HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis results.

    For scalar-output models, index arrays have shape ``(D,)``.
    For multi-output models, index arrays have shape ``(K, D)``.
    For time-series multi-output, index arrays have shape ``(T, K, D)``.

    Attributes:
        R2_HSIC: Normalized first-order HSIC index in [0, 1].
        T_HSIC: Total-order HSIC index.
        p_values: Permutation p-values for R2_HSIC.
        hsic_raw: Unnormalized HSIC(X_i, Y) values.
        problem: Problem definition used for the analysis.
    """

    R2_HSIC: Array
    T_HSIC: Array
    p_values: Array
    hsic_raw: Array
    problem: Problem

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "R2_HSIC": self.R2_HSIC.shape,
            "T_HSIC": self.T_HSIC.shape,
            "p_values": self.p_values.shape,
            "hsic_raw": self.hsic_raw.shape,
        }
        return f"HSICResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            Dataset with variables R2_HSIC, T_HSIC, p_values, hsic_raw.
        """
        param_names = list(self.problem.names)
        r2_arr = np.asarray(self.R2_HSIC)
        ndim = r2_arr.ndim

        if ndim == 1:
            dims = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            K = r2_arr.shape[0]
            onames = _default_output_names(K, self.problem)
            dims = ("output", "param")
            coords = {"output": onames, "param": param_names}
        elif ndim == 3:
            T = r2_arr.shape[0]
            K = r2_arr.shape[1]
            onames = _default_output_names(K, self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims = ("time", "output", "param")
            coords = {"time": tcoords, "output": onames, "param": param_names}
        else:
            raise ValueError(f"Unexpected R2_HSIC.ndim={ndim}")

        data_vars: dict = {
            "R2_HSIC": (dims, r2_arr),
            "T_HSIC": (dims, np.asarray(self.T_HSIC)),
            "p_values": (dims, np.asarray(self.p_values)),
            "hsic_raw": (dims, np.asarray(self.hsic_raw)),
        }
        return xr.Dataset(data_vars, coords=coords)
