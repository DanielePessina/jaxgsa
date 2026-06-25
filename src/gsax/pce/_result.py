"""PCE result dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax.problem import Problem


@dataclass
class PCEResult:
    """Polynomial chaos expansion sensitivity analysis results.

    Stores first-order (S1) and total-order (ST) Sobol indices computed
    analytically from the expansion coefficients, plus the fitted
    coefficients and multi-index for emulation.

    Attributes:
        S1: First-order Sobol indices, shape ``(D,)``.
        ST: Total-order Sobol indices, shape ``(D,)``.
        S2: Second-order interaction indices, shape ``(D, D)`` with NaN
            on the diagonal. Upper and lower triangles are symmetric.
        problem: Problem definition.
        coefficients: Fitted PCE coefficients, shape ``(n_terms,)``.
        multi_index: Multi-index array, shape ``(n_terms, D)``.
        order: Total polynomial degree used.
        loo_rmse: Leave-one-out cross-validation RMSE, or None.
    """

    S1: Array
    ST: Array
    S2: Array
    problem: Problem
    coefficients: Array
    multi_index: np.ndarray
    order: int
    loo_rmse: Array | None = None

    def __repr__(self) -> str:
        n_terms = self.coefficients.shape[0]
        return (
            f"PCEResult(D={len(self.problem.names)}, order={self.order}, "
            f"n_terms={n_terms})"
        )

    def to_dataset(self) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset."""
        param_names = list(self.problem.names)
        coords: dict = {"param": param_names}
        data_vars: dict = {
            "S1": (("param",), np.asarray(self.S1)),
            "ST": (("param",), np.asarray(self.ST)),
        }

        data_vars["S2"] = (
            ("param_i", "param_j"),
            np.asarray(self.S2),
        )
        coords["param_i"] = param_names
        coords["param_j"] = param_names

        if self.loo_rmse is not None:
            data_vars["loo_rmse"] = ((), np.asarray(self.loo_rmse))

        return xr.Dataset(data_vars, coords=coords)
