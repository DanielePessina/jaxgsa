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

    Stores Sobol indices computed analytically from the expansion
    coefficients, plus the fitted coefficients and multi-index so the
    surrogate can be reused for prediction via ``pce.emulate``.

    Attributes:
        S1: First-order Sobol indices, shape ``(D,)``. Fraction of output
            variance explained by each parameter acting alone.
        ST: Total-order Sobol indices, shape ``(D,)``. Fraction of output
            variance involving each parameter, interactions included;
            ``ST - S1`` measures how strongly a parameter interacts.
        S2: Second-order interaction indices, shape ``(D, D)`` with NaN
            on the diagonal. Upper and lower triangles are symmetric.
        problem: Problem definition.
        coefficients: Fitted PCE coefficients, shape ``(n_terms,)``.
            ``coefficients[0]`` is the constant (mean) term.
        multi_index: Per-term polynomial degrees, shape ``(n_terms, D)``;
            row ``t`` gives the degree of each input in term ``t``.
        order: Effective total polynomial degree used (may be lower than
            requested if the sample budget forced a reduction).
        loo_rmse: Leave-one-out cross-validation RMSE, or None. In the
            units of ``Y``; compare against ``Y.std()`` -- a ratio near or
            above 1 means the surrogate (and its indices) is unreliable.
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
        return f"PCEResult(D={len(self.problem.names)}, order={self.order}, n_terms={n_terms})"

    def to_dataset(self) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset."""
        param_names = list(self.problem.names)
        # S1 and ST are 1-D vectors indexed by parameter name.
        coords: dict = {"param": param_names}
        data_vars: dict = {
            "S1": (("param",), np.asarray(self.S1)),
            "ST": (("param",), np.asarray(self.ST)),
        }

        # S2 is a symmetric (D x D) matrix; separate coord names (param_i, param_j)
        # avoid xarray dimension-name conflicts with the 1-D "param" coord.
        data_vars["S2"] = (
            ("param_i", "param_j"),
            np.asarray(self.S2),
        )
        coords["param_i"] = param_names
        coords["param_j"] = param_names

        # LOO RMSE is a scalar diagnostic (no dimensions).
        if self.loo_rmse is not None:
            data_vars["loo_rmse"] = ((), np.asarray(self.loo_rmse))

        return xr.Dataset(data_vars, coords=coords)
