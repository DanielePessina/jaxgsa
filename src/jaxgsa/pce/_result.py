"""PCE result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem

if TYPE_CHECKING:
    from jaxgsa.shapley import ShapleyResult


@dataclass
class PCEResult(SurrogateResult):
    """Polynomial chaos expansion sensitivity analysis results.

    Stores Sobol indices computed analytically from the expansion
    coefficients, plus the fitted coefficients and multi-index so the
    surrogate can be reused through :meth:`predict`.

    Index arrays mirror the layout of the ``Y`` passed to ``pce.analyze``:
    ``(D,)`` for scalar output ``(N,)``, ``(K, D)`` for multi-output
    ``(N, K)``, and ``(T, K, D)`` for time-series output ``(N, T, K)``.

    Attributes:
        S1: First-order Sobol indices, shape ``(D,)`` / ``(K, D)`` /
            ``(T, K, D)``. Fraction of output variance explained by each
            parameter acting alone.
        ST: Total-order Sobol indices, shape ``(D,)`` / ``(K, D)`` /
            ``(T, K, D)``. Fraction of output variance involving each
            parameter, interactions included. ``ST - S1`` measures how
            strongly a parameter interacts.
        S2: Second-order interaction indices, shape ``(D, D)`` /
            ``(K, D, D)`` / ``(T, K, D, D)``, with NaN on the diagonal.
            Upper and lower triangles are symmetric.
        problem: Problem definition used for the analysis.
        coefficients: Fitted PCE coefficients, shape ``(n_terms,)`` /
            ``(K, n_terms)`` / ``(T, K, n_terms)``, with the term axis last.
            ``coefficients[..., 0]`` is the constant (mean) term of each
            output slice. All slices share one basis.
        multi_index: Per-term polynomial degrees, shape ``(n_terms, D)``.
            Row ``t`` gives the degree of each input in term ``t``.
        order: Effective total polynomial degree used. It may be lower than
            requested if the sample budget forced a reduction. It is a single
            int, because the shared basis serves every output slice.
        loo_rmse: Leave-one-out cross-validation RMSE per output slice, shape
            ``()`` / ``(K,)`` / ``(T, K)``, or None. It is in the units of
            ``Y``. Compare it against ``Y.std()``: a ratio near or above 1
            means the surrogate (and its indices) is unreliable.
        explained_variance: Fraction of each output slice's sample variance
            captured by the expansion, shape ``()`` / ``(K,)`` / ``(T, K)``
            to match ``loo_rmse``, or None. Computed as the sum of squared
            non-constant coefficients divided by ``Var(Y)`` for that slice,
            and NaN for constant (zero-variance) slices. Values well below 1
            mean the surrogate misses variance. Values above 1 mean it
            attributes more variance than the data holds (overfit).
        streamed: True when the fit ran the row-streamed path, False when it
            ran in one pass. Both paths solve the same normal equations and
            report the same leave-one-out error; they differ only in float32
            summation order and in peak memory. The streamed path engages
            when ``batch_size`` is an explicit int, or when the one-pass fit
            would exceed the memory budget (see
            :func:`jaxgsa.config.set_memory_budget`). Read it when a fit takes
            much longer than expected: True means the budget engaged.
    """

    S1: Array
    ST: Array
    S2: Array
    problem: Problem
    coefficients: Array
    multi_index: np.ndarray
    order: int
    loo_rmse: Array | None = None
    explained_variance: Array | None = None
    streamed: bool = False

    def _predict_plan(self, X: Array) -> _PredictPlan:
        """Plan a batched evaluation of the fitted expansion at ``X``.

        Rebuilds the polynomial basis at ``X`` and contracts it with the
        coefficients fitted by ``pce.analyze``. No model evaluations are
        needed. The basis tensors carry ``~3 * n_terms`` transient floats
        per prediction row, which prices the automatic batch size.
        See :meth:`predict` for the full contract.
        """
        from jaxgsa.pce._analyze import _pce_predict_plan

        return _pce_predict_plan(self, X)

    def shapley(self) -> "ShapleyResult":
        """Compute Shapley effects from this fitted PCE decomposition.

        Splits each expansion term's variance contribution equally among the
        parameters active in it. This needs no extra model evaluations and no
        refit.

        Returns:
            ShapleyResult with per-parameter Shapley effects ``Sh`` (summing
            to 1 per output slice), the matching ``S1``/``ST`` bounds, and
            this result's ``explained_variance`` diagnostic.

        Raises:
            ValueError: If this result carries no ``explained_variance``
                diagnostic (e.g. constructed by hand without one).

        Warns:
            JaxgsaWarning: If ``explained_variance`` indicates a pathological
                fit (well below 1, or above 1, which is an overfit), making
                the Shapley effects unreliable.
        """
        from jaxgsa.shapley._analyze import _shapley_result_from_variances

        explained = self.explained_variance
        if explained is None:
            raise ValueError("PCEResult does not contain explained-variance diagnostics")
        # Orthonormality makes each squared non-constant coefficient a
        # partial variance; multi_index[1:] > 0 IS the membership matrix.
        partial = self.coefficients[..., 1:] ** 2
        membership = np.asarray(self.multi_index[1:] > 0)
        return _shapley_result_from_variances(
            partial,
            membership,
            explained,
            total=partial.sum(axis=-1),
            problem=self.problem,
            backend="pce",
            order=self.order,
        )

    def __repr__(self) -> str:
        """Return a concise summary showing the size of the fitted basis."""
        n_terms = self.coefficients.shape[-1]
        return f"PCEResult(D={len(self.problem.names)}, order={self.order}, n_terms={n_terms})"

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the ``time`` dimension of
                time-series results. Defaults to ``0..T-1``.

        Returns:
            An ``xr.Dataset`` with variables ``S1``, ``ST``, ``S2`` and, when
            present, ``loo_rmse`` and ``explained_variance``.
        """
        s1 = np.asarray(self.S1)
        dims, coords = _dims_and_coords(s1.ndim, s1.shape, self.problem, time_coords)
        data_vars: dict = {
            "S1": (dims, s1),
            "ST": (dims, np.asarray(self.ST)),
        }

        # S2 is a symmetric (..., D, D) matrix; separate coord names
        # (param_i, param_j) avoid xarray dimension-name conflicts with the
        # 1-D "param" coord used by S1/ST.
        param_names = list(self.problem.names)
        data_vars["S2"] = (
            (*dims[:-1], "param_i", "param_j"),
            np.asarray(self.S2),
        )
        coords["param_i"] = param_names
        coords["param_j"] = param_names

        # LOO RMSE is a per-slice diagnostic: no dims for scalar output,
        # (output,) / (time, output) otherwise.
        if self.loo_rmse is not None:
            data_vars["loo_rmse"] = (dims[:-1], np.asarray(self.loo_rmse))
        if self.explained_variance is not None:
            data_vars["explained_variance"] = (
                dims[:-1],
                np.asarray(self.explained_variance),
            )

        return xr.Dataset(data_vars, coords=coords)
