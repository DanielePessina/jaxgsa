"""PCE result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _dims_and_coords
from gsax.problem import Problem

if TYPE_CHECKING:
    from gsax.shapley import ShapleyResult


@dataclass
class PCEResult:
    """Polynomial chaos expansion sensitivity analysis results.

    Stores Sobol indices computed analytically from the expansion
    coefficients, plus the fitted coefficients and multi-index so the
    surrogate can be reused through :meth:`predict`.

    Index arrays mirror the layout of the ``Y`` passed to ``pce.analyze``:
    leading dims are ``()`` for scalar ``(N,)`` outputs, ``(K,)`` for
    multi-output ``(N, K)``, and ``(T, K)`` for time-series ``(N, T, K)``.

    Attributes:
        S1: First-order Sobol indices, shape ``(..., D)``. Fraction of output
            variance explained by each parameter acting alone.
        ST: Total-order Sobol indices, shape ``(..., D)``. Fraction of output
            variance involving each parameter, interactions included;
            ``ST - S1`` measures how strongly a parameter interacts.
        S2: Second-order interaction indices, shape ``(..., D, D)`` with NaN
            on the diagonal. Upper and lower triangles are symmetric.
        problem: Problem definition.
        coefficients: Fitted PCE coefficients, shape ``(..., n_terms)`` with
            the term axis last. ``coefficients[..., 0]`` is the constant
            (mean) term of each output slice; all slices share one basis.
        multi_index: Per-term polynomial degrees, shape ``(n_terms, D)``;
            row ``t`` gives the degree of each input in term ``t``.
        order: Effective total polynomial degree used (may be lower than
            requested if the sample budget forced a reduction). A single
            int — the shared basis serves every output slice.
        loo_rmse: Leave-one-out cross-validation RMSE per output slice
            (shape ``(...)``: scalar / ``(K,)`` / ``(T, K)``), or None. In
            the units of ``Y``; compare against ``Y.std()`` -- a ratio near
            or above 1 means the surrogate (and its indices) is unreliable.
        explained_variance: Fraction of each output slice's sample variance
            captured by the expansion (shape ``(...)``: scalar / ``(K,)`` /
            ``(T, K)``, matching ``loo_rmse``), or None. Computed as the sum
            of squared non-constant coefficients divided by ``Var(Y)`` for
            that slice; NaN for constant (zero-variance) slices. Values well
            below 1 mean the surrogate misses variance; values above 1 mean
            it attributes more variance than the data holds (overfit).
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

    def predict(self, X: Array, *, batch_size: int | None = None) -> Array:
        """Predict outputs at new input rows using the fitted expansion.

        Rebuilds the polynomial basis at ``X`` and applies the coefficients
        fitted by ``pce.analyze`` -- no model evaluations are needed.
        Accuracy degrades outside the input region the surrogate was fitted
        on.

        Args:
            X: (N_new, D) new input points, in the same physical units as
                the ``X`` passed to ``pce.analyze``.
            batch_size: Rows of ``X`` to predict per batch. The basis tensors
                are linear in the batch size with a large per-row constant
                (``~3 * n_terms`` floats per row), so single-shot evaluation
                at large ``N_new`` can exhaust memory. ``None`` (default)
                derives a batch size from a fixed transient-memory budget
                (~512 MiB); an int fixes the rows per batch, and
                ``batch_size >= N_new`` forces a single-shot call. Each row's
                term contraction is independent, so batching only perturbs
                predictions at the level of floating-point reassociation.

        Returns:
            Predicted outputs mirroring the training ``Y`` layout:
            ``(N_new,)``, ``(N_new, K)``, or ``(N_new, T, K)``.

        Raises:
            ValueError: If ``X`` is not 2-D with one column per problem
                parameter, or ``batch_size`` is not a positive integer.
        """
        from gsax.pce._analyze import _predict_pce

        return _predict_pce(self, X, batch_size=batch_size)

    def shapley(self) -> "ShapleyResult":
        """Compute Shapley effects from this fitted PCE decomposition.

        Distributes each expansion term's variance contribution equally among
        the parameters active in it -- no extra model evaluations or refit.

        Returns:
            ShapleyResult with per-parameter Shapley effects ``Sh`` (summing
            to 1 per output slice), the matching ``S1``/``ST`` bounds, and
            this result's ``explained_variance`` diagnostic.

        Raises:
            ValueError: If this result carries no ``explained_variance``
                diagnostic (e.g. constructed by hand without one).

        Warns:
            UserWarning: If ``explained_variance`` indicates a pathological
                fit (well below 1, or above 1 -- overfit), making the Shapley
                effects unreliable.
        """
        from gsax.shapley._analyze import _shapley_from_pce

        return _shapley_from_pce(self)

    def __repr__(self) -> str:
        n_terms = self.coefficients.shape[-1]
        return f"PCEResult(D={len(self.problem.names)}, order={self.order}, n_terms={n_terms})"

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Optional coordinate values for the ``time``
                dimension of time-series results; defaults to ``0..T-1``.
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
