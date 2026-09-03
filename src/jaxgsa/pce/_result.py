"""PCE result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa._core.surrogate import SurrogateResult, _PredictPlan
from jaxgsa.problem import Problem

if TYPE_CHECKING:
    from jaxgsa.shapley import ShapleyResult


@dataclass(frozen=True, repr=False)
class PCEResult(SchemaResult, SurrogateResult):
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
        invalid: What the non-finite check found in ``(X, Y)`` and what the
            ``on_invalid`` policy did about it. See
            :class:`jaxgsa._core.invalid.InvalidReport`. ``n_invalid == 0``
            means the check ran and found nothing.
        loo_rmse: Leave-one-out cross-validation RMSE per output slice, shape
            ``()`` / ``(K,)`` / ``(T, K)``, or None. It is in the units of
            ``Y``. Compare it against ``Y.std()``: a ratio near or above 1
            means the surrogate (and its indices) is unreliable.
        explained_variance: Fraction of each output slice's sample variance
            captured by the expansion, shape ``()`` / ``(K,)`` / ``(T, K)``
            to match ``loo_rmse``, or None. Computed as the sample variance
            of the fitted values divided by the sample variance of ``Y`` for
            that slice, both over the rows that were fitted, and NaN for
            constant (zero-variance) slices. This is the coefficient of
            determination of the fit, so it lies in ``[0, 1]``: values near 1
            mean the expansion reproduces the sample, and values well below 1
            mean it misses variance. It measures the fit *in sample* and says
            nothing about prediction; ``loo_rmse`` is the out-of-sample
            number, and a high ``explained_variance`` beside a ``loo_rmse``
            near ``Y.std()`` is the signature of an overfit.
        streamed: True when the fit ran the row-streamed path, False when it
            ran in one pass. Both paths solve the same normal equations and
            report the same leave-one-out error; they differ only in float32
            summation order and in peak memory. The streamed path engages
            when ``batch_size`` is an explicit int, or when the one-pass fit
            would exceed the memory budget (see
            :func:`jaxgsa.config.set_memory_budget`). Read it when a fit takes
            much longer than expected: True means the budget engaged.
        S1_conf: Bootstrap confidence interval for ``S1``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``. Each replicate refits the expansion on a row
            resample, so the interval measures the sampling variability of
            ``(X, Y)`` propagated through the fit. It says nothing about
            truncation error: every replicate uses the same basis and
            inherits the same bias. ``loo_rmse`` and ``explained_variance``
            are what report that.
        ST_conf: Bootstrap confidence interval for ``ST``, as ``S1_conf``.
        S2_conf: Bootstrap confidence interval for ``S2``, as ``S1_conf``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the draws themselves when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    S1: Array
    ST: Array
    S2: Array
    problem: Problem
    coefficients: Array
    multi_index: np.ndarray
    order: int
    invalid: InvalidReport
    loo_rmse: Array | None = None
    explained_variance: Array | None = None
    streamed: bool = False
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", interval=True),
            FieldSpec("ST", interval=True),
            FieldSpec("S2", "pair", interval=True),
            FieldSpec("loo_rmse", "slice"),
            FieldSpec("explained_variance", "slice"),
        ),
        meta=("order", "streamed"),
    )

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
            JaxgsaWarning: If ``explained_variance`` sits well below 1, so the
                expansion missed much of the sample variance and the Shapley
                effects it carries are unreliable.
        """
        from jaxgsa.shapley._engine import _shapley_result_from_variances

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
            problem=self.problem,
            backend="pce",
            order=self.order,
            invalid=self.invalid,
        )
