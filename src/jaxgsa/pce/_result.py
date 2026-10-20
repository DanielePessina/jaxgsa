"""PCE result dataclass."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import (
    CIInfo,
    FieldSpec,
    ResultSchema,
    SchemaResult,
    aggregate_indices,
)
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
        rmse: In-sample root-mean-square error of the fit per output slice,
            shape ``()`` / ``(K,)`` / ``(T, K)``, or None. It is in the units
            of ``Y``.
        cv_rmse: Exact leave-one-out cross-validation RMSE per output slice,
            same shape as ``rmse``, or None. It is in the units of ``Y``.
            Compare it against ``Y.std()``: a ratio near or above 1 means the
            surrogate (and its indices) is unreliable.
        explained_variance: Fraction of each output slice's sample variance
            captured by the expansion, shape ``()`` / ``(K,)`` / ``(T, K)``
            to match ``rmse``, or None. Computed as ``1 - rmse**2 / Var(Y)``
            for that slice, over the rows that were fitted, and NaN for
            constant (zero-variance) slices. This is the coefficient of
            determination of the fit, so it lies in ``[0, 1]``: values near 1
            mean the expansion reproduces the sample, and values well below 1
            mean it misses variance. It measures the fit *in sample* and says
            nothing about prediction; ``cv_rmse`` is the out-of-sample
            number, and a high ``explained_variance`` beside a ``cv_rmse``
            near ``Y.std()`` is the signature of an overfit.
        streamed: True when the fit ran the row-streamed path, False when it
            ran in one pass. Both paths solve the same least squares by QR and
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
            inherits the same bias. ``cv_rmse`` and ``explained_variance``
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
    rmse: Array | None = None
    cv_rmse: Array | None = None
    explained_variance: Array | None = None
    streamed: bool = False
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    ci: CIInfo | None = None
    _aggregated: bool = field(default=False, repr=False, compare=False)

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", interval=True),
            FieldSpec("ST", interval=True),
            FieldSpec("S2", "pair", interval=True),
            FieldSpec("rmse", "slice"),
            FieldSpec("cv_rmse", "slice"),
            FieldSpec("explained_variance", "slice"),
        ),
        meta=("order", "streamed"),
    )

    def _require_surrogate(self, method: str) -> None:
        """Raise if this result was aggregated and so holds indices only."""
        if self._aggregated:
            raise ValueError(
                "this PCEResult was aggregated over outputs, so it holds indices only "
                f"and is not a surrogate. Call {method} on the result before aggregate()."
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

        self._require_surrogate("predict()")

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
            ValueError: If this result was aggregated over outputs, or
                carries no ``explained_variance`` diagnostic (e.g. constructed
                by hand without one).

        Warns:
            JaxgsaWarning: If ``explained_variance`` sits well below 1, so the
                expansion missed much of the sample variance and the Shapley
                effects it carries are unreliable.
        """
        from jaxgsa.shapley._engine import _shapley_result_from_variances

        self._require_surrogate("shapley()")
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

    def aggregate(self, *, weights: Literal["variance", "equal"] = "variance") -> Self:
        """Collapse the output axes into one index per input.

        Computes ``S_agg_i = sum_t w_t * S_{i,t}`` over every output slice
        ``t`` (the ``T`` and ``K`` axes together). With ``weights="variance"``
        the weight is ``w_t = V_t / sum(V)``, where ``V_t`` is the variance
        of output ``t``. This is the generalised Sobol index of Gamboa,
        Janon, Klein and Lagnoux (2014, Electron. J. Stat. 8:575-603,
        doi:10.1214/14-EJS895), also used by Lamboni, Monod and Makowski
        (2011, Reliab. Eng. Syst. Saf. 96:450-459,
        doi:10.1016/j.ress.2010.12.002). It is the share of the total output
        variance that the input explains. With ``weights="equal"`` every
        output counts the same, ``w_t = 1 / n_outputs``.

        The method applies to every index field this result holds. The
        result keeps its class and its output axes, now of size 1, so
        ``to_dataset`` and the printed summary work as before.

        The weight of an output is the variance of the fitted surrogate under
        the input measure. The basis is orthonormal, so that variance is the
        sum of the squared non-constant coefficients. This is the same
        variance the indices divide by.

        The aggregated result is a summary, not a surrogate. Its
        ``coefficients`` are ``NaN``, and ``rmse``, ``cv_rmse`` and
        ``explained_variance`` are ``None``, because none of them aggregates
        across outputs. :meth:`predict` and :meth:`shapley` raise
        ``ValueError`` on it. Call them before ``aggregate()``.

        A slice of zero variance (a constant output) has weight 0 and adds
        nothing. A result with one output slice comes back with the same
        index values.

        Confidence intervals do not aggregate bound by bound. When the
        analysis ran with ``keep_replicates=True``, each bootstrap draw is
        aggregated and the interval is drawn again from those draws, at the
        same level and with the same rule. The weights stay fixed at the
        point-estimate values in every draw, so the interval leaves out the
        sampling noise of the weights themselves. Otherwise the ``*_conf`` fields
        and ``ci`` become ``None`` and one ``JaxgsaWarning`` says that
        ``keep_replicates=True`` is needed.

        The output axes collapse to size 1. A result with no output axes
        (``S1`` of shape ``(D,)``) is returned unchanged.

        Args:
            weights: ``"variance"`` (default) or ``"equal"``.

        Returns:
            A result of the same class with output axes of size 1.

        Raises:
            ValueError: If ``weights`` is not one of the two names.

        Warns:
            JaxgsaWarning: If the result has bootstrap intervals but kept no
                replicates, so the intervals are dropped.
        """
        coefficients = jnp.asarray(self.coefficients)
        variance = jnp.sum(coefficients[..., 1:] ** 2, axis=-1)
        # A constant output has NaN indices, but its fitted coefficients are
        # only near zero. Give it weight 0, as the exact variance would. A
        # failed fit has non-finite coefficients: its variance is unknown, not
        # zero, so its NaN weight makes the aggregate NaN.
        constant = jnp.isnan(jnp.asarray(self.S1)).all(axis=-1) & jnp.isfinite(coefficients).all(
            axis=-1
        )
        variance = jnp.where(constant, 0.0, variance)
        updates = aggregate_indices(self, weights=weights, variance=variance)
        if updates:
            lead = updates["S1"].shape[:-1]
            n_terms = np.shape(self.coefficients)[-1]
            updates["coefficients"] = jnp.full((*lead, n_terms), jnp.nan)
            updates["rmse"] = None
            updates["cv_rmse"] = None
            updates["explained_variance"] = None
            updates["_aggregated"] = True
        return dataclasses.replace(self, **updates)
