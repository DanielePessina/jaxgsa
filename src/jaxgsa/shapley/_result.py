"""Shapley-effect result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(frozen=True, repr=False)
class ShapleyResult(SchemaResult):
    """Shapley-effect sensitivity analysis results.

    Stores Shapley effects computed analytically from a fitted surrogate's
    variance decomposition, alongside the first-order (S1) and total-order
    (ST) indices derived from the same decomposition so the ordering
    ``S1 <= Sh <= ST`` is visible at a glance.

    On a correlated problem with ``backend="hdmr"`` and
    ``include_correlative=True``, ``Sh`` is not the conditional-variance
    Shapley effect (Owen 2014; Owen & Prieur 2017). It is an ANCOVA variance
    allocation: the HDMR structural terms (``Sa``) plus the correlative terms
    (``Sb``) that Cost-of-play credits back to each parameter through the
    Shapley formula. The two quantities agree only when the inputs are
    independent. On an exact linear-Gaussian check (D=2, rho=0.5) the true
    Shapley effects are ``[0.339, 0.661]`` against this allocation's
    ``[0.284, 0.716]``; on an asymmetric D=3 check, ``[0.565, 0.409, 0.026]``
    against ``[0.675, 0.320, 0.005]``. See ``include_correlative`` below.

    All indices are normalized by the surrogate's total decomposed variance
    ``sum_u V_u``, so ``Sh.sum(axis=-1)`` is exactly 1 (the Shapley-value
    efficiency property; Owen 2014). ``explained_variance`` reports
    separately how much of the output variance the surrogate captured: the
    coefficient of determination of the fit for the ``"pce"`` backend, and
    the decomposed fraction ``sum_u V_u / Var(Y)`` for ``"hdmr"``.

    For the ``"hdmr"`` backend the indices are normalized by ``sum_u V_u``
    rather than ``Var(Y)``. They therefore relate to HDMR indices by a factor
    of ``explained_variance``: multiply to recover the HDMR scale. For PCE,
    S1/ST match the fitted result exactly.

    Shapes mirror the layout of the ``Y`` used to fit the source surrogate:
    ``(D,)`` for scalar output, ``(K, D)`` for multi-output, and
    ``(T, K, D)`` for time-resolved analyses.

    Attributes:
        Sh: Shapley effects, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
            Sums to 1 along the parameter axis.
        S1: First-order indices from the same surrogate, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        ST: Total-order indices from the same surrogate, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        problem: Problem definition used for the analysis.
        backend: Surrogate backend used, ``"hdmr"`` or ``"pce"``.
        explained_variance: How much of the output variance the surrogate
            captured, shape ``()`` / ``(K,)`` / ``(T, K)``. The quantity
            depends on the backend. For ``"pce"`` it is the coefficient of
            determination of the fit, the sample variance of the fitted
            values over the sample variance of ``Y``, so it lies in
            ``[0, 1]``; read ``loo_rmse`` on the PCE result for the
            out-of-sample view. For ``"hdmr"`` it is ``sum_u V_u / Var(Y)``,
            which can go above 1 when an overfit surrogate over-counts
            shared variance. Both are close to 1 for a good fit and below 1
            when truncation or fit error leaves variance unexplained.
        order: Effective surrogate order actually used. For ``"pce"`` it is
            the polynomial degree, which may be reduced from the requested
            value to fit the sample budget. For ``"hdmr"`` it is the HDMR
            expansion order.
        invalid: What the non-finite check found in ``(X, Y)`` and what the
            ``on_invalid`` policy did about it, as reported by whichever
            backend ran. See :class:`jaxgsa._core.invalid.InvalidReport`.
        include_correlative: Whether the correlative ANCOVA variance (``Sb``)
            was folded into the allocation (HDMR backend only). When ``True``
            the indices credit variance shared through input correlation, so
            ``Sh``/``S1``/``ST`` may be negative and the ordering
            ``S1 <= Sh <= ST`` need not hold. Efficiency (``Sh`` sums to 1) is
            preserved regardless. ``Sh`` under ``True`` is an ANCOVA variance
            allocation, not the conditional-variance Shapley effect; see the
            class docstring.
        Sh_conf: Bootstrap confidence interval for ``Sh``, shape ``(2, ...)``
            for ``[lower, upper]``. ``None`` when ``n_bootstrap=0``. Each
            replicate refits the backend surrogate on a row resample, so the
            interval measures the sampling variability of ``(X, Y)``
            propagated through the fit. It says nothing about how well the
            surrogate represents the model: every replicate shares the same
            basis, so a systematic misfit sits inside every interval.
            ``explained_variance`` is what reports that. Available only from
            ``jaxgsa.shapley.analyze``; ``result.shapley()`` on a fitted
            surrogate has one fit and cannot resample it.
        S1_conf: Bootstrap confidence interval for ``S1``, as ``Sh_conf``.
        ST_conf: Bootstrap confidence interval for ``ST``, as ``Sh_conf``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the draws themselves when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    Sh: Array
    S1: Array
    ST: Array
    problem: Problem
    backend: Literal["hdmr", "pce"]
    explained_variance: Array
    order: int
    invalid: InvalidReport
    include_correlative: bool = False
    Sh_conf: Array | None = None
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="Sh",
        fields=(
            FieldSpec("Sh", interval=True),
            FieldSpec("S1", interval=True),
            FieldSpec("ST", interval=True),
            FieldSpec("explained_variance", "slice"),
        ),
        meta=("backend", "order", "include_correlative"),
    )
