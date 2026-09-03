"""Result container for Kucherenko dependent-input sensitivity indices."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(frozen=True, repr=False)
class KucherenkoResult(SchemaResult):
    """Kucherenko sensitivity indices, returned by :func:`jaxgsa.kucherenko.analyze`.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    ``S1`` estimates ``V(E(Y|X_i)) / V(Y)`` and ``ST`` estimates
    ``E(V(Y|X_{~i})) / V(Y)`` under the problem's declared dependence
    structure. Under independent inputs they are the classic Sobol' first-order
    and total-order indices. Under a declared correlation ``S1`` is
    correlation-inclusive and matches ``jaxgsa.vkoga``'s ``S_TC``, and ``ST``
    is correlation-exclusive and matches ``S_TU``. In that case ``ST >= S1`` no
    longer holds in general.

    Attributes:
        S1: First-order (correlation-inclusive) indices, shape ``(..., D)``.
            Rank parameters by ``S1`` to decide which ones to measure.
        ST: Total-order (correlation-exclusive) indices, shape ``(..., D)``.
            Rank parameters by ``ST`` to decide which ones to fix.
        problem: Problem definition used for the analysis.
        variance: Output variance under the joint input measure, one value per
            output slice (shape ``()``, ``(K,)``, or ``(T, K)``).
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. See :class:`jaxgsa._core.invalid.InvalidReport`.
        S1_conf: Bootstrap confidence interval for ``S1``, shape ``(2, ...)``
            for ``[lower, upper]``. ``None`` when ``n_bootstrap=0``.
        ST_conf: The same for ``ST``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without a
            bootstrap. The resampling unit is one base point, not one design
            row. ``variance`` carries no interval, because it is the
            denominator of both indices rather than a sensitivity measure.
            See :class:`jaxgsa._core.result.CIInfo`.
    """

    S1: Array
    ST: Array
    problem: Problem
    variance: Array
    invalid: InvalidReport
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", interval=True),
            FieldSpec("ST", interval=True),
            FieldSpec("variance", "slice"),
        ),
        meta=("is_correlated",),
    )

    @property
    def is_correlated(self) -> bool:
        """Return ``True`` when the problem declares a non-identity correlation."""
        return self.problem.has_correlated_inputs
