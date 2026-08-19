"""Result container for Borgonovo delta sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class DeltaResult(SchemaResult):
    """Borgonovo delta sensitivity analysis results.

    The delta index is moment-independent: it says how much knowing a
    parameter shifts the whole output density, on a [0, 1] scale. This
    container holds that index and the given-data first-order Sobol index
    computed from the same class partition. It also holds the optional
    bootstrap confidence intervals.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    Attributes:
        delta: Borgonovo delta index per parameter, shape ``(..., D)``. A
            value of 0 means the output distribution does not change with
            the parameter, and 1 means the parameter fully determines it.
            The index is bias-corrected when the analysis ran with
            ``bias_correct=True`` and ``n_bootstrap > 0``, and the
            corrected estimate can fall marginally below 0 for weak
            parameters.
        delta_conf: Bootstrap confidence interval for ``delta``,
            shape ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        S1: Given-data first-order Sobol index per parameter, shape
            ``(..., D)``.
        S1_conf: Bootstrap confidence interval for ``S1``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    delta: Array
    delta_conf: Array | None
    S1: Array
    S1_conf: Array | None
    problem: Problem
    invalid: InvalidReport
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="delta",
        fields=(
            FieldSpec("delta", "param", interval=True),
            FieldSpec("S1", "param", interval=True),
        ),
    )
