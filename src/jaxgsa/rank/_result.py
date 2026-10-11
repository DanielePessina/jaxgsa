"""Result container for rank-based first-order Sobol' indices."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(frozen=True, repr=False)
class RankResult(SchemaResult):
    """Rank-based first-order Sobol' analysis results.

    The index is the Gamboa et al. (2022) rank estimator of
    ``S1_i = Var(E[Y | X_i]) / Var(Y)``. Index arrays have shape ``(D,)`` for a
    scalar output, ``(K, D)`` for a multi-output model, and ``(T, K, D)`` for a
    time-resolved analysis.

    Attributes:
        S1: First-order indices per parameter, shape ``(..., D)``. The
            estimator is not clipped. It never exceeds 1, but a small sample can
            give a negative value. A value near 0 means the parameter alone
            explains little of the output variance. ``NaN`` when the output
            has zero variance.
        S1_conf: Bootstrap confidence interval for ``S1``, shape ``(2, ...)``
            for ``[lower, upper]``. ``None`` when ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    S1: Array
    S1_conf: Array | None
    problem: Problem
    invalid: InvalidReport
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="S1",
        fields=(FieldSpec("S1", "param", interval=True),),
    )
