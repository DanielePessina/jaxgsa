"""Result container for PAWN distribution-based sensitivity indices."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(frozen=True, repr=False)
class PAWNResult(SchemaResult):
    """PAWN sensitivity analysis results.

    The PAWN index is the Kolmogorov-Smirnov distance between the
    unconditional output CDF and the CDF conditional on one parameter. The
    per-bin distances are aggregated across the conditioning bins by median,
    max, or mean.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    Attributes:
        pawn: PAWN sensitivity indices per parameter, shape ``(..., D)``, in
            [0, 1]. A value of 0 means fixing the parameter leaves the output
            distribution unchanged, and larger values mean stronger influence.
        pawn_conf: Bootstrap confidence interval for ``pawn``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        n_valid_bins: How many conditioning bins actually contributed to each
            index, same shape as ``pawn``. A bin contributes only when it
            holds at least two samples; an under-filled bin yields ``NaN``
            from the KS kernel and is dropped by the nan-aware aggregation.
            Bin occupancy depends on the inputs alone, so the count is
            constant across the output (T/K) axes; it is broadcast to
            ``pawn``'s shape so the exported dataset aligns the two. A
            parameter whose count is well below ``n_bins`` (or below its
            level count, for a categorical parameter) rests its index on few
            KS values: use fewer bins or more samples.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    pawn: Array
    pawn_conf: Array | None
    n_valid_bins: Array
    problem: Problem
    invalid: InvalidReport
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="pawn",
        fields=(
            FieldSpec("pawn", "param", interval=True),
            FieldSpec("n_valid_bins", "param"),
        ),
    )
