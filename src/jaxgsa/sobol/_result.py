"""Result container for Sobol sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False, frozen=True)
class SobolResult(SchemaResult):
    """Sobol sensitivity analysis results, returned by :func:`jaxgsa.sobol.analyze`.

    Stores first-order (S1), total-order (ST), and optionally second-order (S2)
    Sobol indices, with optional bootstrap confidence intervals. Call
    :meth:`to_dataset` for a labeled xarray view keyed by parameter, output and
    time names.

    Index shapes track the shape of the analyzed output ``Y``. ``D`` is the
    number of parameters, ``K`` the number of outputs, and ``T`` the number of
    time steps. A scalar output gives ``(D,)``, a multi-output analysis gives
    ``(K, D)``, and a time-resolved analysis gives ``(T, K, D)``.

    Attributes:
        S1: First-order Sobol indices, shape ``(D,)`` / ``(K, D)`` /
            ``(T, K, D)``.
        ST: Total-order Sobol indices, same shape as ``S1``.
        S2: Second-order Sobol indices, shape ``(D, D)`` / ``(K, D, D)`` /
            ``(T, K, D, D)``, or ``None`` when they were not computed.
            ``S2[j, k]`` and ``S2[k, j]`` are estimated independently, from
            the same formula with j and k swapped, and are averaged
            together: the two differ by Monte Carlo noise, not
            floating-point drift, and averaging cuts S2's RMSE by 5-16%
            (measured on Ishigami). This is a deliberate departure from
            SALib, which reports the upper triangle alone. The diagonal
            holds a parameter's interaction with itself, which is undefined,
            so it is set to ``NaN``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found and what it did about it. A
            report with ``n_invalid == 0`` means the check ran and found
            nothing. See :class:`jaxgsa._core.invalid.InvalidReport`.
        S1_conf: Bootstrap confidence bounds on ``S1``, shape ``(2, D)`` /
            ``(2, K, D)`` / ``(2, T, K, D)``, or ``None`` without a bootstrap.
            The leading axis holds ``[lower, upper]``.
        ST_conf: Bootstrap confidence bounds on ``ST``, same shape and layout
            as ``S1_conf``.
        S2_conf: Bootstrap confidence bounds on ``S2``, shape ``(2, D, D)`` /
            ``(2, K, D, D)`` / ``(2, T, K, D, D)``, or ``None`` without a
            bootstrap. Symmetric with a ``NaN`` diagonal, like ``S2``.
        estimator: Which estimator pair produced the indices. Six are
            available and they disagree at finite sample size, so a stored
            result is ambiguous without it. See :func:`jaxgsa.sobol.analyze`.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    S1: Array  # (D,), (K, D), or (T, K, D)
    ST: Array  # same shape as S1
    S2: Array | None  # (..., D, D), symmetric, diagonal NaN
    problem: Problem
    invalid: InvalidReport
    S1_conf: Array | None = None  # (2, *S1.shape)
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    ci: CIInfo | None = None
    estimator: str = "saltelli-jansen"

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", "param", interval=True),
            FieldSpec("ST", "param", interval=True),
            FieldSpec("S2", "pair", interval=True),
        ),
        meta=("estimator",),
    )
