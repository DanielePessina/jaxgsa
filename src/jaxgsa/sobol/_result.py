"""Result container for Sobol sensitivity analysis."""

import dataclasses
from dataclasses import dataclass
from typing import Literal, Self

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import (
    CIInfo,
    FieldSpec,
    ResultSchema,
    SchemaResult,
    aggregate_indices,
)
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
            floating-point drift. Averaging usually lowers the error but
            does not always (measured on Ishigami: 25% and 13% lower at
            ``base_n`` 64 and 1024, 9% higher at 256; see
            :func:`jaxgsa.sobol._analyze._symmetrize_s2`). This is a
            deliberate departure from SALib, which reports the upper
            triangle alone. The diagonal
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
        variance: Variance of each output slice, shape ``()`` / ``(K,)`` /
            ``(T, K)``, or ``None`` for a result saved before this field
            existed. It is the variance over every row of the expanded
            Saltelli layout, which is the number the estimator divides by
            when it standardizes the outputs. :meth:`aggregate` uses it as
            the weight of each output.
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
    variance: Array | None = None

    _schema = ResultSchema(
        primary="S1",
        fields=(
            FieldSpec("S1", "param", interval=True),
            FieldSpec("ST", "param", interval=True),
            FieldSpec("S2", "pair", interval=True),
            FieldSpec("variance", "slice"),
        ),
        meta=("estimator",),
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

        The ``variance`` of the returned result holds the summed variance of
        the outputs.

        A slice of zero variance (a constant output) has weight 0 and adds
        nothing. A result with one output slice comes back with the same
        index values.

        Confidence intervals do not aggregate bound by bound. When the
        analysis ran with ``keep_replicates=True``, each bootstrap draw is
        aggregated and the interval is drawn again from those draws, at the
        same level and with the same rule. The weights stay fixed at the
        point-estimate values in every draw, so the interval leaves out the
        sampling noise of the weights themselves. Otherwise the new ``*_conf``
        fields are ``NaN``.

        Args:
            weights: ``"variance"`` (default) or ``"equal"``.

        Returns:
            A result of the same class with output axes of size 1.

        Raises:
            ValueError: If ``weights`` is not one of the two names, or if
                ``weights="variance"`` and the result carries no ``variance``
                (an old result). Use ``weights="equal"`` then..
        """
        updates = aggregate_indices(self, weights=weights, variance=self.variance)
        if self.variance is not None and updates:
            updates["variance"] = self.variance.sum(keepdims=True)
        return dataclasses.replace(self, **updates)
