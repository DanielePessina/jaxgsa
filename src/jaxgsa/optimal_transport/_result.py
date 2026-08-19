"""Result container for optimal-transport sensitivity indices."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass(repr=False)
class OTResult(SchemaResult):
    """Optimal-transport sensitivity analysis results.

    Holds the normalized optimal-transport index. The index is the
    class-averaged squared 2-Wasserstein distance between the conditional
    and unconditional output distributions, on a [0, 1] scale. The
    container also holds the split of that index into an advective
    (location-shift) and a diffusive (spread/shape) component, plus
    optional bootstrap confidence intervals.

    Index shapes depend on the analysis mode:

    - ``"univariate"``: one index per output column. Index arrays have
      shape ``(D,)`` for a scalar output, ``(K, D)`` for a multi-output
      model, and ``(T, K, D)`` for a time-resolved analysis.
    - ``"multivariate"``: one index per parameter over the joint output
      distribution, shape ``(D,)``.
    - ``"trajectory"``: one index per parameter per output, shape
      ``(K, D)``. Each output's time course is one point cloud.

    Confidence-interval fields add a leading axis: shape ``(2, ...)`` for
    ``[lower, upper]``.

    Attributes:
        ot: Total optimal-transport index per parameter, shape
            ``(..., D)``. 0 means the parameter leaves the output
            distribution unchanged. 1 means the parameter determines the
            output distribution fully.
        ot_conf: Bootstrap confidence interval for ``ot``,
            shape ``(2, ..., D)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        advective: Location-shift component, shape ``(..., D)``. It is the
            class-averaged squared distance between the conditional and
            unconditional output means, on the same normalized scale. It
            equals half the given-data first-order Sobol index.
        advective_conf: Bootstrap confidence interval for
            ``advective``, shape ``(2, ..., D)`` for ``[lower, upper]``.
            ``None`` when ``n_bootstrap=0``.
        diffusive: Spread/shape component ``ot - advective``, shape
            ``(..., D)``. It captures changes in the dispersion and in the
            higher moments of the output distribution.
        diffusive_conf: Bootstrap confidence interval for
            ``diffusive``, shape ``(2, ..., D)`` for ``[lower, upper]``.
            ``None`` when ``n_bootstrap=0``.
        S1: Given-data first-order Sobol index, shape ``(..., D)``. It is the
            advective component rescaled to the Sobol convention:
            ``S1 = Var(E[Y | X_i]) / Var(Y)`` with both variances taken as
            population (``ddof=0``) variances, exactly matching
            ``jaxgsa.borgonovo``'s ``S1``. The OT normalizer uses the
            unbiased (``ddof=1``) sample variance, so ``2 * advective`` alone
            would carry a factor ``(N - 1) / N`` against that convention;
            this field absorbs it, so no ddof caveat is left for the reader.
            In the point-cloud modes the variances generalize to the trace of
            the output covariance.
        S1_conf: Bootstrap confidence interval for ``S1``, shape
            ``(2, ..., D)``. It is the exactly rescaled ``advective_conf``
            (every bootstrap resample has the same size ``N``, so the ddof
            factor is one constant). ``None`` when ``n_bootstrap=0``.
        above_dummy: The total index above the irrelevance floor,
            ``max(ot - ot_dummy, 0)``, shape ``(..., D)``. The dummy baseline
            is the index a synthetic, provably irrelevant parameter receives
            from finite-sample bias (and, in the point-cloud modes, entropic
            bias), so this is the part of ``ot`` that clears that floor. A
            value of 0 means the parameter is indistinguishable from noise at
            this sample size. The name says what it is — the excess above the
            dummy floor — rather than claiming the subtraction removes bias
            in general, which it does only for irrelevant parameters.
            ``None`` unless the analysis ran with ``dummy=True``.
        ot_dummy: Irrelevance baseline, the same shape as ``ot`` without
            the trailing parameter axis. It is the index of a synthetic
            parameter that is independent of the output by construction,
            computed through the identical pipeline. Parameters whose
            ``ot`` is not clearly above this floor are indistinguishable
            from noise. ``None`` unless the analysis ran with
            ``dummy=True``.
        mode: Analysis mode that produced these shapes (``"univariate"``,
            ``"multivariate"``, or ``"trajectory"``).
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without
            a bootstrap. See :class:`jaxgsa._core.result.CIInfo`.
    """

    ot: Array
    ot_conf: Array | None
    advective: Array
    advective_conf: Array | None
    diffusive: Array
    diffusive_conf: Array | None
    S1: Array
    S1_conf: Array | None
    above_dummy: Array | None
    ot_dummy: Array | None
    mode: str
    problem: Problem
    invalid: InvalidReport
    ci: CIInfo | None = None

    _schema = ResultSchema(
        primary="ot",
        fields=(
            FieldSpec("ot", "param", interval=True),
            FieldSpec("advective", "param", interval=True),
            FieldSpec("diffusive", "param", interval=True),
            FieldSpec("S1", "param", interval=True),
            FieldSpec("above_dummy", "param"),
            FieldSpec("ot_dummy", "slice"),
        ),
        meta=("mode",),
    )

    def _base_dims(
        self, time_coords: np.ndarray | list | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Resolve dimensions from ``mode``, not from the array rank.

        Two of the three modes fix their own layout. "multivariate" reduces
        the whole output to one index per parameter, so it is always
        ``(param,)``. "trajectory" treats each output's time course as one
        point cloud, so it is always ``(output, param)``. Reading the rank
        instead would label a trajectory result's leading axis "param" for a
        one-output model, because the array happens to be 1-D.

        Args:
            time_coords: Coordinate values for the time dimension, unused
                outside "univariate" because neither other mode has one.

        Returns:
            The base dims and coordinates for this result's mode.
        """
        if self.mode == "multivariate":
            return ("param",), {"param": list(self.problem.names)}
        ndim = 2 if self.mode == "trajectory" else np.asarray(self.ot).ndim
        return _dims_and_coords(ndim, np.asarray(self.ot).shape, self.problem, time_coords)
