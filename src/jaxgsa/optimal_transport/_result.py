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
        ot_conf: Percentile bootstrap confidence interval for ``ot``,
            shape ``(2, ..., D)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        advective: Location-shift component, shape ``(..., D)``. It is the
            class-averaged squared distance between the conditional and
            unconditional output means, on the same normalized scale. It
            equals half the given-data first-order Sobol index.
        advective_conf: Percentile bootstrap confidence interval for
            ``advective``, shape ``(2, ..., D)`` for ``[lower, upper]``.
            ``None`` when ``n_bootstrap=0``.
        diffusive: Spread/shape component ``ot - advective``, shape
            ``(..., D)``. It captures changes in the dispersion and in the
            higher moments of the output distribution.
        diffusive_conf: Percentile bootstrap confidence interval for
            ``diffusive``, shape ``(2, ..., D)`` for ``[lower, upper]``.
            ``None`` when ``n_bootstrap=0``.
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

    def _dataset_attrs(self) -> dict[str, Any]:
        """Record the analysis mode, which fixes how the shapes read."""
        return {"mode": self.mode}
