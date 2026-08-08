"""Result container for optimal-transport sensitivity indices."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class OTResult:
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

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with the variables ``ot``, ``advective`` and
            ``diffusive``, and with a ``mode`` attribute. Bootstrap
            intervals add ``*_lower`` and ``*_upper`` variables. A
            computed dummy baseline adds ``ot_dummy``.
        """
        dims, coords = _dims_and_coords(self.ot.ndim, self.ot.shape, self.problem, time_coords)

        data_vars: dict = {
            "ot": (dims, np.asarray(self.ot)),
            "advective": (dims, np.asarray(self.advective)),
            "diffusive": (dims, np.asarray(self.diffusive)),
        }

        for name, conf in (
            ("ot", self.ot_conf),
            ("advective", self.advective_conf),
            ("diffusive", self.diffusive_conf),
        ):
            if conf is not None:
                data_vars[f"{name}_lower"] = (dims, np.asarray(conf[0]))
                data_vars[f"{name}_upper"] = (dims, np.asarray(conf[1]))

        if self.ot_dummy is not None:
            # The dummy baseline has no parameter axis: it is the index of
            # one synthetic parameter, per output slice.
            data_vars["ot_dummy"] = (dims[:-1], np.asarray(self.ot_dummy))

        return xr.Dataset(data_vars, coords=coords, attrs={"mode": self.mode})
