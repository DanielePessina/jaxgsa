"""Defines the OTResult dataclass for optimal-transport sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _dims_and_coords
from gsax.problem import Problem


@dataclass
class OTResult:
    """Optimal-transport sensitivity analysis results.

    Stores the normalized optimal-transport index -- the class-averaged
    squared 2-Wasserstein distance between the conditional and
    unconditional output distributions, on a [0, 1] scale -- together
    with its decomposition into an advective (location-shift) and a
    diffusive (spread/shape) component, and optional bootstrap
    confidence intervals.

    Index shapes depend on the analysis mode:

    - ``"univariate"``: one index per output column -- ``(D,)`` for scalar
      outputs, ``(K, D)`` for multi-output, ``(T, K, D)`` for time
      series.
    - ``"multivariate"``: one index per input over the joint output
      distribution -- ``(D,)``.
    - ``"trajectory"``: one index per input per output, treating
      each output's time course as a point cloud -- ``(K, D)``.

    Confidence arrays add a leading axis of size 2 (``[lower, upper]``).

    Attributes:
        ot: Total optimal-transport index per parameter: 0 means the
            output distribution is unaffected by the input, 1 means
            fully determined by it.
        ot_conf: Percentile bootstrap confidence interval
            ``[lower, upper]``, or ``None`` when ``n_bootstrap=0``.
        advective: Location-shift component -- the class-averaged squared
            distance between conditional and unconditional output means,
            on the same normalized scale. Equals half the given-data
            first-order Sobol index.
        advective_conf: Percentile bootstrap confidence interval, or
            ``None`` when ``n_bootstrap=0``.
        diffusive: Spread/shape component ``ot - advective``, capturing
            changes in dispersion and higher moments of the output
            distribution.
        diffusive_conf: Percentile bootstrap confidence interval, or
            ``None`` when ``n_bootstrap=0``.
        ot_dummy: Irrelevance baseline -- the index of a synthetic input
            that is independent of the output by construction, computed
            through the identical pipeline. Inputs whose ``ot`` is not
            clearly above this floor are indistinguishable from noise.
            Same shape as ``ot`` without the trailing parameter axis;
            ``None`` unless the analysis ran with ``dummy=True``.
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
            An ``xr.Dataset`` with variables ``ot``, ``advective`` and
            ``diffusive`` (plus ``*_lower`` / ``*_upper`` when bootstrap
            intervals are present, and ``ot_dummy`` when the dummy
            baseline was computed) and a ``mode`` attribute.
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
            # one synthetic input, per output slice.
            data_vars["ot_dummy"] = (dims[:-1], np.asarray(self.ot_dummy))

        return xr.Dataset(data_vars, coords=coords, attrs={"mode": self.mode})
