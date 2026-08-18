"""Result container for Borgonovo delta sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class DeltaResult:
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
        delta_conf: Percentile bootstrap confidence interval for ``delta``,
            shape ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        S1: Given-data first-order Sobol index per parameter, shape
            ``(..., D)``.
        S1_conf: Percentile bootstrap confidence interval for ``S1``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when
            ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
    """

    delta: Array
    delta_conf: Array | None
    S1: Array
    S1_conf: Array | None
    problem: Problem
    invalid: InvalidReport

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert the results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension, used
                when the index arrays are 3-D. Defaults to integer
                indices.

        Returns:
            An :class:`xarray.Dataset` with the variables ``delta`` and
            ``S1``. It also holds ``delta_lower``, ``delta_upper``,
            ``S1_lower`` and ``S1_upper`` when the analysis produced
            confidence intervals.
        """
        dims, coords = _dims_and_coords(
            self.delta.ndim, self.delta.shape, self.problem, time_coords
        )

        data_vars: dict = {
            "delta": (dims, np.asarray(self.delta)),
            "S1": (dims, np.asarray(self.S1)),
        }

        if self.delta_conf is not None:
            data_vars["delta_lower"] = (dims, np.asarray(self.delta_conf[0]))
            data_vars["delta_upper"] = (dims, np.asarray(self.delta_conf[1]))
        if self.S1_conf is not None:
            data_vars["S1_lower"] = (dims, np.asarray(self.S1_conf[0]))
            data_vars["S1_upper"] = (dims, np.asarray(self.S1_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
