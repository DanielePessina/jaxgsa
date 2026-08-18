"""Result container for PAWN distribution-based sensitivity indices."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class PAWNResult:
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
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
    """

    pawn: Array
    pawn_conf: Array | None
    problem: Problem
    invalid: InvalidReport

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variable ``pawn`` and optionally
            ``pawn_lower`` / ``pawn_upper``.
        """
        dims, coords = _dims_and_coords(self.pawn.ndim, self.pawn.shape, self.problem, time_coords)

        data_vars: dict = {
            "pawn": (dims, np.asarray(self.pawn)),
        }

        if self.pawn_conf is not None:
            data_vars["pawn_lower"] = (dims, np.asarray(self.pawn_conf[0]))
            data_vars["pawn_upper"] = (dims, np.asarray(self.pawn_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
