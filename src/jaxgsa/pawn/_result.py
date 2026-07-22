"""Defines the PAWNResult dataclass for PAWN sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class PAWNResult:
    """PAWN sensitivity analysis results.

    Stores the PAWN index — the Kolmogorov-Smirnov distance between the
    unconditional output CDF and the CDF conditional on each input,
    aggregated (median/max/mean) across conditioning bins — plus optional
    bootstrap confidence intervals.

    For scalar-output models, ``pawn`` has shape ``(D,)``; for
    multi-output models ``(K, D)``; for time-series analyses
    ``(T, K, D)``. ``pawn_conf`` adds a leading axis of size 2.

    Attributes:
        pawn: PAWN sensitivity index per parameter, in [0, 1]. 0 means
            fixing the input leaves the output distribution unchanged
            (non-influential); larger values mean stronger influence.
        pawn_conf: Bootstrap confidence interval ``[lower, upper]``,
            or ``None`` when ``n_bootstrap=0``.
        problem: Problem definition used for the analysis.
    """

    pawn: Array
    pawn_conf: Array | None
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
