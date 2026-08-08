"""Result container for Sobol sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class SobolResult:
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
            ``(T, K, D, D)``, or ``None`` when they were not computed. Only the
            upper triangle is estimated directly. The lower triangle mirrors it
            for convenience. The diagonal holds a parameter's interaction with
            itself, which is undefined, so it is set to ``NaN``.
        problem: Problem definition used for the analysis.
        S1_conf: Bootstrap confidence bounds on ``S1``, shape ``(2, D)`` /
            ``(2, K, D)`` / ``(2, T, K, D)``, or ``None`` without a bootstrap.
            The leading axis holds ``[lower, upper]``.
        ST_conf: Bootstrap confidence bounds on ``ST``, same shape and layout
            as ``S1_conf``.
        S2_conf: Bootstrap confidence bounds on ``S2``, shape ``(2, D, D)`` /
            ``(2, K, D, D)`` / ``(2, T, K, D, D)``, or ``None`` without a
            bootstrap. Symmetric with a ``NaN`` diagonal, like ``S2``.
        nan_counts: Number of ``NaN`` entries per index array, keyed by index
            name, or ``None`` when not recorded. Zero-variance output slices
            are the usual cause. For ``S2`` only the directly estimated upper
            triangle is counted, so the always-``NaN`` diagonal does not
            inflate the count.
    """

    S1: Array  # (D,), (K, D), or (T, K, D)
    ST: Array  # same shape as S1
    S2: Array | None  # (..., D, D), symmetric, diagonal NaN
    problem: Problem
    S1_conf: Array | None = None  # (2, *S1.shape)
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    nan_counts: dict[str, int] | None = None

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "S1": self.S1.shape,
            "ST": self.ST.shape,
            "S2": self.S2.shape if self.S2 is not None else None,
        }
        if self.S1_conf is not None:
            shapes["S1_conf"] = self.S1_conf.shape
        if self.ST_conf is not None:
            shapes["ST_conf"] = self.ST_conf.shape
        if self.S2_conf is not None:
            shapes["S2_conf"] = self.S2_conf.shape
        return f"SobolResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``S1.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``S1``, ``ST``, and optionally
            ``S2``, ``S1_lower/upper``, ``ST_lower/upper``, ``S2_lower/upper``.
        """
        dims_s1, coords = _dims_and_coords(self.S1.ndim, self.S1.shape, self.problem, time_coords)
        param_names = list(self.problem.names)

        data_vars: dict = {
            "S1": (dims_s1, np.asarray(self.S1)),
            "ST": (dims_s1, np.asarray(self.ST)),
        }

        # S2 has two parameter axes (interaction between param_i and param_j),
        # so it uses separate coordinate names to avoid an xarray dimension clash.
        if self.S2 is not None:
            dims_s2 = (*dims_s1[:-1], "param_i", "param_j")
            data_vars["S2"] = (dims_s2, np.asarray(self.S2))
            coords["param_i"] = param_names
            coords["param_j"] = param_names

        # Split the (2, ...) confidence arrays into *_lower and *_upper
        # variables so users can select bounds without integer indexing.
        for name, arr in [
            ("S1", self.S1_conf),
            ("ST", self.ST_conf),
        ]:
            if arr is not None:
                data_vars[f"{name}_lower"] = (dims_s1, np.asarray(arr[0]))
                data_vars[f"{name}_upper"] = (dims_s1, np.asarray(arr[1]))

        if self.S2_conf is not None and self.S2 is not None:
            dims_s2 = (*dims_s1[:-1], "param_i", "param_j")
            data_vars["S2_lower"] = (dims_s2, np.asarray(self.S2_conf[0]))
            data_vars["S2_upper"] = (dims_s2, np.asarray(self.S2_conf[1]))

        return xr.Dataset(data_vars, coords=coords)
