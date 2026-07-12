"""Defines the MorrisResult dataclass for elementary-effects screening."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class MorrisResult:
    """Morris elementary-effects screening measures.

    For scalar-output models, measure arrays have shape ``(D,)``; for
    multi-output models ``(K, D)``; for time-resolved analyses ``(T, K, D)``.
    Confidence interval arrays (``*_conf``) have an extra leading dimension of
    size 2 representing ``[lower, upper]`` bounds.

    Elementary effects are computed in unit-cube coordinates by default
    (``space == "unit"``), making ``mu_star`` directly comparable across
    parameters with different ranges. Use :meth:`to_physical_units` for
    derivative-scale values in the problem's native units.

    Attributes:
        mu: Mean elementary effect; sign cancellation can mask non-monotonic
            influence.
        mu_star: Mean absolute elementary effect (Campolongo et al. 2007), the
            headline importance measure and a proxy for total-order ranking.
        sigma: Standard deviation of the elementary effects (ddof=1); large
            values relative to ``mu_star`` indicate nonlinearity or
            interactions.
        problem: Problem definition used for the analysis.
        mu_conf: Optional bootstrap CI bounds on ``mu``, shape ``(2, ...)``.
        mu_star_conf: Optional bootstrap CI bounds on ``mu_star``.
        sigma_conf: Optional bootstrap CI bounds on ``sigma``.
        space: Coordinate space of the measures, ``"unit"`` or ``"physical"``.
    """

    mu: Array
    mu_star: Array
    sigma: Array
    problem: Problem
    mu_conf: Array | None = None
    mu_star_conf: Array | None = None
    sigma_conf: Array | None = None
    space: Literal["unit", "physical"] = "unit"

    def to_physical_units(self) -> MorrisResult:
        """Return a copy with measures rescaled to physical input units.

        Unit-cube elementary effects divide the output change by a step in
        ``[0, 1]`` coordinates; dividing each measure by the parameter range
        ``high - low`` converts it to a per-physical-unit (derivative-scale)
        effect, comparable to DGSM's mean derivative.

        Returns:
            A new ``MorrisResult`` with ``space == "physical"``.

        Raises:
            ValueError: If the result is already in physical units or the
                problem has no finite bounds.
        """
        if self.space == "physical":
            raise ValueError("Result is already in physical units")
        if self.problem.bounds is None:
            raise ValueError("to_physical_units requires a problem with finite uniform bounds")

        # Ranges broadcast against the trailing parameter axis of every field.
        ranges = np.asarray([high - low for low, high in self.problem.bounds])

        def _scale(arr: Array | None) -> Array | None:
            return None if arr is None else arr / ranges

        return replace(
            self,
            mu=self.mu / ranges,
            mu_star=self.mu_star / ranges,
            sigma=self.sigma / ranges,
            mu_conf=_scale(self.mu_conf),
            mu_star_conf=_scale(self.mu_star_conf),
            sigma_conf=_scale(self.sigma_conf),
            space="physical",
        )

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``mu.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``mu``, ``mu_star``, ``sigma``,
            optionally ``*_lower``/``*_upper`` CI bounds, and a ``space``
            attribute recording the coordinate space.
        """
        param_names = list(self.problem.names)
        ndim = self.mu.ndim

        if ndim == 1:
            dims = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            onames = _default_output_names(self.mu.shape[0], self.problem)
            dims = ("output", "param")
            coords = {"param": param_names, "output": onames}
        elif ndim == 3:
            T = self.mu.shape[0]
            onames = _default_output_names(self.mu.shape[1], self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims = ("time", "output", "param")
            coords = {"param": param_names, "output": onames, "time": tcoords}
        else:
            raise ValueError(f"Unexpected mu.ndim={ndim}")

        data_vars: dict = {
            "mu": (dims, np.asarray(self.mu)),
            "mu_star": (dims, np.asarray(self.mu_star)),
            "sigma": (dims, np.asarray(self.sigma)),
        }

        # Split the (2, ...) confidence arrays into *_lower and *_upper
        # variables so users can select bounds without integer indexing.
        for name, arr in [
            ("mu", self.mu_conf),
            ("mu_star", self.mu_star_conf),
            ("sigma", self.sigma_conf),
        ]:
            if arr is not None:
                data_vars[f"{name}_lower"] = (dims, np.asarray(arr[0]))
                data_vars[f"{name}_upper"] = (dims, np.asarray(arr[1]))

        return xr.Dataset(data_vars, coords=coords, attrs={"space": self.space})
