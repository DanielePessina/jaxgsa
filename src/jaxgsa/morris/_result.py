"""Result container for Morris elementary-effects screening measures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class MorrisResult:
    """Morris elementary-effects screening measures.

    Measure arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    The analysis computes elementary effects in unit-cube coordinates by
    default, which the ``space == "unit"`` field records. In that space
    ``mu_star`` compares directly across parameters with different ranges. Use
    :meth:`to_physical_units` for derivative-scale values in the problem's
    native units.

    Attributes:
        mu: Mean elementary effect, shape ``(..., D)``. The sign is kept, so
            effects of opposite sign can cancel and hide a non-monotonic
            parameter.
        mu_star: Mean absolute elementary effect, shape ``(..., D)``
            (Campolongo et al. 2007). This is the headline importance measure
            and a proxy for total-order ranking.
        sigma: Standard deviation of the elementary effects (ddof=1), shape
            ``(..., D)``. A value that is large next to ``mu_star`` shows
            nonlinearity or interactions.
        problem: Problem definition used for the analysis.
        mu_conf: Bootstrap confidence bounds on ``mu``, shape ``(2, ...)`` for
            ``[lower, upper]``. ``None`` when the analysis ran no bootstrap.
        mu_star_conf: Bootstrap confidence bounds on ``mu_star``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when the analysis ran
            no bootstrap.
        sigma_conf: Bootstrap confidence bounds on ``sigma``, shape
            ``(2, ...)`` for ``[lower, upper]``. ``None`` when the analysis ran
            no bootstrap.
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

        A unit-cube elementary effect divides the output change by a step in
        ``[0, 1]`` coordinates. Dividing each measure by the parameter range
        ``high - low`` turns it into a per-physical-unit effect. That is a
        derivative scale, comparable to DGSM's mean derivative.

        Returns:
            A new :class:`MorrisResult` with ``space == "physical"``.

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
        """Convert the measures to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension, used when
                ``mu.ndim == 3``. Defaults to integer indices.

        Returns:
            An :class:`xarray.Dataset` with the variables ``mu``, ``mu_star``,
            and ``sigma``, each shaped ``(D,)``, ``(K, D)``, or ``(T, K, D)``.
            It also holds ``*_lower`` and ``*_upper`` confidence bounds when
            the analysis ran a bootstrap, and a ``space`` attribute that
            records the coordinate space.
        """
        dims, coords = _dims_and_coords(self.mu.ndim, self.mu.shape, self.problem, time_coords)

        data_vars: dict = {
            "mu": (dims, np.asarray(self.mu)),
            "mu_star": (dims, np.asarray(self.mu_star)),
            "sigma": (dims, np.asarray(self.sigma)),
        }

        # Split each (2, ...) confidence array into *_lower and *_upper
        # variables so users can select a bound without integer indexing.
        for name, arr in [
            ("mu", self.mu_conf),
            ("mu_star", self.mu_star_conf),
            ("sigma", self.sigma_conf),
        ]:
            if arr is not None:
                data_vars[f"{name}_lower"] = (dims, np.asarray(arr[0]))
                data_vars[f"{name}_upper"] = (dims, np.asarray(arr[1]))

        return xr.Dataset(data_vars, coords=coords, attrs={"space": self.space})
