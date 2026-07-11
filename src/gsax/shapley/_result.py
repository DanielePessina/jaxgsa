"""Shapley-effect result dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class ShapleyResult:
    """Shapley-effect sensitivity analysis results.

    Stores Shapley effects computed analytically from a fitted surrogate's
    variance decomposition, alongside the first-order (S1) and total-order
    (ST) indices derived from the same decomposition so the ordering
    ``S1 <= Sh <= ST`` is visible at a glance.

    All indices are normalized by the empirical output variance, so
    ``Sh.sum(axis=-1)`` equals the surrogate's explained-variance fraction
    (close to 1 for a good fit, smaller otherwise).

    Shapes follow the convention ``(T, K, D)`` for time-resolved analyses,
    ``(K, D)`` for multi-output, or ``(D,)`` for scalar output, matching
    the layout of the ``Y`` passed to ``analyze_shapley``.

    Attributes:
        Sh: Shapley effects, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        S1: First-order indices from the same surrogate, same shape.
        ST: Total-order indices from the same surrogate, same shape.
        problem: Problem definition.
        backend: Surrogate backend used, ``"hdmr"`` or ``"pce"``.
    """

    Sh: Array
    S1: Array
    ST: Array
    problem: Problem
    backend: str

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "Sh": self.Sh.shape,
            "S1": self.S1.shape,
            "ST": self.ST.shape,
        }
        return f"ShapleyResult({shapes}, backend={self.backend!r})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``Sh.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``Sh``, ``S1``, and ``ST`` on
            ``param`` (plus ``output``/``time``) dimensions.
        """
        param_names = list(self.problem.names)
        ndim = self.Sh.ndim

        if ndim == 1:
            dims = ("param",)
            coords: dict = {"param": param_names}
        elif ndim == 2:
            onames = _default_output_names(self.Sh.shape[0], self.problem)
            dims = ("output", "param")
            coords = {"param": param_names, "output": onames}
        elif ndim == 3:
            T = self.Sh.shape[0]
            onames = _default_output_names(self.Sh.shape[1], self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims = ("time", "output", "param")
            coords = {"param": param_names, "output": onames, "time": tcoords}
        else:
            msg = f"Unexpected Sh.ndim={ndim}"
            raise ValueError(msg)

        data_vars: dict = {
            "Sh": (dims, np.asarray(self.Sh)),
            "S1": (dims, np.asarray(self.S1)),
            "ST": (dims, np.asarray(self.ST)),
        }

        return xr.Dataset(data_vars, coords=coords)
