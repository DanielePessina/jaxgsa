"""Result container for Kucherenko dependent-input sensitivity indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass(frozen=True)
class KucherenkoResult:
    """Kucherenko sensitivity indices, returned by :func:`jaxgsa.kucherenko.analyze`.

    Index shapes mirror the shape of the analyzed output ``Y``: ``(D,)`` for a
    scalar output, ``(K, D)`` for multi-output, or ``(T, K, D)`` for
    time-resolved analyses.

    ``S1`` estimates ``V(E(Y|X_i)) / V(Y)`` and ``ST`` estimates
    ``E(V(Y|X_{~i})) / V(Y)`` under the problem's declared dependence
    structure. Under independent inputs they are the classic Sobol' first-order
    and total-order indices. Under a declared correlation ``S1`` is
    correlation-inclusive (it matches ``jaxgsa.vkoga``'s ``S_TC``) and ``ST``
    is correlation-exclusive (it matches ``S_TU``), so ``ST >= S1`` no longer
    holds in general.

    Attributes:
        S1: First-order (correlation-inclusive) indices.
        ST: Total-order (correlation-exclusive) indices.
        problem: Problem the design was generated for.
        variance: Output variance under the joint input measure, one value per
            output slice (shape ``()``, ``(K,)``, or ``(T, K)``).
    """

    S1: Array
    ST: Array
    problem: Problem
    variance: Array

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        return (
            f"KucherenkoResult(S1={tuple(self.S1.shape)}, ST={tuple(self.ST.shape)}, "
            f"correlated={self.problem.has_correlated_inputs})"
        )

    @property
    def is_correlated(self) -> bool:
        """Return ``True`` when the problem declares a non-identity correlation."""
        return self.problem.has_correlated_inputs

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                ``S1.ndim == 3``. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``S1``, ``ST``, and ``variance``,
            keyed by ``param`` (and ``output`` / ``time``) coordinates.
        """
        dims, coords = _dims_and_coords(self.S1.ndim, self.S1.shape, self.problem, time_coords)
        data_vars: dict = {
            "S1": (dims, np.asarray(self.S1)),
            "ST": (dims, np.asarray(self.ST)),
            "variance": (dims[:-1], np.asarray(self.variance)),
        }
        return xr.Dataset(
            data_vars,
            coords=coords,
            attrs={"method": "kucherenko", "correlated": bool(self.is_correlated)},
        )
