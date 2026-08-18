"""Result container for HSIC kernel-based sensitivity indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.validation import _dims_and_coords
from jaxgsa.problem import Problem


@dataclass
class HSICResult:
    """HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis results.

    Index arrays have shape ``(D,)`` for a scalar output, ``(K, D)`` for a
    multi-output model, and ``(T, K, D)`` for a time-resolved analysis.

    Attributes:
        R2_HSIC: Normalized first-order HSIC indices, shape ``(..., D)``.
            Each entry is ``HSIC(x_i, Y)`` divided by
            ``sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))`` and lies in [0, 1]. A value
            of 0 means the parameter and the output are independent, and
            larger values mean stronger dependence.
        T_HSIC: Total-order HSIC indices, shape ``(..., D)``. Each entry is
            the fraction of the joint dependence lost when parameter i is
            removed, so it also counts influence carried through interactions
            (analogous to ST).
        p_values: Permutation-test p-values, shape ``(..., D)``, for the null
            hypothesis that ``x_i`` and ``Y`` are independent. Small values
            mean the detected dependence is unlikely to be sampling noise.
        hsic_raw: Unnormalized ``HSIC(x_i, Y)`` values, shape ``(..., D)``.
            They depend on the kernel and on the scale of the data, so
            compare them only within one analysis.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing.
    """

    R2_HSIC: Array
    T_HSIC: Array
    p_values: Array
    hsic_raw: Array
    problem: Problem
    invalid: InvalidReport

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "R2_HSIC": self.R2_HSIC.shape,
            "T_HSIC": self.T_HSIC.shape,
            "p_values": self.p_values.shape,
            "hsic_raw": self.hsic_raw.shape,
        }
        return f"HSICResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with the variables ``R2_HSIC``, ``T_HSIC``,
            ``p_values``, and ``hsic_raw``.
        """
        r2_arr = np.asarray(self.R2_HSIC)
        dims, coords = _dims_and_coords(r2_arr.ndim, r2_arr.shape, self.problem, time_coords)

        data_vars: dict = {
            "R2_HSIC": (dims, r2_arr),
            "T_HSIC": (dims, np.asarray(self.T_HSIC)),
            "p_values": (dims, np.asarray(self.p_values)),
            "hsic_raw": (dims, np.asarray(self.hsic_raw)),
        }
        return xr.Dataset(data_vars, coords=coords)
