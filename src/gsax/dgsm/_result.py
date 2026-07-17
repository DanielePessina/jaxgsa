"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._core.validation import _dims_and_coords
from gsax.problem import Problem


@dataclass
class DGSMResult:
    """Derivative-based global sensitivity measures and Sobol index bounds.

    ``upper_bound`` and ``lower_bound`` bracket the total Sobol index
    ``ST_i`` of each input: an input whose upper bound is near zero is
    provably negligible, while a large lower bound certifies importance.

    Index arrays mirror the output layout: shape ``(D,)`` with scalar
    ``var_y`` for scalar-output models, ``(K, D)`` with ``(K,)`` ``var_y``
    for multi-output, and ``(T, K, D)`` with ``(T, K)`` ``var_y`` for
    time-series outputs.

    Attributes:
        nu: ``E[(df/dx_i)^2]``, the mean squared partial derivative over
            the input distribution — the DGSM importance measure.
        sigma: ``E[df/dx_i]``, the mean (signed) partial derivative; its
            sign indicates the average direction of the effect.
        upper_bound: ``C_i * nu_i / Var(Y)``, the Poincare upper bound on
            ST (``C_i`` is the Poincare constant of input i's marginal).
        lower_bound: ``Var(x_i) * sigma_i^2 / Var(Y)``, the
            Kucherenko-Song lower bound on ST.
        var_y: Output variance.
        problem: Problem definition used for the analysis.
    """

    nu: Array
    sigma: Array
    upper_bound: Array
    lower_bound: Array
    var_y: Array
    problem: Problem

    def to_dataset(self, time_coords: np.ndarray | list | None = None) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Optional coordinate values for the ``time``
                dimension of time-series results; defaults to ``0..T-1``.

        Returns:
            Dataset with variables nu, sigma, upper_bound, lower_bound on
            dims ``(param,)`` / ``(output, param)`` / ``(time, output,
            param)`` matching the result shapes.
        """
        nu_arr = np.asarray(self.nu)
        dims, coords = _dims_and_coords(nu_arr.ndim, nu_arr.shape, self.problem, time_coords)
        data_vars: dict = {
            "nu": (dims, nu_arr),
            "sigma": (dims, np.asarray(self.sigma)),
            "upper_bound": (dims, np.asarray(self.upper_bound)),
            "lower_bound": (dims, np.asarray(self.lower_bound)),
        }
        return xr.Dataset(data_vars, coords=coords)
