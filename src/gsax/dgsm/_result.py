"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


@dataclass
class DGSMResult:
    """Derivative-based global sensitivity measures and Sobol index bounds.

    ``upper_bound`` and ``lower_bound`` bracket the total Sobol index
    ``ST_i`` of each input: an input whose upper bound is near zero is
    provably negligible, while a large lower bound certifies importance.

    For scalar-output models, index arrays have shape ``(D,)`` and
    ``var_y`` is a scalar.  For multi-output models, index arrays have
    shape ``(K, D)`` and ``var_y`` has shape ``(K,)``.

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

    def to_dataset(self) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Returns:
            Dataset with variables nu, sigma, upper_bound, lower_bound.
            For scalar output, dimensions are ``(param,)``.
            For multi-output, dimensions are ``(output, param)``.
        """
        param_names = list(self.problem.names)
        nu_arr = np.asarray(self.nu)
        ndim = nu_arr.ndim

        if ndim == 1:
            # Scalar output: fields are (D,)
            coords: dict = {"param": param_names}
            data_vars: dict = {
                "nu": ("param", nu_arr),
                "sigma": ("param", np.asarray(self.sigma)),
                "upper_bound": ("param", np.asarray(self.upper_bound)),
                "lower_bound": ("param", np.asarray(self.lower_bound)),
            }
        elif ndim == 2:
            # Multi-output: fields are (K, D)
            K = nu_arr.shape[0]
            output_coord = _default_output_names(K, self.problem)
            coords = {"output": output_coord, "param": param_names}
            data_vars = {
                "nu": (("output", "param"), nu_arr),
                "sigma": (("output", "param"), np.asarray(self.sigma)),
                "upper_bound": (("output", "param"), np.asarray(self.upper_bound)),
                "lower_bound": (("output", "param"), np.asarray(self.lower_bound)),
            }
        else:
            raise ValueError(f"Unexpected nu.ndim={ndim}")
        return xr.Dataset(data_vars, coords=coords)
