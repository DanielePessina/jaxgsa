"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from jax import Array

from gsax.problem import Problem


@dataclass
class DGSMResult:
    """Derivative-based global sensitivity measures and Sobol index bounds.

    All index arrays have shape (T, D) where T is the number of outputs
    and D is the number of input parameters. For scalar-output models,
    T = 1.

    Attributes:
        nu: E[(df/dx_i)^2], the DGSM importance measure.
        sigma: E[df/dx_i], the mean partial derivative.
        upper_bound: C_i * nu_i / Var(Y), Poincare upper bound on ST.
        lower_bound: Var_i * sigma_i^2 / Var(Y), Kucherenko-Song lower bound on ST.
        var_y: Output variance, shape (T,).
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
            For scalar output (T=1), dimensions are (param,).
            For multi-output (T>1), dimensions are (output, param).
        """
        param_names = list(self.problem.names)
        T = np.asarray(self.nu).shape[0]

        output_coord = (
            list(self.problem.output_names)
            if self.problem.output_names is not None
            else list(range(T))
        )

        if T == 1:
            coords: dict = {"param": param_names}
            data_vars: dict = {
                "nu": ("param", np.asarray(self.nu)[0]),
                "sigma": ("param", np.asarray(self.sigma)[0]),
                "upper_bound": ("param", np.asarray(self.upper_bound)[0]),
                "lower_bound": ("param", np.asarray(self.lower_bound)[0]),
            }
        else:
            coords = {"output": output_coord, "param": param_names}
            data_vars = {
                "nu": (("output", "param"), np.asarray(self.nu)),
                "sigma": (("output", "param"), np.asarray(self.sigma)),
                "upper_bound": (("output", "param"), np.asarray(self.upper_bound)),
                "lower_bound": (("output", "param"), np.asarray(self.lower_bound)),
            }
        return xr.Dataset(data_vars, coords=coords)
