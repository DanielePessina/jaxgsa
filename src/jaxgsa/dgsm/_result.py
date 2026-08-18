"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class DGSMResult(SchemaResult):
    """Derivative-based global sensitivity measures and Sobol index bounds.

    ``upper_bound`` and ``lower_bound`` bracket the total Sobol index ``ST_i``
    of each input. An input whose upper bound is near zero is provably
    negligible. A large lower bound certifies that the input matters.

    The index arrays mirror the output layout. A scalar-output model gives
    ``(D,)`` indices and a scalar ``var_y``. Multi-output gives ``(K, D)``
    indices and ``(K,)`` ``var_y``. A time series gives ``(T, K, D)`` indices
    and ``(T, K)`` ``var_y``. *D* is the number of parameters, *K* the number
    of outputs, and *T* the number of time steps.

    Attributes:
        nu: Mean squared partial derivative ``E[(df/dx_i)^2]`` over the input
            distribution, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``. This is
            the DGSM importance measure.
        sigma: Mean signed partial derivative ``E[df/dx_i]``, same shape as
            ``nu``. Its sign gives the average direction of the effect.
        upper_bound: Poincare upper bound on ST, ``C_i * nu_i / Var(Y)``, same
            shape as ``nu``. ``C_i`` is the Poincare constant of input i's
            marginal.
        lower_bound: Kucherenko-Song lower bound on ST,
            ``Var(x_i) * sigma_i^2 / Var(Y)``, same shape as ``nu``.
        var_y: Output variance per slice, shape ``()`` / ``(K,)`` / ``(T, K)``.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing. On both calling conventions the
            check covers the derivative as well as the output; a non-finite
            derivative is reported under the source name ``"Y"``.
    """

    nu: Array
    sigma: Array
    upper_bound: Array
    lower_bound: Array
    var_y: Array
    problem: Problem
    invalid: InvalidReport

    # var_y is declared but not exported: the dataset has never carried it,
    # and adding it here would change what an existing netCDF file holds.
    _schema = ResultSchema(
        primary="nu",
        fields=(
            FieldSpec("nu"),
            FieldSpec("sigma"),
            FieldSpec("upper_bound"),
            FieldSpec("lower_bound"),
            FieldSpec("var_y", "slice", dataset=False),
        ),
    )
