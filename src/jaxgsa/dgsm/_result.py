"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
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
            ``nu``, ``sigma`` and ``var_y`` are all reported for the
            standardized output when the analysis ran with
            ``standardize_outputs=True``, which makes ``var_y`` 1.
        problem: Problem definition used for the analysis.
        invalid: What the non-finite check found in the sample, and which
            ``on_invalid`` policy ran. ``invalid.n_invalid == 0`` means the
            check ran and found nothing. On both calling conventions the
            check covers the derivative as well as the output; a non-finite
            derivative is reported under the source name
            ``"Y or its derivative"``.
        nu_conf: Bootstrap confidence interval for ``nu``, shape ``(2, ...)``
            for ``[lower, upper]``. ``None`` when ``n_bootstrap=0``.
        sigma_conf: The same for ``sigma``.
        upper_bound_conf: The same for ``upper_bound``. The interval covers
            the whole ratio: the resample moves ``nu`` and ``var_y`` together,
            because both are averages over the rows that were drawn.
        lower_bound_conf: The same for ``lower_bound``.
        ci: How the intervals were produced: the confidence level, the
            endpoint rule, the resample count, and the bootstrap draws when
            the analysis ran with ``keep_replicates=True``. ``None`` without a
            bootstrap. ``var_y`` carries no interval, because it is the
            denominator of the two bounds rather than a sensitivity measure,
            and its uncertainty is already inside their intervals. See
            :class:`jaxgsa._core.result.CIInfo`.
    """

    nu: Array
    sigma: Array
    upper_bound: Array
    lower_bound: Array
    var_y: Array
    problem: Problem
    invalid: InvalidReport
    nu_conf: Array | None = None
    sigma_conf: Array | None = None
    upper_bound_conf: Array | None = None
    lower_bound_conf: Array | None = None
    ci: CIInfo | None = None

    # var_y is declared but not exported: the dataset has never carried it,
    # and adding it here would change what an existing netCDF file holds.
    _schema = ResultSchema(
        primary="nu",
        fields=(
            FieldSpec("nu", interval=True),
            FieldSpec("sigma", interval=True),
            FieldSpec("upper_bound", interval=True),
            FieldSpec("lower_bound", interval=True),
            FieldSpec("var_y", "slice", dataset=False),
        ),
    )
