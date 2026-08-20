"""Defines the DGSMResult dataclass for derivative-based sensitivity analysis."""

from dataclasses import dataclass

from jax import Array

from jaxgsa._core.invalid import InvalidReport
from jaxgsa._core.result import CIInfo, FieldSpec, ResultSchema, SchemaResult
from jaxgsa.problem import Problem


@dataclass(repr=False)
class DGSMResult(SchemaResult):
    """Derivative-based global sensitivity measures and Sobol index bounds.

    ``upper_bound`` is a proven cap on the total Sobol index ``ST_i`` of each
    input, so an input whose upper bound is near zero is provably negligible.
    ``lower_bound`` is a floor on ``ST_i`` only when input i's marginal is an
    untruncated Gaussian. For a uniform or truncated marginal it is an
    estimate, exact for a response that is linear in that input and able to
    sit above the true ``ST_i`` for a curved one. See ``lower_bound`` below.

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
        lower_bound: ``Var(x_i) * sigma_i^2 / Var(Y)``, same shape as ``nu``.
            Kucherenko & Song (2016) prove this is a lower bound on ``ST_i``
            for a **Gaussian** marginal (their Theorem 6, Section 4.1,
            eq. 31), and only there.
            The proof goes through Stein's identity
            ``Cov(f, x_i) = E[tau(x_i) * df/dx_i]``, whose kernel ``tau``
            equals the constant ``Var(x_i)`` for an untruncated Gaussian and
            for no other marginal this package supports. For ``U(a, b)``,
            ``tau(x) = (x - a)(b - x)/2``, and replacing it by its mean
            ``Var(x_i)`` is an approximation rather than an inequality. So on
            a uniform or truncated-Gaussian input this is exact when the
            response is linear in that input, close when it is nearly linear,
            and can exceed the true ``ST_i`` when it is strongly curved: on
            ``f(p) = 1/p`` with ``p ~ U(0.1, 0.4)`` it reads 1.29, while the
            only input of a one-input model has ``ST = 1`` by definition. The
            paper's lower bounds for uniform inputs (LB1 and LB2) are
            different quantities, needing boundary evaluations and the higher
            moments ``E[x_i^m * df/dx_i]``; neither is computed here.
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
