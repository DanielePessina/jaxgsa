"""DGSM analysis: compute derivative-based sensitivity measures and bounds.

Computes the DGSM moments (nu, sigma) from a JAX-differentiable function
via reverse-mode autodiff, then derives Poincare upper bounds and
Kucherenko-Song lower bounds on the total Sobol index ST.

References:
    Sobol' & Kucherenko (2009). Math. Comp. Sim. 79:3009-3017.
    Kucherenko & Song (2016). Rel. Eng. Sys. Safety 148:81-95.
    Lamboni et al. (2013). Math. Comp. Sim. 87:44-54.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxgsa._core.validation import (
    _prepare_Y,
    _raise_categorical_analysis,
    _raise_correlated_analysis,
    _squeeze_output_axes,
    _validate_output,
    _validate_x,
)
from jaxgsa.dgsm._poincare import axis_constants
from jaxgsa.dgsm._result import DGSMResult
from jaxgsa.problem import Problem


def _promote_jac(jac: Array) -> Array:
    """Promote a Jacobian to canonical 4-D ``(N, T, K, D)`` by axis position.

    This step is positional only. ``analyze`` applies the label-aware
    reinterpretation of the slice axes afterwards, by aligning against the
    canonicalized ``Y``. That pass handles a single-output time series and
    swapped trailing axes.

    Args:
        jac: Jacobian, shape ``(N, D)`` / ``(N, K, D)`` / ``(N, T, K, D)``.

    Returns:
        The Jacobian as a 4-D array of shape ``(N, T, K, D)``.

    Raises:
        ValueError: If ``jac`` is not 2-D, 3-D, or 4-D.
    """
    if jac.ndim == 2:  # (N, D) scalar output
        return jac[:, None, None, :]
    if jac.ndim == 3:  # (N, K, D) multi-output
        return jac[:, None, :, :]
    if jac.ndim == 4:  # (N, T, K, D) time series
        return jac
    raise ValueError(
        f"Jacobian must be 2-D (N, D), 3-D (N, K, D), or 4-D (N, T, K, D), got ndim={jac.ndim}"
    )


def _promote_moments(m: Array) -> Array:
    """Promote a reduced moment ``(*slice, D)`` to canonical ``(T, K, D)``.

    This is the reduced-space analog of :func:`_promote_jac`, which acts on the
    sample axis as well. The autodiff path needs it because it averages the
    moments over N before the output layout is known.

    Args:
        m: Reduced moment, shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.

    Returns:
        The moment as a 3-D array of shape ``(T, K, D)``.
    """
    if m.ndim == 1:  # (D,) scalar output
        return m[None, None, :]
    if m.ndim == 2:  # (K, D) multi-output
        return m[None, :, :]
    return m  # (T, K, D) time series


def _compute_moments(
    fn: Callable,
    X: Array,
    batch_size: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute DGSM moments and forward outputs via reverse-mode autodiff.

    The mean over the sample axis is one vectorized reduction that covers every
    (t, k) output slice at once. The per-slice kernel needs no explicit vmap,
    because the closed form already batches it.

    Args:
        fn: JAX-differentiable function ``(D,) -> ()``, ``(D,) -> (K,)``, or
            ``(D,) -> (T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Number of N sample rows per batch, to limit memory. None
            processes all N rows at once.

    Returns:
        A tuple ``(Y, sigma, nu)``. ``Y`` is the stacked raw ``fn`` output,
        shape ``(N,)`` / ``(N, K)`` / ``(N, T, K)``. ``sigma`` and ``nu`` are
        the reduced raw moments and carry the ``fn`` output's own slice axes,
        shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``. ``analyze`` applies the
        layout canonicalization (promotion and label-driven axis moves) once
        the semantic axes are known.
    """

    # Reverse-mode Jacobian is efficient when K (outputs) < D (inputs).
    # has_aux=True returns both the Jacobian and the forward output from one
    # forward+backward pass, avoiding a redundant model evaluation.
    def _fn_aux(x: Array) -> tuple[Array, Array]:
        y = fn(x)
        return y, y

    combined = jax.jit(jax.vmap(jax.jacrev(_fn_aux, has_aux=True)))

    N = X.shape[0]

    if not batch_size or batch_size <= 0 or N <= batch_size:
        jac, Y = combined(X)
        return Y, jnp.mean(jac, axis=0), jnp.mean(jac**2, axis=0)

    # Batched path: accumulate running sums to bound peak memory.
    # The first full batch triggers XLA compilation; subsequent batches reuse it.
    sum_jac: Array | None = None
    sum_jac2: Array | None = None
    Y_parts: list[Array] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        actual_len = end - start

        # Pad the ragged last batch, so a new shape does not force a JIT
        # recompilation.
        X_chunk = X[start:end]
        if actual_len < batch_size:
            pad_size = batch_size - actual_len
            X_chunk = jnp.concatenate([X_chunk, jnp.zeros((pad_size, X.shape[1]))], axis=0)

        jac, Y_chunk_full = combined(X_chunk)
        Y_parts.append(Y_chunk_full[:actual_len])

        jac = jac[:actual_len]
        sj = jnp.sum(jac, axis=0)
        sj2 = jnp.sum(jac**2, axis=0)
        sum_jac = sj if sum_jac is None else sum_jac + sj
        sum_jac2 = sj2 if sum_jac2 is None else sum_jac2 + sj2

    Y = jnp.concatenate(Y_parts, axis=0)
    assert sum_jac is not None and sum_jac2 is not None
    return Y, sum_jac / N, sum_jac2 / N


def analyze(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    batch_size: int | None = None,
) -> DGSMResult:
    """Compute DGSM sensitivity indices and Sobol index bounds.

    DGSM ranks inputs by how strongly the output reacts to them on average.
    The measure is ``nu_i = E[(df/dx_i)^2]``, the mean squared partial
    derivative over the input distribution. It is the natural pick when the
    model is JAX-differentiable: one autodiff sweep over an ordinary Monte
    Carlo sample replaces a dedicated Sobol design.

    Two inequalities convert the moments into a bracket on the total Sobol
    index:

    - **Upper bound** (Poincare / Sobol-Kucherenko inequality):
      ``ST_i <= C_i * nu_i / Var(Y)``. ``C_i`` is the Poincare constant of
      input i's marginal distribution, the sharpest factor for which the
      inequality holds (see :mod:`jaxgsa.dgsm._poincare`).
    - **Lower bound** (Kucherenko-Song):
      ``ST_i >= Var(x_i) * sigma_i^2 / Var(Y)``, with
      ``sigma_i = E[df/dx_i]``, the mean (signed) derivative.

    An input whose upper bound is near zero is provably negligible.

    There are two calling conventions:

    - **Autodiff path** (primary): pass ``fn`` and ``X``. ``jax.jacrev``
      differentiates the function, and one pass returns both the Jacobian and
      the forward outputs.
    - **Pre-computed path**: pass ``Y`` and ``dfdx``. Use it when the model is
      not JAX-differentiable, or when the Jacobian comes from elsewhere.

    Args:
        problem: Problem definition with D parameters.
        fn: JAX-differentiable function ``(D,) -> ()``, ``(D,) -> (K,)``, or
            ``(D,) -> (T, K)`` for time-series outputs.
        X: Sample matrix in the problem's physical units, shape ``(N, D)``.
        Y: Forward model outputs, shape ``(N,)`` / ``(N, K)`` / ``(N, T, K)``.
        dfdx: Pre-computed Jacobian, mirroring ``Y``'s layout with one extra
            trailing ``(D,)`` axis: ``(N, D)`` for ``(N,)`` Y, ``(N, K, D)``
            for ``(N, K)``, and ``(N, T, K, D)`` for ``(N, T, K)``.
        batch_size: Number of N sample rows per batch on the autodiff path.
            The Jacobian accumulates in batches of this many samples, which
            bounds peak memory. None (default) processes all N samples at once.

    Returns:
        A ``DGSMResult`` with ``nu``, ``sigma``, ``upper_bound``, and
        ``lower_bound``, each of shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``
        mirroring the output layout, plus ``var_y``.

    Raises:
        ValueError: In any of these cases. Neither ``(fn, X)`` nor
            ``(Y, dfdx)`` was given. ``dfdx`` has an unexpected shape, or does
            not match ``Y`` or the problem dimension. ``problem.correlation``
            declares a dependence structure, and the Poincare-inequality bounds
            assume independent inputs. ``problem`` has categorical parameters,
            and a derivative along an unordered level code has no meaning.
    """
    # The Poincare-inequality bound on ST assumes independent inputs; both
    # calling conventions are rejected for a correlated problem.
    _raise_correlated_analysis(problem, "jaxgsa.dgsm.analyze")
    _raise_categorical_analysis(problem, "jaxgsa.dgsm.analyze")
    D = problem.num_vars

    if fn is not None and X is not None:
        X = jnp.asarray(X)
        _validate_x(problem, X)
        Y_out, sigma, nu = _compute_moments(fn, X, batch_size=batch_size)
        Y_valid = _validate_output(Y_out, int(X.shape[0]), problem)
        sigma = _promote_moments(sigma)
        nu = _promote_moments(nu)
    elif Y is not None and dfdx is not None:
        # Pre-computed path: the caller supplies the Jacobian and the forward
        # outputs directly.
        Y_out = jnp.asarray(Y)
        dfdx_arr = jnp.asarray(dfdx)
        Y_valid = _validate_output(Y_out, int(dfdx_arr.shape[0]), problem)
        if dfdx_arr.ndim != Y_valid.ndim + 1:
            raise ValueError("dfdx ndim must equal Y.ndim + 1 and end with the derivative axis")
        if dfdx_arr.shape[-1] != D:
            raise ValueError(
                f"dfdx last dimension ({dfdx_arr.shape[-1]}) must match problem.num_vars ({D})"
            )
        if dfdx_arr.shape[:-1] != Y_valid.shape:
            raise ValueError(
                f"dfdx shape {dfdx_arr.shape} does not match Y shape {Y_valid.shape} "
                f"with trailing D={D}"
            )
        dfdx_arr = _promote_jac(dfdx_arr)
        # One vectorized reduction over N covers every (t, k) slice at once.
        sigma = jnp.mean(dfdx_arr, axis=0)  # E[df/dx_i], (T, K, D)
        nu = jnp.mean(dfdx_arr**2, axis=0)  # E[(df/dx_i)^2]
    else:
        raise ValueError("Provide either (fn, X) or (Y, dfdx)")

    # Canonicalize Y to (N, T, K). The moments were realigned in lockstep with
    # Y's canonicalization above, so their slice axes must already match. A
    # mismatch means dfdx did not mirror Y's layout: wrong ndim, or a
    # transposed Jacobian that inference could not recover. Reject it.
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y_valid)
    if sigma.shape[:2] != Y_3d.shape[1:3]:
        raise ValueError(
            f"dfdx ndim/shape is incompatible with Y: derivative slice dims "
            f"{sigma.shape[:2]} do not match Y's output dims {Y_3d.shape[1:3]}; "
            "dfdx must mirror Y's layout with one extra trailing (D,) axis"
        )

    # Var(Y) per (t, k) output slice, denominator of both bounds.
    var_y = jnp.var(Y_3d, axis=0)  # (T, K)

    # Per-axis constants: C for the Poincare upper bound, Var for the lower.
    C, Var = axis_constants(problem)
    C_jnp = jnp.asarray(C)
    Var_jnp = jnp.asarray(Var)

    # A constant output slice has zero variance; map it to NaN bounds rather
    # than dividing by zero.
    denom = jnp.where(var_y == 0, jnp.nan, var_y)[..., None]  # (T, K, 1)

    # Upper: ST_i <= C_i * nu_i / Var(Y)  (Sobol-Kucherenko inequality).
    # The (D,) constants broadcast against (T, K, D) on the trailing axis.
    upper = C_jnp * nu / denom
    # Lower: ST_i >= Var_i * sigma_i^2 / Var(Y)  (Kucherenko-Song)
    lower = Var_jnp * sigma**2 / denom

    # The upper bound must sit at or above the lower bound. Check it within a
    # numerical tolerance and warn if it does not hold.
    if jnp.any(jnp.isfinite(upper) & jnp.isfinite(lower) & (upper < lower * 0.9)):
        warnings.warn(
            "DGSM: some upper bounds are below lower bounds, suggesting "
            "insufficient samples or numerical issues",
            stacklevel=2,
        )

    # Drop the singleton axes _prepare_Y inserted; var_y has no trailing param
    # axis, so it passes n_trailing=0.
    sigma = _squeeze_output_axes(sigma, squeeze_time, squeeze_output)
    nu = _squeeze_output_axes(nu, squeeze_time, squeeze_output)
    upper = _squeeze_output_axes(upper, squeeze_time, squeeze_output)
    lower = _squeeze_output_axes(lower, squeeze_time, squeeze_output)
    var_y = _squeeze_output_axes(var_y, squeeze_time, squeeze_output, n_trailing=0)

    return DGSMResult(
        nu=nu,
        sigma=sigma,
        upper_bound=upper,
        lower_bound=lower,
        var_y=var_y,
        problem=problem,
    )
