"""Shared input/output validation and labeling for analysis entrypoints.

Validates the canonical ``(N, D)`` input and ``(N,)``/``(N, K)``/``(N, T, K)``
output contracts, promotes outputs to the canonical 3-D layout, squeezes
singleton axes back out of results, and resolves output/time coordinate
labels for result containers.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


def _default_output_names(K: int, problem: Problem) -> list[str]:
    """Resolve output coordinate labels, defaulting to y0, y1, ...

    Args:
        K: Number of output variables.
        problem: Problem definition (may carry output_names).

    Returns:
        List of K string labels.
    """
    if problem.output_names is not None:
        if len(problem.output_names) != K:
            raise ValueError(f"output_names length {len(problem.output_names)} != K={K}")
        return list(problem.output_names)
    return [f"y{i}" for i in range(K)]


def _validate_x(problem: Problem, X: Array) -> None:
    """Validate the shared ``(N, D)`` input-matrix contract.

    Args:
        problem: Problem definition with ``num_vars`` parameters.
        X: Input sample matrix, expected shape ``(N, D)``.

    Raises:
        ValueError: If X is not 2-D or its column count does not match the
            problem.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, D), got ndim={X.ndim}")
    if X.shape[1] != problem.num_vars:
        raise ValueError(
            f"X has {X.shape[1]} columns but problem has {problem.num_vars} parameters"
        )


def _validate_output(
    Y: Array,
    n_expected: int | None,
    problem: Problem | None = None,
) -> Array:
    """Validate a model output using the canonical jaxgsa axis layouts.

    Accepted layouts are ``(N,)``, ``(N, K)``, and ``(N, T, K)``. The sample
    axis must be first. jaxgsa deliberately does not infer or transpose axes.

    Args:
        Y: User-supplied output array, 1-D to 3-D.
        n_expected: Expected sample count, or ``None`` when it is derived by
            the calling method.
        problem: Optional problem definition used to validate output labels.

    Returns:
        The validated JAX array.

    Raises:
        ValueError: If the rank or leading sample dimension is invalid.
    """
    Y = jnp.asarray(Y)
    if Y.ndim not in (1, 2, 3):
        raise ValueError(f"Y must be 1-D (N,), 2-D (N, K), or 3-D (N, T, K), got ndim={Y.ndim}")
    if n_expected is not None and Y.shape[0] != n_expected:
        raise ValueError(
            f"Y has {Y.shape[0]} sample rows but {n_expected} were expected; "
            "pass Y as (N,), (N, K), or (N, T, K)"
        )
    if problem is not None and problem.output_names is not None:
        n_outputs = 1 if Y.ndim == 1 else Y.shape[-1]
        if len(problem.output_names) != n_outputs:
            raise ValueError(
                f"output_names length {len(problem.output_names)} does not match "
                f"the output axis K={n_outputs}"
            )
    return Y


# Named in every correlated-problem rejection so the error is actionable. The
# variance-based routes come first: they answer the same question the refused
# method was asked, so they are what the user most likely wants.
_CORRELATION_TOLERANT_METHODS = (
    "jaxgsa.vkoga (variance-based indices from given data, through a kernel "
    "surrogate), jaxgsa.kucherenko (variance-based indices from its own "
    "conditional-copula design), jaxgsa.optimal_transport, jaxgsa.borgonovo, "
    "jaxgsa.hdmr (whose ANCOVA Sb term quantifies the correlation-induced "
    "contribution, and whose result supports "
    "shapley(include_correlative=True)), jaxgsa.shapley with "
    'backend="hdmr", jaxgsa.hsic, or jaxgsa.pawn'
)


def _categorical_param_names(problem: Problem) -> list[str]:
    """Return the names of the problem's categorical parameters."""
    # Imported lazily: this module is type-checking-only on Problem.
    from jaxgsa.problem import _categorical_dims

    return [problem.names[d] for d, _ in _categorical_dims(problem)]


def _raise_categorical_and_correlated(problem: Problem, method: str, names: list[str]) -> None:
    """Reject a problem that is both categorical and correlated.

    This combination has no route. Every categorical-tolerant variance-based
    sampler (``jaxgsa.sobol``) refuses a declared correlation, and every
    correlation-tolerant variance-based method (``jaxgsa.vkoga``,
    ``jaxgsa.kucherenko``) refuses a categorical parameter. Naming only one of
    the two faults would send the user to a method that then refuses the
    other, so both design-side gates route the combined case here.

    Args:
        problem: Problem the design was requested for.
        method: Fully qualified sampler name for the error message.
        names: Names of the categorical parameters, for the message.

    Raises:
        ValueError: Always. Callers check the combined condition first.
    """
    del problem
    raise ValueError(
        f"{method} cannot build a design for this problem: parameters {names} "
        "are categorical and problem.correlation declares a dependence "
        "structure, and no variance-based method supports categorical plus "
        "correlated inputs yet. The categorical-tolerant samplers refuse a "
        "correlation, and the correlation-tolerant ones refuse a categorical "
        "parameter, so naming either route alone would send you in a circle. "
        "Draw correlated samples with jaxgsa.sampling.monte_carlo and analyze "
        "them with jaxgsa.optimal_transport or jaxgsa.borgonovo, which handle "
        "both. To drop one of the two constraints, use "
        "problem.with_correlation(None) or declare the parameter continuous."
    )


def _raise_correlated_design(problem: Problem, method: str) -> None:
    """Reject a correlated problem in a structured design sampler.

    A structured design places its points assuming independent marginals; the
    downstream estimators are undefined — not merely approximate — under a
    declared correlation, so sampling refuses up front rather than producing a
    design whose analysis would be silently wrong. ``method`` names the
    sampler, so this text does not have to list the callers.

    A problem that is also categorical gets the combined message instead: the
    correlated-only text would recommend methods that then refuse it for being
    categorical. This check therefore does not depend on being ordered before
    or after :func:`_raise_categorical_design`.

    Args:
        problem: Problem the design was requested for.
        method: Fully qualified sampler name for the error message.

    Raises:
        ValueError: If ``problem`` declares a non-identity correlation. The
            message names the combined dead end when ``problem`` also declares
            a categorical parameter.
    """
    if not problem.has_correlated_inputs:
        return
    names = _categorical_param_names(problem)
    if names:
        _raise_categorical_and_correlated(problem, method, names)
    raise ValueError(
        f"{method} builds a structured design whose sensitivity indices assume "
        "independent inputs, but problem.correlation declares a dependence "
        "structure. Draw correlated samples with jaxgsa.sampling.monte_carlo "
        "(which honors problem.correlation) and analyze them with a "
        f"correlation-tolerant given-data method: {_CORRELATION_TOLERANT_METHODS}. "
        "To analyze the independent problem instead, drop the matrix with "
        "problem.with_correlation(None)."
    )


def _raise_correlated_analysis(problem: Problem, method: str) -> None:
    """Reject a correlated problem in a correlation-naive analyzer.

    Args:
        problem: Problem the analysis was requested for.
        method: Fully qualified analyzer name for the error message.

    Raises:
        ValueError: If ``problem`` declares a non-identity correlation.
    """
    if not problem.has_correlated_inputs:
        return
    raise ValueError(
        f"{method} computes indices that assume independent inputs, and they are "
        "invalid — not merely approximate — when problem.correlation declares a "
        "dependence structure. Use a correlation-tolerant given-data method "
        f"instead: {_CORRELATION_TOLERANT_METHODS}. To analyze the independent "
        "problem instead, drop the matrix with problem.with_correlation(None)."
    )


# Named in every categorical-problem rejection so the error is actionable.
_CATEGORICAL_TOLERANT_METHODS = (
    "jaxgsa.optimal_transport, jaxgsa.borgonovo, and jaxgsa.pawn (one "
    "conditioning class per level), or the design-based jaxgsa.sobol "
    "Saltelli pipeline"
)


def _raise_categorical_design(problem: Problem, method: str) -> None:
    """Reject a categorical problem in an incompatible design sampler.

    The Morris and eFAST designs walk or sweep a continuous input space;
    an unordered categorical axis has no meaningful step or frequency, so
    sampling refuses up front. The copula conditionals of
    ``jaxgsa.kucherenko.sample`` refuse it for a different reason: they need a
    continuous marginal CDF on every coordinate.

    A problem that is also correlated gets the combined message instead, so
    this check does not depend on being ordered before or after
    :func:`_raise_correlated_design`.

    Args:
        problem: Problem the design was requested for.
        method: Fully qualified sampler name for the error message.

    Raises:
        ValueError: If ``problem`` declares any categorical parameter. The
            message names the combined dead end when ``problem`` also declares
            a non-identity correlation.
    """
    names = _categorical_param_names(problem)
    if not names:
        return
    if problem.has_correlated_inputs:
        _raise_categorical_and_correlated(problem, method, names)
    raise ValueError(
        f"{method} requires continuous (orderable) inputs, but parameters "
        f"{names} are categorical. Use jaxgsa.sobol.sample (the Saltelli "
        "column-swap scheme is distribution-agnostic; it requires a problem "
        "with no declared correlation), or analyze given data with "
        "jaxgsa.optimal_transport or jaxgsa.borgonovo."
    )


def _raise_categorical_analysis(problem: Problem, method: str) -> None:
    """Reject a categorical problem in a categorical-naive analyzer.

    Args:
        problem: Problem the analysis was requested for.
        method: Fully qualified analyzer name for the error message.

    Raises:
        ValueError: If ``problem`` declares any categorical parameter.
    """
    names = _categorical_param_names(problem)
    if not names:
        return
    raise ValueError(
        f"{method} treats every input as continuous, but parameters {names} "
        "are categorical (unordered level codes); its indices would depend on "
        "the arbitrary code order. Use a categorical-aware method instead: "
        f"{_CATEGORICAL_TOLERANT_METHODS}."
    )


def _validate_xy_inputs(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    correlation_ok: bool = False,
    categorical_ok: bool = False,
    method: str = "this method",
) -> Array:
    """Validate the shared ``(problem, X, Y)`` contract of given-data methods.

    Validates X and Y against the canonical jaxgsa layouts, using X's row count
    as the expected sample count. No axis inference or transposition is
    performed.

    Args:
        problem: Problem definition with ``num_vars`` parameters.
        X: Input sample matrix, expected shape ``(N, D)``.
        Y: Model output with shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        correlation_ok: Capability flag: whether the calling method's indices
            remain valid when ``problem.correlation`` declares a dependence
            structure. Correlation-tolerant methods (rank/CDF-based, or
            ANCOVA-decomposing) pass ``True``; the default ``False`` rejects
            a correlated problem with an actionable ``ValueError``.
        categorical_ok: Capability flag: whether the calling method handles
            categorical (unordered) parameters correctly. Methods that
            condition on one class per level pass ``True``; the default
            ``False`` rejects a categorical problem with an actionable
            ``ValueError``.
        method: Fully qualified caller name used in the rejection message.

    Returns:
        Y as a validated JAX array.

    Raises:
        ValueError: If X or Y violates the shared shape contract, the
            problem is correlated and ``correlation_ok`` is ``False``, or
            the problem has categorical parameters and ``categorical_ok``
            is ``False``.
    """
    if not correlation_ok:
        _raise_correlated_analysis(problem, method)
    if not categorical_ok:
        _raise_categorical_analysis(problem, method)
    _validate_x(problem, X)
    return _validate_output(Y, int(X.shape[0]), problem)


def _squeeze_output_axes(
    arr: Array,
    squeeze_time: bool,
    squeeze_output: bool,
    *,
    n_trailing: int = 1,
) -> Array:
    """Remove the singleton T/K axes that ``_prepare_Y`` inserted.

    The ``(T, K)`` slice axes are located immediately before ``n_trailing``
    trailing axes and addressed relative to the end, so any leading axes (a
    confidence array's ``[lower, upper]`` axis, for example) ride through the
    ``Ellipsis`` untouched. ``n_trailing`` says how many axes follow ``K``:
    ``1`` for the usual ``(..., T, K, D)`` point/confidence arrays, ``2`` for
    ``(..., T, K, D, D)`` pair matrices, and ``0`` for per-slice ``(..., T, K)``
    scalars.

    Args:
        arr: Array whose axes are ``(..., T, K) + n_trailing`` trailing axes.
        squeeze_time: Whether the T axis was inserted (drop it).
        squeeze_output: Whether the K axis was inserted (drop it).
        n_trailing: Number of axes after K.

    Returns:
        The array with the inserted singleton axes removed.
    """
    tail = (slice(None),) * n_trailing
    if squeeze_time and squeeze_output:
        return arr[(Ellipsis, 0, 0) + tail]
    if squeeze_time:
        return arr[(Ellipsis, 0, slice(None)) + tail]
    return arr


def _dims_and_coords(
    ndim: int,
    shape: tuple[int, ...],
    problem: Problem,
    time_coords: "np.ndarray | list | None" = None,
) -> tuple[tuple[str, ...], dict]:
    """Resolve xarray dimension names and coordinates for a result array.

    Shared by every result class's ``to_dataset`` so the ``param`` /
    ``output`` / ``time`` schema stays consistent across methods.

    Args:
        ndim: Number of dimensions of the index array (1, 2, or 3).
        shape: Shape of the index array; used to recover K (and T).
        problem: Problem definition supplying parameter and output names.
        time_coords: Optional coordinate values for the time dimension when
            ``ndim == 3``; defaults to integer indices.

    Returns:
        A tuple ``(dims, coords)`` suitable for constructing an
        ``xr.Dataset``.

    Raises:
        ValueError: If ``ndim`` is not 1, 2, or 3.
    """
    param_names = list(problem.names)
    if ndim == 1:
        return ("param",), {"param": param_names}
    if ndim == 2:
        K = shape[0]
        onames = _default_output_names(K, problem)
        return ("output", "param"), {"output": onames, "param": param_names}
    if ndim == 3:
        T = shape[0]
        K = shape[1]
        onames = _default_output_names(K, problem)
        tcoords = list(time_coords) if time_coords is not None else list(range(T))
        return (
            ("time", "output", "param"),
            {"time": tcoords, "output": onames, "param": param_names},
        )
    raise ValueError(f"Unexpected index array ndim={ndim}")


def _prepare_Y(
    Y: Array,
) -> tuple[Array, bool, bool]:
    """Promote Y to a canonical 3-D shape (N, T, K).

    Args:
        Y: Model output array with 1, 2, or 3 dimensions.

    Returns:
        A tuple ``(Y_3d, squeeze_time, squeeze_output)`` indicating which
        singleton dimensions were inserted and should be removed later.
    """
    squeeze_time = False
    squeeze_output = False
    # Normalize to 3-D so downstream kernels always see (samples, T, K)
    # without shape-dependent branching.
    if Y.ndim == 1:  # scalar output per sample -- both T and K are singleton
        Y = Y[:, None, None]
        squeeze_time = True
        squeeze_output = True
    elif Y.ndim == 2:  # multi-output but single timestep -- only T is singleton
        Y = Y[:, None, :]
        squeeze_time = True
    return Y, squeeze_time, squeeze_output


def _warn_zero_variance_slices(
    Y: Array,
    output_names: tuple[str, ...] | None = None,
    var_per_slice: Array | None = None,
) -> None:
    """Check for zero-variance output slices before analysis and warn.

    Args:
        Y: Model output array with shape ``(n_expanded, ...)`` where
            trailing dims are ``()``, ``(K,)``, or ``(T, K)``.
        output_names: Optional names for the K output dimension.
        var_per_slice: Optional pre-computed per-slice sample variance of
            ``Y`` over its first axis (any shape reshapeable to the flat
            slice axis). Pass it when the caller already computed the same
            variance, so it is not recomputed here.
    """
    # Collapse trailing dims so variance is computed per (t, k) slice.
    flat = Y.reshape(Y.shape[0], -1)
    n_outputs = flat.shape[1]

    # Recover K from the original shape to map flat indices back to named outputs.
    trailing = Y.shape[1:]
    if len(trailing) == 0:
        K = 1
    elif len(trailing) == 1:
        K = trailing[0]
    else:
        K = trailing[1]

    if output_names is not None and len(output_names) != K:
        # Diagnostics should not mask the entrypoint's primary shape error.
        output_names = None

    def _fmt_k(k: int) -> str:
        if output_names is not None:
            return f"k={k} ('{output_names[k]}')"
        return f"k={k}"

    # Sample variance along axis 0; zero means the output is constant
    # and Sobol indices become 0/0 = NaN.
    if var_per_slice is None:
        var_per_slice = jnp.var(flat, axis=0)
    zero_mask = var_per_slice.reshape(-1) == 0
    n_zero = int(jnp.sum(zero_mask))

    if n_zero == 0:
        return

    if n_outputs == 1:
        if output_names is not None and len(output_names) == 1:
            msg = f"jaxgsa: output '{output_names[0]}' has zero variance — all indices will be NaN"
        else:
            msg = "jaxgsa: output has zero variance — all indices will be NaN"
        warnings.warn(msg, stacklevel=2)
        return

    # Materialize indices eagerly -- this is a rare warning path, not a hot loop.
    zero_indices = [int(i) for i in jnp.where(zero_mask)[0]]

    if len(trailing) == 1:  # single-timestep: flat index equals output index k
        labels = [_fmt_k(k) for k in zero_indices]
        if len(labels) > 5:  # cap displayed labels to keep warnings readable
            shown = ", ".join(labels[:5])
            extra = f"... and {len(labels) - 5} more"
            label_str = f"{shown}, {extra}"
        else:
            label_str = ", ".join(labels)
        warnings.warn(
            f"jaxgsa: {n_zero}/{n_outputs} output(s) have zero variance "
            f"({label_str}) — corresponding indices will be NaN",
            stacklevel=2,
        )
    elif len(trailing) == 2:  # multi-timestep: flat index encodes (t, k) in row-major order
        affected = []
        for idx in zero_indices:
            # divmod decomposes flat index into (time_step, output_column)
            t, k = divmod(idx, K)
            affected.append(f"(t={t}, {_fmt_k(k)})")
        if len(affected) > 5:  # cap displayed labels to keep warnings readable
            shown = ", ".join(affected[:5])
            extra = f"... and {len(affected) - 5} more"
            label_str = f"{shown}, {extra}"
        else:
            label_str = ", ".join(affected)
        warnings.warn(
            f"jaxgsa: {n_zero}/{n_outputs} output slice(s) have zero variance "
            f"[{label_str}] — corresponding indices will be NaN",
            stacklevel=2,
        )


def _prenormalize_outputs(Y: Array) -> tuple[Array, Array, Array, Array]:
    """Standardize outputs over the sample axis.

    Args:
        Y: Output array with shape ``(N, ...)``. The first axis is treated as
            the sample axis, and all trailing axes are normalized
            independently.

    Returns:
        A tuple ``(Y_norm, y_mean, y_std, safe_scale)`` where:
            - ``Y_norm`` is the centered/scaled output array.
            - ``y_mean`` is the per-output-slice mean.
            - ``y_std`` is the per-output-slice original standard deviation.
            - ``safe_scale`` is the divisor actually used, with zeros replaced
              by ``1.0`` to avoid division by zero.
    """
    # Centering + scaling stabilizes Sobol variance estimators when output
    # magnitudes vary across slices (prevents large-magnitude slices from
    # dominating numerical precision).
    y_mean = jnp.mean(Y, axis=0)
    y_std = jnp.std(Y, axis=0)
    # Replace zero std with 1.0 so division doesn't produce NaN; the
    # corresponding zero-variance output slices remain all-zero after scaling.
    safe_scale = jnp.where(y_std == 0, jnp.ones_like(y_std), y_std)
    Y_norm = (Y - y_mean) / safe_scale
    return Y_norm, y_mean, y_std, safe_scale
