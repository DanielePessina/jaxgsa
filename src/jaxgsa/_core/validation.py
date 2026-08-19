"""Shared input/output validation and labeling for analysis entrypoints.

The helpers here do four jobs. They check the canonical ``(N, D)`` input and
``(N,)`` / ``(N, K)`` / ``(N, T, K)`` output contracts. They promote outputs to
the canonical 3-D layout, and squeeze the inserted singleton axes back out of
the results. They resolve output and time coordinate labels for the result
containers. Finally, they gate the methods that cannot handle a correlated or
categorical problem.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.warning_types import JaxgsaWarning

if TYPE_CHECKING:
    from jaxgsa.problem import Problem


class YLayout(Enum):
    """Which rank of ``Y`` the caller passed, before the promotion to 3-D.

    Every analysis works on ``(N, T, K)``, and ``_prepare_Y`` inserts whatever
    axes are missing. The result arrays then have to come back out at the
    caller's own rank. That bookkeeping used to be two same-typed booleans,
    ``squeeze_time`` and ``squeeze_output``, which admit a fourth combination
    the promotion cannot produce (an inserted K axis with a real T axis) and
    which are easy to hand over in the wrong order. One value with three
    members can only say something true.

    Attributes:
        SCALAR: The caller passed ``(N,)``; both T and K were inserted.
        MULTI_OUTPUT: The caller passed ``(N, K)``; only T was inserted.
        TIME_SERIES: The caller passed ``(N, T, K)``; nothing was inserted.
    """

    SCALAR = 1
    MULTI_OUTPUT = 2
    TIME_SERIES = 3

    def squeeze(self, arr: Array, *, n_trailing: int = 1) -> Array:
        """Undo the promotion, for one result field.

        The ``(T, K)`` slice axes sit immediately before ``n_trailing``
        trailing axes, and are addressed relative to the end. The ``Ellipsis``
        therefore leaves any leading axes alone, such as a confidence array's
        ``[lower, upper]`` axis. ``n_trailing`` says how many axes follow
        ``K``: ``1`` for the usual ``(..., T, K, D)`` point and confidence
        arrays, ``2`` for ``(..., T, K, D, D)`` pair matrices, and ``0`` for
        per-slice ``(..., T, K)`` scalars.

        Args:
            arr: Array whose axes are ``(..., T, K)`` plus ``n_trailing``
                trailing axes.
            n_trailing: Number of axes after K.

        Returns:
            The array with the inserted singleton axes removed.
        """
        tail = (slice(None),) * n_trailing
        if self is YLayout.SCALAR:
            return arr[(Ellipsis, 0, 0) + tail]
        if self is YLayout.MULTI_OUTPUT:
            return arr[(Ellipsis, 0, slice(None)) + tail]
        return arr


def _default_output_names(K: int, problem: Problem) -> list[str]:
    """Resolve the output coordinate labels, defaulting to ``y0``, ``y1``, ....

    Args:
        K: Number of output variables.
        problem: Problem definition, which may carry ``output_names``.

    Returns:
        A list of ``K`` string labels.
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
        ValueError: If ``X`` is not 2-D, or its column count does not match
            the problem.
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


# The parenthetical each recommended method carries. The *set* of methods is
# read from the registry; only this extra prose is written by hand, and a
# method with no entry here is simply named without one. Keying the prose
# separately is what stops the list drifting: a new correlation-tolerant
# method appears in the message the moment it registers, note or no note.
_CORRELATION_NOTES: dict[str, str] = {
    "vkoga": " (variance-based indices from given data, through a kernel surrogate)",
    "kucherenko": " (variance-based indices from its own conditional-copula design)",
    "hdmr": (
        " (whose ANCOVA Sb term quantifies the correlation-induced contribution, "
        "and whose result supports shapley(include_correlative=True))"
    ),
}

# The variance-based routes come first: they answer the same question the
# refused method was asked, so they are what the user most likely wants. Any
# method not named here sorts after these, alphabetically.
_CORRELATION_ORDER = ("vkoga", "kucherenko", "optimal_transport", "borgonovo", "hdmr")

# The one route the registry cannot express. ``jaxgsa.shapley`` declares
# correlation="refuses" because its default backend does refuse; the hdmr
# backend does not. A capability that depends on an argument has no place in a
# per-method record, so it is named here instead.
_EXTRA_CORRELATION_ROUTES = ('jaxgsa.shapley with backend="hdmr"',)

_CATEGORICAL_NOTES: dict[str, str] = {
    "pawn": " (one conditioning class per level)",
    "sobol": " Saltelli pipeline",
}

_CATEGORICAL_ORDER = ("optimal_transport", "borgonovo", "pawn")


def _tolerant_names(kind: str, order: tuple[str, ...]) -> list[tuple[str, bool]]:
    """Return the registered methods that tolerate ``kind``, in message order.

    Args:
        kind: ``"correlation"`` or ``"categorical"``.
        order: Namespace names to list first, in that order. Anything the
            registry holds beyond this list follows, alphabetically, so a new
            method still reaches the message without an edit here.

    Returns:
        A list of ``(name, is_design_based)`` pairs.
    """
    # Imported here, not at module level: the registry fills up as the method
    # packages import, and this module is imported before they do.
    from jaxgsa._core.registry import methods

    def rank(name: str) -> tuple[int, str]:
        return (order.index(name), "") if name in order else (len(order), name)

    accepting = sorted(
        (spec for spec in methods().values() if getattr(spec, kind) == "accepts"),
        key=lambda spec: rank(spec.name),
    )
    return [(spec.name, spec.is_design_based) for spec in accepting]


def _correlation_tolerant_methods() -> str:
    """Name every correlation-tolerant route, for a rejection message."""
    named = _tolerant_names("correlation", _CORRELATION_ORDER)
    items = [f"jaxgsa.{n}{_CORRELATION_NOTES.get(n, '')}" for n, _ in named]
    items.extend(_EXTRA_CORRELATION_ROUTES)
    return f"{', '.join(items[:-1])}, or {items[-1]}"


def _categorical_tolerant_methods() -> str:
    """Name every categorical-tolerant route, for a rejection message."""
    named = _tolerant_names("categorical", _CATEGORICAL_ORDER)
    given = [f"jaxgsa.{n}{_CATEGORICAL_NOTES.get(n, '')}" for n, design in named if not design]
    design = [f"jaxgsa.{n}{_CATEGORICAL_NOTES.get(n, '')}" for n, is_d in named if is_d]
    listed = f"{', '.join(given[:-1])}, and {given[-1]}" if len(given) > 1 else given[0]
    if not design:
        return listed
    return f"{listed}, or the design-based {', '.join(design)}"


def _categorical_param_names(problem: Problem) -> list[str]:
    """Return the names of the problem's categorical parameters."""
    # Imported lazily: this module is type-checking-only on Problem.
    from jaxgsa.problem import _categorical_dims

    return [problem.names[d] for d, _ in _categorical_dims(problem)]


def _raise_categorical_and_correlated(method: str, names: list[str], *, design: bool) -> None:
    """Reject a problem that is both categorical and correlated.

    This combination has no variance-based route. The categorical-tolerant
    variance-based method (``jaxgsa.sobol``) refuses a declared correlation.
    The correlation-tolerant variance-based methods (``jaxgsa.vkoga``,
    ``jaxgsa.kucherenko``) refuse a categorical parameter. Naming only one of
    the two faults would send the user to a method that then refuses the
    other. All four gates therefore route the combined case here: design and
    analysis, categorical and correlated.

    Both ends of the library reach the same dead end, so both recommend the
    same three methods: ``jaxgsa.optimal_transport``, ``jaxgsa.borgonovo`` and
    ``jaxgsa.pawn``. Those are the only ones that accept both faults. Only the
    opening clause and the call to action differ. A design caller has no
    samples yet and is told how to draw them, while an analysis caller already
    holds ``(X, Y)``.

    Args:
        method: Fully qualified sampler or analyzer name for the message.
        names: Names of the categorical parameters, for the message.
        design: Whether the refusing gate builds a design (``True``) or
            analyzes given data (``False``). Selects the wording only.

    Raises:
        ValueError: Always. Callers check the combined condition first.
    """
    lead = "cannot build a design for" if design else "cannot analyze"
    if design:
        action = (
            "Draw correlated samples with jaxgsa.sampling.monte_carlo and "
            "analyze them with jaxgsa.optimal_transport, jaxgsa.borgonovo or "
            "jaxgsa.pawn, which handle both."
        )
    else:
        action = (
            "Analyze the same (X, Y) data with jaxgsa.optimal_transport, "
            "jaxgsa.borgonovo or jaxgsa.pawn, which handle both."
        )
    raise ValueError(
        f"{method} {lead} this problem: parameters {names} "
        "are categorical and problem.correlation declares a dependence "
        "structure, and no variance-based method supports categorical plus "
        "correlated inputs yet. The categorical-tolerant methods refuse a "
        "correlation, and the correlation-tolerant ones refuse a categorical "
        "parameter, so naming either route alone would send you in a circle. "
        f"{action} To drop one of the two constraints, use "
        "problem.with_correlation(None) or declare the parameter continuous."
    )


def _raise_correlated_design(problem: Problem, method: str) -> None:
    """Reject a correlated problem in a structured design sampler.

    A structured design places its points assuming independent marginals.
    Under a declared correlation the downstream estimators are undefined, not
    merely approximate. Sampling therefore refuses up front, rather than
    handing back a design whose analysis would be silently wrong. ``method``
    names the sampler, so this text does not have to list the callers.

    A problem that is also categorical gets the combined message instead. The
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
        _raise_categorical_and_correlated(method, names, design=True)
    raise ValueError(
        f"{method} builds a structured design whose sensitivity indices assume "
        "independent inputs, but problem.correlation declares a dependence "
        "structure. Draw correlated samples with jaxgsa.sampling.monte_carlo "
        "(which honors problem.correlation) and analyze them with a "
        f"correlation-tolerant given-data method: {_correlation_tolerant_methods()}. "
        "To analyze the independent problem instead, drop the matrix with "
        "problem.with_correlation(None)."
    )


def _raise_correlated_analysis(problem: Problem, method: str) -> None:
    """Reject a correlated problem in a correlation-naive analyzer.

    The indices such a method computes are invalid, not merely approximate,
    once ``problem.correlation`` declares a dependence structure, so the gate
    raises instead of warning.

    A problem that is also categorical gets the combined message instead. The
    correlated-only text would recommend methods that then refuse it for being
    categorical. This check therefore does not depend on being ordered before
    or after :func:`_raise_categorical_analysis`.

    Args:
        problem: Problem the analysis was requested for.
        method: Fully qualified analyzer name for the error message.

    Raises:
        ValueError: If ``problem`` declares a non-identity correlation. The
            message names the combined dead end when ``problem`` also declares
            a categorical parameter.
    """
    if not problem.has_correlated_inputs:
        return
    names = _categorical_param_names(problem)
    if names:
        _raise_categorical_and_correlated(method, names, design=False)
    raise ValueError(
        f"{method} computes indices that assume independent inputs, and they are "
        "invalid — not merely approximate — when problem.correlation declares a "
        "dependence structure. Use a correlation-tolerant given-data method "
        f"instead: {_correlation_tolerant_methods()}. To analyze the independent "
        "problem instead, drop the matrix with problem.with_correlation(None)."
    )


def _raise_categorical_design(problem: Problem, method: str) -> None:
    """Reject a categorical problem in an incompatible design sampler.

    The Morris and eFAST designs walk or sweep a continuous input space. An
    unordered categorical axis has no meaningful step or frequency, so
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
        _raise_categorical_and_correlated(method, names, design=True)
    raise ValueError(
        f"{method} requires continuous (orderable) inputs, but parameters "
        f"{names} are categorical. Use jaxgsa.sobol.sample (the Saltelli "
        "column-swap scheme is distribution-agnostic; it requires a problem "
        "with no declared correlation), or analyze given data with "
        "jaxgsa.optimal_transport, jaxgsa.borgonovo or jaxgsa.pawn."
    )


def _raise_categorical_analysis(problem: Problem, method: str) -> None:
    """Reject a categorical problem in a categorical-naive analyzer.

    A problem that is also correlated gets the combined message instead. The
    categorical-only text recommends the design-based ``jaxgsa.sobol``
    pipeline, which refuses a declared correlation, so it would send the user
    in a circle. This check therefore does not depend on being ordered before
    or after :func:`_raise_correlated_analysis`.

    Args:
        problem: Problem the analysis was requested for.
        method: Fully qualified analyzer name for the error message.

    Raises:
        ValueError: If ``problem`` declares any categorical parameter. The
            message names the combined dead end when ``problem`` also declares
            a non-identity correlation.
    """
    names = _categorical_param_names(problem)
    if not names:
        return
    if problem.has_correlated_inputs:
        _raise_categorical_and_correlated(method, names, design=False)
    raise ValueError(
        f"{method} treats every input as continuous, but parameters {names} "
        "are categorical (unordered level codes); its indices would depend on "
        "the arbitrary code order. Use a categorical-aware method instead: "
        f"{_categorical_tolerant_methods()}."
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

    Checks ``X`` and ``Y`` against the canonical jaxgsa layouts, taking ``X``'s
    row count as the expected sample count. jaxgsa never infers or transposes
    an axis.

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
            is ``False``. A problem that trips both faults gets the combined
            message, whichever of the two checks fires first.
    """
    if not correlation_ok:
        _raise_correlated_analysis(problem, method)
    if not categorical_ok:
        _raise_categorical_analysis(problem, method)
    _validate_x(problem, X)
    return _validate_output(Y, int(X.shape[0]), problem)


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


def _prepare_Y(Y: Array) -> tuple[Array, YLayout]:
    """Promote Y to the canonical 3-D shape ``(N, T, K)``.

    Args:
        Y: Model output array with 1, 2, or 3 dimensions.

    Returns:
        A tuple ``(Y_3d, layout)``, where ``layout`` records the rank the
        caller passed so :meth:`YLayout.squeeze` can undo the promotion.
    """
    # Normalize to 3-D so downstream kernels always see (samples, T, K)
    # without shape-dependent branching.
    if Y.ndim == 1:  # scalar output per sample -- both T and K are singleton
        return Y[:, None, None], YLayout.SCALAR
    if Y.ndim == 2:  # multi-output but single timestep -- only T is singleton
        return Y[:, None, :], YLayout.MULTI_OUTPUT
    return Y, YLayout.TIME_SERIES


# What a constant output slice does to the numbers, per method family. A
# variance-based method divides by that variance and reports NaN. A
# distribution-comparison method (Borgonovo, PAWN, optimal transport) finds
# every conditional distribution identical to the unconditional one and
# reports an exact 0, and Morris divides by the step size and reports 0 for
# all three of its measures. Those are three different things to tell a user,
# and the wording is the only part that differs, so it lives here rather than
# in three copies of the whole check.
_ZERO_VARIANCE_OUTCOMES: dict[str, tuple[str, str]] = {
    "nan": ("all indices will be NaN", "corresponding indices will be NaN"),
    "zero": ("all indices will be 0", "corresponding indices will be 0"),
    "morris": (
        "the screening measures (mu, mu_star, sigma) will be 0",
        "the corresponding screening measures (mu, mu_star, sigma) will be 0",
    ),
}


def _zero_variance_message(
    zero_mask: np.ndarray,
    *,
    n_outputs: int,
    K: int,
    n_trailing: int,
    output_names: tuple[str, ...] | None,
    outcome: str,
) -> str | None:
    """Build the zero-variance warning text, or ``None`` when nothing is wrong.

    Split out of :func:`_warn_zero_variance_slices` so the same wording serves
    the eager path and the host callback that runs under ``jit``/``vmap``. It
    reads concrete values only, which is exactly why it cannot run while
    tracing.

    Args:
        zero_mask: Flat boolean mask over the ``n_outputs`` slices, True where
            the slice is constant. Must be concrete, not a tracer.
        n_outputs: Number of flattened output slices.
        K: Size of the output dimension, used to split a flat index into
            ``(t, k)`` for a time-series output.
        n_trailing: Number of trailing dimensions of the caller's ``Y``: 0 for
            ``(N,)``, 1 for ``(N, K)``, 2 for ``(N, T, K)``.
        output_names: Optional names for the K output dimension, already
            length-checked against ``K``.
        outcome: Which consequence to report; a key of
            :data:`_ZERO_VARIANCE_OUTCOMES`.

    Returns:
        The warning message, or ``None`` when no slice is constant.
    """
    single_tail, plural_tail = _ZERO_VARIANCE_OUTCOMES[outcome]
    mask = np.asarray(zero_mask).reshape(-1)
    n_zero = int(mask.sum())
    if n_zero == 0:
        return None

    if n_outputs == 1:
        if output_names is not None and len(output_names) == 1:
            return f"jaxgsa: output '{output_names[0]}' has zero variance — {single_tail}"
        return f"jaxgsa: output has zero variance — {single_tail}"

    def _fmt_k(k: int) -> str:
        if output_names is not None:
            return f"k={k} ('{output_names[k]}')"
        return f"k={k}"

    zero_indices = [int(i) for i in np.flatnonzero(mask)]

    def _join(labels: list[str]) -> str:
        if len(labels) > 5:  # cap displayed labels to keep warnings readable
            return f"{', '.join(labels[:5])}, ... and {len(labels) - 5} more"
        return ", ".join(labels)

    if n_trailing == 1:  # single-timestep: flat index equals output index k
        label_str = _join([_fmt_k(k) for k in zero_indices])
        return (
            f"jaxgsa: {n_zero}/{n_outputs} output(s) have zero variance "
            f"({label_str}) — {plural_tail}"
        )
    if n_trailing == 2:  # multi-timestep: flat index encodes (t, k) in row-major order
        affected = []
        for idx in zero_indices:
            t, k = divmod(idx, K)
            affected.append(f"(t={t}, {_fmt_k(k)})")
        return (
            f"jaxgsa: {n_zero}/{n_outputs} output slice(s) have zero variance "
            f"[{_join(affected)}] — {plural_tail}"
        )
    # A scalar output (no trailing dims) has n_outputs == 1 and was handled above.
    return None


def _warn_zero_variance_slices(
    Y: Array,
    output_names: tuple[str, ...] | None = None,
    var_per_slice: Array | None = None,
    *,
    outcome: str = "nan",
    stacklevel: int = 2,
) -> None:
    """Check for zero-variance output slices before analysis and warn.

    A constant slice turns every index into ``0 / 0``, so the analysis still
    runs and reports NaN for that slice. The other slices are unaffected, so
    this warns rather than raising.

    The check works under ``jax.jit`` and ``jax.vmap``. It reads no value on
    the host while tracing: when ``Y`` is abstract, the warning is emitted from
    a :func:`jax.debug.callback` that runs when the compiled function executes.
    The message is the same, but it appears at run time rather than at trace
    time, so it repeats on every call of the compiled function and once per
    batch element under ``vmap``, and ``stacklevel`` does not apply to it.

    Args:
        Y: Model output array with shape ``(n_expanded, ...)`` where
            trailing dims are ``()``, ``(K,)``, or ``(T, K)``.
        output_names: Optional names for the K output dimension.
        var_per_slice: Optional pre-computed per-slice sample variance of
            ``Y`` over its first axis (any shape reshapeable to the flat
            slice axis). Pass it when the caller already computed the same
            variance, so it is not recomputed here.
        outcome: Which consequence to report, ``"nan"`` or ``"zero"``. See
            :data:`_ZERO_VARIANCE_OUTCOMES`.
        stacklevel: Frames to skip so the warning points at the user's
            ``analyze()`` call rather than at this helper.
    """
    _ = _ZERO_VARIANCE_OUTCOMES[outcome]  # fail fast on a bad outcome key
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

    # Sample variance along axis 0; zero means the output is constant
    # and Sobol indices become 0/0 = NaN.
    if var_per_slice is None:
        var_per_slice = jnp.var(flat, axis=0)
    zero_mask = var_per_slice.reshape(-1) == 0

    def emit(mask: Array | np.ndarray, level: int) -> None:
        """Warn about the constant slices `mask` marks, if there are any."""
        msg = _zero_variance_message(
            np.asarray(mask),
            n_outputs=n_outputs,
            K=K,
            n_trailing=len(trailing),
            output_names=output_names,
            outcome=outcome,
        )
        if msg is not None:
            warnings.warn(msg, stacklevel=level, category=JaxgsaWarning)

    if isinstance(zero_mask, jax.core.Tracer):
        # Under jit or vmap the mask is abstract: which slices are constant is
        # not known while tracing, and `warnings.warn` cannot run there anyway.
        # `jax.debug.callback` moves the decision to execution time, where the
        # values are real, so the warning still reaches the user -- once per
        # call of the compiled function, and once per batch element under vmap.
        # The stacklevel is 1 because the callback runs from XLA, not from a
        # Python frame chain that leads back to the user's `analyze()` call.
        jax.debug.callback(lambda mask: emit(mask, 1), zero_mask)
        return

    emit(zero_mask, stacklevel)


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
