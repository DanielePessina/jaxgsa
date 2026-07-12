"""Shared output helpers for analysis entrypoints."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from gsax.problem import Problem


class LayoutOps(NamedTuple):
    """Transformations :func:`_infer_output_layout_ops` applied to reach the
    canonical layout, so companion arrays (a Jacobian, an emulator prediction)
    can be moved in lockstep instead of re-derived from shapes.

    The fields record, in application order:
        sample_axis: axis moved to the front (rule 1), or ``None`` if unmoved.
        inserted_output_axis: a singleton K axis was appended (rule 2, a single
            labeled output with several columns: ``(n, T) -> (n, T, 1)``).
        swapped_tk: the trailing two axes were exchanged (rule 3,
            ``(n, K, T) -> (n, T, K)``).

    An all-default ``LayoutOps()`` is the identity (Y was already canonical).
    ``inserted_output_axis`` and ``swapped_tk`` are mutually exclusive (2-D vs
    3-D branch).
    """

    sample_axis: int | None = None
    inserted_output_axis: bool = False
    swapped_tk: bool = False


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


def _infer_output_layout_ops(
    Y: Array,
    problem: Problem,
    n_expected: int | None,
    *,
    stacklevel: int = 3,
) -> tuple[Array, LayoutOps]:
    """Resolve a user-supplied output array to the canonical axis layout.

    gsax's canonical layouts are ``(n,)``, ``(n, K)``, and ``(n, T, K)``, but
    users do not always hand over exactly that. This helper infers the
    semantic axes from the two signals available at every public entry point:
    the expected sample count ``n_expected`` identifies the sample axis, and
    ``len(problem.output_names)`` (when set) identifies the output axis K; a
    remaining axis is time. The ladder is strict — exact canonical shapes pass
    silently, unambiguously recoverable layouts are fixed with a
    ``UserWarning`` naming the transformation, and ambiguous ones raise. It
    never guesses.

    Rules, in order:
      1. Sample axis: if ``Y.shape[0] != n_expected`` but exactly one axis has
         that length, it is moved first (with a warning); no matching axis
         raises. When the leading axis already matches, position is trusted
         even if another axis coincidentally matches too.
      2. 2-D ``(n, M)``: with exactly one entry in ``problem.output_names``,
         the columns are T timepoints of that single output and Y is reshaped
         to ``(n, M, 1)``. With several entries, ``M`` must equal
         ``len(output_names)`` (multi-output). Without ``output_names``, 2-D
         always means ``(n, K)``.
      3. 3-D ``(n, A, B)``: expected ``(n, T, K)``. If ``output_names`` is set
         and only the middle axis matches its length, the trailing axes are
         swapped (with a warning); if neither trailing axis matches, raises.

    Args:
        Y: User-supplied output array, 1-D to 3-D.
        problem: Problem definition; ``output_names`` (when set) pins K.
        n_expected: Expected sample count (X rows for given-data methods, the
            design's unique row count for Sobol/Morris), or ``None`` when the
            caller cannot know it independently (rule 1 is skipped).
        stacklevel: Passed through to ``warnings.warn`` so the layout warnings
            point at the user's call site. Direct callers use the default 3;
            the given-data wrappers pass 4 to skip their extra frame.

    Returns:
        A tuple ``(Y, ops)`` where ``Y`` is in canonical layout — ``(n,)``,
        ``(n, K)``, or ``(n, T, K)`` — and ``ops`` records the transformations
        applied so companion arrays can be moved in lockstep.

    Raises:
        ValueError: If ``Y`` is not 1-D/2-D/3-D, no axis matches the expected
            sample rows, or a labeled output axis cannot be located.
    """
    Y = jnp.asarray(Y)
    if Y.ndim not in (1, 2, 3):
        raise ValueError(f"Y must be 1-D (N,), 2-D (N, K), or 3-D (N, T, K), got ndim={Y.ndim}")

    sample_axis: int | None = None
    inserted_output_axis = False
    swapped_tk = False

    if n_expected is not None and Y.shape[0] != n_expected:
        matches = [ax for ax, size in enumerate(Y.shape) if size == n_expected]
        if len(matches) == 1:
            warnings.warn(
                f"gsax: Y has shape {tuple(Y.shape)} but {n_expected} sample rows were "
                f"expected; interpreting axis {matches[0]} as the sample axis and "
                "moving it first",
                stacklevel=stacklevel,
            )
            sample_axis = matches[0]
            Y = jnp.moveaxis(Y, matches[0], 0)
        else:
            ambiguity = "no unambiguous" if matches else "no"
            raise ValueError(
                f"Y has shape {tuple(Y.shape)} with {ambiguity} axis matching the "
                f"expected {n_expected} sample rows; pass Y as (n,), (n, K), or "
                f"(n, T, K) with n={n_expected}"
            )

    K_labeled = len(problem.output_names) if problem.output_names is not None else None

    if Y.ndim == 2 and K_labeled is not None:
        M = Y.shape[1]
        if K_labeled == 1:
            # One labeled output: the columns are timepoints, not outputs.
            # Flow as genuine (n, T, 1) so results keep the labeled output axis.
            Y = Y[:, :, None]
            inserted_output_axis = True
        elif M != K_labeled:
            raise ValueError(
                f"Y has {M} columns but problem.output_names lists {K_labeled} "
                "outputs; a 2-D Y must have one column per named output "
                "(pass (n, T, K) for multi-output time series)"
            )
    elif Y.ndim == 3 and K_labeled is not None and Y.shape[2] != K_labeled:
        if Y.shape[1] == K_labeled:
            warnings.warn(
                f"gsax: Y has shape {tuple(Y.shape)} but only its middle axis "
                f"matches the {K_labeled} named outputs; interpreting Y as "
                "(n, K, T) and swapping the trailing axes to (n, T, K)",
                stacklevel=stacklevel,
            )
            Y = jnp.swapaxes(Y, 1, 2)
            swapped_tk = True
        else:
            raise ValueError(
                f"3-D Y has shape {tuple(Y.shape)} but no trailing axis matches "
                f"the {K_labeled} entries in problem.output_names; expected "
                "(n, T, K)"
            )
    return Y, LayoutOps(sample_axis, inserted_output_axis, swapped_tk)


def _infer_output_layout(
    Y: Array,
    problem: Problem,
    n_expected: int | None,
    *,
    stacklevel: int = 3,
) -> Array:
    """Canonical-layout shim returning only Y (see :func:`_infer_output_layout_ops`).

    Keeps the historical three-positional-argument ``Array`` contract for
    callers that do not need the ops record. The ``+1`` on ``stacklevel``
    compensates for this extra frame so warnings still point at user code.
    """
    return _infer_output_layout_ops(Y, problem, n_expected, stacklevel=stacklevel + 1)[0]


def _validate_xy_inputs_ops(problem: Problem, X: Array, Y: Array) -> tuple[Array, LayoutOps]:
    """Validate the shared ``(problem, X, Y)`` contract, returning the ops record.

    Like :func:`_validate_xy_inputs` but also returns the :class:`LayoutOps`
    describing how Y was canonicalized, for callers (PCE, HDMR) that need to
    mirror the layout on emulator predictions.

    Args:
        problem: Problem definition with ``num_vars`` parameters.
        X: Input sample matrix, expected shape ``(N, D)``.
        Y: Model output, 1-D, 2-D, or 3-D (layout inferred when recoverable).

    Returns:
        A tuple ``(Y, ops)`` with Y in canonical layout.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, or Y's layout cannot be resolved against X's row count.
    """
    _validate_x(problem, X)
    return _infer_output_layout_ops(Y, problem, int(X.shape[0]), stacklevel=4)


def _validate_xy_inputs(problem: Problem, X: Array, Y: Array) -> Array:
    """Validate the shared ``(problem, X, Y)`` contract of given-data methods.

    Validates X and resolves Y to the canonical layout via
    :func:`_infer_output_layout_ops`, using X's row count as the expected
    sample count. Callers must use the returned Y.

    Args:
        problem: Problem definition with ``num_vars`` parameters.
        X: Input sample matrix, expected shape ``(N, D)``.
        Y: Model output, 1-D, 2-D, or 3-D (layout inferred when recoverable).

    Returns:
        ``Y`` in canonical ``(N,)`` / ``(N, K)`` / ``(N, T, K)`` layout.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, or Y's layout cannot be resolved against X's row count.
    """
    _validate_x(problem, X)
    return _infer_output_layout_ops(Y, problem, int(X.shape[0]), stacklevel=4)[0]


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
    """Promote Y to a canonical 3-D shape (n_total, T, K).

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
) -> None:
    """Check for zero-variance output slices before analysis and warn.

    Args:
        Y: Model output array with shape ``(n_expanded, ...)`` where
            trailing dims are ``()``, ``(K,)``, or ``(T, K)``.
        output_names: Optional names for the K output dimension.
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
        raise ValueError(
            f"len(output_names)={len(output_names)} does not match number of outputs K={K}"
        )

    def _fmt_k(k: int) -> str:
        if output_names is not None:
            return f"k={k} ('{output_names[k]}')"
        return f"k={k}"

    # Sample variance along axis 0; zero means the output is constant
    # and Sobol indices become 0/0 = NaN.
    var_per_slice = jnp.var(flat, axis=0)
    zero_mask = var_per_slice == 0
    n_zero = int(jnp.sum(zero_mask))

    if n_zero == 0:
        return

    if n_outputs == 1:
        if output_names is not None and len(output_names) == 1:
            msg = f"gsax: output '{output_names[0]}' has zero variance — all indices will be NaN"
        else:
            msg = "gsax: output has zero variance — all indices will be NaN"
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
            f"gsax: {n_zero}/{n_outputs} output(s) have zero variance "
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
            f"gsax: {n_zero}/{n_outputs} output slice(s) have zero variance "
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
