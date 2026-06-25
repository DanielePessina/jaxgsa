"""Shared output helpers for analysis entrypoints."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


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
    if Y.ndim == 1:
        Y = Y[:, None, None]
        squeeze_time = True
        squeeze_output = True
    elif Y.ndim == 2:
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
    import warnings

    flat = Y.reshape(Y.shape[0], -1)
    n_outputs = flat.shape[1]

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

    zero_indices = [int(i) for i in jnp.where(zero_mask)[0]]

    if len(trailing) == 1:
        labels = [_fmt_k(k) for k in zero_indices]
        if len(labels) > 5:
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
    elif len(trailing) == 2:
        affected = []
        for idx in zero_indices:
            t, k = divmod(idx, K)
            affected.append(f"(t={t}, {_fmt_k(k)})")
        if len(affected) > 5:
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
    y_mean = jnp.mean(Y, axis=0)
    y_std = jnp.std(Y, axis=0)
    safe_scale = jnp.where(y_std == 0, jnp.ones_like(y_std), y_std)
    Y_norm = (Y - y_mean) / safe_scale
    return Y_norm, y_mean, y_std, safe_scale
