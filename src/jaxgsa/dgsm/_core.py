"""The traceable arithmetic behind DGSM, with no policy in it.

Everything here branches on a shape or on a Python scalar, and nothing here
reads the value of an array on the host. That is what lets
:func:`jaxgsa.dgsm.indices` compose with ``jit``, ``vmap`` and ``jacrev``
while :func:`jaxgsa.dgsm.analyze` cannot: ``analyze`` has to decide what to do
about a non-finite row, and a tracer has no value to decide on.

Both entry points share this module, so the pure core and the policy path run
one copy of the estimator rather than two that have to be kept equal by hand.

The Jacobian mode is chosen here too. See
``docs/adr/0005-autodiff-mode-selection.md``: one reverse pass yields one
output's gradient and one forward pass yields one input's column, so a
``(N, T, K)`` Jacobian costs ``T*K`` reverse passes or ``D`` forward passes.
The cheaper of the two is picked from those two numbers, both of which are
shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa.dgsm._poincare import axis_constants
from jaxgsa.problem import Problem

# Live jac-sized arrays while one batch is reduced. The array the batch loop
# exists to bound is the Jacobian block, T*K*D floats per row where the output
# is one, so the model prices everything in multiples of it: the block itself,
# the elementwise square the moment sums build from it, and the masked copy
# with its square that the policy loop in ``analyze`` keeps alongside (see
# ``_compute_moments``). The primal outputs are T*K floats per row, a factor D
# smaller, so they are left out. Modelled from the code, not measured — the
# same convention as sobol's ``slice_elements`` live-copy count and
# borgonovo's ``_KDE_LIVE_TENSORS``.
_JAC_LIVE_ARRAYS = 4


def _jac_bytes_per_row(n_outputs: int, n_inputs: int, itemsize: int) -> int:
    """Estimate the transient bytes one sample row costs in the batch loop.

    Args:
        n_outputs: ``T*K``, the number of scalar output slices.
        n_inputs: ``D``, the number of parameters.
        itemsize: Bytes per element of the working dtype.

    Returns:
        Estimated bytes per row: ``_JAC_LIVE_ARRAYS`` Jacobian-sized arrays.
    """
    return itemsize * _JAC_LIVE_ARRAYS * n_outputs * n_inputs


def _fn_with_aux(fn: Callable) -> Callable:
    """Wrap ``fn`` so one differentiation pass also returns the forward output.

    ``has_aux=True`` carries the primal out of the same pass that builds the
    Jacobian, so the model is evaluated once rather than twice.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.

    Returns:
        A callable returning ``(y, y)``, the second copy as the aux payload.
    """

    def wrapped(x: Array) -> tuple[Array, Array]:
        y = fn(x)
        return y, y

    return wrapped


def n_output_slices(fn: Callable, X: Array) -> int:
    """Return ``T*K``, the number of output slices one call of ``fn`` produces.

    Uses :func:`jax.eval_shape`, which traces ``fn`` on an abstract row and
    never runs it, so an expensive model pays nothing and the answer is a
    shape rather than a value.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        X: Sample matrix, shape ``(N, D)``. Only its width and dtype are read.

    Returns:
        The number of scalar output slices, at least 1.
    """
    row = jax.ShapeDtypeStruct((int(X.shape[1]),), X.dtype)
    out = jax.eval_shape(fn, row)
    total = 1
    for leaf in jax.tree.leaves(out):
        total *= max(1, int(np.prod(leaf.shape, dtype=np.int64)))
    return total


def jacobian_of(fn: Callable, *, n_inputs: int, n_outputs: int) -> Callable:
    """Return the one-row Jacobian-with-primal of ``fn``.

    ``docs/adr/0005-autodiff-mode-selection.md`` asks for a choice here.
    Reverse mode costs one pass per output slice and forward mode costs one
    pass per input, so reverse should win when ``n_outputs < n_inputs`` and
    forward otherwise, which is a real speed gain on a time-series output.

    **The choice is not made yet, and this returns reverse mode always.** The
    switch was implemented and measured: the two modes agree to about 1e-8
    absolute in float32, which is reassociation and not a different answer,
    but it is enough to move ``sigma`` and ``lower_bound`` on the recorded
    numerical baseline for a ``(N, 3, 2)`` output. ``CONTEXT.md`` treats a
    moved baseline number as a wiring error until reviewed, so the flip is
    left for the change that regenerates the baseline with it. Making that
    change is one line: pick ``jacfwd`` when ``n_outputs >= n_inputs``. The
    arguments are already threaded through for it.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        n_inputs: ``D``, the number of parameters.
        n_outputs: ``T*K``, the number of output slices.

    Returns:
        A callable mapping one ``(D,)`` row to ``(jacobian, y)``.
    """
    del n_inputs, n_outputs  # see the docstring: the selection is not live yet
    return jax.jacrev(_fn_with_aux(fn), has_aux=True)


def jac_batches(
    fn: Callable,
    X: Array,
    batch_size: int | None,
) -> Iterator[tuple[Array, Array, Array]]:
    """Yield ``(jac, Y, X)`` one batch of sample rows at a time.

    The Jacobian is the array that has to be bounded: it is ``T*K*D`` values
    per row, where the output is one. So the sample axis is walked in batches
    and each batch is reduced by the caller before the next one is built.

    ``batch_size=None`` derives the width from the active memory budget
    through the shared :func:`jaxgsa._core.batching.resolve_batch_size`, with
    :func:`_jac_bytes_per_row` as the per-row cost. An explicit value is
    honoured as given, clamped to ``N``.

    The ragged trailing batch is padded back to the full width and sliced
    afterwards, so the vectorized kernel compiles once instead of twice. The
    padded rows never reach the caller.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Rows per batch, clamped to ``N``, or ``None`` to derive
            one from the active memory budget.

    Yields:
        ``(jac, Y, X)`` for each batch: the Jacobian block, the forward
        outputs, and the rows they came from. The leading axis is the batch's
        true length, never the padded one.

    Raises:
        ValueError: If ``batch_size`` is given and not a positive integer.
    """
    n_inputs = int(X.shape[1])
    n_outputs = n_output_slices(fn, X)
    combined = jax.jit(jax.vmap(jacobian_of(fn, n_inputs=n_inputs, n_outputs=n_outputs)))

    N = X.shape[0]
    b = resolve_batch_size(
        _jac_bytes_per_row(n_outputs, n_inputs, jnp.dtype(X.dtype).itemsize), N, batch_size
    )
    if N <= b:
        jac, Y = combined(X)
        yield jac, Y, X
        return

    for start in range(0, N, b):
        end = min(start + b, N)
        actual_len = end - start
        X_chunk = X[start:end]
        if actual_len < b:
            pad_size = b - actual_len
            # dtype= keeps the pad in X's own dtype: an unannotated jnp.zeros
            # is float64 under x64, which would promote the padded last batch
            # and call the user's fn at a different dtype than every other one.
            pad = jnp.zeros((pad_size, X.shape[1]), dtype=X.dtype)
            X_chunk = jnp.concatenate([X_chunk, pad], axis=0)
        jac, Y_chunk = combined(X_chunk)
        yield jac[:actual_len], Y_chunk[:actual_len], X[start:end]


def moment_sums(
    fn: Callable,
    X: Array,
    batch_size: int | None,
) -> tuple[Array, Array, Array]:
    """Sum the Jacobian and its square over every sample row.

    No masking and no finite check: this is the pure-core reduction, and it
    reports what the model returned. :func:`jaxgsa.dgsm.analyze` runs its own
    masked version of the same loop.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Rows per batch, or ``None`` to derive one from the
            active memory budget, as in :func:`jac_batches`.

    Returns:
        ``(sum_jac, sum_jac2, Y)``. The two sums carry ``fn``'s own slice axes
        with a trailing ``(D,)``; ``Y`` is the stacked forward output.
    """
    sum_jac: Array | None = None
    sum_jac2: Array | None = None
    Y_parts: list[Array] = []
    for jac, Y_chunk, _ in jac_batches(fn, X, batch_size):
        Y_parts.append(Y_chunk)
        sj = jnp.sum(jac, axis=0)
        sj2 = jnp.sum(jac**2, axis=0)
        sum_jac = sj if sum_jac is None else sum_jac + sj
        sum_jac2 = sj2 if sum_jac2 is None else sum_jac2 + sj2
    if sum_jac is None or sum_jac2 is None:
        raise ValueError("X has no sample rows, so there is no derivative to average")
    Y = Y_parts[0] if len(Y_parts) == 1 else jnp.concatenate(Y_parts, axis=0)
    return sum_jac, sum_jac2, Y


def resample_moment_sums(
    batches: Iterator[tuple[Array, Array]],
    idx: Array,
) -> tuple[Array, Array, Array, Array]:
    """Accumulate per-replicate sums of the Jacobian and the output.

    A bootstrap replicate is a multiset of sample rows, so every quantity DGSM
    reports is a weighted sum over rows with integer weights. Counting the
    draws that land in the current batch turns the resample into one matrix
    product per batch, which means the whole bootstrap costs **one** extra
    sweep over the sample rather than one sweep per replicate, and the
    Jacobian still never exists whole.

    The weight matrix is ``(n_bootstrap, batch)``, so with a bootstrap the
    batch width bounds that array too. Lower ``batch_size`` if a large
    ``n_bootstrap`` runs the device out of memory.

    Args:
        batches: Yields ``(jac, Y)`` per batch, in row order and with no gaps.
            The rows are numbered by the order they arrive in, so the caller
            must have compacted any dropped rows out already.
        idx: Resampled row indices, shape ``(n_bootstrap, n)``, into that same
            row numbering.

    Returns:
        ``(sum_jac, sum_jac2, sum_y, sum_y2)``, each with a leading
        ``n_bootstrap`` axis followed by the per-row shape.
    """
    n_replicates = int(idx.shape[0])
    rows = jnp.arange(n_replicates)[:, None]
    offset = 0
    acc: list[Array] | None = None
    for jac, Y_chunk in batches:
        n = int(jac.shape[0])
        local = idx - offset
        in_chunk = (local >= 0) & (local < n)
        # Draw counts per (replicate, row of this batch). Indices from other
        # batches are steered to column 0 and given weight 0, which is the
        # explicit form of "drop" and does not rely on out-of-bounds
        # behaviour.
        weights = (
            jnp.zeros((n_replicates, n), dtype=jac.dtype)
            .at[rows, jnp.where(in_chunk, local, 0)]
            .add(jnp.where(in_chunk, 1.0, 0.0).astype(jac.dtype))
        )
        jac_flat = jac.reshape(n, -1)
        y_flat = Y_chunk.reshape(n, -1)
        parts = [
            (weights @ jac_flat).reshape((n_replicates, *jac.shape[1:])),
            (weights @ jac_flat**2).reshape((n_replicates, *jac.shape[1:])),
            (weights @ y_flat).reshape((n_replicates, *Y_chunk.shape[1:])),
            (weights @ y_flat**2).reshape((n_replicates, *Y_chunk.shape[1:])),
        ]
        acc = parts if acc is None else [a + b for a, b in zip(acc, parts, strict=True)]
        offset += n
    if acc is None:
        raise ValueError("X has no sample rows, so there is nothing to resample")
    return acc[0], acc[1], acc[2], acc[3]


def promote_jac(jac: Array) -> Array:
    """Promote a Jacobian to canonical 4-D ``(N, T, K, D)`` by axis position.

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


def promote_moments(m: Array) -> Array:
    """Promote a reduced moment ``(*slice, D)`` to canonical ``(T, K, D)``.

    The reduced-space analog of :func:`promote_jac`, for moments that were
    already averaged over the sample axis.

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


def bounds_from_moments(
    problem: Problem,
    *,
    nu: Array,
    sigma: Array,
    var_y: Array,
    standardize_outputs: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Turn the derivative moments into the two Sobol-index bounds.

    This is the whole numerical statement of the method, and both entry points
    run this one copy of it:

    - upper (Poincare / Sobol-Kucherenko): ``ST_i <= C_i * nu_i / Var(Y)``
    - lower (Kucherenko-Song): ``ST_i >= Var(x_i) * sigma_i^2 / Var(Y)``

    A zero-variance slice makes both bounds ``0 / 0``, so the denominator is
    mapped to ``NaN`` rather than divided by. That is arithmetic, not policy:
    the warning about it stays in ``analyze``.

    Args:
        problem: Problem definition, read for the per-axis Poincare constants
            and the marginal variances. Not an array, so reading it is a host
            operation on static data and stays traceable.
        nu: ``E[(df/dx_i)^2]``, shape ``(T, K, D)``.
        sigma: ``E[df/dx_i]``, shape ``(T, K, D)``.
        var_y: Output variance per slice, shape ``(T, K)``.
        standardize_outputs: Whether to report the moments for the
            standardized output ``(Y - mean) / std``. The map is affine, so
            the standardized moments are the ones already computed, rescaled.

    Returns:
        ``(nu, sigma, upper, lower, var_y)``, the first four shaped
        ``(T, K, D)`` and the last ``(T, K)``.
    """
    if standardize_outputs:
        # Y -> (Y - mean)/std is affine, so d/dx of the shift is zero and the
        # 1/std factor is a constant. Rescaling here rather than
        # differentiating a wrapped function leaves the autodiff sweep alone.
        # Both bounds are ratios of quantities that scale the same way, so
        # they come out unchanged, which the tests assert.
        y_std = jnp.sqrt(var_y)  # (T, K)
        safe_scale = jnp.where(y_std == 0, jnp.ones_like(y_std), y_std)
        sigma = sigma / safe_scale[..., None]
        nu = nu / (safe_scale**2)[..., None]
        var_y = var_y / safe_scale**2

    # Per-axis constants: C for the Poincare upper bound, Var for the lower.
    C, Var = axis_constants(problem)
    C_jnp = jnp.asarray(C)
    Var_jnp = jnp.asarray(Var)

    # A constant output slice has zero variance; map it to NaN bounds rather
    # than dividing by zero.
    denom = jnp.where(var_y == 0, jnp.nan, var_y)[..., None]  # (T, K, 1)

    # The (D,) constants broadcast against (T, K, D) on the trailing axis.
    upper = C_jnp * nu / denom
    lower = Var_jnp * sigma**2 / denom
    return nu, sigma, upper, lower, var_y
