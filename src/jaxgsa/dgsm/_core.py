"""The traceable arithmetic behind DGSM, with no policy in it.

Everything here branches on a shape or on a Python scalar, and nothing here
reads the value of an array on the host. That is what lets
:func:`jaxgsa.dgsm.indices` compose with ``jit``, ``vmap`` and ``jacrev``
while :func:`jaxgsa.dgsm.analyze` cannot: ``analyze`` has to decide what to do
about a non-finite row, and a tracer has no value to decide on.

Both entry points share this module, so the pure core and the policy path run
one copy of the estimator rather than two that have to be kept equal by hand.

The Jacobian mode is chosen here too. One reverse pass yields one output's
gradient and one forward pass yields one input's column, so a ``(N, T, K)``
Jacobian costs ``T*K`` reverse passes or ``D`` forward passes. The cheaper of
the two is picked from those two numbers, both of which are shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core.batching import resolve_batch_size
from jaxgsa._core.bootstrap import draw_rows, resample_keys
from jaxgsa._core.invalid import _row_finite
from jaxgsa.dgsm._poincare import axis_constants
from jaxgsa.problem import Problem

# Live jac-sized arrays while one batch is reduced. The array the batch loop
# exists to bound is the Jacobian block, T*K*D floats per row where the output
# is one, so the model prices everything in multiples of it: the block itself,
# the elementwise square the moment sums build from it, and the masked copy
# with its square that the same loop keeps alongside (see
# ``jacobian_sums``). The primal outputs are T*K floats per row, a factor D
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


def slice_count(out: object) -> int:
    """Return ``T*K``, the number of scalar slices in one evaluated ``fn`` output.

    Shared by :func:`n_output_slices`, which builds ``out`` itself with
    :func:`jax.eval_shape`, and by
    :func:`jaxgsa.dgsm._analyze._check_point_callable`, which already has an
    ``out`` from its own shape check and would otherwise trace the same row a
    second time to learn the same number.

    Args:
        out: A pytree of shaped leaves, such as the result of
            :func:`jax.eval_shape`.

    Returns:
        The number of scalar output slices, at least 1.
    """
    total = 1
    for leaf in jax.tree.leaves(out):
        total *= max(1, int(np.prod(leaf.shape, dtype=np.int64)))
    return total


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
    return slice_count(jax.eval_shape(fn, row))


def jacobian_mode(*, n_inputs: int, n_outputs: int) -> str:
    """Return the autodiff mode the two shape numbers select.

    Reverse mode costs one pass per output slice and forward mode costs one
    pass per input, so forward wins when there are more output slices than
    inputs. The tie goes to reverse mode: the costs are equal there, and
    reverse is what every scalar-output call has always run.

    Args:
        n_inputs: ``D``, the number of parameters.
        n_outputs: ``T*K``, the number of output slices.

    Returns:
        ``"forward"`` when ``n_outputs > n_inputs``, else ``"reverse"``.
    """
    return "forward" if n_outputs > n_inputs else "reverse"


@lru_cache(maxsize=32)
def _get_jacobian_kernel(fn: Callable, n_inputs: int, n_outputs: int) -> Callable:
    """Build and cache the jitted, vmapped Jacobian-with-primal kernel for one model.

    ``jacobian_of`` returns a fresh closure on every call, and wrapping a
    fresh closure in ``jax.jit`` builds a fresh compilation cache entry too:
    two calls to ``analyze`` on the same model would otherwise retrace and
    recompile from nothing. Caching this builder on ``(fn, n_inputs,
    n_outputs)`` — the same three quantities that decide the kernel's shape
    and mode — reuses one compiled kernel across calls, the way
    :func:`jaxgsa.efast._analyze._get_efast_kernel` memoises its kernel by
    design shape. A plain function is hashable by identity, so two
    structurally equal but distinct function objects (for example two
    lambdas) still get one cache entry each.

    Not every model object is hashable: a model written as a plain
    ``@dataclass`` with ``__call__`` has ``__hash__`` set to ``None``.
    :func:`jac_batches` tests for that and builds the kernel without the
    cache in that case, so an unhashable model still runs. It pays one trace
    per call, which is what every model paid before this cache existed.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        n_inputs: ``D``, the number of parameters.
        n_outputs: ``T*K``, the number of output slices.

    Returns:
        A compiled callable mapping a batch of ``(D,)`` rows, vmapped over
        the leading axis, to ``(jacobian, y)``.
    """
    return jax.jit(jax.vmap(jacobian_of(fn, n_inputs=n_inputs, n_outputs=n_outputs)))


def jacobian_of(fn: Callable, *, n_inputs: int, n_outputs: int) -> Callable:
    """Return the one-row Jacobian-with-primal of ``fn``.

    The autodiff mode is selected by :func:`jacobian_mode`: ``jax.jacfwd``
    when ``n_outputs > n_inputs``, ``jax.jacrev`` otherwise. The two modes compute
    the same Jacobian; float arithmetic order differs, so results can move at
    float precision where the mode changed. That move was reviewed once, in
    the baseline regeneration recorded in ``scripts/baseline/README.md``.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        n_inputs: ``D``, the number of parameters.
        n_outputs: ``T*K``, the number of output slices.

    Returns:
        A callable mapping one ``(D,)`` row to ``(jacobian, y)``, both in
        ``result_type(x, y)``.
    """
    if jacobian_mode(n_inputs=n_inputs, n_outputs=n_outputs) == "forward":
        jac_fn = jax.jacfwd(_fn_with_aux(fn), has_aux=True)
    else:
        jac_fn = jax.jacrev(_fn_with_aux(fn), has_aux=True)

    def promoted(x: Array) -> tuple[Array, Array]:
        # The Jacobian takes X's dtype and the primal takes fn's. Both go to
        # their common dtype, so float32 X with a float64 model accumulates
        # and reports every field in float64, as JAX promotion would.
        jac, y = jac_fn(x)
        dtype = jnp.result_type(jac, y)
        return jac.astype(dtype), jnp.asarray(y).astype(dtype)

    return promoted


def _is_hashable(obj: object) -> bool:
    """Tell whether ``hash(obj)`` works.

    ``isinstance(obj, Hashable)`` is not enough: a frozen dataclass with an
    array field has a ``__hash__`` that raises ``TypeError`` when called.

    Args:
        obj: Any object.

    Returns:
        ``True`` if ``obj`` can key a cache.
    """
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def jac_batches(
    fn: Callable,
    X: Array,
    batch_size: int | None,
    *,
    n_outputs: int | None = None,
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
        n_outputs: ``T*K``, if the caller already traced ``fn`` once (for
            example through :func:`jaxgsa.dgsm._analyze._check_point_callable`)
            and knows the answer. ``None`` (default) traces it here through
            :func:`n_output_slices`.

    Yields:
        ``(jac, Y, X)`` for each batch: the Jacobian block, the forward
        outputs, and the rows they came from. The leading axis is the batch's
        true length, never the padded one.

    Raises:
        ValueError: If ``batch_size`` is given and not a positive integer.
    """
    n_inputs = int(X.shape[1])
    if n_outputs is None:
        n_outputs = n_output_slices(fn, X)
    if _is_hashable(fn):
        combined = _get_jacobian_kernel(fn, n_inputs, n_outputs)
    else:
        # A model that cannot key the cache (a plain dataclass, or a frozen
        # one with array fields) gets its kernel built directly, not an error.
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


def _mask_rows(array: Array, row_ok: Array) -> Array:
    """Zero out the rows a mask rejects, without arithmetic on their values.

    ``jnp.where`` is required here rather than a multiply by ``0.0``: a
    multiply leaves ``NaN * 0 == NaN``, which is exactly the value the mask
    exists to remove.

    Args:
        array: Array whose leading axis is the sample axis.
        row_ok: Boolean mask of shape ``(N,)``.

    Returns:
        ``array`` with the rejected rows replaced by zeros.
    """
    return jnp.where(row_ok.reshape((-1,) + (1,) * (array.ndim - 1)), array, 0.0)


def _join(parts: list[Array]) -> Array:
    """Concatenate batch pieces back into one sample-axis array.

    Args:
        parts: One array per batch, all sharing their trailing shape.

    Returns:
        The pieces stacked along the sample axis. A single piece is returned
        unchanged, so the unbatched path copies nothing.
    """
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


@dataclass(frozen=True)
class _JacobianSums:
    """Unreduced derivative sums from one autodiff sweep, in two versions.

    The autodiff path makes ``Y`` itself, inside a jitted ``vmap`` of the
    one-row Jacobian, and reduces each batch of Jacobian rows to a running
    total as soon as it has it. A row that is already inside that total
    cannot be taken out again: ``NaN`` plus anything is ``NaN``. So the batch
    loop keeps two totals side by side, one over every row and one over the
    clean rows only, and a caller picks the one its policy calls for. The
    masking happens where the row still exists, which is inside the loop.

    Keeping both costs one extra pair of sums per batch. The Jacobian itself
    is what the batch loop exists to bound, and neither total holds a sample
    axis, so peak memory is unchanged.

    ``jac_finite`` and ``row_ok`` stay JAX arrays here, not NumPy ones:
    :func:`moment_sums` reuses this same loop from inside
    :func:`jaxgsa.dgsm.indices`, which must survive ``jit``, ``vmap`` and
    ``jacrev``, and a NumPy conversion forces a concrete host read that a
    tracer cannot give. The policy path in ``analyze`` runs eagerly, so it
    converts them itself where it consumes them.

    Attributes:
        Y: Stacked raw ``fn`` output, shape ``(N,)`` / ``(N, K)`` /
            ``(N, T, K)``.
        jac_finite: For each row, whether every entry of its Jacobian block is
            finite, shape ``(N,)``. Reported separately from ``Y`` because a
            derivative can blow up where the output does not, and ``nu`` is
            built from the derivative.
        row_ok: For each row, whether ``X``, ``Y`` and the Jacobian are all
            finite, shape ``(N,)``. This is the mask the ``*_kept`` totals
            used.
        sum_jac: Sum of the Jacobian over every row, shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        sum_jac2: Sum of the squared Jacobian over every row, same shape.
        sum_jac_kept: Sum of the Jacobian over the rows in ``row_ok``.
        sum_jac2_kept: Sum of the squared Jacobian over those same rows.
        resample: The four bootstrap sums of :func:`resample_moment_sums` over
            every row, or ``None`` when no ``idx`` was given.
    """

    Y: Array
    jac_finite: Array
    row_ok: Array
    sum_jac: Array
    sum_jac2: Array
    sum_jac_kept: Array
    sum_jac2_kept: Array
    resample: tuple[Array, Array, Array, Array] | None = None


def jacobian_sums(
    fn: Callable,
    X: Array,
    batch_size: int | None,
    *,
    n_outputs: int | None = None,
    counts: Array | None = None,
) -> _JacobianSums:
    """Sum the Jacobian and its square over every sample row, masked and not.

    Both :func:`moment_sums`, the pure core, and
    :func:`jaxgsa.dgsm.analyze` reduce the same batches of
    :func:`jac_batches` the same way; this is that one shared loop. Every
    batch is reduced twice: once over all of its rows, and once over the rows
    where ``X``, the output and the Jacobian are all finite. Producing both
    totals is mechanical arithmetic over already-computed batches, not a
    policy choice — the choice of which total to read is made by the caller.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Rows per batch, or ``None`` to derive one from the
            active memory budget, as in :func:`jac_batches`.
        n_outputs: ``T*K``, passed straight to :func:`jac_batches` for a
            caller that already knows it.
        counts: Bootstrap draw counts from :func:`draw_counts`, shape
            ``(n_bootstrap, N)``. When given, the bootstrap sums of
            :func:`resample_moment_sums` over every row are accumulated in
            the same sweep.

    Returns:
        A :class:`_JacobianSums` holding both pairs of sums, the stacked
        outputs, the two per-row finite masks and, with ``counts``, the
        bootstrap sums.

    Raises:
        ValueError: If ``X`` has no sample rows.
    """
    sum_jac: Array | None = None
    sum_jac2: Array | None = None
    sum_jac_kept: Array | None = None
    sum_jac2_kept: Array | None = None
    Y_parts: list[Array] = []
    jac_finite_parts: list[Array] = []
    row_ok_parts: list[Array] = []
    resample: list[Array] | None = None
    offset = 0
    shift: Array | None = None

    for jac, Y_chunk, X_chunk in jac_batches(fn, X, batch_size, n_outputs=n_outputs):
        Y_parts.append(Y_chunk)
        if counts is not None:
            if shift is None:
                shift = _output_shift(Y_chunk)
            parts = _resample_batch_sums(jac, Y_chunk, counts, offset, shift)
            resample = (
                parts
                if resample is None
                else [a + b for a, b in zip(resample, parts, strict=True)]
            )
            offset += int(jac.shape[0])
        # The mask is built and applied here, before the sums below fold this
        # batch into the running totals. After that point the batch is gone.
        chunk_jac_finite = _row_finite(jac)
        chunk_row_ok = chunk_jac_finite & _row_finite(Y_chunk) & _row_finite(X_chunk)
        jac_finite_parts.append(chunk_jac_finite)
        row_ok_parts.append(chunk_row_ok)
        jac_kept = _mask_rows(jac, chunk_row_ok)

        sj = jnp.sum(jac, axis=0)
        sj2 = jnp.sum(jac**2, axis=0)
        sjk = jnp.sum(jac_kept, axis=0)
        sjk2 = jnp.sum(jac_kept**2, axis=0)
        sum_jac = sj if sum_jac is None else sum_jac + sj
        sum_jac2 = sj2 if sum_jac2 is None else sum_jac2 + sj2
        sum_jac_kept = sjk if sum_jac_kept is None else sum_jac_kept + sjk
        sum_jac2_kept = sjk2 if sum_jac2_kept is None else sum_jac2_kept + sjk2

    if sum_jac is None or sum_jac2 is None or sum_jac_kept is None or sum_jac2_kept is None:
        raise ValueError("X has no sample rows, so there is no derivative to average")

    return _JacobianSums(
        Y=_join(Y_parts),
        jac_finite=_join(jac_finite_parts),
        row_ok=_join(row_ok_parts),
        sum_jac=sum_jac,
        sum_jac2=sum_jac2,
        sum_jac_kept=sum_jac_kept,
        sum_jac2_kept=sum_jac2_kept,
        resample=None
        if resample is None
        else (resample[0], resample[1], resample[2], resample[3]),
    )


def moment_sums(
    fn: Callable,
    X: Array,
    batch_size: int | None,
    *,
    n_outputs: int | None = None,
) -> tuple[Array, Array, Array]:
    """Sum the Jacobian and its square over every sample row.

    No masking read: this is the pure-core reduction, and it reports what the
    model returned. It shares its loop with :func:`jacobian_sums`, which also
    computes the masked totals :func:`jaxgsa.dgsm.analyze` needs, and just
    reads off the unmasked pair.

    Args:
        fn: One-sample model, ``(D,) -> ()`` / ``(K,)`` / ``(T, K)``.
        X: Sample matrix, shape ``(N, D)``.
        batch_size: Rows per batch, or ``None`` to derive one from the
            active memory budget, as in :func:`jac_batches`.
        n_outputs: ``T*K``, passed straight to :func:`jac_batches` for a
            caller that already knows it.

    Returns:
        ``(sum_jac, sum_jac2, Y)``. The two sums carry ``fn``'s own slice axes
        with a trailing ``(D,)``; ``Y`` is the stacked forward output.
    """
    sums = jacobian_sums(fn, X, batch_size, n_outputs=n_outputs)
    return sums.sum_jac, sums.sum_jac2, sums.Y


def resample_moment_sums(
    batches: Iterator[tuple[Array, Array]],
    counts: Array,
) -> tuple[Array, Array, Array, Array]:
    """Accumulate per-replicate sums of the Jacobian and the output.

    A bootstrap replicate is a multiset of sample rows, so every quantity DGSM
    reports is a weighted sum over rows with integer weights. Counting the
    draws that land in the current batch turns the resample into one matrix
    product per batch, which means the whole bootstrap costs **one** extra
    sweep over the sample rather than one sweep per replicate, and the
    Jacobian still never exists whole.

    The draw-count matrix is ``(n_bootstrap, N)`` int32, built once by
    :func:`draw_counts` and sliced per batch. The index draws behind it are
    freed as soon as it exists. A large ``n_bootstrap`` times a large ``N`` can run the device
    out of memory.

    Args:
        batches: Yields ``(jac, Y)`` per batch, in row order and with no gaps.
            The rows are numbered by the order they arrive in, so the caller
            must have compacted any dropped rows out already.
        counts: Draw counts from :func:`draw_counts`, shape
            ``(n_bootstrap, n)``, over that same row numbering.

    Returns:
        ``(sum_jac, sum_jac2, sum_y, sum_y2)``, each with a leading
        ``n_bootstrap`` axis followed by the per-row shape.
    """
    acc: list[Array] | None = None
    offset = 0
    shift: Array | None = None
    for jac, Y_chunk in batches:
        if shift is None:
            shift = _output_shift(Y_chunk)
        parts = _resample_batch_sums(jac, Y_chunk, counts, offset, shift)
        acc = parts if acc is None else [a + b for a, b in zip(acc, parts, strict=True)]
        offset += int(jac.shape[0])
    if acc is None:
        raise ValueError("X has no sample rows, so there is nothing to resample")
    return acc[0], acc[1], acc[2], acc[3]


def draw_counts(key: Array, n_bootstrap: int, n_rows: int) -> Array:
    """Draw bootstrap rows with replacement and count them per replicate.

    The index draws are local to this function, so only the count matrix
    stays alive afterwards. That is one ``(n_bootstrap, n_rows)`` int32 array
    less than keeping the indices next to the counts.

    Args:
        key: PRNG key for the row draws.
        n_bootstrap: Number of replicates.
        n_rows: Number of rows to draw from, which is also the resample size.

    Returns:
        Integer array of shape ``(n_bootstrap, n_rows)``: how often each
        replicate drew each row.
    """
    return _draw_counts(key, n_bootstrap, n_rows)


@partial(jax.jit, static_argnames=("n_bootstrap", "n_rows"))
def _draw_counts(key: Array, n_bootstrap: int, n_rows: int) -> Array:
    """Count each replicate's draws one replicate at a time, with ``lax.map``.

    Replicate ``r`` uses its own key, so no draw depends on the batch width,
    and the random-bit transient is one row, not the whole matrix.
    """

    def one(k: Array) -> Array:
        return jnp.bincount(draw_rows(k, n_rows), length=n_rows).astype(jnp.int32)

    return jax.lax.map(one, resample_keys(key, n_bootstrap))


def _output_shift(Y_chunk: Array) -> Array:
    """Return one constant per output slice, to centre the output sums on.

    The variance is shift-invariant, but ``sum(y**2)/n - mean(y)**2`` loses
    every digit to cancellation when ``|mean(y)|`` dwarfs the spread. Shifting
    by the mean of the first batch before the sums removes that offset.

    Args:
        Y_chunk: Output rows of the first batch.

    Returns:
        Array of shape ``(T*K,)`` in the dtype of ``Y_chunk``.
    """
    return jnp.mean(Y_chunk.reshape(Y_chunk.shape[0], -1), axis=0)


@jax.jit
def _weighted_sums(
    weights: Array, jac: Array, Y_chunk: Array, shift: Array
) -> tuple[Array, Array, Array, Array]:
    """Weighted sums of one batch for every bootstrap replicate.

    Args:
        weights: Draw counts, shape ``(n_bootstrap, batch)``.
        jac: Jacobian rows of this batch.
        Y_chunk: Output rows of this batch.
        shift: Per-slice constant from :func:`_output_shift`, subtracted from
            the outputs before they are summed.

    Returns:
        ``(sum_jac, sum_jac2, sum_y, sum_y2)`` with a leading ``n_bootstrap``
        axis. The two output sums are of the shifted outputs.
    """
    n_replicates, n = weights.shape
    w = weights.astype(jac.dtype)
    jac_flat = jac.reshape(n, -1)
    y_flat = Y_chunk.reshape(n, -1) - shift
    return (
        (w @ jac_flat).reshape((n_replicates, *jac.shape[1:])),
        (w @ jac_flat**2).reshape((n_replicates, *jac.shape[1:])),
        (w @ y_flat).reshape((n_replicates, *Y_chunk.shape[1:])),
        (w @ y_flat**2).reshape((n_replicates, *Y_chunk.shape[1:])),
    )


def _resample_batch_sums(
    jac: Array, Y_chunk: Array, counts: Array, offset: int, shift: Array
) -> list[Array]:
    """Weighted sums of one batch for every bootstrap replicate.

    The output sums are of ``Y - shift``. That leaves ``Var(Y)`` as it was and
    keeps the raw-moment form stable when ``Y`` has a large offset.

    Args:
        jac: Jacobian rows of this batch.
        Y_chunk: Output rows of this batch.
        counts: Draw counts of every row, shape ``(n_bootstrap, N)``, from
            :func:`draw_counts`.
        offset: Number of rows that came before this batch.
        shift: Per-slice constant, from :func:`_output_shift` on the first
            batch. The same value must be used for every batch.

    Returns:
        ``[sum_jac, sum_jac2, sum_y, sum_y2]`` for this batch, each with a
        leading ``n_bootstrap`` axis.
    """
    n = int(jac.shape[0])
    return list(_weighted_sums(counts[:, offset : offset + n], jac, Y_chunk, shift))


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
    mu: Array,
    variance: Array,
    standardize_outputs: bool,
    zero_variance: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Turn the derivative moments into the two Sobol-index bounds.

    This is the whole numerical statement of the method, and both entry points
    run this one copy of it:

    - upper (Poincare / Sobol-Kucherenko): ``ST_i <= C_i * nu_i / Var(Y)``,
      which holds for every marginal this package supports.
    - lower: ``Var(x_i) * mu_i^2 / Var(Y)``. Kucherenko & Song (2016),
      Theorem 6 (Section 4.1, eq. 31), prove ``ST_i >=`` this for a Gaussian
      marginal only. On a uniform or truncated marginal it is an estimate.
      It is exact for a linear response and can overshoot ``ST_i`` for a
      curved one. See
      :class:`jaxgsa.dgsm.DGSMResult` for why the condition is on the
      marginal.

    A zero-variance slice makes both bounds ``0 / 0``, so it is divided by a
    safe 1 and the bounds are then set to ``NaN``, which keeps gradients of
    the other slices finite. That is arithmetic, not policy:
    the warning about it stays in ``analyze``. A constant float slice rarely
    has a bit-exact zero variance (its mean rounds), so callers pass the
    warning's own test as ``zero_variance``; without it a constant slice
    would report bounds of ``0 / round-off = 0``.

    Args:
        problem: Problem definition, read for the per-axis Poincare constants
            and the marginal variances. Not an array, so reading it is a host
            operation on static data and stays traceable.
        nu: ``E[(df/dx_i)^2]``, shape ``(T, K, D)``.
        mu: ``E[df/dx_i]``, shape ``(T, K, D)``.
        variance: Output variance per slice, shape ``(T, K)``.
        standardize_outputs: Whether to report the moments for the
            standardized output ``(Y - mean) / std``. The map is affine, so
            the standardized moments are the ones already computed, rescaled.
        zero_variance: Boolean mask broadcastable to ``variance``, ``True``
            for a slice the zero-variance warning flags (see
            :func:`jaxgsa._core.validation._zero_variance_mask`). ``None``
            flags only a variance of exactly 0.

    Returns:
        ``(nu, mu, upper, lower, variance)``, the first four shaped
        ``(T, K, D)`` and the last ``(T, K)``.
    """
    zero = variance == 0
    if zero_variance is not None:
        zero = zero | zero_variance
    if standardize_outputs:
        # Y -> (Y - mean)/std is affine, so d/dx of the shift is zero and the
        # 1/std factor is a constant. Rescaling here rather than
        # differentiating a wrapped function leaves the autodiff sweep alone.
        # Both bounds are ratios of quantities that scale the same way, so
        # they come out unchanged, which the tests assert.
        # sqrt'(0) is infinite, so a zero variance must not reach the sqrt:
        # its zero cotangent would become 0 * inf = NaN under grad.
        safe_scale = jnp.sqrt(jnp.where(zero, jnp.ones_like(variance), variance))  # (T, K)
        mu = mu / safe_scale[..., None]
        nu = nu / (safe_scale**2)[..., None]
        variance = variance / safe_scale**2

    # Per-axis constants: C for the Poincare upper bound, Var for the lower.
    C, Var = axis_constants(problem)
    # NumPy float64 constants; cast so float32 data keeps float32 bounds.
    C_jnp = jnp.asarray(C, dtype=nu.dtype)
    Var_jnp = jnp.asarray(Var, dtype=nu.dtype)

    # A constant output slice has zero variance; map it to NaN bounds rather
    # than dividing by zero. Double where: divide by a safe 1 on a constant
    # slice, then put the NaN back. A NaN denominator would send 0 / NaN =
    # NaN cotangents into nu and mu, and through them into every slice's
    # gradient with respect to X.
    zero_b = zero[..., None]  # (T, K, 1)
    safe_var = jnp.where(zero_b, jnp.ones_like(variance)[..., None], variance[..., None])

    # The (D,) constants broadcast against (T, K, D) on the trailing axis.
    upper = jnp.where(zero_b, jnp.nan, C_jnp * nu / safe_var)
    lower = jnp.where(zero_b, jnp.nan, Var_jnp * mu**2 / safe_var)
    return nu, mu, upper, lower, variance
