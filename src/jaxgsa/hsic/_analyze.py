"""HSIC index estimators for kernel-based sensitivity analysis.

The estimators compute R2-HSIC (normalized first-order) and Total HSIC
indices from arbitrary (X, Y) sample pairs. They use Gaussian RBF kernels
and pick the bandwidth with the median heuristic.

Array shape conventions used throughout:
    N  — number of samples
    D  — number of parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

Total HSIC uses augmented kernels k*(x,x') = 1 + k_c(x,x') per Larsen &
Alexanderian (2026), where k_c is the centered kernel. The product of
augmented kernels captures all interaction orders, not just the highest.
This gives correct total indices for additive models.

No bootstrap. HSIC is the only index method here that offers no
``n_bootstrap``, and the omission is deliberate rather than unfinished work.
The uncertainty statement HSIC already makes is the permutation ``p_value``:
it resamples the pairing between ``x_i`` and ``Y`` under the null of
independence and reports where the measured index falls in that null. A row
bootstrap would answer the same question about the same quantity, and worse.
The index is a V-statistic, a double sum over all N^2 sample pairs, so
resampling rows with replacement duplicates rows and the duplicates land on
the diagonal of the kernel matrix, where the kernel is exactly 1. The
resampled statistic is therefore biased upward by construction, and its
spread mixes that bias with the sampling variability it was meant to
measure. So HSIC reports a p-value and no interval, and ``key`` is still
required — the permutation test draws randomness even though nothing here
bootstraps.

References:
    Gretton et al. (2005). JMLR 6:2075-2129.
    Da Veiga (2015). Rel. Eng. Sys. Safety 142:346-362.
    Larsen & Alexanderian (2026). arXiv:2603.00849.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from jaxgsa._core.entry import at_least, prepare, require
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import (
    _prenormalize_outputs,
    _prepare_Y,
)
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hsic._result import HSICResult
from jaxgsa.problem import Problem

_MIN_SAMPLES = 4

_UINT_BY_ITEMSIZE: dict[int, type] = {2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}
"""Unsigned integer type of the same width as a float type, by byte count."""


def _linear_quantile_plan(q: float, n_elements: int) -> tuple[int, int, Array]:
    """Reproduce the index and weight arithmetic of ``jnp.quantile``.

    ``jnp.quantile(a, q)`` with the default ``method="linear"`` reads two
    order statistics of the sorted array and blends them. The blend is done
    in the dtype a bare Python float gets, and the position is computed in
    that dtype too, so the rounding of ``q * (n - 1)`` is part of the answer.
    This helper repeats exactly that arithmetic on the host, which is possible
    because both ``q`` and the element count are static.

    Args:
        q: Quantile in [0, 1], as a Python float.
        n_elements: Number of elements the quantile is taken over.

    Returns:
        ``(low_rank, high_rank, high_weight)``. The ranks are 0-based
        positions into the ascending sort, and the weight is the share of the
        higher of the two, a scalar of the default float dtype.
    """
    dtype = jnp.result_type(float)
    pos = np.asarray(q, dtype=dtype) * np.asarray(n_elements - 1, dtype=dtype)
    low = np.floor(pos)
    high = np.ceil(pos)
    high_weight = np.asarray(pos - low, dtype=dtype)
    return int(low), int(high), jnp.asarray(high_weight)


def _from_order_key(key: Array, sign_bit: Array, dtype: DTypeLike) -> Array:
    """Undo the total-ordering bit transform and read the float back.

    Args:
        key: Order key produced by the transform in
            :func:`_linear_quantile_by_selection`.
        sign_bit: The sign bit of the unsigned type the key lives in.
        dtype: Float dtype to bitcast back to.

    Returns:
        The float whose order key is ``key``.
    """
    raw = jnp.where(key & sign_bit != 0, key ^ sign_bit, ~key)
    return jax.lax.bitcast_convert_type(raw, dtype)


def _linear_quantile_by_selection(values: Array, q: float) -> Array:
    """Take a linear-interpolated quantile without sorting the input.

    ``jnp.quantile`` sorts. On the ``(N, N)`` matrix of pairwise squared
    distances that is an ``N^2 log N^2`` sort of a million elements for every
    output slice, and it measured as 92% of the whole HSIC analysis. Only two
    adjacent order statistics are ever read, so a *selection* answers the same
    question, and a selection needs no sort.

    The selection bisects on the bit pattern of the value rather than on the
    value itself. The raw unsigned integer sharing the storage is monotone in
    the float only for non-negative values -- a negative float has its sign
    bit set, so it reads as a *large* unsigned integer, and ``-inf`` would
    come back as the maximum instead of the minimum. The standard total
    ordering fixes that: flip every bit of a negative, set the sign bit of a
    non-negative. The result is monotone across the whole float line, so
    bisecting the integer range is bisecting the value range, and a fixed
    number of steps (one per bit) lands on an exact element of the input.
    Each step is one compare-and-count pass over the data, which is
    bandwidth-bound and parallel, where a sort is neither.

    The result is bit-for-bit what ``jnp.quantile`` returns: the same two
    order statistics, blended with the same weights, in the same dtype. NaN
    is handled the same way too — ``jnp.quantile`` poisons the whole array if
    any element is NaN, so a NaN anywhere gives NaN here as well.

    Args:
        values: Values to take the quantile over. Any shape; it is treated
            as flat. Negatives and infinities are ordered correctly.
        q: Quantile in [0, 1], as a Python float.

    Returns:
        Scalar quantile, in the dtype ``jnp.quantile`` would return.

    Raises:
        TypeError: If the float width of ``values`` has no unsigned integer
            counterpart to bisect over.
    """
    flat = values.ravel()
    itemsize = jnp.dtype(flat.dtype).itemsize
    if itemsize not in _UINT_BY_ITEMSIZE:
        raise TypeError(f"cannot select order statistics of dtype {flat.dtype}")
    uint = _UINT_BY_ITEMSIZE[itemsize]
    sign_bit = jnp.asarray(1, uint) << jnp.asarray(itemsize * 8 - 1, uint)
    raw = jax.lax.bitcast_convert_type(flat, uint)
    bits = jnp.where(raw & sign_bit != 0, ~raw, raw | sign_bit)

    low_rank, high_rank, high_weight = _linear_quantile_plan(q, flat.shape[0])

    def _step(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        """Halve the candidate bit-pattern interval once."""
        lo, hi = bounds
        mid = lo + (hi - lo) // 2
        enough = jnp.sum(bits <= mid) >= low_rank + 1
        return jnp.where(enough, lo, mid + 1), jnp.where(enough, mid, hi)

    lo_bits, _ = jax.lax.fori_loop(0, itemsize * 8, _step, (jnp.zeros((), uint), jnp.max(bits)))
    low_value = _from_order_key(lo_bits, sign_bit, flat.dtype)

    # The high rank is the low rank plus one, so its value is either the same
    # element again (when the low value is repeated) or the next distinct
    # value up. Both are one masked reduction away, which is cheaper than a
    # second bisection.
    n_at_or_below = jnp.sum(bits <= lo_bits)
    all_ones = ~jnp.zeros((), uint)
    above = jnp.where(bits > lo_bits, bits, all_ones)
    next_value = _from_order_key(jnp.min(above), sign_bit, flat.dtype)
    high_value = jnp.where(n_at_or_below >= high_rank + 1, low_value, next_value)

    # Blend the pair through ``jnp.quantile`` itself rather than by writing
    # the weighted sum out here. On a two-element array the same function
    # reduces to exactly this blend -- the sort is a no-op, the position is
    # ``high_weight``, and the two weights come back unchanged -- so the last
    # bit is guaranteed to agree with the sorting implementation even when the
    # sum lands on a tie that a fused multiply-add rounds one way and a
    # separate multiply and add round the other.
    blended = jnp.quantile(jnp.stack([low_value, high_value]), high_weight)
    return jnp.where(jnp.any(jnp.isnan(flat)), jnp.nan, blended)


def _median_bandwidth_sq(x: Array) -> Array:
    """Compute the median-heuristic squared bandwidth for one variable.

    The median heuristic takes the median of the ``M = (N^2 - N) / 2``
    off-diagonal pairwise squared distances. The N diagonal zeros are excluded
    because they carry no information about the spread of the sample.

    Building the strict upper triangle to do that costs two index arrays and a
    gathered copy. It is not needed: the diagonal zeros are the *smallest*
    entries of the full ``(N, N)`` matrix, so they only shift the target
    position in the sorted array, and a quantile of the full matrix reaches the
    same value.

    Derivation. Sort all ``N^2`` entries. The N zeros occupy positions
    ``0 .. N-1``. Each off-diagonal value appears twice, so the j-th smallest
    upper-triangle value (0-based) occupies positions ``N + 2j`` and
    ``N + 2j + 1``. Write ``S = N + M = (N^2 + N) / 2``.

    - M odd: the upper-triangle median is element ``j = (M - 1) / 2``, which
      occupies positions ``S - 1`` and ``S``. Any position in ``[S-1, S]``
      selects it.
    - M even: the median averages elements ``j = M/2 - 1`` and ``j = M/2``,
      which end at position ``S - 1`` and start at position ``S``. Linear
      interpolation at position ``S - 0.5`` returns exactly that average.

    Position ``S - 0.5`` therefore serves both parities, and a linear-
    interpolated quantile puts the target at ``q * (N^2 - 1)``. Solving for q
    and substituting S gives the expression below. It reproduces
    ``jnp.median`` of the strict upper triangle exactly at every N tested from
    4 to 1024.

    The quantile is taken by selection rather than by sorting — see
    :func:`_linear_quantile_by_selection`, which returns exactly what
    ``jnp.quantile`` would return and is roughly 40x faster here.

    Args:
        x: Values for one variable, shape ``(N,)``.

    Returns:
        Scalar median of the off-diagonal squared distances, floored at 1e-20.
    """
    N = x.shape[0]
    dists_sq = (x[:, None] - x[None, :]) ** 2
    q = (N**2 + N - 1) / (2 * (N**2 - 1))
    return jnp.maximum(_linear_quantile_by_selection(dists_sq, q), 1e-20)


def _resolve_bandwidth_sq(x: Array, bandwidth: float | None) -> Array:
    """Resolve the squared Gaussian bandwidth for one variable.

    Args:
        x: Values for one variable, shape ``(N,)``.
        bandwidth: Fixed bandwidth, or None for the median heuristic.

    Returns:
        Scalar squared bandwidth.
    """
    if bandwidth is None:
        return _median_bandwidth_sq(x)
    return jnp.asarray(bandwidth, dtype=x.dtype) ** 2


def _build_kernel(x: Array, sigma_sq: Array, batch_size: int | None) -> Array:
    """Build a Gaussian RBF kernel matrix from a resolved squared bandwidth.

    The matrix is built in row blocks when ``batch_size`` asks for it. Blocking
    bounds the working memory of the build, not the size of the result: the
    blocks are concatenated back into one ``(N, N)`` matrix either way.

    Args:
        x: Values for one variable, shape ``(N,)``.
        sigma_sq: Scalar squared bandwidth.
        batch_size: Number of rows per block, or None to build in one step.

    Returns:
        Kernel matrix, shape ``(N, N)``.
    """
    N = x.shape[0]
    if batch_size is None or batch_size <= 0 or N <= batch_size:
        return jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * sigma_sq))

    rows = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        rows.append(jnp.exp(-((x[start:end, None] - x[None, :]) ** 2) / (2.0 * sigma_sq)))
    return jnp.concatenate(rows, axis=0)


def _center_kernel(K: Array) -> Array:
    """Center a kernel matrix: K_c = HKH where H = I - (1/n)11^T.

    Args:
        K: Kernel matrix, shape ``(N, N)``.

    Returns:
        Centered kernel matrix, shape ``(N, N)``.
    """
    row_mean = jnp.mean(K, axis=1, keepdims=True)
    col_mean = jnp.mean(K, axis=0, keepdims=True)
    grand_mean = jnp.mean(K)
    return K - row_mean - col_mean + grand_mean


def _hsic_v(K: Array, L: Array) -> Array:
    """Compute the biased V-statistic HSIC estimate for two kernels.

    The trace formula below avoids forming explicit centering matrices:
        HSIC = U/n^2 - 2V/n^3 + W/n^4
    where U = sum(K*L), V = sum(colsums(K)*colsums(L)), W = sum(K)*sum(L).

    Args:
        K: Kernel matrix, shape ``(N, N)``.
        L: Kernel matrix, shape ``(N, N)``.

    Returns:
        Scalar HSIC value.
    """
    n = K.shape[0]
    n_f = jnp.asarray(n, dtype=K.dtype)
    U = jnp.sum(K * L)
    col_K = jnp.sum(K, axis=0)
    col_L = jnp.sum(L, axis=0)
    V = jnp.dot(col_K, col_L)
    W = jnp.sum(K) * jnp.sum(L)
    return U / n_f**2 - 2.0 * V / n_f**3 + W / n_f**4


def _build_one_kernel(
    x: Array,
    bandwidth: float | None,
    batch_size: int | None,
) -> Array:
    """Build one kernel matrix for one variable.

    How the bandwidth is chosen and how the matrix is built are independent
    questions, so the bandwidth is resolved first and the build follows.

    Args:
        x: Values for one variable, shape ``(N,)``.
        bandwidth: Fixed bandwidth, or None for the median heuristic.
        batch_size: Row-block size for the kernel matrix, or None.

    Returns:
        Kernel matrix, shape ``(N, N)``.
    """
    return _build_kernel(x, _resolve_bandwidth_sq(x, bandwidth), batch_size)


def _build_input_kernels(
    X_unit: Array,
    bandwidth: float | None,
    batch_size: int | None,
) -> list[Array]:
    """Build one kernel matrix per parameter.

    Args:
        X_unit: Inputs mapped to [0, 1], shape ``(N, D)``.
        bandwidth: Fixed bandwidth, or None for the median heuristic.
        batch_size: Row-block size for the kernel matrix, or None.

    Returns:
        List of D kernel matrices, each of shape ``(N, N)``.
    """
    D = X_unit.shape[1]
    return [_build_one_kernel(X_unit[:, d], bandwidth, batch_size) for d in range(D)]


def _augmented_kernels(Ks: list[Array]) -> list[Array]:
    """Build augmented kernels: K*_d = 1 + center(K_d).

    The augmented kernel carries a constant term. The product of augmented
    kernels therefore captures all interaction orders, not just the highest.
    Correct total HSIC indices need this.

    Args:
        Ks: List of D raw kernel matrices, each of shape ``(N, N)``.

    Returns:
        List of D augmented kernel matrices, each of shape ``(N, N)``.
    """
    return [jnp.ones_like(K) + _center_kernel(K) for K in Ks]


def _complement_kernels(Ks: list[Array]) -> tuple[Array, list[Array]]:
    """Build the full product kernel and every complement product kernel.

    A prefix-suffix pass gives all D complements in linear time.

    Args:
        Ks: List of D kernel matrices, each of shape ``(N, N)``.

    Returns:
        ``(K_full, complements)``. ``K_full`` is the Hadamard product of all
        Ks, shape ``(N, N)``. ``complements[d]`` is the product of all Ks
        except d, each of shape ``(N, N)``.
    """
    D = len(Ks)
    if D == 1:
        return Ks[0], [jnp.ones_like(Ks[0])]

    prefix = [jnp.ones_like(Ks[0])] * D
    for i in range(1, D):
        prefix[i] = prefix[i - 1] * Ks[i - 1]

    suffix = [jnp.ones_like(Ks[0])] * D
    for i in range(D - 2, -1, -1):
        suffix[i] = suffix[i + 1] * Ks[i + 1]

    K_full = prefix[-1] * Ks[-1]
    compls = [prefix[d] * suffix[d] for d in range(D)]
    return K_full, compls


@lru_cache(maxsize=32)
def _get_hsic_kernel(n_perms: int, bandwidth: float | None, batch_size: int | None):
    """Return a JIT-compiled HSIC kernel for every output slice.

    The atomic unit is one output slice: it builds that slice's output
    kernel, forms the indices against the shared input kernels, and runs the
    permutation test. The returned callable maps that unit over all the
    slices in one compiled call, so the whole analysis costs one dispatch
    instead of one per slice, and no result travels through the host in
    between.

    Everything that depends only on ``X`` -- the input kernels, their
    augmented products and complements, and the self-HSIC values -- is a
    plain argument here. The caller builds it once and hands the same arrays
    to the one call.

    The map is :func:`jax.lax.map`, which runs the atomic kernel one slice at
    a time. So one slice's working set is live however many slices the output
    has, and slice i of a hundred returns bit-for-bit what a scalar analysis
    of that slice returns.

    The chunked ``vmap`` the rest of the library uses was measured here and
    rejected. ``jax.lax.map(..., batch_size=c)`` and ``jax.vmap`` over a chunk
    of c slices ran at the same speed as this scan on a CPU backend -- the
    per-slice work is a chain of ``(N, N)`` reductions that already saturates
    the machine and gains nothing from a batch axis -- while letting XLA
    reassociate those reductions across the chunk. That moved indices by up
    to 1e-3 relative, because the V-statistic is a difference of three large
    terms and float32 cancellation magnifies any change of summation order.
    A chunk that is no faster and changes the answer is not worth a keyword,
    so there is none. On a device that does reward a batch axis the trade
    would have to be measured again.

    Args:
        n_perms: Number of permutations for the permutation test (static;
            it sets the scan length and the number of split keys).
        bandwidth: Fixed Gaussian bandwidth for the output kernel, or None
            for the median heuristic (static; it selects the code path).
        batch_size: Row-block size for the output kernel build, or None
            (static; it sets the number of blocks).

    Returns:
        A jitted callable that returns ``(R2_HSIC, T_HSIC, p_values,
        hsic_raw)``, each of shape ``(total, D)``. It takes the stacked input
        kernels, the precomputed self-HSIC values, the ``(N, total)`` output
        columns, and one PRNG key per column.
    """

    def _one_slice(
        Ks_stack: Array,
        K_aug_compls_stack: Array,
        K_aug_full: Array,
        hsic_xxs: Array,
        y_col: Array,
        key: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Compute HSIC indices for one output slice.

        Args:
            Ks_stack: Stacked raw input kernels ``(D, N, N)``.
            K_aug_compls_stack: Stacked complement augmented kernels
                ``(D, N, N)``.
            K_aug_full: Product of all augmented input kernels ``(N, N)``.
            hsic_xxs: Precomputed self-HSIC ``HSIC(K_d, K_d)`` of shape
                ``(D,)`` (output-independent, so computed once).
            y_col: Single output column, shape ``(N,)``.
            key: PRNG key for the permutation test.

        Returns:
            ``(R2_HSIC, T_HSIC, p_values, hsic_raw)`` each of shape ``(D,)``.
        """
        # The barrier stops XLA fusing the kernel build into the reductions
        # that read it. Fused, the exponential is recomputed inside each
        # reduction loop and the sums accumulate in a different order, which
        # moves the last bit of the indices away from what the same kernel
        # returns when it is handed a materialized matrix. HSIC reads L a few
        # dozen times per slice, so materializing it once is the cheaper
        # arrangement anyway.
        L = jax.lax.optimization_barrier(_build_one_kernel(y_col, bandwidth, batch_size))
        N = L.shape[0]
        eps = jnp.finfo(L.dtype).eps

        hsic_yy = _hsic_v(L, L)
        zero_var = hsic_yy < eps

        hsic_xys = jax.vmap(lambda K: _hsic_v(K, L))(Ks_stack)
        denoms = jnp.sqrt(jnp.maximum(hsic_xxs * hsic_yy, eps**2))
        r2 = jnp.where(zero_var, jnp.nan, hsic_xys / denoms)

        hsic_full = _hsic_v(K_aug_full, L)
        full_thresh = jnp.maximum(jnp.abs(hsic_full) * eps * 100, eps)
        hsic_full_safe = jnp.where(jnp.abs(hsic_full) < full_thresh, jnp.nan, hsic_full)
        hsic_compls = jax.vmap(lambda K: _hsic_v(K, L))(K_aug_compls_stack)
        t_hsic = jnp.where(zero_var, jnp.nan, 1.0 - hsic_compls / hsic_full_safe)

        perm_keys = jax.random.split(key, n_perms)

        def _scan_body(null_counts: Array, pkey: Array) -> tuple[Array, None]:
            """Add one permutation draw's exceedance counts to the running total."""
            perm = jax.random.permutation(pkey, N)
            L_perm = L[perm][:, perm]
            perm_hsics = jax.vmap(lambda K: _hsic_v(K, L_perm))(Ks_stack)
            return (
                null_counts + (perm_hsics >= hsic_xys).astype(null_counts.dtype),
                None,
            )

        null_counts, _ = jax.lax.scan(_scan_body, jnp.zeros_like(hsic_xys), perm_keys)
        p_vals = (null_counts + 1.0) / (n_perms + 1.0)

        return r2, t_hsic, p_vals, hsic_xys

    def _impl(
        Ks_stack: Array,
        K_aug_compls_stack: Array,
        K_aug_full: Array,
        hsic_xxs: Array,
        Y_cols: Array,
        keys: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Map the single-slice kernel over the output columns.

        Args:
            Ks_stack: Stacked raw input kernels ``(D, N, N)``.
            K_aug_compls_stack: Stacked complement augmented kernels
                ``(D, N, N)``.
            K_aug_full: Product of all augmented input kernels ``(N, N)``.
            hsic_xxs: Precomputed self-HSIC ``HSIC(K_d, K_d)`` ``(D,)``.
            Y_cols: Output columns, shape ``(N, total)``.
            keys: One PRNG key per column, shape ``(total,)``.

        Returns:
            ``(R2_HSIC, T_HSIC, p_values, hsic_raw)`` each ``(total, D)``.
        """

        def _step(column: tuple[Array, Array]) -> tuple[Array, Array, Array, Array]:
            """Run the atomic kernel on one (column, key) pair."""
            y_col, key = column
            return _one_slice(Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, y_col, key)

        return jax.lax.map(_step, (Y_cols.T, keys))

    return jax.jit(_impl)


def _warn_single_precision() -> None:
    """Warn that float32 costs the HSIC V-statistic most of its digits.

    The V-statistic is a difference of three sums over the whole ``(N, N)``
    kernel matrix, and the three are of the same magnitude while the answer is
    small. Cancellation therefore eats the leading digits, and what survives
    is set by the summation order. Reordering the sample rows -- a change that
    cannot alter the true value -- moves a measured index. On Ishigami with
    ``N = 2048`` the shift is ``2e-4`` relative in float32 against ``2e-13``
    in float64, and it grows with ``N`` because the sums do. That is the same
    size as the gap HSIC is often asked to resolve between two weak
    parameters.
    """
    # Read the flag off the config object directly; config.read() raises for
    # flags that were never explicitly set.
    if not getattr(jax.config, "jax_enable_x64", False):
        warnings.warn(
            "jaxgsa.hsic: JAX is in single precision; the HSIC V-statistic cancels three "
            "large sums against each other, so float32 leaves only about three or four "
            "correct digits and the index changes with the order of the sample rows. Small "
            "indices and close rankings are not reliable. Enable float64 with "
            'jax.config.update("jax_enable_x64", True) before the analysis.',
            stacklevel=3,
            category=JaxgsaWarning,
        )


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    key: Array,
    bandwidth: float | None = None,
    batch_size: int | None = None,
    prenormalize: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Compute HSIC indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It builds the same
    kernels, runs the same permutation test and returns the same numbers, but
    it does nothing else: no non-finite check, no row dropping, no
    zero-variance warning, no single-precision warning, no sample-count
    check, no :class:`jaxgsa.hsic.HSICResult`, and no read of any array value
    on the host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad``
    and ``jax.jacrev``, which :func:`analyze` cannot, because a policy
    decision needs a concrete value and a tracer has none.

    Every branch here is on a shape or on a Python scalar. ``n_perms``,
    ``bandwidth``, ``batch_size`` and ``prenormalize`` are all static: they
    are read while the graph is being traced and never at run time.
    ``bandwidth`` selects between the median heuristic and a fixed value,
    ``batch_size`` sets how many row blocks the kernel build unrolls into,
    and both are baked into the compiled kernel, which is keyed on them (see
    :func:`_get_hsic_kernel`). Changing one recompiles; neither costs a
    device-side conditional.

    No bootstrap and no confidence interval, here or on :func:`analyze`. The
    permutation ``p_values`` are the uncertainty statement, and the module
    docstring says why a row bootstrap of a V-statistic would be worse than
    redundant.

    Tier T4 (behavioural contract): the returned arrays must equal
    ``R2_HSIC``, ``T_HSIC``, ``p_values`` and ``hsic_raw`` of ``analyze``'s
    result on clean outputs, and the function must survive ``jit``, ``vmap``
    and ``jit(jacrev(...))``. Checked in ``tests/test_hsic.py``.

    One limit on tracing ``X``. A *truncated* Gaussian marginal sends the
    column through ``scipy.stats.truncnorm.cdf``, which reads the values on
    the host, so a problem with one cannot have ``X`` be a tracer. Uniform
    and untruncated Gaussian marginals stay on the device, and ``Y`` is
    always traceable whatever the marginals are.

    Use :func:`analyze` for ordinary analysis. Nothing here checks the
    inputs, so a single NaN silently poisons every index.

    Args:
        problem: Problem definition with D parameters, used only for the
            marginal CDFs that map ``X`` to the unit interval.
        X: Input samples in physical units, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)`` or ``(N, T, K)``.
        n_perms: Number of permutations for the p-value test, as in
            :func:`analyze`.
        key: JAX PRNG key driving the permutation test. Required, and
            keyword-only with no default: the p-values are random, and a
            constant fallback would silently correlate repeated analyses.
        bandwidth: Fixed Gaussian bandwidth, or ``None`` for the median
            heuristic, as in :func:`analyze`. The median heuristic selects
            an order statistic through a bit-pattern bisection, which has no
            derivative, so a gradient taken with respect to ``X`` treats the
            bandwidth as a constant. Pass an explicit ``bandwidth`` when that
            matters.
        batch_size: Row-block size for each ``(N, N)`` kernel build, as in
            :func:`analyze`.
        prenormalize: Standardize each output slice before the analysis, as
            in :func:`analyze`.

    Returns:
        ``(R2_HSIC, T_HSIC, p_values, hsic_raw)``, each of shape
        ``(D,)``, ``(K, D)`` or ``(T, K, D)`` to match the rank of ``Y``,
        exactly as ``analyze`` reports them.

    Raises:
        ValueError: If any parameter is categorical. The Gaussian input
            kernel would read a level code as a distance.
        TypeError: If the float width of ``X`` or ``Y`` has no unsigned
            integer counterpart for the median-heuristic selection.
    """
    D = problem.num_vars
    X_unit = cdf_to_unit_interval(X, problem)

    Y_3d, layout = _prepare_Y(Y)
    _N, T, K = Y_3d.shape

    if prenormalize:
        Y_3d, _, _, _ = _prenormalize_outputs(Y_3d)

    # Build input kernels and augmented products once (independent of Y).
    Ks = _build_input_kernels(X_unit, bandwidth, batch_size)
    Ks_stack = jnp.stack(Ks)
    Ks_aug = _augmented_kernels(Ks)
    K_aug_full, K_aug_compls = _complement_kernels(Ks_aug)
    K_aug_compls_stack = jnp.stack(K_aug_compls)

    # Self-HSIC HSIC(K_d, K_d) is output-independent, so compute it once
    # here and share it across every output slice.
    hsic_xxs = jax.vmap(lambda K: _hsic_v(K, K))(Ks_stack)

    # One column per output slice, in the same (t, k) order the indices are
    # written back in, and one key per column. Each slice reads its own key,
    # so the permutation draws do not depend on how the columns are grouped.
    N = X_unit.shape[0]
    total = T * K
    Y_cols = Y_3d.reshape(N, total)
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(total))

    kernel = _get_hsic_kernel(n_perms, bandwidth, batch_size)
    r2_all, t_all, p_all, raw_all = kernel(
        Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, Y_cols, keys
    )
    return (
        layout.squeeze(r2_all.reshape(T, K, D)),
        layout.squeeze(t_all.reshape(T, K, D)),
        layout.squeeze(p_all.reshape(T, K, D)),
        layout.squeeze(raw_all.reshape(T, K, D)),
    )


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    key: Array | None = None,
    bandwidth: float | None = None,
    batch_size: int | None = None,
    prenormalize: bool = False,
    on_invalid: OnInvalid = "raise",
) -> HSICResult:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) sensitivity indices.

    HSIC uses kernel embeddings to measure the statistical dependence between
    each parameter and the output. It therefore finds nonlinear and
    non-monotonic relationships that correlation-based screening misses. HSIC
    is a given-data method: any (X, Y) sample pair works, and no special
    sampling design is required. The function reports two indices:

    - **R2-HSIC**: ``HSIC(x_i, Y)`` normalized by the geometric mean of the
      self-similarities, ``HSIC(x_i, Y) / sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))``.
      It lies in [0, 1]. A value of 0 means x_i and Y are independent. This is
      the first-order view.
    - **Total HSIC (T_HSIC)**: the fraction of the joint dependence lost when
      x_i is removed. It is analogous to a total-order Sobol index, so it also
      counts influence carried through interactions with other parameters.

    A permutation test supplies p-values for the null hypothesis that x_i and
    Y are independent. HSIC can therefore screen out non-influential
    parameters with a significance level attached.

    **No bootstrap, and no confidence interval.** Almost every other method
    here takes ``n_bootstrap``; HSIC does not, on purpose. The ``p_values``
    are already the uncertainty statement, and they are the right one for
    this estimator: the permutation test breaks the pairing between ``x_i``
    and ``Y`` and reports where the measured index sits in the resulting
    null distribution. A row bootstrap would be redundant against that, and
    it would also be wrong in a way the p-value is not. HSIC is a
    V-statistic, a double sum over all ``N^2`` pairs of rows, so resampling
    rows with replacement repeats rows, the repeats land on the kernel
    diagonal where the kernel equals 1, and the resampled index is biased
    upward by construction. The interval would then mix that bias with the
    sampling spread it was meant to show. ``key`` is still required despite
    the absence of a bootstrap, because the permutation test draws
    randomness of its own.

    :func:`jaxgsa.hsic.indices` is the transformable core: the same numbers
    as bare arrays, with none of the checks, so it composes with ``jit``,
    ``vmap`` and ``jacrev``.

    Correlated parameters are supported. HSIC is a dependence measure and
    assumes no input independence, so a declared ``problem.correlation`` does
    not invalidate the indices. Each index then measures the parameter's total
    association with the output, which includes influence carried through its
    correlated partners. A parameter that the model ignores can therefore
    score above 0 when it correlates with an influential parameter. That
    reading is correct, not an estimation error.

    Args:
        problem: Problem definition with D parameters.
        X: Input samples in physical units, shape ``(N, D)``.
        Y: Model outputs, shape ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
            For outputs of large magnitude, set ``prenormalize=True`` to
            avoid float overflow in the distance computation.
        n_perms: Number of random permutations for the p-value test. More
            permutations give finer p-value resolution at linearly higher
            cost. The smallest attainable p-value is ``1 / (n_perms + 1)``,
            so the default of 200 resolves down to p ~ 0.005.
        key: JAX PRNG key that drives the permutation test. Required: the
            p-values are random, so there is no sensible default. Write
            ``jax.random.key(0)`` if you only want reproducibility. A key
            rather than an integer seed because a key can be split, so
            nested or repeated analyses draw independent permutations
            instead of silently correlated ones.
        bandwidth: Fixed Gaussian-kernel bandwidth applied to all parameters
            and to the output. None (default) selects it per variable with
            the median heuristic (median pairwise distance), a robust default.
        batch_size: Row-block size for building each ``(N, N)`` kernel matrix.
            It bounds the working memory of the build, **not** the kernel
            matrix: the blocks are concatenated back into one ``(N, N)``
            array, so peak memory stays of order ``N^2`` in every case. None
            (default) builds each matrix in one step.
        prenormalize: If True, standardize each output slice to mean 0 and
            unit standard deviation before the analysis.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. ``X`` and ``Y`` are
            checked together, so a bad input takes its own output with it.
            See :mod:`jaxgsa._core.invalid`.

    Returns:
        An :class:`HSICResult` with ``R2_HSIC``, ``T_HSIC``, ``p_values``, and
        ``hsic_raw``, each shaped ``(D,)``, ``(K, D)``, or ``(T, K, D)``, and
        the non-finite report in ``invalid``.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, X and Y have differing row counts, ``n_perms < 1``,
            ``N < 4``, ``bandwidth`` is non-positive or non-finite,
            ``on_invalid`` is not one of the three policies, the non-finite
            policy refuses the sample, ``key`` is ``None``, or
            ``problem`` has categorical parameters. Categorical parameters
            are rejected because the Gaussian input kernel reads a level code
            as a distance, and the arbitrary code order makes that
            meaningless.

    Warns:
        JaxgsaWarning: If an output slice has zero variance, or if JAX is in
            single precision. The V-statistic subtracts three sums of the same
            magnitude, so float32 keeps only three or four correct digits and
            the index moves with the order of the sample rows.
    """
    from jaxgsa.hsic import SPEC

    ctx = prepare(
        SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("n_perms", n_perms, 1),
            require(
                bandwidth is None or (bandwidth > 0 and math.isfinite(bandwidth)),
                f"bandwidth must be positive and finite, got {bandwidth}",
            ),
        ),
        min_kept=_MIN_SAMPLES,
    )
    X, invalid = ctx.inputs, ctx.invalid
    # A sample count, not a scalar argument: it is checked once the shape
    # contract has held, and a sample this small costs nothing to have read.
    if X.shape[0] < _MIN_SAMPLES:
        raise ValueError(f"N must be >= {_MIN_SAMPLES} for HSIC, got {X.shape[0]}")
    # Checked after the preamble and the shape contract, so a problem this
    # method refuses outright (categorical parameters, a sample too small)
    # still reports that, and not a missing key.
    if key is None:
        raise ValueError("key is required for the permutation test")
    _warn_single_precision()

    r2_all, t_all, p_all, raw_all = indices(
        problem,
        X,
        ctx.Y3,
        n_perms=n_perms,
        key=key,
        bandwidth=bandwidth,
        batch_size=batch_size,
        prenormalize=prenormalize,
    )

    r2_all = ctx.squeeze(r2_all)
    t_all = ctx.squeeze(t_all)
    p_all = ctx.squeeze(p_all)
    raw_all = ctx.squeeze(raw_all)

    return HSICResult(
        R2_HSIC=r2_all,
        T_HSIC=t_all,
        p_values=p_all,
        hsic_raw=raw_all,
        problem=problem,
        invalid=invalid,
    )
