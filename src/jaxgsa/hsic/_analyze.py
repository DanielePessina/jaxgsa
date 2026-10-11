"""HSIC index estimators for kernel-based sensitivity analysis.

The estimators compute R2-HSIC (normalized first-order) and Total HSIC
indices from arbitrary (X, Y) sample pairs. They use Gaussian RBF kernels
and pick the bandwidth with the median heuristic. A caller's ``bandwidth``
multiplies that heuristic rather than replacing it, so every width carries
the scale of its own variable and the indices are invariant under
``Y -> a*Y + b`` and under a rescaling of any input.

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
    Gretton, Bousquet, Smola & Schölkopf (2005). Measuring statistical
    dependence with Hilbert-Schmidt norms. ALT 2005, LNCS 3734:63-77.
    (The ``tr(KHLH)/n^2`` V-statistic estimator used here is from this
    paper, not from Gretton et al.'s separate JMLR 6:2075-2129 paper on
    kernel independence tests.)
    Da Veiga (2015). Global sensitivity analysis with dependence measures.
    J. Stat. Comput. Simul. 85(7):1283-1305, doi 10.1080/00949655.2014.945932.
    Larsen & Alexanderian (2026). arXiv:2603.00849.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache, partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.entry import at_least, prepare, require
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.precision import x64_enabled
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import _prepare_Y, as_key
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.hsic._result import HSICResult
from jaxgsa.problem import Problem

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = ("n_perms", "key", "bandwidth", "on_invalid")

_MIN_SAMPLES = 4

_UINT_BY_ITEMSIZE: dict[int, type] = {2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}
"""Unsigned integer type of the same width as a float type, by byte count."""


def _linear_quantile_plan(q: float, n_elements: int) -> tuple[int, int, Array]:
    """Reproduce the index and weight arithmetic of ``jnp.quantile``.

    ``jnp.quantile(a, q)`` with the default ``method="linear"`` reads two
    order statistics of the sorted array and blends them. The position
    ``q * (n - 1)`` is computed in the dtype ``q`` promotes to, which for a
    bare Python float is the default float dtype, and only the finished
    value is cast back to the dtype of ``a``. So the rounding of that
    product is part of the answer, and it is the *default* float dtype that
    decides it, not the dtype of ``a``. This helper repeats exactly that
    arithmetic on the host, which is possible because both ``q`` and the
    element count are static.

    Taking the dtype from ``a`` instead breaks the match. With x64 on and a
    ``float32`` ``a`` of about ``2^23`` elements or more, the target
    position is a half-integer that ``float32`` cannot hold: ``jnp.quantile``
    still computes ``8390655.5`` in ``float64`` and returns ``8390656.0``,
    while a ``float32`` plan collapses to the single rank ``8390655`` and
    returns ``8390655.0``. The float32 rounding at that size is
    ``jnp.quantile``'s own, and reproducing it is this helper's job.

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


def _count_dtype() -> DTypeLike:
    """Return the widest integer dtype the current JAX mode really supports.

    Asking for ``int64`` while ``jax_enable_x64`` is off makes JAX truncate to
    ``int32`` and emit a ``UserWarning`` on every call. Choosing the dtype from
    the mode gives the same arithmetic with no warning in either mode.

    Returns:
        ``jnp.int64`` in x64 mode, otherwise ``jnp.int32``.
    """
    return jnp.int64 if x64_enabled() else jnp.int32


def _pair_boundaries(y: Array, t: Array) -> tuple[Array, Array]:
    """Find, in every row, where the squared distances cross a threshold.

    ``y`` is sorted ascending. In row ``i`` the value
    ``fl((y_j - y_i)^2)`` does not decrease as ``j`` moves away from ``i`` in
    either direction. Floating-point subtraction and squaring both round
    monotonically, so the claim holds exactly, not just in real arithmetic.
    The entries ``<= t`` of a row are therefore one contiguous block around
    the diagonal, and two binary searches find its ends.

    Args:
        y: Sorted values, shape ``(N,)``.
        t: Scalar threshold of the same dtype, not NaN.

    Returns:
        ``(right, left)`` integer arrays of shape ``(N,)``. ``right[i]`` is
        the first ``j >= i`` with ``d2(i, j) > t`` (``N`` if none).
        ``left[i]`` is the first ``j <= i`` with ``d2(i, j) <= t``. Row ``i``
        holds ``right[i] - left[i]`` entries ``<= t``, the diagonal included.
    """
    n = y.shape[0]
    i = jnp.arange(n, dtype=jnp.int32)

    def d2(j: Array) -> Array:
        return (y[j] - y) ** 2

    right_lo, right_hi = i, jnp.full((n,), n, dtype=jnp.int32)
    left_lo, left_hi = jnp.zeros((n,), dtype=jnp.int32), i
    for _ in range(n.bit_length()):
        active = right_lo < right_hi
        mid = (right_lo + right_hi) // 2
        ok = d2(jnp.minimum(mid, n - 1)) <= t
        right_lo = jnp.where(active & ok, mid + 1, right_lo)
        right_hi = jnp.where(active & ~ok, mid, right_hi)

        active = left_lo < left_hi
        mid = (left_lo + left_hi) // 2
        ok = d2(mid) <= t
        left_hi = jnp.where(active & ok, mid, left_hi)
        left_lo = jnp.where(active & ~ok, mid + 1, left_lo)
    return right_lo, left_lo


def _pairwise_quantile_by_selection(x: Array, q: float) -> Array:
    """Take a linear quantile of all ``N^2`` squared distances without forming them.

    This returns bit for bit what ``jnp.quantile`` returns on
    ``(x[:, None] - x[None, :]) ** 2``, but it never builds the ``(N, N)``
    matrix. After one sort of ``x``, every row of that matrix is
    unimodal around its diagonal entry (see :func:`_pair_boundaries`), so the
    number of entries ``<= t`` costs ``O(N log N)``. A bisection over the bit
    pattern of the answer then finds the exact order statistic. Squared
    distances are non-negative, so the raw bit pattern is already ordered
    like the value. The pair set is the full matrix: the ``N`` zero
    diagonal entries and both copies of each pair.

    NaN follows ``jnp.quantile``. Any NaN or infinity in ``x`` puts a NaN in the
    distance matrix (``inf - inf`` on the diagonal), so the result is NaN.
    The selection is not differentiable.

    Args:
        x: Values, shape ``(N,)``.
        q: Quantile in [0, 1], as a Python float.

    Returns:
        Scalar quantile, in the dtype of ``x``.

    Raises:
        TypeError: If the float width of ``x`` has no unsigned integer
            counterpart.
    """
    itemsize = jnp.dtype(x.dtype).itemsize
    if itemsize not in _UINT_BY_ITEMSIZE:
        raise TypeError(f"cannot select order statistics of dtype {x.dtype}")
    uint = _UINT_BY_ITEMSIZE[itemsize]
    n_bits = itemsize * 8
    n = x.shape[0]
    count_dtype = _count_dtype()

    finite = jnp.all(jnp.isfinite(x))
    y = jnp.sort(jax.lax.stop_gradient(x))
    inf_bits = jax.lax.bitcast_convert_type(jnp.asarray(jnp.inf, dtype=x.dtype), uint)
    one = jnp.asarray(1, dtype=uint)

    def count_at_or_below(bits: Array) -> tuple[Array, Array, Array]:
        # Patterns above infinity are NaN. Every entry is below them, which
        # is the same count as at infinity.
        t = jax.lax.bitcast_convert_type(jnp.minimum(bits, inf_bits), x.dtype)
        right, left = _pair_boundaries(y, t)
        return jnp.sum((right - left).astype(count_dtype)), right, left

    def select(rank: int) -> Array:
        """Return the bit pattern of the entry at 0-based ``rank``."""
        r = jnp.asarray(rank, dtype=count_dtype)

        def step(k: Array, ans: Array) -> Array:
            cand = ans | (one << (n_bits - 2 - k).astype(uint))
            below, _, _ = count_at_or_below(cand - one)
            return jnp.where(below <= r, cand, ans)

        # The sign bit is never set: every entry is non-negative.
        return jax.lax.fori_loop(0, n_bits - 1, step, jnp.zeros((), dtype=uint))

    low_rank, high_rank, high_weight = _linear_quantile_plan(q, n * n)
    lo_bits = select(low_rank)
    low_value = jax.lax.bitcast_convert_type(lo_bits, x.dtype)

    # The next distinct value above ``low_value`` sits at a row boundary:
    # the first entry past the block of entries ``<= low_value``.
    n_at_or_below, right, left = count_at_or_below(lo_bits)
    top = jnp.asarray(jnp.inf, dtype=x.dtype)
    right_next = jnp.where(right < n, (y[jnp.minimum(right, n - 1)] - y) ** 2, top)
    left_next = jnp.where(left > 0, (y[jnp.maximum(left - 1, 0)] - y) ** 2, top)
    next_value = jnp.minimum(jnp.min(right_next), jnp.min(left_next))
    high_value = jnp.where(n_at_or_below >= high_rank + 1, low_value, next_value)

    blended = jnp.quantile(jnp.stack([low_value, high_value]), high_weight)
    return jnp.where(finite, blended, jnp.nan)


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
    :func:`_pairwise_quantile_by_selection`, which returns exactly what
    ``jnp.quantile`` would return and is roughly 40x faster here.

    Args:
        x: Values for one variable, shape ``(N,)``.

    Returns:
        Scalar median of the off-diagonal squared distances, floored at 1e-20.
    """
    N = x.shape[0]
    q = (N**2 + N - 1) / (2 * (N**2 - 1))
    return jnp.maximum(_pairwise_quantile_by_selection(x, q), 1e-20)


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


def _build_one_kernel(x: Array, bandwidth: float) -> Array:
    """Build a Gaussian RBF kernel matrix for one variable.

    ``bandwidth`` is a multiplier on the median heuristic, not a width: the
    resolved squared bandwidth is ``bandwidth**2 * median_sq``, so a
    multiplier of exactly 1.0 uses the heuristic itself, bit for bit. The
    heuristic already carries the scale of ``x``, so the kernel -- and every
    index built from it -- is unchanged under ``x -> a*x + b``.

    There is deliberately no row blocking here, and no batching keyword
    anywhere in HSIC. The estimator's peak memory is the *resident* kernel
    stacks -- the D raw input kernels, their D augmented complements, the full
    augmented product and one output kernel, ~``(2D + 1) * N^2`` floats at
    once -- and a row-blocked build only bounded a transient of one build
    while that stack stayed resident. It was measured useless and removed.

    Args:
        x: Values for one variable, shape ``(N,)``.
        bandwidth: Positive multiplier on the median-heuristic bandwidth.

    Returns:
        Kernel matrix, shape ``(N, N)``.
    """
    sigma_sq = jnp.asarray(bandwidth, dtype=x.dtype) ** 2 * _median_bandwidth_sq(x)
    return jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * sigma_sq))


def _augmented_kernels(Ks_stack: Array) -> Array:
    """Build augmented kernels: K*_d = 1 + center(K_d).

    The augmented kernel carries a constant term. The product of augmented
    kernels therefore captures all interaction orders, not just the highest.
    Correct total HSIC indices need this.

    Takes and returns a stacked array, not a list. Building each input
    kernel as a Python list element and stacking it, alongside a second list
    for its augmented form, doubled the resident kernel memory for no
    reason: this and :func:`_complement_kernels` both work on the stack the
    caller already holds, so :func:`indices` never has to keep more than the
    two stacks it actually reads plus the full product -- ``(2D + 1) * N^2``
    floats, not ``5D + 1``.

    Args:
        Ks_stack: Stacked raw kernel matrices, shape ``(D, N, N)``.

    Returns:
        Stacked augmented kernel matrices, shape ``(D, N, N)``.
    """
    return jax.vmap(lambda K: jnp.ones_like(K) + _center_kernel(K))(Ks_stack)


def _complement_kernels(Ks_stack: Array) -> tuple[Array, Array]:
    """Build the full product kernel and every complement product kernel.

    A prefix-suffix pass gives all D complements in linear time. The pass
    itself still walks a Python list -- each step depends on the last, so
    there is no batched primitive for it -- but that list is local to this
    function and is freed on return; only the stacked complements go back
    to the caller. See :func:`_augmented_kernels` for why the stack-in,
    stack-out shape matters here.

    Args:
        Ks_stack: Stacked kernel matrices, shape ``(D, N, N)``.

    Returns:
        ``(K_full, complements)``. ``K_full`` is the Hadamard product of all
        of ``Ks_stack``, shape ``(N, N)``. ``complements[d]`` is the product
        of every entry of ``Ks_stack`` except ``d``, stacked to ``(D, N, N)``.
    """
    D = Ks_stack.shape[0]
    if D == 1:
        return Ks_stack[0], jnp.ones_like(Ks_stack)

    prefix = [jnp.ones_like(Ks_stack[0])] * D
    for i in range(1, D):
        prefix[i] = prefix[i - 1] * Ks_stack[i - 1]

    suffix = [jnp.ones_like(Ks_stack[0])] * D
    for i in range(D - 2, -1, -1):
        suffix[i] = suffix[i + 1] * Ks_stack[i + 1]

    K_full = prefix[-1] * Ks_stack[-1]
    return K_full, jnp.stack([prefix[d] * suffix[d] for d in range(D)])


@partial(jax.jit, static_argnames=("total", "bandwidth"))
def _input_setup(
    X_unit: Array, key: Array, total: int, bandwidth: float
) -> tuple[Array, Array, Array, Array, Array]:
    """Build everything that depends only on ``X`` and the key, in one jit.

    Running this op by op outside ``jit`` cost far more in dispatch than in
    arithmetic. The input kernels and augmented products are built once
    (independent of Y). They are stacked directly and the augmented and
    complement kernels are built from the stack, so no per-input Python list
    of ``(N, N)`` matrices survives to the kernel call -- resident memory is
    the two stacks and the full product, ``(2D + 1) * N^2`` floats.

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        key: PRNG key for the permutation test.
        total: Number of output slices (static).
        bandwidth: Multiplier on the median-heuristic bandwidth (static).

    Returns:
        ``(Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, keys)`` with
        the shapes :func:`_get_hsic_kernel` documents, and ``keys`` of shape
        ``(total,)`` holding ``fold_in(key, i)`` for each slice.
    """
    D = X_unit.shape[1]
    Ks_stack = jnp.stack([_build_one_kernel(X_unit[:, d], bandwidth) for d in range(D)])
    K_aug_full, K_aug_compls_stack = _complement_kernels(_augmented_kernels(Ks_stack))
    # Self-HSIC HSIC(K_d, K_d) is output-independent, so compute it once
    # here and share it across every output slice.
    hsic_xxs = jax.vmap(lambda K: _hsic_v(K, K))(Ks_stack)
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(total))
    return Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, keys


@lru_cache(maxsize=32)
def _get_hsic_kernel(n_perms: int, bandwidth: float):
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
        bandwidth: Multiplier on the median-heuristic bandwidth of the
            output kernel (static; it is a compile-time constant).

    Returns:
        A jitted callable that returns ``(R2, T, p_values,
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
            ``(R2, T, p_values, hsic_raw)`` each of shape ``(D,)``.
        """
        # The barrier stops XLA fusing the kernel build into the reductions
        # that read it. Fused, the exponential is recomputed inside each
        # reduction loop and the sums accumulate in a different order, which
        # moves the last bit of the indices away from what the same kernel
        # returns when it is handed a materialized matrix. HSIC reads L a few
        # dozen times per slice, so materializing it once is the cheaper
        # arrangement anyway.
        L = jax.lax.optimization_barrier(_build_one_kernel(y_col, bandwidth))
        N = L.shape[0]
        eps = jnp.finfo(L.dtype).eps

        hsic_yy = _hsic_v(L, L)
        zero_var = hsic_yy < eps

        hsic_xys = jax.vmap(lambda K: _hsic_v(K, L))(Ks_stack)
        # A constant input column has an all-ones kernel whose self-HSIC
        # vanishes exactly, and it provably depends on no output. Those
        # parameters report exactly 0 (matching PAWN), not the NaN of a
        # degenerate output, so only zero_var (the output's own degeneracy)
        # yields NaN.
        zero_input = hsic_xxs < eps
        denoms = jnp.sqrt(jnp.maximum(hsic_xxs * hsic_yy, eps**2))
        r2 = jnp.where(zero_var, jnp.nan, jnp.where(zero_input, 0.0, hsic_xys / denoms))

        hsic_full = _hsic_v(K_aug_full, L)
        hsic_full_safe = jnp.where(jnp.abs(hsic_full) < eps, jnp.nan, hsic_full)
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
            ``(R2, T, p_values, hsic_raw)`` each ``(total, D)``.
        """

        def _step(column: tuple[Array, Array]) -> tuple[Array, Array, Array, Array]:
            """Run the atomic kernel on one (column, key) pair."""
            y_col, key = column
            return _one_slice(Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, y_col, key)

        return jax.lax.map(_step, (Y_cols.T, keys))

    return jax.jit(_impl)


def _warn_constant_inputs(problem: Problem, X: Array) -> None:
    """Warn that a parameter is constant in the sample.

    A constant column's Gaussian kernel is identically 1, so its R2-HSIC is
    exactly 0 and its dependence on any output is provably zero. The kernel
    applies that value on its own; this only says it out loud, because a
    constant input is almost always a modelling mistake worth naming.

    Args:
        problem: Problem definition, used to name the parameters.
        X: Input samples, shape ``(N, D)``.

    Warns:
        JaxgsaWarning: If any column of ``X`` is constant (``max == min``),
            naming the parameter.
    """
    X_host = np.asarray(X)
    const = np.max(X_host, axis=0) == np.min(X_host, axis=0)
    dims = np.flatnonzero(const)
    if dims.size == 0:
        return
    names = ", ".join(repr(problem.names[d]) for d in dims)
    plural = "s" if dims.size > 1 else ""
    warnings.warn(
        f"jaxgsa.hsic: parameter{plural} {names} {'are' if dims.size > 1 else 'is'} "
        "constant in the sample, so their Gaussian kernel is identically 1 "
        "and R2 is exactly 0 — a constant input has provably zero "
        "dependence on any output",
        stacklevel=2,
        category=JaxgsaWarning,
    )


def _warn_single_precision() -> None:
    """Warn that float32 costs the HSIC V-statistic most of its digits.

    The V-statistic is a difference of three sums over the whole ``(N, N)``
    kernel matrix, and the three are of the same magnitude while the answer is
    small. Cancellation therefore eats the leading digits, and what survives
    is set by the summation order. Reordering the sample rows -- a change that
    cannot alter the true value -- moves a measured index. On Ishigami with
    ``N = 2048`` the shift is ``4e-4`` relative in float32 against ``8e-13``
    in float64, and it grows with ``N`` because the sums do. That is the same
    size as the gap HSIC is often asked to resolve between two weak
    parameters.

    The damage is not spread evenly over the parameters. What the rounding
    costs is roughly a fixed *absolute* amount, set by the epsilon of the
    three sums, so the *relative* error on an index is that amount divided by
    the index. On Ishigami at ``N = 128`` the three sums all sit between 0.3
    and 0.65 while the indices they produce run from ``3e-4`` to ``1e-1``, a
    cancellation ratio of 6 for the strongest parameter and 2100 for the
    weakest. The measured relative shift follows that ratio times the machine
    epsilon of the dtype in both precisions. So a strong parameter keeps most
    of its float32 digits and a weak one keeps almost none, which is exactly
    the wrong way round for reading a ranking off the small end.
    """
    if not x64_enabled():
        warnings.warn(
            "jaxgsa.hsic: JAX is in single precision; the HSIC V-statistic cancels three "
            "large sums against each other, so float32 leaves only about three or four "
            "correct digits and the index changes with the order of the sample rows. Small "
            "indices and close rankings are not reliable. Enable float64 with "
            'jax.config.update("jax_enable_x64", True) before the analysis.',
            stacklevel=3,
            category=JaxgsaWarning,
        )


class HSICIndices(NamedTuple):
    """HSIC indices as plain arrays, in the order :func:`indices` returns them.

    Tuple unpacking works: ``R2, T, p_values, hsic_raw = indices(...)``.

    Attributes:
        R2: Normalized first-order HSIC indices, as ``HSICResult.R2``.
        T: Total-order HSIC indices, as ``HSICResult.T``.
        p_values: Permutation-test p-values, as ``HSICResult.p_values``.
        hsic_raw: Unnormalized ``HSIC(x_i, Y)``, as ``HSICResult.hsic_raw``.
    """

    R2: Array
    T: Array
    p_values: Array
    hsic_raw: Array


def indices(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    key: Array | int,
    bandwidth: float = 1.0,
) -> HSICIndices:
    """Compute HSIC indices as plain arrays, with no diagnostics.

    This is the transformable core of :func:`analyze`. It builds the same
    kernels, runs the same permutation test and returns the same numbers, but
    it does nothing else: no non-finite check, no row dropping, no
    zero-variance warning, no single-precision warning, no sample-count
    check, no :class:`jaxgsa.hsic.HSICResult`, and no read of any array value
    on the host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad``
    and ``jax.jacrev``, which :func:`analyze` cannot, because a policy
    decision needs a concrete value and a tracer has none.

    Every branch here is on a shape or on a Python scalar. ``n_perms`` and
    ``bandwidth`` are both static: they are read while the graph is being
    traced and never at run time. ``bandwidth`` scales the median-heuristic
    width and is baked into the compiled kernel, which is keyed on it (see
    :func:`_get_hsic_kernel`). Changing it recompiles; it costs no
    device-side conditional.

    Peak memory is the resident kernel stacks, ~``(2D + 1) * N^2`` floats at
    once, and no keyword bounds it — see :func:`analyze` for why there is
    deliberately no batching keyword here.

    No bootstrap and no confidence interval, here or on :func:`analyze`. The
    permutation ``p_values`` are the uncertainty statement, and the module
    docstring says why a row bootstrap of a V-statistic would be worse than
    redundant.

    Tier T4 (behavioural contract): the returned arrays must equal
    ``R2``, ``T``, ``p_values`` and ``hsic_raw`` of ``analyze``'s
    result on clean outputs, and the function must survive ``jit``, ``vmap``
    and ``jit(jacrev(...))``. Checked in ``tests/test_hsic.py``.

    ``X`` is traceable whatever the marginals are. The unit-interval
    transform stays on device for uniform, Gaussian and truncated-Gaussian
    columns alike, and the T4 test covers all three.

    A parameter that is constant in the sample yields an exact 0 for
    ``R2`` from inside the kernel, as in :func:`analyze`; only the
    warning is missing here, because a warning needs a concrete value.

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
        bandwidth: Multiplier on the median-heuristic bandwidth, as in
            :func:`analyze`.

            Gradients are always taken at a frozen bandwidth. The median
            heuristic picks an order statistic through a bit-pattern
            bisection, which carries no derivative, and the multiplier rides
            on top of it, so a gradient with respect to ``X`` or ``Y`` never
            sees the width move. There is no way to opt out of that: every
            bandwidth is a multiple of the same selected median. What comes
            back is the exact derivative of the estimator whose bandwidth is
            held at the value the heuristic chose for the point being
            differentiated at, which is the usual convention for a
            median-heuristic kernel. Read it that way, and remember that a
            finite difference of the same call is *not* the same quantity:
            it also moves the median, so it will not match.

    Returns:
        An :class:`HSICIndices` named tuple ``(R2, T, p_values, hsic_raw)``,
        each of shape ``(D,)``, ``(K, D)`` or ``(T, K, D)`` to match the rank
        of ``Y``, exactly as ``analyze`` reports them.

    Raises:
        ValueError: If any parameter is categorical. The Gaussian input
            kernel would read a level code as a distance.
        TypeError: If the float width of ``X`` or ``Y`` has no unsigned
            integer counterpart for the median-heuristic selection.
    """
    key = as_key(key, "hsic")
    D = problem.n_params
    X_unit = cdf_to_unit_interval(X, problem)

    Y_3d, layout = _prepare_Y(Y)
    _N, T, K = Y_3d.shape

    # One column per output slice, in the same (t, k) order the indices are
    # written back in, and one key per column. Each slice reads its own key,
    # so the permutation draws do not depend on how the columns are grouped.
    N = X_unit.shape[0]
    total = T * K
    Y_cols = Y_3d.reshape(N, total)
    Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, keys = _input_setup(
        X_unit, key, total, bandwidth
    )

    kernel = _get_hsic_kernel(n_perms, bandwidth)
    r2_all, t_all, p_all, raw_all = kernel(
        Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, Y_cols, keys
    )
    return HSICIndices(
        R2=layout.squeeze(r2_all.reshape(T, K, D)),
        T=layout.squeeze(t_all.reshape(T, K, D)),
        p_values=layout.squeeze(p_all.reshape(T, K, D)),
        hsic_raw=layout.squeeze(raw_all.reshape(T, K, D)),
    )


def analyze(
    problem: Problem,
    X: Array,
    Y: Array | IrregularY,
    *,
    n_perms: int = 200,
    key: Array | int,
    bandwidth: float = 1.0,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
) -> HSICResult | IrregularResult:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) sensitivity indices.

    HSIC uses kernel embeddings to measure the statistical dependence between
    each parameter and the output. It therefore finds nonlinear and
    non-monotonic relationships that correlation-based screening misses. HSIC
    is a given-data method: any (X, Y) sample pair works, and no special
    sampling design is required. The function reports two indices:

    - **R2-HSIC**: ``HSIC(x_i, Y)`` normalized by the geometric mean of the
      self-similarities, ``HSIC(x_i, Y) / sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))``.
      It lies in [0, 1]. A value of 0 means x_i and Y are independent. A
      parameter that is constant in the sample yields exactly 0: its kernel
      is identically 1, and a constant input has provably zero dependence on
      any output. This is the first-order view.
    - **Total HSIC (T)**: the fraction of the joint dependence lost when
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

    **Memory is quadratic in N, and no keyword bounds it.** The estimator
    keeps the D raw input kernels, their D augmented complement products,
    the full augmented product and one output kernel resident at once —
    peak memory is ~``(2D + 1) * N^2`` floats. There is deliberately no
    batching keyword: a row-blocked kernel build only bounded a transient
    of the build while the resident stacks stayed, so it was measured
    useless and removed. If the sample does not fit, thin ``N`` (HSIC
    converges quickly in N) or lower ``D`` by screening first.

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
            The indices do not depend on the units: the bandwidth tracks the
            data, so ``Y`` and ``a*Y + b`` give the same answer. Outputs of
            extreme magnitude are still worth rescaling by hand, because the
            squared distances the kernel builds can overflow float32 long
            before the index would care; ``(Y - Y.mean(0)) / Y.std(0)``
            changes nothing else.
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
        bandwidth: Multiplier on the Gaussian-kernel bandwidth, applied to
            every parameter and to the output. The width itself is always
            chosen per variable by the median heuristic, and this scales it:
            ``1.0`` (default) is exactly the heuristic, ``0.5`` halves every
            width, ``2.0`` doubles them. Precisely, the standard deviation of
            the Gaussian is ``bandwidth * sqrt(m)``, where ``m`` is the median
            of the off-diagonal *squared* pairwise distances of that variable.
            Squaring is monotone on distances, so ``sqrt(m)`` is the median
            pairwise distance itself — the two agree to 1.6e-11 relative at
            ``N = 257``, and differ only for a tiny sample, where the linear
            interpolation between the two middle order statistics is done on
            the squares.

            The multiplier is relative on purpose. The heuristic carries the
            scale of each variable, so every bandwidth this method uses does
            too, and the indices are unchanged under ``Y -> a*Y + b`` and
            under a rescaling of any input. There is deliberately no way to
            ask for an absolute width: an absolute number would have to be
            read against units this method otherwise never sees, and one
            fixed value cannot be right for both a parameter mapped to
            ``[0, 1]`` and an output measured in megapascals.
        on_invalid: What to do about a row of ``X`` or ``Y`` that holds a
            non-finite value. ``"raise"`` (default) refuses the sample,
            ``"drop"`` removes those rows and analyzes the rest, and
            ``"propagate"`` warns and computes anyway. ``X`` and ``Y`` are
            checked together, so a bad input takes its own output with it.
            See :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``T``. Pass ``False`` for a silent run.
    Returns:
        An :class:`HSICResult` with ``R2``, ``T``, ``p_values``, and
        ``hsic_raw``, each shaped ``(D,)``, ``(K, D)``, or ``(T, K, D)``, the
        non-finite report in ``invalid``, and the ``bandwidth`` and
        ``n_perms`` the run used. Those two are on the result because the
        index moves with the bandwidth, so a stored HSIC number is ambiguous
        without it.

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
        JaxgsaWarning: If an output slice has zero variance (all indices NaN),
            if a parameter is constant in the sample (its R2 is exactly
            0), or if JAX is in single precision. The V-statistic subtracts
            three sums of the same magnitude, so float32 keeps only three or
            four correct digits and the index moves with the order of the
            sample rows.
    """
    key = as_key(key, "hsic")
    if _is_irregular_y(Y):
        return analyze_irregular(
            analyze,
            channel_data=X,
            problem=problem,
            Y=Y,
            n_expected=int(X.shape[0]),
            design_based=False,
            verbose=verbose,
            kwargs=_fwd_kwargs(locals(), _IRREGULAR_KWARGS),
        )
    Y = cast(Array, Y)

    from jaxgsa.hsic import _SPEC

    ctx = prepare(
        _SPEC,
        problem,
        Y,
        X=X,
        on_invalid=on_invalid,
        checks=(
            at_least("n_perms", n_perms, 1),
            require(
                bandwidth > 0 and math.isfinite(bandwidth),
                f"bandwidth must be positive and finite, got {bandwidth}",
            ),
        ),
        min_kept=_MIN_SAMPLES,
    )
    assert ctx.X is not None  # a given-data method always passes X to prepare()
    X, invalid = ctx.X, ctx.invalid
    # A sample count, not a scalar argument: it is checked once the shape
    # contract has held, and a sample this small costs nothing to have read.
    if X.shape[0] < _MIN_SAMPLES:
        raise ValueError(f"N must be >= {_MIN_SAMPLES} for HSIC, got {X.shape[0]}")
    # Checked after the preamble and the shape contract, so a problem this
    # method refuses outright (categorical parameters, a sample too small)
    # still reports that, and not a missing key.
    if key is None:
        raise ValueError("key is required for the permutation test")
    _warn_constant_inputs(problem, X)
    _warn_single_precision()

    t0 = _verbose.tic()
    r2_all, t_all, p_all, raw_all = indices(
        problem,
        X,
        ctx.Y3,
        n_perms=n_perms,
        key=key,
        bandwidth=bandwidth,
    )

    r2_all = ctx.squeeze(r2_all)
    t_all = ctx.squeeze(t_all)
    p_all = ctx.squeeze(p_all)
    raw_all = ctx.squeeze(raw_all)

    result = HSICResult(
        R2=r2_all,
        T=t_all,
        p_values=p_all,
        hsic_raw=raw_all,
        problem=problem,
        invalid=invalid,
        bandwidth=bandwidth,
        n_perms=n_perms,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.T)
        n_runs, T, K = ctx.Y3.shape
        _verbose.analysis_summary(
            method="jaxgsa.hsic.analyze",
            problem=problem,
            n_runs=int(n_runs),
            T=T,
            K=K,
            invalid=invalid,
            timings=[("estimator (includes compile on the first call)", elapsed)],
            notes=[
                f"n_perms: {n_perms}",
                f"bandwidth: {bandwidth} (median-heuristic multiplier)",
            ],
            index_name="T",
            values=result.T,
        )
    return result
