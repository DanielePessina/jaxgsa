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
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import get_memory_budget
from jaxgsa._core.entry import _user_stacklevel, at_least, prepare, require
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


_MAX_N_32BIT = 65_535
"""Largest N whose ``N^2`` pair count fits a ``uint32``, the x64-off counter."""


def _count_dtype() -> DTypeLike:
    """Return the integer dtype that counts the ``N^2`` pairs of the median heuristic.

    Asking for ``int64`` while ``jax_enable_x64`` is off makes JAX truncate to
    ``int32`` and emit a ``UserWarning`` on every call, so the dtype follows
    the mode. Without x64 the count is unsigned: ``int32`` overflows once
    ``N^2`` passes ``2^31 - 1`` (``N > 46,340``), but ``uint32`` holds every
    count up to ``N = 65,535``. :func:`_check_sample_size` refuses larger N.
    All counts and ranks are non-negative and nothing subtracts them, so the
    unsigned type changes no result.

    Returns:
        ``jnp.int64`` in x64 mode, otherwise ``jnp.uint32``.
    """
    return jnp.int64 if x64_enabled() else jnp.uint32


def _check_sample_size(n: int) -> None:
    """Refuse a sample too large for the 32-bit pair count of the median heuristic.

    N is a static shape, so the check also works under ``jit``. It runs
    before any kernel is built.

    Args:
        n: Number of samples.

    Raises:
        ValueError: If x64 is off and ``n`` exceeds 65,535.
    """
    if n > _MAX_N_32BIT and not x64_enabled():
        raise ValueError(
            f"jaxgsa.hsic: N = {n} is too large without x64; the median-heuristic "
            f"bandwidth counts all N^2 sample pairs, and that count only fits 32 bits "
            f"for N <= {_MAX_N_32BIT}. Enable x64 (64-bit integers and floats) with "
            'jax.config.update("jax_enable_x64", True), or thin the sample. At this N '
            "the dense (N, N) kernels need many GB of memory anyway."
        )


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

    # A rolled loop, not a Python unroll: this runs inside a bisection that is
    # itself inlined once per input, and unrolling made the first call
    # compile for tens of seconds at N = 1024.
    def body(_: Array, carry: tuple[Array, Array, Array, Array]) -> tuple[Array, ...]:
        right_lo, right_hi, left_lo, left_hi = carry
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
        return right_lo, right_hi, left_lo, left_hi

    init = (i, jnp.full((n,), n, dtype=jnp.int32), jnp.zeros((n,), dtype=jnp.int32), i)
    right_lo, _, left_lo, _ = jax.lax.fori_loop(0, n.bit_length(), body, init)
    return right_lo, left_lo


def _pairwise_quantile_by_selection(x: Array, q: float, *, skip_zero: bool = False) -> Array:
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
        skip_zero: If true, ignore the zero-distance pairs (the diagonal and
            tied values) and return the median of the non-zero pairs. ``q``
            is then not used for columns with tied values. For a column with
            no ties the result equals the ``q`` quantile, bit for bit. If
            every pair is zero, the result is 0.

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

    def select(rank: int | Array) -> Array:
        """Return the bit pattern of the entry at 0-based ``rank``."""
        r = jnp.asarray(rank, dtype=count_dtype)

        def step(k: Array, ans: Array) -> Array:
            cand = ans | (one << (n_bits - 2 - k).astype(uint))
            below, _, _ = count_at_or_below(cand - one)
            return jnp.where(below <= r, cand, ans)

        # The sign bit is never set: every entry is non-negative.
        return jax.lax.fori_loop(0, n_bits - 1, step, jnp.zeros((), dtype=uint))

    low_rank, high_rank, high_weight = _linear_quantile_plan(q, n * n)
    low_r: int | Array = low_rank
    high_r: int | Array = high_rank
    n_zero: Array | None = None
    if skip_zero:
        # Zero distances: the N diagonal entries plus every tied pair.
        n_zero = count_at_or_below(jnp.zeros((), dtype=uint))[0]
        n_nonzero_pairs = (jnp.asarray(n * n, dtype=count_dtype) - n_zero) // 2
        is_tie_free = n_zero == n
        # The non-zero entries fill the top of the sorted array, each twice.
        # Positions Z + M - 1 and Z + M, weight 1/2, give their median.
        dyn_low = n_zero + n_nonzero_pairs - 1
        low_r = jnp.where(is_tie_free, jnp.asarray(low_rank, dtype=count_dtype), dyn_low)
        high_r = jnp.where(is_tie_free, jnp.asarray(high_rank, dtype=count_dtype), dyn_low + 1)
        high_weight = jnp.where(
            is_tie_free, high_weight, jnp.asarray(0.5, dtype=high_weight.dtype)
        )
    lo_bits = select(low_r)
    low_value = jax.lax.bitcast_convert_type(lo_bits, x.dtype)

    # The next distinct value above ``low_value`` sits at a row boundary:
    # the first entry past the block of entries ``<= low_value``.
    n_at_or_below, right, left = count_at_or_below(lo_bits)
    top = jnp.asarray(jnp.inf, dtype=x.dtype)
    right_next = jnp.where(right < n, (y[jnp.minimum(right, n - 1)] - y) ** 2, top)
    left_next = jnp.where(left > 0, (y[jnp.maximum(left - 1, 0)] - y) ** 2, top)
    next_value = jnp.minimum(jnp.min(right_next), jnp.min(left_next))
    high_value = jnp.where(
        n_at_or_below >= jnp.asarray(high_r, dtype=count_dtype) + 1, low_value, next_value
    )
    if n_zero is not None:
        # Every pair is zero: no non-zero distance exists. Return 0.
        all_zero = n_zero == jnp.asarray(n * n, dtype=count_dtype)
        low_value = jnp.where(all_zero, 0.0, low_value)
        high_value = jnp.where(all_zero, 0.0, high_value)

    blended = jnp.quantile(jnp.stack([low_value, high_value]), high_weight)
    return jnp.where(finite, blended, jnp.nan)


def _median_bandwidth_sq(x: Array) -> Array:
    """Compute the median-heuristic squared bandwidth for one variable.

    The median heuristic takes the median of the non-zero pairwise squared
    distances. For a column with no tied values these are the
    ``M = (N^2 - N) / 2`` off-diagonal pairs. Zero distances (the diagonal and
    tied values) carry no information about the spread of the sample, and an
    atom holding more than half of the pairs would otherwise pull the median
    to 0 and collapse the kernel to an equality test. A column with
    near-ties (not exact ones) can still collapse; :func:`_warn_collapsed_median`
    warns about it.

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
        Scalar median of the non-zero squared distances, floored at 1e-20.
    """
    N = x.shape[0]
    q = (N**2 + N - 1) / (2 * (N**2 - 1))
    return jnp.maximum(_pairwise_quantile_by_selection(x, q, skip_zero=True), 1e-20)


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

    Args:
        x: Values for one variable, shape ``(N,)``.
        bandwidth: Positive multiplier on the median-heuristic bandwidth.

    Returns:
        Kernel matrix, shape ``(N, N)``.
    """
    return _kernel_from_median(x, _median_bandwidth_sq(x), bandwidth)


def _kernel_from_median(x: Array, median_sq: Array, bandwidth: float) -> Array:
    """Build a Gaussian RBF kernel matrix from a known median squared distance.

    Args:
        x: Values for one variable, shape ``(N,)``.
        median_sq: Median-heuristic squared bandwidth of ``x``, a scalar.
        bandwidth: Positive multiplier on the median-heuristic bandwidth.

    Returns:
        Kernel matrix, shape ``(N, N)``.
    """
    sigma_sq = jnp.asarray(bandwidth, dtype=x.dtype) ** 2 * median_sq
    return jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * sigma_sq))


def _hsic_centered(Kc: Array, Lc: Array) -> Array:
    """Compute the V-statistic HSIC from two centred kernels.

    With ``H = I - (1/n) 1 1^T``, ``tr(K H L H) = sum(HKH * HLH)``, so the
    estimate of :func:`_hsic_v` is one elementwise product sum once both
    kernels are centred. It needs one pass over each matrix instead of
    three, and it does not subtract three large sums from each other, so
    float32 keeps its digits. Measured on Ishigami with a dummy fourth input
    at ``N = 128, 2048, 4096``: every float32 ``R2`` is within ``3e-6``
    relative of float64, the dummy input (``R2`` down to ``6e-4``)
    included, and a reorder of the sample rows moves an index by at most
    ``2e-7``. That is why :func:`analyze` does not warn in single precision.

    Args:
        Kc: Centred kernel matrix, shape ``(N, N)``.
        Lc: Centred kernel matrix, shape ``(N, N)``.

    Returns:
        Scalar HSIC value.
    """
    n = Kc.shape[-1]
    return jnp.sum(Kc * Lc) / jnp.asarray(n, dtype=Kc.dtype) ** 2


class _Plan(NamedTuple):
    """How many ``(N, N)`` input-side matrices each pass keeps live at once.

    Attributes:
        input_chunk: Centred input kernels held during the R2 and
            permutation pass.
        total_chunk: Complement products held during the total-index pass.
    """

    input_chunk: int
    total_chunk: int


_R_FIXED_MATRICES = 4
"""Matrices the first-order pass holds besides the input chunk: the centred
output kernel, one kernel being built and its transients. Calibrated against
the peak process RSS at ``N = 4096``, which sits about two matrices above
XLA's own temp-buffer estimate."""

_T_FIXED_MATRICES = 6
"""Matrices the total-index pass holds besides the chunk: the running prefix,
one augmented kernel being built, the full product, the centred output kernel
and transients. Calibrated against the peak process RSS at ``N = 4096``."""


def _plan(d: int, n: int, itemsize: int) -> _Plan:
    """Size the chunks of the two passes from the memory budget.

    The working set of each pass is a fixed number of ``(N, N)`` matrices plus
    one per chunk member, so the budget of :func:`get_memory_budget` gives the
    chunk widths directly. Every width is at least 1: the estimator needs a
    few ``(N, N)`` matrices whatever the budget, and a budget below that is
    exceeded by those few.

    Args:
        d: Number of inputs.
        n: Number of samples.
        itemsize: Bytes per float.

    Returns:
        The chunk widths.
    """
    n_mat = get_memory_budget() // (n * n * itemsize)
    input_chunk = max(1, min(d, n_mat - _R_FIXED_MATRICES))
    total_chunk = max(1, min(d, n_mat - _T_FIXED_MATRICES))
    return _Plan(input_chunk, total_chunk)


def _pad_columns(X: Array, medians: Array, chunk: int) -> tuple[Array, Array, int]:
    """Pad the inputs with constant columns to a whole number of chunks.

    A constant column has an all-ones kernel, whose centred form is exactly
    0. So a padded column adds an exact 0 to the first-order pass, whose
    results for it are dropped anyway, and an augmented kernel of exactly 1
    to the total-index products, which leaves them unchanged.

    Args:
        X: Inputs on the unit interval, shape ``(N, D)``.
        medians: Median-heuristic squared bandwidths, shape ``(D,)``.
        chunk: Inputs per chunk.

    Returns:
        ``(X_pad, medians_pad, n_chunks)``.
    """
    n, d = X.shape
    n_chunks = -(-d // chunk)
    pad = n_chunks * chunk - d
    X_pad = jnp.concatenate([X, jnp.zeros((n, pad), dtype=X.dtype)], axis=1)
    medians_pad = jnp.concatenate([medians, jnp.ones((pad,), dtype=medians.dtype)])
    return X_pad, medians_pad, n_chunks


def _centered_kernel(x: Array, median_sq: Array, bandwidth: float) -> Array:
    """Build the centred Gaussian kernel ``HKH`` of one variable.

    The kernel is symmetric, so its column means are its row means and one
    reduction gives both. Nothing forces the raw kernel to be written out:
    XLA may fuse the exponential into the reduction and into whatever reads
    the result.

    Args:
        x: Values for one variable, shape ``(N,)``.
        median_sq: Median-heuristic squared bandwidth of ``x``, a scalar.
        bandwidth: Positive multiplier on the median-heuristic bandwidth.

    Returns:
        Centred kernel matrix, shape ``(N, N)``.
    """
    K = _kernel_from_median(x, median_sq, bandwidth)
    r = jnp.mean(K, axis=1)
    return K - r[:, None] - r[None, :] + jnp.mean(r)


def _column(X: Array, medians: Array, j: Array) -> tuple[Array, Array]:
    """Read column ``j`` and its median behind an optimisation barrier.

    Inside a loop ``j`` is traced. Without the barrier XLA fuses the dynamic
    slice into the ``(N, N)`` exponential that reads it, and the fused loop
    no longer vectorises: building one kernel inside a ``fori_loop`` measured
    56 ms at ``N = 4096`` against 8 ms with the column read out first.

    Args:
        X: Samples, shape ``(N, C)``.
        medians: Median-heuristic squared bandwidths, shape ``(C,)``.
        j: Column index (may be traced).

    Returns:
        ``(x, median_sq)``: the column, shape ``(N,)``, and its median.
    """
    return jax.lax.optimization_barrier((X[:, j], medians[j]))


def _out_kernel(Y_cols: Array, y_med: Array, t: Array, bandwidth: float) -> Array:
    """Build the centred kernel of output slice ``t``.

    The barrier stops XLA fusing the kernel build into the reductions that
    read it, which would recompute the exponential inside each of them. The
    kernel is read many times per slice, so materialising it once is cheaper.

    Args:
        Y_cols: Output columns, shape ``(N, total)``.
        y_med: Median-heuristic squared bandwidths, shape ``(total,)``.
        t: Slice index (may be traced).
        bandwidth: Multiplier on the median-heuristic bandwidth.

    Returns:
        Centred output kernel, shape ``(N, N)``.
    """
    y, med = _column(Y_cols, y_med, t)
    return jax.lax.optimization_barrier(_centered_kernel(y, med, bandwidth))


@jax.jit
def _medians(X_unit: Array, Y_cols: Array) -> tuple[Array, Array]:
    """Return the median-heuristic squared bandwidths of every input and output.

    One vmapped selection instead of one inlined copy per variable halves the
    compile time.

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.

    Returns:
        ``(x_med, y_med)``, shapes ``(D,)`` and ``(total,)``.
    """
    pick = jax.vmap(_median_bandwidth_sq, in_axes=1)
    return pick(X_unit), pick(Y_cols)


@partial(jax.jit, static_argnames=("n_perms", "bandwidth", "chunk"))
def _first_order_pass(
    X_unit: Array,
    Y_cols: Array,
    x_med: Array,
    y_med: Array,
    keys: Array,
    *,
    n_perms: int,
    bandwidth: float,
    chunk: int,
) -> tuple[Array, Array, Array, Array]:
    """Compute the HSIC values behind R2 and the permutation test.

    The inputs are taken ``chunk`` at a time. A chunk of centred input
    kernels is built once and shared by every output slice and every
    permutation, so the input kernels are never rebuilt per slice and never
    all held at once. For each slice the centred output kernel is built once;
    each permutation reads it through two ``take`` gathers that XLA fuses
    into the reduction, so no permuted ``(N, N)`` matrix is written.

    Centring commutes with a permutation, so the permuted centred kernel is
    the centred permuted kernel, and each HSIC value is one product sum
    (:func:`_hsic_centered`). Slice ``t`` splits ``keys[t]`` into its
    ``n_perms`` permutation keys, as a scalar analysis of that slice does, so
    the draws do not depend on the chunking.

    Peak memory is about ``chunk + 4`` matrices of shape ``(N, N)``
    (``_R_FIXED_MATRICES``, measured as process RSS).

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.
        x_med: Input median-heuristic squared bandwidths, shape ``(D,)``.
        y_med: Output median-heuristic squared bandwidths, shape ``(total,)``.
        keys: One PRNG key per output slice, shape ``(total,)``.
        n_perms: Number of permutations (static).
        bandwidth: Multiplier on the median-heuristic bandwidth (static).
        chunk: Input kernels held at once (static).

    Returns:
        ``(hsic_yy, hsic_xx, hsic_xy, counts)``: the output self-HSIC
        ``(total,)``, the input self-HSIC ``(D,)``, ``HSIC(x_d, y_t)`` of
        shape ``(total, D)``, and the number of permutations whose HSIC
        reached the unpermuted one, ``(total, D)``.
    """
    n, d = X_unit.shape
    total = Y_cols.shape[1]

    def self_hsic(t: Array) -> Array:
        Lc = _out_kernel(Y_cols, y_med, t, bandwidth)
        return _hsic_centered(Lc, Lc)

    hsic_yy = jax.lax.map(self_hsic, jnp.arange(total))
    Xr, mr, n_chunks = _pad_columns(X_unit, x_med, chunk)
    perm_keys = jax.vmap(lambda k: jax.random.split(k, n_perms))(keys)

    def one_chunk(start: Array) -> tuple[Array, Array, Array]:
        """Self-HSIC, HSIC and exceedance counts of one input chunk."""

        def in_kernel(j: Array) -> Array:
            return _centered_kernel(*_column(Xr, mr, j), bandwidth)

        Kc = jax.lax.map(in_kernel, start + jnp.arange(chunk))
        n_sq = jnp.asarray(n, dtype=Kc.dtype) ** 2

        def score(Lc: Array) -> Array:
            """HSIC of every kernel of the chunk against one centred kernel."""
            return jnp.sum(Kc * Lc[None], axis=(1, 2)) / n_sq

        hxx = jax.lax.map(lambda K: _hsic_centered(K, K), Kc)

        def one_slice(args: tuple[Array, Array]) -> tuple[Array, Array]:
            t, pkeys = args
            Lc = _out_kernel(Y_cols, y_med, t, bandwidth)
            raw = score(Lc)

            def body(counts: Array, pkey: Array) -> tuple[Array, None]:
                """Add one permutation draw's exceedance counts to the running total."""
                perm = jax.random.permutation(pkey, n)
                # Left unmaterialised on purpose: XLA fuses the two takes
                # into the reduction in score(), which measured 4x faster
                # than writing the permuted kernel out and reading it back.
                Lp = jnp.take(jnp.take(Lc, perm, axis=0), perm, axis=1)
                return counts + (score(Lp) >= raw).astype(counts.dtype), None

            counts, _ = jax.lax.scan(body, jnp.zeros((chunk,), dtype=jnp.int32), pkeys)
            return raw, counts

        raw, counts = jax.lax.map(one_slice, (jnp.arange(total), perm_keys))
        return hxx, raw, counts

    hxx, raw, counts = jax.lax.map(one_chunk, jnp.arange(n_chunks) * chunk)
    hxx = hxx.reshape(-1)[:d]
    raw = jnp.moveaxis(raw, 0, 1).reshape(total, -1)[:, :d]
    counts = jnp.moveaxis(counts, 0, 1).reshape(total, -1)[:, :d]
    return hsic_yy, hxx, raw, counts


def _total_streamed(
    X_unit: Array,
    Y_cols: Array,
    x_med: Array,
    y_med: Array,
    bandwidth: float,
    chunk: int,
) -> tuple[Array, Array]:
    """Compute the HSIC values behind the total index, chunk by chunk.

    Total HSIC uses augmented kernels ``A_d = 1 + center(K_d)`` (Larsen &
    Alexanderian, 2026): the constant term makes the product over inputs
    carry every interaction order, which correct total indices need. For a
    chunk of inputs, the product of the augmented kernels of the earlier
    chunks is carried over and the product of the later ones is streamed one
    kernel at a time. A prefix-suffix pass inside the chunk
    then writes each complement product ``prod_{e != d} A_e`` in place.
    Each augmented kernel is rebuilt from ``X`` when it is needed instead of
    being stored, so peak memory is about ``chunk + 6`` matrices of shape
    ``(N, N)`` (``_T_FIXED_MATRICES``) and the cost is about ``D^2 / (2 chunk)``
    kernel builds.

    The output kernel is centred, so the products need not be:
    ``sum(C * HLH) = sum(HCH * HLH)``.

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.
        x_med: Input median-heuristic squared bandwidths, shape ``(D,)``.
        y_med: Output median-heuristic squared bandwidths, shape ``(total,)``.
        bandwidth: Multiplier on the median-heuristic bandwidth (static).
        chunk: Complement products held at once (static).

    Returns:
        ``(hsic_compls, hsic_full)``: ``HSIC(prod_{e != d} A_e, y_t)`` of
        shape ``(total, D)`` and ``HSIC(prod_e A_e, y_t)`` of shape
        ``(total,)``.
    """
    n, d = X_unit.shape
    total = Y_cols.shape[1]
    Xt, mt, n_chunks = _pad_columns(X_unit, x_med, chunk)
    n_pad = n_chunks * chunk

    def aug(j: Array) -> Array:
        """Augmented kernel ``1 + center(K_j)``; exactly ones for a padded column."""
        return 1.0 + _centered_kernel(*_column(Xt, mt, j), bandwidth)

    def one_chunk(before: Array, start: Array) -> tuple[Array, tuple[Array, Array]]:
        """Complement and full-product HSIC of one input chunk, every slice.

        ``before`` is the product of the augmented kernels of every earlier
        chunk, carried from chunk to chunk; only the product after the chunk
        is streamed. The forward pass is never differentiated (see
        :func:`_total_core`), so the dynamic loop bound is fine.
        """
        ones = jnp.ones((n, n), dtype=Xt.dtype)
        after = jax.lax.fori_loop(start + chunk, n_pad, lambda k, acc: acc * aug(k), ones)
        buf = jnp.zeros((chunk, n, n), dtype=Xt.dtype).at[chunk - 1].set(after)

        def back(i: int, buf: Array) -> Array:
            """Write the suffix product (times ``out``) into slot ``chunk - 2 - i``."""
            k = chunk - 2 - i
            return buf.at[k].set(buf[k + 1] * aug(start + k + 1))

        buf = jax.lax.fori_loop(0, chunk - 1, back, buf)

        def fwd(k: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            """Multiply the running prefix into slot ``k``, then extend it."""
            buf, prefix = carry
            return buf.at[k].set(prefix * buf[k]), prefix * aug(start + k)

        C, through = jax.lax.fori_loop(0, chunk, fwd, (buf, before))
        F = C[chunk - 1] * aug(start + chunk - 1)

        def one_slice(t: Array) -> tuple[Array, Array]:
            Lc = _out_kernel(Y_cols, y_med, t, bandwidth)
            hc = jax.lax.map(lambda Ck: _hsic_centered(Ck, Lc), C)
            return hc, _hsic_centered(F, Lc)

        return through, jax.lax.map(one_slice, jnp.arange(total))

    ones = jnp.ones((n, n), dtype=Xt.dtype)
    _, (hc, hf) = jax.lax.scan(one_chunk, ones, jnp.arange(n_chunks) * chunk)
    hsic_compls = jnp.moveaxis(hc, 0, 1).reshape(total, -1)[:, :d]
    # Every chunk forms the same full product; chunk 0's is the one used.
    return hsic_compls, hf[0]


def _total_listed(
    X_unit: Array, Y_cols: Array, x_med: Array, y_med: Array, bandwidth: float
) -> tuple[Array, Array]:
    """Compute the same values as :func:`_total_streamed` with every kernel stored.

    This is the form the backward pass differentiates. Reverse mode through
    the streamed loops saves the running product of every step of every
    chunk, ``n_chunks * D`` matrices, which grows without bound as the
    budget shrinks. Differentiating this form costs a fixed few ``D``
    matrices instead: the augmented kernels and their prefix, suffix and
    complement products. Each slice is checkpointed, so its output kernel is
    rebuilt in the backward pass rather than kept for every slice.

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.
        x_med: Input median-heuristic squared bandwidths, shape ``(D,)``.
        y_med: Output median-heuristic squared bandwidths, shape ``(total,)``.
        bandwidth: Multiplier on the median-heuristic bandwidth.

    Returns:
        Same as :func:`_total_streamed`.
    """
    d = X_unit.shape[1]
    total = Y_cols.shape[1]
    aug = [1.0 + _centered_kernel(X_unit[:, e], x_med[e], bandwidth) for e in range(d)]
    prefix = [jnp.ones_like(aug[0])] * d
    for i in range(1, d):
        prefix[i] = prefix[i - 1] * aug[i - 1]
    suffix = [jnp.ones_like(aug[0])] * d
    for i in range(d - 2, -1, -1):
        suffix[i] = suffix[i + 1] * aug[i + 1]
    C = jnp.stack([prefix[e] * suffix[e] for e in range(d)])
    F = prefix[-1] * aug[-1]

    @jax.checkpoint
    def one_slice(t: Array) -> tuple[Array, Array]:
        Lc = _out_kernel(Y_cols, y_med, t, bandwidth)
        hc = jnp.sum(C * Lc[None], axis=(1, 2)) / jnp.asarray(Lc.shape[0], Lc.dtype) ** 2
        return hc, _hsic_centered(F, Lc)

    return jax.lax.map(one_slice, jnp.arange(total))


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _total_core(
    X_unit: Array, Y_cols: Array, x_med: Array, y_med: Array, bandwidth: float, chunk: int
) -> tuple[Array, Array]:
    """Streamed forward values, with the gradient of :func:`_total_listed`."""
    return _total_streamed(X_unit, Y_cols, x_med, y_med, bandwidth, chunk)


def _total_core_fwd(
    X_unit: Array, Y_cols: Array, x_med: Array, y_med: Array, bandwidth: float, chunk: int
):
    """Forward rule: keep only the inputs as residuals."""
    out = _total_streamed(X_unit, Y_cols, x_med, y_med, bandwidth, chunk)
    return out, (X_unit, Y_cols, x_med, y_med)


def _total_core_bwd(bandwidth: float, chunk: int, residuals, cotangents):
    """Backward rule: differentiate the stored-kernel form."""
    del chunk
    _, vjp = jax.vjp(lambda X, Y, xm, ym: _total_listed(X, Y, xm, ym, bandwidth), *residuals)
    return vjp(cotangents)


_total_core.defvjp(_total_core_fwd, _total_core_bwd)


@partial(jax.jit, static_argnames=("bandwidth", "chunk"))
def _total_pass(
    X_unit: Array,
    Y_cols: Array,
    x_med: Array,
    y_med: Array,
    *,
    bandwidth: float,
    chunk: int,
) -> tuple[Array, Array]:
    """Compute the HSIC values behind the total index.

    The values come from :func:`_total_streamed`, whose memory follows the
    budget. The gradient comes from :func:`_total_listed` (see there why).

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.
        x_med: Input median-heuristic squared bandwidths, shape ``(D,)``.
        y_med: Output median-heuristic squared bandwidths, shape ``(total,)``.
        bandwidth: Multiplier on the median-heuristic bandwidth (static).
        chunk: Complement products held at once (static).

    Returns:
        ``(hsic_compls, hsic_full)``, shapes ``(total, D)`` and ``(total,)``.
    """
    return _total_core(X_unit, Y_cols, x_med, y_med, bandwidth, chunk)


@partial(jax.jit, static_argnames=("n_perms",))
def _combine(
    hsic_yy: Array,
    hsic_xx: Array,
    hsic_xy: Array,
    counts: Array,
    hsic_compls: Array,
    hsic_full: Array,
    *,
    n_perms: int,
) -> tuple[Array, Array, Array, Array]:
    """Turn the HSIC values of both passes into the reported indices.

    Args:
        hsic_yy: Output self-HSIC, shape ``(total,)``.
        hsic_xx: Input self-HSIC, shape ``(D,)``.
        hsic_xy: ``HSIC(x_d, y_t)``, shape ``(total, D)``.
        counts: Permutation exceedance counts, shape ``(total, D)``.
        hsic_compls: Complement-product HSIC, shape ``(total, D)``.
        hsic_full: Full-product HSIC, shape ``(total,)``.
        n_perms: Number of permutations (static).

    Returns:
        ``(R2, T, p_values, hsic_raw)``, each of shape ``(total, D)``.
    """
    eps = jnp.finfo(hsic_xy.dtype).eps
    zero_var = (hsic_yy < eps)[:, None]
    # A constant input column has an all-ones kernel whose centred form is
    # exactly 0, and it provably depends on no output. Those parameters
    # report exactly 0 (matching PAWN), not the NaN of a degenerate output,
    # so only zero_var (the output's own degeneracy) yields NaN.
    zero_input = (hsic_xx < eps)[None, :]
    denoms = jnp.sqrt(jnp.maximum(hsic_xx[None, :] * hsic_yy[:, None], eps**2))
    r2 = jnp.where(zero_var, jnp.nan, jnp.where(zero_input, 0.0, hsic_xy / denoms))
    full_safe = jnp.where(jnp.abs(hsic_full) < eps, jnp.nan, hsic_full)[:, None]
    t_hsic = jnp.where(zero_var, jnp.nan, 1.0 - hsic_compls / full_safe)
    p_vals = (counts.astype(hsic_xy.dtype) + 1.0) / (n_perms + 1.0)
    # A degenerate output has no dependence to test: R2 and T are NaN, so the
    # p-value is NaN too, not a confident 1.0.
    p_vals = jnp.where(zero_var, jnp.nan, p_vals)
    return r2, t_hsic, p_vals, hsic_xy


def _run_kernel(
    X_unit: Array, Y_cols: Array, keys: Array, n_perms: int, bandwidth: float, plan: _Plan
) -> tuple[Array, Array, Array, Array]:
    """Run both passes for every output slice.

    The passes are separate compiled calls on purpose. In one program XLA
    keeps both working sets live at once (measured: 19 ``(N, N)`` matrices
    where each pass alone needs 10 or 13), because buffers inside different
    loops are not shared. Under an outer ``jit`` they are inlined into one
    program again, which is correct but forgoes that saving.

    Args:
        X_unit: Inputs mapped to the unit interval, shape ``(N, D)``.
        Y_cols: Output columns, shape ``(N, total)``.
        keys: One PRNG key per output slice, shape ``(total,)``.
        n_perms: Number of permutations.
        bandwidth: Multiplier on the median-heuristic bandwidth.
        plan: Chunk widths from :func:`_plan`.

    Returns:
        ``(R2, T, p_values, hsic_raw)``, each of shape ``(total, D)``.
    """
    x_med, y_med = _medians(X_unit, Y_cols)
    hsic_yy, hsic_xx, hsic_xy, counts = _first_order_pass(
        X_unit,
        Y_cols,
        x_med,
        y_med,
        keys,
        n_perms=n_perms,
        bandwidth=bandwidth,
        chunk=plan.input_chunk,
    )
    hsic_compls, hsic_full = _total_pass(
        X_unit, Y_cols, x_med, y_med, bandwidth=bandwidth, chunk=plan.total_chunk
    )
    return _combine(hsic_yy, hsic_xx, hsic_xy, counts, hsic_compls, hsic_full, n_perms=n_perms)


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


_NEAR_COLLAPSE_RATIO = 1e-6
"""Warn when the median non-zero squared distance is below this share of the variance."""

_COLLAPSE_CHECK_MAX_ROWS = 2048
"""Rows used by the host near-collapse check (an even stride subsample)."""


def _nonzero_pair_sq_dists(col: np.ndarray) -> np.ndarray:
    """Return the non-zero squared distances over all pairs ``i < j`` of a column.

    The pairs are built row by row into one flat buffer. This avoids the full
    ``(n, n)`` matrix and the ``triu_indices`` arrays.

    Args:
        col: Finite samples, shape ``(n,)``.

    Returns:
        Squared distances above zero, in no particular order.
    """
    n = col.size
    buf = np.empty(n * (n - 1) // 2, dtype=np.float64)
    pos = 0
    for i in range(n - 1):
        m = n - 1 - i
        np.subtract(col[i + 1 :], col[i], out=buf[pos : pos + m])
        pos += m
    np.square(buf, out=buf)
    return buf[buf > 0]


def _collapsed_median_labels(values: np.ndarray) -> list[int]:
    """Return the columns whose median heuristic nearly collapses.

    The heuristic takes the median over the non-zero pairwise squared
    distances, so exact atoms are handled. Near-ties are not: a column where
    most values sit within a tiny distance of each other (a spike plus
    noise of order 1e-6) makes that median tiny against the spread of the
    column, and the kernel then separates only the near-equal values. A
    column is flagged when the median non-zero squared distance is below
    ``1e-6`` times its variance. A constant column is left alone, since it
    has its own warning. Large samples are checked on an even stride
    subsample of at most 2048 rows.

    Args:
        values: Samples, shape ``(N, C)``.

    Returns:
        Indices of the affected columns.
    """
    n = values.shape[0]
    if n > _COLLAPSE_CHECK_MAX_ROWS:
        values = values[:: -(-n // _COLLAPSE_CHECK_MAX_ROWS)]
    bad = []
    for c in range(values.shape[1]):
        col = values[:, c].astype(np.float64)
        col = col[np.isfinite(col)]
        var = float(np.var(col)) if col.size else 0.0
        if var == 0.0:
            continue
        nonzero = _nonzero_pair_sq_dists(col)
        if nonzero.size and float(np.median(nonzero)) < _NEAR_COLLAPSE_RATIO * var:
            bad.append(c)
    return bad


def _warn_collapsed_median(problem: Problem, X: Array, Y3: Array, stacklevel: int) -> None:
    """Warn when the median heuristic nearly collapses for an input or an output slice.

    The heuristic ignores exact ties, so atoms (a spike of zeros) are fine.
    Near-ties are not: when the median non-zero squared distance is below
    1e-6 times the variance, the Gaussian kernel reads about 1 for
    near-equal values and 0 for the rest, and the indices react strongly to
    tiny perturbations. The check reads concrete values, so it lives here and
    not in :func:`indices`. ``bandwidth`` multiplies the heuristic, so it
    cannot rescue a collapsed one.

    Args:
        problem: Problem definition, used to name the parameters.
        X: Input samples, shape ``(N, D)``.
        Y3: Output samples, shape ``(N, T, K)``.
        stacklevel: Stack level that points the warning at the caller's line.

    Warns:
        JaxgsaWarning: If an input or an output slice has a median non-zero
            squared distance below 1e-6 times its variance.
    """
    x_host = np.asarray(X)
    y_host = np.asarray(Y3)
    n, n_t, n_k = y_host.shape
    parts = [f"parameter {problem.names[d]!r}" for d in _collapsed_median_labels(x_host)]
    for c in _collapsed_median_labels(y_host.reshape(n, n_t * n_k)):
        t, k = divmod(c, n_k)
        parts.append(f"output slice (t={t}, k={k})")
    if not parts:
        return
    warnings.warn(
        f"jaxgsa.hsic: the median heuristic failed for {', '.join(parts)}: most values lie "
        "within a tiny distance of each other (near-ties), so the median non-zero squared "
        "distance is below 1e-6 times the variance and the kernel only tests whether two "
        "values are nearly equal. The indices then react strongly to tiny changes in the "
        "other samples. The bandwidth multiplier cannot fix this. Round the near-equal "
        "values to exact ties, analyse the non-zero part of the variable separately, or "
        "use a method that does not rely on a bandwidth.",
        stacklevel=stacklevel,
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
    zero-variance warning, no constant-input warning, no sample-count
    check, no :class:`jaxgsa.hsic.HSICResult`, and no read of any array value
    on the host. So it composes with ``jax.jit``, ``jax.vmap``, ``jax.grad``
    and ``jax.jacrev``, which :func:`analyze` cannot, because a policy
    decision needs a concrete value and a tracer has none.

    Every branch here is on a shape or on a Python scalar. ``n_perms`` and
    ``bandwidth`` are both static: they are read while the graph is being
    traced and never at run time. ``bandwidth`` scales the median-heuristic
    width and is baked into the compiled passes, which are keyed on it.
    Changing it recompiles; it costs no device-side conditional. How many
    input kernels a pass holds at once is also static: it follows from the
    shapes and the memory budget (:func:`_plan`), and is part of the same
    cache key.

    Peak memory follows ``jaxgsa.config.set_memory_budget``, as described
    under :func:`analyze`. Under an outer ``jit`` the two passes are compiled
    into one program, and XLA may then hold both working sets at once.

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
        ValueError: If x64 is off and ``N > 65,535``. The median heuristic
            counts all ``N^2`` sample pairs, and without x64 that count has
            only 32 bits. Enable x64 or thin the sample.
        TypeError: If the float width of ``X`` or ``Y`` has no unsigned
            integer counterpart for the median-heuristic selection.
    """
    key = as_key(key, "hsic")
    _check_sample_size(X.shape[0])
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
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(total))
    itemsize = jnp.result_type(X_unit.dtype, Y_cols.dtype).itemsize
    plan = _plan(D, N, itemsize)
    r2_all, t_all, p_all, raw_all = _run_kernel(X_unit, Y_cols, keys, n_perms, bandwidth, plan)
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

    **Memory is quadratic in N and follows the global memory budget.** The
    estimator works on dense ``(N, N)`` kernels, and it holds only as many
    input kernels at once as ``jaxgsa.config.set_memory_budget`` allows,
    building the rest again when they are needed. A smaller budget costs
    time, not accuracy. The estimator always needs about six ``(N, N)``
    matrices, so at a large ``N`` a small budget is exceeded by that
    floor; then thin ``N`` (HSIC converges quickly in N). There is no
    per-call batching keyword.

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
            Also raised if x64 is off and ``N > 65,535``: the median
            heuristic counts all ``N^2`` sample pairs, and without x64 that
            count has only 32 bits. Enable x64 with
            ``jax.config.update("jax_enable_x64", True)`` or thin the sample.

    Warns:
        JaxgsaWarning: If an output slice has zero variance (all indices NaN),
            if a parameter is constant in the sample (its R2 is exactly
            0).
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
    _check_sample_size(X.shape[0])
    # Checked after the preamble and the shape contract, so a problem this
    # method refuses outright (categorical parameters, a sample too small)
    # still reports that, and not a missing key.
    if key is None:
        raise ValueError("key is required for the permutation test")
    _warn_constant_inputs(problem, X)
    _warn_collapsed_median(problem, X, ctx.Y3, _user_stacklevel())

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
