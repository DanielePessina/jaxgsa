"""HSIC analysis: kernel-based sensitivity indices.

Computes R2-HSIC (normalized first-order) and Total HSIC indices from
arbitrary (X, Y) sample pairs using Gaussian RBF kernels with the
median heuristic for bandwidth selection.

The V-statistic HSIC estimator with centering matrix H is equivalent
to using augmented kernels k*(x,x') = 1 + k(x,x') because HK*H = HKH.
This means the theoretical guarantees from Larsen & Alexanderian (2026)
— monotonicity under marginalization — hold without explicitly
constructing augmented kernels.

References:
    Gretton et al. (2005). JMLR 6:2075-2129.
    Da Veiga (2015). Rel. Eng. Sys. Safety 142:346-362.
    Larsen & Alexanderian (2026). arXiv:2603.00849.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from gsax._normalization import _prenormalize_outputs, _prepare_Y, _warn_zero_variance_slices
from gsax._transforms import cdf_to_unit_interval
from gsax.hsic._result import HSICResult
from gsax.problem import Problem


def _median_bandwidth_sq(dists_sq: Array) -> Array:
    """Compute squared bandwidth from a pairwise squared-distance matrix.

    Uses the upper triangle (excluding diagonal) to avoid bias from
    the N diagonal zeros per the standard median heuristic definition.

    Args:
        dists_sq: (N, N) pairwise squared distances.

    Returns:
        Scalar median of off-diagonal squared distances, floored at 1e-20.
    """
    n = dists_sq.shape[0]
    idx = jnp.triu_indices(n, k=1)
    upper = dists_sq[idx]
    return jnp.maximum(jnp.median(upper), 1e-20)


def _build_kernel_median(x: Array) -> Array:
    """Build Gaussian RBF kernel matrix with median heuristic bandwidth.

    Computes pairwise squared distances once and derives both the
    bandwidth (via the median heuristic on upper-triangle distances)
    and the kernel matrix.

    Args:
        x: 1-D array of N values.

    Returns:
        (N, N) kernel matrix.
    """
    dists_sq = (x[:, None] - x[None, :]) ** 2
    median_sq = _median_bandwidth_sq(dists_sq)
    return jnp.exp(-dists_sq / (2.0 * median_sq))


def _build_kernel_fixed(x: Array, sigma: Array) -> Array:
    """Build Gaussian RBF kernel matrix with a fixed bandwidth.

    Args:
        x: 1-D array of N values.
        sigma: Kernel bandwidth.

    Returns:
        (N, N) kernel matrix.
    """
    dists_sq = (x[:, None] - x[None, :]) ** 2
    return jnp.exp(-dists_sq / (2.0 * sigma**2))


def _build_kernel_chunked(x: Array, sigma: Array, chunk_size: int) -> Array:
    """Build Gaussian kernel matrix in row blocks to limit peak memory.

    Args:
        x: 1-D array of N values.
        sigma: Kernel bandwidth.
        chunk_size: Number of rows per block.

    Returns:
        (N, N) kernel matrix.
    """
    N = x.shape[0]
    rows = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        block = jnp.exp(-((x[start:end, None] - x[None, :]) ** 2) / (2.0 * sigma**2))
        rows.append(block)
    return jnp.concatenate(rows, axis=0)


def _median_bandwidth(x: Array) -> Array:
    """Compute bandwidth via the median heuristic (upper triangle only).

    Args:
        x: 1-D array of N values.

    Returns:
        Scalar bandwidth sigma.
    """
    dists_sq = (x[:, None] - x[None, :]) ** 2
    return jnp.sqrt(_median_bandwidth_sq(dists_sq))


def _hsic_v(K: Array, L: Array) -> Array:
    """Biased V-statistic HSIC estimator.

    Uses the efficient trace formula avoiding explicit centering matrices:
        HSIC = U/n^2 - 2V/n^3 + W/n^4
    where U = sum(K*L), V = sum(colsums(K)*colsums(L)), W = sum(K)*sum(L).

    Precision follows the input kernel dtype. When ``jax_enable_x64``
    is active, kernels are float64 and the subtraction avoids
    cancellation. In float32 mode the bias is negligible for
    typical sample sizes (N < 10000).

    Args:
        K: (N, N) kernel matrix.
        L: (N, N) kernel matrix.

    Returns:
        Scalar HSIC value.
    """
    n = K.shape[0]
    orig_dtype = K.dtype
    n_f = jnp.asarray(n, dtype=orig_dtype)
    Kc = K
    Lc = L
    U = jnp.sum(Kc * Lc)
    col_K = jnp.sum(Kc, axis=0)
    col_L = jnp.sum(Lc, axis=0)
    V = jnp.dot(col_K, col_L)
    W = jnp.sum(Kc) * jnp.sum(Lc)
    result = U / n_f**2 - 2.0 * V / n_f**3 + W / n_f**4
    return result.astype(orig_dtype)


def _build_input_kernels(
    X_unit: Array,
    bandwidth: float | None,
    chunk_size: int | None,
) -> list[Array]:
    """Build kernel matrices for all D input dimensions.

    Args:
        X_unit: (N, D) inputs on [0, 1].
        bandwidth: Fixed bandwidth or None for median heuristic.
        chunk_size: Block size for kernel matrix, or None.

    Returns:
        List of D kernel matrices, each (N, N).
    """
    N, D = X_unit.shape
    use_chunked = chunk_size is not None and chunk_size > 0 and N > chunk_size

    Ks: list[Array] = []
    for d in range(D):
        xi = X_unit[:, d]
        if bandwidth is None and not use_chunked:
            Ki = _build_kernel_median(xi)
        elif bandwidth is not None and not use_chunked:
            Ki = _build_kernel_fixed(xi, jnp.asarray(bandwidth, dtype=xi.dtype))
        else:
            if bandwidth is None:
                sigma = _median_bandwidth(xi)
            else:
                sigma = jnp.asarray(bandwidth, dtype=xi.dtype)
            assert chunk_size is not None
            Ki = _build_kernel_chunked(xi, sigma, chunk_size)
        Ks.append(Ki)
    return Ks


def _complement_kernels(Ks: list[Array]) -> list[Array]:
    """Build complement product kernels via prefix-suffix products.

    For each dimension d, the complement kernel is the Hadamard product
    of all input kernels except d. This avoids element-wise division
    which is numerically unsafe when kernel entries underflow.

    Args:
        Ks: List of D kernel matrices, each (N, N).

    Returns:
        List of D complement kernel matrices, each (N, N).
    """
    D = len(Ks)
    if D == 1:
        return [jnp.ones_like(Ks[0])]

    # prefix[i] = product of Ks[0..i-1], suffix[i] = product of Ks[i+1..D-1]
    prefix = [jnp.ones_like(Ks[0])] * D
    for i in range(1, D):
        prefix[i] = prefix[i - 1] * Ks[i - 1]

    suffix = [jnp.ones_like(Ks[0])] * D
    for i in range(D - 2, -1, -1):
        suffix[i] = suffix[i + 1] * Ks[i + 1]

    return [prefix[d] * suffix[d] for d in range(D)]


def _compute_slice(
    Ks: list[Array],
    y_col: Array,
    bandwidth: float | None,
    key: Array,
    n_perms: int,
    chunk_size: int | None,
) -> tuple[Array, Array, Array, Array]:
    """Compute HSIC indices for a single (t, k) output slice.

    Uses pre-built input kernel matrices and builds only the output
    kernel per slice.

    Args:
        Ks: Pre-built input kernel matrices, one per input dimension.
        y_col: (N,) single output column.
        bandwidth: Fixed bandwidth or None for median heuristic.
        key: PRNG key for permutation test.
        n_perms: Number of permutations.
        chunk_size: Block size for kernel matrix, or None.

    Returns:
        (R2_HSIC, T_HSIC, p_values, hsic_raw) each of shape (D,).
    """
    N = y_col.shape[0]
    D = len(Ks)
    use_chunked = chunk_size is not None and chunk_size > 0 and N > chunk_size

    if bandwidth is None and not use_chunked:
        L = _build_kernel_median(y_col)
    elif bandwidth is not None and not use_chunked:
        L = _build_kernel_fixed(y_col, jnp.asarray(bandwidth, dtype=y_col.dtype))
    else:
        if bandwidth is None:
            sigma = _median_bandwidth(y_col)
        else:
            sigma = jnp.asarray(bandwidth, dtype=y_col.dtype)
        assert chunk_size is not None
        L = _build_kernel_chunked(y_col, sigma, chunk_size)

    hsic_yy = _hsic_v(L, L)

    hsic_xys: list[Array] = []
    r2s: list[Array] = []
    K_full = jnp.ones((N, N), dtype=Ks[0].dtype)

    for d in range(D):
        Ki = Ks[d]
        hsic_xy = _hsic_v(Ki, L)
        hsic_xx = _hsic_v(Ki, Ki)
        denom = jnp.sqrt(jnp.maximum(hsic_xx * hsic_yy, 1e-30))

        hsic_xys.append(hsic_xy)
        r2s.append(hsic_xy / denom)
        K_full = K_full * Ki

    hsic_xys_arr = jnp.stack(hsic_xys)
    r2_arr = jnp.stack(r2s)

    # Total HSIC via complement product kernels (no division)
    hsic_full = _hsic_v(K_full, L)
    hsic_full_safe = jnp.where(jnp.abs(hsic_full) < 1e-30, jnp.nan, hsic_full)

    K_compls = _complement_kernels(Ks)
    t_hsics = []
    for d in range(D):
        hsic_compl = _hsic_v(K_compls[d], L)
        t_hsics.append(1.0 - hsic_compl / hsic_full_safe)

    t_arr = jnp.stack(t_hsics)

    # Permutation test using cached kernel matrices (Phipson-Smyth correction)
    perm_keys = jax.random.split(key, n_perms)
    null_counts = jnp.zeros(D)
    for pkey in perm_keys:
        perm = jax.random.permutation(pkey, N)
        L_perm = L[perm][:, perm]
        perm_hsics = jnp.stack([_hsic_v(Ks[d], L_perm) for d in range(D)])
        null_counts = null_counts + (perm_hsics >= hsic_xys_arr).astype(jnp.float32)

    p_vals = (null_counts + 1.0) / (n_perms + 1.0)

    return r2_arr, t_arr, p_vals, hsic_xys_arr


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    seed: int = 0,
    bandwidth: float | None = None,
    chunk_size: int | None = None,
    prenormalize: bool = False,
) -> HSICResult:
    """Compute HSIC sensitivity indices.

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)`` in physical units.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
            For outputs with large magnitude, set ``prenormalize=True``
            to avoid float overflow in distance computation.
        n_perms: Number of permutations for p-value computation.
        seed: Random seed for permutation test reproducibility.
        bandwidth: Fixed kernel bandwidth. None uses the median heuristic.
        chunk_size: Block size for N x N kernel matrix computation.
            None computes the full matrix at once.
        prenormalize: If True, standardize Y before analysis.

    Returns:
        HSICResult with R2_HSIC, T_HSIC, p_values, and hsic_raw.

    Raises:
        ValueError: If X is not 2-D, column count doesn't match problem,
            row counts of X and Y differ, n_perms < 1, or bandwidth
            is non-positive / non-finite.
    """
    D = problem.num_vars
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")
    if bandwidth is not None and (bandwidth <= 0 or not math.isfinite(bandwidth)):
        raise ValueError(f"bandwidth must be positive and finite, got {bandwidth}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, D), got ndim={X.ndim}")
    if X.shape[1] != D:
        raise ValueError(f"X has {X.shape[1]} columns but problem has {D} parameters")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]} rows")

    _warn_zero_variance_slices(Y, problem.output_names)

    X_unit = cdf_to_unit_interval(X, problem)

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _N, T, K = Y_3d.shape

    if prenormalize:
        Y_3d, _, _, _ = _prenormalize_outputs(Y_3d)

    key = jax.random.key(seed)

    # Build input kernels once — they depend only on X_unit, not Y.
    Ks = _build_input_kernels(X_unit, bandwidth, chunk_size)

    r2_all = jnp.empty((T, K, D))
    t_all = jnp.empty((T, K, D))
    p_all = jnp.empty((T, K, D))
    raw_all = jnp.empty((T, K, D))

    for t in range(T):
        for k in range(K):
            subkey = jax.random.fold_in(key, t * K + k)
            y_col = Y_3d[:, t, k]

            r2, t_hsic, p_vals, raw = _compute_slice(
                Ks, y_col, bandwidth, subkey, n_perms, chunk_size
            )

            r2_all = r2_all.at[t, k].set(r2)
            t_all = t_all.at[t, k].set(t_hsic)
            p_all = p_all.at[t, k].set(p_vals)
            raw_all = raw_all.at[t, k].set(raw)

    if squeeze_time and squeeze_output:
        r2_all = r2_all[0, 0]
        t_all = t_all[0, 0]
        p_all = p_all[0, 0]
        raw_all = raw_all[0, 0]
    elif squeeze_time:
        r2_all = r2_all[0]
        t_all = t_all[0]
        p_all = p_all[0]
        raw_all = raw_all[0]

    return HSICResult(
        R2_HSIC=r2_all,
        T_HSIC=t_all,
        p_values=p_all,
        hsic_raw=raw_all,
        problem=problem,
    )
