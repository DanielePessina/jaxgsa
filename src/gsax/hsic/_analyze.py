"""HSIC analysis: kernel-based sensitivity indices.

Computes R2-HSIC (normalized first-order) and Total HSIC indices from
arbitrary (X, Y) sample pairs using Gaussian RBF kernels with the
median heuristic for bandwidth selection.

Array shape conventions used throughout:
    N  — number of samples
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)

For total HSIC, augmented kernels k*(x,x') = 1 + k_c(x,x') are used
per Larsen & Alexanderian (2026), where k_c is the centered kernel.
The product of augmented kernels captures all interaction orders,
not just the highest, giving correct total indices for additive models.

References:
    Gretton et al. (2005). JMLR 6:2075-2129.
    Da Veiga (2015). Rel. Eng. Sys. Safety 142:346-362.
    Larsen & Alexanderian (2026). arXiv:2603.00849.
"""

from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from gsax._normalization import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from gsax._transforms import cdf_to_unit_interval
from gsax.hsic._result import HSICResult
from gsax.problem import Problem

_MIN_SAMPLES = 4


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


def _center_kernel(K: Array) -> Array:
    """Center a kernel matrix: K_c = HKH where H = I - (1/n)11^T.

    Args:
        K: (N, N) kernel matrix.

    Returns:
        (N, N) centered kernel matrix.
    """
    row_mean = jnp.mean(K, axis=1, keepdims=True)
    col_mean = jnp.mean(K, axis=0, keepdims=True)
    grand_mean = jnp.mean(K)
    return K - row_mean - col_mean + grand_mean


def _hsic_v(K: Array, L: Array) -> Array:
    """Biased V-statistic HSIC estimator.

    Uses the efficient trace formula avoiding explicit centering matrices:
        HSIC = U/n^2 - 2V/n^3 + W/n^4
    where U = sum(K*L), V = sum(colsums(K)*colsums(L)), W = sum(K)*sum(L).

    Args:
        K: (N, N) kernel matrix.
        L: (N, N) kernel matrix.

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
    chunk_size: int | None,
    N: int,
) -> Array:
    """Build a single kernel matrix with the appropriate strategy.

    Args:
        x: 1-D array of N values.
        bandwidth: Fixed bandwidth or None for median heuristic.
        chunk_size: Block size for kernel matrix, or None.
        N: Sample count (used for chunking decision).

    Returns:
        (N, N) kernel matrix.
    """
    use_chunked = chunk_size is not None and chunk_size > 0 and N > chunk_size
    if bandwidth is None and not use_chunked:
        return _build_kernel_median(x)
    if bandwidth is not None and not use_chunked:
        return _build_kernel_fixed(x, jnp.asarray(bandwidth, dtype=x.dtype))
    if bandwidth is None:
        sigma = _median_bandwidth(x)
    else:
        sigma = jnp.asarray(bandwidth, dtype=x.dtype)
    if chunk_size is None:
        raise ValueError("chunk_size must not be None in chunked path")
    return _build_kernel_chunked(x, sigma, chunk_size)


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
    return [_build_one_kernel(X_unit[:, d], bandwidth, chunk_size, N) for d in range(D)]


def _augmented_kernels(Ks: list[Array]) -> list[Array]:
    """Build augmented kernels: K*_d = 1 + center(K_d).

    The augmented kernel includes a constant term so that the product
    of augmented kernels captures all interaction orders, not just the
    highest. This is required for correct total HSIC indices.

    Args:
        Ks: List of D raw kernel matrices.

    Returns:
        List of D augmented kernel matrices.
    """
    return [jnp.ones_like(K) + _center_kernel(K) for K in Ks]


def _complement_kernels(Ks: list[Array]) -> tuple[Array, list[Array]]:
    """Build full product and complement product kernels via prefix-suffix.

    Args:
        Ks: List of D kernel matrices, each (N, N).

    Returns:
        Tuple of (K_full, complements) where K_full is the Hadamard
        product of all Ks, and complements[d] is the product of all
        Ks except d.
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
def _get_hsic_kernel(n_perms: int):
    """Return a JIT-compiled HSIC slice kernel for the given ``n_perms``.

    Args:
        n_perms: Number of permutations for the permutation test (static;
            it sets the scan length and the number of split keys).

    Returns:
        A jitted callable that computes ``(R2_HSIC, T_HSIC, p_values,
        hsic_raw)`` for one output slice from the stacked input kernels,
        the precomputed self-HSIC values, the output kernel, and a key.
    """

    def _impl(
        Ks_stack: Array,
        K_aug_compls_stack: Array,
        K_aug_full: Array,
        hsic_xxs: Array,
        L: Array,
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
            L: Output kernel matrix ``(N, N)``.
            key: PRNG key for the permutation test.

        Returns:
            ``(R2_HSIC, T_HSIC, p_values, hsic_raw)`` each of shape ``(D,)``.
        """
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

    return jax.jit(_impl)


def _compute_slice(
    Ks_stack: Array,
    K_aug_compls_stack: Array,
    K_aug_full: Array,
    hsic_xxs: Array,
    y_col: Array,
    bandwidth: float | None,
    key: Array,
    n_perms: int,
    chunk_size: int | None,
) -> tuple[Array, Array, Array, Array]:
    """Compute HSIC indices for a single (t, k) output slice.

    Args:
        Ks_stack: Stacked raw input kernels ``(D, N, N)``.
        K_aug_compls_stack: Stacked complement augmented kernels ``(D, N, N)``.
        K_aug_full: Product of all augmented input kernels ``(N, N)``.
        hsic_xxs: Precomputed self-HSIC ``HSIC(K_d, K_d)`` ``(D,)``
            (output-independent, so built once by the caller).
        y_col: ``(N,)`` single output column.
        bandwidth: Fixed bandwidth or None for median heuristic.
        key: PRNG key for permutation test.
        n_perms: Number of permutations.
        chunk_size: Block size for kernel matrix, or None.

    Returns:
        (R2_HSIC, T_HSIC, p_values, hsic_raw) each of shape ``(D,)``.
    """
    N = y_col.shape[0]
    L = _build_one_kernel(y_col, bandwidth, chunk_size, N)
    return _get_hsic_kernel(n_perms)(Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, L, key)


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
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) sensitivity indices.

    HSIC quantifies the statistical dependence between each input and the
    output using kernel embeddings, so it picks up nonlinear and
    non-monotonic relationships that correlation-based screening misses.
    It is a given-data method: any (X, Y) sample pair works — no special
    sampling design is required. Two indices are reported:

    - **R2-HSIC**: HSIC(x_i, Y) normalized by the geometric mean of the
      self-similarities, ``HSIC(x_i, Y) / sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))``.
      Lies in [0, 1]; 0 means x_i and Y are independent (first-order view).
    - **Total HSIC (T_HSIC)**: fraction of the joint dependence lost when
      x_i is removed, analogous to a total-order Sobol index — it also
      counts influence carried through interactions with other inputs.

    A permutation test supplies p-values for the null hypothesis that
    x_i and Y are independent, making HSIC useful for screening out
    non-influential inputs with a significance level attached.

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)`` in physical units.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
            For outputs with large magnitude, set ``prenormalize=True``
            to avoid float overflow in distance computation.
        n_perms: Number of random permutations for the p-value test.
            More permutations give finer p-value resolution (the smallest
            attainable p-value is ``1 / (n_perms + 1)``) at linearly
            higher cost; 200 (default) resolves down to p ~ 0.005.
        seed: Random seed for permutation test reproducibility.
        bandwidth: Fixed Gaussian-kernel bandwidth applied to all inputs
            and the output. None (default) selects it per variable via the
            median heuristic (median pairwise distance), a robust default.
        chunk_size: Row-block size for building each ``(N, N)`` kernel
            matrix, bounding peak memory for large N. None computes each
            full matrix at once.
        prenormalize: If True, standardize each output slice to mean 0 and
            unit standard deviation before analysis.

    Returns:
        HSICResult with R2_HSIC, T_HSIC, p_values, and hsic_raw.

    Raises:
        ValueError: If X is not 2-D, column count doesn't match problem,
            row counts of X and Y differ, n_perms < 1, N < 4, or
            bandwidth is non-positive / non-finite.
    """
    D = problem.num_vars
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")
    if bandwidth is not None and (bandwidth <= 0 or not math.isfinite(bandwidth)):
        raise ValueError(f"bandwidth must be positive and finite, got {bandwidth}")
    Y = _validate_xy_inputs(problem, X, Y)
    if X.shape[0] < _MIN_SAMPLES:
        raise ValueError(f"N must be >= {_MIN_SAMPLES} for HSIC, got {X.shape[0]}")

    _warn_zero_variance_slices(Y, problem.output_names)

    X_unit = cdf_to_unit_interval(X, problem)

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _N, T, K = Y_3d.shape

    if prenormalize:
        Y_3d, _, _, _ = _prenormalize_outputs(Y_3d)

    key = jax.random.key(seed)

    # Build input kernels and augmented products once (independent of Y).
    Ks = _build_input_kernels(X_unit, bandwidth, chunk_size)
    Ks_stack = jnp.stack(Ks)
    Ks_aug = _augmented_kernels(Ks)
    K_aug_full, K_aug_compls = _complement_kernels(Ks_aug)
    K_aug_compls_stack = jnp.stack(K_aug_compls)

    # Self-HSIC HSIC(K_d, K_d) is output-independent, so compute it once
    # here and share it across every output slice.
    hsic_xxs = jax.vmap(lambda K: _hsic_v(K, K))(Ks_stack)

    r2_all = jnp.empty((T, K, D))
    t_all = jnp.empty((T, K, D))
    p_all = jnp.empty((T, K, D))
    raw_all = jnp.empty((T, K, D))

    for t in range(T):
        for k in range(K):
            subkey = jax.random.fold_in(key, t * K + k)
            y_col = Y_3d[:, t, k]

            r2, t_hsic, p_vals, raw = _compute_slice(
                Ks_stack,
                K_aug_compls_stack,
                K_aug_full,
                hsic_xxs,
                y_col,
                bandwidth,
                subkey,
                n_perms,
                chunk_size,
            )

            r2_all = r2_all.at[t, k].set(r2)
            t_all = t_all.at[t, k].set(t_hsic)
            p_all = p_all.at[t, k].set(p_vals)
            raw_all = raw_all.at[t, k].set(raw)

    r2_all = _squeeze_output_axes(r2_all, squeeze_time, squeeze_output)
    t_all = _squeeze_output_axes(t_all, squeeze_time, squeeze_output)
    p_all = _squeeze_output_axes(p_all, squeeze_time, squeeze_output)
    raw_all = _squeeze_output_axes(raw_all, squeeze_time, squeeze_output)

    return HSICResult(
        R2_HSIC=r2_all,
        T_HSIC=t_all,
        p_values=p_all,
        hsic_raw=raw_all,
        problem=problem,
    )
