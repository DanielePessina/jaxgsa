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

from jaxgsa._core.entry import at_least, prepare, require
from jaxgsa._core.invalid import OnInvalid
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa._core.validation import (
    _prenormalize_outputs,
)
from jaxgsa.hsic._result import HSICResult
from jaxgsa.problem import Problem

_MIN_SAMPLES = 4


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

    Position ``S - 0.5`` therefore serves both parities, and ``jnp.quantile``
    with the default linear interpolation puts the target at
    ``q * (N^2 - 1)``. Solving for q and substituting S gives the expression
    below. It reproduces ``jnp.median`` of the strict upper triangle exactly at
    every N tested from 4 to 1024.

    Args:
        x: Values for one variable, shape ``(N,)``.

    Returns:
        Scalar median of the off-diagonal squared distances, floored at 1e-20.
    """
    N = x.shape[0]
    dists_sq = (x[:, None] - x[None, :]) ** 2
    q = (N**2 + N - 1) / (2 * (N**2 - 1))
    return jnp.maximum(jnp.quantile(dists_sq, q), 1e-20)


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
def _get_hsic_kernel(n_perms: int):
    """Return a JIT-compiled HSIC slice kernel for the given ``n_perms``.

    Args:
        n_perms: Number of permutations for the permutation test (static;
            it sets the scan length and the number of split keys).

    Returns:
        A jitted callable that returns ``(R2_HSIC, T_HSIC, p_values,
        hsic_raw)``, each of shape ``(D,)``, for one output slice. It takes
        the stacked input kernels, the precomputed self-HSIC values, the
        output kernel, and a PRNG key.
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
    batch_size: int | None,
) -> tuple[Array, Array, Array, Array]:
    """Compute HSIC indices for a single (t, k) output slice.

    Args:
        Ks_stack: Stacked raw input kernels ``(D, N, N)``.
        K_aug_compls_stack: Stacked complement augmented kernels ``(D, N, N)``.
        K_aug_full: Product of all augmented input kernels ``(N, N)``.
        hsic_xxs: Precomputed self-HSIC ``HSIC(K_d, K_d)`` ``(D,)``
            (output-independent, so built once by the caller).
        y_col: Single output column, shape ``(N,)``.
        bandwidth: Fixed bandwidth, or None for the median heuristic.
        key: PRNG key for the permutation test.
        n_perms: Number of permutations.
        batch_size: Row-block size for the kernel matrix, or None.

    Returns:
        ``(R2_HSIC, T_HSIC, p_values, hsic_raw)``, each of shape ``(D,)``.
    """
    L = _build_one_kernel(y_col, bandwidth, batch_size)
    return _get_hsic_kernel(n_perms)(Ks_stack, K_aug_compls_stack, K_aug_full, hsic_xxs, L, key)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    n_perms: int = 200,
    seed: int = 0,
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
        seed: Random seed that makes the permutation test reproducible.
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
            policy refuses the sample, or
            ``problem`` has categorical parameters. Categorical parameters
            are rejected because the Gaussian input kernel reads a level code
            as a distance, and the arbitrary code order makes that
            meaningless.
    """
    from jaxgsa.hsic import SPEC

    D = problem.num_vars

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

    X_unit = cdf_to_unit_interval(X, problem)

    Y_3d = ctx.Y3
    _N, T, K = Y_3d.shape

    if prenormalize:
        Y_3d, _, _, _ = _prenormalize_outputs(Y_3d)

    key = jax.random.key(seed)

    # Build input kernels and augmented products once (independent of Y).
    Ks = _build_input_kernels(X_unit, bandwidth, batch_size)
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
                batch_size,
            )

            r2_all = r2_all.at[t, k].set(r2)
            t_all = t_all.at[t, k].set(t_hsic)
            p_all = p_all.at[t, k].set(p_vals)
            raw_all = raw_all.at[t, k].set(raw)

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
