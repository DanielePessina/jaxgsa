"""RS-HDMR (Random Sampling High-Dimensional Model Representation) analysis.

Provides ``analyze_hdmr`` for computing ANCOVA-based sensitivity indices from
arbitrary (X, Y) pairs using B-spline surrogate modelling, and ``emulate_hdmr``
for prediction with the fitted surrogate.
"""

import itertools
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax._normalization import _prenormalize_outputs, _prepare_Y, _warn_zero_variance_slices
from gsax._transforms import cdf_to_unit_interval
from gsax.expansions.hdmr._engine import (
    _build_B1,
    _build_B2,
    _build_B3,
    _compute_f_crits,
    _make_hdmr_kernel,
)
from gsax.expansions.hdmr._result import HDMREmulator, HDMRResult
from gsax.problem import Problem


def _normalize_X(X: Array, problem: Problem) -> Array:
    """Normalize X to [0, 1] via per-dimension marginal CDF."""
    return cdf_to_unit_interval(X, problem)


@lru_cache(maxsize=None)
def _get_hdmr_static_data(D: int, maxorder: int, m: int) -> tuple:
    """Cache host-side HDMR term metadata and basis index tables."""
    c1 = tuple(range(D))
    c2 = tuple(itertools.combinations(range(D), 2)) if maxorder >= 2 else tuple()
    c3 = tuple(itertools.combinations(range(D), 3)) if maxorder >= 3 else tuple()
    n1 = D
    n2 = len(c2)
    n3 = len(c3)
    n = n1 + n2 + n3
    m1 = m + 3
    m2 = m1**2
    m3 = m1**3
    beta2 = (
        np.asarray(list(itertools.product(range(m1), repeat=2)), dtype=np.int32)
        if n2 > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    beta3 = (
        np.asarray(list(itertools.product(range(m1), repeat=3)), dtype=np.int32)
        if n3 > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    return c1, c2, c3, n1, n2, n3, n, m1, m2, m3, beta2, beta3


@lru_cache(maxsize=None)
def _get_batched_hdmr_kernel(
    D: int,
    maxorder: int,
    m: int,
    maxiter: int,
    lambdax: float,
    N: int,
):
    """Cache the final batched HDMR wrapper by semantic signature."""
    _, _, _, n1, n2, n3, n, m1, m2, m3, _, _ = _get_hdmr_static_data(D, maxorder, m)
    kernel = _make_hdmr_kernel(
        maxorder,
        m1,
        n1,
        maxiter,
        m2,
        m3,
        n2,
        n3,
        n,
        lambdax,
        N,
    )
    return jax.jit(jax.vmap(kernel, in_axes=(None, None, None, 0, None)))


def _build_term_labels(
    problem: Problem,
    c1: tuple[int, ...],
    c2: tuple[tuple[int, int], ...],
    c3: tuple[tuple[int, int, int], ...],
) -> tuple[str, ...]:
    """Build human-readable term labels."""
    names = problem.names
    labels = [names[i] for i in c1]
    labels += ["/".join(names[i] for i in combo) for combo in c2]
    labels += ["/".join(names[i] for i in combo) for combo in c3]
    return tuple(labels)


def _compute_ST(
    S: Array,
    c2: Array,
    c3: Array,
    n1: int,
) -> Array:
    """Compute total-order indices by summing S over terms involving each param."""
    ST = S[..., :n1]  # First order terms map 1:1 to parameters.

    n2 = c2.shape[0]
    S2 = S[..., n1 : n1 + n2]
    ST = ST.at[..., c2[:, 0]].add(S2)
    ST = ST.at[..., c2[:, 1]].add(S2)

    S3 = S[..., n1 + n2 :]
    ST = ST.at[..., c3[:, 0]].add(S3)
    ST = ST.at[..., c3[:, 1]].add(S3)
    ST = ST.at[..., c3[:, 2]].add(S3)

    return ST



def _squeeze_hdmr(
    Sa: Array,
    Sb: Array,
    S: Array,
    ST: Array,
    squeeze_time: bool,
    squeeze_output: bool,
) -> tuple:
    """Remove singleton T/K dims from HDMR result arrays."""
    arrays = [Sa, Sb, S, ST]

    if squeeze_time and squeeze_output:
        arrays = [a[0, 0] for a in arrays]
    elif squeeze_time:
        arrays = [a[0] for a in arrays]

    return tuple(arrays)


def _reshape_emulator_value(
    value: Array,
    T: int,
    K_out: int,
    squeeze_time: bool,
    squeeze_output: bool,
) -> Array:
    """Reshape flattened per-output emulator state back to the analyzed layout."""
    value = value.reshape((T, K_out) + value.shape[1:])

    if squeeze_time and squeeze_output:
        return value[0, 0]
    if squeeze_time:
        return value[0]
    return value


def analyze_hdmr(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    prenormalize: bool = False,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    chunk_size: int = 2048,
) -> HDMRResult:
    """Compute sensitivity indices via RS-HDMR with B-spline surrogate modelling.

    Works with **any** set of (X, Y) pairs -- no structured sampling required.
    Decomposes the input-output relationship into hierarchical component
    functions using B-spline regression, then derives ANCOVA-based sensitivity
    indices.

    Args:
        problem: Parameter names and bounds.
        X: (N, D) input samples.
        Y: (N,), (N, K), or (N, T, K) model outputs. A 2D array is always
            interpreted as (N, K); for time-series with a single output,
            reshape to (N, T, 1).
        prenormalize: When ``True``, standardize each output slice over the
            sample axis before fitting the surrogate by subtracting its mean
            and dividing by its standard deviation once globally. The fitted
            emulator is still returned on the original output scale. Defaults
            to ``False``.
        maxorder: Maximum HDMR expansion order (1, 2, or 3).
        maxiter: Maximum backfitting iterations for first-order terms.
        m: Number of B-spline intervals (basis size = m + 3 per dimension).
        lambdax: Tikhonov regularization parameter.
        chunk_size: Maximum number of (T, K) output combos per vmap batch.

    Returns:
        HDMRResult with Sa, Sb, S, ST, emulator, etc.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    N, D = X.shape
    if D != problem.num_vars:
        raise ValueError(f"X has {D} columns but problem defines {problem.num_vars} parameters")
    if N < 300:
        raise ValueError(f"Need at least 300 samples, got {N}")
    if maxorder not in (1, 2, 3):
        raise ValueError(f"maxorder must be 1, 2, or 3, got {maxorder}")
    if D < maxorder:
        import warnings

        maxorder = min(maxorder, D)
        warnings.warn(
            f"gsax: maxorder clamped to {maxorder} (need D >= maxorder, got D={D})",
            stacklevel=2,
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    lambdax = float(lambdax)

    # Build terms
    c1, c2, c3, n1, n2, n3, n, m1, m2, m3, beta2_host, beta3_host = _get_hdmr_static_data(
        D,
        maxorder,
        m,
    )
    term_labels = _build_term_labels(problem, c1, c2, c3)
    c2_idx = jnp.asarray(c2, dtype=int) if n2 > 0 else jnp.zeros((0, 2), dtype=int)
    c3_idx = jnp.asarray(c3, dtype=int) if n3 > 0 else jnp.zeros((0, 3), dtype=int)
    beta2 = jnp.asarray(beta2_host, dtype=int)
    beta3 = jnp.asarray(beta3_host, dtype=int)

    # Normalize X
    X_n = _normalize_X(X, problem)

    # Build B-spline bases
    B1 = _build_B1(X_n, m)  # (N, m1, D)
    B2 = _build_B2(B1, c2_idx, beta2) if n2 > 0 else jnp.zeros((N, 1, 1))
    B3 = _build_B3(B1, c3_idx, beta3) if n3 > 0 else jnp.zeros((N, 1, 1))

    # Precompute F critical values
    f_crits = _compute_f_crits(0.95, m1, m2, m3, N)

    # Promote Y to 3D
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K_out = Y_3d.shape
    if prenormalize:
        Y_3d, y_mean, y_std, _ = _prenormalize_outputs(Y_3d)
    else:
        y_mean = jnp.zeros(Y_3d.shape[1:], dtype=Y_3d.dtype)
        y_std = jnp.ones(Y_3d.shape[1:], dtype=Y_3d.dtype)

    _warn_zero_variance_slices(Y_3d, output_names=problem.output_names)

    Y_flat = Y_3d.transpose(1, 2, 0).reshape(T * K_out, N)
    total = T * K_out
    cs = min(chunk_size, total)

    sa_parts, sb_parts, s_parts, rmse_parts = [], [], [], []
    c1_parts, c2_parts, c3_parts, f0_parts = [], [], [], []
    select_sum = jnp.zeros(n)

    batched_kernel = _get_batched_hdmr_kernel(
        D,
        maxorder,
        m,
        maxiter,
        lambdax,
        N,
    )
    for start in range(0, total, cs):
        end = min(start + cs, total)
        sa, sb, s, sel, rmse_val, c1_coef, c2_coef, c3_coef, f0_val = batched_kernel(
            B1,
            B2,
            B3,
            Y_flat[start:end],
            f_crits,
        )
        sa_parts.append(sa)
        sb_parts.append(sb)
        s_parts.append(s)
        rmse_parts.append(rmse_val)
        c1_parts.append(c1_coef)
        f0_parts.append(f0_val)
        select_sum = select_sum + jnp.sum(sel, axis=0)
        if n2 > 0:
            c2_parts.append(c2_coef)
        if n3 > 0:
            c3_parts.append(c3_coef)

    # Reshape to (T, K, n_terms) / (T, K, D)
    Sa_out = jnp.concatenate(sa_parts).reshape(T, K_out, n)
    Sb_out = jnp.concatenate(sb_parts).reshape(T, K_out, n)
    S_out = jnp.concatenate(s_parts).reshape(T, K_out, n)
    ST_out = _compute_ST(S_out, c2_idx, c3_idx, n1)

    # Squeeze
    Sa_out, Sb_out, S_out, ST_out = _squeeze_hdmr(
        Sa_out,
        Sb_out,
        S_out,
        ST_out,
        squeeze_time,
        squeeze_output,
    )

    C1_out = _reshape_emulator_value(
        jnp.concatenate(c1_parts),
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    C2_out = None
    C3_out = None
    if n2 > 0:
        C2_out = _reshape_emulator_value(
            jnp.concatenate(c2_parts),
            T,
            K_out,
            squeeze_time,
            squeeze_output,
        )
    if n3 > 0:
        C3_out = _reshape_emulator_value(
            jnp.concatenate(c3_parts),
            T,
            K_out,
            squeeze_time,
            squeeze_output,
        )
    f0_out = _reshape_emulator_value(
        jnp.concatenate(f0_parts),
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    y_mean_out = _reshape_emulator_value(
        y_mean.reshape(T * K_out, 1),
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    y_mean_out = jnp.squeeze(y_mean_out, axis=-1)
    y_std_out = _reshape_emulator_value(
        y_std.reshape(T * K_out, 1),
        T,
        K_out,
        squeeze_time,
        squeeze_output,
    )
    y_std_out = jnp.squeeze(y_std_out, axis=-1)

    # Build emulator dict
    emulator: HDMREmulator = {
        "C1": C1_out,
        "C2": C2_out,
        "C3": C3_out,
        "f0": f0_out,
        "prenormalize": prenormalize,
        "y_mean": y_mean_out,
        "y_std": y_std_out,
        "m": m,
        "maxorder": maxorder,
        "c2": list(c2),
        "c3": list(c3),
    }

    return HDMRResult(
        Sa=Sa_out,
        Sb=Sb_out,
        S=S_out,
        ST=ST_out,
        problem=problem,
        terms=term_labels,
        emulator=emulator,
        select=select_sum,
        rmse=_reshape_emulator_value(
            jnp.concatenate(rmse_parts), T, K_out, squeeze_time, squeeze_output
        )
        * y_std_out,
    )


def _emulator_contract(B: Array, C: Array) -> Array:
    """Contract basis B (N, m, j) with coefficients C, summing over terms.

    Dispatches on C.ndim to handle scalar (m, j), multi-output (K, m, j),
    and time-series (T, K, m, j) coefficient layouts.
    """
    if C.ndim == 2:
        return jnp.sum(jnp.einsum("rmj,mj->rj", B, C), axis=1)
    if C.ndim == 3:
        return jnp.sum(jnp.einsum("rmj,kmj->rkj", B, C), axis=2)
    return jnp.sum(jnp.einsum("rmj,tkmj->rtkj", B, C), axis=3)


def emulate_hdmr(result: HDMRResult, X_new: Array) -> Array:
    """Predict at new input points using the fitted HDMR surrogate.

    Note: This function is not JIT-compatible because ``HDMRResult`` is not a
    registered JAX pytree type.

    Args:
        result: HDMRResult from ``analyze_hdmr`` (must have ``emulator`` set).
        X_new: (N_new, D) new input points within the problem bounds.

    Returns:
        Y_pred: (N_new,), (N_new, K), or (N_new, T, K) predicted outputs.
            When the emulator was fit with ``prenormalize=True``, predictions
            are inverse-transformed back to the original output scale before
            being returned.
    """
    em = result.emulator
    if em is None:
        raise ValueError("HDMRResult has no emulator (emulator is None)")

    X_new = jnp.asarray(X_new)
    maxorder = em["maxorder"]
    C1 = em["C1"]
    f0 = em["f0"]
    prenormalize = em["prenormalize"]
    y_mean = em["y_mean"]
    y_std = em["y_std"]

    # Normalize
    X_n = _normalize_X(X_new, result.problem)

    # Build bases and compute predictions
    B1 = _build_B1(X_n, em["m"])  # (N_new, m1, D)
    Y_total = _emulator_contract(B1, C1)

    if maxorder >= 2 and em["C2"] is not None:
        _, _, _, _, _, _, _, _, _, _, beta2_host, _ = _get_hdmr_static_data(
            result.problem.num_vars,
            maxorder,
            em["m"],
        )
        B2 = _build_B2(
            B1,
            jnp.asarray(em["c2"], dtype=int),
            jnp.asarray(beta2_host, dtype=int),
        )
        Y_total = Y_total + _emulator_contract(B2, em["C2"])

    if maxorder >= 3 and em["C3"] is not None:
        _, _, _, _, _, _, _, _, _, _, _, beta3_host = _get_hdmr_static_data(
            result.problem.num_vars,
            maxorder,
            em["m"],
        )
        B3 = _build_B3(
            B1,
            jnp.asarray(em["c3"], dtype=int),
            jnp.asarray(beta3_host, dtype=int),
        )
        Y_total = Y_total + _emulator_contract(B3, em["C3"])

    Y_pred = Y_total + f0
    if prenormalize:
        Y_pred = Y_pred * y_std + y_mean
    return Y_pred
