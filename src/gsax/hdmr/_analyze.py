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

from gsax._normalization import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
    _warn_zero_variance_slices,
)
from gsax._transforms import cdf_to_unit_interval
from gsax.hdmr._engine import (
    _build_B1,
    _build_B2,
    _build_B3,
    _compute_f_crits,
    _make_hdmr_kernel,
)
from gsax.hdmr._result import HDMREmulator, HDMRResult
from gsax.problem import Problem


@lru_cache(maxsize=None)
def _get_hdmr_static_data(D: int, maxorder: int, m: int) -> tuple:
    """Cache host-side HDMR term metadata and basis index tables."""
    # Enumerate all parameter index combinations up to maxorder.
    # c1: single dimensions, c2: pairs, c3: triples.
    c1 = tuple(range(D))
    c2 = tuple(itertools.combinations(range(D), 2)) if maxorder >= 2 else tuple()
    c3 = tuple(itertools.combinations(range(D), 3)) if maxorder >= 3 else tuple()
    n1 = D
    n2 = len(c2)
    n3 = len(c3)
    # Total number of HDMR component functions across all active orders.
    n = n1 + n2 + n3
    # m1 = basis functions per dimension (m intervals + 3 for cubic B-spline
    # boundary support); m2, m3 = tensor-product basis sizes for orders 2, 3.
    m1 = m + 3
    m2 = m1**2
    m3 = m1**3
    # beta tables enumerate all multi-index pairs/triples into the 1-D basis.
    # They turn the tensor product into a flat index -> (i, j[, k]) lookup
    # so the basis can be built via gather + elementwise multiply.
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
    # Caching by (D, maxorder, m, maxiter, lambdax, N) avoids re-tracing the
    # JIT+vmap wrapper when analyze_hdmr is called repeatedly with the same
    # structural parameters but different data.
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
    # in_axes=(None, None, None, 0, None): B1/B2/B3 basis matrices and f_crits
    # are shared across all output slices, while Y is batched along axis 0 so
    # each (T*K) output slice gets its own response vector.
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
    # ST_j = S_j + sum_{i<j or j<i} S_{ij} + sum_{i<j<k, j in {i,j,k}} S_{ijk}
    # i.e. total-order for param j includes its first-order term plus every
    # interaction term (2nd and 3rd order) that contains j.
    ST = S[..., :n1]  # First-order terms map 1:1 to parameters.

    # Scatter-add each 2nd-order term S_{ab} to both parameters a and b.
    n2 = c2.shape[0]
    S2 = S[..., n1 : n1 + n2]
    ST = ST.at[..., c2[:, 0]].add(S2)
    ST = ST.at[..., c2[:, 1]].add(S2)

    # Scatter-add each 3rd-order term S_{abc} to all three participating params.
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
    return tuple(_squeeze_output_axes(a, squeeze_time, squeeze_output) for a in (Sa, Sb, S, ST))


def _reshape_emulator_value(
    value: Array,
    T: int,
    K_out: int,
    squeeze_time: bool,
    squeeze_output: bool,
) -> Array:
    """Reshape flattened per-output emulator state back to the analyzed layout."""
    value = value.reshape((T, K_out) + value.shape[1:])
    return _squeeze_output_axes(value, squeeze_time, squeeze_output, n_trailing=value.ndim - 2)


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
    """Compute sensitivity indices via RS-HDMR (public entry point).

    Validates ``(X, Y)``, warns once about any zero-variance output slice, then
    delegates to :func:`_analyze_hdmr_core`. See that function for the full
    parameter and return documentation; ``analyze_shapley``'s HDMR backend
    calls the core directly on an already-canonical Y to avoid re-validating.
    """
    X = jnp.asarray(X)
    Y, ops = _validate_xy_inputs(problem, X, jnp.asarray(Y))
    # A constant output slice makes every index 0/0 = NaN; warn once up front,
    # in the public wrapper only, so callers routing through the core (Shapley)
    # do not double-warn.
    _warn_zero_variance_slices(_prepare_Y(Y)[0], output_names=problem.output_names)
    return _analyze_hdmr_core(
        problem,
        X,
        Y,
        prenormalize=prenormalize,
        maxorder=maxorder,
        maxiter=maxiter,
        m=m,
        lambdax=lambdax,
        chunk_size=chunk_size,
        inserted_output_axis=ops.inserted_output_axis,
    )


def _analyze_hdmr_core(
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
    inserted_output_axis: bool = False,
) -> HDMRResult:
    """Fit RS-HDMR on an already-canonical Y (no re-validation, no warn).

    Compute sensitivity indices via RS-HDMR with B-spline surrogate modelling.

    Works with **any** set of (X, Y) pairs -- no structured sampling required,
    so it suits existing datasets and expensive models where Sobol/eFAST
    sampling schemes are unaffordable. Decomposes the input-output
    relationship into hierarchical component functions (one per parameter,
    parameter pair, ...) via B-spline regression, then derives ANCOVA-based
    sensitivity indices from the fitted components. Unlike pure Sobol
    estimators, the ANCOVA split into structural (Sa) and correlative (Sb)
    parts remains meaningful when inputs are correlated.

    Args:
        problem: Parameter names and distributions.
        X: (N, D) input samples.
        Y: (N,), (N, K), or (N, T, K) model outputs. A 2D array is read as
            (N, K) unless ``problem.output_names`` has exactly one entry, in
            which case the columns are T timepoints of that single output.
        prenormalize: When ``True``, standardize each output slice over the
            sample axis (subtract mean, divide by standard deviation) before
            fitting, which puts disparate output magnitudes on an equal
            numerical footing. The indices are ratios and unaffected;
            predictions from the returned emulator are still on the original
            output scale. Defaults to ``False``.
        maxorder: Maximum HDMR expansion order (1, 2, or 3): the largest
            interaction size modelled. Order 2 (default) captures pairwise
            interactions; 3 adds triples but the term count and fit cost grow
            combinatorially. Clamped to D (with a warning) when D < maxorder.
        maxiter: Maximum backfitting iterations for the first-order terms.
            The default rarely needs raising; iteration stops early once the
            coefficients stop changing.
        m: Number of B-spline intervals per dimension (basis size m + 3).
            Larger m resolves sharper features of the component functions but
            multiplies the coefficient count (per-term basis grows as
            (m+3)^order) and needs more samples to avoid overfitting.
        lambdax: Tikhonov regularization strength. Increase for noisy Y or
            small N (smoother, more stable components); decrease if genuine
            sharp features are being oversmoothed.
        chunk_size: Maximum number of (T, K) output slices fitted per vmap
            batch. Caps peak device memory for large T*K; smaller values
            trade speed for memory.

    Returns:
        HDMRResult with per-term indices Sa, Sb, S, per-parameter ST,
        human-readable term labels, F-test selection counts, the fitted
        emulator (usable with ``emulate_hdmr``), and its RMSE.

    Raises:
        ValueError: If ``N < 300``, ``maxorder`` is not 1/2/3, or
            ``chunk_size < 1``.
    """
    N, D = X.shape
    # B-spline regression with backfitting needs a reasonable sample size
    # to avoid overfitting; 300 is a practical lower bound.
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

    # Build term metadata (cached on host; only computed once per D/maxorder/m).
    c1, c2, c3, n1, n2, n3, n, m1, m2, m3, beta2_host, beta3_host = _get_hdmr_static_data(
        D,
        maxorder,
        m,
    )
    term_labels = _build_term_labels(problem, c1, c2, c3)
    # Transfer index tables to device; empty arrays for inactive orders.
    c2_idx = jnp.asarray(c2, dtype=int) if n2 > 0 else jnp.zeros((0, 2), dtype=int)
    c3_idx = jnp.asarray(c3, dtype=int) if n3 > 0 else jnp.zeros((0, 3), dtype=int)
    beta2 = jnp.asarray(beta2_host, dtype=int)
    beta3 = jnp.asarray(beta3_host, dtype=int)

    # CDF transform: maps each dimension's marginal to U[0,1] so the B-spline
    # basis operates on a uniform domain regardless of the original distribution.
    X_n = cdf_to_unit_interval(X, problem)

    # Build B-spline bases for all orders. Bases are shared across output
    # slices (only Y changes), so they are computed once here.
    B1 = _build_B1(X_n, m)  # (N, m1, D)
    B2 = _build_B2(B1, c2_idx, beta2) if n2 > 0 else jnp.zeros((N, 1, 1))
    B3 = _build_B3(B1, c3_idx, beta3) if n3 > 0 else jnp.zeros((N, 1, 1))

    # F critical values at alpha=0.95, precomputed outside JIT to avoid
    # re-running the bisection solver on every vmap lane.
    f_crits = _compute_f_crits(0.95, m1, m2, m3, N)

    # Promote Y to canonical (N, T, K) layout; track which dims were singleton
    # so the output arrays can be squeezed back to the user's original shape.
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K_out = Y_3d.shape
    if prenormalize:
        # Standardize each (t, k) slice to zero-mean, unit-variance before
        # fitting. The scale factors are stored so emulate_hdmr can invert.
        Y_3d, y_mean, y_std, _ = _prenormalize_outputs(Y_3d)
    else:
        y_mean = jnp.zeros(Y_3d.shape[1:], dtype=Y_3d.dtype)
        y_std = jnp.ones(Y_3d.shape[1:], dtype=Y_3d.dtype)

    # (N,T,K) -> (T,K,N) -> (T*K, N): move sample axis last, then flatten
    # time x output into a single batch dim so each row is an independent
    # response vector for the vmapped kernel.
    Y_flat = Y_3d.transpose(1, 2, 0).reshape(T * K_out, N)
    total = T * K_out
    cs = min(chunk_size, total)

    # Accumulate chunk results; select_sum aggregates F-test pass counts
    # across chunks to report how many output slices found each term significant.
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
    # Process output slices in chunks to bound peak device memory when T*K
    # is large. Each chunk is a vmap batch of independent HDMR fits.
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

    # Reassemble chunks into the full (T, K, n_terms) index arrays.
    Sa_out = jnp.concatenate(sa_parts).reshape(T, K_out, n)
    Sb_out = jnp.concatenate(sb_parts).reshape(T, K_out, n)
    S_out = jnp.concatenate(s_parts).reshape(T, K_out, n)
    # Aggregate per-term S into per-parameter total-order indices.
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
    y_mean_out = _squeeze_output_axes(y_mean, squeeze_time, squeeze_output, n_trailing=0)
    y_std_out = _squeeze_output_axes(y_std, squeeze_time, squeeze_output, n_trailing=0)

    # Bundle all fitted state needed to reconstruct predictions at new points.
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
        c2=tuple(c2),
        c3=tuple(c3),
        n1=n1,
        emulator=emulator,
        select=select_sum,
        # RMSE is computed on the standardized scale inside the kernel;
        # multiply by y_std to report it on the original output scale.
        rmse=_reshape_emulator_value(
            jnp.concatenate(rmse_parts), T, K_out, squeeze_time, squeeze_output
        )
        * y_std_out,
        _inserted_output_axis=inserted_output_axis,
    )


def _emulator_contract(B: Array, C: Array) -> Array:
    """Contract basis B (N, m, j) with coefficients C, summing over terms.

    Dispatches on C.ndim to handle scalar (m, j), multi-output (K, m, j),
    and time-series (T, K, m, j) coefficient layouts.
    """
    # Dispatch on C.ndim to handle different output layouts:
    #   2D (m, j):      scalar output   -> einsum("rmj,mj->rj")   -> sum over j terms
    #   3D (K, m, j):   multi-output    -> einsum("rmj,kmj->rkj") -> sum over j terms
    #   4D (T, K, m, j): time-series    -> einsum("rmj,tkmj->rtkj") -> sum over j terms
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

    # Apply the same CDF -> [0,1] transform used during fitting.
    X_n = cdf_to_unit_interval(X_new, result.problem)

    # Reconstruct prediction as f0 + sum of component functions.
    # Start with first-order: sum_j B1_j @ C1_j.
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

    # Add grand mean to recover the full surrogate prediction.
    Y_pred = Y_total + f0
    # Undo the standardization applied during fitting, if any.
    if prenormalize:
        Y_pred = Y_pred * y_std + y_mean
    # If inference inserted a singleton K axis at fit time, drop it so the
    # prediction mirrors the training Y's original (N_new, T) rank.
    if result._inserted_output_axis:
        Y_pred = Y_pred[..., 0]
    return Y_pred
