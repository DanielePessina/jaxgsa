"""Optimal-transport sensitivity analysis from given data.

Implements the optimal-transport sensitivity indices of Borgonovo,
Figalli, Plischke & Savare (2024). For each input the sample is split
into equal-frequency classes by the input's rank; the index is the
class-weighted squared 2-Wasserstein distance between the conditional
output distribution of each class and the unconditional one, normalized
by twice the output variance (the theoretical maximum of the averaged
distance) so it lies in [0, 1].

Every index is decomposed into an *advective* component -- the
class-averaged squared distance between conditional and unconditional
means, i.e. how much the input relocates the output distribution -- and
a *diffusive* remainder capturing changes in spread and shape. Because
the advective numerator is exactly ``Var(E[Y|X_i])``, the advective
component equals half the given-data first-order Sobol index, which
anchors the decomposition to the variance-based world.

Three modes cover gsax's output shapes. ``"separate"`` treats every
output column independently with the closed-form 1-D optimal transport
(sorted-quantile coupling; no solver): the unconditional sample supplies
quantiles at the N uniform mass points and each conditional class is
evaluated at the same points through its nearest-rank empirical quantile
function (midpoint rule ``j = floor((i + 0.5) * n_m / N)``, exact
whenever the class size divides N). ``"joint"`` and ``"joint-over-time"``
treat the (flattened or per-output) output vector as a point cloud and
transport the unconditional cloud onto each class with entropic
regularization (log-domain Sinkhorn, see
:mod:`gsax.optimal_transport._solver`), reporting the unregularized cost
of the entropic plan.

The original sample and any bootstrap resamples run through a single
scanned path (the original sample is replicate 0, gathered via the
identity permutation), so point estimates and confidence intervals share
one code path. Class partitions are rebuilt per replicate from the
resampled inputs, exactly as in :mod:`gsax.borgonovo`.

This is a clean-room implementation from the published equations; it is
numerically validated against POT (MIT-licensed) and analytic closed
forms in the test suite.

References:
    Borgonovo, Figalli, Plischke & Savare (2024). Global sensitivity
    analysis via optimal transport. Management Science.
    doi:10.1287/mnsc.2023.01796.
"""

from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from gsax._bootstrap import _percentile_ci
from gsax._normalization import (
    _prenormalize_outputs,
    _prepare_Y,
    _squeeze_output_axes,
    _validate_xy_inputs,
)
from gsax._partition import _build_class_indices, _class_layout
from gsax.optimal_transport._result import OTResult
from gsax.optimal_transport._solver import _sinkhorn_w2
from gsax.problem import Problem

# Target element budget for the default per-chunk working set of the
# "separate" mode. The dominant intermediate is the per-column conditional
# quantile tensor of size ``chunk_columns * D * M * N`` elements, so the
# default chunk width keeps it near this many float32 elements (~256 MB).
_CHUNK_ELEM_BUDGET = 1 << 26

_MODES = ("separate", "joint", "joint-over-time")


@jax.jit
def _ot_1d_kernel(
    Y_cols: Array,
    all_idx: Array,
    all_cls_idx: Array,
    mask: Array,
    counts: Array,
) -> tuple[Array, Array, Array, Array]:
    """Per-column 1-D optimal-transport indices for every replicate.

    Uses the closed-form 1-D coupling: both empirical quantile functions
    are evaluated on the N uniform mass points of the full sample, the
    conditional one through the nearest-rank (midpoint rule) lookup into
    the class's sorted members. No transport solver is involved.

    Args:
        Y_cols: Output columns ``(N, C)``.
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        all_cls_idx: Class indices ``(R, D, M, P)`` into the original
            sample, from :func:`gsax._partition._build_class_indices`.
        mask: Class validity mask ``(M, P)``.
        counts: True class sizes ``(M,)``.

    Returns:
        ``(ot, advective, diffusive, degenerate)`` with index arrays of
        shape ``(R, C, D)`` and a ``(R, C)`` flag marking constant
        (zero-variance) replicate columns, which yield zero indices.
    """
    dtype = jnp.result_type(Y_cols.dtype, jnp.float32)
    Y_cols = Y_cols.astype(dtype)
    maskf = mask.astype(dtype)
    countsf = counts.astype(dtype)
    N = Y_cols.shape[0]
    n_total = countsf.sum()  # == N
    weights = countsf / n_total  # (M,)

    # Nearest-rank conditional quantile index for every full-sample mass
    # point (midpoint rule); j < n_m always, so padding is never touched.
    i_grid = jnp.arange(N, dtype=dtype) + 0.5
    j = jnp.floor(i_grid[None, :] * countsf[:, None] / n_total).astype(jnp.int32)
    j = jnp.minimum(j, counts.astype(jnp.int32)[:, None] - 1)  # (M, N)

    def _col_stats(y: Array, r: Array, cls_idx: Array):
        """OT/advective/diffusive indices for one column and replicate."""
        y_r = y[r]  # resampled column
        y_sorted = jnp.sort(y_r)
        mean_r = y_r.mean()
        var_r = jnp.var(y_r, ddof=1)
        degenerate = y_r.max() == y_r.min()

        y_cls = y[cls_idx]  # (D, M, P) resampled class members
        # Pads sort to the tail as +inf and are unreachable through j.
        y_cls_sorted = jnp.sort(jnp.where(mask, y_cls, jnp.inf), axis=-1)
        j_full = jnp.broadcast_to(j[None], (y_cls.shape[0],) + j.shape)
        q = jnp.take_along_axis(y_cls_sorted, j_full, axis=-1)  # (D, M, N)
        w2 = ((y_sorted[None, None, :] - q) ** 2).mean(axis=-1)  # (D, M)

        cls_mean = (y_cls * maskf).sum(axis=-1) / countsf  # (D, M)
        adv = (cls_mean - mean_r) ** 2
        # W2^2 >= (mean shift)^2 mathematically; the clamp only absorbs
        # float cancellation noise near zero.
        diff = jnp.maximum(w2 - adv, 0.0)

        V = 2.0 * var_r
        V_safe = jnp.where(V > 0, V, 1.0)

        def _aggregate(per_class: Array) -> Array:
            val = (weights[None, :] * per_class).sum(axis=-1) / V_safe
            return jnp.where(V > 0, val, 0.0)

        return _aggregate(w2), _aggregate(adv), _aggregate(diff), degenerate

    def _one_replicate(carry, xs):
        r, cls_idx = xs
        out = jax.vmap(lambda y: _col_stats(y, r, cls_idx))(Y_cols.T)
        return carry, out

    _, (ot, adv, diff, degen) = jax.lax.scan(_one_replicate, None, (all_idx, all_cls_idx))
    return ot, adv, diff, degen


@jax.jit
def _joint_kernel(
    Z: Array,
    all_idx: Array,
    all_cls_idx: Array,
    mask: Array,
    counts: Array,
    epsilon: Array,
    max_iter: Array,
    tol: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Joint (point-cloud) optimal-transport indices for every replicate.

    For every input and class, transports the unconditional output cloud
    onto the class's conditional cloud with entropic regularization and
    aggregates the per-class costs into one index per input.

    Args:
        Z: Output point cloud ``(N, E)`` (already standardized when
            requested by the caller).
        all_idx: Replicate row indices ``(R, N)`` (row 0 is the identity).
        all_cls_idx: Class indices ``(R, D, M, P)``.
        mask: Class validity mask ``(M, P)``.
        counts: True class sizes ``(M,)``.
        epsilon: Entropic regularization strength (scalar).
        max_iter: Sinkhorn iteration cap (scalar).
        tol: Sinkhorn marginal stopping tolerance (scalar).

    Returns:
        ``(ot, advective, diffusive, errs, degenerate)`` with index arrays
        of shape ``(R, D)``, per-solve marginal residuals ``(R, D, M)``,
        and a ``(R,)`` flag marking constant (zero-variance) replicate
        clouds, which yield zero indices.
    """
    dtype = jnp.result_type(Z.dtype, jnp.float32)
    Z = Z.astype(dtype)
    maskf = mask.astype(dtype)
    countsf = counts.astype(dtype)
    n_total = countsf.sum()  # == N
    weights = countsf / n_total
    M = mask.shape[0]
    log_b_all = jnp.where(mask, -jnp.log(countsf)[:, None], -jnp.inf)  # (M, P)

    def _one_replicate(carry, xs):
        r, cls_idx = xs  # (N,), (D, M, P)
        Z_r = Z[r]  # resampled cloud
        mean_all = Z_r.mean(axis=0)
        V = 2.0 * jnp.var(Z_r, axis=0, ddof=1).sum()  # == 2 * Tr(Cov)
        V_safe = jnp.where(V > 0, V, 1.0)
        sq_full = (Z_r**2).sum(axis=-1)  # (N,)
        D = cls_idx.shape[0]

        def _one_class(dm: Array):
            """Transport cost of the full cloud onto class dm % M of input dm // M."""
            idx = cls_idx[dm // M, dm % M]  # (P,)
            m = dm % M
            Z_c = Z[idx]  # (P, E)
            # Squared Euclidean cost block (N, P); padded columns carry
            # zero target mass, so their (finite) costs are inert.
            C = sq_full[:, None] + (Z_c**2).sum(axis=-1)[None, :] - 2.0 * (Z_r @ Z_c.T)
            C = jnp.maximum(C, 0.0)
            cost, err = _sinkhorn_w2(C, log_b_all[m], epsilon, max_iter, tol)
            cls_mean = (Z_c * maskf[m][:, None]).sum(axis=0) / countsf[m]
            adv = ((cls_mean - mean_all) ** 2).sum()
            return cost, adv, err

        # Sequential map keeps peak memory at one (N, P) cost block.
        costs, advs, errs = jax.lax.map(_one_class, jnp.arange(D * M))
        w2 = costs.reshape(D, M)
        adv = advs.reshape(D, M)
        diff = jnp.maximum(w2 - adv, 0.0)

        def _aggregate(per_class: Array) -> Array:
            val = (weights[None, :] * per_class).sum(axis=-1) / V_safe
            return jnp.where(V > 0, val, 0.0)

        out = (_aggregate(w2), _aggregate(adv), _aggregate(diff), errs.reshape(D, M), V <= 0)
        return carry, out

    _, (ot, adv, diff, errs, degen) = jax.lax.scan(_one_replicate, None, (all_idx, all_cls_idx))
    return ot, adv, diff, errs, degen


def _boot_conf(vals_all: Array, hat: Array, degen_all: Array, conf_level: float) -> Array:
    """Percentile CI endpoints with degenerate replicates neutralized.

    A constant bootstrap resample carries no information; it contributes
    the point estimate instead of a spurious zero so it neither widens
    nor shifts the interval (borgonovo convention).

    Args:
        vals_all: Replicate values ``(R, ..., D)`` (row 0 is the point
            estimate).
        hat: Point estimate ``(..., D)``.
        degen_all: Degeneracy flags ``(R, ...)``.
        conf_level: Two-sided confidence level.

    Returns:
        ``(2, ..., D)`` array of ``[lower, upper]`` endpoints.
    """
    degen_boot = degen_all[1:][..., None]
    boot_vals = jnp.where(degen_boot, hat[None], vals_all[1:])
    return _percentile_ci(boot_vals, conf_level)


def analyze(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    mode: Literal["separate", "joint", "joint-over-time"] = "separate",
    n_partitions: int = 25,
    standardize: bool = True,
    epsilon: float = 0.01,
    max_iter: int = 1000,
    tol: float | None = None,
    dummy: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    seed: int = 0,
    chunk_size: int | None = None,
) -> OTResult:
    """Compute optimal-transport sensitivity indices from given data.

    The OT index measures how much knowing an input's value displaces the
    *entire* output distribution: it is the class-averaged squared
    2-Wasserstein distance between the output distribution conditional on
    the input and the unconditional one, normalized to [0, 1] by twice
    the output variance. 0 means the output distribution is unaffected by
    the input, 1 means it is fully determined by it. Unlike variance-based
    indices it reacts to changes in spread, tails and shape; the returned
    decomposition separates the location-shift (``advective``) part --
    exactly half the given-data first-order Sobol index -- from the
    spread/shape (``diffusive``) remainder. Any ``(X, Y)`` sample works;
    no special design is needed.

    Conditioning classes are built from the inputs' ordinal *ranks*, which
    are invariant under monotone transforms -- the estimator is therefore
    distribution-free in X and works unchanged for uniform, Gaussian,
    truncated-Gaussian, or mixed marginals (no CDF transform involved).
    Correlated inputs are supported: the index is well-defined under
    dependence and then measures each input's *total* association with
    the output, including effects mediated by correlated inputs (the
    same reading as the given-data S1, whose half remains the advective
    component).

    Args:
        problem: Problem definition with D parameters.
        X: Input sample matrix ``(N, D)``.
        Y: Model output ``(N,)``, ``(N, K)``, or ``(N, T, K)``.
        mode: Output treatment. ``"separate"`` (default) scores every
            output column independently with exact 1-D optimal transport
            (indices ``(T, K, D)``, squeezed). ``"joint"`` treats the
            whole (flattened) output vector as one point cloud and yields
            a single index per input over the joint output distribution
            (``(D,)``), using entropic Sinkhorn transport.
            ``"joint-over-time"`` does the same per output, treating each
            output's time course as the cloud (``(K, D)``); requires a
            3-D ``Y``.
        n_partitions: Number of equal-frequency conditioning classes per
            input. More classes localize the conditioning (less
            discretization bias) but leave fewer samples per class (more
            estimation noise); ~25 is customary for the OT index at
            N >= 2500.
        standardize: Joint modes only: divide each output column by its
            standard deviation before building the transport cost, so no
            single output dominates the joint distance through its units.
            Ignored in ``"separate"`` mode (each column is normalized by
            its own variance regardless).
        epsilon: Joint modes only: entropic regularization strength,
            relative to the cost matrix scaled to [0, 1]. Smaller values
            approach exact transport at the price of more iterations.
        max_iter: Joint modes only: Sinkhorn iteration cap per solve.
        tol: Joint modes only: stopping tolerance on the L1 target-
            marginal violation. ``None`` selects ``1e-9`` in float64 and
            ``1e-6`` in float32 (tighter is unresolvable there). A single
            warning is emitted if any solve fails to converge.
        dummy: Also push one synthetic input -- independent of the output
            by construction -- through the identical pipeline and report
            its index as ``ot_dummy``. This estimates the index floor a
            fully irrelevant input receives from finite-sample and
            (in joint modes) entropic bias; inputs not clearly above it
            are indistinguishable from noise.
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. 0 (default) skips them (``*_conf`` are ``None``).
            Joint modes solve ``n_bootstrap * D * n_partitions``
            transport problems -- keep it modest there.
        conf_level: Confidence level for percentile bootstrap intervals.
        seed: Random seed for bootstrap resampling and the dummy input.
        chunk_size: ``"separate"`` mode: number of flattened ``T*K``
            output columns processed per kernel call; ``None`` picks a
            memory-aware default. Accepted but inert in joint modes
            (their peak memory is bounded by one ``(N, N/M)`` cost block
            per solve).

    Returns:
        OTResult with total, advective and diffusive indices, optional
        confidence intervals, and the optional dummy baseline. All
        indices are 0 for a constant (zero-variance) output slice rather
        than NaN. In joint modes the entropic and finite-sample bias
        keeps indices of irrelevant inputs strictly positive -- compare
        against ``ot_dummy`` rather than against 0.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, Y is not 1-D/2-D/3-D, X and Y have differing row
            counts, ``mode`` is unknown, ``mode="joint-over-time"`` is
            used with a non-3-D Y, ``n_partitions`` is not in
            ``[2, N // 2]``, ``epsilon <= 0``, ``max_iter < 1``,
            ``tol <= 0``, ``n_bootstrap < 0``, ``conf_level`` is not in
            ``(0, 1)``, or ``chunk_size`` is not a positive integer.
    """
    X = jnp.asarray(X)
    Y, _ = _validate_xy_inputs(problem, X, Y)

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode == "joint-over-time" and Y.ndim != 3:
        raise ValueError(f"mode='joint-over-time' requires a 3-D (N, T, K) Y, got ndim={Y.ndim}")
    N = X.shape[0]
    if not 2 <= n_partitions <= N // 2:
        raise ValueError(f"n_partitions must be in [2, N//2={N // 2}], got {n_partitions}")
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol is not None and not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K = Y_3d.shape
    D = problem.num_vars
    M = int(n_partitions)

    dtype = jnp.result_type(Y_3d.dtype, jnp.float32)
    if tol is None:
        tol = 1e-9 if dtype == jnp.float64 else 1e-6

    key_boot, key_dummy = jax.random.split(jax.random.PRNGKey(seed))

    X_eff = X
    if dummy:
        # A synthetic input that is independent of Y by construction; only
        # its ranks matter, so a permutation of 0..N-1 suffices. It rides
        # through the pipeline as one extra input column.
        dummy_col = jax.random.permutation(key_dummy, N).astype(X.dtype)
        X_eff = jnp.concatenate([X, dummy_col[:, None]], axis=1)
    D_eff = X_eff.shape[1]

    # Replicate 0 is the identity permutation (the original sample); the
    # remaining rows are the bootstrap resamples. Building them together
    # means the point estimate and its interval share one code path.
    identity = jnp.arange(N, dtype=jnp.int32)[None, :]
    if n_bootstrap > 0:
        boot = jax.random.randint(key_boot, (n_bootstrap, N), 0, N, dtype=jnp.int32)
        all_idx = jnp.concatenate([identity, boot], axis=0)
    else:
        all_idx = identity

    take_np, mask_np, sizes_np = _class_layout(N, M)
    take = jnp.asarray(take_np)
    mask = jnp.asarray(mask_np)
    counts = jnp.asarray(sizes_np)
    # Rank the inputs once for every replicate, reused by every kernel call.
    all_cls_idx = _build_class_indices(X_eff, all_idx, take)  # (R, D_eff, M, P)

    R = all_idx.shape[0]
    errs: Array | None = None

    if mode == "separate":
        Y_cols = Y_3d.reshape(N, T * K)
        total = T * K
        if chunk_size is None:
            chunk_size = max(1, _CHUNK_ELEM_BUDGET // (D_eff * M * N))
        cs = min(chunk_size, total)
        parts: tuple[list, list, list, list] = ([], [], [], [])
        for start in range(0, total, cs):
            out = _ot_1d_kernel(Y_cols[:, start : start + cs], all_idx, all_cls_idx, mask, counts)
            for part, arr in zip(parts, out):
                part.append(arr)
        ot_all = jnp.concatenate(parts[0], axis=1).reshape(R, T, K, D_eff)
        adv_all = jnp.concatenate(parts[1], axis=1).reshape(R, T, K, D_eff)
        diff_all = jnp.concatenate(parts[2], axis=1).reshape(R, T, K, D_eff)
        degen_all = jnp.concatenate(parts[3], axis=1).reshape(R, T, K)
    else:
        eps_s = jnp.asarray(epsilon, dtype)
        max_iter_s = jnp.asarray(max_iter, jnp.int32)
        tol_s = jnp.asarray(tol, dtype)
        if mode == "joint":
            Z = Y_3d.reshape(N, T * K)
            if standardize:
                Z, _, _, _ = _prenormalize_outputs(Z)
            ot_all, adv_all, diff_all, errs, degen_all = _joint_kernel(
                Z, all_idx, all_cls_idx, mask, counts, eps_s, max_iter_s, tol_s
            )  # (R, D_eff)
        else:  # joint-over-time
            outs = []
            for k in range(K):
                Z_k = Y_3d[:, :, k]
                if standardize:
                    Z_k, _, _, _ = _prenormalize_outputs(Z_k)
                outs.append(
                    _joint_kernel(
                        Z_k, all_idx, all_cls_idx, mask, counts, eps_s, max_iter_s, tol_s
                    )
                )
            ot_all = jnp.stack([o[0] for o in outs], axis=1)  # (R, K, D_eff)
            adv_all = jnp.stack([o[1] for o in outs], axis=1)
            diff_all = jnp.stack([o[2] for o in outs], axis=1)
            errs = jnp.stack([o[3] for o in outs], axis=1)  # (R, K, D_eff, M)
            degen_all = jnp.stack([o[4] for o in outs], axis=1)  # (R, K)

    if errs is not None:
        # One host sync for the whole analysis; the solver itself never
        # raises (exceptions cannot cross a traced while_loop).
        n_bad = int((errs > tol).sum())
        if n_bad:
            warnings.warn(
                f"gsax: {n_bad} of {errs.size} Sinkhorn solves did not reach "
                f"tol={tol:g} within max_iter={max_iter}; results use the last "
                "iterate (consider raising max_iter or epsilon)",
                stacklevel=2,
            )

    ot_hat = ot_all[0]
    adv_hat = adv_all[0]
    diff_hat = diff_all[0]

    ot_conf: Array | None = None
    adv_conf: Array | None = None
    diff_conf: Array | None = None
    if n_bootstrap > 0:
        ot_conf = _boot_conf(ot_all, ot_hat, degen_all, conf_level)
        adv_conf = _boot_conf(adv_all, adv_hat, degen_all, conf_level)
        diff_conf = _boot_conf(diff_all, diff_hat, degen_all, conf_level)

    ot_dummy: Array | None = None
    if dummy:
        # The synthetic column is the last input; split it off everywhere.
        ot_dummy = ot_hat[..., D]
        ot_hat, adv_hat, diff_hat = ot_hat[..., :D], adv_hat[..., :D], diff_hat[..., :D]
        if ot_conf is not None and adv_conf is not None and diff_conf is not None:
            ot_conf = ot_conf[..., :D]
            adv_conf = adv_conf[..., :D]
            diff_conf = diff_conf[..., :D]

    if mode == "separate":
        ot_hat = _squeeze_output_axes(ot_hat, squeeze_time, squeeze_output)
        adv_hat = _squeeze_output_axes(adv_hat, squeeze_time, squeeze_output)
        diff_hat = _squeeze_output_axes(diff_hat, squeeze_time, squeeze_output)
        if ot_conf is not None and adv_conf is not None and diff_conf is not None:
            ot_conf = _squeeze_output_axes(ot_conf, squeeze_time, squeeze_output)
            adv_conf = _squeeze_output_axes(adv_conf, squeeze_time, squeeze_output)
            diff_conf = _squeeze_output_axes(diff_conf, squeeze_time, squeeze_output)
        if ot_dummy is not None:
            ot_dummy = _squeeze_output_axes(ot_dummy, squeeze_time, squeeze_output, n_trailing=0)

    return OTResult(
        ot=ot_hat,
        ot_conf=ot_conf,
        advective=adv_hat,
        advective_conf=adv_conf,
        diffusive=diff_hat,
        diffusive_conf=diff_conf,
        ot_dummy=ot_dummy,
        mode=mode,
        problem=problem,
    )
