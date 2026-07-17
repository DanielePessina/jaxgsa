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

Three modes cover gsax's output shapes. ``"univariate"`` treats every
output column independently with the closed-form 1-D optimal transport
(sorted-quantile coupling; no solver): the unconditional sample supplies
quantiles at the N uniform mass points and each conditional class is
evaluated at the same points through its nearest-rank empirical quantile
function (midpoint rule ``j = floor((i + 0.5) * n_m / N)``, exact
whenever the class size divides N). ``"multivariate"`` and ``"trajectory"``
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

The estimator is implemented from the paper's published equations and
numerically validated against POT and analytic closed forms in the
test suite.

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
import numpy as np
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
# "univariate" mode. The dominant intermediate is the per-column conditional
# quantile tensor of size ``chunk_columns * D * M * N`` elements, so the
# default chunk width keeps it near this many float32 elements (~256 MB).
_CHUNK_ELEM_BUDGET = 1 << 26

_MODES = ("univariate", "multivariate", "trajectory")


def _aggregate_normalized(per_class: Array, weights: Array, V: Array) -> Array:
    """Class-weighted average of per-class costs, normalized by ``V``.

    This is the defining [0, 1] normalization of the OT index, shared by
    the 1-D and joint kernels so the two modes can never drift apart.

    Args:
        per_class: Per-class costs ``(D, M)``.
        weights: Class weights ``n_m / N`` summing to 1, shape ``(M,)``.
        V: Normalizer ``2 * Var`` (scalar); a non-positive value marks a
            constant output and yields exactly 0 instead of NaN.

    Returns:
        Normalized indices ``(D,)``.
    """
    V_safe = jnp.where(V > 0, V, 1.0)
    val = (weights[None, :] * per_class).sum(axis=-1) / V_safe
    return jnp.where(V > 0, val, 0.0)


def _quantile_ranks(sizes: "np.ndarray", N: int) -> "np.ndarray":
    """Nearest-rank conditional quantile index for every full-sample mass point.

    Midpoint rule ``j = floor((i + 0.5) * n_m / N)``; ``j < n_m`` always,
    so class padding is never touched. Computed host-side in float64
    because the products exceed float32's exact-integer range for large N
    (and the table depends only on the static partition layout anyway).

    Args:
        sizes: True class sizes ``(M,)`` from
            :func:`gsax._partition._class_layout`.
        N: Number of samples.

    Returns:
        int32 lookup table ``(M, N)`` into each class's sorted members.
    """
    i_grid = np.arange(N, dtype=np.float64) + 0.5
    j = np.floor(i_grid[None, :] * sizes[:, None].astype(np.float64) / N).astype(np.int32)
    return np.minimum(j, (sizes - 1)[:, None].astype(np.int32))


@jax.jit
def _ot_1d_kernel(
    Y_cols: Array,
    all_idx: Array,
    all_cls_idx: Array,
    mask: Array,
    counts: Array,
    j: Array,
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
        j: Quantile lookup table ``(M, N)`` from :func:`_quantile_ranks`.

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
    weights = countsf / N  # (M,)

    def _col_stats(y: Array, r: Array, cls_idx: Array):
        """OT/advective/diffusive indices for one column and replicate."""
        y_r = y[r]  # resampled column
        y_sorted = jnp.sort(y_r)
        mean_r = y_r.mean()
        V = 2.0 * jnp.var(y_r, ddof=1)
        # Same predicate the aggregation zeroes on, so the CI
        # neutralization can never disagree with the reported zeros.
        degenerate = ~(V > 0)

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

        return (
            _aggregate_normalized(w2, weights, V),
            _aggregate_normalized(adv, weights, V),
            _aggregate_normalized(diff, weights, V),
            degenerate,
        )

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
        ``(ot, advective, diffusive, n_bad, degenerate)`` with index
        arrays of shape ``(R, D)``, a per-replicate count ``(R,)`` of
        Sinkhorn solves whose marginal residual stayed above ``tol``, and
        a ``(R,)`` flag marking constant (zero-variance) replicate
        clouds, which yield zero indices.
    """
    dtype = jnp.result_type(Z.dtype, jnp.float32)
    Z = Z.astype(dtype)
    maskf = mask.astype(dtype)
    countsf = counts.astype(dtype)
    N = Z.shape[0]
    weights = countsf / N
    M = mask.shape[0]
    log_b_all = jnp.where(mask, -jnp.log(countsf)[:, None], -jnp.inf)  # (M, P)

    def _one_replicate(carry, xs):
        r, cls_idx = xs  # (N,), (D, M, P)
        Z_r = Z[r]  # resampled cloud
        mean_all = Z_r.mean(axis=0)
        V = 2.0 * jnp.var(Z_r, axis=0, ddof=1).sum()  # == 2 * Tr(Cov)
        sq_full = (Z_r**2).sum(axis=-1)  # (N,)
        D = cls_idx.shape[0]

        def _one_class(dm: Array):
            """Transport cost of the full cloud onto class dm % M of input dm // M."""
            idx = cls_idx[dm // M, dm % M]  # (P,)
            m = dm % M
            Z_c = Z[idx]  # (P, E)
            # Squared Euclidean cost block (N, P). Padded columns carry
            # zero target mass, so zeroing their costs is exact -- and
            # necessary: pads are clamped duplicates of a real sample, and
            # an outlier there would otherwise set the solver's max-cost
            # scale and change the effective regularization per class.
            C = sq_full[:, None] + (Z_c**2).sum(axis=-1)[None, :] - 2.0 * (Z_r @ Z_c.T)
            C = jnp.maximum(C, 0.0) * maskf[m][None, :]
            cost, err = _sinkhorn_w2(C, log_b_all[m], epsilon, max_iter, tol)
            cls_mean = (Z_c * maskf[m][:, None]).sum(axis=0) / countsf[m]
            adv = ((cls_mean - mean_all) ** 2).sum()
            return cost, adv, err

        # Sequential map keeps peak memory at one (N, P) cost block.
        costs, advs, errs = jax.lax.map(_one_class, jnp.arange(D * M))
        w2 = costs.reshape(D, M)
        adv = advs.reshape(D, M)
        diff = jnp.maximum(w2 - adv, 0.0)

        out = (
            _aggregate_normalized(w2, weights, V),
            _aggregate_normalized(adv, weights, V),
            _aggregate_normalized(diff, weights, V),
            (errs > tol).sum(),
            ~(V > 0),
        )
        return carry, out

    _, (ot, adv, diff, n_bad, degen) = jax.lax.scan(_one_replicate, None, (all_idx, all_cls_idx))
    return ot, adv, diff, n_bad, degen


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
    mode: Literal["univariate", "multivariate", "trajectory"] = "univariate",
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
        mode: Output treatment. ``"univariate"`` (default) scores every
            output column independently with exact 1-D optimal transport
            (indices ``(T, K, D)``, squeezed). ``"multivariate"`` treats the
            whole (flattened) output vector as one point cloud and yields
            a single index per input over the joint output distribution
            (``(D,)``), using entropic Sinkhorn transport.
            ``"trajectory"`` does the same per output, treating each
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
            Ignored in ``"univariate"`` mode (each column is normalized by
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
            (in the point-cloud modes) entropic bias; inputs not
            clearly above it
            are indistinguishable from noise.
        n_bootstrap: Number of bootstrap resamples for confidence
            intervals. 0 (default) skips them (``*_conf`` are ``None``).
            Joint modes solve ``n_bootstrap * D * n_partitions``
            transport problems -- keep it modest there.
        conf_level: Confidence level for percentile bootstrap intervals.
        seed: Random seed for bootstrap resampling and the dummy input.
        chunk_size: ``"univariate"`` mode: number of flattened ``T*K``
            output columns processed per kernel call; ``None`` picks a
            memory-aware default. Accepted but inert in the point-cloud
            modes
            (their peak memory is bounded by one ``(N, N/M)`` cost block
            per solve).

    Returns:
        OTResult with total, advective and diffusive indices, optional
        confidence intervals, and the optional dummy baseline. All
        indices are 0 for a constant (zero-variance) output slice rather
        than NaN. In the point-cloud modes the entropic and
        finite-sample bias
        keeps indices of irrelevant inputs strictly positive -- compare
        against ``ot_dummy`` rather than against 0.

    Raises:
        ValueError: If X is not 2-D, its column count does not match the
            problem, Y is not 1-D/2-D/3-D, X and Y have differing row
            counts, ``mode`` is unknown, ``mode="trajectory"`` is
            used with a non-3-D Y, ``n_partitions`` is not in
            ``[2, N // 2]``, ``epsilon <= 0``, ``max_iter < 1``,
            ``tol <= 0``, ``n_bootstrap < 0``, ``conf_level`` is not in
            ``(0, 1)``, or ``chunk_size`` is not a positive integer.
    """
    X = jnp.asarray(X)
    Y = _validate_xy_inputs(problem, X, Y)

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode == "trajectory" and Y.ndim != 3:
        raise ValueError(f"mode='trajectory' requires a 3-D (N, T, K) Y, got ndim={Y.ndim}")
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
    M = int(n_partitions)

    dtype = jnp.result_type(Y_3d.dtype, jnp.float32)
    if tol is None:
        tol = 1e-9 if dtype == jnp.float64 else 1e-6

    key_boot, key_dummy = jax.random.split(jax.random.PRNGKey(seed))

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
    quantile_j = jnp.asarray(_quantile_ranks(sizes_np, N))
    # Rank the inputs once for every replicate, reused by every kernel call.
    all_cls_idx = _build_class_indices(X, all_idx, take)  # (R, D, M, P)

    # `_run` maps replicate indices + class indices to
    # (ot, advective, diffusive, degenerate) arrays with the input axis
    # last, and accumulates Sinkhorn convergence stats for the one
    # end-of-analysis warning. Defining it per mode lets the dummy
    # baseline below reuse the identical estimator.
    n_bad_parts: list[Array] = []
    n_solves = 0

    if mode == "univariate":
        Y_cols = Y_3d.reshape(N, T * K)
        total = T * K

        def _run(idx: Array, cls_idx: Array) -> tuple[Array, Array, Array, Array]:
            R_run, D_run = idx.shape[0], cls_idx.shape[1]
            cs = chunk_size
            if cs is None:
                cs = max(1, _CHUNK_ELEM_BUDGET // (D_run * M * N))
            cs = min(cs, total)
            parts: tuple[list, list, list, list] = ([], [], [], [])
            for start in range(0, total, cs):
                out = _ot_1d_kernel(
                    Y_cols[:, start : start + cs], idx, cls_idx, mask, counts, quantile_j
                )
                for part, arr in zip(parts, out):
                    part.append(arr)
            ot, adv, diff, degen = (jnp.concatenate(p, axis=1) for p in parts)
            return (
                ot.reshape(R_run, T, K, D_run),
                adv.reshape(R_run, T, K, D_run),
                diff.reshape(R_run, T, K, D_run),
                degen.reshape(R_run, T, K),
            )
    else:
        eps_s = jnp.asarray(epsilon, dtype)
        max_iter_s = jnp.asarray(max_iter, jnp.int32)
        tol_s = jnp.asarray(tol, dtype)
        # One point cloud for "multivariate" (flattened output), one per output
        # for "trajectory"; the runner stacks per-cloud results on a
        # new output axis only in the latter case.
        if mode == "multivariate":
            clouds = [Y_3d.reshape(N, T * K)]
        else:
            clouds = [Y_3d[:, :, k] for k in range(K)]
        if standardize:
            clouds = [_prenormalize_outputs(Z)[0] for Z in clouds]

        def _run(idx: Array, cls_idx: Array) -> tuple[Array, Array, Array, Array]:
            nonlocal n_solves
            n_solves += len(clouds) * idx.shape[0] * cls_idx.shape[1] * M
            outs = [
                _joint_kernel(Z, idx, cls_idx, mask, counts, eps_s, max_iter_s, tol_s)
                for Z in clouds
            ]
            n_bad_parts.append(sum(o[3].sum() for o in outs))
            if mode == "multivariate":
                ot, adv, diff, _, degen = outs[0]  # (R, D) / (R,)
                return ot, adv, diff, degen
            return (
                jnp.stack([o[0] for o in outs], axis=1),  # (R, K, D)
                jnp.stack([o[1] for o in outs], axis=1),
                jnp.stack([o[2] for o in outs], axis=1),
                jnp.stack([o[4] for o in outs], axis=1),  # (R, K)
            )

    ot_all, adv_all, diff_all, degen_all = _run(all_idx, all_cls_idx)

    ot_dummy: Array | None = None
    if dummy:
        # A synthetic input that is independent of Y by construction; only
        # its ranks matter, so a permutation of 0..N-1 suffices. It runs
        # through the identical estimator as a single-replicate pass (the
        # baseline needs no bootstrap interval).
        dummy_col = jax.random.permutation(key_dummy, N)[:, None]
        dummy_cls_idx = _build_class_indices(dummy_col, identity, take)  # (1, 1, M, P)
        ot_dummy = _run(identity, dummy_cls_idx)[0][0][..., 0]

    if n_bad_parts:
        # One host sync for the whole analysis; the solver itself never
        # raises (exceptions cannot cross a traced while_loop).
        n_bad = int(sum(n_bad_parts))
        if n_bad:
            warnings.warn(
                f"gsax: {n_bad} of {n_solves} Sinkhorn solves did not reach "
                f"tol={tol:g} within max_iter={max_iter}; results use the last "
                "iterate (consider raising max_iter or epsilon)",
                stacklevel=2,
            )

    hats = {"ot": ot_all[0], "advective": adv_all[0], "diffusive": diff_all[0]}
    confs: dict[str, Array | None] = dict.fromkeys(hats, None)
    if n_bootstrap > 0:
        for name, vals in (("ot", ot_all), ("advective", adv_all), ("diffusive", diff_all)):
            ci = _boot_conf(vals, hats[name], degen_all, conf_level)
            if mode == "univariate":
                ci = _squeeze_output_axes(ci, squeeze_time, squeeze_output)
            confs[name] = ci

    if mode == "univariate":
        hats = {
            name: _squeeze_output_axes(val, squeeze_time, squeeze_output)
            for name, val in hats.items()
        }
        if ot_dummy is not None:
            ot_dummy = _squeeze_output_axes(ot_dummy, squeeze_time, squeeze_output, n_trailing=0)

    return OTResult(
        ot=hats["ot"],
        ot_conf=confs["ot"],
        advective=hats["advective"],
        advective_conf=confs["advective"],
        diffusive=hats["diffusive"],
        diffusive_conf=confs["diffusive"],
        ot_dummy=ot_dummy,
        mode=mode,
        problem=problem,
    )
