"""Kucherenko index estimators over the conditional-copula design.

Write ``y = X_i`` and ``z = X_{~i}``. The design supplies, for each base point
``k``: a joint row ``(y_k, z_k)``, a first-order row ``(y_k, z'_k)`` with
``z' ~ p(z | y_k)``, and a total row ``(y'_k, z_k)`` with ``y' ~ p(y | z_k)``.
The implemented estimators are the single-loop forms of Kucherenko, Tarantola
& Annoni (2012), per output slice with ``N`` base points:

.. code-block:: text

    f0_A   = (1/N) sum_k f(y_k, z_k)
    f0_B,i = (1/N) sum_k f(y_k, z'_k)
    V      = (1/N) sum_k f(y_k, z_k)^2  -  f0_A^2

    S1_i = [ (1/N) sum_k f(y_k, z_k) f(y_k, z'_k)  -  f0_A f0_B,i ] / V
    ST_i = [ (1/(2N)) sum_k ( f(y_k, z_k) - f(y'_k, z_k) )^2 ] / V

``S1_i`` estimates ``V(E(Y|X_i)) / V(Y)``. The product pairs two draws that
share ``y_k``, so its expectation is ``E[E(Y|y)^2]``. Subtracting the mean
product then leaves the variance of the conditional mean. The mean correction
uses the product of the two block means, since each block estimates ``f0``.
That cancels the shared Monte-Carlo error of the two blocks. With ``f0_A^2``
the estimator carries a strictly larger O(1/N) bias. The implementation
subtracts one shared shift (the joint-block mean) from both factors before the
product. The estimator is algebraically identical, but the arithmetic stays
centered when ``Y`` carries a large mean.

``ST_i`` estimates ``E(V(Y|X_{~i})) / V(Y)``. The squared difference pairs two
draws of ``y`` under the same ``z_k``, which is the Jansen form of the
conditional variance.

Under an identity correlation the conditional draws are plain independent
draws and both formulas reduce exactly to the classic Saltelli column-swap
estimators: the Sobol'/Homma-Saltelli form for ``S1`` and the Jansen form for
``ST``.

**This module has no pure core.** Every other method exposes an ``indices()``
that returns bare arrays and survives ``jit``, ``vmap`` and ``jacrev``.
Kucherenko does not, and is not going to: it is host NumPy end to end, and
that is why it is the fastest method in the library. Its work is many small
operations on modest arrays, where dispatch and host-device transfer dominate,
so a JAX rewrite measured slower, and in float32 less accurate as well. That
is a declared exemption (see ``jaxgsa._core.registry.MethodRecord.pure_core``),
not a gap: no traceable core exists to export.

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
    Jansen (1999). Comput. Phys. Commun. 117:35-43.
"""

from __future__ import annotations

from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.bootstrap import interval
from jaxgsa._core.entry import _user_stacklevel, bootstrap_checks, prepare
from jaxgsa._core.invalid import OnInvalid, _unit_of_row_for_policy
from jaxgsa._core.irregular import (
    IrregularResult,
    IrregularY,
    _fwd_kwargs,
    _is_irregular_y,
    analyze_irregular,
)
from jaxgsa._core.result import CIInfo
from jaxgsa._core.sampling import require_design
from jaxgsa._core.validation import (
    _warn_zero_variance_slices,
    as_key,
)
from jaxgsa.kucherenko._result import KucherenkoResult
from jaxgsa.kucherenko._sampling import KucherenkoSamples

# Keyword-only parameters the irregular engine forwards unchanged to the
# per-channel recursive analyze() calls. verbose is excluded:
# the engine owns it.
_IRREGULAR_KWARGS = (
    "n_bootstrap",
    "conf_level",
    "ci_method",
    "key",
    "on_invalid",
    "keep_replicates",
)


def _closed_estimate(
    f_joint: np.ndarray, f_block: np.ndarray, safe_variance: np.ndarray
) -> np.ndarray:
    """Single-loop closed (first-order) estimator for a stack of blocks.

    Args:
        f_joint: Joint-block outputs, shape ``(N, S)``.
        f_block: Outputs of blocks that keep some columns of the joint rows,
            shape ``(M, N, S)``.
        safe_variance: Output variance with ``NaN`` where it is not positive,
            shape ``(S,)``.

    Returns:
        ``V(E(Y | kept columns)) / V(Y)`` per block, shape ``(M, S)``.
    """
    N = f_joint.shape[0]
    # Shift both factors by one shared constant (the joint-block mean)
    # before the product. The estimator is algebraically unchanged: for any
    # shift c, mean((a-c)(b-c)) - mean(a-c) mean(b-c) == mean(ab) - mean(a)
    # mean(b). Numerically it is decisive: the uncentered product form loses
    # the O(1) covariance in rounding when Y carries a large mean, because
    # mean(f_joint f_first) and f0_joint f0_first are both O(mean^2). The ST
    # estimator is a squared difference and the variance is a central moment,
    # so both are already shift-safe.
    f0_joint = f_joint.mean(axis=0)  # (S,)
    g_joint = f_joint - f0_joint  # (N, S), centered
    g_block = f_block - f0_joint[None, None]  # (M, N, S), same shift
    # The cross-moment is a sum over the sample axis of a per-(m, s) dot
    # product. The broadcast-multiply-then-mean form materialises a
    # (M, N, S) intermediate; einsum accumulates the same dot without it.
    # Both operate on the *centred* factors, so the shift-safety the
    # estimator was designed for is unchanged (the two agree to ~1e-17 in
    # float64, pure reassociation noise).
    return (
        np.einsum("dns,ns->ds", g_block, g_joint) / N
        - g_joint.mean(axis=0)[None] * g_block.mean(axis=1)
    ) / safe_variance


def _jansen_estimate(
    f_joint: np.ndarray, f_block: np.ndarray, safe_variance: np.ndarray
) -> np.ndarray:
    """Jansen squared-difference estimator of ``E(V(Y | kept columns)) / V(Y)``.

    The arguments are as in :func:`_closed_estimate`. The result has shape
    ``(M, S)`` and is exactly 0 when the model ignores the redrawn columns.
    """
    return 0.5 * ((f_joint[None] - f_block) ** 2).mean(axis=1) / safe_variance


def _estimate(
    f_joint: np.ndarray,
    f_first: np.ndarray,
    f_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run both single-loop estimators on one block of base points.

    One copy of the arithmetic, so a bootstrap replicate cannot drift from
    the point estimate it is an interval around.

    Args:
        f_joint: Joint-block outputs, shape ``(N, S)``.
        f_first: First-order block outputs, shape ``(D, N, S)``.
        f_total: Total block outputs, shape ``(D, N, S)``.

    Returns:
        ``(S1, ST, variance)``, the first two shaped ``(D, S)`` and the last
        ``(S,)``.
    """
    variance = f_joint.var(axis=0)  # (S,)
    safe_variance = np.where(variance > 0.0, variance, np.nan)
    S1 = _closed_estimate(f_joint, f_first, safe_variance)
    # ST is a squared difference and the variance a central moment, so both
    # are already shift-safe.
    ST = _jansen_estimate(f_joint, f_total, safe_variance)
    return S1, ST, variance


def _estimate_subsets(
    f_a: np.ndarray, f_b: np.ndarray, f_blocks: np.ndarray, n_params: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed and total index of every subset from the ``subsets="all"`` design.

    Block ``m`` keeps the columns of the bitmask ``m`` and redraws the others
    from their conditional. It gives two estimates:

    * the closed index of ``m``, ``V(E(Y | X_m)) / V(Y)``, with the same
      arithmetic as ``S1`` (Kucherenko, Tarantola & Annoni 2012);
    * the total index of the complement of ``m``,
      ``E(V(Y | X_m)) / V(Y)``, with the same Jansen form as ``ST``.

    The variance is taken over the two independent joint blocks ``A`` and
    ``B`` together. The full set has closed and total index 1, and the empty
    set is left out.

    Args:
        f_a: Outputs of the joint block ``A``, shape ``(N, S)``.
        f_b: Outputs of the independent joint block ``B``, shape ``(N, S)``.
        f_blocks: Outputs of blocks for the masks ``1 .. 2^D - 2``, shape
            ``(2^D - 2, N, S)``.
        n_params: Number of parameters ``D``.

    Returns:
        ``(closed, total, variance)``. The first two are shaped
        ``(2^D - 1, S)`` with mask ``m`` at row ``m - 1``. ``variance`` is
        ``(S,)``.
    """
    n_masks = 2**n_params - 1
    variance = np.concatenate([f_a, f_b], axis=0).var(axis=0)
    safe_variance = np.where(variance > 0.0, variance, np.nan)
    closed_blocks = _closed_estimate(f_a, f_blocks, safe_variance)
    total_of_complement = _jansen_estimate(f_a, f_blocks, safe_variance)
    one = np.where(variance > 0.0, 1.0, np.nan)
    closed = np.concatenate([closed_blocks, one[None]], axis=0)
    # total(u) comes from the block that keeps the complement of u.
    complement = np.arange(1, n_masks) ^ n_masks  # for u = 1 .. 2^D - 2
    total = np.concatenate([total_of_complement[complement - 1], one[None]], axis=0)
    return closed, total, variance


def analyze(
    samples: KucherenkoSamples,
    Y: Array | IrregularY,
    *,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> KucherenkoResult | IrregularResult:
    """Compute Kucherenko first-order and total indices from model outputs.

    The estimators run on the actual model outputs. No surrogate is fitted.
    This is the design-based counterpart to :func:`jaxgsa.vkoga.analyze`. It
    estimates the same two conditional-variance quantities, but it evaluates
    the model on a conditional-copula design instead of querying a fitted
    kernel surrogate. Under independent inputs the indices are the classic
    Sobol' ``S1`` and ``ST``.

    A design drawn with ``subsets="all"`` also gives the closed index
    ``closed`` and the total index ``subset_total`` of every subset of the
    inputs. ``result.shapley()`` and ``result.pme()`` turn them into exact
    Shapley effects and proportional marginal effects. The singleton entries
    come from the singleton blocks, and the variance is taken over the two
    independent joint blocks ``A`` and ``B`` together.

    Args:
        samples: Design from :func:`jaxgsa.kucherenko.sample`.
        Y: Model outputs evaluated at each row of
            ``samples.samples``, in the same row order. Accepted
            shapes: ``(n_runs,)``, ``(n_runs, K)``, or ``(n_runs, T, K)``.
            Indices are computed independently for every output slice.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
            ``0`` (default) computes no intervals. The resampling unit is one
            **base point**, which carries the joint row and all ``2D``
            conditional rows drawn around it. Base points are i.i.d., so
            resampling them is a clean bootstrap of the design; resampling
            individual rows would not be, because dropping part of a
            conditional block would leave the estimator reading rows that no
            longer line up. ``S1`` and ``ST`` get intervals, and so do
            ``closed`` and ``subset_total`` on a ``subsets="all"`` design. ``variance``
            does not: it is the denominator of both indices, a property of
            the joint block rather than a sensitivity measure, and its
            uncertainty is already inside the two index intervals.
        conf_level: Confidence level for the bootstrap intervals.
        ci_method: How the interval endpoints are formed. ``"quantile"``
            (default) reads them off the empirical bootstrap distribution.
            ``"gaussian"`` centres them on the point estimate and takes
            ``+/- z * sd`` of the bootstrap draws, which is smoother for a
            small ``n_bootstrap`` but assumes the draws are normal.
        key: A ``jax.random`` key for the bootstrap resampling. Required when
            ``n_bootstrap > 0``. Pass ``jax.random.key(0)`` if you have an
            integer seed.
        on_invalid: What to do about non-finite model outputs. The unit here
            is one base point, which appears once in the joint block and once
            in each of the ``2D`` conditional blocks, so a single bad value
            removes all ``2D + 1`` of its rows. ``"raise"`` (the default)
            refuses the sample, ``"propagate"`` lets the value reach the
            indices, and ``"drop"`` analyzes the surviving base points. See
            :mod:`jaxgsa._core.invalid`.
        verbose: If ``True`` (default), print a short summary to stdout: the
            problem and the data, the wall-clock timing, and the top
            parameters by ``ST``. Pass ``False`` for a silent run.
        keep_replicates: Keep the per-resample indices on
            ``KucherenkoResult.ci.replicates``. Off by default because they
            are large: ``n_bootstrap`` copies of two index arrays, or of four
            on a ``subsets="all"`` design. ``result.shapley()`` and
            ``result.pme()`` need them to give an interval.

    Returns:
        A :class:`KucherenkoResult` with ``S1`` and ``ST`` shaped ``(D,)``,
        ``(K, D)``, or ``(T, K, D)`` to mirror ``Y``, the optional ``S1_conf``
        and ``ST_conf`` intervals with the ``ci`` record that describes them,
        plus the ``invalid`` report.

    Raises:
        TypeError: If ``samples`` is not a ``KucherenkoSamples``.
        ValueError: If ``Y``'s shape violates the output contract; if
            ``on_invalid`` is not one of the three policies; if the sample
            holds a non-finite value under ``on_invalid="raise"``; if fewer
            than 2 base points survive a drop; if ``n_bootstrap`` is negative,
            ``conf_level`` is not in ``(0, 1)``, ``ci_method`` is not
            ``"quantile"`` or ``"gaussian"``, or ``n_bootstrap > 0`` and no
            ``key`` was given.

    Warns:
        JaxgsaWarning: If any output slice has zero variance (its indices are
            NaN), or if non-finite outputs are found under a policy that does
            not raise.
    """
    require_design(
        samples, KucherenkoSamples, "jaxgsa.kucherenko.analyze", "jaxgsa.kucherenko.sample"
    )
    key = as_key(key, "kucherenko")
    if _is_irregular_y(Y):
        return analyze_irregular(
            analyze,
            channel_data=samples,
            problem=samples.problem,
            Y=Y,
            n_expected=int(samples.n_runs),
            design_based=True,
            verbose=verbose,
            kwargs=_fwd_kwargs(locals(), _IRREGULAR_KWARGS),
        )
    Y = cast(Array, Y)

    from jaxgsa.kucherenko import _SPEC

    problem = samples.problem
    N = samples.base_n
    D = samples.n_params
    all_subsets = samples.subsets == "all"
    n_blocks = samples.n_blocks

    ctx = prepare(
        _SPEC,
        problem,
        Y,
        on_invalid=on_invalid,
        checks=bootstrap_checks(n_bootstrap, ci_method, conf_level, key),
        n_expected=samples.n_runs,
        # A base point k feeds the joint row and every conditional row with
        # the same k, so one non-finite output anywhere invalidates all of
        # them. The design is block-major, so base point k sits at rows
        # k, N + k, 2N + k, … rather than in a contiguous run.
        n_units=N,
        # The map is the interleaved base-point pattern, built on the host
        # only where the check reads it (never under on_invalid='skip').
        unit_of_row=_unit_of_row_for_policy(on_invalid, lambda: np.tile(np.arange(N), n_blocks)),
        min_kept=2,
        # The denominator of both indices is the variance of the joint block
        # alone, not of the whole design, so the zero-variance check has to
        # wait until that block has been separated out below.
        warn_zero_variance=False,
    )
    keep, invalid = ctx.keep, ctx.invalid
    Y_canonical = ctx.Y3
    n_time, n_out = Y_canonical.shape[1], Y_canonical.shape[2]
    n_slices = n_time * n_out
    # (n_blocks, N, S) — block-major, matching the design's stacking order.
    F = np.asarray(Y_canonical, dtype=np.float64).reshape(n_blocks, N, n_slices)
    if not keep.all():
        F = F[:, keep, :]

    f_joint = F[0]  # (N, S)
    if all_subsets:
        f_other = F[1]  # (N, S) — independent joint block B
        f_sub = F[2:]  # (2^D - 2, N, S) — block for mask m at row m - 1
        f_first = f_total = None
    else:
        f_first = F[1 : D + 1]  # (D, N, S) — z redrawn given x_i
        f_total = F[D + 1 :]  # (D, N, S) — x_i redrawn given z

    # Warned on the caller's own output rank, so a (N, K) output reads "k=1"
    # and not the made-up "(t=0, k=1)" the inserted time axis would give.
    _warn_zero_variance_slices(
        f_joint.reshape((-1, *ctx.Y.shape[1:])),
        output_names=problem.output_names,
        method="jaxgsa.kucherenko.analyze",
        stacklevel=_user_stacklevel(),
    )
    t0 = _verbose.tic()
    closed = subset_total = None
    # The singleton indices of a subsets="all" design sit at the masks 2^i:
    # S1_i from the block that keeps {i}, ST_i from the block that keeps every
    # column but i (read through the total index of {i}).
    singles = (1 << np.arange(D)) - 1
    if all_subsets:
        closed, subset_total, variance = _estimate_subsets(f_joint, f_other, f_sub, D)
        S1, ST = closed[singles], subset_total[singles]
    else:
        assert f_first is not None and f_total is not None
        S1, ST, variance = _estimate(f_joint, f_first, f_total)

    def _shape(index: np.ndarray) -> Array:
        """Reshape a ``(..., M, S)`` block to the output contract, ``M`` last."""
        lead = index.shape[:-2]
        arr = jnp.asarray(
            np.moveaxis(index, -2, -1).reshape(*lead, n_time, n_out, index.shape[-2])
        )
        return ctx.squeeze(arr, n_trailing=1)

    S1_conf: Array | None = None
    ST_conf: Array | None = None
    closed_conf: Array | None = None
    subset_total_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        assert key is not None  # refused by the missing-key check above
        n_kept = f_joint.shape[0]
        idx = np.asarray(
            jax.random.randint(key, shape=(n_bootstrap, n_kept), minval=0, maxval=n_kept)
        )

        # One transpose makes each resample's gather contiguous.
        f_sub_t = np.ascontiguousarray(f_sub.transpose(1, 0, 2)) if all_subsets else None

        def replicate(i: np.ndarray) -> tuple[np.ndarray, ...]:
            """Index estimates of one resample of the base points.

            One resample takes the same base points out of every block, so
            every conditional row still sits beside the joint row it was
            drawn around.
            """
            if all_subsets:
                assert f_sub_t is not None
                return _estimate_subsets(f_joint[i], f_other[i], f_sub_t[i].transpose(1, 0, 2), D)[
                    :2
                ]
            assert f_first is not None and f_total is not None
            return _estimate(f_joint[i], f_first[:, i], f_total[:, i])[:2]

        replicates = [replicate(i) for i in idx]
        boot_first = np.stack([r[0] for r in replicates])  # (R, D or 2^D - 1, S)
        boot_second = np.stack([r[1] for r in replicates])
        # _shape already squeezes to the caller's own rank; interval() reads
        # a shape only to broadcast [lower, upper], so it works the same on
        # this squeezed layout as on the raw (T, K, D) one.
        point = {"S1": _shape(S1), "ST": _shape(ST)}
        if all_subsets:
            assert closed is not None and subset_total is not None
            point |= {"closed": _shape(closed), "subset_total": _shape(subset_total)}
            draws = {
                "S1": _shape(boot_first[:, singles]),
                "ST": _shape(boot_second[:, singles]),
                "closed": _shape(boot_first),
                "subset_total": _shape(boot_second),
            }
        else:
            draws = {"S1": _shape(boot_first), "ST": _shape(boot_second)}
        confs, ci = interval(
            point,
            draws,
            level=conf_level,
            method=ci_method,
            n_bootstrap=n_bootstrap,
            keep_replicates=keep_replicates,
        )
        S1_conf, ST_conf = confs["S1"], confs["ST"]
        if all_subsets:
            closed_conf, subset_total_conf = confs["closed"], confs["subset_total"]

    variance_shaped = ctx.squeeze(jnp.asarray(variance.reshape(n_time, n_out)), n_trailing=0)
    result = KucherenkoResult(
        S1=_shape(S1),
        ST=_shape(ST),
        problem=problem,
        variance=variance_shaped,
        invalid=invalid,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        ci=ci,
        closed=None if closed is None else _shape(closed),
        subset_total=None if subset_total is None else _shape(subset_total),
        closed_conf=closed_conf,
        subset_total_conf=subset_total_conf,
    )

    if verbose:
        elapsed = _verbose.stop(t0, result.S1, result.ST)
        n_kept = f_joint.shape[0]
        _verbose.analysis_summary(
            method="jaxgsa.kucherenko.analyze",
            problem=problem,
            dependence=problem.dependence,
            n_runs=n_blocks * n_kept,
            T=n_time,
            K=n_out,
            invalid=invalid,
            timings=[("compute", elapsed)],
            notes=[
                f"design: {problem.dependence} "
                f"({'subsets=all, 2^D' if all_subsets else '2D+1'} = {n_blocks} "
                f"blocks of {n_kept} base points)"
            ],
            index_name="ST",
            values=result.ST,
            conf=result.ST_conf,
        )
    return result
