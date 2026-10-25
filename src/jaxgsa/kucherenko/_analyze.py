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

from typing import Literal, NamedTuple, cast, overload

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.batching import get_memory_budget, resolve_batch_size
from jaxgsa._core.bootstrap import draw_row_matrix, interval
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


class _HostBlocks:
    """Design blocks of ``Y`` on the host, converted to float64 one block at a time.

    The estimators read one ``(N, S)`` block at a time, so the float64 copy of
    the whole design (twice the size of a float32 ``Y``) never has to exist.
    Indexing with an integer converts that block (and drops the removed base
    points); indexing with a slice returns a narrower lazy view. An
    ``ndarray`` of shape ``(M, N, S)`` supports the same two operations, so
    every estimator below takes either.
    """

    def __init__(
        self,
        Y_blocks: np.ndarray,
        keep: np.ndarray | None,
        cols: slice,
        rows: range | None = None,
    ) -> None:
        """Wrap ``(n_blocks, N, S)`` outputs at their own dtype.

        Args:
            Y_blocks: The design outputs, blocks first. Not copied.
            keep: Boolean mask of the surviving base points, shape ``(N,)``,
                or ``None`` when every point survives.
            cols: The output slices to read.
            rows: The blocks this view covers. ``None`` means all of them.
        """
        self._Y = Y_blocks
        self._keep = keep
        self._cols = cols
        self._rows = range(Y_blocks.shape[0]) if rows is None else rows

    def __len__(self) -> int:
        """Number of blocks in the view."""
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...
    @overload
    def __getitem__(self, index: slice) -> _HostBlocks: ...
    def __getitem__(self, index: int | slice) -> np.ndarray | _HostBlocks:
        """Return block ``index`` as float64 ``(N, S)``, or a narrower view."""
        if isinstance(index, slice):
            return _HostBlocks(self._Y, self._keep, self._cols, self._rows[index])
        block = np.asarray(self._Y[self._rows[index], :, self._cols], dtype=np.float64)
        return block if self._keep is None else block[self._keep]

    def materialize(self) -> np.ndarray:
        """Every block of the view at once, shape ``(M, N, S)``, in float64."""
        return np.stack([self[m] for m in range(len(self))])


_Blocks = np.ndarray | _HostBlocks
"""An ``(M, N, S)`` stack of blocks: an array, or a lazy :class:`_HostBlocks`."""


class _JointMoments(NamedTuple):
    """What the per-block estimators reuse from the joint block.

    Attributes:
        f_joint: Joint-block outputs, shape ``(N, S)``.
        f0: Joint-block mean, the shared shift, shape ``(S,)``.
        g_joint: Joint block centred on ``f0``, shape ``(N, S)``.
        g_joint_mean: Mean of ``g_joint`` (rounding-level, kept for exactness),
            shape ``(S,)``.
        safe_variance: Output variance with ``NaN`` where it is not positive,
            shape ``(S,)``.
    """

    f_joint: np.ndarray
    f0: np.ndarray
    g_joint: np.ndarray
    g_joint_mean: np.ndarray
    safe_variance: np.ndarray

    @classmethod
    def of(cls, f_joint: np.ndarray, safe_variance: np.ndarray) -> _JointMoments:
        """Compute the moments of ``f_joint`` once, for every block after it."""
        f0 = f_joint.mean(axis=0)
        g_joint = f_joint - f0
        return cls(f_joint, f0, g_joint, g_joint.mean(axis=0), safe_variance)


def _closed_one(joint: _JointMoments, f_block: np.ndarray) -> np.ndarray:
    """Single-loop closed (first-order) estimate for one block.

    Both factors are shifted by one shared constant (the joint-block mean)
    before the product. The estimator is algebraically unchanged: for any
    shift c, mean((a-c)(b-c)) - mean(a-c) mean(b-c) == mean(ab) - mean(a)
    mean(b). Numerically it is decisive: the uncentered product form loses the
    O(1) covariance in rounding when Y carries a large mean, because
    mean(f_joint f_first) and f0_joint f0_first are both O(mean^2). The cross
    moment is an einsum, a per-slice dot product that never forms the
    ``(N, S)`` product.

    Args:
        joint: Moments of the joint block.
        f_block: Outputs of a block that keeps some columns of the joint rows,
            shape ``(N, S)``.

    Returns:
        ``V(E(Y | kept columns)) / V(Y)``, shape ``(S,)``.
    """
    g_block = f_block - joint.f0
    n = joint.f_joint.shape[0]
    cov = np.einsum("ns,ns->s", g_block, joint.g_joint) / n
    return (cov - joint.g_joint_mean * g_block.mean(axis=0)) / joint.safe_variance


def _jansen_one(joint: _JointMoments, f_block: np.ndarray) -> np.ndarray:
    """Jansen squared-difference estimate of ``E(V(Y | kept columns)) / V(Y)``.

    It is a squared difference over a central-moment variance, so it is
    shift-safe as it stands, and exactly 0 when the model ignores the redrawn
    columns.

    Args:
        joint: Moments of the joint block.
        f_block: Block outputs, shape ``(N, S)``.

    Returns:
        Shape ``(S,)``.
    """
    diff = joint.f_joint - f_block
    np.square(diff, out=diff)
    return 0.5 * diff.mean(axis=0) / joint.safe_variance


def _closed_estimate(
    f_joint: np.ndarray, f_block: _Blocks, safe_variance: np.ndarray
) -> np.ndarray:
    """Closed (first-order) estimate of every block in a stack, one block at a time.

    Args:
        f_joint: Joint-block outputs, shape ``(N, S)``.
        f_block: Block outputs, shape ``(M, N, S)``.
        safe_variance: Output variance with ``NaN`` where it is not positive,
            shape ``(S,)``.

    Returns:
        Shape ``(M, S)``. See :func:`_closed_one`.
    """
    joint = _JointMoments.of(f_joint, safe_variance)
    out = np.empty((len(f_block), f_joint.shape[1]))
    for m in range(len(f_block)):
        out[m] = _closed_one(joint, f_block[m])
    return out


def _jansen_estimate(
    f_joint: np.ndarray, f_block: _Blocks, safe_variance: np.ndarray
) -> np.ndarray:
    """Jansen estimate of every block in a stack, one block at a time.

    The arguments are as in :func:`_closed_estimate`. The result has shape
    ``(M, S)``. See :func:`_jansen_one`.
    """
    joint = _JointMoments.of(f_joint, safe_variance)
    out = np.empty((len(f_block), f_joint.shape[1]))
    for m in range(len(f_block)):
        out[m] = _jansen_one(joint, f_block[m])
    return out


def _estimate(
    f_joint: np.ndarray,
    f_first: _Blocks,
    f_total: _Blocks,
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
    f_a: np.ndarray, f_b: np.ndarray, f_blocks: _Blocks, n_params: int
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
    # One pass over the blocks for both estimates, so a lazy block is
    # converted once.
    joint = _JointMoments.of(f_a, safe_variance)
    closed_blocks = np.empty((len(f_blocks), f_a.shape[1]))
    total_of_complement = np.empty_like(closed_blocks)
    for m in range(len(f_blocks)):
        block = f_blocks[m]
        closed_blocks[m] = _closed_one(joint, block)
        total_of_complement[m] = _jansen_one(joint, block)
    one = np.where(variance > 0.0, 1.0, np.nan)
    closed = np.concatenate([closed_blocks, one[None]], axis=0)
    # total(u) comes from the block that keeps the complement of u.
    complement = np.arange(1, n_masks) ^ n_masks  # for u = 1 .. 2^D - 2
    total = np.concatenate([total_of_complement[complement - 1], one[None]], axis=0)
    return closed, total, variance


# Live float64 (N,) columns per output slice while the point estimators run
# one block at a time: the joint block and its centred copy, the converted
# block and its centred copy or squared difference, and the second joint
# block plus the pooled copy for the variance of subsets="all". The automatic
# slice chunk width is sized against this and the memory budget.
_KUCHERENKO_LIVE_COPIES = 6

# Largest (chunk, N) weight matrix the bootstrap builds, in elements (32 MiB).
_BOOT_WEIGHT_ELEMENTS = 1 << 22


def _count_weights(rows: np.ndarray, n: int) -> np.ndarray:
    """Resample counts as weights: ``W[r, k]`` is (times base point k drawn) / n.

    Args:
        rows: Resample indices, shape ``(R, n)``.
        n: Number of base points.

    Returns:
        Float64 array of shape ``(R, n)``. The mean of any per-point quantity
        ``x`` over resample ``r`` is ``W[r] @ x``.
    """
    W = np.empty((rows.shape[0], n), dtype=np.float64)
    for r, row in enumerate(rows):
        W[r] = np.bincount(row, minlength=n)
    W /= n
    return W


def _bootstrap_closed(
    W: np.ndarray,
    c_a: np.ndarray,
    f_blocks: np.ndarray,
    safe_variance: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Closed estimate of every block for a chunk of resamples.

    Equals :func:`_closed_estimate` on the gathered rows ``f[rows[r]]``. The
    covariance is taken about each resample's own means, ``sum_n w_rn
    (a_n - m_a,r)(b_n - m_b,r)``, in two passes. The one-pass form
    ``W@(ab) - m_a m_b`` cancels badly when a resample misses a far-off group
    of points. The cost is a ``(r, N, S)`` temporary, so the caller sizes the
    chunk ``r`` from the memory budget.

    Args:
        W: Count weights, shape ``(r, N)``.
        c_a: Joint block centred about each resample's mean, shape
            ``(r, N, S)``.
        f_blocks: Block outputs, shape ``(M, N, S)``.
        safe_variance: Resample variance with ``NaN`` where undefined,
            shape ``(r, S)``.
        shift: Global centring constant, shape ``(S,)``.

    Returns:
        Array of shape ``(r, M, S)``.
    """
    out = np.empty((W.shape[0], f_blocks.shape[0], c_a.shape[2]))
    for m in range(f_blocks.shape[0]):
        g_b = f_blocks[m] - shift
        c_b = g_b[None] - (W @ g_b)[:, None, :]
        out[:, m] = np.einsum("rn,rns,rns->rs", W, c_a, c_b) / safe_variance
    return out


def _bootstrap_jansen(
    W: np.ndarray, f_a: np.ndarray, f_blocks: np.ndarray, safe_variance: np.ndarray
) -> np.ndarray:
    """Jansen estimate of every block for a chunk of resamples.

    Equals :func:`_jansen_estimate` on gathered rows. The arguments are as in
    :func:`_bootstrap_closed`; the result has shape ``(R, M, S)``.
    """
    out = np.empty((W.shape[0], f_blocks.shape[0], f_a.shape[1]))
    for m in range(f_blocks.shape[0]):
        out[:, m] = 0.5 * (W @ (f_a - f_blocks[m]) ** 2) / safe_variance
    return out


def _bootstrap_estimates(
    rows: np.ndarray,
    f_joint: np.ndarray,
    f_blocks: np.ndarray,
    f_total: np.ndarray | None,
    f_other: np.ndarray | None,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap replicates of the index estimates, without gathering blocks.

    Each resample takes the same base points out of every block. The result
    is the point estimator applied to the gathered rows of every resample,
    in exact arithmetic. In floating point it agrees to rounding error of
    the centred sums (checked at rtol 1e-6, including a resample that misses
    outliers 1e8 sigma away). Peak memory is a few ``(r, N, S)`` blocks, with
    ``r`` set from the memory budget.

    Args:
        rows: Resample indices, shape ``(R, N)``.
        f_joint: Joint-block outputs, shape ``(N, S)``.
        f_blocks: First-order blocks ``(D, N, S)`` or, with ``f_other``, the
            subset blocks ``(2^D - 2, N, S)``.
        f_total: Total blocks ``(D, N, S)``, or ``None`` for ``subsets="all"``.
        f_other: Independent joint block ``B`` for ``subsets="all"``, else
            ``None``.
        n_params: Number of parameters ``D``.

    Returns:
        ``(first, second)`` replicates, each ``(R, M, S)`` with ``M = D`` or
        ``2^D - 1``.
    """
    n, n_slices = f_joint.shape
    shift = f_joint.mean(axis=0)
    g_a = f_joint - shift
    g_b = None if f_other is None else f_other - shift
    point_var = (
        f_joint.var(axis=0)
        if f_other is None
        else np.concatenate([f_joint, f_other], axis=0).var(axis=0)
    )
    # The weight matrix is (r, N); the centred temporaries are (r, N, S) and
    # about four are live at once, so r is bounded by both.
    tensor_rows = get_memory_budget() // (4 * 8 * n * n_slices)
    chunk = max(1, min(_BOOT_WEIGHT_ELEMENTS // n, tensor_rows))
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for start in range(0, rows.shape[0], chunk):
        W = _count_weights(rows[start : start + chunk], n)
        c_a = g_a[None] - (W @ g_a)[:, None, :]  # about each resample's mean
        if g_b is None:
            var = np.einsum("rn,rns,rns->rs", W, c_a, c_a)
        else:
            # Variance of the pooled A and B rows of the resample, about
            # the pooled mean.
            pooled = 0.5 * (W @ g_a + W @ g_b)
            ca = g_a[None] - pooled[:, None, :]
            cb = g_b[None] - pooled[:, None, :]
            var = 0.5 * (
                np.einsum("rn,rns,rns->rs", W, ca, ca) + np.einsum("rn,rns,rns->rs", W, cb, cb)
            )
            del ca, cb
        safe = np.where((var > 0.0) & (point_var > 0.0), var, np.nan)
        closed = _bootstrap_closed(W, c_a, f_blocks, safe, shift)
        if f_other is None:
            assert f_total is not None
            first_parts.append(closed)
            second_parts.append(_bootstrap_jansen(W, f_joint, f_total, safe))
        else:
            jansen = _bootstrap_jansen(W, f_joint, f_blocks, safe)
            one = np.where(np.isfinite(safe), 1.0, np.nan)[:, None]
            complement = np.arange(1, 2**n_params - 1) ^ (2**n_params - 1)
            first_parts.append(np.concatenate([closed, one], axis=1))
            second_parts.append(np.concatenate([jansen[:, complement - 1], one], axis=1))
    return np.concatenate(first_parts), np.concatenate(second_parts)


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
    layout = ctx.layout
    n_time, n_out = ctx.Y3.shape[1], ctx.Y3.shape[2]
    out_axes = ctx.Y.shape[1:]
    n_slices = n_time * n_out
    # A host view of the caller's Y, blocks first. On a CPU backend this is
    # zero-copy, and NumPy's reshape of it is a view. ctx is dropped so the
    # promoted copy it holds (ctx.Y3, a full copy of Y for rank 1 or 2) does
    # not stay alive through the estimators.
    Y_blocks = np.asarray(ctx.Y).reshape(n_blocks, N, n_slices)
    del ctx
    n_kept = int(keep.sum())

    t0 = _verbose.tic()
    idx = None
    if n_bootstrap > 0:
        assert key is not None  # refused by the missing-key check above
        idx = np.asarray(draw_row_matrix(key, n_bootstrap, n_kept))

    # The estimators work in float64 on the host. Converting all of Y at
    # once costs twice its float32 size, plus the estimators' temporaries on
    # top. Instead the point estimates read one (N, S) block at a time
    # (_HostBlocks), and only when even that is over the memory budget are
    # the output slices taken a chunk at a time. Every estimate is per slice,
    # so neither choice changes an index. The bootstrap reads every block of
    # a chunk many times, so it converts its chunk once, sized to half the
    # budget; it sizes its own resample temporaries from the budget.
    per_slice = 2 * n_blocks * N * 8 if n_bootstrap > 0 else _KUCHERENKO_LIVE_COPIES * N * 8
    width = resolve_batch_size(per_slice, n_slices, None)
    keep_mask = None if keep.all() else keep
    parts: list[tuple[np.ndarray, ...]] = []
    boot_parts: list[tuple[np.ndarray, np.ndarray]] = []
    joint_parts: list[np.ndarray] = []
    for start in range(0, n_slices, width):
        lazy = _HostBlocks(Y_blocks, keep_mask, slice(start, start + width))
        F: _Blocks = lazy.materialize() if idx is not None else lazy
        f_joint = F[0]  # (N, w)
        joint_parts.append(f_joint)
        if all_subsets:
            f_other = F[1]  # (N, w) — independent joint block B
            f_sub = F[2:]  # (2^D - 2, N, w) — block for mask m at row m - 1
            parts.append(_estimate_subsets(f_joint, f_other, f_sub, D))
            if idx is not None:
                assert isinstance(f_sub, np.ndarray)
                boot_parts.append(_bootstrap_estimates(idx, f_joint, f_sub, None, f_other, D))
        else:
            f_first = F[1 : D + 1]  # (D, N, w) — z redrawn given x_i
            f_total = F[D + 1 :]  # (D, N, w) — x_i redrawn given z
            parts.append(_estimate(f_joint, f_first, f_total))
            if idx is not None:
                assert isinstance(f_first, np.ndarray) and isinstance(f_total, np.ndarray)
                boot_parts.append(_bootstrap_estimates(idx, f_joint, f_first, f_total, None, D))
        del F

    # The joint block is 1 / n_blocks of the design, so the whole of it is
    # cheap to hold for the one zero-variance warning over every slice.
    f_joint = joint_parts[0] if len(joint_parts) == 1 else np.concatenate(joint_parts, axis=1)
    del joint_parts
    _warn_zero_variance_slices(
        # Read on the host, like the rest of this module (see its docstring).
        cast(Array, f_joint.reshape((-1, *out_axes))),
        output_names=problem.output_names,
        method="jaxgsa.kucherenko.analyze",
        stacklevel=_user_stacklevel(),
    )
    first_est, second_est, variance = (
        np.concatenate([part[i] for part in parts], axis=-1) for i in range(3)
    )
    closed = subset_total = None
    singles = (1 << np.arange(D)) - 1
    if all_subsets:
        closed, subset_total = first_est, second_est
        S1, ST = closed[singles], subset_total[singles]
    else:
        S1, ST = first_est, second_est

    def _shape(index: np.ndarray) -> Array:
        """Reshape a ``(..., M, S)`` block to the output contract, ``M`` last."""
        lead = index.shape[:-2]
        arr = jnp.asarray(
            np.moveaxis(index, -2, -1).reshape(*lead, n_time, n_out, index.shape[-2])
        )
        return layout.squeeze(arr, n_trailing=1)

    S1_conf: Array | None = None
    ST_conf: Array | None = None
    closed_conf: Array | None = None
    subset_total_conf: Array | None = None
    ci: CIInfo | None = None
    if n_bootstrap > 0:
        boot_first, boot_second = (
            np.concatenate([part[i] for part in boot_parts], axis=-1) for i in range(2)
        )  # (R, D or 2^D - 1, S)
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

    variance_shaped = layout.squeeze(jnp.asarray(variance.reshape(n_time, n_out)), n_trailing=0)
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
