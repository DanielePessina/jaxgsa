"""The Sobol estimator formulas, one function per named estimator.

A Saltelli design gives four output vectors per base point: ``A``, ``B``,
``AB_j`` (``A`` with column j taken from ``B``) and, when the design was
drawn with ``calc_second_order=True``, ``BA_j``. Several different
estimators turn those vectors into a first-order index ``S1`` and a
total-order index ``ST``. They all converge to the same numbers, and they
differ in how much sampling noise they carry, and in whether they can
return a value outside ``[0, 1]``.

``jaxgsa.sobol.analyze`` and ``jaxgsa.sobol.indices`` choose one with
``estimator=``. This module holds every formula, and nothing else: no
shape handling, no diagnostics, no policy.

The named schemes
-----------------

Each name picks a first-order formula and a total-order formula together.

=================== ============================ =========================
Name                First order                  Total order
=================== ============================ =========================
``saltelli-jansen`` Sobol'-Mauntz, Saltelli 2010 Jansen 1999
``jansen``          Jansen 1999                  Jansen 1999
``janon-monod``     Janon 2014 / Monod 2006      Janon 2014 / Monod 2006
``martinez``        Martinez 2011                Martinez 2011
``mauntz-kucherenko`` Sobol' 2007                Sobol' 2007
``azzini-rosati``   Azzini, Mara, Rosati 2021    Azzini, Mara, Rosati 2021
=================== ============================ =========================

``saltelli-jansen`` is the default: Jansen's total-order estimator is a
mean of squares, so it can never go negative, and it is also SALib's
default, which keeps cross-library comparisons plain. Users screen on
``ST``, and a negative ``ST`` invites the clipping that jaxgsa refuses to
do.

A note on the first-order label. The default's first-order formula is
``E[B (AB_j - A)] / Var(Y)``. That is not the Saltelli (2002) estimator,
which is the plain cross-moment ``E[B AB_j] - f0^2``. It is the improved
form of Sobol', Tarantola, Gatelli, Kucherenko and Mauntz (2007), which
Saltelli et al. (2010) tabulate and recommend. So ``mauntz-kucherenko``
and ``saltelli-jansen`` share their first-order formula on purpose, and
differ only in the total order.

Variance normaliser
-------------------

``saltelli-jansen``, ``jansen`` and ``mauntz-kucherenko`` divide by a
pooled output variance over ``concat(A, B)``, that is ``2N`` points.
``janon-monod`` and ``martinez`` build their own denominator from the
same pair of vectors they use in the numerator, which is the point of
those estimators. ``azzini-rosati`` shares one denominator between its
first-order and total-order formulas, which is what makes it hold
``S1 <= ST`` for every sample.

Sample cost
-----------

Every scheme except ``azzini-rosati`` reads only ``A``, ``B`` and ``AB``,
so it runs on the ``N(D+2)`` design. ``azzini-rosati`` also reads ``BA``,
so it needs the ``N(2D+2)`` design, that is ``calc_second_order=True``.

Negative estimates
------------------

Every first-order estimator here can return a negative index. That is not
a defect, and it is not a sign that the sample is too small.

The mechanism is the same for all of them. The index is formed as a
difference of two correlated Monte Carlo estimates: a cross-moment minus
a squared mean, or one minus a mean square. The difference is unbiased
but noisy, so when the true index is near zero the sampling error is
larger than the index, and roughly half the estimates land below zero.
Owen (2013) states it directly: the cross-moment form "has very large
variance when tau^2_u << mu^2", whereas a squared-difference form "is a
sum of squares, hence nonnegative. If the true tau^2_u = 0, then the
estimate is 0 with probability one".

Note what that last sentence does *not* say about first order. A Jansen
first-order estimate is *one minus* a sum of squares, so it is bounded
above by 1 and unbounded below (measured -0.59 on Sobol-G at
``base_n = 32``); only the total order is a bare sum of squares and
therefore non-negative. ``ST`` cannot go negative; ``S1`` can.

jaxgsa's measurements agree, on Sobol-G with 40 seeds at
``base_n`` = 64 / 256 / 1024 / 4096: the default ``saltelli-jansen``
pairing returns a negative first-order value for one parameter or
another in 100 / 88 / 65 / 30 per cent of runs, ``jansen`` in
50 / 57 / 30 / 45 per cent, and ``azzini-rosati`` in 0 per cent, every
time. The rate falls with N for the default pairing; it does not fall
monotonically for ``jansen``. Negative values are expected, not a bug.

What is non-negative, exactly and on every sample:

* the total order of ``saltelli-jansen`` and ``jansen``, both Jansen's
  mean of squares.
* nothing in ``janon-monod``, ``martinez`` or ``mauntz-kucherenko``.

``azzini-rosati`` is the only scheme that holds ``S1_j <= ST_j`` on every
sample. The others cross that bound often at a small N: on Sobol-G with
64 base points, roughly 40 per cent of their per-parameter estimates do.

Investigate a negative value only if it is large, if it appears for a
parameter whose index is demonstrably not near zero, or if it grows with
N. Do not clip a negative value before ranking parameters. Clipping is a
display choice and nothing more: it biases upward in exactly the
near-zero regime where the ranking decision is being made.

References
----------

- Jansen, M.J.W. (1999). Analysis of variance designs for model output.
  *Computer Physics Communications* 117(1-2):35-43.
- Sobol', I.M., Tarantola, S., Gatelli, D., Kucherenko, S., Mauntz, W.
  (2007). Estimating the approximation error when fixing unessential
  factors in global sensitivity analysis. *Reliability Engineering and
  System Safety* 92(7):957-960.
- Saltelli, A. et al. (2010). Variance based sensitivity analysis of
  model output. *Computer Physics Communications* 181(2):259-270.
- Monod, H., Naud, C., Makowski, D. (2006). Uncertainty and sensitivity
  analysis for crop models. In *Working with Dynamic Crop Models*.
- Janon, A., Klein, T., Lagnoux, A., Nodet, M., Prieur, C. (2014).
  Asymptotic normality and efficiency of two Sobol index estimators.
  *ESAIM: Probability and Statistics* 18:342-364.
- Martinez, J.-M. (2011). Analyse de sensibilite globale par
  decomposition de la variance. GdR MASCOT-NUM presentation.
- Azzini, I., Mara, T.A., Rosati, R. (2021). Comparison of two sets of
  Monte Carlo estimators of Sobol' indices. *Environmental Modelling and
  Software* 144:105167.
- Owen, A.B. (2013). Better estimation of small Sobol' sensitivity
  indices. *ACM TOMACS* 23(2):11.
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jax import Array

Estimator = Literal[
    "saltelli-jansen",
    "jansen",
    "janon-monod",
    "martinez",
    "mauntz-kucherenko",
    "azzini-rosati",
]
"""The name of a first-order plus total-order estimator pair."""

ESTIMATORS: tuple[str, ...] = (
    "saltelli-jansen",
    "jansen",
    "janon-monod",
    "martinez",
    "mauntz-kucherenko",
    "azzini-rosati",
)
"""Every accepted ``estimator=`` value, in the order the error message lists them."""

DEFAULT_ESTIMATOR: Estimator = "saltelli-jansen"
"""The estimator used when the caller does not ask for one."""

NEEDS_BA: frozenset[str] = frozenset({"azzini-rosati"})
"""Estimators that read the BA blocks, so they need ``calc_second_order=True``."""


def requires_second_order_design(estimator: str) -> bool:
    """Report whether an estimator needs the BA blocks of the Saltelli design.

    Args:
        estimator: One of :data:`ESTIMATORS`.

    Returns:
        True when the estimator can only run on a design drawn with
        ``calc_second_order=True``.
    """
    return estimator in NEEDS_BA


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _pooled_inv_var(A: Array, B: Array) -> Array:
    """Return ``1 / Var(concat(A, B))``, or NaN when the output is constant.

    Centering before squaring avoids the catastrophic cancellation that
    afflicts the naive ``E[x^2] - E[x]^2`` form for large-magnitude outputs.
    The divisor is ``2N``, matching ``jnp.var``'s ``ddof=0`` convention.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        Scalar Array with the reciprocal pooled variance.
    """
    N = A.shape[0]
    pooled_mean = (jnp.mean(A) + jnp.mean(B)) / 2.0
    A_c = A - pooled_mean
    B_c = B - pooled_mean
    var = (jnp.sum(A_c**2) + jnp.sum(B_c**2)) / (2 * N)
    return jnp.where(var == 0, jnp.nan, 1.0 / var)


def _safe_ratio(numerator: Array, denominator: Array) -> Array:
    """Divide, returning NaN where the denominator is exactly zero.

    Args:
        numerator: Any shape.
        denominator: Broadcastable to ``numerator``.

    Returns:
        The elementwise ratio, NaN where the denominator vanished.
    """
    safe = jnp.where(denominator == 0, 1.0, denominator)
    return jnp.where(denominator == 0, jnp.nan, numerator / safe)


def _paired_ratio(u: Array, v: Array) -> Array:
    """Janon-Monod ratio for a pair of output vectors that share some inputs.

    Both the numerator and the denominator use the same pooled mean
    ``E[(u + v) / 2]`` and the same pooled second moment, taken over the two
    vectors together. That is the whole idea of the estimator: a shared,
    self-consistent normaliser has an asymptotic variance no larger than the
    classical one, with equality only at an index of exactly 0 or 1.

    Args:
        u: One member of the pair, shape ``(N, 1)`` or ``(N, D)``.
        v: The other member, broadcastable against ``u``, shape ``(N, D)``.

    Returns:
        The estimated variance fraction explained by the inputs the pair
        holds in common, shape ``(D,)``.
    """
    mean_uv = jnp.mean((u + v) / 2.0, axis=0)
    numerator = jnp.mean(u * v, axis=0) - mean_uv**2
    denominator = jnp.mean((u**2 + v**2) / 2.0, axis=0) - mean_uv**2
    return _safe_ratio(numerator, denominator)


def _correlation(u: Array, v: Array) -> Array:
    """Column-wise Pearson correlation between ``u`` and each column of ``v``.

    Args:
        u: One member of the pair, shape ``(N, 1)`` or ``(N, D)``.
        v: The other member, shape ``(N, D)``.

    Returns:
        Correlation per column, shape ``(D,)``. NaN where either vector has
        zero spread.
    """
    u_c = u - jnp.mean(u, axis=0)
    v_c = v - jnp.mean(v, axis=0)
    covariance = jnp.mean(u_c * v_c, axis=0)
    spread = jnp.sqrt(jnp.mean(u_c**2, axis=0) * jnp.mean(v_c**2, axis=0))
    return _safe_ratio(covariance, spread)


# ---------------------------------------------------------------------------
# First-order plus total-order formulas. Signature: (A, AB, B) -> (S1, ST)
# ---------------------------------------------------------------------------


def _jansen(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Jansen (1999) for both orders.

    Both numerators are mean squared differences. That makes the total
    order non-negative and, unlike the first-order estimators, exactly
    unbiased: ``E[(A - AB_j)^2] / 2 = V_Tj``, so an inert input reads
    ``ST = 0`` (measured 40/40 out of 40 runs on Sobol-G). First order is
    *one minus* such a term, so it is bounded above by 1 and can still
    come out negative::

        S1_j = 1 - E[(B - AB_j)^2] / (2 Var(Y))
        ST_j =     E[(A - AB_j)^2] / (2 Var(Y))

    ``B`` and ``AB_j`` share only column j, so their squared difference
    estimates the variance that everything *except* j explains. ``A`` and
    ``AB_j`` share everything except column j, so theirs estimates the
    variance j is responsible for, interactions included.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    inv_var = _pooled_inv_var(A, B)
    S1 = 1.0 - 0.5 * jnp.mean((B[:, None] - AB) ** 2, axis=0) * inv_var
    ST = 0.5 * jnp.mean((A[:, None] - AB) ** 2, axis=0) * inv_var
    return S1, ST


def _janon_monod(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Janon et al. (2014), the estimator introduced by Monod et al. (2006).

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    # (B, AB_j) share column j alone, so their shared variance is S1_j.
    S1 = _paired_ratio(B[:, None], AB)
    # (A, AB_j) share every column but j, so their shared variance is the
    # part j is not responsible for; one minus it is the total order.
    ST = 1.0 - _paired_ratio(A[:, None], AB)
    return S1, ST


def _martinez(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Martinez (2011): the same pairings as Janon-Monod, read as correlations.

    Using the empirical correlation coefficient is what lets Martinez's
    intervals come from a Fisher transform rather than a bootstrap. jaxgsa
    does not use that interval; the point estimate is the same quantity
    either way.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    S1 = _correlation(B[:, None], AB)
    ST = 1.0 - _correlation(A[:, None], AB)
    return S1, ST


def _mauntz_kucherenko(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Sobol' et al. (2007), the improved formulas for both orders::

        S1_j = E[B (AB_j - A)] / Var(Y)
        ST_j = E[A (A - AB_j)] / Var(Y)

    The first-order formula is the one the default scheme also uses. The
    total-order formula differs from Jansen: it is not a sum of squares, so
    it can return a small negative value. It also carries the uncentred
    second moment ``E[A^2]``, which would make it noisier next to a large
    output mean; in practice this is moot, because every caller standardizes
    ``A`` and ``B`` to mean 0 before either estimator formula runs (see
    :func:`jaxgsa.sobol._analyze._separate_output_values`).

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    inv_var = _pooled_inv_var(A, B)
    S1 = jnp.mean(B[:, None] * (AB - A[:, None]), axis=0) * inv_var
    ST = jnp.mean(A[:, None] * (A[:, None] - AB), axis=0) * inv_var
    return S1, ST


def _azzini_rosati(A: Array, AB: Array, BA: Array, B: Array) -> tuple[Array, Array]:
    """Azzini, Mara and Rosati (2021), which reads all four blocks.

    Write ``u_j = AB_j - A`` and ``v_j = B - BA_j``. Both are the effect of
    moving column j from its A value to its B value, once with the other
    columns held at A and once with them held at B. Then::

        S1_j = 2 * sum(u_j v_j)         / sum((A - B)^2 + (BA_j - AB_j)^2)
        ST_j = sum(u_j^2 + v_j^2)       / sum((A - B)^2 + (BA_j - AB_j)^2)

    One denominator serves both, and ``2uv <= u^2 + v^2`` for every sample,
    so this scheme holds ``S1_j <= ST_j`` exactly, on any sample, which
    neither Saltelli nor Jansen does. First order can still go negative,
    when the cross term is negative. The shared denominator also makes both
    formulas robust to a large output mean, because every term in it is a
    difference, so the mean cancels before anything is squared.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        BA: Model outputs from each cross-matrix BA_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    u = AB - A[:, None]
    v = B[:, None] - BA
    # E[(A - B)^2] = 2 Var(Y), and AB_j and BA_j share no column at all, so
    # E[(BA_j - AB_j)^2] = 2 Var(Y) too. The sum therefore estimates 4 Var(Y).
    denominator = jnp.sum((A - B)[:, None] ** 2 + (BA - AB) ** 2, axis=0)
    S1 = _safe_ratio(2.0 * jnp.sum(u * v, axis=0), denominator)
    ST = _safe_ratio(jnp.sum(u**2 + v**2, axis=0), denominator)
    return S1, ST


def _saltelli_jansen(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """The default pairing: Sobol'-Mauntz first order, Jansen total order.

    Built from the two formulas this module already has, rather than a
    separately maintained fused kernel: ``_mauntz_kucherenko``'s ``S1`` and
    ``_jansen``'s ``ST`` are exactly the formulas ``saltelli-jansen`` uses.
    An earlier version of this module hand-fused the two into one function
    that computed the pooled variance once and shared it, and kept a comment
    claiming that fusion was needed for the default scheme to produce
    bit-for-bit its historical numbers. It was verified bitwise identical to
    this construction, in float32 and float64, under ``jit``
    (``scratchpad/fix/sobol/repro_fused_vs_generic.py``), so the fusion
    bought nothing but a second place to keep the same two formulas in sync.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        ``(S1, ST)``, each of shape ``(D,)``.
    """
    S1, _ = _mauntz_kucherenko(A, AB, B)
    _, ST = _jansen(A, AB, B)
    return S1, ST


_FIRST_TOTAL: dict[str, Callable[[Array, Array, Array], tuple[Array, Array]]] = {
    "saltelli-jansen": _saltelli_jansen,
    "jansen": _jansen,
    "janon-monod": _janon_monod,
    "martinez": _martinez,
    "mauntz-kucherenko": _mauntz_kucherenko,
}


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def first_total_kernel(
    estimator: str,
) -> Callable[[Array, Array, Array], tuple[Array, Array]]:
    """Return the ``(A, AB, B) -> (S1, ST)`` function for a named estimator.

    Args:
        estimator: One of :data:`ESTIMATORS`, excluding those in
            :data:`NEEDS_BA`.

    Returns:
        A pure JAX function of the three output vectors.

    Raises:
        KeyError: If the name needs the BA blocks, or is not an estimator.
            Both entry points validate the name and the design before
            reaching here, so either is a programming error.
    """
    # requires_second_order_design (backed by NEEDS_BA) is the one place that
    # says which estimators need the BA blocks; this used to ask the same
    # question a second way, by dict membership, which could drift from it.
    if requires_second_order_design(estimator):
        raise KeyError(
            f"{estimator!r} has no first-order-only kernel; use second_order_kernel for it"
        )
    return _FIRST_TOTAL[estimator]


def second_order_kernel(
    estimator: str,
) -> Callable[[Array, Array, Array, Array], tuple[Array, Array, Array]]:
    """Return the ``(A, AB, BA, B) -> (S1, ST, S2)`` function for an estimator.

    The second-order formula is Saltelli (2002) for every scheme::

        V_jk  = E[BA_j AB_k - A B] / Var(Y)
        S2_jk = V_jk - S1_j - S1_k

    Only the ``S1_j`` and ``S1_k`` it subtracts come from the chosen scheme.
    No published pairwise estimator exists for most of these schemes, and
    inventing one would be worse than reusing the one that is published.

    Args:
        estimator: One of :data:`ESTIMATORS`.

    Returns:
        A pure JAX function of the four output vectors.
    """

    def kernel(A: Array, AB: Array, BA: Array, B: Array) -> tuple[Array, Array, Array]:
        if estimator == "azzini-rosati":
            S1, ST = _azzini_rosati(A, AB, BA, B)
        else:
            S1, ST = _FIRST_TOTAL[estimator](A, AB, B)
        inv_var = _pooled_inv_var(A, B)
        Vjk = (
            jnp.mean(
                BA[:, :, None] * AB[:, None, :] - (A * B)[:, None, None],
                axis=0,
            )
            * inv_var
        )
        S2 = Vjk - S1[:, None] - S1[None, :]
        return S1, ST, S2

    return kernel
