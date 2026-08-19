"""Internal helpers for RS-HDMR sensitivity analysis.

Everything here is pure JAX. The module evaluates the cubic B-spline basis,
builds its tensor products, and holds the F-distribution critical values and
the term-to-order map the F-test needs. The fit itself, including the
backfitting solve, the ANCOVA split and the F-test, is in
:mod:`jaxgsa.hdmr._fit`.
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------


def _bspline_basis(x: Array, m: int) -> Array:
    """Evaluate m+3 cubic B-spline basis functions at points x in [0, 1].

    Uses the standard piecewise cubic polynomial for uniform B-splines with
    knot spacing 1/m, scaled by m^3 to match SALib convention.

    Args:
        x: Evaluation points in [0, 1], shape ``(N,)``.
        m: Number of B-spline intervals.

    Returns:
        B: Basis matrix, shape ``(N, m+3)``.
    """
    n_basis = m + 3
    i = jnp.arange(n_basis, dtype=x.dtype)
    # Map each x to the local coordinate u within the support of each basis
    # function. The shift +3 centres the 4-wide support window [0,4) so that
    # the first basis function activates at x=0.
    u = x[:, None] * m - i[None, :] + 3.0

    # Cubic B-spline N_3(u): four degree-3 polynomial segments on support
    # intervals [0,1), [1,2), [2,3), [3,4). The partition of unity property
    # (sum of all basis values = 1) follows from the Cox-de Boor recursion.
    p1 = u**3 / 6.0
    p2 = (-3 * u**3 + 12 * u**2 - 12 * u + 4) / 6.0
    p3 = (3 * u**3 - 24 * u**2 + 60 * u - 44) / 6.0
    p4 = (4 - u) ** 3 / 6.0

    val = jnp.where(
        u < 0,
        0.0,
        jnp.where(
            u < 1,
            p1,
            jnp.where(
                u < 2,
                p2,
                jnp.where(
                    u < 3,
                    p3,
                    jnp.where(u <= 4, p4, 0.0),
                ),
            ),
        ),
    )
    # Scale by m^3: SALib convention so that integral over [0,1] of each basis
    # function is O(1) regardless of the number of knot intervals m.
    # Clamp negatives from floating-point undershoot at support boundaries.
    return jnp.maximum(val * (m**3), 0.0)


def _build_B1(X_n: Array, m: int) -> Array:
    """Build first-order B-spline basis for all dimensions.

    Args:
        X_n: Normalized inputs in [0, 1], shape ``(N, D)``.
        m: Number of B-spline intervals.

    Returns:
        B1: First-order basis, shape ``(N, m1, D)`` with m1 = m + 3.
    """
    # vmap over columns of X_n: each dimension gets its own 1-D basis evaluated
    # independently, then axes are permuted to (sample, basis, dimension).
    B1_T = jax.vmap(partial(_bspline_basis, m=m))(X_n.T)  # (D, N, m1)
    return B1_T.transpose(1, 2, 0)  # (N, m1, D)


@jax.jit
def _build_B2(B1: Array, c2: Array, beta: Array) -> Array:
    """Build second-order tensor product basis.

    JIT-compiled so XLA fuses the two index gathers into the multiply instead
    of materializing both (N, m2, n2) gather products as eager transients.

    Args:
        B1: First-order basis, shape ``(N, m1, D)``.
        c2: Integer parameter index pairs, shape ``(n2, 2)``.
        beta: Tensor-product basis index table, shape ``(m1^2, 2)``.

    Returns:
        B2: Second-order basis, shape ``(N, m1^2, n2)``.
    """
    # beta maps each flat index in [0, m1^2) to a pair of 1D basis indices,
    # replacing the explicit outer product B_i(x_a) otimes B_j(x_b) with
    # column-gather + elementwise multiply: phi_{ij}(x) = B_i(x_a) * B_j(x_b).
    left = B1[:, :, c2[:, 0]][:, beta[:, 0], :]  # (N, m2, n2)
    right = B1[:, :, c2[:, 1]][:, beta[:, 1], :]  # (N, m2, n2)
    return left * right


@jax.jit
def _build_B3(B1: Array, c3: Array, beta: Array) -> Array:
    """Build third-order tensor product basis.

    JIT-compiled so XLA fuses the three index gathers into the multiply
    instead of materializing all (N, m3, n3) gather products as eager
    transients.

    Args:
        B1: First-order basis, shape ``(N, m1, D)``.
        c3: Integer parameter index triples, shape ``(n3, 3)``.
        beta: Tensor-product basis index table, shape ``(m1^3, 3)``.

    Returns:
        B3: Third-order basis, shape ``(N, m1^3, n3)``.
    """
    # Three-way tensor product: phi_{ijk}(x) = B_i(x_a) * B_j(x_b) * B_k(x_c),
    # constructed via the same index-gather strategy as _build_B2.
    a = B1[:, :, c3[:, 0]][:, beta[:, 0], :]  # (N, m3, n3)
    b = B1[:, :, c3[:, 1]][:, beta[:, 1], :]
    c = B1[:, :, c3[:, 2]][:, beta[:, 2], :]
    return a * b * c


# ---------------------------------------------------------------------------
# F-test
# ---------------------------------------------------------------------------


def _f_ppf(q: float, d1: float, d2: float) -> float:
    """Compute F-distribution percent point function via bisection on betainc.

    Pure JAX, no scipy dependency. The arguments are the degrees of freedom,
    which come from the basis size and the sample count, so every value here
    is a Python float known before any array is touched.

    The bisection reads its own comparison on the host, and that would raise
    inside a trace, because a ``jnp`` call on a constant returns a tracer
    there. ``jax.ensure_compile_time_eval`` is what states the intent: these
    are compile-time constants, so evaluate them eagerly wherever this runs.
    Without it, ``hdmr.indices`` could not be jitted. With it, the value is
    the same one the eager path computes, bit for bit.
    """
    # The F-distribution CDF can be written as I_x(d1/2, d2/2) where
    # x = d1*F / (d1*F + d2) and I_x is the regularized incomplete beta.
    # Bisection on x in [0, 1] inverts that relationship.
    a, b = d1 / 2.0, d2 / 2.0
    lo, hi = 0.0, 1.0
    with jax.ensure_compile_time_eval():
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if float(jax.scipy.special.betainc(a, b, mid)) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
    # Convert from beta variable x back to F value: F = d2*x / (d1*(1-x)).
    x = (lo + hi) / 2.0
    return float(d2 * x / (d1 * (1.0 - x + 1e-30)))


@lru_cache(maxsize=128)
def _compute_f_crits(alpha: float, m1: int, m2: int, m3: int, R: int) -> Array:
    """Precompute F critical values for each order (outside JIT).

    Cached because it is expensive and it is pure. Each of the three values
    costs up to 100 bisection steps, and every step reads a
    ``jax.scipy.special.betainc`` back on the host, so the call takes about
    9 ms -- for a small fit that was the single largest fixed cost of
    ``analyze``, paid again on every call and on every bootstrap replicate
    with the same arguments.

    The cache key is five Python scalars and nothing else, which is what ADR
    0018 requires of anything reachable from a compilation cache: no array,
    no list, no container a pytree flattener would walk into. ``alpha``,
    ``m1``, ``m2`` and ``m3`` come from the method's own constants and the
    basis sizes; ``R`` is a row count. None of them is data. The cached value
    is a JAX array, which is immutable, so handing the same one to several
    callers cannot let one of them disturb another.

    Args:
        alpha: Significance level of the test.
        m1: Basis count of a first-order term.
        m2: Basis count of a second-order term.
        m3: Basis count of a third-order term.
        R: Number of sample rows.

    Returns:
        f_crits: Critical F values ``[order1, order2, order3]``, shape
            ``(3,)``.
    """
    crits = []
    for p in (m1, m2, m3):
        if R > p:
            crits.append(_f_ppf(alpha, float(p), float(R - p)))
        else:
            # Fewer samples than basis functions makes the F-test undefined.
            # Set inf so no term at this order can pass the significance test.
            crits.append(float("inf"))
    return jnp.array(crits)


def term_order_map(
    n: int,
    n1: int,
    n2: int,
    m1: int,
    m2: int,
    m3: int,
    f_crits: Array,
) -> tuple[Array, Array, Array]:
    """Map every HDMR term to the interaction order it belongs to.

    Terms are laid out in one flat axis: the ``n1`` single-parameter terms
    first, then the ``n2`` pairs, then the triples. The F-test needs three
    per-term lookups derived from that layout, and both the in-memory kernel
    (:func:`_f_test`) and the row-streamed fit
    (:func:`jaxgsa.hdmr._stream._fit_hdmr_streamed`) need exactly the same
    three. This is their single definition: the streamed fit is compared
    against the in-memory one by exact equality on the selected term set, so
    the two must agree bit for bit, which a second copy of the expressions
    cannot guarantee.

    Args:
        n: Total number of terms.
        n1: Number of first-order terms.
        n2: Number of second-order terms.
        m1: Basis count of a first-order term.
        m2: Basis count of a second-order term.
        m3: Basis count of a third-order term.
        f_crits: Critical F values per order, shape ``(3,)``.

    Returns:
        Tuple ``(p, f_crit_per_term, order_idx)``, each of shape ``(n,)``:
        the basis count of each term as float32, its critical F value, and
        its interaction order as a 0/1/2 index into the null-model tables.
    """
    term_idx = jnp.arange(n)
    p = jnp.where(
        term_idx < n1,
        m1,
        jnp.where(term_idx < n1 + n2, m2, m3),
    ).astype(jnp.float32)
    f_crit_per_term = jnp.where(
        term_idx < n1,
        f_crits[0],
        jnp.where(term_idx < n1 + n2, f_crits[1], f_crits[2]),
    )
    order_idx = jnp.where(term_idx < n1, 0, jnp.where(term_idx < n1 + n2, 1, 2))
    return p, f_crit_per_term, order_idx
