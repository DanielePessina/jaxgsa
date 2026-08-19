"""The default Sobol estimator pair, and the fused kernels that apply it.

This module holds the ``saltelli-jansen`` scheme, which is what
``estimator=`` selects when the caller does not choose. The other five
schemes live in :mod:`jaxgsa.sobol._estimators`, which also documents the
whole menu and the references.

Attribution, because the labels are easy to get wrong. The first-order
formula here is the improved form of Sobol', Tarantola, Gatelli,
Kucherenko and Mauntz (2007), which Saltelli et al. (2010) tabulate and
recommend. It is *not* the Saltelli (2002) estimator, which is the plain
cross-moment. The total-order formula is Jansen (1999).

Notation
--------
- N : number of base samples drawn from the input parameter space.
- A, B : two independent ``(N, D)`` input sample matrices, where D is the
  number of parameters. The arrays passed to these functions are the
  matching model outputs, each of shape ``(N,)``.
- AB_j : model output when column j of A is replaced by column j of B.
  All other columns stay from A. Shape ``(N,)``.
- BA_j : model output when column j of B is replaced by column j of A.
  All other columns stay from B. Shape ``(N,)``.

Variance estimation
-------------------
All estimators normalise by a pooled output variance computed over the
concatenation of A and B, that is ``var(concat(A, B))`` with shape
``(2N,)``. Pooling both base-sample vectors gives a more robust variance
estimate than using A or B alone. It doubles the effective sample size and
stays unbiased, because A and B are identically distributed.
"""

import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Fused kernels: compute the pooled variance ONCE and derive all indices from
# it, vectorising all D parameters via broadcasting. A per-parameter form
# would recompute var(concat(A, B)) for every parameter j; fusing saves that
# D-fold cost on the most expensive reduction and JIT-compiles as one kernel.
# ---------------------------------------------------------------------------


def _fused_first_total(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Compute all S1 and ST indices with a single variance computation.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        S1: first-order Sobol indices, shape ``(D,)``.
        ST: total-order Sobol indices, shape ``(D,)``.
    """
    N = A.shape[0]
    # Centered-sum variance: Var = E[(x - mu)^2] over pooled A, B.
    # Centering before squaring avoids catastrophic cancellation that
    # afflicts the naive E[x^2] - E[x]^2 form for large-magnitude outputs.
    pooled_mean = (jnp.mean(A) + jnp.mean(B)) / 2.0
    A_c = A - pooled_mean
    B_c = B - pooled_mean
    # Divide by 2N (not 2N-1) to match jnp.var's default ddof=0 convention
    var = (jnp.sum(A_c**2) + jnp.sum(B_c**2)) / (2 * N)
    # Pre-invert once; multiply is cheaper than D separate divides in XLA
    inv_var = jnp.where(var == 0, jnp.nan, 1.0 / var)

    # [:, None] broadcasts (N,) to (N, 1), computing all D indices at once
    # without vmap: each column j of AB is paired with the full A and B.
    # Sobol' et al. (2007) S1 estimator: E[B * (AB_j - A)] / Var(Y)
    S1 = jnp.mean(B[:, None] * (AB - A[:, None]), axis=0) * inv_var  # (D,)
    # Jansen (1999) ST estimator: E[(A - AB_j)^2] / (2 Var(Y))
    ST = 0.5 * jnp.mean((A[:, None] - AB) ** 2, axis=0) * inv_var  # (D,)

    return S1, ST


def _fused_second_order(A: Array, AB: Array, BA: Array, B: Array) -> tuple[Array, Array, Array]:
    """Compute all S1, ST, and S2 indices with a single variance computation.

    Args:
        A: Model outputs from the A base matrix, shape ``(N,)``.
        AB: Model outputs from each cross-matrix AB_j, shape ``(N, D)``.
        BA: Model outputs from each cross-matrix BA_j, shape ``(N, D)``.
        B: Model outputs from the B base matrix, shape ``(N,)``.

    Returns:
        S1: first-order Sobol indices, shape ``(D,)``.
        ST: total-order Sobol indices, shape ``(D,)``.
        S2: second-order interaction indices, shape ``(D, D)``. Only the
            upper triangle is valid.
    """
    N = A.shape[0]

    # Centered-sum variance (see _fused_first_total for rationale).
    pooled_mean = (jnp.mean(A) + jnp.mean(B)) / 2.0
    A_c = A - pooled_mean
    B_c = B - pooled_mean
    var = (jnp.sum(A_c**2) + jnp.sum(B_c**2)) / (2 * N)
    inv_var = jnp.where(var == 0, jnp.nan, 1.0 / var)

    # S1 and ST use the same Sobol'-Mauntz/Jansen pair as _fused_first_total
    S1 = jnp.mean(B[:, None] * (AB - A[:, None]), axis=0) * inv_var  # (D,)
    ST = 0.5 * jnp.mean((A[:, None] - AB) ** 2, axis=0) * inv_var  # (D,)

    # Outer-product trick: BA[:,j] * AB[:,k] for all (j,k) pairs at once.
    # BA[:, :, None] is (N,D,1), AB[:, None, :] is (N,1,D); their product
    # is (N,D,D), giving the full joint-variance matrix V_jk in one pass.
    # This avoids a nested loop over D*D pairs that would be JIT-hostile.
    Vjk = (
        jnp.mean(
            BA[:, :, None] * AB[:, None, :] - (A * B)[:, None, None],
            axis=0,
        )
        * inv_var
    )  # (D, D)
    # ANOVA decomposition via broadcasting: S1[:, None] is (D,1) and
    # S1[None, :] is (1,D), subtracting marginal effects from all (D,D) pairs.
    S2 = Vjk - S1[:, None] - S1[None, :]  # (D, D)

    return S1, ST, S2
