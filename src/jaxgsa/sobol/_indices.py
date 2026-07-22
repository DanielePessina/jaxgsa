"""Sobol sensitivity index estimators (first-order, total-order, second-order).

Implements the Saltelli (2010) estimators for variance-based global
sensitivity analysis using a Sobol quasi-random sampling design.

Notation
--------
- N : number of base samples drawn from the input parameter space.
- A, B : two independent (N, D) input sample matrices, where D is the
  number of parameters.  The arrays passed to these functions are the
  corresponding *model outputs*, each of shape (N,).
- AB_j : model output when column j of A is replaced by column j of B
  (all other columns remain from A).  Shape (N,).
- BA_j : model output when column j of B is replaced by column j of A
  (all other columns remain from B).  Shape (N,).

Variance estimation
-------------------
All estimators normalise by a *pooled* output variance computed over the
concatenation of A and B, i.e. ``var(concat(A, B))`` with shape (2N,).
Pooling both base-sample vectors gives a more robust variance estimate
than using A or B alone, because it doubles the effective sample size
while remaining unbiased (A and B are identically distributed).
"""

import jax.numpy as jnp
from jax import Array


def first_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Estimate the first-order (main-effect) Sobol index for parameter j.

    Uses the Saltelli (2010) estimator::

        S1_j = E[B * (AB_j - A)] / Var(Y)

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the first-order index S1_j.
    """
    # Pool A and B (identically distributed) to double effective sample size
    # for a more robust variance estimate: Var(Y) ~ var(concat(A, B)).
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    # B*(AB_j - A) isolates how changing only parameter j (A->B) shifts output
    numerator = jnp.mean(B * (AB_j - A))
    # Zero variance means constant output; index is undefined, not zero
    return jnp.where(var == 0, jnp.nan, numerator / var)


def total_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Estimate the total-order Sobol index for parameter j.

    Uses the Jansen (1999) estimator::

        ST_j = (1/2) * E[(A - AB_j)^2] / Var(Y)

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the total-order index ST_j.
    """
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    # Jansen (1999) estimator: ST_j = E[(A - AB_j)^2] / (2 Var(Y)).
    # Measures total variance attributable to param j (including interactions).
    # Preferred over Sobol (1993) because it is non-negative by construction.
    numerator = 0.5 * jnp.mean((A - AB_j) ** 2)
    return jnp.where(var == 0, jnp.nan, numerator / var)


def second_order(A: Array, AB_j: Array, AB_k: Array, BA_j: Array, B: Array) -> Array:
    """Estimate the second-order Sobol interaction index between parameters j and k.

    Uses the Saltelli (2002) estimator::

        V_jk  = E[BA_j * AB_k - A * B] / Var(Y)
        S2_jk = V_jk - S1_j - S1_k

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        AB_k: (N,) model outputs from the cross-matrix where column k
            of A is replaced by column k of B.
        BA_j: (N,) model outputs from the cross-matrix where column j
            of B is replaced by column j of A.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the second-order interaction index S2_jk.
    """
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    # V_jk estimates the joint variance contribution of params j and k
    Vjk = jnp.where(var == 0, jnp.nan, jnp.mean(BA_j * AB_k - A * B) / var)
    # Subtract marginal effects to isolate the pure interaction
    Sj = first_order(A, AB_j, B)
    Sk = first_order(A, AB_k, B)
    # Sobol ANOVA decomposition: the second-order interaction is the joint
    # variance contribution V_jk minus both marginal first-order effects,
    # isolating the purely synergistic effect between parameters j and k.
    return Vjk - Sj - Sk


# ---------------------------------------------------------------------------
# Fused kernels: compute variance ONCE, derive all indices from it.
# The per-parameter functions above recompute pooled var(concat(A,B)) for
# every parameter j independently.  These fused variants compute it once
# and vectorise all D parameters via broadcasting, giving D-fold savings
# on the most expensive reduction and enabling efficient JIT compilation.
# ---------------------------------------------------------------------------


def _fused_first_total(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Compute all S1 and ST indices with a single variance computation.

    Args:
        A:  (N,) model outputs from the A base matrix.
        AB: (N, D) model outputs from each cross-matrix AB_j.
        B:  (N,) model outputs from the B base matrix.

    Returns:
        S1: (D,) first-order Sobol indices.
        ST: (D,) total-order Sobol indices.
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
    # Saltelli (2010) S1 estimator: E[B * (AB_j - A)] / Var(Y)
    S1 = jnp.mean(B[:, None] * (AB - A[:, None]), axis=0) * inv_var  # (D,)
    # Jansen (1999) ST estimator: E[(A - AB_j)^2] / (2 Var(Y))
    ST = 0.5 * jnp.mean((A[:, None] - AB) ** 2, axis=0) * inv_var  # (D,)

    return S1, ST


def _fused_second_order(A: Array, AB: Array, BA: Array, B: Array) -> tuple[Array, Array, Array]:
    """Compute all S1, ST, and S2 indices with a single variance computation.

    Args:
        A:  (N,) model outputs from the A base matrix.
        AB: (N, D) model outputs from each cross-matrix AB_j.
        BA: (N, D) model outputs from each cross-matrix BA_j.
        B:  (N,) model outputs from the B base matrix.

    Returns:
        S1: (D,) first-order Sobol indices.
        ST: (D,) total-order Sobol indices.
        S2: (D, D) second-order interaction indices (upper triangle valid).
    """
    N = A.shape[0]

    # Centered-sum variance (see _fused_first_total for rationale).
    pooled_mean = (jnp.mean(A) + jnp.mean(B)) / 2.0
    A_c = A - pooled_mean
    B_c = B - pooled_mean
    var = (jnp.sum(A_c**2) + jnp.sum(B_c**2)) / (2 * N)
    inv_var = jnp.where(var == 0, jnp.nan, 1.0 / var)

    # S1 and ST use the same Saltelli/Jansen estimators as _fused_first_total
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
