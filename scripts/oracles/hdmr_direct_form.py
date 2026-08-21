"""Cross-check jaxgsa's RS-HDMR against a direct-form reference in NumPy.

Provenance, as `scripts/oracles/README.md` requires:

1. **Tier**: T4 (internal consistency), with the F quantile at T2. The fit
   itself has no external oracle -- see "Why T4" below.
2. **Oracle**: this script's `direct_form_hdmr`, an independent NumPy/SciPy
   implementation of the RS-HDMR estimator, plus `scipy.stats.f.ppf` for the
   F-test critical values.
3. **Version**: NumPy 2.4.2, SciPy 1.18.0, Python 3.12.13.
4. **Date run**: 2026-08-19.
5. **Script**: this file. Run it with
   `uv run --extra dev scripts/oracles/hdmr_direct_form.py`.

## Why this exists

Until 2026-08-19 `jaxgsa.hdmr` had two fit paths. The "in-memory" one solved
the component regressions in sample space, one output slice at a time; the
"streamed" one accumulated Gram matrices over row batches and solved in
coefficient space. Each checked the other, and `tests/test_hdmr_streaming.py`
was that check.

Unifying the two onto the Gram formulation removed the sample-space
implementation, and with it the only second opinion on the estimator. Copying
the deleted kernel into the test suite would not restore it: a test that
retypes the source's own formula is a mirror, not an oracle. An oracle must
come from outside the library under test. So the direct form lives here
instead, re-derived rather than copied:

- pure NumPy and SciPy, no JAX, no import from `src/jaxgsa/`;
- float64 throughout, so the comparison is not reading float32 noise;
- `scipy.stats.f.ppf` instead of the bisection on `betainc` that jaxgsa rolls
  by hand, which makes the critical values an actual external check;
- sample-space normal equations and an explicit Python backfitting loop, which
  is the formulation the library no longer contains.

One thing is shared by construction and cannot be otherwise: the cubic
B-spline basis, including the `m**3` scaling that follows SALib's convention.
Two implementations of *different* bases would disagree for reasons that say
nothing about the fit. The basis here is written from the Cox-de Boor
piecewise cubic rather than transcribed, but it is the same mathematical
object on purpose. What this script cross-checks is the estimator: the
backfitting solve, the sequential higher-order residual, the ANCOVA split and
the F-test.

## Why T4 and not higher

SALib's `analyze.hdmr` is the only other implementation of this estimator, and
it differs from jaxgsa in the sampling it assumes, in its bootstrap-driven
term selection, and in returning `Sa`/`Sb`/`S` only for terms its own F-test
kept. Pinning a literal against it would pin those choices, not the ANCOVA
decomposition. Li et al. (2010) publish no numeric table for the Ishigami
function that this could be checked against, so there is no T1 either. That
leaves an independent re-implementation, which is T4 by definition, and the
honest label for it is T4.

## What the recorded literals are

`EXPECTED` holds the direct-form `Sa`, `Sb`, `S` and `ST` for two problems, in
float64, computed by this script on the date above. They are the pre-
unification numbers in the sense that matters: they are what the sample-space
formulation produces, independent of the Gram rewrite that replaced it. They
are recorded here rather than in a test on purpose -- nothing in the suite
asserts them, and the script is the thing a future reader re-runs by hand.

The script prints three columns: the recorded literal, what the direct form
computes now, and what `jaxgsa.hdmr.analyze` computes now. It exits 0 when all
three agree to `TOL`, and 1 otherwise.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.stats import f as f_dist

# Agreement required between the direct form and jaxgsa, in float64. The two
# solve the same normal equations by different routes -- sample space against
# coefficient space -- so they agree to solver noise on a matrix whose
# condition number the Tikhonov term bounds, not to machine epsilon.
TOL = 1e-9

# Fit settings shared by every case below.
MAXORDER = 2
MAXITER = 100
M_INTERVALS = 2
LAMBDAX = 0.01
ALPHA = 0.95


# ---------------------------------------------------------------------------
# The direct form: NumPy only
# ---------------------------------------------------------------------------


def bspline_basis(x: np.ndarray, m: int) -> np.ndarray:
    """Evaluate the m+3 cubic B-spline basis functions at points x in [0, 1].

    Uniform knots at spacing 1/m. The Cox-de Boor recursion for degree 3 on a
    uniform knot vector collapses to one cubic polynomial per unit of the
    local coordinate, which is what the four branches below are. The `m**3`
    scaling is SALib's convention and jaxgsa follows it.

    Args:
        x: Evaluation points in [0, 1], shape ``(N,)``.
        m: Number of B-spline intervals.

    Returns:
        Basis matrix, shape ``(N, m + 3)``.
    """
    n_basis = m + 3
    i = np.arange(n_basis, dtype=np.float64)
    u = x[:, None] * m - i[None, :] + 3.0
    val = np.zeros_like(u)
    s = (u >= 0) & (u < 1)
    val[s] = u[s] ** 3 / 6.0
    s = (u >= 1) & (u < 2)
    val[s] = (-3 * u[s] ** 3 + 12 * u[s] ** 2 - 12 * u[s] + 4) / 6.0
    s = (u >= 2) & (u < 3)
    val[s] = (3 * u[s] ** 3 - 24 * u[s] ** 2 + 60 * u[s] - 44) / 6.0
    s = (u >= 3) & (u <= 4)
    val[s] = (4 - u[s]) ** 3 / 6.0
    return np.maximum(val * (m**3), 0.0)


def build_bases(X_n: np.ndarray, m: int, c2: list[tuple[int, int]]):
    """Build the first- and second-order bases.

    Args:
        X_n: Inputs mapped to [0, 1], shape ``(N, D)``.
        m: Number of B-spline intervals.
        c2: Parameter index pairs, in term order.

    Returns:
        ``(B1, B2)`` of shapes ``(N, m1, D)`` and ``(N, m1**2, n2)``.
    """
    N, D = X_n.shape
    m1 = m + 3
    B1 = np.stack([bspline_basis(X_n[:, j], m) for j in range(D)], axis=2)
    beta = list(itertools.product(range(m1), repeat=2))
    B2 = np.empty((N, m1 * m1, len(c2)))
    for t, (a, b) in enumerate(c2):
        for q, (p, r) in enumerate(beta):
            B2[:, q, t] = B1[:, p, a] * B1[:, r, b]
    return B1, B2


def direct_form_hdmr(X_n: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Fit one output slice by the sample-space direct form.

    Solves each component regression against the sample vectors themselves
    rather than against accumulated Gram matrices: the first order by
    coordinate-descent backfitting with a precomputed solver matrix per
    dimension, the second order in one shot against the first-order residual.

    Args:
        X_n: Inputs mapped to [0, 1], shape ``(N, D)``.
        y: One output slice, shape ``(N,)``.

    Returns:
        Mapping with ``Sa``, ``Sb``, ``S`` (per term), ``ST`` (per parameter),
        ``select`` (per term) and ``rmse``.
    """
    N, D = X_n.shape
    m1 = M_INTERVALS + 3
    m2 = m1 * m1
    c2 = list(itertools.combinations(range(D), 2))
    n1, n2 = D, len(c2)
    B1, B2 = build_bases(X_n, M_INTERVALS, c2)

    f0 = y.mean()
    V_Y = y.var()
    Y_res = y - f0

    # First order: T1[j] = (B1_j^T B1_j + lambda I)^-1 B1_j^T, fixed for the
    # whole backfit because the basis does not change.
    lam1 = LAMBDAX * np.eye(m1)
    T1 = np.empty((n1, m1, N))
    for j in range(n1):
        Bj = B1[:, :, j]
        T1[j] = np.linalg.solve(Bj.T @ Bj + lam1, Bj.T)

    C1 = np.stack([T1[j] @ Y_res for j in range(n1)], axis=1)  # (m1, n1)
    var_old = (C1**2).sum(axis=0)
    for _ in range(MAXITER):
        for j in range(n1):
            all_contrib = np.einsum("rmj,mj->r", B1, C1)
            j_contrib = B1[:, :, j] @ C1[:, j]
            C1[:, j] = T1[j] @ (Y_res - all_contrib + j_contrib)
        var_new = (C1**2).sum(axis=0)
        if np.abs(var_new - var_old).max() <= 1e-3:
            var_old = var_new
            break
        var_old = var_new
    Y_em1 = np.einsum("rmj,mj->rj", B1, C1)

    # Second order: one shot per term against the first-order residual.
    Y_res2 = Y_res - Y_em1.sum(axis=1)
    lam2 = LAMBDAX * np.eye(m2)
    C2 = np.empty((m2, n2))
    for t in range(n2):
        Bt = B2[:, :, t]
        C2[:, t] = np.linalg.solve(Bt.T @ Bt + lam2, Bt.T @ Y_res2)
    Y_em2 = np.einsum("rmj,mj->rj", B2, C2)

    Y_em = np.concatenate([Y_em1, Y_em2], axis=1)  # (N, n)

    # ANCOVA (Li et al., 2010): structural, total, correlative.
    Y_em_c = Y_em - Y_em.mean(axis=0, keepdims=True)
    Sa = Y_em.var(axis=0) / V_Y
    S = (Y_em_c * (y - y.mean())[:, None]).mean(axis=0) / V_Y
    Y0 = Y_em.sum(axis=1)
    Y0_minus = Y0[:, None] - Y_em
    Sb = (Y_em_c * (Y0_minus - Y0_minus.mean(axis=0, keepdims=True))).mean(axis=0) / V_Y

    # SCSA total: every term containing a parameter, summed onto it.
    ST = S[:n1].copy()
    for t, (a, b) in enumerate(c2):
        ST[a] += S[n1 + t]
        ST[b] += S[n1 + t]

    # F-test, hierarchical nulls. scipy supplies the critical values.
    r0 = y - f0
    r1 = r0 - Y_em[:, :n1].sum(axis=1)
    nulls = [r0, r1]
    crit = [f_dist.ppf(ALPHA, p, N - p) if N > p else np.inf for p in (m1, m2)]
    select = np.empty(n1 + n2)
    for i in range(n1 + n2):
        order = 0 if i < n1 else 1
        p = m1 if order == 0 else m2
        r_null = nulls[order]
        SSR0 = float(r_null @ r_null)
        resid = r_null - Y_em[:, i]
        SSR1 = float(resid @ resid)
        F = ((SSR0 - SSR1) / p) / (SSR1 / max(N - p, 1.0))
        select[i] = float(F > crit[order])

    rmse = float(np.sqrt(np.mean((y - (f0 + Y0)) ** 2)))
    return {"Sa": Sa, "Sb": Sb, "S": S, "ST": ST, "select": select, "rmse": np.array(rmse)}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def ishigami_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ishigami on 3 uniform inputs over [-pi, pi], N = 2000, seed 20260819.

    Returns:
        ``(X, X_n, y)``: raw inputs, inputs mapped to [0, 1], and the output.
    """
    rng = np.random.default_rng(20260819)
    X = rng.uniform(-np.pi, np.pi, size=(2000, 3))
    y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    return X, (X + np.pi) / (2 * np.pi), y


def sobol_g_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobol g-function on 4 uniform inputs over [0, 1], N = 2000, seed 20260819.

    Four inputs rather than the usual eight: at maxorder 2 the term count is
    quadratic in D, and 4 keeps the printed table readable while still
    exercising a case with several inactive terms.

    Returns:
        ``(X, X_n, y)``: raw inputs, inputs mapped to [0, 1], and the output.
    """
    rng = np.random.default_rng(20260819)
    X = rng.uniform(0.0, 1.0, size=(2000, 4))
    a = np.array([0.0, 1.0, 4.5, 9.0])
    y = np.prod((np.abs(4.0 * X - 2.0) + a) / (1.0 + a), axis=1)
    return X, X.copy(), y


# Recorded on 2026-08-19 by this script, float64. See the module docstring.
EXPECTED: dict[str, dict[str, list[float]]] = {
    "ishigami": {
        "Sa": [
            0.29100975417503894,
            0.36980450726702874,
            0.0016616000557304776,
            0.0033443362480038885,
            0.2374455877803616,
            0.0032048287365459478,
        ],
        "Sb": [
            -0.0012560346428897918,
            -0.002317808543496159,
            0.0011558789197646795,
            0.0022144973548078894,
            0.0028818394687403453,
            0.003405506910258371,
        ],
        "S": [
            0.288748645504602,
            0.3684565446239413,
            0.002809142684827292,
            0.003579875880771265,
            0.23654624403601965,
            0.004405067807238723,
        ],
        "ST": [
            0.5288747654213929,
            0.37644148831195123,
            0.24376045452808567,
        ],
        "select": [
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    },
    "sobol_g": {
        "Sa": [
            0.7080058717696551,
            0.1857497249978573,
            0.024681896259494778,
            0.00763171828785918,
            0.05491530571781809,
            0.007910417614965085,
            0.0037699548047541924,
            0.0020426772452883557,
            0.001343138425418819,
            0.000602549472473757,
        ],
        "Sb": [
            -0.0017404909208615206,
            0.0035657673312077596,
            -0.0024954181433113,
            0.0013703500665769367,
            0.0033202654349939404,
            0.001597509599547088,
            0.0025519402336855452,
            0.00037503389402177906,
            0.002069132882835623,
            0.0008243353573801588,
        ],
        "S": [
            0.7036333791991969,
            0.1870075463592802,
            0.022279149681843233,
            0.008915828272773088,
            0.05563842780589843,
            0.008691844068816134,
            0.004951108695404822,
            0.0025193584806978138,
            0.0027971968022613737,
            0.00095011573003556,
        ],
        "ST": [
            0.7729147597693163,
            0.24796252944813782,
            0.03444046796139274,
            0.017614249500474844,
        ],
        "select": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
        ],
    },
}


def jaxgsa_values(X: np.ndarray, y: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    """Run the library on the same data, in float64.

    Args:
        X: Raw inputs, shape ``(N, D)``.
        y: Output, shape ``(N,)``.
        lo: Lower bound shared by every input.
        hi: Upper bound shared by every input.

    Returns:
        Mapping with the same keys `direct_form_hdmr` returns.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jaxgsa.hdmr import analyze
    from jaxgsa.problem import Problem

    D = X.shape[1]
    problem = Problem.from_dict({f"x{i}": (lo, hi) for i in range(D)})
    r = analyze(
        problem,
        jnp.asarray(X),
        jnp.asarray(y),
        maxorder=MAXORDER,
        maxiter=MAXITER,
        m=M_INTERVALS,
        lambdax=LAMBDAX,
    )
    return {
        "Sa": np.asarray(r.Sa),
        "Sb": np.asarray(r.Sb),
        "S": np.asarray(r.S),
        "ST": np.asarray(r.ST),
        "select": np.asarray(r.select),
        "rmse": np.asarray(r.rmse),
    }


def main() -> int:
    """Print the comparison table and report agreement.

    Returns:
        0 when the recorded literals, the direct form and jaxgsa all agree to
        ``TOL``, 1 otherwise.
    """
    print(f"oracle: direct-form RS-HDMR in NumPy {np.__version__}, F quantile from SciPy")
    print(f"tolerance: {TOL:g} (float64)")
    ok = True
    for name, builder, lo, hi in (
        ("ishigami", ishigami_case, -np.pi, np.pi),
        ("sobol_g", sobol_g_case, 0.0, 1.0),
    ):
        X, X_n, y = builder()
        direct = direct_form_hdmr(X_n, y)
        lib = jaxgsa_values(X, y, lo, hi)
        recorded = EXPECTED.get(name)
        print(f"\n{name}  (N={X.shape[0]}, D={X.shape[1]}, maxorder={MAXORDER})")
        print(
            f"  {'field':7} {'idx':>3} {'recorded':>22} {'direct form':>22} {'jaxgsa':>22}  status"
        )
        for field in ("Sa", "Sb", "S", "ST", "select"):
            d = np.atleast_1d(direct[field])
            g = np.atleast_1d(lib[field])
            e = np.asarray(recorded[field]) if recorded else d
            for i in range(d.size):
                agree = abs(d[i] - g[i]) <= TOL and abs(d[i] - e[i]) <= TOL
                ok = ok and agree
                print(
                    f"  {field:7} {i:>3} {e[i]:>22.15g} {d[i]:>22.15g} {g[i]:>22.15g}"
                    f"  {'ok' if agree else 'MISMATCH'}"
                )
        if not recorded:
            print(f"  (no literals recorded for {name}; printing the direct form as the column)")
    print()
    if ok:
        print("Direct form, jaxgsa and the recorded literals agree.")
        return 0
    print("Disagreement: see the MISMATCH rows above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
