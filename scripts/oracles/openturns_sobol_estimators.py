"""Record OpenTURNS' Sobol index estimators as a T2 oracle for jaxgsa.

Provenance, as ``scripts/oracles/README.md`` requires:

1. **Tier**: T2 (external library at test time).
2. **Oracle**: OpenTURNS' ``SaltelliSensitivityAlgorithm``,
   ``JansenSensitivityAlgorithm``, ``MartinezSensitivityAlgorithm`` and
   ``MauntzKucherenkoSensitivityAlgorithm``. It is BSD-licensed and
   independent of jaxgsa, so it is a real oracle rather than a mirror of our
   own formulas.
3. **Version**: OpenTURNS 1.27.
4. **Date run**: 2026-08-21.
5. **Script**: this file. Run it with
   ``uv run --with openturns scripts/oracles/openturns_sobol_estimators.py``.

OpenTURNS is not a jaxgsa dependency, not even a development one, so this
script runs locally and its output is pinned into
``tests/test_sobol_estimators.py``. That follows the same pattern as
``salib_delta_class_counts.py``.

Conventions differ, and the differences are worth stating because they set
the tolerance in the test. Every OpenTURNS estimator here divides by the
variance of the A block alone and centres the outputs by the mean of the
whole design; jaxgsa divides by the variance pooled over A and B and does
not pre-centre. Both are consistent, and they differ by O(1/sqrt(N)).
The exceptions are Martinez, which is a correlation coefficient and so is
invariant to both choices, and Jansen, whose numerator matches jaxgsa's
exactly while its denominator differs by a constant factor.

The comparison uses a plain Monte Carlo design, not the Saltelli QMC one,
so that the test can rebuild the exact same inputs from a numpy seed
without depending on jaxgsa's sampler.

As ``scripts/oracles/README.md`` requires, this script also checks its own
output against the literals recorded in ``tests/test_sobol_estimators.py``
(``OPENTURNS_ISHIGAMI`` below) and exits non-zero if OpenTURNS now disagrees
with what is pinned there.
"""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import openturns as ot  # noqa: E402

from jaxgsa.benchmarks import ishigami  # noqa: E402
from jaxgsa.sobol import _estimators as estimators  # noqa: E402

N = 2**14
D = 3
SEED = 7

# Pinned into tests/test_sobol_estimators.py as OPENTURNS_ISHIGAMI. Keep the
# two copies equal: this dict is what the test asserts jaxgsa against, and
# this script checks that OpenTURNS still produces it.
OPENTURNS_ISHIGAMI = {
    "jansen": (
        [0.325659913086, 0.449937662035, 0.005534417301],
        [0.556417110084, 0.441889970259, 0.242806411634],
    ),
    "martinez": (
        [0.309152505070, 0.440555663071, -0.009814482997],
        [0.562965078161, 0.443822006127, 0.243499042677],
    ),
    "mauntz-kucherenko": (
        [0.306839712130, 0.438305521513, -0.004547979241],
        [0.568104352653, 0.446389449976, 0.245736508513],
    ),
}
REFERENCE_ATOL = 1e-9
"""How close a fresh OpenTURNS run must stay to the pinned literals above."""


def build_design() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build the A, B and AB output vectors of a plain Monte Carlo design.

    Returns:
        ``(yA, yB, yAB, A, B, AB)``: the model outputs with shapes ``(N,)``,
        ``(N,)`` and ``(N, D)``, followed by the input blocks that produced
        them, each ``(N, D)``.
    """
    rng = np.random.default_rng(SEED)
    A = rng.uniform(-np.pi, np.pi, (N, D))
    B = rng.uniform(-np.pi, np.pi, (N, D))
    columns = np.arange(D)
    AB = np.stack([np.where(columns == j, B, A) for j in range(D)], axis=1)

    def f(X: np.ndarray) -> np.ndarray:
        return np.asarray(ishigami.evaluate(jnp.asarray(X)))

    return f(A), f(B), np.stack([f(AB[:, j]) for j in range(D)], axis=1), A, B, AB


def main() -> int:
    """Print the OpenTURNS indices, jaxgsa's disagreement, and self-check.

    Returns:
        ``0`` when every estimator in ``OPENTURNS_ISHIGAMI`` still matches
        this run's OpenTURNS output within ``REFERENCE_ATOL``, ``1``
        otherwise. A mismatch means the pinned literals in
        ``tests/test_sobol_estimators.py`` are stale and must be
        regenerated from this script's output.
    """
    yA, yB, yAB, A, B, AB = build_design()

    # OpenTURNS reads one stacked sample in the block order [A; B; E_1; ...; E_d].
    X_ot = ot.Sample(np.vstack([A, B] + [AB[:, j] for j in range(D)]))
    Y_ot = ot.Sample(np.concatenate([yA, yB] + [yAB[:, j] for j in range(D)])[:, None])

    algorithms = {
        "saltelli": ot.SaltelliSensitivityAlgorithm(X_ot, Y_ot, N),
        "jansen": ot.JansenSensitivityAlgorithm(X_ot, Y_ot, N),
        "martinez": ot.MartinezSensitivityAlgorithm(X_ot, Y_ot, N),
        "mauntz-kucherenko": ot.MauntzKucherenkoSensitivityAlgorithm(X_ot, Y_ot, N),
    }
    # OpenTURNS' "Saltelli" first order is the plain cross-moment form, which
    # jaxgsa does not offer; its "MauntzKucherenko" is the improved form that
    # jaxgsa's default and mauntz-kucherenko schemes both use.
    pairs = {
        "jansen": "jansen",
        "martinez": "martinez",
        "mauntz-kucherenko": "mauntz-kucherenko",
    }

    a, b, ab = jnp.asarray(yA), jnp.asarray(yB), jnp.asarray(yAB)
    print(f"N = {N}, seed = {SEED}, Ishigami")
    print(f"analytic S1 = {ishigami.ANALYTICAL_S1}")
    print(f"analytic ST = {ishigami.ANALYTICAL_ST}\n")

    ok = True
    for ot_name, algorithm in algorithms.items():
        s1 = np.array(algorithm.getFirstOrderIndices())
        st = np.array(algorithm.getTotalOrderIndices())
        print(f'"{ot_name}": (')
        print(f"    {np.array2string(s1, separator=', ', precision=12)},")
        print(f"    {np.array2string(st, separator=', ', precision=12)},")
        print("),")
        if ot_name in pairs:
            mine_s1, mine_st = estimators.first_total_kernel(pairs[ot_name])(a, ab, b)
            print(
                f"    # jaxgsa {pairs[ot_name]}: max|dS1| = "
                f"{np.max(np.abs(np.asarray(mine_s1) - s1)):.3e}, max|dST| = "
                f"{np.max(np.abs(np.asarray(mine_st) - st)):.3e}"
            )

        pinned = OPENTURNS_ISHIGAMI.get(ot_name)
        if pinned is None:
            print(
                "    # not pinned in tests/test_sobol_estimators.py, so nothing to check. "
                "Printed for reference."
            )
            continue
        pinned_s1, pinned_st = (np.asarray(v) for v in pinned)
        d_s1 = np.max(np.abs(s1 - pinned_s1))
        d_st = np.max(np.abs(st - pinned_st))
        agrees = bool(d_s1 < REFERENCE_ATOL and d_st < REFERENCE_ATOL)
        ok &= agrees
        status = "OK" if agrees else "MISMATCH"
        print(
            f"    # self-check against tests/test_sobol_estimators.py: "
            f"max|dS1|={d_s1:.3e}, max|dST|={d_st:.3e}  [{status}]"
        )

    if ok:
        print(
            "\nSelf-check PASSED: OpenTURNS still matches every literal pinned in "
            "tests/test_sobol_estimators.py. Saltelli is printed but not pinned "
            "there, so it is not checked."
        )
    else:
        print(
            "\nSelf-check FAILED: OpenTURNS disagrees with the literals pinned "
            "in tests/test_sobol_estimators.py. Regenerate OPENTURNS_ISHIGAMI "
            "there from this script's printed output."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
