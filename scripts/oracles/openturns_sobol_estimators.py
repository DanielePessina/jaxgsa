"""Record OpenTURNS' Sobol index estimators as a T2 oracle for jaxgsa.

OpenTURNS ships four ``SobolIndicesAlgorithm`` implementations that overlap
with jaxgsa's ``estimator=`` menu: Saltelli, Jansen, Martinez and
Mauntz-Kucherenko. It is BSD-licensed and independent of jaxgsa, so it is a
real oracle rather than a mirror of our own formulas.

OpenTURNS is not a jaxgsa dependency, not even a development one, so this
script runs locally and its output is pinned into
``tests/test_sobol_estimators.py``. That follows the same pattern as
``salib_delta_class_counts.py``.

Run it with::

    uv run --with openturns scripts/oracles/openturns_sobol_estimators.py

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
"""

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


def build_design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the A, B and AB output vectors of a plain Monte Carlo design.

    Returns:
        ``(yA, yB, yAB)`` with shapes ``(N,)``, ``(N,)`` and ``(N, D)``.
    """
    rng = np.random.default_rng(SEED)
    A = rng.uniform(-np.pi, np.pi, (N, D))
    B = rng.uniform(-np.pi, np.pi, (N, D))
    columns = np.arange(D)
    AB = np.stack([np.where(columns == j, B, A) for j in range(D)], axis=1)

    def f(X: np.ndarray) -> np.ndarray:
        return np.asarray(ishigami.evaluate(jnp.asarray(X)))

    return f(A), f(B), np.stack([f(AB[:, j]) for j in range(D)], axis=1), A, B, AB


def main() -> None:
    """Print the OpenTURNS indices and jaxgsa's largest disagreement with them."""
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


if __name__ == "__main__":
    main()
