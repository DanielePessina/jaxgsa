"""Regenerate the Plischke class counts from SALib itself.

Oracle: SALib 1.5.2, tier T2. Run it with the development extra::

    uv run --extra dev scripts/oracles/salib_delta_class_counts.py

The numbers it prints are the literals in
``tests/test_borgonovo.py::TestPlischkeHeuristic::test_matches_reference_class_counts``.

SALib computes the class count ``M`` inside ``SALib.analyze.delta.analyze`` and
never returns it. To read it without copying the rule, this script calls
``analyze`` for real and captures the partition array ``m`` that ``analyze``
hands to ``bias_reduced_delta``. That array has ``M + 1`` edges, so ``M`` falls
out of its length. The stub also keeps the run cheap: the delta estimator never
executes.

The script exits 0 when SALib agrees with the literals in the test, and 1 when
it does not.
"""

from __future__ import annotations

import sys
from importlib.metadata import version

import numpy as np
import SALib.analyze.delta as salib_delta

# The literals as they stand in tests/test_borgonovo.py today.
EXPECTED: dict[int, int] = {
    100: 4,
    500: 5,
    1000: 6,
    1500: 9,
    5000: 18,
    10000: 22,
    100000: 47,
}


def salib_class_count(n_samples: int) -> int:
    """Ask SALib how many classes it would use for a sample of this size.

    Args:
        n_samples: Number of model evaluations.

    Returns:
        The class count ``M`` that ``SALib.analyze.delta.analyze`` picks.
    """
    captured: list[np.ndarray] = []

    def spy(Y, Ygrid, X_i, m, num_resamples, conf_level, y_resamples):
        """Stand in for ``bias_reduced_delta`` and record the partition."""
        captured.append(np.asarray(m))
        return 0.0, 0.0

    original = salib_delta.bias_reduced_delta
    salib_delta.bias_reduced_delta = spy
    try:
        rng = np.random.default_rng(0)
        problem = {"num_vars": 1, "names": ["x"], "bounds": [[0.0, 1.0]]}
        X = rng.random((n_samples, 1))
        Y = X[:, 0].copy()
        salib_delta.analyze(
            problem,
            X,
            Y,
            num_resamples=1,
            method="delta",
            print_to_console=False,
        )
    finally:
        salib_delta.bias_reduced_delta = original

    if not captured:
        raise RuntimeError("SALib did not reach bias_reduced_delta; the spy caught nothing.")
    # analyze() builds m as linspace(0, y_resamples, M + 1).
    return len(captured[0]) - 1


def main() -> int:
    """Print the table and report whether SALib matches the test literals.

    Returns:
        0 when every value agrees, 1 otherwise.
    """
    print(f"oracle: SALib {version('salib')} (SALib.analyze.delta.analyze)")
    print()
    print(f"{'N':>8}  {'SALib':>6}  {'in test':>8}  status")
    ok = True
    for n_samples, expected in EXPECTED.items():
        got = salib_class_count(n_samples)
        agrees = got == expected
        ok = ok and agrees
        print(f"{n_samples:>8}  {got:>6}  {expected:>8}  {'ok' if agrees else 'MISMATCH'}")
    print()
    if ok:
        print("All values agree with tests/test_borgonovo.py.")
        return 0
    print("SALib disagrees with the literals in tests/test_borgonovo.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
