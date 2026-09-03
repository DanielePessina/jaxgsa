"""The verbose seam prints what it promises, and only when asked.

Tier T4 (internal consistency): there is no external oracle for a printed
summary; what these tests prove is that ``verbose=True`` prints the three
sections, that ``verbose=False`` prints nothing, and that the sections say
what the workflow really did.

The rest of the suite silences :func:`jaxgsa._core.verbose.emit` through an
autouse fixture in ``conftest.py``. Every test here carries the
``verbose_output`` marker, which that fixture honours by leaving the seam
live, so ``capsys`` sees the real output.

The coverage is a representative subset, not all seventeen entry points: one
design-based method (sobol), one given-data method (pawn), one
surrogate-backed method (pce), and two samplers. The signature rule — every
``analyze()`` and every ``sample()`` takes ``verbose: bool = True`` — is
pinned for all seventeen in ``tests/test_vocabulary.py``, and every method
routes through the one helper this file exercises.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxgsa

pytestmark = pytest.mark.verbose_output

PROBLEM = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0), "x3": (-1.0, 1.0)})


def _model(X: jax.Array) -> jax.Array:
    """Tiny non-additive model, cheap enough to run per test."""
    return X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 0] * X[:, 2]


def _given_data(n: int = 256) -> tuple[jax.Array, jax.Array]:
    X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n, seed=0))
    return X, _model(X)


def _sobol_inputs() -> tuple[jaxgsa.sobol.SobolSamples, jax.Array]:
    sr = jaxgsa.sobol.sample(PROBLEM, n_samples=128, seed=0, verbose=False)
    return sr, _model(jnp.asarray(sr.samples))


# ---------------------------------------------------------------------------
# analyze(): the three sections, on a representative subset
# ---------------------------------------------------------------------------


def test_sobol_analyze_prints_the_three_sections(capsys):
    sr, Y = _sobol_inputs()
    jaxgsa.sobol.analyze(sr, Y)
    out = capsys.readouterr().out

    assert "jaxgsa.sobol.analyze" in out
    # Section 1: the problem and the data.
    assert "problem: D=3" in out
    assert "marginals: uniform=3" in out
    assert "correlation: independent" in out
    assert "invalid: none found" in out
    assert "Saltelli groups" in out
    # Section 2: timings, with the resolved chunk width.
    assert "timing:" in out
    assert " s" in out
    assert "slice_chunk_size:" in out
    # Section 3: every parameter ranked by the headline index.
    assert "results: top 3 of 3 parameters by ST" in out
    for name in PROBLEM.names:
        assert name in out


def test_sobol_analyze_verbose_false_prints_nothing(capsys):
    sr, Y = _sobol_inputs()
    jaxgsa.sobol.analyze(sr, Y, verbose=False)
    assert capsys.readouterr().out == ""


def test_bootstrapped_sobol_prints_intervals(capsys):
    sr, Y = _sobol_inputs()
    jaxgsa.sobol.analyze(sr, Y, n_bootstrap=16, key=jax.random.key(0))
    out = capsys.readouterr().out
    # A ranked line with an interval reads "1. x2  ST=0.62  [0.55, 0.69]".
    results = out[out.index("results:") :]
    assert "[" in results and "]" in results


def test_pawn_analyze_prints_and_can_be_silenced(capsys):
    X, Y = _given_data()
    jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bins=4)
    out = capsys.readouterr().out
    assert "jaxgsa.pawn.analyze" in out
    assert "problem: D=3" in out
    assert "timing:" in out
    assert "by PAWN" in out

    jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bins=4, verbose=False)
    assert capsys.readouterr().out == ""


def test_pce_analyze_prints_and_can_be_silenced(capsys):
    X, Y = _given_data(n=512)
    jaxgsa.pce.analyze(PROBLEM, X, Y, order=2)
    out = capsys.readouterr().out
    assert "jaxgsa.pce.analyze" in out
    assert "timing:" in out
    assert "by ST" in out

    jaxgsa.pce.analyze(PROBLEM, X, Y, order=2, verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# sample(): the design narration
# ---------------------------------------------------------------------------


def test_sobol_sample_narrates_the_design(capsys):
    jaxgsa.sobol.sample(PROBLEM, n_samples=128, seed=0)
    out = capsys.readouterr().out
    assert out.startswith("jaxgsa.sobol.sample: D=3")
    assert "n_runs=" in out
    assert "duplicates_removed=" in out

    jaxgsa.sobol.sample(PROBLEM, n_samples=128, seed=0, verbose=False)
    assert capsys.readouterr().out == ""


def test_morris_sample_narrates_the_design(capsys):
    jaxgsa.morris.sample(PROBLEM, n_trajectories=8, seed=0)
    out = capsys.readouterr().out
    assert out.startswith("jaxgsa.morris.sample: D=3")
    assert "duplicates_removed=" in out

    jaxgsa.morris.sample(PROBLEM, n_trajectories=8, seed=0, verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# The layout note: (T, K) outputs say they were averaged
# ---------------------------------------------------------------------------


def test_multi_output_summary_says_it_averaged_over_slices(capsys):
    sr, Y = _sobol_inputs()
    Y2 = jnp.stack([Y, 2.0 * Y + 1.0], axis=-1)
    jaxgsa.sobol.analyze(sr, Y2)
    out = capsys.readouterr().out
    assert "T=1 x K=2" in out
    assert "mean over 2 output slices" in out
