"""Golden-fixture generator for the jaxgsa JavaScript port.

For each of the six hand-ported methods (sobol, kucherenko, morris, pce,
hdmr, shapley) this script builds one fixed deterministic test case on the
Ishigami benchmark (D=3, uniform [-pi, pi], scalar output) and writes the
inputs plus the expected results as JSON under ``goldens/v0.9.0/<method>.json``.

All math runs in float64 (``jax_enable_x64=True``), all random draws use the
same fixed seed, and no RNG is ever left unseeded, so re-running the script
must reproduce the files byte-for-byte. The JSON stores plain ``float`` lists
via ``numpy.asarray(...).tolist()``.

Regeneration:

    uv run python goldens/generate.py

The script prints a one-line summary per method and exits non-zero if any
method raises.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaxgsa  # noqa: E402
from jaxgsa import hdmr, kucherenko, morris, pce, shapley, sobol  # noqa: E402
from jaxgsa.benchmarks import ishigami  # noqa: E402

# Fixed, documented in goldens/README.md. Everything downstream of this seed
# (designs, cloud draws, fits) is deterministic.
SEED = 42

# Per-method sizes (small but stable).
N_SOBOL_BASE = 512      # sobol base points (passed as base_n=...)
N_KUCHERENKO = 512      # kucherenko base points
N_TRAJECTORIES = 40     # morris trajectories
N_CLOUD = 512           # pce / hdmr / shapley rows

# PCE surrogate order. Default order=3 leaves >50% of Ishigami's variance
# unexplained (warns, ev ~0.5); order=9 gives ev > 0.999 with indices stable
# to ~1e-7 across ridge values, so the golden pins a well-conditioned fit.
PCE_ORDER = 9

PROBLEM = ishigami.PROBLEM
JAXGSA_VERSION = jaxgsa.__version__

OUT_DIR = Path(__file__).resolve().parent / "v0.9.0"

# Tolerances the TypeScript tests must use. Design methods are exact
# estimator arithmetic on stored data, so they are allowed tight bands;
# surrogate fits (pce/hdmr/shapley) are more sensitive to kernel scheduling,
# so their bands are two orders of magnitude wider.
TOLERANCES = {
    "sobol": {"rtol": 1e-7, "atol": 1e-9},
    "kucherenko": {"rtol": 1e-7, "atol": 1e-9},
    "morris": {"rtol": 1e-7, "atol": 1e-9},
    "pce": {"rtol": 1e-5, "atol": 1e-6},
    "hdmr": {"rtol": 1e-5, "atol": 1e-6},
    "shapley": {"rtol": 1e-5, "atol": 1e-6},
}


def evaluate(X: np.ndarray) -> np.ndarray:
    """Ishigami outputs as a plain float64 numpy vector, in row order."""
    Y = ishigami.evaluate(jnp.asarray(X))
    return np.asarray(Y, dtype=np.float64)


def to_list(a) -> list:
    """Array -> plain JSON-native list of Python floats."""
    return np.asarray(a).tolist()


def doc(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    expected: dict,
    config: dict,
    design: dict | None = None,
) -> dict:
    """Assemble one golden document per the schema in goldens/README.md."""
    document = {
        "method": method,
        "jaxgsa_version": JAXGSA_VERSION,
        "n": int(x.shape[0]),
        "x": to_list(x),
        "y": to_list(y),
        "expected": {k: to_list(v) for k, v in expected.items()},
        "tolerance": TOLERANCES[method],
        "config": config,
    }
    if design is not None:
        document["design"] = design
    return document


def summarize(method: str, doc_: dict) -> None:
    s1 = doc_["expected"].get("S1")
    extra = f", S1 in [{min(s1):.6f}, {max(s1):.6f}]" if s1 is not None else ""
    print(
        f"{method:12s} n={doc_['n']:5d} x={np.asarray(doc_['x']).shape} "
        f"y={np.asarray(doc_['y']).shape} keys={sorted(doc_['expected'])}{extra}"
    )


def build_sobol() -> dict:
    # base_n=512 gives the requested 512 base points; n_samples is ignored
    # when base_n is given. First/total only: 512 * (D+2) = 2560 rows.
    sr = sobol.sample(
        PROBLEM, n_samples=1, base_n=N_SOBOL_BASE,
        calc_second_order=False, seed=SEED, verbose=False,
    )
    X = np.asarray(sr.samples)
    Y = evaluate(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sobol.analyze(sr, jnp.asarray(Y), verbose=False)
    return doc(
        "sobol", X, Y,
        {"S1": res.S1, "ST": res.ST},
        {"base_n": N_SOBOL_BASE, "calc_second_order": False, "seed": SEED},
    )


def build_kucherenko() -> dict:
    # 512 base points x (2D + 1) = 7 blocks -> 3584 rows. The independent
    # problem fires the expected JaxgsaWarning ("design reduces to the
    # Saltelli column-swap scheme"); it is intentional and silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kr = kucherenko.sample(PROBLEM, n_samples=N_KUCHERENKO, seed=SEED, verbose=False)
        X = np.asarray(kr.samples)
        Y = evaluate(X)
        res = kucherenko.analyze(kr, jnp.asarray(Y), verbose=False)
    return doc(
        "kucherenko", X, Y,
        {"S1": res.S1, "ST": res.ST, "variance": res.variance},
        {"n_samples": N_KUCHERENKO, "seed": SEED},
    )


def build_morris() -> dict:
    # 40 trajectories x (D + 1) = 160 expanded rows, deduplicated to the
    # unique design stored below. The estimator needs the design bookkeeping
    # (expansion map + elementary-effect indices) that the JS analyze reads,
    # so it is stored in the "design" block rather than reconstructed.
    mr = morris.sample(PROBLEM, n_trajectories=N_TRAJECTORIES, seed=SEED, verbose=False)
    X = np.asarray(mr.samples)
    Y = evaluate(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = morris.analyze(mr, jnp.asarray(Y), verbose=False)
    return doc(
        "morris", X, Y,
        {"mu": res.mu, "mu_star": res.mu_star, "sigma": res.sigma},
        {"n_trajectories": N_TRAJECTORIES, "num_levels": 4, "method": "trajectory", "seed": SEED},
        design={
            "n_expanded": mr.n_expanded,
            "n_trajectories": mr.n_trajectories,
            "num_levels": mr.num_levels,
            "expanded_to_unique": to_list(mr.expanded_to_unique),
            "ee_idx_after": to_list(mr.ee_idx_after),
            "ee_idx_before": to_list(mr.ee_idx_before),
            "ee_delta": to_list(mr.ee_delta),
        },
    )


def build_cloud() -> tuple[np.ndarray, np.ndarray]:
    """Shared deterministic space-filling design for the given-data methods.

    The A block (first N_CLOUD rows) of the same scrambled-Sobol design used
    by the sobol golden: 512 unique, space-filling rows in physical units.
    """
    sr = sobol.sample(
        PROBLEM, n_samples=1, base_n=N_SOBOL_BASE,
        calc_second_order=False, seed=SEED, verbose=False,
    )
    X = np.asarray(sr.samples)[:N_CLOUD]
    return X, evaluate(X)


def build_pce() -> dict:
    X, Y = build_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pce.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(Y), order=PCE_ORDER, verbose=False)
    return doc(
        "pce", X, Y,
        {"S1": res.S1, "ST": res.ST},
        {"order": PCE_ORDER, "ridge": 1e-8, "n": N_CLOUD, "seed": SEED},
    )


def build_hdmr() -> dict:
    X, Y = build_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = hdmr.analyze(PROBLEM, jnp.asarray(X), jnp.asarray(Y), verbose=False)
    return doc(
        "hdmr", X, Y,
        {"S1": res.S1, "ST": res.ST},
        {"n": N_CLOUD, "seed": SEED, "maxorder": 2, "maxiter": 100, "lambdax": 0.01},
    )


def build_shapley() -> dict:
    X, Y = build_cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = shapley.analyze(
            PROBLEM, jnp.asarray(X), jnp.asarray(Y), backend="pce", order=PCE_ORDER, verbose=False
        )
    return doc(
        "shapley", X, Y,
        {"S1": res.S1, "ST": res.ST},
        {"backend": "pce", "order": PCE_ORDER, "n": N_CLOUD, "seed": SEED},
    )


def main() -> int:
    builders = {
        "sobol": build_sobol,
        "kucherenko": build_kucherenko,
        "morris": build_morris,
        "pce": build_pce,
        "hdmr": build_hdmr,
        "shapley": build_shapley,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in builders.items():
        d = build()
        out = OUT_DIR / f"{name}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
            fh.write("\n")
        summarize(name, d)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - the caller must see the failure
        print(f"goldens/generate.py failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)