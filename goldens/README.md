# Golden fixtures for the jaxgsa JavaScript port

Deterministic reference outputs for the six hand-ported methods, produced by
the Python package `jaxgsa` **0.9.0** in full float64. The TypeScript tests
validate the wasm port against these files.

## Files

```
goldens/
  generate.py            the generator (uv run python goldens/generate.py)
  README.md              this file
  v0.9.0/
    sobol.json
    kucherenko.json
    morris.json
    pce.json
    hdmr.json
    shapley.json
```

## Test case

Every method runs on the same problem: the **Ishigami benchmark**
(`jaxgsa.benchmarks.ishigami`), D=3 uniform on `[-pi, pi]`, scalar output

```
f(x) = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)
```

Fixed settings (see `generate.py` for the exact call signatures):

| method     | design                                            | rows   |
|------------|---------------------------------------------------|--------|
| sobol      | scrambled Sobol, `base_n=512`, first/total only   | 2560   |
| kucherenko | conditional design, `n_samples=512` (7 blocks)    | 3584   |
| morris     | `n_trajectories=40`, 4-level trajectory design    | 61     |
| pce        | Sobol A-block cloud (512 rows), `order=9`         | 512    |
| hdmr       | same 512-row cloud, defaults (`maxorder=2`)       | 512    |
| shapley    | same 512-row cloud, `backend="pce"`, `order=9`    | 512    |

All random draws use `seed=42`; no RNG is ever unseeded.

- **sobol** / **kucherenko** / **morris** are design methods: the stored `x`
  is the design, `y` its model outputs, and `expected` the estimator output.
- **pce** / **hdmr** / **shapley** are given-data methods: `x`/`y` is the
  shared 512-row space-filling cloud (the A block of the same scrambled Sobol
  sequence used by the sobol golden), and `expected` the surrogate indices.
  PCE uses `order=9` because the default `order=3` leaves ~50% of the output
  variance unexplained (fires a `JaxgsaWarning`, indices describe a poor
  surrogate); order 9 reaches `explained_variance > 0.999` and the indices
  are stable to ~1e-7 across ridge values.

## Schema

Flat and explicit; one file per method:

```json
{
  "method": "sobol",
  "jaxgsa_version": "0.9.0",
  "n": 2560,
  "x": [[...], ...],
  "y": [...],
  "expected": {"S1": [...], "ST": [...]},
  "tolerance": {"rtol": 1e-7, "atol": 1e-9},
  "config": {...}
}
```

| key                | type   | meaning                                              |
|--------------------|--------|------------------------------------------------------|
| `method`           | string | one of the six method names                           |
| `jaxgsa_version`   | string | package version that produced the file                |
| `n`                | int    | row count of `x` / `y`                                |
| `x`                | list   | `(n, 3)` design or input cloud, float64               |
| `y`                | list   | `(n,)` Ishigami outputs, float64                      |
| `expected`         | object | method-specific index arrays (see below)              |
| `tolerance`        | object | `rtol` / `atol` the TS tests must use                 |
| `config`           | object | seed and parameters that produced the file (informational) |
| `design`           | object | design bookkeeping the estimator needs (morris only)  |

`expected` keys per method:

- `sobol`: `S1`, `ST`
- `kucherenko`: `S1`, `ST`, `variance`
- `morris`: `mu`, `mu_star`, `sigma`
- `pce` / `hdmr` / `shapley`: `S1`, `ST`

`design` (morris only) carries the bookkeeping the elementary-effect estimator
reads, exactly as `MorrisSamples` stores it: `expanded_to_unique` (map from each
expanded row to its unique row), `ee_idx_after` / `ee_idx_before` (expanded-row
indices of the perturbed/reference points, `(r, D)`), `ee_delta` (signed
unit-cube steps, `(r, D)`), plus `n_expanded`, `n_trajectories`, `num_levels`.

All arrays are plain JSON lists of Python `float` (64-bit double precision),
stored verbatim — no rounding, no truncation.

## Tolerances

The `tolerance` block in each file is the source of truth; the TS tests must
use `|actual - expected| <= atol + rtol * |expected|` per element.

- Design methods (`sobol`, `kucherenko`, `morris`): `rtol=1e-7, atol=1e-9`.
  The estimators are exact arithmetic on stored data; the band absorbs
  float64 trig differences in the JS evaluation of Ishigami.
- Surrogate methods (`pce`, `hdmr`, `shapley`): `rtol=1e-5, atol=1e-6`.
  The fits are more sensitive to kernel scheduling (basis construction,
  solver path); two orders of magnitude wider.

## Determinism and regeneration policy

Regenerate with:

```
uv run python goldens/generate.py
```

The script enables `jax_enable_x64`, uses `seed=42` for every draw, silences
the expected `JaxgsaWarning`s (kucherenko on an independent problem), prints
a one-line summary per method, and exits non-zero on any failure.

Two successive runs must be **byte-identical** (verified by `shasum`).
Determinism holds because:

- scrambled Sobol designs come from `jax.random` with a fixed key;
- the cloud and all fits run single-threaded CPU JAX in float64;
- results are converted with `numpy.asarray(...).tolist()` (stable ordering).

Policy:

- Golden files change **only** when the jaxgsa version that defines the
  reference changes. Regenerate, commit all six files together, and bump the
  directory name (`v0.9.0/` -> `vX.Y.Z/`) for a breaking reference change.
- If a regeneration diffs unexpectedly, treat it as a bug report: reproduce
  the divergence before committing new goldens.
- Do not hand-edit golden JSON; edit `generate.py` and regenerate.