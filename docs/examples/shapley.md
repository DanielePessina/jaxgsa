# Shapley Effects (Fair Variance Allocation)

Shapley effects allocate the output variance fairly across inputs: each
interaction's variance is split equally among its participants (Owen, 2014;
Song, Nelson & Staum, 2016), so the shares sum to exactly 1 with no gaps and
no double counting. jaxgsa computes them analytically from a fitted
surrogate's variance decomposition — PCE (default) or RS-HDMR — with no
permutation Monte Carlo and no extra model runs. The result carries Sh
alongside the first-order (S1) and total-order (ST) indices from the same
surrogate, so the bracketing `S1 <= Sh <= ST` is visible at a glance.

When to use Shapley effects:

- You want a single, fairly allocated importance score per parameter that
  sums to exactly 1 — for ranking, reporting, or budget allocation.
- Interactions matter and you want them attributed to their participants
  rather than omitted (S1) or counted once per participant (ST).
- You have existing (X, Y) pairs from any sampling strategy — no structured
  design required — and your inputs are independent (required in this
  version).

A companion marimo notebook lives at
[`examples/shapley_gsa.py`](https://github.com/danielepessina/jaxgsa/blob/master/examples/shapley_gsa.py).
Run it interactively with `uv run marimo edit examples/shapley_gsa.py`.

## Import style

Shapley effects are methods on fitted PCE and HDMR results:

```python
import jaxgsa
# jaxgsa.pce.analyze(...).shapley()
# jaxgsa.hdmr.analyze(...).shapley()
```

## Scalar example (Ishigami)

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

# Any (X, Y) pairs work — no structured design required
X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=42))
Y = evaluate(X)

# PCE effects are exact within the fitted polynomial.
# Ishigami's sines need a degree-8 polynomial; the default order=3
# under-fits here and would trigger the explained_variance warning.
result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=8).shapley()

print("Sh:", result.Sh)        # (D,) fair variance shares
print("sum:", result.Sh.sum()) # exactly 1 (Shapley efficiency)
print("S1:", result.S1)        # (D,) first-order, same surrogate
print("ST:", result.ST)        # (D,) total-order, same surrogate
print("explained_variance:", result.explained_variance)  # ~1.03 — good fit
print("order:", result.order)  # effective polynomial degree used
```

Interpreting the indices:

- **Sh sums to exactly 1.** Every interaction's variance is split equally
  among its participants, so the shares partition the decomposed variance
  with no gaps (unlike S1, whose sum falls short of 1) and no double
  counting (unlike ST, whose sum exceeds 1).
- **x3 shows what Shapley adds.** Its first-order index is exactly zero —
  it acts only through the `x1`–`x3` interaction — yet its Shapley effect
  is clearly positive (about 0.12) because it owns half of that
  interaction's variance. S1 would dismiss x3 entirely; ST counts the
  interaction once for x1 and once for x3.
- **Bracketing.** Under independent inputs `S1 <= Sh <= ST` holds
  elementwise, and all three come from the same surrogate fit, so they are
  directly comparable. Ishigami has a single two-way interaction, so
  `Sh = (S1 + ST) / 2` here.

## Ground-truth check

The Ishigami, linear, and Sobol-G benchmarks ship analytical Shapley
effects (`ANALYTICAL_SHAPLEY`), so you can validate against ground truth
rather than another implementation:

```python
import numpy as np
from jaxgsa.benchmarks import ishigami

print("estimated: ", np.round(result.Sh, 4))            # [0.4362 0.4418 0.122]
print("analytical:", np.round(ishigami.ANALYTICAL_SHAPLEY, 4))  # [0.4357 0.4424 0.1218]
```

## Backend selection

The `backend` argument picks the surrogate that supplies the partial
variances:

- **`backend="pce"`** (default) reads subset variances off orthonormal
  polynomial coefficients (Sudret, 2008) — exact within the fitted
  polynomial. Scalar `(N,)` outputs only; a non-scalar `Y` raises
  `ValueError`. Knobs: `order` (default 3), `ridge`, `fit_ratio`.
- **`backend="hdmr"`** fits the RS-HDMR B-spline surrogate and uses its
  structural (ANCOVA) component variances, truncated at `maxorder`.
  Supports `(N,)`, `(N, K)`, and `(N, T, K)` outputs. Knobs: `maxorder`
  (default 2), `m`, `maxiter`, `lambdax`, `prenormalize`, `slice_chunk_size`.

Backend-specific keywords are validated: explicitly setting a knob that
belongs to the non-selected backend (e.g. `backend="pce"` with
`maxorder=3`) raises `ValueError`.

## Multi-output example (HDMR backend)

When Y has shape `(N, K)`, the indices have shape `(K, D)` and each output
row of Sh sums to 1. Time-series outputs `(N, T, K)` produce `(T, K, D)`.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=42))
Y1 = evaluate(X)
Y2 = jnp.sum(X**2, axis=1)  # purely additive: S1 = Sh = ST = 1/3 each
Y_multi = jnp.column_stack([Y1, Y2])

result = jaxgsa.hdmr.analyze(PROBLEM, X, Y_multi).shapley()

print("Sh shape:", result.Sh.shape)          # (K, D) = (2, 3)
print("row sums:", result.Sh.sum(axis=-1))   # [1. 1.]
print("explained_variance:", result.explained_variance)  # (K,)
```

## The explained_variance diagnostic

Indices are normalized by the surrogate's total decomposed variance
`sum_u V_u`. Thus Sh always sums to exactly 1, even when the surrogate
fits poorly. The fit-quality signal is reported separately in
`explained_variance = sum_u V_u / Var(Y)`: close to 1 for a good fit,
below 1 when truncation or fit error leaves variance unexplained, above 1
when an overfit surrogate over-counts shared variance. A `UserWarning` is
emitted when it drops below 0.5 or exceeds 1.3 — check it before trusting
the allocation.

```python
result_low = jaxgsa.pce.analyze(PROBLEM, X, Y1, order=2).shapley()
# UserWarning: jaxgsa: surrogate explained_variance is below 0.5 ...
print(result_low.Sh.sum())              # still exactly 1
print(result_low.explained_variance)    # ~0.4 — do not trust these shares
```

Because of this normalization, `backend="pce"` returns S1/ST that match
`jaxgsa.pce.analyze` exactly, while `backend="hdmr"` indices relate to
`jaxgsa.hdmr.analyze`'s (which normalize by `Var(Y)`) by a factor of
`explained_variance`.

## xarray export

`ShapleyResult.to_dataset()` converts results to a labeled
`xarray.Dataset`, just like the other jaxgsa result types.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:  (output: 2, param: 3)
# Data variables: Sh, S1, ST, explained_variance

print(ds.Sh.sel(param="x1"))
print(ds.explained_variance)
```

For time-series results, pass `time_coords` to label the time dimension.
`explained_variance` carries no `param` dimension — it is scalar,
`(output,)`, or `(time, output)`.

## Shape rules

| Y shape | backend | Sh / S1 / ST shape | explained_variance |
|---------|---------|--------------------|--------------------|
| `(N,)` | pce or hdmr | `(D,)` | `()` |
| `(N, K)` | pce or hdmr | `(K, D)` | `(K,)` |
| `(N, T, K)` | pce or hdmr | `(T, K, D)` | `(T, K)` |

D is always the last axis. Without `problem.output_names`, a 2D `Y` is always
read as `(N, K)`; with exactly one entry in `output_names`, a 2D `Y` is read
as `(N, T)` — timepoints of that single output — and flows through as
`(N, T, 1)`. Passing a pre-reshaped `(N, T, 1)` array also works.

## Practical caveats

- **Independent inputs are assumed (v1).** The Shapley value is especially
  attractive for dependent inputs, but the dependent-input formulation
  needs conditional-variance estimation and is future work — do not rely
  on the indices when inputs are strongly correlated.
- Both backends accept scalar `(N,)`, multi-output `(N, K)`, and
  time-series `(N, T, K)` `Y`.
- Interactions beyond the surrogate's truncation (`order` for PCE,
  `maxorder` for HDMR) are absent from the allocation — raise the order
  until `explained_variance` stabilizes near 1.
- The HDMR backend inherits `jaxgsa.hdmr.analyze`'s input contract: at least 300
  samples and `maxorder` in `{1, 2, 3}` (clamped with a warning when
  `D < maxorder`).
- Setting a keyword that belongs to the non-selected backend raises
  `ValueError` rather than being silently ignored.

## See also

- [PCE](/examples/pce) for the default backend's surrogate and its
  emulator/LOO-RMSE workflow.
- [RS-HDMR](/examples/hdmr) for the multi-output backend's surrogate.
- [Basic Example](/examples/basic) for the Sobol workflow when you can
  afford a structured Saltelli design.
- [Borgonovo Delta](/examples/borgonovo) for a moment-independent
  importance measure from the same given-data setting.
- [Methods](/guide/methods) for the theory behind Shapley effects and when
  to choose them over S1/ST.
- [API Reference](/api/#shapley-results) for full parameter
  documentation.
