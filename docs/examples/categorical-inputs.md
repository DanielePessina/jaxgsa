# Categorical Inputs

Use this page when an input parameter is a choice, not a number. Examples:
a material grade, a solver variant, an on/off switch with more than two
states. jaxgsa calls these categorical marginals. A categorical
parameter has `L` unordered levels with declared probabilities. Samples
carry the integer level codes `0 .. L-1` (as floats) — codes, never
physical values. Your model maps each code to whatever the level means.

## Declare a categorical parameter

```python
import numpy as np

import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "temperature": (300.0, 400.0),
        "catalyst": {
            "dist": "categorical",
            "probs": [0.5, 0.3, 0.2],
            "labels": ["Pt", "Pd", "Ni"],  # optional, for reporting only
        },
    }
)

print(problem.has_categorical_inputs)  # True
print(problem.categorical_labels)      # {'catalyst': ('Pt', 'Pd', 'Ni')}
```

`probs` must be positive and sum to 1. A small rounding error is
renormalized; a sum that is clearly off raises `ValueError`. `labels` is
optional and defaults to `"0" .. "L-1"`. Labels never enter the sample
matrix. Use `problem.categorical_labels` to map codes back to names in
your own reports.

## Sample and evaluate

`jaxgsa.sampling.monte_carlo` draws every marginal, categorical included.
The categorical column holds the codes `0.0`, `1.0`, `2.0` with the
declared frequencies.

```python
X = jaxgsa.sampling.monte_carlo(problem, n=8192, seed=0)
codes = X[:, 1].astype(int)

# The model maps each level code to its physical effect.
rate_constant = np.array([1.0, 1.8, 0.6])  # one entry per level
Y = np.exp(-rate_constant[codes] * (X[:, 0] - 300.0) / 100.0)
```

## Analyze with optimal transport, Borgonovo delta, and PAWN

All three given-data methods condition on one class per level for a
categorical column. Continuous columns keep their usual conditioning:
equal-frequency rank classes for optimal transport and Borgonovo delta,
equal-probability bins for PAWN. The indices depend only on the level
partition. Relabeling the levels does not change them.

```python
ot_result = jaxgsa.optimal_transport.analyze(problem, X, Y)
print(ot_result.ot)         # one index per parameter, [0, 1]
print(ot_result.advective)  # mean-shift part (= S1 / 2)

delta_result = jaxgsa.borgonovo.analyze(problem, X, Y)
print(delta_result.delta)   # density-based index per parameter
print(delta_result.S1)      # given-data first-order Sobol index

pawn_result = jaxgsa.pawn.analyze(problem, X, Y)
print(pawn_result.pawn)     # KS-based index per parameter, [0, 1]
```

A declared level with no observed samples is dropped from the class
average, with a `UserWarning`. `n_partitions` / `n_classes` / `n_bins`
apply to the continuous columns only. PAWN gives a level with too few
samples a `NaN` KS value and drops it from the median, max, or mean over
bins, so a rare level cannot distort the index.

### Delta needs a continuous output

`borgonovo.analyze` supports a continuous output distribution only. A
categorical input is fine. A categorical or otherwise discrete output is
not. The estimator compares densities on a shared grid, and an atom is a
spike no grid resolves. `analyze` checks the output first and raises
`ValueError` when a column takes at most 20 distinct values and those
values are fewer than 1% of the samples:

```python
Y_discrete = rate_constant[codes]  # 3 distinct values, no noise
try:
    jaxgsa.borgonovo.analyze(problem, X, Y_discrete)
except ValueError as e:
    print(e)
# jaxgsa.borgonovo.analyze supports a continuous output distribution only,
# but the output takes only 3 distinct values in 8192 samples. ... Use
# jaxgsa.optimal_transport.analyze for a discrete output: it compares
# empirical distributions directly and needs no density.
```

Optimal transport and PAWN both accept a discrete output. Use one of them
instead. A continuous output rounded to a few decimals is not refused, and
neither is a constant column, whose exact answer is `delta = S1 = 0`.

### Delta on a near-deterministic level

A categorical level often maps to one output value plus a small amount of
noise. The conditional density is then a spike. The delta estimator
compares densities on a shared output grid of `grid_size` points, and it
cannot resolve a spike much narrower than one grid step. jaxgsa widens
such a class to a bandwidth the grid can integrate and emits a
`UserWarning`.

Two things follow from this:

- The delta of such an input depends on `grid_size` and is biased low.
  On a three-level model with true delta `2/3` and negligible noise, the
  estimate is 0.56 at `grid_size=50` and about 0.61 at `grid_size=100` and
  above. The bias does not go away as `N` grows. Read delta on a
  near-deterministic level as a ranking signal, not a calibrated number.
  `grid_size` is the knob that moves it.
- If an estimate still leaves `[0, 1]` by more than 0.05, the computation
  failed. `analyze` raises `ValueError` naming the parameter, the observed
  value, and both knobs. The value is never clipped: a clipped value looks
  plausible and is still wrong. A confidence bound outside the range only
  warns, because the point estimate is the contract and the interval is a
  diagnostic.

`degenerate_tol` and `degenerate_bandwidth` let you override when a class
counts as too narrow and how wide it is made. The defaults suit most work.

## Analyze with Sobol' (the Saltelli scheme)

The Saltelli design works because its estimators only ever copy coordinate
values between sample rows — they never need an ordering on them.

```python
sr = jaxgsa.sobol.sample(problem, 2**13, seed=0)
codes = sr.samples[:, 1].astype(int)
Y = np.exp(-rate_constant[codes] * (sr.samples[:, 0] - 300.0) / 100.0)
result = jaxgsa.sobol.analyze(sr, Y)
print(result.S1, result.ST)
```

One caveat: a categorical column collapses whole probability bins onto
one code, so low-cardinality problems have few distinct rows. The sampler
normally inflates the design until it has `n_samples` unique rows; for
categorical problems it stops when the achievable distinct-row count is
reached and keeps duplicate rows, with a `UserWarning`. Duplicate rows
are valid Saltelli samples — deduplication only saves model evaluations.

::: warning `sr.samples` is an evaluation set, not a sample
`sr.samples` holds only the unique rows to evaluate. Deduplication removes
repeated rows, so the empirical frequencies of a column in `sr.samples` do
not match the declared marginal. With `probs = [0.9, 0.1]` the `sr.samples`
column shows about `[0.84, 0.16]`, not `[0.9, 0.1]`. Categorical dedup
rates are high, so the distortion is easy to see.

The declared marginal is exact in the expanded design, which
`jaxgsa.sobol.analyze` rebuilds through `sr.expanded_to_unique`. The
indices are therefore correct. Evaluate `sr.samples` and pass the outputs
to `analyze`; never reuse `sr.samples` on its own as a Monte Carlo design.
For a plain sample of the declared distribution, use
`jaxgsa.sampling.monte_carlo`.
:::

## Methods that refuse categorical inputs

Every other method treats inputs as continuous. Its indices would depend
on the arbitrary code order, so it raises a clear error instead:

```python
try:
    jaxgsa.morris.sample(problem, n_trajectories=16)
except ValueError as e:
    print(e)
# jaxgsa.morris.sample requires continuous (orderable) inputs, but
# parameters ['catalyst'] are categorical. Use jaxgsa.sobol.sample
# (the Saltelli column-swap scheme is distribution-agnostic), or
# analyze given data with jaxgsa.optimal_transport, jaxgsa.borgonovo, or
# jaxgsa.pawn.
```

The same applies to `efast.sample`, `dgsm.analyze`, `pce.analyze`,
`hdmr.analyze`, `hsic.analyze`, and `shapley.analyze`.

Correlation is also rejected for categorical parameters: a
`problem.correlation` entry touching one raises `ValueError` (polychoric
coupling is future work). Identity rows and columns are fine.

## Map codes back to labels

Result arrays keep the parameter axis; codes only appear in `X`. When you
report per-level statistics yourself, translate codes through the labels:

```python
labels = problem.categorical_labels["catalyst"]
for code in range(len(labels)):
    sel = X[:, 1] == code
    print(f"{labels[code]}: mean Y = {Y[sel].mean():.3f}")
```

## Related docs

- [Non-Uniform Inputs](/examples/non-uniform-inputs)
- [Correlated Inputs](/examples/correlated-inputs)
- [Methods guide](/guide/methods)
