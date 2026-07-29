# Categorical Inputs

Use this page when an input parameter is a choice, not a number. Examples:
a material grade, a solver variant, an on/off switch with more than two
states. jaxgsa calls these **categorical** marginals. A categorical
parameter has `L` unordered levels with declared probabilities. Samples
carry the **integer level codes** `0 .. L-1` (as floats) — codes, never
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

## Analyze with optimal transport and Borgonovo delta

Both given-data methods condition on **one class per level** for a
categorical column. Continuous columns keep their usual equal-frequency
rank classes. The indices depend only on the level partition. Relabeling
the levels does not change them.

```python
ot_result = jaxgsa.optimal_transport.analyze(problem, X, Y)
print(ot_result.ot)         # one index per parameter, [0, 1]
print(ot_result.advective)  # mean-shift part (= S1 / 2)

delta_result = jaxgsa.borgonovo.analyze(problem, X, Y)
print(delta_result.delta)   # density-based index per parameter
print(delta_result.S1)      # given-data first-order Sobol index
```

A declared level with no observed samples is dropped from the class
average, with a `UserWarning`. `n_partitions` / `n_classes` apply to the
continuous columns only.

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
# analyze given data with jaxgsa.optimal_transport or jaxgsa.borgonovo.
```

The same applies to `efast.sample`, `dgsm.analyze`, `pce.analyze`,
`hdmr.analyze`, `hsic.analyze`, `pawn.analyze`, and `shapley.analyze`.

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
