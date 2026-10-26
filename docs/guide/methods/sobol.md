# Sobol' indices

Sobol' indices split the output variance into shares. Each share belongs to
one input, or to a group of inputs that act together. The method answers one
question: how much of the output spread goes away when you know an input
exactly?

jaxgsa reports three indices:

- `S1`, the first-order index. It is the share of the output variance that an
  input explains alone.
- `ST`, the total-order index. It is the share that an input explains alone
  and through every interaction with other inputs.
- `S2`, the second-order index. It is the share that a pair of inputs explains
  together, and that neither input explains alone.

All three are fractions of the output variance, so they lie in `[0, 1]` up to
sampling noise. The gap `ST - S1` is the part of an input's effect that comes
from interactions.

Sobol' is a design-based method. `jaxgsa.sobol.sample` builds a Saltelli
design, you run the model on every row of it, and `jaxgsa.sobol.analyze`
reads the outputs. The estimators depend on the row layout of the design, so
an ordinary `(X, Y)` sample cannot replace it. The full signature and the
estimator references are on the [API page](/api/sobol).

## Import style

```python
from jaxgsa import sobol
# sobol.sample(...)
# sobol.analyze(...)
```

## Scalar example (Ishigami)

The Ishigami function has three inputs and known exact indices, so the
estimates can be checked.

```python
import numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import ANALYTICAL_S1, ANALYTICAL_ST, PROBLEM, evaluate

design = jaxgsa.sobol.sample(PROBLEM, base_n=4096, seed=0)
Y = evaluate(design.samples)  # your model here
result = jaxgsa.sobol.analyze(design, Y)

np.set_printoptions(precision=3, suppress=True)
print("S1:", np.asarray(result.S1), "exact:", ANALYTICAL_S1)
print("ST:", np.asarray(result.ST), "exact:", ANALYTICAL_ST)
print("S2[x1, x3]:", float(result.S2[0, 2]))
```

```text
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=4096, requested_runs>=32768, n_runs=32768, n_expanded=32768, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=32768 runs, T=1 x K=1 output slice
    invalid: none found in 4096 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.2859 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5571
    2. x2  ST=0.4421
    3. x3  ST=0.2434
S1: [0.316 0.443 0.002] exact: [0.314 0.442 0.   ]
ST: [0.557 0.442 0.243] exact: [0.558 0.442 0.244]
S2[x1, x3]: 0.24150460958480835
```

The first line comes from `sample`. It reports the design size: 4096 base
points and 32,768 model runs. The block after it comes from `analyze`. Pass
`verbose=False` to either call to silence it.

Every estimate is within 0.003 of the exact value. Read the three inputs in
turn:

- `x1` has `S1 = 0.316` and `ST = 0.557`. About 0.24 of its effect comes from
  interactions.
- `x2` has `S1` equal to `ST`. It acts alone.
- `x3` has `S1` near 0 and `ST = 0.243`. It has no effect alone. All of its
  effect comes from an interaction.

`S2[x1, x3] = 0.242` names the partner. The exact value is 0.244. The other
two pairs are near 0. A small `S1` alone is therefore not enough to fix an
input at a nominal value. Read `ST` for that.

## Design size

`sample` takes exactly one of two size arguments:

- `base_n` sets the number of base points. It must be a power of two.
- `n_samples` sets the minimum number of model runs. jaxgsa picks the
  smallest power-of-two `base_n` that reaches it.

Giving both, or neither, raises `ValueError`. For $D$ inputs a design costs
$N(2D+2)$ runs with second-order indices, and $N(D+2)$ runs without them.
`calc_second_order=False` drops `S2` and the extra rows:

```python
d1 = jaxgsa.sobol.sample(PROBLEM, base_n=4096, seed=0, calc_second_order=False, verbose=False)
r1 = jaxgsa.sobol.analyze(d1, evaluate(d1.samples), verbose=False)
print("runs:", design.n_runs, "vs", d1.n_runs, "S2:", r1.S2)
```

```text
runs: 32768 vs 20480 S2: None
```

`design.downsample(base_n, Y)` cuts a larger evaluated design down to a
smaller `base_n`, with no new model runs. The
[convergence example](/examples/basic) uses it to compare budgets.

`seed` makes the scrambled design reproducible. `scramble=False` with a `seed`
raises, because the unscrambled sequence has no randomness for the seed to
change.

## Confidence intervals

`n_bootstrap` resamples the base points and returns intervals in `S1_conf`,
`ST_conf` and `S2_conf`. A bootstrap needs a `key`:

```python
import jax

boot = jaxgsa.sobol.analyze(design, Y, n_bootstrap=500, key=jax.random.key(0), verbose=False)
print("ST_conf:")
print(np.asarray(boot.ST_conf))
```

```text
ST_conf:
[[0.512 0.423 0.23 ]
 [0.601 0.46  0.258]]
```

Row 0 holds the lower bounds and row 1 the upper bounds. The bootstrap treats
the base points as an independent random sample. A scrambled Sobol' sequence
converges faster than that, so the interval is not a calibrated 95% statement.
At a large `base_n` it is too wide. At a small `base_n` it can be too narrow.
The [API page](/api/sobol) gives the measured coverage. The
[bootstrap example](/examples/bootstrap) shows the intervals in more detail.

## Estimators

`analyze` takes `estimator=`. The default is `"saltelli-jansen"`. The other
names are `"jansen"`, `"janon-monod"`, `"martinez"`, `"mauntz-kucherenko"`
and `"azzini-rosati"`. All of them converge to the same indices. They differ
in sampling noise at a small `base_n`, and in whether an estimate can fall
outside `[0, 1]`. The default `ST` estimator is a mean of squares, so `ST` is
never negative. `"azzini-rosati"` needs a design with
`calc_second_order=True`. The [estimator table](/api/sobol#choosing-an-estimator)
lists the source of each one.

## What it returns

`SobolResult` has these fields:

| Field | Content |
| --- | --- |
| `S1`, `ST` | One index per input, with the leading axes of `Y`. |
| `S2` | A symmetric `(D, D)` matrix per output slice, with a NaN diagonal. `None` without second order. |
| `S1_conf`, `ST_conf`, `S2_conf` | Lower and upper bounds on a leading axis of length 2. `None` without a bootstrap. |
| `variance` | The output variance of each slice. |
| `estimator`, `ci`, `problem`, `invalid` | The estimator name, the interval settings, the problem, and the non-finite report. |

`result.to_dataset()` gives a labeled `xarray.Dataset`. See
[xarray output](/examples/xarray). `result.aggregate()` collapses many outputs
into one index per input, with each output weighted by its variance.

## Shape rules

`Y` holds one output per row of `design.samples`, in the same row order.
Accepted shapes are `(n_runs,)`, `(n_runs, K)` and `(n_runs, T, K)`. jaxgsa
computes the indices for every `(t, k)` slice. See
[Multi-output and time series](/examples/multi-output). Channels on different
time grids go in as a ragged list. See
[Irregular output grids](/examples/irregular-outputs).

## Inside a JAX transformation

`jaxgsa.sobol.indices(design, Y)` returns the same `S1`, `ST` and `S2` as
plain arrays, with no checks and no printing. It works under `jax.jit`,
`jax.vmap` and `jax.jacrev`. With `design.transform(theta)`, you can
differentiate an index with respect to the input bounds. See
[Differentiate Sobol indices](/guide/differentiable-sobol).

## Limits

- The inputs must be independent. `sobol.sample` refuses a problem with a
  declared correlation. The column swap in the Saltelli scheme assumes that
  each column can take any value whatever the other columns hold.
  [Kucherenko](/guide/methods/kucherenko) builds a design that keeps the
  declared dependence.
- The design is fixed. Rows from another study cannot be used.
- Categorical inputs are accepted. See
  [Categorical inputs](/examples/categorical-inputs).
- Small negative estimates of `S1` or `S2` are sampling noise around 0.
- `on_invalid` sets the policy for non-finite outputs. The default `"raise"`
  stops the analysis. See [Handle failed runs](/guide/failed-runs).

## See also

- [Sobol API](/api/sobol) for the full signature, the estimator references
  and the `slice_chunk_size` memory setting.
- [How many model runs are enough?](/examples/basic) for a convergence check.
- [Getting started](/guide/getting-started) for a first Sobol' analysis.
- [Morris](/guide/methods/morris) for elementary effects.
  `design.to_morris()` reads them from the same Saltelli design at no extra
  model runs.
- [Rank-based S1](/guide/methods/rank) for first-order indices from an
  existing `(X, Y)` sample.
- [Save and reload](/examples/save-load) for running the model in a separate
  job.
