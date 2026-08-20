# Categorical Inputs

Use this page when an input parameter is a choice, not a number. A material
grade, a solver variant, a switch with more than two positions. By the end of
it you will have sensitivity indices that rank such a choice against your
numeric inputs, and you will know which methods accept it and which refuse it.

jaxgsa calls these categorical marginals. A categorical parameter has `L`
unordered levels with declared probabilities. Samples carry the integer level
codes `0 .. L-1`, stored as floats. They are codes, never physical values. Your
model maps each code to whatever the level means.

The whole page runs on one example: a reaction rate that depends on a
temperature and on which of three catalysts you pick.

## Declare a categorical parameter

```python
import jax

jax.config.update("jax_enable_x64", True)

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

print(problem.has_categorical_inputs)
print(problem.categorical_labels)
```

```
True
{'catalyst': ('Pt', 'Pd', 'Ni')}
```

`probs` must be positive and sum to 1. A small rounding error is renormalized;
a sum that is clearly off raises `ValueError`. `labels` is optional and
defaults to `"0" .. "L-1"`. Labels never enter the sample matrix. Use
`problem.categorical_labels` to map codes back to names in your own reports.

The `probs` are not decoration. They are the distribution the indices are
defined against. A level you will use half the time and a level you will use
2% of the time contribute to the output variance in proportion to how often
they occur, so declaring `[0.5, 0.3, 0.2]` when the real mix is uniform gives
you indices for a plant you do not operate.

## Sample and evaluate

`jaxgsa.sampling.monte_carlo` draws every marginal, categorical included.

```python
X = jaxgsa.sampling.monte_carlo(problem, n=8192, seed=0)
codes = X[:, 1].astype(int)
print("observed level frequencies:", np.round(np.bincount(codes) / len(codes), 3))

# The model maps each level code to its physical effect.
rate_constant = np.array([1.0, 1.8, 0.6])  # one entry per level
Y = np.exp(-rate_constant[codes] * (X[:, 0] - 300.0) / 100.0)
```

```
observed level frequencies: [0.498 0.301 0.2  ]
```

The column holds `0.0`, `1.0`, `2.0` at the declared frequencies. N=8192 is
generous for two parameters; it is chosen so the rarest level (`Ni`, at 0.2)
still gets about 1600 samples, which is what keeps its conditional
distribution estimable.

## Analyze with optimal transport, Borgonovo delta, and PAWN

These three are given-data methods: they take any `(X, Y)` pairs and need no
structured design. They all work by conditioning, which means splitting the
samples into classes by the value of one input and comparing the output
distribution within each class against the overall one. For a categorical
column each level is its own class, which is exactly the right split.
Continuous columns keep their usual conditioning: equal-frequency rank classes
for optimal transport and Borgonovo delta, equal-probability bins for PAWN. The
indices depend only on the level partition, so relabeling the levels does not
change them.

```python
ot_result = jaxgsa.optimal_transport.analyze(problem, X, Y)
```

```
jaxgsa.optimal_transport.analyze
  problem: D=2 (temperature, catalyst)
    marginals: uniform=1, categorical=1
    correlation: independent
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 8192 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.4148 s
    mode: univariate
    epsilon: 0.01
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 2 of 2 parameters by ot
    1. temperature  ot=0.5539
    2. catalyst     ot=0.1541
```

The `marginals: uniform=1, categorical=1` line is worth checking on every run.
It is where a categorical parameter that silently arrived as a float range
shows up.

```python
delta_result = jaxgsa.borgonovo.analyze(problem, X, Y, verbose=False)
pawn_result = jaxgsa.pawn.analyze(problem, X, Y, verbose=False)

print("ot        :", np.round(ot_result.ot, 3))
print("advective :", np.round(ot_result.advective, 3))
print("delta     :", np.round(delta_result.delta, 3))
print("delta's S1:", np.round(delta_result.S1, 3))
print("pawn      :", np.round(pawn_result.pawn, 3))
print("valid bins:", pawn_result.n_valid_bins)
```

```
ot        : [0.554 0.154]
advective : [0.372 0.11 ]
delta     : [0.511 0.217]
delta's S1: [0.743 0.22 ]
pawn      : [0.478 0.313]
valid bins: [10 3]
```

All three indices lie in `[0, 1]` and rank `catalyst` on the same scale as
`temperature`, so you read one ranking over both kinds of input. All three
agree on the ordering: temperature first, catalyst second. They disagree on the
gap, because they measure different distances between distributions, and that
is expected. Do not compare an `ot` of 0.154 against a `pawn` of 0.313 as if
they were the same quantity. Compare within a column.

Two extras worth reading here. `advective` is the share of the
optimal-transport index that comes from a shift in the output mean, as opposed
to a change in the distribution's shape. For `catalyst` it is 0.110 out of
0.154, so most of what the catalyst does is move the mean. `n_valid_bins` is
`[10, 3]`: ten quantile bins for the continuous temperature, and exactly three
for the catalyst, one per level. If that second number ever comes back below
`L`, a declared level was too rare to estimate and PAWN dropped it.

A declared level with no observed samples is dropped from the class average,
with a `JaxgsaWarning`. `n_partitions` / `n_classes` / `n_bins` apply to the
continuous columns only. PAWN gives a level with too few samples a `NaN` KS
value and drops it from the median, max, or mean over bins, so a rare level
cannot distort the index.

### Delta needs a continuous output

`borgonovo.analyze` supports a continuous output distribution only. A
categorical input is fine. A categorical or otherwise discrete output is
not. The estimator compares densities on a shared grid, and an atom is a
spike no grid resolves. `analyze` checks the output first and raises
`ValueError` when a column takes at most 20 distinct values and those
values are fewer than 1% of the samples:

```python
Y_discrete = rate_constant[codes]  # 3 distinct values, no noise
jaxgsa.borgonovo.analyze(problem, X, Y_discrete)
```

```
ValueError: jaxgsa.borgonovo.analyze supports a continuous output distribution
only, but the output takes only 3 distinct values in 8192 samples. The delta
estimator compares Gaussian kernel density estimates on a shared output grid; an
atomic density is a spike that no grid resolves, so the index would report the
grid resolution, not the model. Use jaxgsa.optimal_transport.analyze for a
discrete output: it compares empirical distributions directly and needs no
density.
```

Optimal transport and PAWN both accept a discrete output. Use one of them
instead. A continuous output rounded to a few decimals is not refused, and
neither is a constant column, whose exact answer is `delta = S1 = 0`.

### Delta on a near-deterministic level

A categorical level often maps to one output value plus a small amount of
noise. The conditional density is then a spike. The delta estimator compares
densities on a shared output grid of `grid_size` points, and it cannot resolve
a spike much narrower than one grid step. jaxgsa widens such a class to a
bandwidth the grid can integrate and emits a `JaxgsaWarning`.

Here is the size of that effect. A three-level model whose levels map to 0, 1
and 2 with noise of 1e-3 has true `delta = 2/3` for the level parameter:

```python
rng = np.random.default_rng(0)
pc = jaxgsa.Problem.from_dict(
    {"x": (0.0, 1.0), "lvl": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}}
)
Xc = jaxgsa.sampling.monte_carlo(pc, n=16384, seed=0)
lvl = Xc[:, 1].astype(int)
Yc = np.array([0.0, 1.0, 2.0])[lvl] + 1e-3 * rng.standard_normal(len(lvl))

for gs in (50, 100, 200):
    r = jaxgsa.borgonovo.analyze(pc, Xc, Yc, grid_size=gs, verbose=False)
    print(f"grid_size={gs:3d}: delta={np.round(r.delta, 3)}")
```

```
grid_size= 50: delta=[0.206 0.545]
grid_size=100: delta=[0.206 0.613]
grid_size=200: delta=[0.206 0.656]
```

Against a true 2/3 = 0.667 the estimate is biased low and climbing with
`grid_size`, while the continuous `x` sits unmoved at 0.206. More samples will
not fix this; the bias is set by the grid, not by `N`. So read delta on a
near-deterministic level as a ranking signal, not a calibrated number, and
raise `grid_size` if you need the number itself.

If an estimate leaves `[0, 1]` by more than 0.05, the computation failed.
`analyze` raises `ValueError` naming the parameter, the observed value, and the
knob to turn. The message reads what the run actually did: it names
`degenerate_bandwidth` and the fraction that would fix it when a class was
floored, and `degenerate_tol` when no class was. The value is never clipped,
because a clipped value looks plausible and is still wrong. A confidence bound
outside the range only warns, because the point estimate is the contract and
the interval is a diagnostic.

`degenerate_tol` and `degenerate_bandwidth` let you override when a class
counts as too narrow and how wide it is made. The defaults suit most work.
Neither is refused on the setting alone. `degenerate_bandwidth` only ever
reaches a class already called degenerate, so on data with no such class it
changes nothing at any value, and even on a degenerate class a narrow kernel
only breaks the run if a grid point lands on the peak. Raising `degenerate_tol`
biases the answer instead: a class whose own bandwidth sits between the floor
and the tolerance is *narrowed* to the floor, which inflates delta for the very
classes the higher tolerance said to distrust. That stays a valid computation,
so it is a bias to know about, not an error.

## Analyze with Sobol' (the Saltelli scheme)

Sobol' indices need the Saltelli design, a structured set of sample rows built
by swapping columns between two base matrices. That design works with a
categorical input because its estimators only ever copy coordinate values
between sample rows. They never need an ordering on those values. Nothing in
the estimator asks whether `Pd` is greater than `Pt`.

```python
sr = jaxgsa.sobol.sample(problem, 2**13, seed=0)
codes = sr.samples[:, 1].astype(int)
Y_sobol = np.exp(-rate_constant[codes] * (sr.samples[:, 0] - 300.0) / 100.0)
result = jaxgsa.sobol.analyze(sr, Y_sobol)
```

```
jaxgsa.sobol.sample: D=2, mode=second-order, base_n=4096, requested_runs>=8192, n_runs=13266, n_expanded=24576, duplicates_removed=11310 (46.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=2 (temperature, catalyst)
    marginals: uniform=1, categorical=1
    correlation: independent
    output: N=24576 runs, T=1 x K=1 output slice
    invalid: none found in 4096 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.4138 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 2 of 2 parameters by ST
    1. temperature  ST=0.7729
    2. catalyst     ST=0.2574
```

`S1` is `[0.739, 0.226]` and `ST` is `[0.773, 0.257]`, one entry per parameter,
`catalyst` included, so the choice of catalyst is ranked against the
temperature on the same variance scale. The gap between `S1` and `ST` is about
0.03 for both, which is the `temperature x catalyst` interaction: the catalyst
changes how fast the rate falls with temperature, so the two are not additive.

Compare that `S1` with the `delta's S1` printed earlier, `[0.743, 0.220]`. Two
different estimators on two different designs, agreeing to about 0.006. That
cross-check costs one line and is the fastest way to catch a code-mapping
mistake in your own model wrapper.

Note the `duplicates_removed=11310 (46.0%)` in the sampler line. A categorical
column collapses whole probability bins onto one code, so low-cardinality
problems have few distinct rows. The sampler normally inflates the design until
it has `n_samples` unique rows. For categorical problems it stops when the
achievable distinct-row count is reached and keeps duplicate rows, with a
`JaxgsaWarning`:

```python
p_flags = jaxgsa.Problem.from_dict(
    {
        "a": {"dist": "categorical", "probs": [0.5, 0.5]},
        "b": {"dist": "categorical", "probs": [0.5, 0.5]},
    }
)
jaxgsa.sobol.sample(p_flags, 4096, seed=0)
```

```
JaxgsaWarning: jaxgsa.sobol.sample: the requested n_samples=4096 unique rows
cannot be reached because the problem has only 4 possible distinct rows. The
design is returned with 4 unique rows and keeps its duplicate rows. Duplicates
are valid Saltelli samples; deduplication only saves model evaluations
```

Two binary flags give four possible rows, so the design cannot be wider than
that. Duplicate rows are valid Saltelli samples. Deduplication only saves model
evaluations.

::: warning `sr.samples` is an evaluation set, not a sample
`sr.samples` holds only the unique rows to evaluate. Deduplication removes
repeated rows, so the empirical frequencies of a column in `sr.samples` do not
match the declared marginal:

```python
p2 = jaxgsa.Problem.from_dict(
    {"x": (0.0, 1.0), "flag": {"dist": "categorical", "probs": [0.9, 0.1]}}
)
sr2 = jaxgsa.sobol.sample(p2, 2**12, seed=0, verbose=False)
col = sr2.samples[:, 1].astype(int)
expanded = sr2.samples[sr2.expanded_to_unique][:, 1].astype(int)

print("sr.samples freq:", np.round(np.bincount(col) / len(col), 3))
print("expanded freq  :", np.round(np.bincount(expanded) / len(expanded), 3))
```

```
sr.samples freq: [0.839 0.161]
expanded freq  : [0.9 0.1]
```

The declared marginal is exact in the expanded design, which
`jaxgsa.sobol.analyze` rebuilds through `sr.expanded_to_unique`. The indices
are therefore correct. Evaluate `sr.samples` and pass the outputs to `analyze`;
never reuse `sr.samples` on its own as a Monte Carlo design. For a plain sample
of the declared distribution, use `jaxgsa.sampling.monte_carlo`.
:::

## Which methods accept a categorical input

Four do: `optimal_transport`, `borgonovo`, `pawn`, and `sobol`. Every other
method treats inputs as continuous, so its indices would depend on the
arbitrary order of the level codes, and it raises instead:

```python
jaxgsa.morris.sample(problem, n_trajectories=16)
```

```
ValueError: jaxgsa.morris.sample requires continuous (orderable) inputs, but
parameters ['catalyst'] are categorical. Use jaxgsa.sobol.sample (the Saltelli
column-swap scheme is distribution-agnostic; it requires a problem with no
declared correlation), or analyze given data with jaxgsa.optimal_transport,
jaxgsa.borgonovo or jaxgsa.pawn.
```

Morris and eFAST are the two that cannot be repaired, and they fail for the
same underlying reason at two different points:

- Morris measures an elementary effect, the output change divided by a step
  along one input axis. `(Y(Pd) - Y(Pt)) / (1 - 0)` is arithmetic on a
  difference of labels. Relabel the catalysts and the number changes. There is
  no step size on an unordered set. `sobol.SobolSamples.to_morris` refuses for
  the same reason, even though the Saltelli design it converts from is
  perfectly valid.
- eFAST drives each input along a periodic search curve and reads the output
  spectrum. The curve needs the input to move continuously, and a categorical
  input can only jump between codes. The Fourier coefficients would describe
  the jumps in the coding, not the model.

The rest refuse because their machinery needs a continuous marginal:
`dgsm.analyze` differentiates with respect to the input, `pce.analyze` builds
orthogonal polynomials in it, `hdmr.analyze` and `vkoga.analyze` map it through
its inverse CDF (a step function, so not invertible), `hsic.analyze` puts a
distance kernel on it, `shapley.analyze` inherits its backend's limits, and
`kucherenko.sample` conditions on a copula that has no meaning for unordered
codes.

Correlation is also rejected for categorical parameters: a `problem.correlation`
entry touching one raises `ValueError` (polychoric coupling is future work).
Identity rows and columns are fine. Note the combined case. For a categorical
problem whose continuous parameters are correlated, no variance-based route
exists yet: `sobol.sample` refuses the correlation, and the design and analyzer
errors say so instead of pointing you in a circle. Optimal transport and
Borgonovo delta are the supported analyses there.

## Map codes back to labels

Result arrays keep the parameter axis; codes only appear in `X`. When you
report per-level statistics yourself, translate codes through the labels:

```python
labels = problem.categorical_labels["catalyst"]
for code in range(len(labels)):
    sel = X[:, 1] == code
    print(f"{labels[code]}: mean Y = {Y[sel].mean():.3f}")
```

```
Pt: mean Y = 0.627
Pd: mean Y = 0.468
Ni: mean Y = 0.755
```

That spread, 0.468 to 0.755, is what the `catalyst` index of 0.15 to 0.31 is
made of. Printing it is worth the two lines: an index tells you a parameter
matters, and only the per-level means tell you which way to move.

## Related docs

- [Non-Uniform Inputs](/examples/non-uniform-inputs)
- [Correlated Inputs](/examples/correlated-inputs)
- [Methods guide](/guide/methods)
