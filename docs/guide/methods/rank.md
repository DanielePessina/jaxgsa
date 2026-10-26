# Rank-based S1: first-order Sobol' indices from given data

The rank estimator gives the first-order Sobol' index of every input from one
`(X, Y)` sample. The index is the share of the output variance that an input
explains alone:

$$
S_i = \frac{\mathrm{Var}(E[Y \mid X_i])}{\mathrm{Var}(Y)}.
$$

This is the same `S1` that [Sobol'](/guide/methods/sobol) reports. The
difference is the data. Sobol' needs its own structured design. The rank
estimator takes any sample of inputs and outputs, for example a Monte Carlo
study, a Latin hypercube, or runs from an earlier project.

The idea is simple. For each input, sort the rows by that input. Pair each
output with the output of the next row in the sort. When the output depends
strongly on the input, neighbouring rows have similar outputs, and the pairs
are strongly correlated. The index is the covariance of the pairs divided by
the output variance. The estimator is from
[Gamboa, Gremaud, Klein and Lagnoux (2022)](https://doi.org/10.3150/21-BEJ1421).
It needs one sort per input and it has no tuning parameter: no bin count, no
kernel and no surrogate. The formula and the references are on the
[API page](/api/rank).

## Import style

```python
from jaxgsa import rank
# rank.analyze(...)
```

## A first run on Ishigami

```python
import numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import ANALYTICAL_S1, PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=4000, seed=42)
Y = evaluate(X)  # or the outputs you already have
result = jaxgsa.rank.analyze(PROBLEM, X, Y)

np.set_printoptions(precision=3, suppress=True)
print("S1:", np.asarray(result.S1), "exact:", ANALYTICAL_S1)
```

```text
jaxgsa.rank.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=4000 runs, T=1 x K=1 output slice
    invalid: none found in 4000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1561 s
    slice_chunk_size: 1 (resolved from the memory budget)
  results: top 3 of 3 parameters by S1
    1. x2  S1=0.4617
    2. x1  S1=0.3072
    3. x3  S1=0.01836
S1: [0.307 0.462 0.018] exact: [0.314 0.442 0.   ]
```

The block above `S1:` is the run summary. Pass `verbose=False` to silence it.

The ranking matches the exact answer: `x2`, then `x1`, then `x3`. Each
estimate is within 0.02 of the exact value. At $N = 4000$ the standard error
is about 0.02 per index, so these errors are sampling noise.

`x3` reads 0.018 against an exact 0. `x3` affects Ishigami only through an
interaction with `x1`. A first-order index cannot see that interaction, and
the rank estimator gives no total-order index. The method therefore cannot
tell an input with no effect from an input that acts only through
interactions.

## Sample size

The error falls as the sample grows. The same model on fresh samples of three
sizes:

```python
for n in (500, 2000, 16000):
    Xn = jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=n, seed=1)
    r = jaxgsa.rank.analyze(PROBLEM, Xn, evaluate(Xn), verbose=False)
    err = np.max(np.abs(np.asarray(r.S1) - ANALYTICAL_S1))
    print(f"N={n:>6}  S1={np.asarray(r.S1)}  largest error={err:.3f}")
```

```text
N=   500  S1=[ 0.365  0.439 -0.004]  largest error=0.051
N=  2000  S1=[ 0.333  0.409 -0.004]  largest error=0.034
N= 16000  S1=[ 0.319  0.429 -0.007]  largest error=0.014
```

The index is not clipped. A small negative value, such as `-0.004` for `x3`,
is sampling noise around 0. The index never exceeds 1.

## Confidence intervals

`n_bootstrap` gives intervals in `S1_conf`. A bootstrap needs a `key`:

```python
import jax

boot = jaxgsa.rank.analyze(
    PROBLEM, X, Y, n_bootstrap=500, key=jax.random.key(0), verbose=False
)
print("S1_conf:")
print(np.asarray(boot.S1_conf))
```

```text
S1_conf:
[[ 0.263  0.427 -0.05 ]
 [ 0.348  0.5    0.112]]
```

Row 0 holds the lower bounds and row 1 the upper bounds. All three intervals
contain the exact value. The interval for `x3` contains 0.

The usual bootstrap draws rows with replacement. That fails here: a repeated
row sorts next to its own copy, and the pair adds a large positive bias. So
jaxgsa draws each resample as a random 80% of the rows without replacement,
and rescales the spread to the full sample size. It also measures the bias of
order $1/N$ and reports an interval that covers both the plain and the
bias-corrected estimate. The [API page](/api/rank#confidence-intervals) gives
the measured coverage. Treat intervals at $N$ below about 50 with care.

## What it returns

`RankResult` has these fields:

| Field | Content |
| --- | --- |
| `S1` | One index per input, with the leading axes of `Y`. `NaN` for an output with zero variance. |
| `S1_conf` | Lower and upper bounds on a leading axis of length 2. `None` without a bootstrap. |
| `ci`, `problem`, `invalid` | The interval settings, the problem, and the non-finite report. |

`result.to_dataset()` gives a labeled `xarray.Dataset`. See
[xarray output](/examples/xarray).

## Options

| Argument | Default | Effect |
| --- | --- | --- |
| `n_bootstrap` | `0` | Number of resamples for `S1_conf`. |
| `conf_level` | `0.95` | Confidence level of the interval. |
| `ci_method` | `"quantile"` | `"quantile"` or `"gaussian"` interval endpoints. |
| `key` | `None` | A `jax.random` key. Required when `n_bootstrap > 0`. |
| `slice_chunk_size` | `None` | Output slices per batch. A memory setting; it changes no index. |
| `on_invalid` | `"raise"` | Policy for non-finite rows: `"raise"`, `"drop"` or `"propagate"`. |
| `verbose` | `True` | Prints the run summary. |

## Shape rules

`X` has shape `(N, D)`. `Y` has shape `(N,)`, `(N, K)` or `(N, T, K)`. jaxgsa
computes the index for every `(t, k)` slice, so `S1` has shape `(D,)`,
`(K, D)` or `(T, K, D)`. See
[Multi-output and time series](/examples/multi-output).

## Correlated inputs

Supported. The index $\mathrm{Var}(E[Y \mid X_i]) / \mathrm{Var}(Y)$ is defined
for any joint distribution of the inputs. Under dependence it includes what
an input carries through its correlated partners. An input that the model
ignores can score above 0 when it is correlated with an influential input.
See [Dependent inputs](/guide/dependent-inputs).

## Limits

- First order only. There is no `ST` and no `S2`.
- Categorical inputs are refused. Their level codes are heavily tied, and the
  rank pairing is not defined for ties. [PAWN](/guide/methods/pawn),
  [Borgonovo delta](/guide/methods/borgonovo),
  [optimal transport](/guide/methods/optimal-transport) and
  [Sobol'](/guide/methods/sobol) accept categorical inputs.
- Many tied values in a continuous input bias the index, because row order
  then decides which outputs pair. A few ties change nothing of substance.
- `analyze` needs at least 3 rows.
- `jaxgsa.rank.indices(problem, X, Y)` works under `jax.jit` and `jax.vmap`.
  The gradient with respect to `Y` is defined. The gradient with respect to
  `X` is 0 almost everywhere, because the estimator reads only the ranks of
  `X`.

## See also

- [Rank API](/api/rank) for the formula, the interval method and the check
  against the R package `sensitivity`.
- [Sobol'](/guide/methods/sobol) for `S1`, `ST` and `S2` from a structured
  design.
- [PAWN](/guide/methods/pawn), [Borgonovo delta](/guide/methods/borgonovo) and
  [HSIC](/guide/methods/hsic) for other given-data indices.
- [Use existing data](/guide/existing-data) for the given-data workflow.
