# Getting started

jaxgsa answers one question: which of your model's inputs move its output? You
give it input samples and the outputs your model produced for them. It gives
back sensitivity indices, numbers that rank the inputs and measure the
interactions between them. Everything runs in JAX, so the estimators are
JIT-compiled and the same code runs on CPU, GPU, or TPU.

This page runs one Sobol analysis end to end. Producing the numbers is the easy
half. The harder half is knowing what they mean and when to distrust them, so
most of this page is about reading the output rather than making it.

## Installation

```bash
pip install jaxgsa
# or, with uv:
uv add jaxgsa
```

The development version:

```bash
pip install git+https://github.com/DanielePessina/jaxgsa.git
```

To work on jaxgsa itself:

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev   # or: pip install -e ".[dev]"
```

## Citing jaxgsa

If you use jaxgsa in research, cite the exact version that produced your
results. Use the metadata in the repository's
[`CITATION.cff`](https://github.com/DanielePessina/jaxgsa/blob/master/CITATION.cff).
If a version-specific DOI is available, use that DOI; otherwise cite the
corresponding Git tag or GitHub release. Also cite the primary paper for each
sensitivity method you use, as listed in the [methods guide](/guide/methods).

## Your first analysis

Four steps: define the inputs, draw the design, run your model, analyze.

The model here is the Ishigami function. Its exact indices are known on paper,
which lets you check your reading of the output against the truth. Swap in your
own model at step 3.

```python
import jax.numpy as jnp
import jaxgsa

# 1. Name each input and give it a range.
problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

# 2. Draw the design. Sobol estimators need a specific row layout, so use
#    jaxgsa.sobol.sample() instead of random points.
design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)

# 3. Run your model on every row of design.samples, shape (n_runs, D).
def model(X):
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )

Y = model(design.samples)

# 4. Analyze.
result = jaxgsa.sobol.analyze(design, Y)
```

Both calls print, because `verbose=True` is the default:

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=512, requested_runs>=4096, n_runs=4096, n_expanded=4096, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4096 runs, T=1 x K=1 output slice
    invalid: none found in 512 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.6519 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.6266
    2. x2  ST=0.44
    3. x3  ST=0.2423
```

`verbose=True` is the default on all thirteen `analyze()` functions and on the
four design samplers, `jaxgsa.sobol.sample`, `jaxgsa.morris.sample`,
`jaxgsa.efast.sample`, and `jaxgsa.kucherenko.sample`.
`jaxgsa.sampling.monte_carlo` is the exception and takes no `verbose` keyword
at all. Pass `verbose=False` to silence any call that has one.

## What the summary block tells you

Read it top to bottom. Every line is there to catch a mistake before you act on
the indices.

The `sample` line reports the budget it actually spent. You asked for at least
4096 unique model runs. jaxgsa picked `base_n=512`, the smallest power of two
whose Saltelli expansion of `base_n * (2D + 2)` reaches your request, giving
`n_runs=4096`. `duplicates_removed=0 (0.0%)` matters in low dimensions, where
the Saltelli construction repeats rows and jaxgsa strips the repeats so you do
not pay to evaluate the same input twice.

`problem: D=3 (x1, x2, x3)` and `marginals: uniform=3` echo the `Problem` back
to you. If you meant a Gaussian input and see `uniform=3`, you built the wrong
problem, and the indices below are answers to a different question.

`correlation: independent` says no dependence structure was declared. Sobol
indices assume independent inputs, and `jaxgsa.sobol.sample` refuses a
correlated `Problem` outright rather than returning a number that looks fine.

`output: N=4096 runs, T=1 x K=1 output slice` is jaxgsa telling you how it
interpreted the shape of your `Y`. One scalar output per run here. If you pass a
`(N, T, K)` array of time-resolved outputs and this line says `T=1`, your array
was the wrong shape.

`invalid: none found in 512 Saltelli groups (policy 'raise')` is the non-finite
check. It reports groups, not rows, because one failed model run condemns the
whole Saltelli group it sits in. See [when a model run
fails](#when-a-model-run-fails) below.

The timing line includes XLA compilation on the first call in a process. A
second analysis of the same shape reuses the compiled kernel and is much faster,
so do not read 0.65 s as the cost of the estimator. `slice_chunk_size: 1` is the
batching width jaxgsa derived from its memory budget; with one output slice
there is nothing to batch. See [Configuration](/guide/configuration).

`estimator: saltelli-jansen` names which of the six estimator pairs produced
these numbers. They disagree at finite sample size, so an index without its
estimator is ambiguous. It is also stored on `result.estimator`.

The results section ranks by `ST` and shows the top five parameters, or all of
them when there are fewer.

## Reading the numbers

The full arrays live on the result object:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("S1:", np.asarray(result.S1))
print("ST:", np.asarray(result.ST))
print("S2:")
print(np.asarray(result.S2))
```

```
S1: [0.3387 0.4421 0.0155]
ST: [0.6266 0.44   0.2423]
S2:
[[    nan -0.0356  0.2128]
 [-0.0356     nan  0.0054]
 [ 0.2128  0.0054     nan]]
```

Each number is a share of the output variance.

`S1[i]` is the variance `x_i` explains on its own, averaged over everything the
other inputs do. Here `x2` carries 0.44 of the variance by itself.

`ST[i]` adds every interaction `x_i` takes part in. This is the one to use when
you want to fix an input to a constant and stop sampling it. `ST` near zero is
the licence to do that. `S1` near zero is not.

`x3` is the whole reason that distinction exists. Its `S1` is 0.0155, near
nothing, but its `ST` is 0.2423. On its own `x3` does nothing. Through its
interaction with `x1` it accounts for a quarter of the variance. Screening on
`S1` alone would have thrown it away.

`S2[i, j]` is the pairwise interaction share. The diagonal is `NaN`, because a
parameter's interaction with itself is not defined, and the matrix is mirrored
so you can index it either way. `S2[0, 2] = 0.2128` finds exactly the `x1`-`x3`
interaction that the `ST`-minus-`S1` gap pointed at. The `-0.0356` for `x1`-`x2`
is a variance share estimated as negative, which is the estimator telling you
the true value is small compared with its own sampling noise. jaxgsa does not
clip negative estimates to zero, because the clip would hide that signal.

## How much of that is sampling noise

The indices above came from 4096 model runs. They are estimates. Before you
report one, ask how wide it is, and bootstrap resampling answers that:

```python
import jax

boot = jaxgsa.sobol.analyze(
    design, Y, n_bootstrap=1000, key=jax.random.key(0), verbose=False
)
print("S1 lower/upper:")
print(np.asarray(boot.S1_conf))
print("ST lower/upper:")
print(np.asarray(boot.ST_conf))
```

```
S1 lower/upper:
[[ 0.2529  0.3674 -0.065 ]
 [ 0.4245  0.5199  0.0988]]
ST lower/upper:
[[0.4953 0.3853 0.2091]
 [0.7702 0.5005 0.2799]]
```

`S1_conf` and `ST_conf` are bounds, not half-widths. The leading axis holds
`[lower, upper]`, so column `i` of row 0 and row 1 bracket index `i` at the 95%
level.

Now read the point estimates again with those bounds beside them. `ST[0] =
0.6266` is really 0.50 to 0.77, an interval a quarter of the total variance
wide. `S1[2] = 0.0155` is really -0.065 to 0.099, an interval straddling zero,
which is the correct statement that 4096 runs cannot tell `x3`'s direct effect
apart from nothing. `ST[1] = 0.44` sits in 0.39 to 0.50 and is the only index
here you could quote to two digits.

The fix is more runs. Sobol convergence goes as roughly `1/sqrt(N)`, so
narrowing an interval by 4x costs 16x the model evaluations. Here is 131072 runs
of the same model:

```python
big = jaxgsa.sobol.sample(problem, n_samples=131072, seed=42, verbose=False)
big_result = jaxgsa.sobol.analyze(big, model(big.samples), verbose=False)

print("S1:", np.asarray(big_result.S1))
print("ST:", np.asarray(big_result.ST))
print("S2:")
print(np.asarray(big_result.S2))
```

```
S1: [0.3128 0.4426 0.0007]
ST: [0.5572 0.4426 0.2437]
S2:
[[   nan 0.0009 0.2439]
 [0.0009    nan 0.    ]
 [0.2439 0.        nan]]
```

The exact values for this function, with `a = 7` and `b = 0.1` on
$[-\pi, \pi]^3$, are `S1 = 0.3139, 0.4424, 0` and `ST = 0.5576, 0.4424,
0.2437`. Every index now matches to three decimals, and `S2[0, 2] = 0.2439`
against an exact `0.2437`. The `-0.0356` noise in the `x1`-`x2` cell collapsed
to `0.0009`. Your own model has no such table to check against, which is the
reason to run the bootstrap.

::: tip
Convergence is per index, not per analysis. `ST[1]` was already good at 4096
runs while `ST[0]` needed 32x more. Size your budget against the index you
actually care about.
:::

## When a model run fails

Real models return `NaN`. jaxgsa checks for it and, by default, refuses to
continue:

```python
Y_broken = Y.at[17].set(jnp.nan)
jaxgsa.sobol.analyze(design, Y_broken)
```

```
ValueError: jaxgsa.sobol.analyze: 1 of 512 Saltelli groups hold a non-finite
value (NaN or inf) in Y. Non-finite rows: [17]. They condemn Saltelli groups
[2], which covers 8 rows. An index computed from the rest is a different
quantity from the one you asked for, so this raises by default. Investigate
those runs, or pass on_invalid='drop' to analyze the remainder, or
on_invalid='propagate' to let the value reach the indices.
```

One bad row takes eight rows with it, because the Sobol estimator reads a
Saltelli group as a unit. That is why the error names groups. Dropping the
group is a real option, and `on_invalid='drop'` does it, but understand that you
are then estimating over a design with a hole in it. Chasing down run 17 is
usually the better answer.

## Controlling jaxgsa's warnings

Not everything jaxgsa objects to is fatal. A result that is degraded but still
valid gets a warning instead: a float64 array being truncated to float32, PAWN
keeping too few usable bins, VKOGA trained on a correlated design. Every one of
them carries the same category, `jaxgsa.JaxgsaWarning`, so you can act on
jaxgsa's warnings without touching NumPy's, SciPy's, or JAX's.

`JaxgsaWarning` subclasses `UserWarning`, so a filter you already have on
`UserWarning` keeps working.

```python
import warnings
from jaxgsa import JaxgsaWarning

# Fail the run on anything jaxgsa warns about. Good for CI and for a
# production pipeline where a degraded index must not pass silently.
warnings.simplefilter("error", JaxgsaWarning)

# Silence them, when you have read the warning and accepted it.
warnings.simplefilter("ignore", JaxgsaWarning)

# Show every occurrence instead of only the first from each call site.
warnings.simplefilter("always", JaxgsaWarning)

# Or capture them, to log or assert on.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", JaxgsaWarning)
    result = jaxgsa.sobol.analyze(design, Y)
for w in caught:
    print(w.category.__name__, w.message)
```

Escalating to `"error"` is worth doing once on a new pipeline. jaxgsa's
warnings name a specific defect in your setup, and reading them early is
cheaper than discovering the defect in a published index.

## Defining a problem

A `Problem` gives each input a name and a distribution. jaxgsa calls the inputs
parameters in code.

```python
from jaxgsa import Problem

problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})
```

A bare `(low, high)` tuple means uniform. `from_dict` also takes
`GaussianSpec`, `CategoricalSpec`, and the plain-dict form of each. Names must
be strings and must be unique; a duplicate name raises rather than silently
overwriting a parameter. See [Non-Uniform
Inputs](/examples/non-uniform-inputs) for the Gaussian marginals and the
truncation rules.

## Saving a design

Sampling and model evaluation are usually separate jobs. The model may run on a
cluster or take hours. `jaxgsa.sobol.sample()` returns a `SobolSamples` you can
write to disk and reload, and the reloaded object keeps the Saltelli metadata
that `analyze()` needs.

```python
design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
design.save("runs/ishigami")

restored = jaxgsa.sobol.SobolSamples.load("runs/ishigami")
Y = model(restored.samples)
result = jaxgsa.sobol.analyze(restored, Y)
```

This writes `runs/ishigami.npz` holding the sample matrix, the problem
definition, and the expansion metadata. The parent directory must already
exist; `save` will not create it.

Never rebuild a design by hand from a saved `X` matrix. The Saltelli row order
is what the estimator reads, and a shuffled or re-sorted matrix gives numbers
that look plausible and are wrong.

## Where to go next

- [Methods](/guide/methods) compares the thirteen methods, and tells you when Sobol is the wrong one
- [Basic Example (Ishigami)](/examples/basic) is this analysis written out as a script
- [Bootstrap CIs](/examples/bootstrap) goes further into confidence intervals for `S1`, `ST`, and `S2`
- [Multi-Output & Time-Series](/examples/multi-output) moves from a scalar `Y` to `(N, K)` and `(N, T, K)`
- [Non-Uniform Inputs](/examples/non-uniform-inputs) mixes uniform, Gaussian, and truncated Gaussian marginals
- [Correlated Inputs](/examples/correlated-inputs) covers what to do when your inputs are not independent
- [RS-HDMR](/examples/hdmr) analyzes arbitrary `(X, Y)` pairs you already have, with no special design
- [Save and Reload Samples](/examples/save-load) covers the full persistence workflow
- [xarray Output](/examples/xarray) exports labeled datasets with named parameters, outputs, and times
- [Configuration](/guide/configuration) covers precision, the memory budget, and the batching contract
- [API Reference](/api/) has the signatures, shape contracts, and result objects
