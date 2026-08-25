---
name: jaxgsa
description: Use when writing, reviewing, or documenting code that uses jaxgsa for global sensitivity analysis in JAX. Covers defining a Problem, drawing a Sobol, Morris, eFAST, or Kucherenko design, drawing plain Monte Carlo samples, and running any of the thirteen analysis methods (sobol, morris, efast, kucherenko, pce, hdmr, shapley, dgsm, hsic, pawn, borgonovo, optimal_transport, vkoga), including correlated and categorical inputs, confidence intervals, batching, and xarray export.
---

# jaxgsa

Global sensitivity analysis in JAX. Every method lives in its own namespace, and
that namespace holds the sampling and analysis commands for the method.

```python
import jaxgsa

jaxgsa.Problem, jaxgsa.UniformSpec, jaxgsa.GaussianSpec, jaxgsa.CategoricalSpec
jaxgsa.sampling            # plain draws and correlation helpers
jaxgsa.sobol, jaxgsa.morris, jaxgsa.efast, jaxgsa.kucherenko   # build own design
jaxgsa.pce, jaxgsa.hdmr, jaxgsa.shapley, jaxgsa.dgsm, jaxgsa.hsic,
jaxgsa.pawn, jaxgsa.borgonovo, jaxgsa.optimal_transport, jaxgsa.vkoga  # given data
jaxgsa.benchmarks, jaxgsa.config
```

This page covers everything shared by all thirteen methods. **The per-method
call, its keywords, a worked example and its caveats live in four reference
files. Read the one for the family you need before writing a method call.**

| File | Methods |
| --- | --- |
| `reference/variance-based.md` | `sobol`, `efast`, `pce`, `hdmr`, `shapley` |
| `reference/dependent-inputs.md` | `kucherenko`, `vkoga`, and declaring a correlation |
| `reference/screening.md` | `morris`, `dgsm` |
| `reference/moment-independent.md` | `hsic`, `pawn`, `borgonovo`, `optimal_transport` |

## Quick start

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(design.samples)          # (n_runs, D) in, (N,) or (N, K) or (N, T, K) out
result = jaxgsa.sobol.analyze(design, Y)

result.S1, result.ST, result.S2
```

`verbose=True` is the default on all thirteen `analyze()` functions and on the
four design samplers. The printed block is a receipt. It echoes the marginals,
the inferred `(N, T, K)` shape, the invalid-run count, and the top-k ranking, so
a transposed `Y` shows up before you plot anything. Read it on the first run and
pass `verbose=False` in loops and tests. `jaxgsa.sampling.monte_carlo` has no
`verbose` keyword.

## Problem

The direct constructor takes uniform marginals only.

```python
jaxgsa.Problem(
    names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    output_names: tuple[str, ...] | None = None,
    correlation=None,
    correlation_type: Literal["latent", "spearman"] = "latent",
)
```

`from_dict` takes any mix of marginals, in dict insertion order, which is the
model's column order.

```python
problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),                                        # uniform shorthand
        "x2": jaxgsa.GaussianSpec(mean=0.0, variance=4.0, high=3.0),
        "x3": {"dist": "categorical", "probs": [0.5, 0.5], "labels": ["off", "on"]},
    },
    output_names=("y",),
    truncate_gaussians=None,     # e.g. 1e-4 to bound every Gaussian at its own quantiles
    correlation=None,
    correlation_type="latent",
)
```

Each value may be a `(low, high)` tuple, a spec dataclass (`UniformSpec`,
`GaussianSpec`, `CategoricalSpec`), or the matching plain dict
(`UniformInputSpec`, `GaussianInputSpec`, `CategoricalInputSpec`). The dict form
is a TypedDict, so a problem written that way is JSON-expressible.

Properties: `names`, `bounds`, `output_names`, `num_vars`, `input_specs`,
`correlation`, `has_correlated_inputs`, `has_non_uniform_inputs`,
`has_categorical_inputs`, `categorical_labels`.

Keep in mind:

- `problem.bounds` is `None` as soon as any marginal is not uniform. Read
  `input_specs` and branch with `isinstance` instead.
- Names must be unique strings. Both rules are enforced at construction, because
  a bad name only fails much later and much less clearly.
- Set `output_names` whenever `T` and `K` could be confused. Its length must
  equal the trailing axis of `Y`, and the mismatch is caught before array work.
- `Problem` is frozen, so `problem.with_correlation(R)` returns a copy. Pass
  `None` to drop a matrix.

Categorical parameters carry integer level codes `0 .. L-1` as floats, never
physical values. `probs` needs at least two positive entries summing to 1 within
`1e-3`, and a spec that misses is rejected rather than rescaled. `labels` is
reporting metadata only. Four methods accept categorical inputs: `sobol`,
`borgonovo`, `optimal_transport` and `pawn`. Everything else raises a
`ValueError`, because a derivative, a level spacing or a Fourier sweep along an
unordered code has no meaning.

For declaring a correlation, see `reference/dependent-inputs.md`.

## Sampling

```python
# Design builders. The method must be able to run your model at its points.
jaxgsa.sobol.sample(problem, n_samples, *, base_n=None, calc_second_order=True,
                    scramble=True, seed=None, verbose=True)      -> SobolSamples
jaxgsa.morris.sample(problem, n_trajectories, *, num_levels=4,
                     method="trajectory", scramble=True, seed=None,
                     truncation_quantile=1e-4, verbose=True)     -> MorrisSamples
jaxgsa.efast.sample(problem, n_per_curve, *, M=4, seed=None,
                    verbose=True)                                -> EFASTSamples
jaxgsa.kucherenko.sample(problem, n_samples, *, scramble=True, seed=None,
                         verbose=True)                           -> KucherenkoSamples

# Plain draws and correlation helpers, for the nine given-data methods.
jaxgsa.sampling.monte_carlo(problem, n, *, seed=None)     -> np.ndarray  (n, D)
jaxgsa.sampling.correlate(X, problem, *, seed=None)       -> np.ndarray  (N, D)
jaxgsa.sampling.fit_correlation(problem, X)               -> np.ndarray  (D, D)
jaxgsa.sampling.correlation_from_covariance(cov)          -> np.ndarray  (D, D)
```

`seed` is keyword-only everywhere. It accepts an int, an existing
`np.random.Generator`, or `None` for fresh OS entropy.

Keep in mind:

- **`sobol.sample` takes the total budget, not the base count.**
  `sample(problem, 8192)` on `D=3` gives 8192 model runs from `base_n=1024`, not
  8192 times 8. `efast.sample` (per curve) and `morris.sample` (per trajectory)
  do multiply. Check `samples.samples.shape[0]` if you are not sure.
- Duplicate rows are removed, so `n_runs` is below `n_expanded`. Evaluate
  exactly the rows in `samples.samples`, in order, and pass the outputs straight
  to `analyze`, which rebuilds the blocks itself.
- The seed feeds the Owen scramble, not the sequence. Keep `scramble=True`.
  Scrambling is what removes the Sobol sequence's structural duplicates, and
  without it a Saltelli design loses real blocks to deduplication.
  `kucherenko.sample` raises on `scramble=False, seed=...` rather than accept a
  seed that would do nothing. `efast.sample` has no `scramble`, because its
  randomness is the phase shift of each search curve.
- `monte_carlo` honors `problem.correlation` through the NORTA construction and
  returns level codes for categorical columns. An independent problem keeps the
  plain uniform path bit for bit, so old seeds reproduce old samples.

### Design objects

All four carry `save(path)` and `load(path)` as one compressed NPZ file holding
the sample matrix and a JSON metadata blob. `SobolSamples` and `MorrisSamples`
also have `downsample(base_n, Y=None)`, which returns a prefix-nested smaller
design and the matching subset of outputs, so you can plot convergence against
sample size from one evaluated design.

`SobolSamples` fields: `samples`, `n_runs`, `n_expanded`, `expanded_to_unique`,
`base_n`, `n_params`, `calc_second_order`, `problem`, `unit`,
`expanded_to_unit`.

```python
# Reuse one design under different input ranges. unit holds the quasi-random
# points before any distribution is applied, and transform applies a new set.
narrow = samples.transform({"x1": {"low": 0.0, "high": 1.0}, ...})
wide   = samples.transform({"x1": {"low": -1.0, "high": 2.0}, ...})
```

`theta` has type `jaxgsa.Theta`, a mapping keyed by parameter name and then by
the field names of that parameter's distribution. `transform` is written in JAX
and is differentiable with respect to `theta`. It raises for a problem with
categorical parameters, because a categorical inverse CDF is a step function.

`SobolSamples.to_morris()` reinterprets an evaluated Saltelli design as a radial
Morris design, for free screening measures. See `reference/screening.md`.

## Shape contract

| Output | Shape |
| --- | --- |
| Scalar | `(N,)` |
| Multiple outputs | `(N, K)` |
| Time series with multiple outputs | `(N, T, K)` |

The sample axis is first and the output axis is last. jaxgsa never infers,
transposes, or inserts axes. **A 2-D `Y` is always `(N, K)`.** A single
time-varying output must be written `(N, T, 1)`. The index arrays follow:
`(D,)` for a scalar, `(K, D)` for multi-output, `(T, K, D)` for a time series.
Check that shape once, on the first run, and a transposed array cannot survive
to your plots.

## Cross-cutting keywords

Every `analyze()` takes `verbose` and `on_invalid`. Eleven of the thirteen take
the bootstrap block. Which batching keyword exists depends on the axis the
method batches over.

### Failed model runs

`on_invalid: "raise" | "propagate" | "drop"`, default `"raise"`. The default
refuses, because an index computed from part of a sample is a different quantity
from the one you asked for, and `analyze()` is cheap to run again.

`"drop"` removes the whole block a bad value sits in: a Saltelli group, a Morris
trajectory, a Kucherenko base point, or one row for a given-data method.
`jaxgsa.efast.analyze` accepts only `"raise"` and `"propagate"`, because its
ordered Fourier sweep changes meaning if a point is removed.

Every result carries `result.invalid`.

```python
result.invalid.n_invalid        # blocks that held a bad value
result.invalid.unit_indices     # which blocks
result.invalid.bad_row_indices  # the rows that actually failed
result.invalid.row_indices      # every row those blocks cover, which "drop" removes
result.invalid.sources          # bad values in X, in Y, or both
result.invalid.unit             # ROW, SALTELLI_GROUP, TRAJECTORY, BASE_POINT, CURVE
```

Both index arrays refer to the array as you passed it, so they name model runs
you can find. For a block design they differ by a large factor: one failed run
inside an eFAST search curve gives one entry in `bad_row_indices` and 257 in
`row_indices`.

### Confidence intervals

```python
result = jaxgsa.sobol.analyze(
    samples, Y,
    n_bootstrap=1000,          # default 0, meaning no interval
    conf_level=0.95,
    ci_method="quantile",      # or "gaussian"
    key=jax.random.key(0),     # REQUIRED once n_bootstrap > 0, raises without it
    keep_replicates=True,      # keeps result.ci.replicates
)
result.S1_conf   # (2, ...), the leading axis holds [lower, upper]
result.ci.level, result.ci.method, result.ci.n_bootstrap, result.ci.replicates
```

`"quantile"` reads the endpoints off the empirical bootstrap distribution.
`"gaussian"` centres them on the point estimate at plus or minus z times the
standard deviation of the draws, which is smoother at small `n_bootstrap` but
assumes the draws are normal.

`result.ci` is `None` when no bootstrap ran. `keep_replicates` defaults to
`False` because the draws are large: 1000 resamples of a `(T=100, K=5, D=20)`
index array is 80 MB, more than the rest of the result put together.

`efast` and `hsic` have no bootstrap, by decision. eFAST has nothing to resample
inside one search curve, since removing a point changes what the estimator
computes rather than shrinking the sample. HSIC reports permutation `p_values`
instead, because a row bootstrap duplicates rows onto the kernel diagonal where
the kernel is exactly 1, which biases the resampled index upward by
construction.

The four surrogate-backed methods (`pce`, `hdmr`, `vkoga`, `shapley`) refit
their surrogate on every replicate, so an interval there costs an order of
magnitude more than a row resample on a direct estimator.

### Batching

| Keyword | Batches over | On |
| --- | --- | --- |
| `batch_size` | sample rows | `dgsm`, `hdmr`, `pce`, `vkoga`, surrogate `predict` |
| `slice_chunk_size` | `(T, K)` output slices | `sobol`, `hdmr`, `pawn`, `borgonovo`, `efast`, `optimal_transport` |
| `resample_chunk_size` | bootstrap replicates | `morris` |

The value is clamped to the axis it sizes and never selects a different
algorithm, so the answer does not depend on it beyond float summation order.
`None` derives a width from the memory budget, and an explicit value always
wins.

`hsic`, `shapley`, and `kucherenko` take none of the three.
`jaxgsa.hsic.analyze(..., batch_size=...)` raises `TypeError`, because HSIC
holds `2D+1` matrices of shape `(N, N)` at once and only `N` bounds that.

### indices(), the traceable core

Eleven namespaces expose `indices(...)`, which returns raw arrays, checks
nothing, builds no result, and is safe under `jax.jit`, `jax.vmap`, and
`jax.grad`. `kucherenko` and `vkoga` do not have one, because Kucherenko is host
NumPy from end to end and VKOGA's index stage is a host quasi-Monte-Carlo loop.
Pair `sobol.indices` with `SobolSamples.transform` to differentiate an index
with respect to the input distribution parameters.

### Warnings

```python
import warnings
from jaxgsa import JaxgsaWarning

warnings.filterwarnings("error", category=JaxgsaWarning)   # good default for CI
```

Every jaxgsa warning is a `JaxgsaWarning`, a subclass of `UserWarning`. jaxgsa
warns when a run is degraded but still valid: a surrogate that does not fit, a
repaired correlation matrix, a zero-variance output slice, a bound that excludes
nothing, a design thinned by deduplication. Each of those produces numbers that
look fine and mean less than they appear to, so escalate to `"error"` in a
pipeline. The reference files say which warning each method raises and what to
do about it.

### Configuration

```python
import jax
jax.config.update("jax_enable_x64", True)     # BEFORE any analysis

jaxgsa.config.enable_compilation_cache("~/.cache/jaxgsa-jax")
jaxgsa.config.set_memory_budget(512)          # MiB by default
jaxgsa.config.set_memory_budget(1.5, unit="gb")
jaxgsa.config.get_memory_budget()
```

JAX computes in 32-bit by default and downcasts a `float64` `Y` you pass it.
jaxgsa warns when it sees this. HSIC and VKOGA need float64 outright, for the
reasons in their sections. `set_memory_budget` reads megabytes unless `unit=`
says otherwise, and an old byte-count call raises rather than silently meaning
512 TB.

## Method capabilities

| Method | Reports | Own design | Correlated | Categorical | Bootstrap CI |
|---|---|:--:|:--:|:--:|---|
| `borgonovo` | delta, S1 | no | yes | yes | `n_bootstrap` |
| `dgsm` | bounds on ST | no | no | no | `n_bootstrap` |
| `efast` | S1, ST | yes | no | no | none |
| `hdmr` | Sa, Sb, S per term, surrogate | no | yes | no | `n_bootstrap` |
| `hsic` | dependence measure | no | yes | no | none |
| `kucherenko` | S1, ST under dependence | yes | yes | no | `n_bootstrap` |
| `morris` | mu\*, sigma | yes | no | no | `n_bootstrap` |
| `optimal_transport` | W2 squared index, advective and diffusive | no | yes | yes | `n_bootstrap` |
| `pawn` | KS distance | no | yes | yes | `n_bootstrap` |
| `pce` | S1, S2, ST, surrogate | no | no | no | `n_bootstrap` |
| `shapley` | allocation summing to 1 | no | pce no, hdmr yes | no | `n_bootstrap` |
| `sobol` | S1, S2, ST | yes | no | yes | `n_bootstrap` |
| `vkoga` | S_TC, S_TU, S_U, S_C, S_IU, surrogate | no | yes | no | `n_bootstrap` |

A "no" in the Correlated or Categorical column is a refusal with a `ValueError`
that names the parameters and the alternatives, never a silent approximation.

Two of the "yes" answers carry conditions. `hdmr` accepts a correlated problem
but its `ST` is then not a total-effect index, and `shapley` accepts one only on
`backend="hdmr"`. Both are in `reference/dependent-inputs.md`.

## Choosing a method

- Can run the model, independent inputs, want the reference answer: `sobol`.
- Expensive model with many parameters: screen with `morris` at r(D+1) runs, or
  with `dgsm` if the model is JAX-differentiable, fix the inert parameters, then
  spend the rest of the budget on `sobol` for the survivors.
- Only existing (X, Y) data: any given-data method.
- Correlated inputs: `kucherenko` if you can still run the model, `vkoga` or
  `hdmr` if you cannot, or a moment-independent method.
- Skewed or heavy-tailed output, where variance is the wrong summary: `pawn`,
  `borgonovo`, `optimal_transport`.
- Want to know whether a parameter shifts the output or reshapes it:
  `optimal_transport`.
- Want one fair number per parameter that sums to 1: Shapley effects.
- Want a reusable surrogate too: `pce`, `hdmr`, `vkoga`.

Then read the reference file for the family you picked.

| Method | What the number is | File |
| --- | --- | --- |
| `sobol` | Exact variance fractions from a Saltelli design, with `S2` | `variance-based.md` |
| `efast` | The same `S1` and `ST` from Fourier search curves | `variance-based.md` |
| `pce` | Variance fractions read off a fitted polynomial | `variance-based.md` |
| `hdmr` | Per-term variance shares from a B-spline fit, split structural and correlative | `variance-based.md` |
| `shapley` | One allocation per parameter, summing to 1, from a PCE or HDMR fit | `variance-based.md` |
| `kucherenko` | Conditional-variance `S1` and `ST` under a declared copula | `dependent-inputs.md` |
| `vkoga` | The same two from a kernel surrogate, plus the correlated and uncorrelated split | `dependent-inputs.md` |
| `morris` | Mean absolute elementary effect, a screening rank | `screening.md` |
| `dgsm` | Derivative moments and bounds on `ST` | `screening.md` |
| `hsic` | Kernel dependence, with a permutation p-value | `moment-independent.md` |
| `pawn` | KS distance between conditional and unconditional CDFs | `moment-independent.md` |
| `borgonovo` | Half L1 distance between conditional and unconditional densities | `moment-independent.md` |
| `optimal_transport` | Normalized squared Wasserstein distance, split mean-shift and shape | `moment-independent.md` |

## Export

Every result has `to_dataset(time_coords=None)`, which returns a labeled
`xarray.Dataset` on `param`, `output` and `time` dimensions. Point estimates
come first, then the `*_lower` and `*_upper` halves of every confidence
interval, so you select a bound by name instead of by integer index.

```python
ds = result.to_dataset(time_coords=np.linspace(0, 10, T))
ds["S1"].sel(param="x1")
```

`output_names` on the `Problem` names the `output` coordinate. Scalar provenance
such as `SobolResult.estimator` or `HSICResult.bandwidth` goes into
`ds.attrs` rather than into a variable.

## Benchmarks

`jaxgsa.benchmarks` holds analytical test functions whose Sobol indices are
known in closed form: `ishigami`, `sobol_g`, `linear`, `gaussian_linear` and
`oakley_ohagan`. Each provides `PROBLEM`, a JAX `evaluate(X)`, the precomputed
`ANALYTICAL_S1`, `ANALYTICAL_ST` and `ANALYTICAL_S2`, and an
`analytical_indices(...)` for non-default parameters.

```python
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate, ANALYTICAL_S1
```

Use them to check an estimator, or to measure how fast it converges. Every
example in the reference files runs against one of them, so the printed values
can be checked against the truth.
