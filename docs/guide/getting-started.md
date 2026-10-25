# Getting started

This tutorial answers a practical question: **which inputs matter to my
model's output?** You will run a Sobol analysis, read its result, and check how
much uncertainty remains.

## Install

```bash
uv add jaxgsa
```

Python 3.12 or newer is required. If you are not using uv, install with
`pip install jaxgsa`.

## Run your first analysis

Sobol analysis compares model outputs from a structured set of input values.
The Ishigami test model has three inputs and a known answer, so it lets us
check whether we have interpreted the estimates correctly.

Save this as `first_analysis.py` and run it with `uv run first_analysis.py`:

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})

design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42, verbose=False)


def model(X):
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )


Y = model(design.samples)
result = jaxgsa.sobol.analyze(design, Y, verbose=False)

print("S1:", jnp.round(result.S1, 2))
print("ST:", jnp.round(result.ST, 2))
```

The four actions are the same for your model: describe its input ranges, draw
a design, evaluate your model at `design.samples`, then pass the design and
outputs to `analyze()`. Keep the output rows in the same order as the samples.

For this seed and budget, the estimates are approximately:

```text
S1: [0.34 0.44 0.02]
ST: [0.63 0.44 0.24]
```

The columns follow the input order `x1`, `x2`, `x3`. Each index estimates a
share of output variation:

| Input | `S1`: effect on its own | `ST`: including interactions |
| --- | ---: | ---: |
| `x1` | 0.34 | 0.63 |
| `x2` | 0.44 | 0.44 |
| `x3` | 0.02 | 0.24 |

**What did we learn?** `x2` matters on its own. `x3` has almost no effect on
its own, yet matters when another input changes. Here that partner is `x1`.
Judging inputs by `S1` alone would miss `x3`. A gap between `S1` and `ST` is a
clue to interaction, though small gaps can also come from sampling noise.

## Check how certain the ranking is

These are estimates from a finite number of model runs. For example, `S1` for
`x3` is 0.02 although its exact value is zero. Use bootstrap intervals before
reporting a precise index:

```python
import jax
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict({
    "x1": (-jnp.pi, jnp.pi),
    "x2": (-jnp.pi, jnp.pi),
    "x3": (-jnp.pi, jnp.pi),
})
design = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42, verbose=False)


def model(X):
    return (
        jnp.sin(X[:, 0])
        + 7.0 * jnp.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * jnp.sin(X[:, 0])
    )


Y = model(design.samples)
boot = jaxgsa.sobol.analyze(
    design, Y, n_bootstrap=1000, key=jax.random.key(0), verbose=False
)
print("S1 lower and upper bounds:\n", boot.S1_conf)
print("ST lower and upper bounds:\n", boot.ST_conf)
```

Each confidence array has a lower row and an upper row. The lower bound of a
near-zero index can fall below zero, as it does for `S1` of `x3`. Sampling error
causes this. Read it as "no detectable effect on its own". If a range is too wide
for your decision, run more model evaluations. The [Ishigami example](/examples/basic)
shows how estimates change as the sample budget grows. Your real model has no
known answer to compare with, so intervals and convergence checks matter more.

## Check the setup before trusting the numbers

- Give each input its actual range or distribution. The indices answer a
  question about those distributions, not just about the model formula.
- Evaluate the exact rows in `design.samples`. Sobol analysis needs this
  design's row layout and metadata; arbitrary random samples will not work.
- Keep invalid runs visible. By default, `analyze()` raises when a model output
  is `NaN` or infinite. Investigate failed runs before considering
  `on_invalid="drop"`.
- Keep the inputs independent for Sobol indices. For dependent inputs, use a
  method from the [methods guide](/guide/methods).

### When a model run fails

By default, `analyze()` raises if any output is `NaN` or infinite. Find and
correct the failed model run first. For Sobol analysis, one bad row affects its
whole structured group. If you choose `on_invalid="drop"`, jaxgsa removes the
affected groups and reports how many were lost. See the
[API reference](/api/) for the other policies.

The calls print a run summary by default. We passed `verbose=False` above so
the first result is easy to see. On your own data, turn the summary on and
check the input distributions, output shape, invalid-run count, and actual
sample budget. See [configuration](/guide/configuration) for memory and
precision settings.

## Reading the verbose output

With `verbose=True`, `sample()` and `analyze()` print a short report. This is
the Sobol' output for the README example (`n_samples=4096`, `seed=42`). Timings vary by machine:

```text
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=512, requested_runs>=4096, n_runs=4096, n_expanded=4096, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=4096 runs, T=1 x K=1 output slice
    invalid: none found in 512 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.2622 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.6266
    2. x2  ST=0.44
    3. x3  ST=0.2423
```

The `sample()` line has these fields:

- `D`: the number of input parameters.
- `mode`: `second-order` if the design also supports `S2`, or `first/total-order` if you set `calc_second_order=False`.
- `base_n`: the number of base points. It is a power of 2. The full Saltelli design has `base_n * (2D + 2)` rows with second order, or `base_n * (D + 2)` without.
- `requested_runs>=`: the run count you asked for with `n_samples`. jaxgsa picks the smallest `base_n` that meets it, so `n_runs` is never below this value.
- `n_runs`: the number of unique rows. Evaluate your model on exactly these rows.
- `n_expanded`: the row count of the full Saltelli layout before duplicate rows are removed. It is never below `n_runs`.
- `duplicates_removed`: `n_expanded - n_runs`, with its share of `n_expanded`. Duplicates occur, for example, with discrete inputs. You run each unique row only once.
- `scramble`: whether the Sobol' sequence is randomised. Keep it `True`.

The `analyze()` report has these parts:

- `problem`: the input count and names, the marginal distribution kinds, and the dependence kind.
- `output`: `N` is the number of output rows used. `T x K` is the output layout, and each (T, K) pair is one output slice.
- `invalid`: the result of the NaN and infinity check. The unit is `Saltelli groups` for Sobol', because one bad row spoils its whole group. The policy is your `on_invalid` setting.
- `timing`: the time spent in the estimators. The first call includes JIT compilation. `slice_chunk_size` is how many output slices run together in one batch. `resolved from the memory budget` means jaxgsa chose it; `user-set` means you passed it. `estimator` is the formula pair in use, `saltelli-jansen` by default.
- `results`: the top five inputs ranked by the headline index (`ST` for Sobol'). With a bootstrap, each line also shows the confidence interval. With several output slices, the values are the mean over slices.

## Where to go next

- [Concepts](/guide/concepts) explains what the indices mean.
- [Choosing a method](/guide/methods) helps you start with the right analysis.
- [Ishigami example](/examples/basic) explores sampling error and convergence.
- [Multi-output and time series](/examples/multi-output) extends the workflow
  beyond one scalar output.
- [API reference](/api/) lists the full signatures and result fields.

If you use jaxgsa in research, cite the version you ran and the paper for
your sensitivity method (see the [Sobol' references](/api/sobol#references)
for this tutorial's estimators). To cite jaxgsa in general, use the
concept DOI
[10.5281/zenodo.22099117](https://doi.org/10.5281/zenodo.22099117).
It always resolves to the latest version. For reproducibility, cite the
version DOI instead. Find it on the Zenodo record of the release you ran.
See [`CITATION.cff`](https://github.com/DanielePessina/jaxgsa/blob/master/CITATION.cff)
and the [methods guide](/guide/methods).
