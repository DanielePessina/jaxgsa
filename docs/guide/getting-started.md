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

boot = jaxgsa.sobol.analyze(
    design, Y, n_bootstrap=1000, key=jax.random.key(0), verbose=False
)
print("S1 lower and upper bounds:\n", boot.S1_conf)
print("ST lower and upper bounds:\n", boot.ST_conf)
```

Each confidence array has a lower row and an upper row. If a range is too wide
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

## Where to go next

- [Concepts](/guide/concepts) explains what the indices mean.
- [Choosing a method](/guide/methods) helps you start with the right analysis.
- [Ishigami example](/examples/basic) explores sampling error and convergence.
- [Multi-output and time series](/examples/multi-output) extends the workflow
  beyond one scalar output.
- [API reference](/api/) lists the full signatures and result fields.

If you use jaxgsa in research, cite the version you ran and the paper for
your sensitivity method. For v1.0.2, use
[10.5281/zenodo.23057233](https://doi.org/10.5281/zenodo.23057233)
(the archived v1.0.0 record is
[10.5281/zenodo.23054912](https://doi.org/10.5281/zenodo.23054912)).
See [`CITATION.cff`](https://github.com/DanielePessina/jaxgsa/blob/master/CITATION.cff)
and the [methods guide](/guide/methods).
