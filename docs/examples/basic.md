# How many model runs are enough?

The [first analysis](/guide/getting-started) showed how to compute and read
Sobol indices. This example asks a different question: **how stable are those
numbers when the sample budget changes?**

We use the Ishigami model because its exact indices are known. Your own model
will not come with an answer key, but the same convergence check still helps.

## Compare a small run with the exact answer

Save the code below as `basic.py` and run `uv run basic.py`. The package
includes the model and its exact indices, so you can focus on the estimates.

```python
import numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import (
    ANALYTICAL_S1,
    ANALYTICAL_ST,
    PROBLEM,
    evaluate,
)

design = jaxgsa.sobol.sample(
    PROBLEM, n_samples=4096, seed=42, calc_second_order=False, verbose=False
)
Y = evaluate(design.samples)
result = jaxgsa.sobol.analyze(design, Y, verbose=False)

np.set_printoptions(precision=3, suppress=True)
print("S1 estimate:", np.asarray(result.S1), "exact:", ANALYTICAL_S1)
print("ST estimate:", np.asarray(result.ST), "exact:", ANALYTICAL_ST)
```

The exact `S1` values are about `[0.314, 0.442, 0]`; the exact `ST` values
are about `[0.558, 0.442, 0.244]`. The third input's direct effect is zero,
but its total effect is about a quarter of the variation because it interacts
with the first input.

Your estimates will not be exact. Small negative values and small gaps
between indices can come from sampling noise. Treat an index near zero as an
estimate, not proof that the input has no effect.

We set `calc_second_order=False` because we only need `S1` and `ST` here.
That spends fewer model evaluations per independent base point. If you need
the interaction between a *specific pair*, enable second-order indices and
read `result.S2`.

## Watch the estimates change

One larger design can be downsampled to several smaller budgets. Evaluate the
model once, then compare each subset with the same exact answer:

```python
big = jaxgsa.sobol.sample(
    PROBLEM, n_samples=20_480, seed=42, calc_second_order=False,
    verbose=False,
)
Y_big = np.asarray(evaluate(big.samples))

print(f"{'base points':>11} {'model runs':>11} {'largest S1 error':>17} {'largest ST error':>17}")
for base_n in (64, 256, 1024, 4096):
    design_n, Y_n = big.downsample(base_n, Y_big)
    estimate = jaxgsa.sobol.analyze(design_n, Y_n, verbose=False)
    s1_error = np.max(np.abs(np.asarray(estimate.S1) - ANALYTICAL_S1))
    st_error = np.max(np.abs(np.asarray(estimate.ST) - ANALYTICAL_ST))
    print(f"{base_n:>11} {design_n.n_runs:>11} {s1_error:>17.4f} {st_error:>17.4f}")
```

`base_n` counts the independent starting points in the Sobol design. With
three inputs and `calc_second_order=False`, each base point needs five
structured rows. More base points generally improve the estimates, but a
single increase in budget can make one estimated index worse. Check the
indices that matter to your decision across several budgets.

The exact answer exists only for this teaching model. For a real model, use
[bootstrap confidence intervals](/examples/bootstrap) to describe uncertainty.
Increase the model-run budget if a confidence interval is too wide to support
your conclusion.

## What to carry into your own analysis

1. Choose input distributions that represent the uncertainty you want to
   study. Changing them changes the meaning of the indices.
2. Use `design.samples` in its original order and keep the design object for
   `analyze()`. Sobol estimators depend on the layout.
3. Check `ST` before discarding an input. A small `S1` can hide an
   interaction.
4. Ask whether the uncertainty in an index is small enough for your decision.
   A ranking may be useful before the exact variance shares settle.

See [methods](/guide/methods) to choose an analysis for other data and
[save and reload samples](/examples/save-load) when model evaluation happens
in a separate job.
