# Failed model runs

Some model runs fail. The model then returns `NaN` or an infinite value for
those rows. This page shows what `analyze` does about it and how to see which
runs failed.

## The default: stop

Every `analyze` function takes `on_invalid`. The default is `"raise"`. With
this value, `analyze` raises an error if any input or output is not finite. An
index from a partial sample answers a different question from the one you
asked. Find the failed runs first.

## Drop the failed data

With `on_invalid="drop"`, `analyze` removes the affected data and uses the
rest. What it removes depends on the design:

| Design | Unit removed |
| --- | --- |
| Given data (`rank`, `pce`, `hdmr`, `borgonovo`, `pawn`, ...) | The failed row |
| Sobol | The whole Saltelli group: one base point and all rows built from it |
| Morris | The whole trajectory |
| Kucherenko | The base point in every block it appears in |
| eFAST | Not possible. `"drop"` raises an error |

Sobol drops the whole group because a partial group breaks the A/B/AB/BA
layout. The estimator would read wrong values and raise no error.

```python
import numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

design = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=0, verbose=False)
Y = np.array(evaluate(design.samples))
Y[3] = np.nan  # one model run failed

result = jaxgsa.sobol.analyze(design, Y, on_invalid="drop", verbose=False)
print(result.invalid)
print("groups dropped:", result.invalid.n_invalid, "of", result.invalid.n_units)
print("runs to investigate:", result.invalid.bad_row_indices)
print("S1:", result.S1)
```

Output (the drop also emits a `JaxgsaWarning` that names the dropped group):

```text
InvalidReport(policy='drop', unit='Saltelli group', n_invalid=1/1024, sources=('Y',), unit_indices=[0])
groups dropped: 1 of 1024
runs to investigate: (3,)
S1: [0.32233477 0.43617576 0.00138994]
```

The exact Ishigami values are S1 = [0.314, 0.442, 0.0]. One dropped group out
of 1024 leaves the estimate close to them.

## The InvalidReport

Every result has an `invalid` field. It is an `InvalidReport`. Its main fields:

- `policy`: the policy that ran.
- `unit`: what one bad value removes (a row, a Saltelli group, ...).
- `n_units` and `n_invalid`: how many units existed and how many were affected.
- `row_indices`: every row removed, numbered as in the array you passed.
- `bad_row_indices`: the rows that held a non-finite value. These are the model
  runs to investigate.
- `sources`: which arrays held bad values, for example `"Y"`.

A warning appears when few units remain. A small sample gives a noisy index.

## Other policies

- `"propagate"`: compute anyway. The `NaN` reaches the indices.
- `"skip"`: do not check. This saves time when you already cleaned the data.
  The report always says clean, because nothing was checked.

For given-data methods, see [existing data](/guide/existing-data).
