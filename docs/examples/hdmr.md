# RS-HDMR Example

By the end of this page you will have sensitivity indices computed from
`(X, Y)` pairs you already had, with no special sampling design, plus a fitted
surrogate that predicts the model output at new inputs for almost no cost.

Random sampling high-dimensional model representation (RS-HDMR) writes the
model as a sum of terms. There is one term per input on its own, and with
`maxorder=2` one more term per input pair. Fitting those terms by regression
gives both the surrogate and the indices: the share of output variance a term
accounts for is that term's sensitivity index. Because it is a regression, it
accepts whatever samples you have.

## Import style

The HDMR module lives at `jaxgsa.hdmr`:

```python
from jaxgsa import hdmr
```

## Sensitivity analysis from random samples

This example draws uniform random inputs to stand in for data you already have.
`maxorder=2` asks for pair terms as well as single-input terms.
`slice_chunk_size` caps how many output slices are fitted at once, which bounds
peak memory on large outputs.

```python
import jax
import jax.numpy as jnp
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate
from jaxgsa import hdmr

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

result = hdmr.analyze(
    PROBLEM,
    X,
    Y,
    maxorder=2,
    slice_chunk_size=256,
)

print("S1:", result.S1)
print("ST:", result.ST)
print("Terms:", result.terms)
print("Sa:", result.Sa)
print("Sb:", result.Sb)
print("RMSE:", result.rmse)
```

Start with `result.rmse`, which is the fit error of the surrogate against the
training outputs. The indices describe the surrogate, not the model, so a poor
fit means every other number on the result describes the wrong function. Once
the fit holds up, compare `S1` and `ST` per input. Where `ST` is much larger
than `S1`, the input works through interactions, and `result.Sa` says with
which partner: it carries one entry per term, and `result.terms` names those
terms.

## Use the emulator

The fit leaves behind a surrogate you can call directly. Predicting at inputs
the fit already saw is the cheapest sanity check on it.

```python
Y_pred = result.predict(X[:5])
print("Prediction shape:", Y_pred.shape)
print("Absolute residuals:", jnp.abs(Y[:5] - Y_pred))
```

Compare the printed residuals against the spread of `Y` itself. Residuals that
are a small part of the output range mean the expansion captured the response.
A leftover gap points at effects the chosen `maxorder` and basis cannot
represent, and raising `maxorder` is the way to test that.

HDMR fits the surrogate on the outputs you supply. The stored emulator and
`result.rmse` are on that same scale.

## What to look at

- `result.S1` is the structural first-order contribution extracted from
  `result.Sa`.
- `result.ST` is the total contribution per parameter after summing all terms
  that involve that parameter.
- `result.Sa` is the structural variance fraction per term, and `result.Sb` is
  the correlative one. `Sb` is near zero for independent inputs, as in this
  example, and becomes the correlation diagnostic when the inputs are
  dependent. See [Correlated Inputs](/examples/correlated-inputs).
- `result.terms` tells you which columns in `Sa`, `Sb`, and `S` correspond to
  first-order and interaction terms.
- `result.rmse` helps you decide whether the fitted surrogate is accurate enough
  for downstream interpretation.

## Practical caveats

- `hdmr.analyze()` accepts `(N,)`, `(N, K)`, and `(N, T, K)` outputs, so the
  same shape rules from [Multi-Output & Time-Series](/examples/multi-output)
  still apply.
- HDMR does not use a structured Saltelli design; if you want exact Sobol
  estimators on independent inputs, start from [Basic Example](/examples/basic)
  instead.
- If you want labeled `term`, `param`, `time`, and `output` coordinates, call
  `result.to_dataset()` and continue with
  [xarray Labeled Output](/examples/xarray).

## See also

- [Methods](/guide/methods) for the conceptual difference between Sobol and
  HDMR.
- [xarray Labeled Output](/examples/xarray) for exporting `Sa`, `Sb`, `S`, and
  `ST` to a labeled dataset.
- [Advanced Workflow](/examples/advanced-workflow) for a custom time-series
  model that runs Sobol and HDMR side by side.
