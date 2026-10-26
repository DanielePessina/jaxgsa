# RS-HDMR

This page turns `(X, Y)` pairs you already have into sensitivity indices, with
no special sampling design. You finish with per-term variance shares, a fitted
surrogate that predicts at new inputs for almost nothing, and the fit error
that says whether to believe any of it.

## What the method fits

Random-sampling high-dimensional model representation writes the model as a
sum of pieces:

$$
f(x) \approx f_0 + \sum_i f_i(x_i) + \sum_{i<j} f_{ij}(x_i, x_j)
$$

$f_0$ is the mean output. $f_i$ is a **component function**: a curve in one
input that gives the average shift in output as that input moves, with every
other input averaged out. $f_{ij}$ is a surface in two inputs, holding what
the pair does together that neither does alone. `order` cuts the sum off,
at pairs by default.

jaxgsa builds each component function from B-splines and fits all of them
together by backfitting, then runs an F-test to drop terms that do not earn
their parameters. The sensitivity index of a term is the share of output
variance that term accounts for. Fit and indices are the same computation.
That is the appeal. Whatever samples you have, you get both.

Everything below rests on those component functions being close to the real
ones. They are B-splines with `n_intervals` intervals per input. Pick `n_intervals` too low and
the curves are too stiff to follow the response, the indices shrink, and
nothing warns you. That is the subject of the second section.

## Import style

```python
from jaxgsa import hdmr
```

## Indices from random samples

Uniform random inputs stand in for data you already have. `order=2` asks
for pair terms as well as single-input terms. `n_intervals=4` sets four B-spline
intervals per input, up from the default 2, for reasons the next section
makes concrete.

```python
import jax
import jax.numpy as jnp
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate
from jaxgsa import hdmr

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

result = hdmr.analyze(PROBLEM, X, Y, order=2, n_intervals=4)

print("terms:", result.terms)
print("S1:", result.S1)
print("ST:", result.ST)
print("Sa:", result.Sa)
print("Sb:", result.Sb)
print("rmse:", result.rmse)
print("std(Y):", jnp.std(Y))
```

```text
jaxgsa.hdmr.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.806 s
    order: 2
    slice_chunk_size: 1 (resolved from the memory budget)
    batch_size: 2000 (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5543
    2. x2  ST=0.4441
    3. x3  ST=0.2375
terms: ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')
S1: [0.32048696 0.4472546  0.00075338]
ST: [0.5543085  0.44405735 0.23746695]
Sa: [3.2048696e-01 4.4725460e-01 7.5337599e-04 2.2831899e-05 2.3683792e-01
 1.6588895e-04]
Sb: [-0.00298596 -0.00319192  0.00017364  0.00010799 -0.00016176 -0.00030231]
rmse: 0.048114467
std(Y): 3.7769964
```

The summary block above `terms:` prints by default. Pass
`verbose=False` to silence it. It restates the problem it was handed, which
catches a wrong `bounds` array or a misread output shape before you read an
index.

Read `rmse` first. It is 0.0481 against a `std(Y)` of 3.777, so the surrogate
misses by 1.3% of the output spread. The indices describe the surrogate, not
the model, so a bad `rmse` makes every other field on the result a
description of the wrong function.

Then the per-term arrays. `Sa` and `Sb` have one entry per term and `terms`
names them in order. `Sa[4] = 0.2368` is the x1/x3 pair, and it dwarfs the
other two pairs at 0.00002 and 0.0002. That matches Ishigami, whose only real
interaction is x1 with x3. `S1` and `ST` collapse this to one number per
input. `S1` is the input's own term, and `ST` sums every term the input
appears in. Where `ST` runs far above `S1`, as x3's 0.237 does over its 0.0008, the
input matters only through a partner, and the term-level `Sa` says which one.

`Sb` is the correlative share. It is near zero here because the inputs are
independent, and it becomes the real diagnostic when they are not. See
[Correlated Inputs](/examples/correlated-inputs).

## `n_intervals` is the diagnostic knob

`n_intervals` is the number of B-spline intervals per input, and it sets how much each
component function can bend. The default is 2. Ishigami's $\sin^2(x_2)$ over
$[-\pi, \pi]$ needs more than that. Here is the same fit at four values:

```python
import numpy as np

sigma = float(jnp.std(Y))
for m in (2, 4, 6, 8):
    r = hdmr.analyze(PROBLEM, X, Y, order=2, n_intervals=m, verbose=False)
    print(f"n_intervals={m}  rmse={float(r.rmse):.4f}  rmse/std={float(r.rmse)/sigma:6.1%}"
          f"  S1={np.round(r.S1, 4)}")
```

```text
n_intervals=2  rmse=1.1900  rmse/std= 31.5%  S1=[0.30699998 0.35369998 0.0006    ]
n_intervals=4  rmse=0.0481  rmse/std=  1.3%  S1=[0.3205 0.4473 0.0008]
n_intervals=6  rmse=0.1132  rmse/std=  3.0%  S1=[0.3199 0.4461 0.0009]
n_intervals=8  rmse=0.0264  rmse/std=  0.7%  S1=[0.3209     0.44739997 0.0012    ]
```

Ishigami's analytical `S1` is `[0.3139, 0.4424, 0]`. At the default `n_intervals=2` the
fit reports x2's first-order index as 0.354, low by 20%, and `rmse` sits at a
third of the output spread. Nothing errors. Nothing warns. You get a plausible
table with x2 understated, and the only clue is `rmse`.

At `n_intervals=4` the error drops to 1.3% and x2 lands at 0.447. Past that the
error does not fall steadily: it rises to 3.0% at 6 and falls to 0.7% at 8, while
`S1` barely moves. So `n_intervals` is not "higher is better". Sweep it, watch
`rmse`, and stop when the indices settle. On this problem `n_intervals=4` is enough.

My rule for `rmse / std(Y)`: under 10% the indices are reportable, 10% to 25%
treat the ranking as provisional, above 25% the numbers describe a surrogate
nobody should quote. RS-HDMR reaches those bands more slowly than a polynomial
does on smooth models, which is the honest cost of a local basis.

## Term selection

`result.select` counts how many output slices kept each term. With a scalar
`Y` there is one slice, so it reads as a flag:

```text
select   (term) float32 24B 1.0 1.0 0.0 0.0 1.0 0.0
```

Reading it against `terms = ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')`:
the fit dropped the x3 solo term and the x1/x2 and x2/x3 pairs, all of which
are truly absent from Ishigami. The F-test is a threshold, though, and it can
let noise through near zero. Do not treat a selected term with a tiny share as evidence of an
interaction.

## Use the emulator

The fit leaves a surrogate you can call. Predicting at inputs it already saw
is the cheapest check on it:

```python
Y_pred = result.predict(X[:5])
print("Prediction shape:", Y_pred.shape)
print("Absolute residuals:", jnp.abs(Y[:5] - Y_pred))
```

```text
Prediction shape: (5,)
Absolute residuals: [0.03567696 0.01585853 0.03184509 0.00614166 0.01206064]
```

Those residuals run from 0.006 to 0.036 against a `std(Y)` of 3.777, consistent
with the fitted `rmse` of 0.048. The emulator is on the same scale as the `Y` you
supplied, so a residual you can read against the output range is the whole
check. A gap that stays stubborn as you raise `n_intervals` is a structural miss instead,
and the fix is a different model or more rows. Order 3 is not available.

## When RS-HDMR beats plain Sobol

The classical Sobol estimator needs the Saltelli column-swap design, and
jaxgsa enforces that in the signature: `jaxgsa.sobol.analyze(samples,
Y)` takes a `SobolSamples` object, not an `(X, Y)` pair. There is no way to
hand it 2000 rows of existing runs. If the model has already been run, or was
run by somebody else, or costs a week per evaluation, Sobol is not an option
and RS-HDMR is.

Three more cases. Correlated inputs, where `problem.correlation` is declared
and the ANCOVA split into `Sa` and `Sb` is the point. Multi-output and
time-series `Y`, where one fit covers every slice. And a surrogate as the
deliverable, when the emulator matters as much as the indices.

Against [PCE](/examples/pce), the other given-data surrogate, the trade is
local versus global. B-splines follow kinks and plateaus that a global
polynomial smooths over. On a smooth response PCE often reaches a low error
with fewer terms. On Ishigami the two are close: PCE gives `cv_rmse = 0.074` at
`order=8`, and RS-HDMR gives an in-sample `rmse = 0.048` at `n_intervals=4`.
Fit both when you can. They
are cheap and they disagree informatively.

## Field reference

`result.S1` is the first-order share, the solo term's contribution. `result.ST`
sums every term containing the parameter. `result.Sa` is the structural
variance share per term and `result.Sb` the correlative one. `result.S` is
their sum, and `result.S.sum()` is 0.999 here, close to 1 as the theory
requires. `result.terms` labels the columns of all three. `result.select` is
the F-test flag per term. `result.rmse` is the fit error you check first.

`result.to_dataset()` puts all of it in an `xarray.Dataset` with `term`,
`param`, `param_i` and `param_j` coordinates, plus `time` and `output` where
they apply. See [xarray Labeled Output](/examples/xarray).

## Practical caveats

`hdmr.analyze()` accepts `(N,)`, `(N, K)` and `(N, T, K)` outputs under the
shape rules in [Multi-Output & Time-Series](/examples/multi-output).

`order` must be 1 or 2 (`order=3` raises), and it is clamped with a warning when
`D < order`. Fewer than 300 rows gives a warning, because the
backfitting solve degenerates quietly below that. The call raises
`ValueError` only below `(n_intervals + 3) ** min(order, D) + 1` rows, the
least the fit can use.

`slice_chunk_size` caps how many output slices are fitted at once and
`batch_size` sizes row blocks. Both default to `None`, which lets the memory
budget pick. Neither changes the answer.

Under a declared correlation, `ST` becomes the SCSA total from Li et al.
(2010) rather than a Sobol total-order index. It can go negative and it is not
bounded in `[0, 1]`, so it cannot tell you a parameter is safe to fix.
`hdmr.analyze` warns about this when it applies. For a conditional-variance
total under dependence use [Kucherenko](/examples/kucherenko) or
[VKOGA](/examples/vkoga).

## See also

- [PCE](/examples/pce) for the polynomial given-data surrogate.
- [Shapley Effects](/examples/shapley) for a fair allocation read off this
  fit, including `include_correlative=True` under dependence.
- [Methods](/guide/methods) for the difference between Sobol and HDMR.
- [xarray Labeled Output](/examples/xarray) for the labeled export.
- [Screen first, then quantify](/examples/advanced-workflow) for cutting 20 inputs to 4 with Morris,
  then spending the Sobol budget on the survivors.
