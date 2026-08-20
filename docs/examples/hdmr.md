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
the pair does together that neither does alone. `maxorder` cuts the sum off,
at pairs by default.

jaxgsa builds each component function from B-splines and fits all of them
together by backfitting, then runs an F-test to drop terms that do not earn
their parameters. The sensitivity index of a term is the share of output
variance that term accounts for. Fit and indices are the same computation.
That is the appeal. Whatever samples you have, you get both.

Everything below rests on those component functions being close to the real
ones. They are B-splines with `m` intervals per input. Pick `m` too low and
the curves are too stiff to follow the response, the indices shrink, and
nothing warns you. That is the subject of the second section.

## Import style

```python
from jaxgsa import hdmr
```

## Indices from random samples

Uniform random inputs stand in for data you already have. `maxorder=2` asks
for pair terms as well as single-input terms. `m=4` sets four B-spline
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

result = hdmr.analyze(PROBLEM, X, Y, maxorder=2, m=4)

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
    correlation: independent
    output: N=2000 runs, T=1 x K=1 output slice
    invalid: none found in 2000 rows (policy 'raise')
  timing:
    fit + estimator (includes compile on the first call): 1.993 s
    maxorder: 2
    slice_chunk_size: auto (resolved from the memory budget)
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5615
    2. x2  ST=0.4634
    3. x3  ST=0.2593
terms: ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')
S1: [0.32037902 0.44724527 0.00075385]
ST: [0.56145686 0.4634375  0.25926825]
Sa: [0.32037902 0.44724527 0.00075385 0.00254859 0.23652677 0.01298848]
Sb: [9.9287450e-04 1.7239768e-03 8.0345257e-05 2.9592053e-03 2.0535192e-02
 1.7308122e-02]
rmse: 0.49884835
std(Y): 3.7769964
```

The summary block above `terms:` prints by default in 1.0. Pass
`verbose=False` to silence it. It restates the problem it was handed, which
catches a wrong `bounds` array or a misread output shape before you read an
index.

Read `rmse` first. It is 0.4988 against a `std(Y)` of 3.777, so the surrogate
misses by 13% of the output spread. The indices describe the surrogate, not
the model, so a bad `rmse` makes every other field on the result a
description of the wrong function.

Then the per-term arrays. `Sa` and `Sb` have one entry per term and `terms`
names them in order. `Sa[4] = 0.2365` is the x1/x3 pair, and it dwarfs the
other two pairs at 0.0025 and 0.0130. That matches Ishigami, whose only real
interaction is x1 with x3. `S1` and `ST` collapse this to one number per
input. `S1` is the input's own term, and `ST` sums every term the input
appears in. Where `ST` runs far above `S1`, as x3's 0.259 does over its 0.0008, the
input matters only through a partner, and the term-level `Sa` says which one.

`Sb` is the correlative share. It is near zero here because the inputs are
independent, and it becomes the real diagnostic when they are not. See
[Correlated Inputs](/examples/correlated-inputs).

## `m` is the diagnostic knob

`m` is the number of B-spline intervals per input, and it sets how much each
component function can bend. The default is 2. Ishigami's $\sin^2(x_2)$ over
$[-\pi, \pi]$ needs more than that. Here is the same fit at four values:

```python
import numpy as np

sigma = float(jnp.std(Y))
for m in (2, 4, 6, 8):
    r = hdmr.analyze(PROBLEM, X, Y, maxorder=2, m=m, verbose=False)
    print(f"m={m}  rmse={float(r.rmse):.4f}  rmse/std={float(r.rmse)/sigma:6.1%}"
          f"  S1={np.round(r.S1, 4)}")
```

```text
m=2  rmse=1.2082  rmse/std= 32.0%  S1=[0.30699998 0.3538     0.0006    ]
m=4  rmse=0.4988  rmse/std= 13.2%  S1=[0.3204 0.4472 0.0008]
m=6  rmse=0.6052  rmse/std= 16.0%  S1=[0.3198 0.4461 0.0009]
m=8  rmse=0.7349  rmse/std= 19.5%  S1=[0.3148 0.4443 0.0012]
```

Ishigami's analytical `S1` is `[0.3139, 0.4424, 0]`. At the default `m=2` the
fit reports x2's first-order index as 0.354, low by 20%, and `rmse` sits at a
third of the output spread. Nothing errors. Nothing warns. You get a plausible
table with x2 understated, and the only clue is `rmse`.

At `m=4` the error drops to 13% and x2 lands at 0.447. Past that the error
climbs again, because more intervals means fewer samples per interval and the
splines start chasing noise. So `m` is not "higher is better". Sweep it, watch
`rmse`, and take the minimum. On this problem that is `m=4`.

My rule for `rmse / std(Y)`: under 10% the indices are reportable, 10% to 25%
treat the ranking as provisional, above 25% the numbers describe a surrogate
nobody should quote. RS-HDMR reaches those bands more slowly than a polynomial
does on smooth models, which is the honest cost of a local basis.

## Term selection

`result.select` counts how many output slices kept each term. With a scalar
`Y` there is one slice, so it reads as a flag:

```text
select   (term) float32 24B 1.0 1.0 0.0 0.0 1.0 1.0
```

Reading it against `terms = ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')`:
the fit dropped the x3 solo term and the x1/x2 pair, both of which are truly
absent from Ishigami. It kept x2/x3, which is not. Its `S` share is 0.0169, small
enough to ignore, but the F-test is a threshold and it does let noise through
near zero. Do not treat a selected term with a tiny share as evidence of an
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
Absolute residuals: [0.26264906 0.08427393 0.26891232 0.27363586 0.25042963]
```

Those residuals run around 0.25 against a `std(Y)` of 3.777, consistent with
the fitted `rmse` of 0.499. The emulator is on the same scale as the `Y` you
supplied, so a residual you can read against the output range is the whole
check. A gap that stays stubborn as you raise `m` is a structural miss instead,
and the fix is `maxorder=3`.

## When RS-HDMR beats plain Sobol

The classical Sobol estimator needs the Saltelli column-swap design, and
jaxgsa enforces that in the signature: `jaxgsa.sobol.analyze(sampling_result,
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
polynomial smooths over. On a smooth response PCE usually reaches a lower
error with fewer terms, and on Ishigami it does: `loo_rmse = 0.074` at
`order=8` against RS-HDMR's best `rmse = 0.499`. Fit both when you can. They
are cheap and they disagree informatively.

## Field reference

`result.S1` is the first-order share, the solo term's contribution. `result.ST`
sums every term containing the parameter. `result.Sa` is the structural
variance share per term and `result.Sb` the correlative one. `result.S` is
their sum, and `result.S.sum()` is 1.023 here, close to 1 as the theory
requires. `result.terms` labels the columns of all three. `result.select` is
the F-test flag per term. `result.rmse` is the fit error you check first.

`result.to_dataset()` puts all of it in an `xarray.Dataset` with `term`,
`param`, `param_i` and `param_j` coordinates, plus `time` and `output` where
they apply. See [xarray Labeled Output](/examples/xarray).

## Practical caveats

`hdmr.analyze()` accepts `(N,)`, `(N, K)` and `(N, T, K)` outputs under the
shape rules in [Multi-Output & Time-Series](/examples/multi-output).

`maxorder` must be 1, 2 or 3, and it is clamped with a warning when
`D < maxorder`. Fewer than 300 rows raises `ValueError`, because the
backfitting solve degenerates quietly below that.

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
