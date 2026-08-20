# Kucherenko (Dependent-Input Sobol' Indices)

This page produces two sensitivity indices per input for a model whose inputs
are correlated. The numbers come from running the model itself on a
purpose-built design. No surrogate is fitted anywhere.

A Sobol' index is the share of the output variance that one input accounts
for. Kucherenko indices are Sobol' indices for inputs that are not independent
(Kucherenko, Tarantola & Annoni, 2012). The estimator runs your model on a
conditional-copula design, that is, a sample drawn so that it respects the
declared dependence between the inputs.

This makes the method the design-based counterpart to
[VKOGA](/examples/vkoga). Both estimate the same two conditional-variance
quantities. VKOGA gets them from data you already have, through a surrogate.
Kucherenko gets them from fresh model runs, so it carries no surrogate error.
The price is a dedicated sampling design.

## What this gives you that plain Sobol' cannot

`jaxgsa.sobol.sample` refuses a problem with a declared correlation. That is
not caution. The Saltelli column-swap scheme builds its conditional samples by
copying a column from one base matrix into another, which silently assumes the
column can take any value regardless of the other columns. Under dependence
that assumption is false, and the design puts the model on input combinations
that the joint distribution says never occur. The resulting index is a number
about a distribution nobody declared.

Kucherenko replaces the column swap with a draw from the conditional copula.
The swapped column is sampled from its distribution *given* the columns that
were kept. Every row stays inside the declared joint distribution. That one
change is what buys the two readings below.

## The two indices

Both indices keep their defining formulas under dependence:

- `S1` = $V(E(Y|X_i))/V(Y)$. Correlation-inclusive: what $X_i$ explains
  through itself and through its coupling with the others. Equals VKOGA's
  `S_TC`.
- `ST` = $E(V(Y|X_{\sim i}))/V(Y)$. Correlation-exclusive: what only $X_i$
  can explain. Equals VKOGA's `S_TU`.

The two answer different questions, and under correlation they give different
answers. Read `S1` when you want to know which input to measure more
accurately: a large `S1` means the output tracks that input closely. Read `ST`
when you want to know which input you can fix at a nominal value: a small `ST`
means the other inputs already carry everything that input contributes.

Under independent inputs both reduce exactly to the classic Sobol' `S1` and
`ST`. Under correlation, `ST >= S1` no longer holds. A strongly coupled input
has a large `S1` and a small `ST`, and the gap is the correlation-borne share.

## Workflow

The usual sample / evaluate / analyze split, in five steps:

1. Declare the marginal distribution of each input with
   `Problem.from_dict`. The sampler needs each input's own distribution before
   it can couple them.
2. Attach the dependence with `with_correlation`. The sampler conditions on
   this matrix, and `analyze` reads it back off the design.
3. Draw the design with `kucherenko.sample`. It costs `base_n * (2D + 1)`
   model runs: one joint block, plus two conditional blocks per parameter.
4. Run your model on `ks.samples`. The example uses a linear model so that the
   answer can be checked against a closed form.
5. Call `kucherenko.analyze` on the design and the outputs to get `S1` and
   `ST`.

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)

# Declare the dependence on the problem; the sampler conditions on it.
R = np.array(
    [
        [1.0, 0.6, 0.0],
        [0.6, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
problem = problem.with_correlation(R)

ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)

a = np.array([2.0, 1.0, 0.5])
Y = ks.samples @ a  # your model here

result = jaxgsa.kucherenko.analyze(ks, Y)
print("S1:", np.round(result.S1, 3))
print("ST:", np.round(result.ST, 3))
```

```
jaxgsa.kucherenko.sample: D=3, base_n=4096, n_blocks=7, n_runs=28672, dependence=copula-conditional, scramble=True
jaxgsa.kucherenko.analyze
  problem: D=3 (x1, x2, x3)
    marginals: gaussian=3
    correlation: correlated (Gaussian copula)
    output: N=28672 runs, T=1 x K=1 output slice
    invalid: none found in 4096 base points (policy 'raise')
  timing:
    compute: 0.02234 s
    design: copula-conditional (2D+1 = 7 blocks of 4096 base points)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.3351
    2. x2  ST=0.08364
    3. x3  ST=0.0327
S1: [0.883 0.632 0.031]
ST: [0.335 0.084 0.033]
```

Why these settings. `base_n=4096` costs 28,672 model runs here, and that is
what buys three-decimal agreement with the closed form below. Halve it and the
third decimal goes. The correlation ties `x1` and `x2` at 0.6 and leaves `x3`
independent of both, which is the smallest problem that shows a correlated pair
and an uncorrelated control in the same run. A linear model is used because its
indices have a closed form:

```python
var_y = a @ R @ a  # 7.65
print("closed S1:", np.round((R @ a) ** 2 / var_y, 3))
print("closed ST:", np.round(a**2 * np.array([0.64, 0.64, 1.0]) / var_y, 3))
```

```
closed S1: [0.884 0.633 0.033]
closed ST: [0.335 0.084 0.033]
```

`ST` matches to all three decimals. `S1` is off by 0.001 on `x1` and `x2`.

## Reading the numbers

- `x1` and `x2` have large `S1` (0.88 and 0.63) but small `ST` (0.34 and
  0.08). They share correlated variance. Each explains a large slice of the
  output, but most of that slice is also reachable through the other. So `S1`
  ranks them for measurement, and `ST` ranks them for fixing. Measuring either
  one more accurately pays off. Fixing either one while the other stays free
  does not.
- `x3` has `S1` and `ST` within 0.002 of each other (0.031 and 0.033). It is
  uncorrelated with the rest, so the correlation-inclusive and
  correlation-exclusive readings coincide, as they do for every input of an
  independent problem.

The practical mistake this prevents: `x2` looks like the second most important
input on `S1` and is almost free to fix on `ST`. A single-number ranking hides
one of those two facts, and which one it hides depends on which estimator you
happened to run.

## Independent inputs collapse to classic Sobol'

With no declared correlation, the design is exactly the Saltelli column-swap
scheme and the indices are the classic ones:

```python
plain = problem.with_correlation(None)
ks0 = jaxgsa.kucherenko.sample(plain, 4096, seed=0, verbose=False)
r0 = jaxgsa.kucherenko.analyze(ks0, ks0.samples @ a, verbose=False)

print("S1:", np.round(r0.S1, 3))
print("ST:", np.round(r0.ST, 3))
print("analytic S1 = ST:", np.round(a**2 / (a**2).sum(), 3))
```

```
S1: [0.762 0.19  0.046]
ST: [0.763 0.19  0.048]
analytic S1 = ST: [0.762 0.19  0.048]
```

`S1` and `ST` agree to within 0.002 per input, and both match the analytic
answer. The gap between the two indices closed because the correlation that
opened it is gone.

## Seeding the design

`kucherenko.sample` takes `seed: int | np.random.Generator | None = None`, the
same interface as the other samplers. Pass an int for a reproducible run, a
`Generator` when the design is one draw in a larger seeded workflow, or nothing
at all for a fresh design each call. `seed=0` reproduces the design that older
versions produced by default.

The seed drives the Sobol' scrambling. Asking for an unscrambled sequence and a
seed at the same time is a contradiction, so it raises rather than quietly
ignoring one of the two:

```python
jaxgsa.kucherenko.sample(problem, 128, scramble=False, seed=0)
```

```
ValueError: jaxgsa.kucherenko.sample: seed has no effect with scramble=False.
The unscrambled Sobol' sequence is deterministic, so the seed would do nothing.
Use scramble=True, or drop the seed.
```

Keep `scramble=True` unless you have a specific reason not to. Without
scrambling you get one fixed design and no way to estimate its Monte-Carlo
error, because there is no second draw to compare against.

## Persistence and datasets

`KucherenkoSamples` uses the same one-file NPZ persistence as the other
designs, correlation included:

```python
ks.save("design.npz")
ks2 = jaxgsa.kucherenko.KucherenkoSamples.load("design.npz")
print(ks2.samples.shape, ks2.problem.has_correlated_inputs)
```

```
(28672, 3) True
```

Saving the design matters here because the model runs are the expensive part.
Reloading `design.npz` lets you re-analyze the same evaluations without
redrawing the sample.

`to_dataset()` gives labeled output:

```python
print(result.to_dataset())
```

```
<xarray.Dataset> Size: 80B
Dimensions:   (param: 3)
Coordinates:
  * param     (param) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    S1        (param) float64 24B 0.8835 0.6321 0.03135
    ST        (param) float64 24B 0.3351 0.08364 0.0327
    variance  float64 8B 7.647
Attributes:
    is_correlated:  True
```

`variance` is 7.647 against the closed-form 7.65, and it is the denominator of
both indices.

## Kucherenko or VKOGA?

Both estimate the same two quantities under the same Gaussian copula, and the
test suite pins them to the same closed form, and to each other.

| | Kucherenko | VKOGA |
| --- | --- | --- |
| Model evaluations | `base_n * (2D + 1)` on a dedicated design | Zero new runs; any existing `(X, Y)` |
| Estimation error | Monte-Carlo only | Monte-Carlo plus surrogate error |
| Extras | none | Full five-index split (`S_U`, `S_C`, `S_IU`) and a reusable emulator |

Use Kucherenko when the model is still runnable and cheap enough for the
design. Use VKOGA when the data is fixed, or when you want the finer
correlated/uncorrelated split. When a VKOGA run reports an index outside
`[0, 1]`, or warns about its training design, Kucherenko is the check that
settles it.

## Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global
  sensitivity indices for models with dependent variables. *Computer Physics
  Communications*, 183(4), 937-946.
