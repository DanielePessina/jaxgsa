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
print(ks.n_runs)  # 28672 = 4096 * (2*3 + 1)

Y = ks.samples @ np.array([2.0, 1.0, 0.5])  # your model here

result = jaxgsa.kucherenko.analyze(ks, Y)
print("S1:", np.round(result.S1, 3))  # [0.883 0.632 0.031]
print("ST:", np.round(result.ST, 3))  # [0.335 0.084 0.033]
```

The correlation matrix `R` ties `x1` and `x2` together at 0.6 and leaves `x3`
independent of both. This linear model has a closed form:
`S1 = (R a)_i^2 / (a' R a)`, which gives `[0.884, 0.633, 0.033]`. The estimate
lands within a few 1e-3 of it.

## Reading the numbers

- `x1` and `x2` have large `S1` (0.88 and 0.63) but small `ST` (0.33 and
  0.08). They share correlated variance. Each explains a large slice of the
  output, but most of that slice is also reachable through the other. So `S1`
  ranks them for measurement, and `ST` ranks them for fixing. Measuring either
  one more accurately pays off. Fixing either one while the other stays free
  does not.
- `x3` has `S1 = ST` (0.031 against 0.033). It is uncorrelated with the rest,
  so the correlation-inclusive and correlation-exclusive readings coincide, as
  they do for every input of an independent problem.

## Independent inputs collapse to classic Sobol'

With no declared correlation, the design is exactly the Saltelli column-swap
scheme and the indices are the classic ones:

```python
plain = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x3": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    }
)
ks0 = jaxgsa.kucherenko.sample(plain, 4096, seed=0)
r0 = jaxgsa.kucherenko.analyze(ks0, ks0.samples @ np.array([2.0, 1.0, 0.5]))

print("S1:", np.round(r0.S1, 3))  # [0.762 0.19  0.046]
print("ST:", np.round(r0.ST, 3))  # [0.763 0.19  0.048]
# analytic S1 = ST:                 [0.762 0.19  0.048]
```

The printed `S1` and `ST` agree to within 0.002 per input, and both match the
analytic answer `[0.762, 0.19, 0.048]`. The gap between the two indices closed
because the correlation that opened it is gone.

## Persistence and datasets

`KucherenkoSamples` uses the same one-file NPZ persistence as the other
designs, correlation included:

```python
ks.save("design.npz")
ks2 = jaxgsa.kucherenko.KucherenkoSamples.load("design.npz")

ds = result.to_dataset()  # S1, ST, variance with param/output/time coords
```

Saving the design matters here because the model runs are the expensive part.
Reloading `design.npz` lets you re-analyze the same evaluations without
redrawing the sample.

## Kucherenko or VKOGA?

Both estimate the same two quantities under the same Gaussian copula, and the
test suite pins them to the same closed form, and to each other.

| | Kucherenko | VKOGA |
| --- | --- | --- |
| Model evaluations | `base_n * (2D + 1)` on a dedicated design | Zero new runs; any existing `(X, Y)` |
| Estimation error | Monte-Carlo only | Monte-Carlo plus surrogate error |
| Extras | — | Full five-index split (`S_U`, `S_C`, `S_IU`) and a reusable emulator |

Use Kucherenko when the model is still runnable and cheap enough for the
design. Use VKOGA when the data is fixed, or when you want the finer
correlated/uncorrelated split.

## Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global
  sensitivity indices for models with dependent variables. *Computer Physics
  Communications*, 183(4), 937-946.
