# Kucherenko (Dependent-Input Sobol' Indices)

Kucherenko indices are Sobol' indices for dependent inputs, estimated by
running your actual model on a conditional-copula design (Kucherenko,
Tarantola & Annoni, 2012). No surrogate is fitted. This makes the method the
design-based counterpart to [VKOGA](/examples/vkoga): the same two
conditional-variance quantities, but free of surrogate error — at the price
of a dedicated sampling design.

The two indices keep their defining formulas under dependence:

- **`S1`** = $V(E(Y|X_i))/V(Y)$ — correlation-inclusive: what $X_i$
  explains through itself and through its coupling with the others. Equals
  VKOGA's `S_TC`.
- **`ST`** = $E(V(Y|X_{\sim i}))/V(Y)$ — correlation-exclusive: what only
  $X_i$ can explain. Equals VKOGA's `S_TU`.

Under independent inputs both reduce exactly to the classic Sobol' `S1` and
`ST`. Under correlation, `ST >= S1` no longer holds — a strongly coupled
input has a large `S1` and a small `ST`, and the gap is the
correlation-borne share.

## Workflow

The usual sample / evaluate / analyze split. The design costs
`base_n * (2D + 1)` model runs: one joint block plus two conditional blocks
per parameter.

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

This linear model has a closed form: `S1 = (R a)_i^2 / (a' R a)`, which gives
`[0.884, 0.633, 0.033]`. The estimate lands within a few 1e-3 of it.

## Reading the numbers

- `x1` and `x2` have large `S1` (0.88, 0.63) but small `ST` (0.33, 0.08).
  They share correlated variance: each explains a large slice of the output,
  but most of that slice is also reachable through the other. `S1` ranks them
  for measurement, `ST` ranks them for fixing.
- `x3` has `S1 = ST`. It is uncorrelated with the rest, so the two
  readings coincide, as they do for every input of an independent problem.

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

## Persistence and datasets

`KucherenkoSamples` uses the same one-file NPZ persistence as the other
designs, correlation included:

```python
ks.save("design.npz")
ks2 = jaxgsa.kucherenko.KucherenkoSamples.load("design.npz")

ds = result.to_dataset()  # S1, ST, variance with param/output/time coords
```

## Kucherenko or VKOGA?

Both estimate the same two quantities under the same Gaussian copula, and the
test suite pins them to the same closed form — and to each other.

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
