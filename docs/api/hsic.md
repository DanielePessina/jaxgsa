# HSIC

```python
jaxgsa.hsic.analyze(
    problem, X, Y, *,
    n_perms=200,
    key=None,
    bandwidth=1.0,
    on_invalid="raise",
    verbose=True,
) -> HSICResult
```

HSIC measures kernel dependence between each input and the output. It is not a
variance decomposition, so it reacts to any statistical dependence, including
the non-monotonic kind that a rank correlation reports as zero. A permutation
test attaches a p-value to every index, which makes HSIC a screening tool with
a stated significance level.

Any `(X, Y)` pair works. No design is required.

## A run

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)

X = monte_carlo(problem, 1024, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.hsic.analyze(problem, X, Y, key=jax.random.key(0))
print(np.asarray(res.R2_HSIC).round(4))
print(np.asarray(res.T_HSIC).round(4))
print(np.asarray(res.p_values).round(4))
```

```
jaxgsa.hsic.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=1024 runs, T=1 x K=1 output slice
    invalid: none found in 1024 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 3.172 s
    n_perms: 200
    bandwidth: 1.0 (median-heuristic multiplier)
  results: top 3 of 3 parameters by T_HSIC
    1. x1  T_HSIC=0.7996
    2. x3  T_HSIC=0.1873
    3. x2  T_HSIC=0.07121

[0.135  0.0097 0.0252]
[0.7996 0.0712 0.1873]
[0.005  0.005  0.005 ]
```

That is the Ishigami function. `x3` enters it only through the `x3^4 * sin(x1)`
term, so its first-order `R2_HSIC` is 0.025 while its total `T_HSIC` is 0.187.
The gap is the interaction. All three p-values sit at `1 / (n_perms + 1)`,
the smallest value 200 permutations can resolve, so all three parameters are
significant and none of them is more significant than the others.

`verbose=True` is the default and produced the block above. Pass
`verbose=False` for a silent run.

## Memory is quadratic in N, and nothing bounds it

The estimator holds the `D` input kernels, the `D` augmented complement
products, the full augmented product and one output kernel at the same time.
Peak memory is about `(2D + 1) * N^2` floats. At `D = 10` and `N = 20000` in
float64 that is 67 GB.

**`analyze` takes no `batch_size`, on purpose.** A row-blocked kernel build was
tried and removed: it bounded a transient of the build while the resident
stacks stayed exactly as large, so it bought nothing. If the sample does not
fit, cut `N` (HSIC converges quickly in `N`) or cut `D` by screening with a
cheaper method first. Do not go looking for a batching keyword.

## No bootstrap either

HSIC is the one analysis here without `n_bootstrap`. The `p_values` are already
the uncertainty statement, and a row bootstrap would be worse than redundant.
HSIC is a V-statistic, a double sum over all `N^2` row pairs. Resampling rows
with replacement repeats rows, the repeats land on the kernel diagonal where the
Gaussian kernel equals 1, and the resampled index is biased upward by
construction. The interval would mix that bias with the sampling spread it was
meant to show.

`key` is still required, because the permutation test needs randomness of its
own.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_perms` | `200` | Permutations behind the p-values. Cost is linear in it. The smallest reachable p-value is `1 / (n_perms + 1)`, so 200 bottoms out at 0.005. Raise it only when you need to separate parameters that all hit that floor. |
| `key` | `None`, but required | A `jax.random` key for the permutations. There is no default because the p-values are random. `jax.random.key(0)` if you just want reproducibility. It is a key rather than an int seed so nested analyses can split it and draw independent permutations. |
| `bandwidth` | `1.0` | Multiplier on the Gaussian bandwidth, applied to every input and to the output. The width itself always comes from the median heuristic per variable: the Gaussian standard deviation is `bandwidth * sqrt(m)`, with `m` the median off-diagonal squared pairwise distance. `0.5` halves every width, `2.0` doubles them. There is no absolute-width setting, because one fixed number cannot be right for both a parameter on `[0, 1]` and an output in megapascals. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. `"drop"` removes the `(X, Y)` pair, `"propagate"` warns and computes anyway. `X` and `Y` are checked together, so a bad input takes its output with it. |
| `verbose` | `True` | Prints the summary block shown above. |

Because the bandwidth tracks the data, the indices are unchanged under
`Y -> a*Y + b` and under any rescaling of an input. Rescaling an output of
extreme magnitude by hand is still worth doing: the squared distances can
overflow float32 long before the index would care.

## HSICResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `R2_HSIC` | `(..., D)` | `HSIC(x_i, Y) / sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))`, in `[0, 1]`. The first-order view. 0 means independence. |
| `T_HSIC` | `(..., D)` | Fraction of the joint dependence lost when `x_i` is removed. The analogue of a total-order index, so it counts interactions. |
| `p_values` | `(..., D)` | Permutation-test p-values against the null that `x_i` and `Y` are independent. |
| `hsic_raw` | `(..., D)` | Unnormalized `HSIC(x_i, Y)`. Kernel- and scale-dependent, so compare only within one analysis. |
| `problem` | | The problem the analysis ran on. |
| `invalid` | | What the non-finite check found and which policy ran. |

The leading axes follow the shape contract: `(D,)` for `Y` of shape `(N,)`,
`(K, D)` for `(N, K)`, `(T, K, D)` for `(N, T, K)`.

There is no `ci` field and no `*_conf` field. See the section above.

`res.to_dataset(time_coords=None)` gives the labeled xarray view.

## Correlated inputs

Supported, and the reading changes. HSIC is a dependence measure that assumes
nothing about input independence, so a declared `problem.correlation` does not
invalidate anything. Each index then measures a parameter's total association
with the output, including what it carries through its correlated partners. A
parameter the model never reads can score well above 0 when it correlates with
one that matters. That is the right answer, not an estimation error.

## What it refuses

`analyze` raises `ValueError` for a non-2-D `X`, a column count that disagrees
with the problem, mismatched row counts, `n_perms < 1`, `N < 4`, a non-positive
or non-finite `bandwidth`, an unknown `on_invalid`, a missing `key`, a sample
the non-finite policy refuses, and any categorical parameter. Categorical
parameters are rejected because the Gaussian input kernel would read a level
code as a distance, and the code order is arbitrary.

It warns on a zero-variance output slice, and on float32. The V-statistic
subtracts three sums of the same magnitude, so single precision leaves three or
four correct digits and the index moves with the row order of the sample. Turn
x64 on before you trust a small index or a close ranking.

## Traceable core

`jaxgsa.hsic.indices(problem, X, Y, *, n_perms=200, key, bandwidth=1.0)`
returns `(R2_HSIC, T_HSIC, p_values, hsic_raw)` as bare arrays with none of the
checks, so it composes with `jit`, `vmap` and `jacrev`. `key` is keyword-only
and has no default there either.

See the [HSIC example](/examples/hsic), [Methods](/guide/methods), and the
[API overview](/api/).
