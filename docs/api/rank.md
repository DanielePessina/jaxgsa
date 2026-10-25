# Rank-based S1

```python
jaxgsa.rank.analyze(
    problem, X, Y, *,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    slice_chunk_size=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> RankResult | IrregularResult
```

The rank estimator gives all first-order Sobol' indices from one existing
`(X, Y)` sample. For each parameter, it sorts the rows by that parameter. It
pairs each output with the output of the next-ranked row, and the last rank
pairs with the first. The index is the covariance of the pairs divided by the
output variance:

$$
\hat S_i = \frac{\frac1N\sum_j Y_j\,Y_{N(j)} - \bar Y^2}{\frac1N\sum_j Y_j^2 - \bar Y^2},
$$

where $N(j)$ is the row whose rank in $X_i$ follows the rank of row $j$. If
the output depends strongly on the parameter, rows that are neighbours in the
sort have similar outputs, and the index is large. The estimator targets
$S_i = \mathrm{Var}(E[Y \mid X_i]) / \mathrm{Var}(Y)$.

The estimator is
[Gamboa, Gremaud, Klein and Lagnoux (2022)](https://doi.org/10.3150/21-BEJ1421),
eq. (20) of the arXiv version, with the neighbour map of eq. (8). It builds on
Chatterjee's rank correlation coefficient. It needs one sort per parameter. It
has no bin count and no tuning parameter.

## A run

```python
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)

def ishigami(X):
    return (np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2
            + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0]))

X = monte_carlo(problem, 4000, seed=0)
res = jaxgsa.rank.analyze(problem, X, ishigami(X))
print(np.asarray(res.S1))
```

```
jaxgsa.rank.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=4000 runs, T=1 x K=1 output slice
    invalid: none found in 4000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1746 s
    slice_chunk_size: 1 (resolved from the memory budget)
  results: top 3 of 3 parameters by S1
    1. x2  S1=0.4646
    2. x1  S1=0.2872
    3. x3  S1=0.001759

[0.28715482 0.46459588 0.0017589 ]
```

The Ishigami first-order indices are 0.314, 0.442 and 0. At $N = 4000$ the
standard error is about 0.02 for each index.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_bootstrap` | `0` | Resamples behind `S1_conf`. `0` skips them and leaves `S1_conf` at `None`. See the next section for how a resample is drawn. |
| `conf_level` | `0.95` | Confidence level for the intervals. |
| `ci_method` | `"quantile"` | `"quantile"` reads the endpoints off the empirical resample distribution. `"gaussian"` centres them on the point estimate and takes `+/- z * sd` of the draws. |
| `key` | `None` | A `jax.random` key for the resampling. Required when `n_bootstrap > 0`. Use `jax.random.key(0)` if you have an integer seed. |
| `slice_chunk_size` | `None` | Flattened `T*K` output slices the kernel reads at once. It bounds the peak memory, and the memory of a gradient, for a large `T * K`, and changes no index. `None` derives it from the memory budget. An explicit value must be an integer of at least 1. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. `"drop"` removes the `(X, Y)` pair, `"propagate"` warns and computes anyway. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Keeps the per-resample indices on `result.ci.replicates`. These are the rescaled draws that the interval is built from. |

## Confidence intervals

The usual bootstrap resamples rows with replacement. That does not work for
this estimator. A resample repeats rows. Repeated rows have equal inputs, so
they sort next to each other and pair with their own copy. That adds a large
positive bias. On Ishigami at $N = 4000$, the mean bootstrap draw for `x3`, whose
true index is 0, is about 0.37.

`rank.analyze` draws each resample as a random 80% of the rows, without
replacement, as the R package `sensitivity` does. A smaller sample scatters
less than the full sample by a known factor. The code centres the draws on
their own mean and scales the deviations by $\sqrt{m/(N-m)}$, which is 2 for
$m = 0.8N$.

The estimator also has a bias of order $1/N$. A smaller resample has a larger
bias, so the gap between the resample mean and the point estimate measures
it. jaxgsa forms two intervals. The first is centred on the point estimate.
The second is centred on the point estimate minus the measured bias. The
reported interval is the hull of the two, from the lower of the two lower
bounds to the higher of the two upper bounds. It need not be centred on the
point estimate.

Neither interval alone is enough. If the output is a smooth function of one
input, for example $Y = X_1$, the estimate is about $1 - 6/N$ and its spread
is much smaller than that bias. The first interval then misses the true value
in almost every sample. If the bias is small, as on Ishigami, the bias
estimate is noisy and the second interval alone covers only about 87% of the
time at the 95% level. In a simulation with 300 samples per case, the hull
covered the true value in 96% to 100% of samples on Ishigami ($N = 500$ and
$2000$) and on $Y = X_1$ ($N = 200$ and $1000$). It is about 10% to 15% wider
than the first interval. This rule is a heuristic that the simulation
supports, not an exact result for a rank statistic. It breaks down for very
small samples: for $N = 4$, every resample has 3 rows and the same value, so
the interval has zero width.
Use a few hundred resamples, and treat intervals at $N$ below about 50 with care.

## RankResult

| Field | Shape | Meaning |
| --- | --- | --- |
| `S1` | `(..., D)` | The first-order index. It is not clipped. It never exceeds 1, but a small sample can give a negative value. It is `NaN` for an output with zero variance. |
| `S1_conf` | `(2, ..., D)` | `[lower, upper]`, `None` when `n_bootstrap=0`. |
| `problem` | | The problem the analysis ran on. |
| `invalid` | | What the non-finite check found and which policy ran. |
| `ci` | | Confidence level, endpoint rule, resample count, and the draws when `keep_replicates=True`. |

Leading axes follow the shape contract: `(D,)` for `Y` of shape `(N,)`,
`(K, D)` for `(N, K)`, `(T, K, D)` for `(N, T, K)`.

`res.to_dataset(time_coords=None)` gives the labeled xarray view.

## Correlated inputs

Supported. $\mathrm{Var}(E[Y \mid X_i]) / \mathrm{Var}(Y)$ is defined for any
joint distribution of the inputs. A declared `problem.correlation` or a custom
sampler therefore does not invalidate the index. Under dependence, the index
measures the total first-order influence of a parameter, including what it
carries through correlated partners. A parameter the model ignores can score
above 0 when it correlates with an influential one. That reading is correct.

## Categorical inputs

Refused. The level codes of a categorical parameter are heavily tied, and the
rank pairing is ill defined for ties. Use [PAWN](/api/pawn),
[Borgonovo delta](/api/borgonovo), [optimal transport](/api/optimal-transport)
or [Sobol'](/api/sobol), which accept categorical inputs.

## Ties

A stable sort breaks tied values of a continuous parameter by row order. A few
ties change nothing of substance. Many ties among the inputs bias the index,
because row order then decides which outputs pair.

## What it refuses

`ValueError` for a categorical parameter, a non-2-D `X`, a column count that
disagrees with the problem, mismatched row counts, fewer than 3 rows,
`conf_level` outside `(0, 1)`, an unknown `on_invalid`, `n_bootstrap < 0`, an
unknown `ci_method`, `n_bootstrap > 0` with no `key`, or a sample the
non-finite policy refuses.

`JaxgsaWarning` for a zero-variance output slice. The index is then `NaN`,
because it is a ratio to the output variance.

## Traceable core

`jaxgsa.rank.indices(problem, X, Y, *, slice_chunk_size=None)` returns a `RankIndices` named tuple with one field,
`S1`, holding the index array, with none of the diagnostics, so it composes with `jit` and
`vmap`. It still refuses categorical problems, because that check reads only
static problem metadata. The estimator reads the ranks of `X`, so the gradient
with respect to `X` is 0 almost everywhere. The gradient with respect to `Y` is
not 0.

## Check against R

`tests/test_rank.py` compares the index with `sensitivity::sobolrank` from the
R package `sensitivity`, run as a black box on the same data. R divides by the
sample variance with $N-1$ in the denominator, and jaxgsa follows the paper
with $N$. The two differ by the exact factor $N/(N-1)$, and they agree to
$2\times10^{-4}$ after that factor.

## References

- Gamboa, F., Gremaud, P., Klein, T., and Lagnoux, A. (2022). [Global sensitivity analysis: a novel generation of mighty estimators based on rank statistics](https://doi.org/10.3150/21-BEJ1421). *Bernoulli*, 28(4), 2345–2374. Gives the rank estimator of the first-order Sobol' index.
- Chatterjee, S. (2021). [A new coefficient of correlation](https://doi.org/10.1080/01621459.2020.1758115). *Journal of the American Statistical Association*, 116(536), 2009–2022. Introduces the rank-pairing statistic that the estimator builds on.

See the [Methods](/guide/methods) and the [API overview](/api/).
