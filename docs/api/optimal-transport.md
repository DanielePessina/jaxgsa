# Optimal transport

```python
jaxgsa.optimal_transport.analyze(
    problem, X, Y, *,
    mode="univariate",
    n_partitions=None,
    standardize_outputs=True,
    epsilon=0.01,
    max_iter=1000,
    tol=None,
    dummy=False,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    slice_chunk_size=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> OTResult
```

The OT index is the class-averaged squared 2-Wasserstein distance between the
output distribution conditional on a parameter and the unconditional one,
normalized to `[0, 1]` by twice the output variance. 0 means the parameter
leaves the output distribution alone. 1 means it determines it fully.

`analyze` splits that index into an advective part (location shift) and a
diffusive part (spread and shape). Variance-based indices react only to the
first. Any `(X, Y)` pair works, and the method handles mixed marginals,
categorical parameters and correlated inputs natively.

## A run

```python
import jax
import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
N = 4000
X = monte_carlo(problem, N, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.optimal_transport.analyze(problem, X, Y, dummy=True, key=jax.random.key(0))
for name in ("ot", "advective", "diffusive", "S1", "above_dummy"):
    print(f"{name:12s}", np.asarray(getattr(res, name)).round(4))
print("ot_dummy    ", np.asarray(res.ot_dummy).round(4))
```

```
jaxgsa.optimal_transport.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4000 runs, T=1 x K=1 output slice
    invalid: none found in 4000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.5295 s
    mode: univariate
    epsilon: 0.01
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ot
    1. x2  ot=0.2775
    2. x1  ot=0.2013
    3. x3  ot=0.09772

ot           [0.2013 0.2775 0.0977]
advective    [0.1536 0.2198 0.0037]
diffusive    [0.0477 0.0577 0.094 ]
S1           [0.3072 0.4398 0.0074]
above_dummy  [0.19   0.2662 0.0864]
ot_dummy     0.0114
```

Look at `x3`. Its advective part is 0.004 and its diffusive part is 0.094, so
essentially all of its influence is a change in the shape of the output
distribution rather than a shift of its centre. `S1 = 0.007` reports it as
irrelevant. It is not. That is the case this method exists for.

`verbose=True` is the default and printed the block. Pass `verbose=False` for a
silent run.

## S1 against 2 * advective

`advective` is the location-shift component on the OT normalization: the
class-averaged squared distance between the conditional and unconditional output
means, divided by twice the sample variance. That normalizer uses the unbiased
(`ddof=1`) variance.

`S1` is the same quantity restated on the Sobol convention,
`Var(E[Y | X_i]) / Var(Y)` with both variances taken as population (`ddof=0`)
variances. That is exactly the convention
[`jaxgsa.borgonovo`](/api/borgonovo)'s `S1` uses, so the two are directly
comparable. Concretely:

```
S1 = 2 * advective * N / (N - 1)
```

In the run above `2 * advective * 4000/3999` reproduces
`[0.3072 0.4398 0.0074]` to four decimals. The field exists so that no reader
has to carry a `ddof` caveat around. In the point-cloud modes the variances
generalize to the trace of the output covariance.

`S1_conf` is `advective_conf` rescaled by the same constant. Every bootstrap
resample has size `N`, so the factor is one number, not a per-draw correction.

## The dummy floor

`dummy=True` pushes one synthetic parameter, independent of the output by
construction, through the identical pipeline and reports its index as
`ot_dummy`. That is the score a provably irrelevant parameter picks up from
finite-sample bias, and in the point-cloud modes from entropic bias as well.

`above_dummy` is `max(ot - ot_dummy, 0)`. A value of 0 means the parameter is
indistinguishable from noise at this sample size. The name says what it is, the
excess above the dummy floor, rather than claiming the subtraction removes bias
in general; it does that only for irrelevant parameters.

Both fields are `None` unless the analysis ran with `dummy=True`. `dummy=True`
also requires a `key`, because the synthetic parameter is drawn.

In the run above `ot_dummy = 0.011` and `x3` scores 0.098, well clear of the
floor.

## Modes

| `mode` | What it does | Index shape |
| --- | --- | --- |
| `"univariate"` (default) | Scores every output column independently with exact 1-D optimal transport. | `(D,)`, `(K, D)`, or `(T, K, D)` |
| `"multivariate"` | Treats the whole flattened output vector as one point cloud and gives one index per parameter over the joint output distribution. Entropic Sinkhorn transport. | `(D,)` |
| `"trajectory"` | The same, per output, with each output's time course as the cloud. Requires a 3-D `Y`. | `(K, D)` |

`standardize_outputs`, `epsilon`, `max_iter` and `tol` apply to the two
point-cloud modes only.

## Arguments

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_partitions` | `None` | Equal-frequency conditioning classes per continuous parameter. More classes localize the conditioning and cut the discretization bias, but leave fewer samples per class and raise the noise. About 25 is customary at `N >= 2500`. `None` selects `min(25, N // 2)`. A passed value is validated against `[2, N // 2]`. Categorical parameters ignore it and use one class per level. If every parameter is categorical and `dummy` is false, nothing uses the value and a `JaxgsaWarning` says it is ignored. |
| `standardize_outputs` | `True` | Joint modes only. Divides each output column by its standard deviation before the transport cost is built, so no single output dominates the joint distance through its units. Turn it off only when the outputs already share a scale and their relative magnitudes are meaningful. Ignored in `"univariate"` mode, where each column is normalized by its own variance anyway. |
| `epsilon` | `0.01` | Joint modes only. Entropic regularization strength, relative to the cost matrix scaled to `[0, 1]`. Smaller values approach exact transport and need more iterations. It also sets the entropic part of the dummy floor, so lowering it lowers `ot_dummy`. |
| `max_iter` | `1000` | Joint modes only. Sinkhorn iteration cap per solve. |
| `tol` | `None` | Joint modes only. Stopping tolerance on the L1 target-marginal violation. `None` selects `1e-9` in float64 and `1e-6` in float32, where anything tighter is unresolvable. One warning is emitted if any solve fails to converge. |
| `dummy` | `False` | See above. Requires `key`. |
| `n_bootstrap` | `0` | Row resamples. `0` leaves every `*_conf` at `None`. The joint modes solve `n_bootstrap * D * n_partitions` transport problems, so keep it modest there. |
| `key` | `None` | A `jax.random` key. It feeds two independent consumers, the bootstrap and the synthetic dummy parameter, so it is required when either `n_bootstrap > 0` or `dummy=True`. |
| `slice_chunk_size` | `None` | `"univariate"` only. Flattened `T*K` output columns per kernel call. `None` picks a memory-aware default. The point-cloud modes accept it and ignore it, because one `(N, N/M)` cost block per solve already bounds their peak memory. |
| `on_invalid` | `"raise"` | Policy for non-finite rows. The check reads the real `X` and `Y` you passed, not the synthetic dummy column, and reads them together, so a bad input takes its output with it. |
| `verbose` | `True` | Prints the summary block shown above. |
| `keep_replicates` | `False` | Keeps `n_bootstrap` copies of all three index arrays on `result.ci.replicates`. Worth more here than elsewhere: recomputing an interval at another level then costs nothing, instead of re-solving every transport problem. |

`conf_level` and `ci_method` behave as on every other method.

The old `standardize=` keyword is gone. `standardize_outputs` is the one
spelling.

## OTResult

| Field | Meaning |
| --- | --- |
| `ot` | Total index, `(..., D)`. |
| `advective` | Location-shift component, `(..., D)`. Half the given-data first-order Sobol index on the OT normalization. |
| `diffusive` | `ot - advective`, the dispersion and higher-moment part. |
| `S1` | The advective component on the Sobol `ddof=0` convention. See above. |
| `above_dummy` | `max(ot - ot_dummy, 0)`. `None` without `dummy=True`. |
| `ot_dummy` | The irrelevance floor. Shape of `ot` without the trailing parameter axis. `None` without `dummy=True`. |
| `mode` | The mode that produced these shapes. |
| `ot_conf`, `advective_conf`, `diffusive_conf`, `S1_conf` | `(2, ...)` intervals, `None` when `n_bootstrap=0`. |
| `problem`, `invalid`, `ci` | Problem, non-finite report, interval provenance. |

`res.to_dataset(time_coords=None)` gives the labeled xarray view. It resolves
its dimensions from `mode` rather than from the array rank, so a `"trajectory"`
result on a one-output model still labels its leading axis `output` instead of
mistaking it for `param`.

Every index is 0 for a constant output slice rather than NaN. In the point-cloud
modes the entropic and finite-sample bias keeps irrelevant parameters strictly
positive, so compare those against `ot_dummy` and not against 0.

## Ranks, categoricals and correlation

Conditioning classes come from the ordinal ranks of the parameters. Ranks
survive any monotone transform, so the estimator is distribution-free in `X`. It
works unchanged for uniform, Gaussian, truncated-Gaussian or mixed marginals,
and it applies no CDF transform.

Categorical parameters work natively. Each conditions on one class per level,
with class sizes equal to the observed level counts, so the index depends only
on the level partition and never on the arbitrary code order. Declared levels
with no observed samples are dropped with a warning.

Correlated parameters are supported, and the reading is the total-association
one: a parameter that never enters the model but correlates with one that does
scores non-zero. That is the same reading as the given-data `S1` whose half is
the advective component.

## What it refuses

`ValueError` for a non-2-D `X`, a column count that disagrees with the problem,
a `Y` that is not 1-D/2-D/3-D, mismatched row counts, an unknown `mode`,
`mode="trajectory"` with a non-3-D `Y`, an `n_partitions` outside
`[2, N // 2]`, a categorical column holding values other than its level codes,
`epsilon <= 0`, `max_iter < 1`, `tol <= 0`, `n_bootstrap < 0`, an unknown
`ci_method`, a missing `key` while `n_bootstrap > 0` or `dummy=True`,
`conf_level` outside `(0, 1)`, a `slice_chunk_size` that is not a positive
integer, an unknown `on_invalid`, or a sample the non-finite policy refuses.

`JaxgsaWarning` for a zero-variance output slice, where every conditional
distribution equals the unconditional one and the indices are an exact 0.

## Traceable core

`jaxgsa.optimal_transport.indices(problem, X, Y, *, mode="univariate",
n_partitions=None, standardize_outputs=True, epsilon=0.01, max_iter=1000,
tol=None, slice_chunk_size=None)` returns `(ot, advective, diffusive)` as bare
arrays with none of the checks, so it composes with `jit`, `vmap` and `jacrev`.
`S1`, `above_dummy` and the intervals live on the result object only.

See the [Optimal transport example](/examples/optimal-transport),
[Methods](/guide/methods), and the [API overview](/api/).
