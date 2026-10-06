# VKOGA (correlated-input indices)

```python
jaxgsa.vkoga.analyze(
    problem, X, Y, *,
    correlation=None,
    sampler=None,
    sampler_dimension=None,
    sequence="sobol",
    gamma=None,
    ridge=None,
    max_centers=None,
    n_folds=10,
    n_outer=512,
    n_inner=128,
    n_variance=8192,
    n_bootstrap=0,
    conf_level=0.95,
    ci_method="quantile",
    key=None,
    batch_size=None,
    on_invalid="raise",
    verbose=True,
    keep_replicates=False,
) -> VKOGAResult | IrregularResult
```

`analyze` fits a Vectorial Kernel Orthogonal Greedy Algorithm surrogate to
given `(X, Y)` data, then estimates the five correlated indices of
[Li et al. (2010)](https://doi.org/10.1021/jp9096919) against that surrogate under
a Gaussian copula by default, or a user-supplied joint distribution. This surrogate-based sensitivity workflow follows
[Hilhorst et al. (2024)](https://doi.org/10.1002/cnm.3797). It is the only method in
jaxgsa that separates correlated from uncorrelated contributions out of a
single fitted surrogate, and the only one that gets a correlated total without
re-running the model.

The kernel centres use the target-independent **P-greedy** rule of
[De Marchi et al. (2005)](https://doi.org/10.1007/s10444-004-1829-1): selection
maximises the power function without reading `Y`. The Newton basis and
regularised coefficient fit follow
[Santin & Haasdonk (2021)](https://doi.org/10.1515/9783110498967-009).
The VKOGA name comes from
[Wirtz & Haasdonk (2013)](https://doi.org/10.14658/PUPJ-DRNA-2013-Special_Issue-10),
whose target-dependent selection differs from the rule implemented here.

The design-based alternative is [`jaxgsa.kucherenko`](/api/kucherenko), which
evaluates the real model on a conditional-copula design instead of querying a
surrogate.

## Choose training data that covers the integration support

The training data and the target input distribution are separate choices.
Conditional integration queries the surrogate on the target joint support,
including draws from its conditional distributions. Training must represent
that support accurately. An independent, space-filling design can help when
you can evaluate the model throughout the domain, but it is not a requirement
for conditional-variance indices. Correlated training data can be suitable.
Check `cv_rmse` and whether training covers the points integration will query.

The built-in path warns when it detects correlated training data. This is a
coverage diagnostic, not an automatic fit of the integration copula.

## A run

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jaxgsa
from jaxgsa.sampling import monte_carlo

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
rho = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])

X = monte_carlo(problem, 512, seed=0)
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.vkoga.analyze(
    problem.with_correlation(rho), X, Y, key=jax.random.key(0), max_centers=120
)
for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
    print(name, np.asarray(getattr(res, name)).round(4))
```

```
jaxgsa.vkoga.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: correlated (Gaussian copula)
    output: N=512 runs, T=1 x K=1 output slice
    invalid: none found in 512 rows (policy 'raise')
  timing:
    fit + compute: 3.647 s
    n_centers: 120
    gamma: 7.533
    ridge: 0.01
    batch_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by S_TC
    1. x2  S_TC=0.4992
    2. x1  S_TC=0.3475
    3. x3  S_TC=0.0009807

S_TC [0.3475 0.4992 0.001 ]
S_TU [0.4325 0.4251 0.247 ]
S_U  [0.2534 0.4016 0.0009]
S_C  [0.0941 0.0976 0.    ]
S_IU [0.1791 0.0235 0.2461]
```

`x3` has `S_TC = 0.001` and `S_TU = 0.247`. It has essentially no first-order
effect and a large interaction effect, so it is the case where prioritisation
and fixing disagree completely. `x1` and `x2` each pick up about 0.09 to 0.10 of
`S_C`, the share that arrives through their 0.7 correlation with each other.
`x3` correlates with nothing and its `S_C` is 0.

The same problem run through
[`jaxgsa.kucherenko`](/api/kucherenko) on the real model gives
`S1 = [0.331, 0.523, 0.003]` against `S_TC = [0.348, 0.499, 0.001]`, and
`ST = [0.425, 0.423, 0.243]` against `S_TU = [0.433, 0.425, 0.247]`. The two
methods are estimating the same two quantities, one through a surrogate and one
through 28672 model runs.

## Index reference

The correlated/uncorrelated decomposition follows
[Li et al. (2010)](https://doi.org/10.1021/jp9096919), with the surrogate-based
estimators described by
[Hilhorst et al. (2024)](https://doi.org/10.1002/cnm.3797).

Every index has shape `(..., D)`. The leading axes follow the shape contract:
`(D,)` for `Y` of shape `(N,)`, `(K, D)` for `(N, K)`, `(T, K, D)` for
`(N, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S_TC` | $V(E(Y \mid X_i)) / V(Y)$ | Total correlated. What $X_i$ explains through itself plus what it explains through its correlation with the others. Use it to prioritise. |
| `S_TU` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | Total uncorrelated. What only $X_i$ can explain, correlated pathways removed. Use it to decide what can be fixed. |
| `S_U` | independent part of $S_{TC}$ | The contribution of $X_i$ alone. |
| `S_C` | `S_TC - S_U` | The correlation-borne part. It can be negative when a correlation opposes a direct effect. |
| `S_IU` | `S_TU - S_U` | Independent interactions. |

Under independent inputs the five collapse to the familiar picture: `S_TC` is
the first-order Sobol' index $S_1$, `S_TU` is the total index $S_T$, `S_U`
equals `S_TC`, and `S_C` is zero.

## Dependency structure

`correlation` declares the Gaussian copula the indices are computed under.

| Value | Meaning |
| --- | --- |
| `None` (default) | Read `problem.correlation`. Independent (identity) when the problem declares none. |
| `(D, D)` array | Override the problem's declaration for this call. The matrix must be symmetric with a unit diagonal. A non-positive-definite matrix is repaired by clipping its negative eigenvalues and renormalising the diagonal, with a `JaxgsaWarning`. This is a cheap heuristic, not Higham's nearest correlation matrix; the two can disagree by a visible margin. |

To fit a matrix from observed data use
`jaxgsa.sampling.fit_correlation(problem, X_data)` and attach it with
`problem.with_correlation(...)`. Making that a separate step keeps it explicit
which sample the copula came from. A string value raises `ValueError`.

The built-in matrix used is on `result.correlation`; custom sampling has no
copula matrix.

## Custom input dependence

Pass `sampler` to replace the Gaussian copula, with neither a declared
`problem.correlation` nor a `correlation=` override. The callback contract is
[the same as Kucherenko's](/api/kucherenko#custom-input-dependence).
It transforms package-generated uniform points into joint or conditional
physical samples. It must be deterministic for identical arguments, with no
hidden randomness or mutable sampling state. Dependence is not
estimated from `X`.

`X, Y` train the surrogate; the sampler defines the integration distribution.
Its joint marginals must match `Problem`, because the declared marginal CDFs
still define the coordinate transform and component-function basis.

Custom counts are exact. The default `sequence="sobol"` requires power-of-two
`n_outer`, `n_inner`, and `n_variance`; `sequence="monte_carlo"` permits arbitrary
counts. The required `key` controls integration and bootstrap reproducibility.
Each conditional mean uses two independently randomized blocks of `n_inner`
points. Products of their means estimate squared conditional expectations,
without an IID inner-variance correction. Two joint blocks of `n_variance`
points similarly estimate the output variance. This applies to both sequences
and costs roughly twice as many surrogate predictions as a single-block
integration. Finite-sample index estimates can still be slightly negative.
Increasing counts reduces integration error but cannot repair surrogate error.
The internal `_custom_indices.py` computes the same five VKOGA indices under
the supplied distribution; it does not define new sensitivity indices.
The extra `S_U`, `S_C`, `S_IU` fields inherit the fitted additive-component
approximation and do not establish a unique causal decomposition.

See the [runnable custom-dependence example](/examples/correlated-inputs#custom-dependence-without-a-copula).

## Fit and estimator controls

| Argument | Default | What it changes |
| --- | --- | --- |
| `gamma` | `None` | Gaussian RBF shape parameter. `None` cross-validates over a grid. |
| `ridge` | `None` | RKHS regularisation. `None` cross-validates over a grid. |
| `max_centers` | `None` | Cap on greedily selected kernel centres. `None` means 300, itself capped at `N`. Lower it to cut fit time; too low and the surrogate underfits, which `cv_rmse` reports. |
| `n_folds` | `10` | Folds for the hyperparameter cross-validation. At least 2. |
| `n_outer` | `512` | Outer (conditioning) sample size per parameter. Rounded up to the next power of two for Sobol' balance on the built-in path; exact for custom samplers; Sobol counts must be powers of two. Increasing it reduces outer sampling noise. Built-in `S_TC` also has error from a shared inner block that increasing `n_outer` cannot remove; custom sampling uses two independent inner randomizations. |
| `n_inner` | `128` | Inner (conditional) sample size per outer point, rounded up to a power of two on the built-in path (exact for custom samplers; Sobol counts must be powers of two). On the built-in path, `S_TC` drops the i.i.d. inner-noise correction and leans on a large shared inner block, so do not lower `n_inner` to buy speed there: a small `n_inner` inflates a small `S_TC`, and raising `n_outer` will not fix it. |
| `n_variance` | `8192` | Sample size for the output variance and the component-function fit, rounded up to a power of two on the built-in path; exact for custom samplers; Sobol counts must be powers of two. |
| `key` | `None`, but required | A `jax.random` key driving the index integration: Sobol points by default, or Monte Carlo points for custom samplers. Use `jax.random.key(0)` for reproducibility. |
| `sampler` | `None` | Uniform-point transformation implementing joint and conditional sampling; replaces the Gaussian copula. |
| `sampler_dimension` | `None` | Uniform coordinates required by the sampler; defaults to the number of inputs. |
| `sequence` | `"sobol"` | Custom integration point sequence: `"sobol"` or `"monte_carlo"`. |
| `batch_size` | `None` | Rows per device call for surrogate evaluation. Built-in nested sampling groups outer conditioning rows within the memory budget; custom sampling draws one conditioning row's `n_inner` conditional rows at a time. One complete conditional block is the minimum sampling allocation, even when `batch_size < n_inner`. Changing prediction batching does not change the random sample. |
| `on_invalid` | `"raise"` | One training row is one unit, so `"drop"` removes the `(X, Y)` pair. The check runs before the hyperparameter search on purpose: one non-finite `Y` makes every cross-validation score non-finite, and the caller would otherwise meet a `RuntimeError` about failed kernel solves that names the wrong cause. `"drop"` needs at least `n_folds` rows to survive. |
| `verbose` | `True` | Prints the summary block shown above. |

Leaving both `gamma` and `ridge` to cross-validation costs a 10x10 grid of
k-fold refits, which dominates the runtime. Pass both once you know good
values; the fitted values are reported on the result and in the verbose block.

## Confidence intervals

`n_bootstrap=0` is the default and is the right setting for a routine run.
Every replicate refits the kernel surrogate **and** re-runs the nested
conditional integration, so `n_bootstrap=100` costs roughly a hundred analyses.

What the interval measures is total uncertainty in the reported index given this
training sample. The training rows are resampled with replacement, a new
surrogate is fitted to each resample, and a fresh quasi-random stream integrates
against it. It folds the sampling error of `(X, Y)` together with the
Monte-Carlo error of the integration, which is what you want when the question
is "how much would this index move if I had drawn a different training set".
Fitting once and resampling only the integration would instead measure what
`n_outer`/`n_inner` cost you, which is the part you can shrink for free by
raising them.

Two things stay fixed, so the interval is a conditional statement. `gamma` and
`ridge` are the values cross-validated on the full sample and are not
re-selected per replicate. And the interval is about the surrogate's indices: it
cannot see the error the surrogate makes against the true model. `cv_rmse` is
what reports that.

Intervals cover the five index fields only. `variance`, `rmse`, `cv_rmse` and
`n_centers` describe the reported fit itself, so an interval over other fits
would not be about the thing they name.

`conf_level`, `ci_method` and `keep_replicates` behave as on every other method.

## VKOGAResult

Indices: `S_TC`, `S_TU`, `S_U`, `S_C`, `S_IU`, each `(..., D)`, with the
matching `*_conf` fields of shape `(2, ..., D)` when `n_bootstrap > 0`.

Diagnostics:

| Field | Meaning |
| --- | --- |
| `correlation` | The `(D, D)` copula matrix for built-in integration; `None` for custom samplers. |
| `variance` | Output variance under the correlated input measure, per output slice. |
| `n_centers` | Kernel centres in the fitted surrogate. |
| `gamma`, `ridge` | The hyperparameters the fit used, cross-validated or passed. |
| `rmse` | Training-fit RMSE, per output slice. |
| `cv_rmse` | Cross-validated RMSE. This is the one that tells you whether to trust the indices at all. |
| `is_correlated` | Whether built-in integration used a non-identity correlation; `None` for custom sampling. |
| `dependence_source` | `"gaussian"` or `"custom"`. |
| `invalid`, `ci` | The non-finite report and the interval provenance. |

Operations: `result.predict(X_new, batch_size=None)` evaluates the fitted
surrogate, and `result.to_dataset(time_coords=None)` gives the labeled xarray
view.

`result.shapley()` raises `NotImplementedError`. Shapley effects need a variance
decomposition indexed by parameter subsets. A kernel expansion is a sum over
centres and every centre involves every parameter, so there is no membership
matrix to allocate from. Use [`jaxgsa.hdmr`](/api/hdmr) or
[`jaxgsa.pce`](/api/pce) for Shapley effects.

## Enable float64

The coefficient solve forms $A^\top A$, which squares the condition number of
the cross kernel. float32 cannot carry that for a small `gamma`. Call
`jax.config.update("jax_enable_x64", True)` before fitting. `analyze` emits a
`JaxgsaWarning` when x64 is off.

## What it refuses

`analyze` raises `ValueError` when `X`/`Y` violate the shape contract, when
`gamma` or `ridge` is not finite and positive, when a size argument is out of
range, when `correlation` is neither `None` nor a valid matrix, when `key` is
`None`, when `on_invalid` is unknown or refuses the sample, and in two
method-specific cases:

- Fewer than two parameters. Conditioning on the other parameters means nothing
  at `D = 1`.
- Any categorical parameter. The isotropic RBF needs a continuous CDF map per
  coordinate, and a step-CDF coordinate breaks both the kernel metric and the
  copula conditionals. Use [`jaxgsa.optimal_transport`](/api/optimal-transport),
  [`jaxgsa.borgonovo`](/api/borgonovo), [`jaxgsa.pawn`](/api/pawn), or the
  Saltelli-based Sobol pipeline.

It raises `RuntimeError` when every cross-validation score is non-finite. Since
the non-finite check now runs first, that is a genuine solver failure.

It warns in five cases: a zero-variance output slice; float32; a cross-validated
error that is a large fraction of the output standard deviation, which makes
every index untrustworthy; `S_U` clipped to `S_TU` by a wide margin, which says
the additive component functions cannot represent the model's interactions; and
the correlated-training-design case described at the top of this page.

## References

- Hilhorst, P. L. J., Quicken, S., van de Vosse, F. N. & Huberts, W. (2024). [Efficient sensitivity analysis for biomechanical models with correlated inputs](https://doi.org/10.1002/cnm.3797). *International Journal for Numerical Methods in Biomedical Engineering*, 40(2), e3797. Surrogate-based correlated sensitivity analysis.
- Li, G., Rabitz, H., Yelvington, P. E., Oluwole, O. O., Bacon, F., Kolb, C. E. & Schoendorf, J. (2010). [Global sensitivity analysis for systems with independent and/or correlated inputs](https://doi.org/10.1021/jp9096919). *Journal of Physical Chemistry A*, 114(19), 6022–6032. Structural and correlative variance decomposition.
- De Marchi, S., Schaback, R. & Wendland, H. (2005). [Near-optimal data-independent point locations for radial basis function interpolation](https://doi.org/10.1007/s10444-004-1829-1). *Advances in Computational Mathematics*, 23, 317–330. P-greedy centre selection implemented here.
- Santin, G. & Haasdonk, B. (2021). [Kernel methods for surrogate modeling](https://doi.org/10.1515/9783110498967-009). In *Model Order Reduction, Volume 1: System- and Data-Driven Methods and Algorithms*, 311–354. De Gruyter. Newton basis and regularised kernel approximation.
- Wirtz, D. & Haasdonk, B. (2013). [A vectorial kernel orthogonal greedy algorithm](https://doi.org/10.14658/PUPJ-DRNA-2013-Special_Issue-10). *Dolomites Research Notes on Approximation*, 6, 83–100. Source of the VKOGA name and a target-dependent selection variant.

See the [VKOGA example](/examples/vkoga), [Methods](/guide/methods), and the
[API overview](/api/).
