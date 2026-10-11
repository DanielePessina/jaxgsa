# Kucherenko (dependent-input Sobol' indices)

```python
jaxgsa.kucherenko.sample(problem, n_samples=None, *, base_n=None,
                         subsets="singletons", scramble=True, seed=None,
                         verbose=True)
                         -> KucherenkoSamples

jaxgsa.kucherenko.analyze(samples, Y, *, n_bootstrap=0, conf_level=0.95,
                          ci_method="quantile", key=None, on_invalid="raise",
                          verbose=True, keep_replicates=False)
                          -> KucherenkoResult | IrregularResult
```

`jaxgsa.kucherenko` estimates the Sobol' indices generalised to dependent
inputs ([Kucherenko, Tarantola & Annoni, 2012](https://doi.org/10.1016/j.cpc.2011.12.020)).
It runs the estimators on your
actual model outputs, evaluated on a conditional-copula design. No surrogate is
fitted anywhere. That makes it the design-based counterpart to
[`jaxgsa.vkoga`](/api/vkoga), which reaches the same two quantities by querying
a fitted kernel surrogate instead.

The trade is straightforward. Kucherenko costs `base_n * (2D + 1)` model
evaluations (`base_n * 2^D` with `subsets="all"`) and has no surrogate error. VKOGA costs one `(X, Y)` sample of any
size and inherits whatever the surrogate gets wrong.

`analyze` also accepts ragged outputs. Pass a list or dict of `(times, values)`
pairs; each channel is analyzed on its own grid. See
[Irregular output grids](/examples/irregular-outputs).

:::: warning These indices are not comparable to `jaxgsa.sobol`'s
Under a declared correlation, `kucherenko.S1` is a different estimand from
`sobol.S1`, not the same estimand computed a different way. It is
correlation-inclusive: it counts what $X_i$ explains through its coupling with
the other parameters as well as what it explains alone. `ST` is
correlation-exclusive. `ST >= S1` does not hold. Do not read a `kucherenko`
index against a `sobol` index in the same table or plot unless the problem is
independent, in which case the two estimate the same thing.
::::

::: info On the name
Kucherenko calls these Sobol' sensitivity indices generalised to dependent
inputs; the 2012 paper is titled for the general problem, not for its author.
The module is named `kucherenko` because that is the handle
[UQLab](https://www.uqlab.com/) and [UQpy](https://uqpyproject.readthedocs.io/)
use for this estimator pair, and because a name reading as `sobol` would invite
the comparison the box above warns against. It is one of several competing
generalisations of Sobol' indices to dependent inputs. See
[the dependent-input guide](/guide/dependent-inputs) for
how it differs from VKOGA, HDMR's ANCOVA split, and the ANCOVA Shapley
allocation.
:::

`sample` warns when the problem declares no dependence (`problem.dependence`
is `"independent"`). The conditional redraws
are plain independent draws in that case, so the design collapses to a Saltelli
column-swap scheme and you pay `base_n * (2D + 1)` runs for indices
[`jaxgsa.sobol`](/api/sobol) gives from `base_n * (D + 2)`, with `S2` available
for one more block. Run it on an independent problem only to cross-check a
correlated analysis.

## A run

```python
import numpy as np
import jaxgsa

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
rho = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
corr_problem = problem.with_correlation(rho)

ks = jaxgsa.kucherenko.sample(corr_problem, base_n=4096, seed=0)
X = ks.samples
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.kucherenko.analyze(ks, Y)
print(np.asarray(res.S1).round(4))
print(np.asarray(res.ST).round(4))
```

```
jaxgsa.kucherenko.sample: D=3, base_n=4096, n_blocks=7, n_runs=28672, dependence=gaussian, scramble=True
jaxgsa.kucherenko.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: gaussian copula
    output: N=28672 runs, T=1 x K=1 output slice
    invalid: none found in 4096 base points (policy 'raise')
  timing:
    compute: 0.05247 s
    design: gaussian (2D+1 = 7 blocks of 4096 base points)
  results: top 3 of 3 parameters by ST
    1. x2  ST=0.425
    2. x1  ST=0.4115
    3. x3  ST=0.2431

[ 0.3252  0.5248 -0.0007]
[0.4115 0.425  0.2431]
```

Read `x2` first: `S1 = 0.52` against `ST = 0.43`. Under independence that
ordering is impossible. Here it happens because `S1` collects everything `x2`
explains including what arrives through its 0.7 correlation with `x1`, while
`ST` counts only what `x2` alone can explain. Then read `x3`, which is the
opposite case: `S1 = -0.0007` (zero within noise) and `ST = 0.24`, a parameter with no first-order
effect and a large interaction effect.

Both `sample` and `analyze` print because `verbose=True` is the default on both.

## Index reference

Every index has shape `(..., D)`. The leading axes follow the shape contract:
`(D,)` for `Y` of shape `(n_runs,)`, `(K, D)` for `(n_runs, K)`, `(T, K, D)` for
`(n_runs, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S1` | $V(E(Y \mid X_i)) / V(Y)$ | Correlation-inclusive first order. What $X_i$ explains through itself plus what it explains through its coupling. Equals VKOGA's `S_TC`. |
| `ST` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | Correlation-exclusive total. What only $X_i$ can explain. Equals VKOGA's `S_TU`. |

Under independent inputs both reduce exactly to the classic Sobol' `S1` and
`ST`. Under correlation `ST >= S1` no longer holds, as the run above shows.

## The design

`sample` builds `base_n * (2D + 1)` rows, for built-in and custom dependence
alike. You set the size in one of two ways, the same as
[`jaxgsa.sobol.sample`](/api/sobol). `n_samples` is the minimum number of model
runs. `sample` picks the smallest power-of-two `base_n` whose design reaches
it. `base_n=` sets the base points per block exactly, and it must be a power of
two. The Sobol' sequence keeps its balance guarantees only at powers of two.
At `D = 3` the design has 7 blocks, so `n_samples=28672` and `base_n=4096` give
the same design, and `n_samples=28673` gives `base_n=8192`. Give exactly one of
the two. A `base_n` below 16 warns that the design is degenerate.

With `subsets="all"` the layout is different. See
[Shapley effects and PME](#shapley-effects-and-pme).

The design holds one joint block, then two blocks per parameter. The first keeps
$X_i$ and redraws the rest from $p(\mathbf{X}_{\sim i} \mid X_i)$. The second
keeps the rest and redraws $X_i$ from $p(X_i \mid \mathbf{X}_{\sim i})$. Both
conditionals are closed-form Gaussians in the latent copula space. A custom
sampler supplies the same conditional draws itself.

The dependence comes from the problem: `problem.correlation` (a Gaussian
copula) or `problem.sampler` (a custom sampler, see
[below](#custom-input-dependence)). This sampler is deliberately
exempt from the correlated-design error that `sobol`, `morris` and `efast`
raise, because conditioning on the declared copula is the whole point of the
method. With no declared dependence the design is exactly the Saltelli
column-swap scheme and the analysis reproduces the classic Sobol' indices.

| Argument | Default | What it changes |
| --- | --- | --- |
| `subsets` | `"singletons"` | `"singletons"` builds the `2D + 1` block design. `"all"` builds `2^D` blocks for exact Shapley effects and PME. See [Shapley effects and PME](#shapley-effects-and-pme). |
| `n_samples` | `None` | Minimum number of model runs. The smallest power-of-two `base_n` whose design reaches it is used. Give exactly one of `n_samples` and `base_n`. |
| `base_n` | `None` | Base points per block, a power of two, at least 2. It is the only knob on accuracy: everything else in the design is fixed by `D`. Give exactly one of `n_samples` and `base_n`. |
| `scramble` | `True` | Owen-scrambles the Sobol' sequence. Keep it on, because it is what makes different seeds give statistically independent designs. Off, the sequence is deterministic and repeated runs are identical. |
| `seed` | `None` | `int`, `np.random.Generator`, or `None`. It seeds the scrambling only, so it does nothing with `scramble=False` and passing both raises `ValueError`. The random streams changed in this release, so a seed does not reproduce designs from earlier versions. |
| `verbose` | `True` | Prints the one-line design summary shown above. |

`sample` raises `ValueError` when the problem declares a categorical parameter
(the conditional copula needs continuous marginals; use
[`jaxgsa.optimal_transport`](/api/optimal-transport),
[`jaxgsa.borgonovo`](/api/borgonovo), [`jaxgsa.pawn`](/api/pawn), or the
Saltelli-based Sobol pipeline), when the problem has fewer than two parameters,
when both or neither of `n_samples` and `base_n` are given, when `base_n` is not a power of two, and when `n_samples < 2` or `base_n < 2`.

## Shapley effects and PME

The default design gives one number per input, `S1` and `ST`. Under
dependence these two numbers can disagree a lot. Sometimes you need one share
per input that sums to 1. Draw the design with `subsets="all"` for that.

```python
ks = jaxgsa.kucherenko.sample(problem, base_n=4096, subsets="all", seed=0)
res = jaxgsa.kucherenko.analyze(ks, Y)
res.shapley()   # exact Shapley effects
res.pme()       # proportional marginal effects
```

The `"all"` design holds one block per subset of inputs. Each block keeps a
subset $u$ of the inputs and redraws the others from their conditional law
given $u$. The design has `base_n * 2^D` rows. It also holds a second,
independent joint block, which only pools the variance estimate. The design
refuses `D < 2` and `D > 12`. At `D = 12` and `base_n = 1024` the design has
4 million rows. Plan the memory before you ask for it.

From each block the analysis reads two costs for a subset $u$:

- the closed index $\mathrm{Var}(\mathbb{E}(Y \mid X_u)) / \mathrm{Var}(Y)$, from
  the same single-loop product as `S1`;
- the total index $\mathbb{E}(\mathrm{Var}(Y \mid X_{\sim u})) / \mathrm{Var}(Y)$ of
  the complement, from the same Jansen difference as `ST`.

Both are conditional variances under the joint input law. The two costs give
the same Shapley effects in theory. They differ by sampling noise only.

| Field or method | Meaning |
| --- | --- |
| `closed` | Closed index of every subset, shape `(..., 2^D - 1)`. |
| `subset_total` | Total index of every subset. It is not `ST`. The two agree for single inputs only. |
| `subsets` | Boolean membership matrix, shape `(2^D - 1, D)`. Subset with bitmask `m` (bit `j` is input `j`) sits at row `m - 1`. The last row is the full set, with index exactly 1. |
| `closed_conf`, `subset_total_conf` | `(2, ..., 2^D - 1)` intervals, `None` when `n_bootstrap=0`. |
| `shapley(cost="total")` | Exact Shapley effects ([Owen, 2014](#ref-owen-2014)). Use `cost="closed"` for the closed cost. |
| `pme()` | Proportional marginal effects ([Herin et al., 2024](#ref-herin-2024)). |

`S1` and `ST` of an `"all"` design come from the singleton blocks inside it.
`shapley()` and `pme()` return a `ShapleyResult` with `backend="exact"`.
`explained_variance` is 1 because no surrogate is fitted. For `pme()`, the
`Sh` field holds the PME values. They are not Shapley effects.

Intervals need `n_bootstrap > 0` and `keep_replicates=True`. Each kept draw is
transformed, and the interval comes from the transformed draws. Without kept
draws, `Sh_conf` is NaN. A call to `shapley()` or `pme()` on a result from the
default design raises `ValueError`.

### Shapley or PME

Both methods give nonnegative shares that sum to 1. They treat an input that
the model does not use in different ways.

Take $Y = X_1 + 0.5 X_3$ with $X_2$ correlated with $X_1$ ($\rho = 0.8$).
The model never reads $X_2$.

```python
problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-1.0, 1.0),) * 3)
rho = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
problem = problem.with_correlation(rho)

ks = jaxgsa.kucherenko.sample(problem, base_n=4096, subsets="all", seed=0)
Y = ks.samples[:, 0] + 0.5 * ks.samples[:, 2]
res = jaxgsa.kucherenko.analyze(ks, Y, verbose=False)
print(np.asarray(res.S1).round(3))
print(np.asarray(res.ST).round(3))
print(np.asarray(res.shapley().Sh).round(3))
print(np.asarray(res.pme().Sh).round(3))
```

```
[0.8   0.498 0.2  ]
[0.302 0.    0.2  ]
[0.552 0.248 0.201]
[0.8 0.  0.2]
```

- `S1` gives $X_2$ half of the variance, because $X_2$ carries information
  about $X_1$.
- `ST` gives $X_2$ zero, because nothing is left once the others are known.
- Shapley gives $X_2$ a share of 0.25. It shares the variance of $X_1$ between
  $X_1$ and $X_2$ through their correlation.
- PME gives $X_2$ exactly zero. It gives all of the variance to the inputs
  that the model uses.

Use Shapley when you want to know who explains the output, including by
correlation. Use PME when you want to know which inputs the model uses. PME
gives zero to an input the model does not read. Shapley does not.

PME detects a null input as one whose cost is exactly zero. The Jansen
estimator gives exactly zero when the model ignores the redrawn inputs. An
input that the model uses only a little gets a small positive PME. There is
no tolerance argument.

Shapley and PME of an `"all"` design use the same total cost. The
[Plischke, Rabitti and Borgonovo (2021)](#ref-plischke-2021) paper compares
other ways to estimate Shapley effects. [Iooss and Prieur (2019)](#ref-iooss-2019)
compare Shapley effects with Sobol' indices under dependence.

## Custom input dependence

A problem can carry a custom sampler. Declare it when you build the problem, or
attach it afterwards:

```python
problem = jaxgsa.Problem(names=..., bounds=..., sampler=f, sampler_dimension=2)
problem = jaxgsa.Problem.from_dict(..., sampler=f, sampler_dimension=2)
problem = problem.with_sampler(f, sampler_dimension=2)   # copy with a sampler
problem = problem.with_sampler(None)             # copy without it
```

`sampler_dimension` is the number of uniform coordinates per row. It defaults
to `D`. A problem has either a Gaussian-copula `correlation` or a `sampler`,
never both. Declaring both raises `ValueError`. `problem.dependence` tells you
which one is active: `"independent"`, `"gaussian"` or `"custom"`.

Kucherenko `sample` and VKOGA `analyze` read the sampler from the problem. Its
type is `jaxgsa.sampling.InputSampler`:

```python
sampler(uniform_points, *, conditioning_rows=None, fixed_indices=())
```

The package supplies `(n, sampler_dimension)` uniform points. Return `(n, D)`
finite real physical inputs in `problem.names` order. The default design makes exactly
three kinds of call. A `subsets="all"` design adds a fourth:

| `fixed_indices` | `conditioning_rows` | What to return |
| --- | --- | --- |
| `()` | `None` | Joint draws. |
| `(i,)` | physical rows | Keep column `i` of `conditioning_rows`. Draw the other columns from their conditional given `x_i`. |
| every index except `i` | physical rows | Keep those columns. Draw `x_i` from its conditional given the rest. |
| any sorted subset `u` (`subsets="all"` only) | physical rows | Keep the columns in `u`. Draw the others from their joint conditional given `x_u`. |

Copy the fixed columns exactly. Overwriting fixed columns after a joint draw is
valid only for independent inputs. The package passes copies, so changing the
arguments in place does no harm.

For `subsets="all"`, the sampler must be a deterministic inverse Rosenblatt
map ([Rosenblatt, 1952](/guide/dependent-inputs#references)). That is a
conditional-quantile map. It sends the uniform points to the free columns
through the conditional law given the fixed columns. Rejection sampling and
MCMC do not qualify. The free columns must not depend on hidden state.

The callback must be deterministic: identical arguments give identical outputs.
Do not use hidden randomness or mutable sampling state. All randomness comes
from the package. Kucherenko's `seed` and `scramble`, or VKOGA's required
`key`, control the uniform points. Scrambled Sobol' points are the only
source, and sample sizes round up to the next power of two. The package
assembles the analysis blocks. Shapes, finite values and fixed columns are
checked. Statistical consistency is your responsibility. VKOGA's
`correlation=` override cannot be combined with a custom sampler.

For given-data methods, draw `X` with `jaxgsa.sampling.monte_carlo(problem, n_samples)`.
It sends pseudo-random uniforms through the joint branch of your sampler. This
does not turn an ordinary `X` matrix into a Kucherenko analysis design.
`sobol`, `morris`, `efast` and the independence-only analyses refuse a
custom-sampler problem, exactly as they refuse a correlated one. Saved designs
retain sampled rows and provenance, not callable code. VKOGA's declared
marginals must match the sampler, because its coordinate transform and
component bases still use them. Training data must cover the analysis
distribution (see
[VKOGA training data](/examples/vkoga#training-data-must-cover-the-analysis-distribution)).
Dependence fitting remains explicit, never triggered just by supplying data.

See the [complete example](/examples/correlated-inputs#custom-dependence-without-a-copula).

## Analysis

`analyze` applies the single-loop estimators of
[Kucherenko, Tarantola & Annoni (2012)](https://doi.org/10.1016/j.cpc.2011.12.020): the paired
product over the shared-$X_i$ rows for `S1`, and the
[Jansen (1999)](https://doi.org/10.1016/S0010-4655(98)00154-4) squared difference
over the shared-$\mathbf{X}_{\sim i}$ rows for `ST`. The exact formulas are in
the `jaxgsa.kucherenko._analyze` module docstring.

The resampling unit for `n_bootstrap` is one **base point**, which carries the
joint row and all `2D` conditional rows drawn around it. Monte Carlo base
points are independent. Sobol base points are dependent, so this row bootstrap
does not estimate randomized-QMC error; compare independently scrambled
complete designs for that purpose. Resampling
individual rows would not be, because dropping part of a conditional block
leaves the estimator reading rows that no longer line up. `S1` and `ST` get
intervals; `variance` does not, since it is the shared denominator of both and
its uncertainty is already inside the two index intervals.

`on_invalid` works on the same unit. A base point appears once in the joint
block and once in each of the `2D` conditional blocks, so one non-finite value
removes all `2D + 1` of its rows. `"raise"` is the default, `"drop"` analyzes
the surviving base points and warns, `"propagate"` lets the value reach the
indices. Base points are not contiguous in the array: base point `k` occupies
rows `k`, `N + k`, `2N + k` and so on, and `result.invalid` reports them. See
[Failed model runs](/api/#failed-model-runs).

`analyze` raises `ValueError` on a `Y` that violates the shape contract, on an
unknown `on_invalid`, on a non-finite sample under `"raise"`, when fewer than 2
base points survive a drop, and on the usual bootstrap-argument checks
(`n_bootstrap < 0`, `conf_level` outside `(0, 1)`, an unknown `ci_method`, or
`n_bootstrap > 0` with no `key`).

It warns on an output slice with zero variance. Its indices come back as NaN.
Drop that slice, or widen the input ranges so the output varies.

## KucherenkoResult

| Field | Meaning |
| --- | --- |
| `S1`, `ST` | The two index arrays, `(..., D)`. |
| `S1_conf`, `ST_conf` | `(2, ..., D)` intervals, `None` when `n_bootstrap=0`. |
| `variance` | Output variance under the joint input measure, per output slice. |
| `closed`, `subset_total`, `subsets` | Subset indices and their membership matrix. `None` unless the design used `subsets="all"`. See [Shapley effects and PME](#shapley-effects-and-pme). |
| `dependence` | How the inputs depend on each other: `"independent"`, `"gaussian"` or `"custom"`. |
| `invalid` | What the non-finite check found and which policy ran. |
| `ci` | Confidence level, endpoint rule, resample count, and the draws when `keep_replicates=True`. |

`result.to_dataset(time_coords=None)` gives the labeled xarray view of `S1`,
`ST` and `variance`. With subset indices it adds `closed` and `subset_total` on
a `subset` dimension, labelled like `x1+x3`, and a boolean `subset_members`.

## Aggregating outputs

```python
result.aggregate(*, weights="variance")
```

`aggregate` collapses the `T` and `K` axes into one index per input. With
`weights="variance"` each output counts in proportion to its variance. This is
the generalised Sobol' index of [Gamboa, Janon, Klein and Lagnoux
(2014)](#ref-gamboa-2014), also used by [Lamboni, Monod and Makowski
(2011)](#ref-lamboni-2011). It is the share of the total output variance that
the input explains. With `weights="equal"` every output counts the same.

The result keeps its class. Its output axes have size 1. A constant output has
weight 0. A result with one output slice comes back unchanged. When subset
indices are present they collapse with the same weights, so `shapley()` of the
aggregated result equals the variance-weighted mean of the per-output Shapley
effects.

Intervals do not aggregate bound by bound. With `keep_replicates=True`, each
kept draw is aggregated and the interval is drawn again. Otherwise the
intervals are NaN. The weights stay fixed at their point estimates, so the
interval leaves out the noise in the weights. A ragged result
(`IrregularResult`) has no `aggregate`; aggregate each channel instead.

## KucherenkoSamples

`samples` is the `(base_n * (2D + 1), D)` design in physical units, stored as
`2D + 1` stacked blocks of `base_n` rows: the joint block first, then one
conditional block per parameter for `S1`, then one per parameter for `ST`.
`n_runs`, `base_n`, `n_params` and `problem` describe it.

The design keeps all rows, including repeated points under a deterministic custom sampler: `n_expanded == n_runs`, `expanded_to_unique` is the identity,
and `expand_outputs(Y)` returns `Y` unchanged. The shared
`UniqueDesignSamples` base is here for the NPZ format and the metadata schema,
not for deduplication.

`KucherenkoSamples.save(path)` and `KucherenkoSamples.load(path)` use the same
one-file NPZ format as the other design classes. The stored problem metadata
carries the sampling provenance, so a loaded design analyzes identically.
The file does not store a custom sampler function. The sampled rows and their
outputs are enough for the analysis. A loaded custom problem still reports
`dependence == "custom"`, so the independence checks still refuse it. It
raises if you ask it to draw new rows. To draw a new design, attach the
sampler again with `problem.with_sampler(f, sampler_dimension=k)`.

## References

- **Dependent-input indices and conditional-copula sampling:** Kucherenko, S.,
  Tarantola, S., & Annoni, P. (2012).
  [Estimation of global sensitivity indices for models with dependent variables](https://doi.org/10.1016/j.cpc.2011.12.020).
  *Computer Physics Communications*, 183(4), 937–946.
- **Squared-difference total-order estimator:** Jansen, M. J. W. (1999).
  [Analysis of variance designs for model output](https://doi.org/10.1016/S0010-4655(98)00154-4).
  *Computer Physics Communications*, 117(1–2), 35–43.
- <a id="ref-owen-2014"></a>**Shapley value of Sobol' indices:** Owen, A. B. (2014).
  [Sobol' indices and Shapley value](https://doi.org/10.1137/130936233).
  *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245–251.
- <a id="ref-song-2016"></a>**Shapley effects and the total cost:** Song, E., Nelson, B. L.,
  & Staum, J. (2016).
  [Shapley effects for global sensitivity analysis: theory and computation](https://doi.org/10.1137/15M1048070).
  *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060–1083.
- <a id="ref-iooss-2019"></a>**Shapley effects with correlated inputs:** Iooss, B., &
  Prieur, C. (2019).
  [Shapley effects for sensitivity analysis with correlated inputs: comparisons with Sobol' indices, numerical estimation and applications](https://doi.org/10.1615/Int.J.UncertaintyQuantification.2019028372).
  *International Journal for Uncertainty Quantification*, 9(5), 493–514.
- <a id="ref-plischke-2021"></a>**Computing Shapley effects:** Plischke, E.,
  Rabitti, G., & Borgonovo, E. (2021).
  [Computing Shapley effects for sensitivity analysis](https://doi.org/10.1137/19M1304738).
  *SIAM/ASA Journal on Uncertainty Quantification*, 9(4), 1411–1437.
- <a id="ref-herin-2024"></a>**Proportional marginal effects:** Herin, M.,
  Il Idrissi, M., Chabridon, V., & Iooss, B. (2024).
  [Proportional marginal effects for global sensitivity analysis](https://doi.org/10.1137/22M153032X).
  *SIAM/ASA Journal on Uncertainty Quantification*, 12(2), 667–692.
- <a id="ref-gamboa-2014"></a>**Aggregated indices for multiple outputs:** Gamboa, F.,
  Janon, A., Klein, T., & Lagnoux, A. (2014).
  [Sensitivity analysis for multidimensional and functional outputs](https://doi.org/10.1214/14-EJS895).
  *Electronic Journal of Statistics*, 8(1), 575–603.
- <a id="ref-lamboni-2011"></a>**Aggregated indices for dynamic models:** Lamboni, M.,
  Monod, H., & Makowski, D. (2011).
  [Multivariate sensitivity analysis to measure global contribution of input factors in dynamic models](https://doi.org/10.1016/j.ress.2010.12.002).
  *Reliability Engineering & System Safety*, 96(4), 450–459.

See the [Kucherenko example](/examples/kucherenko), [Methods](/guide/methods),
and the [API overview](/api/).
