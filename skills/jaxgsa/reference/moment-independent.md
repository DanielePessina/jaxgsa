# Moment-independent methods

Four methods that measure how strongly a parameter affects the whole output
distribution rather than its variance. Use them when the output is skewed,
bimodal or heavy-tailed, so `Var(Y)` is a poor denominator, or when you want a
statement variance cannot make.

All four are given-data methods: they take any aligned `(X, Y)` pairs and need
no design. All four accept a correlated problem. `borgonovo`,
`optimal_transport` and `pawn` accept categorical parameters; `hsic` does not.

Their indices are correlation-inclusive. A parameter the model never reads
scores above zero when it correlates with one the model does read. That is the
correct reading of these indices, not an estimation error. For the split between
direct and correlation-borne influence, see `dependent-inputs.md`.

The examples use Ishigami:

```python
import jax
import jax.numpy as jnp
import numpy as np
import jaxgsa
from jaxgsa.benchmarks import ishigami

PROBLEM = ishigami.PROBLEM
X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=4000, seed=0))
Y = ishigami.evaluate(X)
```

The four rank Ishigami the same way and space it differently, which is the point
worth absorbing before picking one:

```
pawn      [0.2484, 0.4022, 0.0868]   KS distance
delta     [0.2149, 0.3387, 0.1606]   half L1 between densities
ot        [0.2013, 0.2775, 0.0977]   normalized W2 squared
R2_HSIC   [0.135,  0.008,  0.025 ]   kernel dependence, at bandwidth 1.0
```

None of them is "the" answer, and the HSIC row is a different number entirely,
for the reason in its section.

## borgonovo

The delta index: the expected L1 distance between the unconditional output
density and the density conditional on one parameter, halved so it lies in
[0, 1].

```python
result = jaxgsa.borgonovo.analyze(PROBLEM, X, Y)

result.delta   # [0.2149, 0.3387, 0.1606]
result.S1      # [0.3057, 0.4208, 0.0026], the given-data first-order Sobol index
```

`delta` and `S1` come from the same class partition at no extra cost, and the
gap between them says a parameter has influence the first-order variance does
not see. Here `x3` has `S1 = 0.003` and `delta = 0.161`.

Signature:

```python
jaxgsa.borgonovo.analyze(problem, X, Y, *, n_classes=None, grid_size=100,
                         bandwidth="silverman", n_bootstrap=0, conf_level=0.95,
                         ci_method="quantile", bias_correct=None, key=None,
                         slice_chunk_size=None, degenerate_tol=...,
                         degenerate_bandwidth="auto", on_invalid="raise",
                         verbose=True, keep_replicates=False)
```

Keep in mind:

- **Continuous outputs only.** `analyze` raises when a column takes at most 20
  distinct values and each repeats at least 5 times on average, because a
  discrete output has atoms no density grid resolves. Use `pawn` or
  `optimal_transport` there. Categorical *parameters* stay supported; the limit
  is on the output.
- If the estimate leaves [0, 1] by more than 0.05 the computation failed and
  `analyze` raises, naming the parameter and the knob that applies. The value is
  never clipped, because a clipped value looks plausible and is still wrong. A
  confidence bound outside the range only warns.
- `n_bootstrap > 0` with `bias_correct` not `False` applies the bias correction,
  including under the `bias_correct=None` default, which warns once per process
  that it did. The corrected estimate can fall marginally below 0 for weak
  parameters. Set `n_bootstrap=0` for the raw plug-in estimate.
- Below about 500 samples, read the ranking and ignore the magnitudes. The
  plug-in estimate is biased upward and the correction can push weak parameters
  below zero.
- `degenerate_tol` says when a conditioning class counts as degenerate and
  `degenerate_bandwidth` says how wide a kernel it gets. The `"auto"` default
  floors it at `max(0.1 * h_full, grid_step)`, never below what the output grid
  can integrate. Raising `degenerate_tol` calls more classes degenerate and can
  hand them a narrower kernel than they had, which biases delta; `analyze` warns
  when the floor fires.

## optimal_transport

A normalized squared 2-Wasserstein distance in [0, 1], split exactly into a
mean-shift part and a spread-and-shape part. This is the only method here that
says *how* a parameter matters.

```python
result = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y, dummy=True,
                                          key=jax.random.key(0))

result.ot           # [0.2013, 0.2775, 0.0977]
result.advective    # [0.1536, 0.2198, 0.0037]   mean shift
result.diffusive    # [0.0477, 0.0577, 0.0940]   spread and shape
result.S1           # 2 * advective, up to an N/(N-1) factor
result.ot_dummy     # [0.0095, 0.0095, 0.0095]   the irrelevance floor
result.above_dummy  # [0.1918, 0.2681, 0.0883]   ot minus that floor
```

Read `x3`: 0.004 of mean shift against 0.094 of shape change. `x3` changes the
spread of the output without moving its mean. No variance-based index says that.
Compare `x1`, which is mostly mean shift.

`ot_dummy` is the index of a synthetic, provably independent column pushed
through the same estimator, so it is the floor below which a value means
nothing. All three read 0.0095 here because every Ishigami parameter shares the
same marginal and class count. `x3`'s 0.098 is ten times its own floor and is
real.

Three modes:

```python
analyze(problem, X, Y, mode="univariate")    # per output column (default)
analyze(problem, X, Y, mode="multivariate")  # one index over the flattened output
analyze(problem, X, Y, mode="trajectory")    # one index per output over the time course
```

`"trajectory"` requires a 3-D `(N, T, K)` `Y` and scores each parameter against
the entire time course jointly, which is how you get one number per parameter
for a whole trajectory.

Signature:

```python
jaxgsa.optimal_transport.analyze(problem, X, Y, *, mode="univariate",
                                 n_partitions=None, standardize_outputs=True,
                                 epsilon=0.03, max_iter=2000, tol=None,
                                 dummy=False, n_bootstrap=0, conf_level=0.95,
                                 ci_method="quantile", key=None,
                                 slice_chunk_size=None, on_invalid="raise",
                                 verbose=True, keep_replicates=False)
```

Keep in mind:

- **In a point-cloud mode, pass `dummy=True`.** `"multivariate"` and
  `"trajectory"` solve entropic transport, whose bias holds an irrelevant
  parameter's index above zero however large `N` is. Reading those indices
  without the floor will make you believe in parameters that do nothing, so
  `analyze` warns when either mode runs with `dummy=False`. In `"univariate"`
  mode the floor is small and the warning does not fire.
- Cost in a point-cloud mode is `(n_bootstrap + 1) * D * n_partitions` Sinkhorn
  solves, times `K` in `"trajectory"` mode, plus one dummy pass. At 100
  replicates, 10 parameters and 25 partitions that is over 25,000 solves. A
  categorical parameter costs its own level count instead of `n_partitions`.
- One parameter at a time, so no `S2` and no total order. The diffusive part
  says influence exists beyond the mean shift; it does not say which parameter
  it is shared with.
- `analyze` warns when Sinkhorn solves did not reach `tol` within `max_iter`,
  and the results then use the last iterate. Raise `max_iter` or `epsilon`.

## pawn

The Kolmogorov-Smirnov distance between the unconditional output CDF and the
CDF conditional on one parameter, aggregated over conditioning bins.

```python
result = jaxgsa.pawn.analyze(PROBLEM, X, Y)

result.pawn           # [0.248, 0.402, 0.087]
result.n_valid_bins   # [10, 10, 10]   <- check this
```

All 10 bins survived for all three parameters, so those indices stand on the
full sample.

Signature:

```python
jaxgsa.pawn.analyze(problem, X, Y, *, n_bins=10, statistic="median",
                    n_bootstrap=0, conf_level=0.95, ci_method="quantile",
                    key=None, slice_chunk_size=None, on_invalid="raise",
                    verbose=True, keep_replicates=False)
```

`statistic` is `"median"`, `"max"` or `"mean"` over the per-bin KS distances.

Keep in mind:

- `n_bins` trades conditioning resolution against sample density per bin. Bins
  holding fewer than 2 samples are dropped, `n_valid_bins` counts what is left,
  and `analyze` warns when a parameter keeps fewer than half its bins. A
  parameter down to 3 bins has a median over 3 numbers.
- Bins are equal-probability on the marginal's own CDF, so a skewed marginal
  does not by itself starve a tail bin. What empties one is a small `N`, a large
  `n_bins`, or samples that land outside the declared marginal.
- Categorical parameters need no binning. The level code already names the
  conditioning class, so PAWN uses one bin per level, `n_bins` does not apply,
  and relabelling the levels gives the same number.
- The KS distance is a supremum, so it reacts to the single largest gap between
  two CDFs and ignores everything else. It is the sharper instrument when a
  parameter moves one part of the range a lot. `delta` or the OT index sees more
  of a small shift across the whole distribution.
- First-order only. No total order, no `S2`.

## hsic

A kernel dependence measure with a permutation test. It answers "is this
parameter doing anything at all" better than it answers "how much".

```python
jax.config.update("jax_enable_x64", True)      # before the analysis

result = jaxgsa.hsic.analyze(PROBLEM, X, Y, n_perms=200, key=jax.random.key(0))

result.R2_HSIC    # [0.135, 0.008, 0.025]   first-order, at bandwidth 1.0
result.T_HSIC     # total, counts interactions
result.p_values   # permutation p-values, the uncertainty statement
result.hsic_raw   # unnormalized, kernel- and scale-dependent
result.bandwidth  # 1.0, carried because the index depends on it
result.n_perms    # 200
```

`key` is required: the permutation test always runs and there is no
`n_bootstrap=0` equivalent that skips it.

The bandwidth sweep is the thing to internalise. On Ishigami at 2000 samples in
float64:

```
bandwidth   R2_HSIC
0.25        [0.058, 0.111, 0.025]     ranks x2 > x1 > x3
0.5         [0.085, 0.070, 0.028]
1.0         [0.135, 0.008, 0.025]     ranks x1 > x3 > x2
2.0         [0.177, 0.002, 0.009]
```

At 0.25 the ranking agrees with `S1 = [0.314, 0.442, 0.0]`. At the default 1.0
`x2` has dropped to 0.008 despite owning 44% of the output variance, because a
wide kernel smooths the twice-oscillating `7 sin^2(x2)` term into a
near-constant.

Signature:

```python
jaxgsa.hsic.analyze(problem, X, Y, *, n_perms=200, key=None, bandwidth=1.0,
                    on_invalid="raise", verbose=True)
```

Keep in mind:

- **Use float64.** The V-statistic cancels three large sums against each other,
  so float32 leaves three or four correct digits and the index changes with the
  order of the sample rows. `analyze` warns.
- **Sweep `bandwidth` before reporting a ranking, and say which value you
  used.** It multiplies the median-heuristic length scale, and as the table
  shows it changes which parameter comes first. The result carries `bandwidth`
  and `n_perms`, and `to_dataset()` writes both into the dataset attributes, so
  a stored index says what produced it.
- Time and memory are O(N squared): about `2D+1` matrices of shape `(N, N)`.
  Nothing chunks it, and `batch_size` raises `TypeError`. Above N of about
  20,000 it is impractical, since N=20,000 with D=5 in float64 is roughly 35 GB.
  Reduce `N`, or screen with a cheaper method first.
- `R2_HSIC` has no units, does not sum to 1, and moves with the bandwidth. For a
  magnitude on a fixed [0, 1] scale use `optimal_transport` or `borgonovo`.
- There is no bootstrap, deliberately. A row bootstrap duplicates rows onto the
  kernel diagonal, where the kernel is exactly 1, which biases the resampled
  index upward by construction. The `p_values` are the uncertainty statement.
- Outputs of extreme magnitude can overflow float32 in the squared distances.
  Rescale with `(Y - Y.mean(0)) / Y.std(0)`, which changes nothing else.
- The `T_HSIC` minus `R2_HSIC` gap says interactions exist, not which pairs.
- Categorical parameters are refused: the RBF kernel would read level codes as
  distances.
