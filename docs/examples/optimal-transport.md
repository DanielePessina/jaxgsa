# Optimal transport: how far knowing an input moves the output distribution

The optimal-transport index asks what it would cost to move the output
distribution you see when an input is free onto the output distribution you
see when it is fixed. That cost is the squared 2-Wasserstein distance,
averaged over the conditioning classes and divided by twice the output
variance so the answer lands in [0, 1]. Borgonovo, Figalli, Plischke and
Savaré introduced it in 2024.

It is a given-data method. Any (X, Y) pairs work, no design is required, and
the analysis adds no model evaluations to whatever you already ran. That
matters most for the modes below that summarize a whole trajectory: you get
one number per input for a 24-point time course out of the same rows a
per-timepoint analysis would use.

## What it measures that a variance index misses

Every OT index splits exactly into two parts, and no other method here does
that.

**advective**, the location shift. This is the part of the movement that is
pure translation, moving the conditional distribution's centre. It equals
exactly half the given-data first-order Sobol index.

**diffusive**, everything else. Spread, tails, skew, shape.

So the OT index contains the variance-based answer and then adds to it. An
input with a large advective part moves the output. An input with a large
diffusive part reshapes it. Sobol sees only the first kind.

## Import style

```python
from jaxgsa import optimal_transport
# optimal_transport.analyze(...)
```

## A first run on Ishigami

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=8192, seed=42))
Y = evaluate(X)

result = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y)

print("ot:       ", result.ot)
print("advective:", result.advective)
print("diffusive:", result.diffusive)
print("S1:       ", result.S1)
print("above_dummy:", result.above_dummy)
```

```
jaxgsa.optimal_transport.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 8192 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1946 s
    mode: univariate
    epsilon: 0.01
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ot
    1. x2  ot=0.2764
    2. x1  ot=0.2013
    3. x3  ot=0.0955
ot:        [0.2012687  0.27641615 0.09550098]
advective: [0.1566093  0.22097291 0.00218869]
diffusive: [0.04465945 0.05544327 0.0933123 ]
S1:        [0.31325683 0.44199976 0.00437791]
above_dummy: None
```

The block above the arrays is the verbose summary, printed because
`verbose=True` is the 1.0 default on every `analyze`. Pass `verbose=False`
for a silent run.

Now read the split. x1 and x2 are dominated by their advective parts
(0.157 of 0.201, and 0.221 of 0.276). Those two inputs move the output, and a
Sobol analysis would have told you as much.

x3 is the opposite. Its advective part is 0.0022, essentially zero, and its
diffusive part is 0.0933, which is almost the whole of its 0.0955 index. That
is correct and it is the point of the method: x3 enters Ishigami only through
`0.1 * x3^4 * sin(x1)`, whose conditional mean over x1 is zero for every x3.
Fixing x3 does not move the output distribution anywhere. It changes how wide
it is. One `analyze` call, and you can see the difference in kind between x1
and x3, not just a difference in magnitude.

`above_dummy` is `None` because this call did not pass `dummy=True`. See
[the irrelevance baseline](#the-irrelevance-baseline).

## S1 is a free cross-check

New in 1.0, the result carries `S1` and `S1_conf`. It is exactly
`2 * advective * N / (N - 1)`, where the `N / (N - 1)` factor puts the
conditional and unconditional variances on the same ddof=0 convention that
[borgonovo](/examples/borgonovo) uses, so the identity `advective = S1 / 2`
holds with no small print.

That makes it a cheap consistency test. Two given-data estimators, entirely
different machinery, one that sorts quantiles and one that fits kernel
densities, should land on the same first-order index:

```python
N = X.shape[0]
print("2 * advective * N/(N-1):", 2.0 * result.advective * N / (N - 1))
print("result.S1:              ", result.S1)

b = jaxgsa.borgonovo.analyze(PROBLEM, X, Y, verbose=False)
print("borgonovo S1:           ", b.S1)
```

```
2 * advective * N/(N-1): [0.31325683 0.44199976 0.00437791]
result.S1:               [0.31325683 0.44199976 0.00437791]
borgonovo S1:            [0.3125267  0.43882066 0.00308204]
```

The first two lines agree bit for bit, because the second is computed from
the first. The third is the real check. Ishigami's analytic first-order
indices are 0.3139, 0.4424 and 0.0000. The OT estimator gives 0.3133, 0.4420
and 0.0044; borgonovo gives 0.3125, 0.4388 and 0.0031. Both are within
Monte Carlo error of the truth and of each other, and neither was told the
answer.

Run this on your own model when a result surprises you. If the two S1 columns
disagree by more than their intervals allow, the disagreement is about your
data, not about the methods.

## The three modes

`mode` decides what counts as one output distribution. Pick it by what
question you are asking, not by the shape of your array.

**`"univariate"`** (the default) scores every output column on its own, using
the exact closed form of 1-D optimal transport, the sorted-quantile coupling.
No iterative solver runs at all, which is why the Ishigami call above took
0.19 s. Indices come back as `(D,)`, `(K, D)` or `(T, K, D)`.

**`"multivariate"`** treats the whole flattened output vector as one point
cloud and returns one index per input over the joint distribution, shape
`(D,)`. Use it when the outputs only make sense together, for example a pair
of concentrations whose ratio is what you care about.

**`"trajectory"`** does the same per output, with each output's time course as
the cloud, and returns `(K, D)`. It requires a 3-D `Y`. Use it when each
output is a curve and you want one number for the whole curve.

The last two transport point clouds, which has no closed form, so they run a
pure-JAX log-domain Sinkhorn solver. They are considerably slower. That is
the price of a joint answer.

Here is a small pharmacokinetic model where the choice matters. Two outputs,
a plasma curve and a tissue curve, over twelve timepoints:

```python
problem = jaxgsa.Problem.from_dict(
    {"dose": (0.5, 1.5), "k_abs": (0.2, 1.0), "k_elim": (0.05, 0.4)},
    output_names=("plasma", "tissue"),
)
t = jnp.linspace(1.0, 24.0, 12)

def curves(X):
    dose, ka, ke = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    plasma = dose * ka / (ka - ke) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))
    tissue = 0.4 * dose * (1.0 - jnp.exp(-ke * t))
    return jnp.stack([plasma, tissue], axis=-1)      # (N, T=12, K=2)

Xpk = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n=2048, seed=7))
Ypk = curves(Xpk)

r_tr = jaxgsa.optimal_transport.analyze(problem, Xpk, Ypk, mode="trajectory")
print(r_tr.ot)
```

```
jaxgsa.optimal_transport.analyze
  problem: D=3 (dose, k_abs, k_elim)
    marginals: uniform=3
    correlation: independent
    output: N=2048 runs, T=12 x K=2 output slices
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 7.453 s
    mode: trajectory
    epsilon: 0.01
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 3 of 3 parameters by ot, mean over 2 output slices
    1. k_elim  ot=0.4594
    2. dose    ot=0.4352
    3. k_abs   ot=0.1186
[[0.26304573 0.15762493 0.6728184 ]
 [0.6074043  0.079532   0.24592847]]
```

Row 0 is plasma, row 1 tissue, in the `output_names` order. Elimination rate
dominates the plasma curve (0.673) and dose dominates the tissue curve
(0.607), which is what the two formulas say.

Now compare against the default mode on the same data, and look at what
absorption rate does over time:

```python
r_uni = jaxgsa.optimal_transport.analyze(problem, Xpk, Ypk, verbose=False)
print("univariate ot shape:", r_uni.ot.shape)
print("plasma, per-time ot for k_abs:", r_uni.ot[:, 0, 1])
```

```
univariate ot shape: (12, 2, 3)
plasma, per-time ot for k_abs: [0.3315365  0.10241234 0.02204981 0.01721671 0.02212655 0.0251486
 0.02644053 0.02720482 0.02776806 0.02825176 0.02877845 0.0295321 ]
```

`k_abs` scores 0.332 at the first timepoint, 0.102 at the second, and then
sits near 0.02 for the remaining ten hours. That is the pharmacology:
absorption governs the early rise and then stops mattering. The trajectory
index for the same input and output is 0.158, which is well above the mean of
that row (0.057) because trajectory mode transports the whole curve as one
object and keeps the correlation between timepoints. Averaging the
per-timepoint indices would have buried the early effect under ten
uninformative hours.

That is the rule for choosing. Use `univariate` when you want to know *when*
an input matters. Use `trajectory` when you want to know *whether* it matters
to the curve. They are different questions and they have different answers
here.

## The irrelevance baseline

In the point-cloud modes, the entropic regularization and ordinary
finite-sample noise together keep even a totally irrelevant input strictly
above zero. Do not compare those indices against 0. Compare them against a
baseline.

`dummy=True` pushes one synthetic input, independent of the output by
construction, through the identical pipeline and reports its index as
`ot_dummy`. New in 1.0, the result also carries `above_dummy`, which is
`max(ot - ot_dummy, 0)`. It is `None` unless you passed `dummy=True`.

The example below adds an `inert` input the model never reads, and analyzes
the pair of 24-hour concentrations jointly:

```python
problem = jaxgsa.Problem.from_dict(
    {"dose": (0.5, 1.5), "k_abs": (0.2, 1.0), "k_elim": (0.05, 0.4),
     "inert": (0.0, 1.0)},
    output_names=("plasma", "tissue"),
)
X4 = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n=2048, seed=7))
Y4 = curves(X4)[:, -1, :]      # (N, K=2), the 24 h pair

result = jaxgsa.optimal_transport.analyze(
    problem, X4, Y4, mode="multivariate", dummy=True, key=jax.random.key(0),
)
print("ot:         ", result.ot)
print("ot_dummy:   ", result.ot_dummy)
print("above_dummy:", result.above_dummy)
```

```
jaxgsa.optimal_transport.analyze
  problem: D=4 (dose, k_abs, k_elim, inert)
    marginals: uniform=4
    correlation: independent
    output: N=2048 runs, T=1 x K=2 output slices
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 5.897 s
    mode: multivariate
    epsilon: 0.01
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 4 of 4 parameters by ot
    1. dose    ot=0.5356
    2. k_elim  ot=0.5137
    3. inert   ot=0.09749
    4. k_abs   ot=0.09711
ot:          [0.5355592  0.09711429 0.5137086  0.0974879 ]
ot_dummy:    0.09990064
above_dummy: [0.43565854 0.         0.41380796 0.        ]
```

This is the whole argument for the dummy in one printout. `k_abs` scores
0.0971 and `inert` scores 0.0975. Without a baseline you would report a small
but real absorption effect at 24 hours. The dummy scores 0.0999, above both
of them, so neither is distinguishable from noise, and `above_dummy` zeroes
both. Which is right: by 24 hours the absorption phase is long over, and
`inert` was never in the model at all.

The floor is not small. It is 0.0999 on a [0, 1] scale, and it eats
everything below about 0.1. Pass `dummy=True` on any point-cloud run where
you intend to call an input unimportant. It costs one more input's worth of
transport solves.

`epsilon` trades entropic bias against solver iterations. Smaller means less
bias and more iterations. The default 0.01 is relative to a cost matrix
scaled to [0, 1].

Note that `key` is required here. It feeds both the bootstrap and the
synthetic dummy input, which are independent consumers, so `analyze` raises
if you set `dummy=True` or `n_bootstrap > 0` without one.

## Standardizing outputs in the joint modes

`standardize_outputs=True` (the default) divides each output column by its
standard deviation before the transport cost is built. This is the 1.0
spelling; the old `standardize=` keyword is gone.

It only applies to the joint modes. `univariate` normalizes each column by
its own variance regardless, so the setting does nothing there.

Turn it off and one output's units decide the answer:

```python
Y2 = jnp.stack([Y, 1000.0 * jnp.sum(X**2, axis=1)], axis=1)

for flag in (True, False):
    r = jaxgsa.optimal_transport.analyze(
        PROBLEM, X, Y2, mode="multivariate",
        standardize_outputs=flag, verbose=False,
    )
    print(f"standardize_outputs={flag}: ot={r.ot}")
```

```
standardize_outputs=True: ot=[0.314138   0.34358105 0.23528925]
standardize_outputs=False: ot=[0.25355068 0.25373605 0.2465351 ]
```

The second output here is a thousand times larger than the first, as it would
be if one were measured in millimetres and the other in metres. With
standardization on, the two outputs contribute comparably and the ranking is
x2 > x1 > x3. With it off, the large output swamps the transport cost
entirely. Analyze that output on its own and you get the same three numbers:

```python
r_big = jaxgsa.optimal_transport.analyze(
    PROBLEM, X, Y2[:, 1:], mode="multivariate", verbose=False,
)
print("the large output alone:", r_big.ot)
```

```
the large output alone: [0.25355035 0.25373566 0.24653472]
```

Agreement to six digits with the unstandardized joint run. The first output
contributed nothing. The ranking has not reversed, it has been flattened out
of existence, which is worse, because a flat result looks like a finding.

Turn it off only when the outputs are already in the same units and their
relative magnitudes are the thing you want weighted.

## Bootstrap intervals

The bootstrap resamples the (X, Y) rows and re-estimates. It costs no model
runs.

```python
result = jaxgsa.optimal_transport.analyze(
    PROBLEM, X, Y, n_bootstrap=200, conf_level=0.95,
    key=jax.random.key(0), verbose=False,
)
print("ot:     ", result.ot)
print("ot_conf:", result.ot_conf)
print("S1:     ", result.S1)
print("S1_conf:", result.S1_conf)
```

```
ot:      [0.2012687  0.27641615 0.09550098]
ot_conf: [[0.19533639 0.2703077  0.09074909]
 [0.21178016 0.2910235  0.10534795]]
S1:      [0.31325674 0.44199976 0.00437791]
S1_conf: [[0.30253428 0.4288565  0.0032733 ]
 [0.32742235 0.46053672 0.01214939]]
```

Every `*_conf` is shaped `(2, ...)`: row 0 lower, row 1 upper. They are
`None` when `n_bootstrap` is 0, which is the default. The three `ot`
intervals here do not overlap, so the ranking is supported by the data. x3's
`S1_conf` is [0.0033, 0.0121], which brackets a first-order index whose true
value is 0, while its `ot_conf` of [0.091, 0.105] is nowhere near zero. Same
input, same rows, and the two intervals disagree because they are measuring
different things.

Keep `n_bootstrap` modest in the point-cloud modes. Each replicate re-solves
`D * n_partitions` transport problems, so 200 replicates on 4 inputs and 25
classes is 20000 Sinkhorn solves.

## Conditioning classes

`n_partitions` sets how many equal-frequency conditioning classes each
continuous input gets. `None` (the default) picks `min(25, N // 2)`. About 25
is customary for this index at N >= 2500.

More classes localize the conditioning and lower the discretization bias, but
leave fewer samples per class and raise the noise. The classes come from the
inputs' ordinal ranks, so the estimator is distribution-free in X: uniform,
Gaussian, truncated Gaussian and mixed marginals all work unchanged, and no
CDF transform is applied. Categorical inputs ignore `n_partitions` and use one
class per level, so the index depends only on the level partition and never on
the arbitrary code order.

## Shape rules

N is the number of samples, T the number of timepoints, K the number of
outputs, D the number of inputs.

| `mode` | `Y` shape | `ot`, `advective`, `diffusive`, `S1` |
|---|---|---|
| `univariate` | `(N,)` | `(D,)` |
| `univariate` | `(N, K)` | `(K, D)` |
| `univariate` | `(N, T, K)` | `(T, K, D)` |
| `multivariate` | any | `(D,)` |
| `trajectory` | `(N, T, K)` only | `(K, D)` |

Every `*_conf` adds a leading axis of size 2. `ot_dummy` has the shape of
`ot` with the parameter axis removed; `above_dummy` has the shape of `ot`.

## Selecting by name with xarray

```python
ds = r_tr.to_dataset()
print(ds)
```

```
<xarray.Dataset> Size: 216B
Dimensions:    (output: 2, param: 3)
Coordinates:
  * output     (output) <U6 48B 'plasma' 'tissue'
  * param      (param) <U6 72B 'dose' 'k_abs' 'k_elim'
Data variables:
    ot         (output, param) float32 24B 0.263 0.1576 ... 0.07953 0.2459
    advective  (output, param) float32 24B 0.09774 0.0355 ... 0.005924 0.141
    diffusive  (output, param) float32 24B 0.1653 0.1221 ... 0.07361 0.105
    S1         (output, param) float32 24B 0.1956 0.07103 ... 0.01185 0.2821
Attributes:
    mode:     trajectory
```

The mode is recorded in `attrs`, which is worth having when a saved dataset
outlives the script that made it. Bootstrap bounds arrive as `*_lower` and
`*_upper`, and `ot_dummy` as its own variable, both only when you asked for
them.

## Other things worth knowing

A constant, zero-variance output slice gives an index of exactly 0 rather
than NaN, together with a warning naming the slice.

Correlated inputs are supported and the index stays well-defined. It then
measures each input's total association with the output, including effects
carried by correlated partners, and the same reading applies to `S1`. An
input the model never reads scores above 0 when it correlates with one the
model does read. That is correct, not an error. Use
[VKOGA](/examples/vkoga) or [Kucherenko](/examples/kucherenko) to split the
direct effect from the borrowed one.

`slice_chunk_size` applies to `univariate` mode only. The point-cloud modes
accept it and ignore it, because one `(N, N/M)` cost block per solve already
bounds their peak memory.

## References

Borgonovo, E., Figalli, A., Plischke, E., & Savaré, G. (2024). Global
sensitivity analysis via optimal transport. *Management Science*.
doi:10.1287/mnsc.2023.01796

## See also

- [Basic example](/examples/basic) for the Sobol variance decomposition.
- [Borgonovo delta](/examples/borgonovo) for the density-distance index and
  the other given-data S1.
- [PAWN](/examples/pawn) for the CDF-based index, which is much cheaper.
- [HSIC](/examples/hsic) for kernel dependence with a significance test.
- [Methods](/guide/methods) for a side-by-side comparison.
- [API reference](/api/#given-data-methods) for every parameter.
