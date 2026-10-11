# Borgonovo delta: a distance between output densities

The delta index measures how far fixing one input moves the whole probability
density of the output. Take the output density with nothing fixed, that is
the unconditional density. Take the density after conditioning on one input,
that is the conditional density. Delta is the expected L1 distance between
the two, halved so it lands in [0, 1]. Geometrically it is the average area
between the two density curves.

Nothing in that definition mentions a mean or a variance, which is what
"moment-independent" means. That is the whole argument for the method.

## What this buys over Sobol

A first-order Sobol index answers one question: how much of the output
variance does this input explain on its own? That question has a blind spot.
Variance is a single number summarizing a whole distribution, and an input
can rearrange the distribution without changing that number, or change the
distribution's shape in a way the variance under-reports. Bimodality, tail
thickness, and skew all live in that blind spot.

Delta has no blind spot of that kind, because it compares the densities
themselves. If fixing an input changes the output distribution in any way at
all, delta is positive.

`analyze` returns both indices from the same conditioning, so the comparison
costs nothing:

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=5000, seed=42))
Y = evaluate(X)

result = jaxgsa.borgonovo.analyze(PROBLEM, X, Y)

print("delta:", result.delta)
print("S1:   ", result.S1)
print("delta_conf:", result.delta_conf)
```

```
jaxgsa.borgonovo.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=5000 runs, T=1 x K=1 output slice
    invalid: none found in 5000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.2515 s
    slice_chunk_size: 1 (resolved from the memory budget)
    grid_size: 100
    bandwidth: silverman
  results: top 3 of 3 parameters by delta
    1. x2  delta=0.3516
    2. x1  delta=0.2106
    3. x3  delta=0.1567
delta: [0.21057463 0.3515927  0.15670761]
S1:    [0.30810872 0.43677974 0.00358097]
delta_conf: None
```

Read the two arrays side by side, entry by entry.

For x1 and x2 the two indices agree on the story. Both say x2 matters most,
x1 second. Where an input acts on the output in the ordinary way, by moving
its centre and its spread together, delta tells you nothing the Sobol index
did not.

x3 is the interesting column. Its `S1` is 0.0036, which is a Monte Carlo
estimate of exactly zero: x3 enters Ishigami only through
`0.1 * x3^4 * sin(x1)`, and averaging that over x1 gives zero for every x3.
A first-order variance analysis reports x3 as inert. Its delta is 0.157,
comfortably third but not close to zero. Fixing x3 does change the output
distribution, it just does so by widening and narrowing it rather than by
moving it. That gap between 0.0036 and 0.157 is what a moment-independent
index adds, in one concrete number.

The verbose block above the arrays comes free, because `verbose=True` is the
1.0 default on every `analyze`. Pass `verbose=False` to silence it. Note
`delta_conf: None`: the default is `n_bootstrap=0`, so no intervals were
computed. The next section shows how to add them.

## Given data, no new model runs

Delta is a given-data method. Any (X, Y) pairs work. There is no design to
satisfy, so a Monte Carlo run, a Latin hypercube, or an existing simulation
log all go straight in, and the analysis costs zero further model
evaluations. The bootstrap below resamples those same rows, so it costs
nothing extra either.

The one restriction is on the output, not on the sampling.

::: warning Continuous outputs only
`borgonovo.analyze` supports a continuous output distribution only. The
estimator compares kernel density estimates on a shared output grid, and a
discrete output has atoms that no grid resolves, so the index would report
the grid resolution rather than the model. `analyze` checks first and raises:

```python
Y_disc = jnp.where(Y > 0, 1.0, 0.0)
jaxgsa.borgonovo.analyze(PROBLEM, X, Y_disc)
```

```
ValueError: jaxgsa.borgonovo.analyze supports a continuous output
distribution only, but the output takes only 2 distinct values in 5000
samples. The delta estimator compares Gaussian kernel density estimates on a
shared output grid; an atomic density is a spike that no grid resolves, so
the index would report the grid resolution, not the model. Use
jaxgsa.optimal_transport.analyze for a discrete output: it compares empirical
distributions directly and needs no density.
```

Use [optimal transport](/examples/optimal-transport) for a discrete output.
A continuous output rounded to a few decimals is not refused, and neither is
a constant column, whose exact answer is `delta = S1 = 0`. Categorical
*inputs* stay supported; the limit applies to the output only.
:::

## The bias_correct flag

The plug-in delta estimate is biased upward, and the reason is worth stating
because it explains why the correction only goes one way. A delta estimate is
a distance between two kernel density estimates. Sampling noise perturbs both
densities, and any perturbation can only push a distance up, never down. Two
identical densities estimated from finite samples separate; two different
ones do not converge. So the noise floor is a positive number added to every
index.

`bias_correct` is a plain bool. It is `False` by default. The two values do
what they say:

- `False` reports the plug-in estimate, always. Adding `n_bootstrap` adds
  intervals and does not change `delta`.
- `True` reports the Plischke correction `2*d_hat - mean(d_boot)`. It needs
  replicates, so it raises `ValueError` when `n_bootstrap` is 0.

Start with the default and add a bootstrap:

```python
result = jaxgsa.borgonovo.analyze(
    PROBLEM, X, Y,
    n_bootstrap=200, conf_level=0.95, key=jax.random.key(0),
)
print("delta (plug-in):", result.delta)
print("95% CI lower:", result.delta_conf[0])
print("95% CI upper:", result.delta_conf[1])
print("S1:", result.S1)
print("S1 CI:", result.S1_conf)
```

```
jaxgsa.borgonovo.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=5000 runs, T=1 x K=1 output slice
    invalid: none found in 5000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 1.227 s
    slice_chunk_size: 1 (resolved from the memory budget)
    grid_size: 100
    bandwidth: silverman
  results: top 3 of 3 parameters by delta
    1. x2  delta=0.3516  [0.3449, 0.3629]
    2. x1  delta=0.2106  [0.2079, 0.226]
    3. x3  delta=0.1567  [0.1529, 0.1703]
delta (plug-in): [0.21057463 0.3515927  0.15670761]
95% CI lower: [0.20792328 0.3449311  0.15285581]
95% CI upper: [0.22595103 0.3628546  0.17025897]
S1: [0.3081087  0.43677974 0.00358097]
S1 CI: [[0.29423395 0.41606882 0.00295663]
 [0.32919294 0.45707515 0.01367938]]
```

`delta` matches the first run exactly: 0.2106 for x1. The estimator is
deterministic given the data, and the bootstrap only adds the interval. Now
ask for the correction on the same replicates:

```python
r_corr = jaxgsa.borgonovo.analyze(
    PROBLEM, X, Y, n_bootstrap=200, bias_correct=True,
    key=jax.random.key(0), verbose=False,
)
print("delta (bias_correct=True):", r_corr.delta)
print("shift:", r_corr.delta - result.delta)
```

```
delta (bias_correct=True): [0.20413923 0.3486844  0.15101331]
shift: [-0.00643539 -0.00290829 -0.0056943 ]
```

The correction moved every index down, by 0.0064, 0.0029 and 0.0057.
Downward on every input, as the upward-bias argument predicts.

Use `bias_correct=True` when the value matters and not only the ranking.
Pair it with `n_bootstrap >= 100`. Keep `False` when you want the raw
estimator, or when you compare against another implementation that does not
correct.

`S1` is never bias-corrected, matching SALib. That is why `S1` is identical
in the runs above.

One consequence to expect: the corrected estimate, and its interval bounds,
can dip marginally below 0 for a weak input at small N. Subtracting an
estimated bias from a small number sometimes overshoots. That is the
correction working, not a failure.

## Checking against ground truth

The `gaussian_linear` benchmark has a semi-analytic delta solution in
`ANALYTICAL_DELTA`, so you can validate against the truth rather than against
another implementation. That makes it the right place to answer "how many
samples does my problem need?".

```python
from jaxgsa.benchmarks import gaussian_linear

print("analytical:", gaussian_linear.ANALYTICAL_DELTA)
for n in (500, 2000, 8000, 32000):
    Xn = jnp.asarray(jaxgsa.sampling.monte_carlo(gaussian_linear.PROBLEM, n_samples=n, seed=42))
    Yn = gaussian_linear.evaluate(Xn)
    r = jaxgsa.borgonovo.analyze(
        gaussian_linear.PROBLEM, Xn, Yn,
        n_bootstrap=100, bias_correct=True, key=jax.random.key(0), verbose=False,
    )
    err = float(jnp.max(jnp.abs(r.delta - jnp.asarray(gaussian_linear.ANALYTICAL_DELTA))))
    print(f"N={n:6d}: delta={r.delta}  max abs error={err:.4f}")
```

```
analytical: [0.08900277 0.20156255 0.38735419]
N=   500: delta=[0.1031889  0.17368464 0.3440713 ]  max abs error=0.0433
N=  2000: delta=[0.10180537 0.19444586 0.35278648]  max abs error=0.0346
N=  8000: delta=[0.09093203 0.18996786 0.36351147]  max abs error=0.0238
N= 32000: delta=[0.08855142 0.19409662 0.37382233]  max abs error=0.0135
```

The ranking is correct at N = 500 already. The *values* take much longer:
error falls from 0.043 to 0.014 over a 64x increase in N, roughly as
`N**-0.28`, which is slower than the `N**-0.5` you may expect. The
conditioning-class count grows with N under the Plischke heuristic, so
samples per class grow more slowly than N, and the KDE bias shrinks slowly
with it.

The practical reading: trust a delta ranking at a few thousand samples. If
you plan to quote a delta value, run this sweep on your own model and find
where it stops moving.

## Choosing n_classes

`n_classes` sets how many equal-frequency conditioning classes each
continuous input is split into. `None` (the default) uses the Plischke
sample-size heuristic, roughly `N**(2/7)` capped at 48, which is identical to
SALib's.

The trade-off mirrors PAWN's bin count. More classes condition more tightly
and lower the discretization bias; fewer samples per class make each
conditional KDE noisier. The default is well tested and I would leave it
alone unless a sweep tells you otherwise. A categorical input ignores the
setting entirely and uses one class per level.

## Multiple outputs

`Y` shaped `(N, K)` gives indices shaped `(K, D)`.

```python
X3 = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=3000, seed=42))
Y1 = evaluate(X3)
Y2 = jnp.sum(X3**2, axis=1)

result = jaxgsa.borgonovo.analyze(
    PROBLEM, X3, jnp.column_stack([Y1, Y2]),
    n_bootstrap=100, bias_correct=True, key=jax.random.key(0), verbose=False,
)
print("delta shape:", result.delta.shape)
print(result.delta)
print(result.to_dataset())
```

```
delta shape: (2, 3)
[[0.21183114 0.345705   0.14884597]
 [0.21189035 0.19872913 0.19872493]]

<xarray.Dataset> Size: 184B
Dimensions:      (output: 2, param: 3)
Coordinates:
  * output       (output) <U2 16B 'y0' 'y1'
  * param        (param) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    delta        (output, param) float32 24B 0.2118 0.3457 ... 0.1987 0.1987
    S1           (output, param) float32 24B 0.3189 0.4414 ... 0.322 0.3448
    delta_lower  (output, param) float32 24B 0.1981 0.3316 ... 0.1834 0.1876
    delta_upper  (output, param) float32 24B 0.2238 0.3592 ... 0.2154 0.2114
    S1_lower     (output, param) float32 24B 0.3003 0.4214 ... 0.2986 0.3241
    S1_upper     (output, param) float32 24B 0.3438 0.4751 ... 0.3559 0.3733
```

Row 0 is the Ishigami output, row 1 the sum of squares. Row 1 is flat at 0.20
to 0.21, which is right for a function that treats its three inputs alike.
Each output gets its own ranking, and an input can lead for one output and
trail for another.

The outputs are named `y0` and `y1` because this `PROBLEM` declares no
`output_names`. Pass `output_names=(...)` to `Problem.from_dict` and those
names appear here, which lets you write
`ds.delta.sel(output="plasma", param="dose")` instead of counting rows.

The dataset carries `delta_lower` and `delta_upper` only because this run
asked for a bootstrap. Without one, it holds `delta` and `S1` alone.

## Shape rules

N is the number of samples, T the number of time steps, K the number of
outputs, D the number of inputs.

| `Y` shape | `delta`, `S1` | `delta_conf`, `S1_conf` |
|---|---|---|
| `(N,)` | `(D,)` | `(2, D)` or `None` |
| `(N, K)` | `(K, D)` | `(2, K, D)` or `None` |
| `(N, T, K)` | `(T, K, D)` | `(2, T, K, D)` or `None` |

## When the estimator refuses an answer

Delta is a half L1 distance between densities, so it must lie in [0, 1]. If
the returned estimate leaves that range by more than 0.05, the computation
failed and `analyze` raises `ValueError` naming the input. It is never
clipped, because a clipped value looks plausible and is still wrong.

The cause is always a conditioning class the output grid cannot resolve. The
message reads what the run actually did and names the knob that governs it:
`grid_size` and `degenerate_bandwidth` when a class was found degenerate and
its kernel was floored, or `grid_size` and `degenerate_tol` when no class was
floored and the floor played no part.

A confidence bound outside [0, 1] only warns. The point estimate is the
contract; the interval is a diagnostic.

Two settings deserve a warning of their own. `degenerate_bandwidth` only ever
reaches a class the estimator already called degenerate, so on smooth data it
cannot change the result at any value. Raising `degenerate_tol` above the
floor fraction does bias the result, and in the direction you would not
guess: a class whose own bandwidth sits between the floor and the tolerance
is then *narrowed* to the floor, which inflates delta for exactly the classes
the higher tolerance said to distrust. That is a valid computation, so it is
a bias to know about rather than an error.

A near-atomic conditional class, which is the normal case for a categorical
level mapping to one output value, makes delta depend on `grid_size` and biases
it low. On a noise-free three-atom model with true delta 2/3, the estimate
goes 0.56 at `grid_size=50`, 0.61 at 100, and 0.61 at 200 and above. The bias
does not vanish as N grows either, so on atomic conditionals treat delta as a
ranking signal, not a calibrated number. Inputs with genuine conditional
spread are unaffected.

## Other things worth knowing

The estimator matches `SALib.analyze.delta` on the same partition, bandwidths
and grid, with two deliberate differences. It is deterministic given the
data, where SALib computes its central estimate on a random resample. And it
returns `delta = S1 = 0` for a constant output where SALib raises.

Peak memory scales with `slice_chunk_size * D * N * grid_size`. Lower
`slice_chunk_size` for long time-series outputs, though the output grid is
evaluated in tiles, so peak memory follows the tile rather than the whole
grid and a narrower chunk is rarely the knob that saves you.

Correlated inputs are supported. The estimator partitions on ordinal ranks
and compares output densities, so a declared `problem.correlation` does not
invalidate it. Under dependence both `delta` and `S1` read as total
association, including effects carried by correlated partners. Neither
separates the direct effect from the borrowed one. Use
[VKOGA](/examples/vkoga) or [Kucherenko](/examples/kucherenko) for that
split; both need continuous inputs.

## See also

- [Basic example](/examples/basic) for the Sobol variance decomposition.
- [PAWN](/examples/pawn), the CDF-based moment-independent method, which
  needs no continuous output.
- [Optimal transport](/examples/optimal-transport) for a distributional index
  that splits into a location shift and a reshape, and its own given-data S1.
- [HSIC](/examples/hsic) for kernel dependence with a significance test.
- [Methods](/guide/methods) for a side-by-side comparison.
- [API reference](/api/#given-data-methods) for every parameter.
