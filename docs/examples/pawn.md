# PAWN: sensitivity read off the output CDF

PAWN works on the output's cumulative distribution function, not on its
variance. For each input it splits the sample into `n_bins` conditioning bins,
builds the output CDF inside each bin, and compares that conditional CDF with
the unconditional one using the Kolmogorov-Smirnov statistic, the largest
vertical gap between two CDF curves. The per-bin gaps are then reduced to one
number per input.

That is the whole method, and the consequence is the reason to use it. A
variance index asks how much the output spread shrinks when you fix an input.
PAWN asks whether the *shape* of the output distribution changes at all. An
input that shifts a tail, splits the output into two modes, or changes the
skew while leaving the variance where it was is invisible to the first
question and obvious to the second. The KS statistic is also invariant under
any monotone transformation of the output, so analyzing `Y` and analyzing
`log(Y)` give the same index.

PAWN is a given-data method. Bring the (X, Y) pairs you already have. There is
no design to satisfy and no extra model run to pay for. A Latin hypercube
study someone ran last year is a valid PAWN input as it stands.

## Import style

```python
from jaxgsa import pawn
# pawn.analyze(...)
```

## A first run on Ishigami

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=5000, seed=42))
Y = evaluate(X)

result = jaxgsa.pawn.analyze(PROBLEM, X, Y)

print("pawn:        ", result.pawn)
print("n_valid_bins:", result.n_valid_bins)
```

```
jaxgsa.pawn.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=5000 runs, T=1 x K=1 output slice
    invalid: none found in 5000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.4539 s
    slice_chunk_size: 1 (resolved from the memory budget)
    statistic: median
    n_bins: 10
  results: top 3 of 3 parameters by PAWN
    1. x2  PAWN=0.4066
    2. x1  PAWN=0.2419
    3. x3  PAWN=0.08334
pawn:         [0.2419315  0.40658885 0.08334284]
n_valid_bins: [10 10 10]
```

The block above the arrays is the verbose summary, printed because
`verbose=True` is the 1.0 default on every `analyze`. It records what was
analyzed and how, which is worth keeping in a log: three uniform inputs, 5000
rows, no non-finite values, median aggregation over 10 bins. Pass
`verbose=False` when you do not want it.

`result.pawn` has one entry per input, ordered as in `PROBLEM`. Each entry is
the median KS distance across that input's conditioning bins, and it lives in
[0, 1]. A conditioning bin is a slice of one input's range; inside a bin that
input is nearly fixed while the others still vary freely, so the conditional
output CDF is the CDF you would see if you knew that input.

Now the part worth pausing on. Ishigami's x3 has a first-order Sobol index of
exactly zero: x3 enters only through the term `0.1 * x3^4 * sin(x1)`, whose
conditional mean over x1 is zero for every x3. A variance-based first-order
analysis reports x3 as having no effect. PAWN gives it 0.083, and the
[bootstrap below](#bootstrap-intervals) puts that comfortably clear of zero.
Fixing x3 does change the output distribution. It widens or narrows it
without moving its centre, and that is exactly the kind of effect the CDF
comparison sees and the variance decomposition does not.

`n_valid_bins` is new in 1.0 and reports how many bins actually contributed
per input. All 10 of 10 here. When that number drops, read
[the sparse-bin warning](#when-bins-run-out-of-samples).

## Choosing n_bins

Bin count is the parameter people get wrong, so it deserves the space.

The bins are equal-width on the CDF-transformed unit interval, which makes
them equal-probability under the input's own marginal. Each holds roughly
`N / n_bins` samples. That single ratio is the whole trade-off. More bins
condition each input more tightly, which reduces the bias from treating a
whole slice as "fixed". Fewer samples per bin make each KS statistic noisier,
and KS noise is one-sided: a KS distance is a maximum, so noise can only push
it up.

Watch both ends of that trade-off in one sweep:

```python
for nb in (5, 10, 20, 50):
    r = jaxgsa.pawn.analyze(PROBLEM, X, Y, n_bins=nb, verbose=False)
    print(f"n_bins={nb:3d}: pawn={r.pawn}  n_valid_bins={r.n_valid_bins}")
```

```
n_bins=  5: pawn=[0.18761349 0.28084224 0.08781385]  n_valid_bins=[5 5 5]
n_bins= 10: pawn=[0.2419315  0.40658885 0.08334284]  n_valid_bins=[10 10 10]
n_bins= 20: pawn=[0.2354129  0.40557504 0.09115449]  n_valid_bins=[20 20 20]
n_bins= 50: pawn=[0.24619085 0.43429956 0.10700866]  n_valid_bins=[50 50 50]
```

Five bins is too few. Each bin covers a fifth of x2's range, x2 is far from
constant inside it, and the conditional CDF is smeared toward the
unconditional one. x2 comes back as 0.281 instead of 0.406, a 30% understatement.

Ten and twenty bins agree to within 0.006 on every input. That agreement is
the signal you want: the answer has stopped depending on the knob.

Fifty bins, at 100 samples per bin, starts drifting upward. Every index rises,
x3 most in relative terms (0.083 to 0.107). That is the one-sided KS noise
showing, not a newly discovered effect.

The recipe: start at the default 10, then run 2x that. If the two agree, keep
the answer. If they do not, you are on the noisy side and you need more
samples rather than a different bin count. Keep at least 100 samples per bin
as a floor, so `n_bins <= N / 100`.

## Choosing the aggregation statistic

One KS distance per bin, then one reduction to a single number. The reduction
decides which bin drives the index.

```python
for s in ("median", "max", "mean"):
    r = jaxgsa.pawn.analyze(PROBLEM, X, Y, statistic=s, verbose=False)
    print(f"{s:7s}: {r.pawn}")
```

```
median : [0.2419315  0.40658885 0.08334284]
max    : [0.2966746  0.49668312 0.20064142]
mean   : [0.22638342 0.350946   0.09694888]
```

`max` is at least as large as the other two for every input, by definition.
The interesting number is the ratio. x1 and x2 rise by about 20% from median
to max; x3 rises by 140%, from 0.083 to 0.201. That says x3 acts strongly in
one part of its range and weakly elsewhere, which is right: the `x3^4` term
is flat near zero and steep at the edges.

Use `median` (the default) when you want a robust summary. Use `max` for
screening, because an input is safe to drop only if *no* bin shifts the
output. Use `mean` when you want every bin to count equally and you already
trust the bins.

## Bootstrap intervals

The bootstrap resamples the (X, Y) rows with replacement and recomputes the
index each time. It costs no model runs, only arithmetic. Set `n_bootstrap`
and pass a `key`; without the key `analyze` raises
`ValueError: key is required when n_bootstrap > 0`.

```python
result = jaxgsa.pawn.analyze(
    PROBLEM, X, Y,
    n_bootstrap=200, conf_level=0.95, key=jax.random.key(0), verbose=False,
)
print("pawn: ", result.pawn)
print("lower:", result.pawn_conf[0])
print("upper:", result.pawn_conf[1])
```

```
pawn:  [0.2419315  0.40658885 0.08334284]
lower: [0.22795124 0.3840406  0.08216228]
upper: [0.2634125  0.42300665 0.09661566]
```

`pawn_conf` is shaped `(2, D)`: row 0 lower, row 1 upper. It is `None` when
`n_bootstrap` is 0, which is the default.

Read these as separation tests. The three intervals here do not overlap, so
the ranking x2 > x1 > x3 is supported by the data, not an artifact of which
5000 rows you happened to draw. x3's interval is [0.082, 0.097], which is the
number that makes the claim above precise: PAWN finds a real effect for the
input whose first-order Sobol index is exactly zero.

The estimate itself does not move, because PAWN's point estimate is not
bias-corrected. Only the interval is new. That differs from
[Borgonovo delta](/examples/borgonovo), where turning on the bootstrap also
changes the reported index.

## When bins run out of samples

A bin needs at least two samples to define a conditional CDF. Bins with fewer
are dropped, and the index rests on whatever survives. This happens when an
input was logged at a handful of settings, which is common with real data:
someone ran the rig at three flow rates, not at 5000 of them.

New in 1.0, `analyze` warns when an input keeps fewer than half its bins, and
`n_valid_bins` tells you which one.

```python
X_coarse = X.at[:, 2].set(jnp.round(X[:, 2] / 2.5) * 2.5)   # x3 logged at 3 settings
Y_coarse = evaluate(X_coarse)

r = jaxgsa.pawn.analyze(PROBLEM, X_coarse, Y_coarse)
print("pawn:        ", r.pawn)
print("n_valid_bins:", r.n_valid_bins)
```

```
JaxgsaWarning: PAWN: parameters 'x3' (3/10) have fewer than half of their
conditioning bins contributing (a bin needs at least 2 samples to define a
conditional CDF; the rest are dropped). The reported indices rest on those few
bins. Use fewer bins (lower n_bins) or more samples.

jaxgsa.pawn.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=5000 runs, T=1 x K=1 output slice
    invalid: none found in 5000 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.3991 s
    slice_chunk_size: 1 (resolved from the memory budget)
    statistic: median
    n_bins: 10
  results: top 3 of 3 parameters by PAWN
    1. x2  PAWN=0.3367
    2. x1  PAWN=0.3177
    3. x3  PAWN=0.07878
pawn:         [0.31772962 0.33670652 0.07877553]
n_valid_bins: [10 10  3]
```

The warning names x3 and says 3 of 10. What it means is that x3's index is a
median over three numbers, so it is far noisier than the other two even
though nothing in the printed value says so.

Two fixes, and which one you pick depends on why the bins are empty.

If the input genuinely has only a few settings, lower `n_bins` to match. The
extra bins were never going to be filled, and asking for them only throws
information away:

```python
r = jaxgsa.pawn.analyze(PROBLEM, X_coarse, Y_coarse, n_bins=3, verbose=False)
print("n_bins=3 pawn:", r.pawn, " n_valid_bins:", r.n_valid_bins)
```

```
n_bins=3 pawn: [0.2720249  0.1078375  0.07877553]  n_valid_bins: [3 3 3]
```

The warning is gone, and x3's index is unchanged at 0.0788, because three
bins was all x3 ever had. The other two inputs move a lot, though, and x2
falls from 0.337 to 0.108. That is the earlier lesson again: three bins
under-resolves a continuous input badly. So this fix silences the warning at
the cost of the inputs that were fine.

If instead you want to keep the resolution on the well-sampled inputs, keep
`n_bins` where it is, accept the warning, and treat x3's number as a weak
signal rather than a measurement. Declaring x3 categorical is the clean
answer when it really is a discrete setting: a categorical input gets one bin
per level, `n_bins` does not apply to it, and relabelling the levels changes
nothing. See [Categorical inputs](/examples/categorical-inputs).

## Multiple outputs

`Y` shaped `(N, K)` gives indices shaped `(K, D)`. One call covers every
output, and the conditioning is built once and shared.

```python
X3 = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=3000, seed=42))
Y1 = evaluate(X3)
Y2 = jnp.sum(X3**2, axis=1)

result = jaxgsa.pawn.analyze(PROBLEM, X3, jnp.column_stack([Y1, Y2]),
                             verbose=False)
print("pawn shape:", result.pawn.shape)
print(result.pawn)
print(result.to_dataset())
```

```
pawn shape: (2, 3)
[[0.25033993 0.4081337  0.08886259]
 [0.20379516 0.2067296  0.21694896]]

<xarray.Dataset> Size: 88B
Dimensions:       (output: 2, param: 3)
Coordinates:
  * output        (output) <U2 16B 'y0' 'y1'
  * param         (param) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    pawn          (output, param) float32 24B 0.2503 0.4081 ... 0.2067 0.2169
    n_valid_bins  (output, param) int32 24B 10 10 10 10 10 10
```

Row 0 is the Ishigami output and keeps the familiar ranking. Row 1 is the sum
of squares, and it is nearly flat at 0.20 to 0.22 because that function
treats all three inputs alike. Two outputs, two different stories, one pass
over the data.

The outputs are named `y0` and `y1` because this `PROBLEM` declares no
`output_names`. Pass them to `Problem.from_dict` and they appear here
instead, which is worth doing the moment you have more than two outputs.
`to_dataset()` then lets you write `ds.pawn.sel(output="plasma", param="dose")`.

Bootstrap bounds reach the dataset as `pawn_lower` and `pawn_upper`, and only
when you asked for them.

## Shape rules

N is the number of samples, T the number of time steps, K the number of
outputs, D the number of inputs.

| `Y` shape | `pawn`, `n_valid_bins` | `pawn_conf` |
|---|---|---|
| `(N,)` | `(D,)` | `(2, D)` or `None` |
| `(N, K)` | `(K, D)` | `(2, K, D)` or `None` |
| `(N, T, K)` | `(T, K, D)` | `(2, T, K, D)` or `None` |

## Memory on long time series

PAWN pushes the flattened `T*K` output columns through one vmapped kernel, in
chunks of at most `slice_chunk_size` columns.

The chunk is not sized by the result. The inner `vmap` builds a full
`(N, n_bins)` ECDF table for every (column, input) pair and holds two such
arrays at once, so peak memory is about
`2 * slice_chunk_size * D * N * n_bins` elements. That is `N` times the size
of the result, which is why the default is derived rather than fixed.

`slice_chunk_size=None` (the default) sizes the chunk against the active
memory budget, which `jaxgsa.config.set_memory_budget` sets. Pass a positive
integer to override:

```python
t = jnp.linspace(0.1, 5.0, 40)
Y_ts = (evaluate(X)[:, None] * jnp.exp(-t)[None, :])[:, :, None]   # (N, T=40, K=1)

r = jaxgsa.pawn.analyze(PROBLEM, X, Y_ts, slice_chunk_size=8, verbose=False)
r_auto = jaxgsa.pawn.analyze(PROBLEM, X, Y_ts, verbose=False)
print("pawn shape:", r.pawn.shape)
print("identical to the auto-sized run:", bool(jnp.array_equal(r.pawn, r_auto.pawn)))
```

```
pawn shape: (40, 1, 3)
identical to the auto-sized run: True
```

Lower it when a long time series exhausts the device. It changes no index:
every output column is computed independently of every other, so the chunked
answer is the unchunked answer exactly, as the printed comparison shows.

## Other things worth knowing

Any (X, Y) pairs work. Monte Carlo, Latin hypercube, a Sobol sequence, or the
leftovers of a Saltelli design you ran for something else.

The KS statistic sharpens with N. Larger samples separate close inputs
better, and there is no upper bound past which more data stops helping.

Correlated inputs are supported. PAWN bins one input and compares output
CDFs, so a declared `problem.correlation` does not invalidate anything. Each
index then reads as total influence, including influence borrowed from
correlated partners. An input the model ignores can score above 0 when it
correlates with one the model reads. That reading is correct. Use
[VKOGA](/examples/vkoga) or [Kucherenko](/examples/kucherenko) to split the
direct effect from the borrowed one.

## See also

- [Basic example](/examples/basic) for the Sobol variance decomposition.
- [Borgonovo delta](/examples/borgonovo), the density-based sibling of this
  method, which also returns a given-data first-order Sobol index.
- [Optimal transport](/examples/optimal-transport) for a distributional index
  that splits into a location shift and a reshape.
- [HSIC](/examples/hsic) for kernel dependence with a significance test.
- [Methods](/guide/methods) for a side-by-side comparison.
- [API reference](/api/#given-data-methods) for every parameter.
