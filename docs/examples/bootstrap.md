# Bootstrap confidence intervals

A Sobol index computed from a finite sample is an estimate. Run the same study
with a different seed and every number moves. The bootstrap puts a bound on how
far, so you can tell a real ranking from an artifact of the sample you happened
to draw.

This is also the output people misread most often, so the middle of this page
is about what the interval covers and what it quietly leaves out.

## One analysis, one interval

Pass `n_bootstrap` and a JAX key. Every `*_conf` array on the result then holds
the bounds for the matching index array.

```python
import jax
import numpy as np

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

design = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=42)
Y = evaluate(design.samples)

result = jaxgsa.sobol.analyze(
    design,
    Y,
    n_bootstrap=500,
    conf_level=0.95,
    ci_method="quantile",
    key=jax.random.key(0),
)

lo, hi = np.asarray(result.S1_conf)
for i, name in enumerate(PROBLEM.names):
    print(f"{name}  S1={result.S1[i]: .4f}  95% CI [{lo[i]: .4f}, {hi[i]: .4f}]")
print(result.ci)
```

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=1024, requested_runs>=8192, n_runs=8192, n_expanded=8192, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 1024 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.9928 s
    slice_chunk_size: 1 (resolved from the memory budget)
    bootstrap slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5551  [0.4804, 0.6329]
    2. x2  ST=0.4398  [0.4049, 0.4844]
    3. x3  ST=0.2411  [0.2156, 0.2707]
x1  S1= 0.3270  95% CI [ 0.2603,  0.3877]
x2  S1= 0.4432  95% CI [ 0.3939,  0.4989]
x3  S1= 0.0113  95% CI [-0.0422,  0.0661]
```

```
CIInfo(level=0.95, method='quantile', n_bootstrap=500, replicates=None)
```

With `n_bootstrap > 0` the verbose table gains bracketed bounds next to each
`ST`, so you can read the ranking and its uncertainty without writing any
printing code.

Look at `x3`. Its `S1` is 0.0113 and the interval runs from -0.0422 to 0.0661,
straight through zero. The Ishigami function's analytic first-order index for
`x3` is exactly 0: it enters the output only through its interaction with `x1`.
The interval is telling you the truth, and the point estimate on its own would
have you reporting a small positive main effect that does not exist. A negative
lower bound is normal here, because the first-order estimator is a difference
of means and is not constrained to be non-negative.

`x1` and `x2` do not overlap, so ranking them is supported. If two intervals
did overlap, the run does not support ranking those two inputs no matter how
far apart the point estimates sit.

`result.ci` records how the interval was made. Store it with the numbers. A
`*_conf` array on its own is an interval of unknown level and unknown method.

## What the interval covers

The bootstrap resamples the rows of `Y` with replacement, recomputes the
indices on each resample, and reads the endpoints off the spread of those
recomputed values. So the interval measures one thing: how much the index moves
when you perturb the sample you already have.

That means it says nothing about three other sources of error.

**Your model.** Every resample uses the same outputs from the same model. If
the model is wrong, the bootstrap will happily give you a tight interval around
a wrong index. Sensitivity analysis apportions the variance of what you
simulated, not of what happens in the world.

**The estimator's bias.** The Sobol estimators are consistent, not unbiased at
finite N. A bias shifts every resample by the same amount, so it never shows up
in the spread. Compare `S1(x3) = 0.0113` above against the analytic 0: the
0.0113 is bias plus noise, and only the noise part is inside the interval.

**The design's structure.** The resampling treats the rows as an independent
sample. They are not. jaxgsa draws a scrambled Sobol' sequence, which is
deliberately more even than random, so the estimator is more accurate than
plain Monte Carlo of the same size. The bootstrap cannot see that, and the
interval comes out too wide. Here is the size of the effect:

```python
design = jaxgsa.sobol.sample(PROBLEM, n_samples=2048, seed=0, verbose=False)
one = jaxgsa.sobol.analyze(
    design, evaluate(design.samples), n_bootstrap=500,
    key=jax.random.key(0), verbose=False,
)
lo, hi = np.asarray(one.S1_conf)
print(f"bootstrap on one design: S1(x1)={one.S1[0]:.4f}  "
      f"95% CI [{lo[0]:.4f}, {hi[0]:.4f}]  width={hi[0] - lo[0]:.4f}")

draws = []
for seed in range(60):
    d = jaxgsa.sobol.sample(PROBLEM, n_samples=2048, seed=seed, verbose=False)
    draws.append(float(jaxgsa.sobol.analyze(d, evaluate(d.samples), verbose=False).S1[0]))
draws = np.array(draws)
print(f"60 fresh designs:        mean={draws.mean():.4f}  sd={draws.std(ddof=1):.4f}  "
      f"2*1.96*sd={2 * 1.96 * draws.std(ddof=1):.4f}")
```

```
bootstrap on one design: S1(x1)=0.2932  95% CI [0.1879, 0.4053]  width=0.2174
60 fresh designs:        mean=0.3151  sd=0.0434  2*1.96*sd=0.1700
```

The analytic value is 0.3139, and the mean over 60 independent designs is
0.3151, so the estimator is essentially unbiased at this size. But the real
95% spread of the estimator is 0.170 wide and the bootstrap claimed 0.217. The
interval is about 28% too wide. That is the safe direction to be wrong in, and
it is worth knowing before you conclude that your study needs four times the
budget.

The honest summary: treat the interval as an upper bound on sampling noise from
this particular design, and treat agreement with a rerun at a different seed as
the real check.

## Narrowing the interval

Raise `n_samples` in `sample()` and run the model again. Nothing else works.
Sobol error falls roughly as the square root of the run count:

```python
for n in (512, 8192, 65536):
    design = jaxgsa.sobol.sample(PROBLEM, n_samples=n, seed=42, verbose=False)
    Y = evaluate(design.samples)
    r = jaxgsa.sobol.analyze(
        design, Y, n_bootstrap=500, key=jax.random.key(0), verbose=False
    )
    lo, hi = np.asarray(r.ST_conf)
    print(f"n_runs={design.n_runs:6d}  ST(x1)={r.ST[0]:.4f}"
          f"  CI [{lo[0]:.4f}, {hi[0]:.4f}]  width={hi[0]-lo[0]:.4f}")
```

```
n_runs=   512  ST(x1)=0.5804  CI [0.2561, 0.9346]  width=0.6786
n_runs=  8192  ST(x1)=0.5551  CI [0.4804, 0.6329]  width=0.1525
n_runs= 65536  ST(x1)=0.5560  CI [0.5222, 0.5859]  width=0.0636
```

512 runs is useless: the interval spans 0.26 to 0.93 and does not even
establish that `x1` explains more than a quarter of the variance. Sixteen times
the budget cuts the width by 4.4x, and another eightfold cut it by 2.4x. So
halving an interval costs roughly four times the model runs. Decide up front
how tight the answer has to be, because you cannot buy the last factor of two
cheaply.

## How many resamples

`n_bootstrap` controls precision of the interval, not precision of the index.
The cost is linear and it is cheap, because a resample recomputes an estimator
over an array you already have and never calls your model.

```python
import time

design = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=42, verbose=False)
Y = evaluate(design.samples)

for nb in (0, 100, 500, 2000, 10000):
    kwargs = {} if nb == 0 else {"key": jax.random.key(0)}
    jaxgsa.sobol.analyze(design, Y, n_bootstrap=nb, verbose=False, **kwargs)  # warm up
    t0 = time.perf_counter()
    r = jaxgsa.sobol.analyze(design, Y, n_bootstrap=nb, verbose=False, **kwargs)
    dt = time.perf_counter() - t0
    if nb == 0:
        print(f"n_bootstrap={nb:6d}  {dt * 1e3:7.1f} ms  no interval")
    else:
        lo, hi = np.asarray(r.S1_conf)
        print(f"n_bootstrap={nb:6d}  {dt * 1e3:7.1f} ms  "
              f"S1(x1) CI [{lo[0]:.4f}, {hi[0]:.4f}]  width={hi[0] - lo[0]:.4f}")
```

```
n_bootstrap=     0      1.4 ms  no interval
n_bootstrap=   100      9.9 ms  S1(x1) CI [0.2551, 0.3806]  width=0.1255
n_bootstrap=   500     36.8 ms  S1(x1) CI [0.2603, 0.3877]  width=0.1274
n_bootstrap=  2000    141.5 ms  S1(x1) CI [0.2642, 0.3909]  width=0.1267
n_bootstrap= 10000    784.9 ms  S1(x1) CI [0.2654, 0.3914]  width=0.1260
```

Timings are from one laptop CPU and exclude the JIT compile, which the warm-up
call absorbs. The pattern is what matters, not the milliseconds.

From 500 upward the endpoints move in the third decimal. 100 is enough to rank
parameters, 500 is enough to report two decimals, and 10000 buys nothing but a
second of compute. Use 500 unless you are plotting the bootstrap distribution
itself.

Two costs do scale, though. The resample loop runs the estimator
`n_bootstrap` times over the full output array, so a `(N, T=100, K=5)` output
multiplies that second by 500. And `keep_replicates=True` stores every draw.

## Keeping the draws

`keep_replicates=True` puts the per-resample index arrays on
`result.ci.replicates`, keyed by index name.

```python
design = jaxgsa.sobol.sample(PROBLEM, n_samples=8192, seed=42, verbose=False)
Y = evaluate(design.samples)

result = jaxgsa.sobol.analyze(
    design, Y, n_bootstrap=500, key=jax.random.key(0),
    keep_replicates=True, verbose=False,
)

draws = result.ci.replicates
print("kept arrays:", sorted(draws))
print("S1 draws shape:", draws["S1"].shape)
print("S2 draws shape:", draws["S2"].shape)

S2 = np.asarray(draws["S2"])
print("S2 draws symmetric:", np.allclose(S2, np.swapaxes(S2, -1, -2), equal_nan=True))
print("S2 draws diagonal:", np.unique(S2[:, 0, 0]))

lo68, hi68 = np.quantile(np.asarray(draws["S1"]), [0.16, 0.84], axis=0)
print("S1 68% CI:", np.round(np.stack([lo68, hi68]), 4).tolist())
print("S1 95% CI:", np.round(np.asarray(result.S1_conf, dtype=float), 4).tolist())
```

```
kept arrays: ['S1', 'S2', 'ST']
S1 draws shape: (500, 3)
S2 draws shape: (500, 3, 3)
S2 draws symmetric: True
S2 draws diagonal: [nan]
S1 68% CI: [[0.2908, 0.4158, -0.0166], [0.3577, 0.4692, 0.0422]]
S1 95% CI: [[0.2603, 0.3939, -0.0422], [0.3877, 0.4989, 0.0661]]
```

The leading axis is the resample. Everything after it is the shape of that
index. Two reasons to keep the draws: re-cut the interval at another level
without re-running the analysis, as the 68% line does above, and plot the
bootstrap distribution when you suspect it is skewed rather than bell-shaped.

New in 1.0, the kept `S2` draws are symmetric with a NaN diagonal, which is
exactly the layout of the reported `S2` and `S2_conf`. Before, the draws held a
raw upper triangle and the two disagreed, so `replicates["S2"][:, 1, 0]` and
`result.S2[1, 0]` were different things. Now `np.quantile` over the draws
reproduces `S2_conf` entry for entry.

They are not free. 1000 resamples of a `(T=100, K=5, D=20)` index array is
80 MB, more than the rest of the result put together. Leave the flag off unless
you are going to use them.

## Endpoint rule and level

`ci_method` chooses how the two endpoints come off the draws.

- `"quantile"` (default) takes the empirical 2.5% and 97.5% quantiles of the
  draws. It follows a skewed bootstrap distribution, which matters near the
  boundaries where an index piles up against 0 or 1.
- `"gaussian"` takes the point estimate plus and minus 1.96 bootstrap standard
  deviations. It is symmetric by construction and closer to the way SALib
  builds its intervals.

`conf_level` is the two-sided level, 0.95 by default. Both settings are
recorded on `result.ci`.

jaxgsa always returns endpoints, never half-widths. SALib reports a half-width,
so a SALib `S1_conf` value is roughly `(hi - lo) / 2` here, not `hi`.

## Shapes

The bootstrap adds a leading axis of length 2 for `[lower, upper]`:

- scalar output: `S1_conf.shape == (2, D)`
- multi-output: `S1_conf.shape == (2, K, D)`
- time-series multi-output: `S1_conf.shape == (2, T, K, D)`

`S2_conf` follows the same rule with two trailing parameter axes, so
`(2, T, K, D, D)` in the time-series case.

## Other things worth knowing

- A `jax.random.key(...)` is required when `n_bootstrap > 0`.
- `sobol.analyze()` standardizes each output slice once over the sample axis
  before the bootstrap starts. The resamples reuse that transformed array;
  they are not re-standardized per resample.
- `n_bootstrap=0` (the default) skips the bootstrap and leaves every `*_conf`
  field and `result.ci` as `None`.
- With `calc_second_order=False` at sampling time, `result.S2` and
  `result.S2_conf` are both `None`.
- The same `n_bootstrap` / `conf_level` / `ci_method` / `key` /
  `keep_replicates` arguments appear on the other methods that bootstrap, so
  Morris `mu_star_conf` and PAWN `pawn_conf` read the same way.

## See also

- [Save and reload a design](/examples/save-load) to bootstrap a stored design.
- [Multi-output and time-series](/examples/multi-output) for the shape rules on
  `(N, K)` and `(N, T, K)` outputs.
- [xarray labeled output](/examples/xarray) for exporting intervals as `_lower`
  and `_upper` dataset variables.
