# Screen first, then quantify

Most models have more inputs than matter. If your model takes ten minutes a
run, a full Sobol analysis of twenty inputs costs 43,008 runs, which is ten
months of wall clock, and most of that budget goes into resolving parameters
whose true index is 0.0001.

The fix is two passes. Run a cheap screening method on all the inputs, throw
away the ones that do nothing, and spend the real budget on the survivors. This
page runs that workflow end to end on a model with a known answer, so you can
see both what it saves and what it costs you in accuracy. On this example the
saving is 31,928 model runs, and the screened answer is closer to the truth
than the full study it replaced.

Every number below comes from running the code as shown.

## The model

The Sobol G-function is a product of `D` V-shaped factors, one per input. Each
factor carries an importance parameter `a_j`. Setting `a_j = 0` makes input `j`
dominant, and `a_j = 99` makes it nearly inert. That lets us build a test case
with a known answer.

```python
import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa.benchmarks.sobol_g import evaluate

A = (0.0, 0.5, 3.0, 9.0) + (99.0,) * 16
D = len(A)

problem = jaxgsa.Problem.from_dict({f"x{i + 1}": (0.0, 1.0) for i in range(D)})


def model(X):
    return evaluate(X, a=A)
```

Twenty inputs. Four of them (`x1` to `x4`) do the work, at four different
strengths so the screen has to resolve more than a single cliff. Each of the
other sixteen has an analytic total-order index of 0.000092, and all sixteen
together sum to 0.0015 against 1.11 for the four that matter. This is the shape
of a real engineering model more often than not: a handful of drivers and a
long tail of parameters somebody added because they were in the spec.

Nothing tells jaxgsa which is which. The workflow has to find out.

## Pass 1: screen with Morris

Morris measures elementary effects. It walks trajectories through the input
space, changing one parameter at a time, and records how much the output moved.
The cost is `n_trajectories * (D + 1)` runs, linear in the number of inputs, so
screening twenty inputs is cheap.

```python
screen_design = jaxgsa.morris.sample(problem, n_trajectories=40, seed=0)
screen = jaxgsa.morris.analyze(screen_design, model(screen_design.samples))
```

```
jaxgsa.morris.sample: D=20, method=trajectory, n_trajectories=40, num_levels=4, n_expanded=840, n_runs=840, duplicates_removed=0 (0.0%)
jaxgsa.morris.analyze
  problem: D=20 (x1, x2, x3, x4, x5, x6, x7, x8, ... 12 more)
    marginals: uniform=20
    correlation: independent
    output: N=840 runs, T=1 x K=1 output slice
    invalid: none found in 40 trajectories (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1016 s
  results: top 5 of 20 parameters by mu_star
    1. x1   mu_star=2.859
    2. x2   mu_star=2.098
    3. x3   mu_star=0.9231
    4. x4   mu_star=0.3451
    5. x16  mu_star=0.04408
```

840 model runs, and the answer is already visible in the verbose summary: the
top four are `x1` through `x4`, and fifth place is an order of magnitude behind
fourth. Morris ranks; it does not give you a variance share. That is the trade,
and at this stage a ranking is all you need.

Why 40 trajectories? `mu_star` is a mean over one elementary effect per
trajectory, so 40 gives each parameter 40 numbers to average. Here is what
happens if you spend less:

```python
for nt in (5, 10, 20, 40, 80):
    d = jaxgsa.morris.sample(problem, n_trajectories=nt, seed=0, verbose=False)
    r = jaxgsa.morris.analyze(d, model(d.samples), verbose=False)
    ms = np.asarray(r.mu_star)
    o = np.argsort(ms)[::-1]
    ratios = ms[o][:-1] / ms[o][1:]
    cut = int(np.argmax(ratios)) + 1
    print(f"n_trajectories={nt:3d}  {d.n_runs:5d} runs  cut after {cut}  "
          f"drop {ratios.max():.1f}x  keeps {sorted(problem.names[i] for i in o[:cut])}")
```

```
n_trajectories=  5    105 runs  cut after 4  drop 5.9x  keeps ['x1', 'x2', 'x3', 'x4']
n_trajectories= 10    210 runs  cut after 4  drop 8.5x  keeps ['x1', 'x2', 'x3', 'x4']
n_trajectories= 20    420 runs  cut after 4  drop 7.7x  keeps ['x1', 'x2', 'x3', 'x4']
n_trajectories= 40    840 runs  cut after 4  drop 7.8x  keeps ['x1', 'x2', 'x3', 'x4']
n_trajectories= 80   1680 runs  cut after 4  drop 8.1x  keeps ['x1', 'x2', 'x3', 'x4']
```

Even 105 runs finds the right four. The screen is robust when the separation is
this wide, and on this model it is. I still ran 40, because you do not know the
separation before you look, and 840 runs is 2% of the budget you are about to
spend. Read the drop factor as your confidence: 5.9x from a five-trajectory
screen is enough to act on, and anything under about 2x is not.

## Read the ranking and pick a cut

The number to cut on is the largest drop in `mu_star`, not a fixed threshold. A
threshold depends on the output's units. A ratio does not.

```python
mu_star = np.asarray(screen.mu_star)
order = np.argsort(mu_star)[::-1]
ratios = mu_star[order][:-1] / mu_star[order][1:]
cut = int(np.argmax(ratios)) + 1

print("rank  param   mu_star    sigma")
for rank, i in enumerate(order, 1):
    marker = "  <- cut" if rank == cut else ""
    print(f"{rank:>4}  {problem.names[i]:<6} {mu_star[i]:7.4f} {float(screen.sigma[i]):8.4f}{marker}")
print(f"\nlargest drop is {ratios.max():.1f}x, after rank {cut}")
```

```
rank  param   mu_star    sigma
   1  x1      2.8590   3.0820
   2  x2      2.0980   2.3943
   3  x3      0.9231   1.0934
   4  x4      0.3451   0.4227  <- cut
   5  x16     0.0441   0.0494
   6  x6      0.0432   0.0518
   7  x12     0.0423   0.0490
   8  x8      0.0418   0.0481
   9  x19     0.0411   0.0504
  10  x9      0.0393   0.0465
  11  x14     0.0390   0.0483
  12  x10     0.0379   0.0466
  13  x18     0.0355   0.0423
  14  x17     0.0350   0.0427
  15  x13     0.0343   0.0409
  16  x11     0.0341   0.0382
  17  x20     0.0336   0.0396
  18  x5      0.0335   0.0422
  19  x15     0.0330   0.0390
  20  x7      0.0328   0.0381
```

```
largest drop is 7.8x, after rank 4
```

Look at ranks 5 through 20. They run from 0.0441 down to 0.0328, a spread of
1.3x across sixteen parameters, and they are shuffled into no meaningful order.
That flat tail is the signature of a group of inert inputs: what you are
reading there is Morris's own noise floor, not their effects. Ranks 1 to 4
stand clear of it and of each other.

The 7.8x drop after rank 4 is the cut. Take it.

If your own study has no such gap, do not force one. A smooth decline means
every input contributes something, and screening will not help you. Widen the
cut, keep more parameters, or accept the full cost.

`sigma` is along for the ride and worth a glance. It is the spread of the
elementary effects. Here `sigma` is comparable to `mu_star` for every parameter,
which says the effects are strongly non-constant across the input space, which
is exactly right for a multiplicative function where every input scales every
other. Compare `sigma` against `mu_star` for the parameters you keep, and take
a large ratio as a warning that the response is nonlinear and interacting, so
the second pass needs `ST` and not just `S1`.

## Pass 2: quantify with Sobol on the survivors

Build a reduced problem from the four survivors, and a wrapper that holds the
other sixteen at their midpoints.

```python
keep = [int(i) for i in sorted(order[:cut])]
print("keeping:", [problem.names[i] for i in keep])

reduced = jaxgsa.Problem.from_dict({problem.names[i]: (0.0, 1.0) for i in keep})
keep_idx = jnp.asarray(keep)


def reduced_model(X_small):
    X = jnp.full((X_small.shape[0], D), 0.5)
    return model(X.at[:, keep_idx].set(X_small))


quant_design = jaxgsa.sobol.sample(reduced, n_samples=10_240, seed=0)
quant = jaxgsa.sobol.analyze(
    quant_design, reduced_model(quant_design.samples),
    n_bootstrap=500, key=jax.random.key(0),
)
```

```
keeping: ['x1', 'x2', 'x3', 'x4']
jaxgsa.sobol.sample: D=4, mode=second-order, base_n=1024, requested_runs>=10240, n_runs=10240, n_expanded=10240, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=4 (x1, x2, x3, x4)
    marginals: uniform=4
    correlation: independent
    output: N=10240 runs, T=1 x K=1 output slice
    invalid: none found in 1024 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.8601 s
    slice_chunk_size: 1 (resolved from the memory budget)
    bootstrap slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 4 of 4 parameters by ST
    1. x1  ST=0.6909  [0.6219, 0.7652]
    2. x2  ST=0.3551  [0.3122, 0.3992]
    3. x3  ST=0.05625  [0.0487, 0.06462]
    4. x4  ST=0.009263  [0.007952, 0.01068]
```

Sobol cost is `base_n * (2 * D + 2)`, and `base_n` is what sets the accuracy of
each index. With four inputs, `base_n = 1024` costs 10,240 runs. With twenty
inputs the same `base_n` costs 43,008. That factor of 4.2 in the `(2D + 2)`
term is the whole reason screening pays: dropping inputs shortens the design
before you multiply by the sample count.

`n_bootstrap=500` adds intervals for nothing extra in model runs, because the
resamples reuse the outputs you already have. See
[bootstrap confidence intervals](/examples/bootstrap) for what those brackets do
and do not cover.

`jnp.full(..., 0.5)` is the choice to think about. The frozen parameters have to
sit somewhere, and where you put them is an assumption. The midpoint is the
usual default. If a frozen parameter has a nominal operating value, use that
instead, and be aware that the indices you report are conditional on it.

## What it saved, and what it cost

Compare against the study we avoided: the same `base_n` on all twenty inputs.

```python
full_design = jaxgsa.sobol.sample(problem, n_samples=1024 * (2 * D + 2), seed=0, verbose=False)
full = jaxgsa.sobol.analyze(full_design, model(full_design.samples), verbose=False)

from jaxgsa.benchmarks.sobol_g import analytical_indices

ST_true = analytical_indices(A)[1]

print("param  ST screened   ST full-20   ST analytic")
for r, i in enumerate(keep):
    print(f"{problem.names[i]:<5} {float(quant.ST[r]):11.4f} {float(full.ST[i]):12.4f} {ST_true[i]:13.4f}")

screened_runs = screen_design.n_runs + quant_design.n_runs
print()
print(f"screened path: {screen_design.n_runs} Morris + {quant_design.n_runs} Sobol = {screened_runs} runs")
print(f"full path:     {full_design.n_runs} runs")
print(f"saved:         {full_design.n_runs - screened_runs} runs ({full_design.n_runs / screened_runs:.2f}x)")
```

```
param  ST screened   ST full-20   ST analytic
x1         0.6909       0.6877        0.6895
x2         0.3551       0.3647        0.3559
x3         0.0562       0.0581        0.0563
x4         0.0093       0.0091        0.0092

screened path: 840 Morris + 10240 Sobol = 11080 runs
full path:     43008 runs
saved:         31928 runs (3.88x)
```

Read the three columns across. The screened numbers are not a degraded version
of the full ones. For `x1`, `x2` and `x3` they are the closer of the two to the
analytic value, by a small margin. That is not luck: at the same `base_n`, both
runs carry similar Monte Carlo noise, and the screened run is not spending any
of its variance budget on sixteen parameters that contribute nothing.

At ten minutes a run, 31,928 saved runs is 222 days of compute you did not
spend.

The screen is not free of risk, and here is the honest accounting of it:

- The screened study reports nothing about `x5` to `x20`. It cannot, because
  they never varied. You get "below the screening threshold" and nothing more
  precise. If a reviewer wants a number for `x11`, you have to go back.
- The reported indices are conditional on the frozen values. Move `x11` from
  0.5 to 0.9 and the four indices shift, by an amount nobody measured.
- Every index is now a share of a smaller variance. More on that below.

## Freezing changes the level, not the shares

A fair question about step 2: if you hold sixteen inputs still, does the output
still vary the way it did?

```python
key = jax.random.key(1)
X = jax.random.uniform(key, (20000, D))
Y_full = model(X)

X_frozen = jnp.full((20000, D), 0.5).at[:, keep_idx].set(X[:, keep_idx])
Y_frozen = model(X_frozen)

print(f"variance, all 20 varying:      {float(jnp.var(Y_full)):.4f}")
print(f"variance, 16 held at midpoint: {float(jnp.var(Y_frozen)):.4f}")
print(f"mean, all 20 varying:          {float(jnp.mean(Y_full)):.4f}")
print(f"mean, 16 held at midpoint:     {float(jnp.mean(Y_frozen)):.4f}")
cv = lambda y: float(jnp.std(y) / jnp.mean(y))
print(f"relative spread (sd/mean): full {cv(Y_full):.4f}  frozen {cv(Y_frozen):.4f}")
```

```
variance, all 20 varying:      0.5805
variance, 16 held at midpoint: 0.4207
mean, all 20 varying:          1.0120
mean, 16 held at midpoint:     0.8619
relative spread (sd/mean): full 0.7528  frozen 0.7525
```

The frozen model has 28% less variance, which looks alarming until you look at
the mean. Each frozen factor evaluates to `a / (1 + a) = 0.99` at the midpoint,
and sixteen of them multiply to 0.85, which is exactly the drop from 1.0120 to
0.8619. The whole output is scaled by that constant, so the variance drops by
its square, 0.85^2 = 0.72. Nothing else changed: the relative spread holds to
three decimals, 0.7528 against 0.7525.

Sobol indices are variance fractions, so a constant scale factor cancels out.
That is why the screened `ST` column matched the analytic values even though
the frozen model's variance is visibly lower. Worth checking in your own study,
because the cancellation is not guaranteed for a model that is not
multiplicative. If the relative spread moves a lot when you freeze, your
"inert" parameters were not inert.

## The mistake that breaks the screen

Rank by `mu` instead of `mu_star` and the workflow quietly picks the wrong
inputs.

```python
by_mu = np.argsort(np.abs(np.asarray(screen.mu)))[::-1][:4]
by_mu_star = np.argsort(np.asarray(screen.mu_star))[::-1][:4]
print("top 4 by |mu|:    ", [problem.names[i] for i in by_mu])
print("top 4 by mu_star: ", [problem.names[i] for i in by_mu_star])
```

```
top 4 by |mu|:     ['x1', 'x2', 'x3', 'x11']
top 4 by mu_star:  ['x1', 'x2', 'x3', 'x4']
```

`mu` averages the signed elementary effects. Every factor of the G-function is
V-shaped, so an input's effect is negative on one side and positive on the
other and the two cancel in the mean. `x4` falls to seventh place by `|mu|` and
the inert `x11` takes its slot. Screen on `mu_star`, the mean of the absolute
effects, which is what it exists for.

Two more ways to lose a real parameter:

- **Cutting on `S1` from a cheap Sobol run.** An input can have `S1 = 0` and a
  large `ST`, meaning it acts only through interactions. The Ishigami function's
  `x3` is the standard example. Cut on `ST` or on `mu_star`, never on `S1`.
- **Screening one output when you care about several.** `mu_star` for a
  time-series output has shape `(T, K, D)`. A parameter that matters only at
  `t = 0` disappears in a mean over time. Take the maximum over the output
  axes, not the mean.

## When to skip the screen

Screening is not always the right call.

- Fewer than about eight inputs. The `(2D + 2)` factor is small, and the
  screening runs are a real fraction of the total.
- The model is fast enough that 43,008 runs is minutes. Then just run Sobol on
  everything and skip the assumptions.
- No clear gap in the `mu_star` ranking. Cutting an even decline throws away
  something you needed.
- You need a defensible number for every parameter, and "below the screening
  threshold" will not survive review.

## See also

- [Morris screening](/examples/morris) for what `mu`, `mu_star` and `sigma`
  measure and how to plot them.
- [Bootstrap confidence intervals](/examples/bootstrap) for the intervals in the
  second pass.
- [Save and reload a design](/examples/save-load) for splitting either pass
  across sessions, which you will want if each model run is ten minutes.
- [RS-HDMR](/examples/hdmr) for the other way to cut cost: fit a surrogate to
  samples you already have and read the indices off the fit.
