# Basic Example (Ishigami)

The Ishigami function is the right first example because you already know the
answer. Its Sobol indices have a closed form, so every number this page prints
can be put next to the exact value. That is the honest way to learn what a
sensitivity estimate is worth at a given sample size.

$$
f(x) = \sin(x_1) + A \sin^2(x_2) + B x_3^4 \sin(x_1),
\qquad x_i \sim U[-\pi, \pi]
$$

with $A = 7$ and $B = 0.1$. The third term is the interesting one. It contains
`x3`, but only multiplied by $\sin(x_1)$. Change `x3` on its own and the average
output does not move. Change `x3` while `x1` is away from zero and it moves a
lot.

Three numbers describe that. `S1` is the share of output variance an input
explains on its own. `ST` is the share it explains on its own plus every
interaction it joins. `S2` is the share carried by one pair acting together.

## Minimal Sobol run

Four steps.

1. Build a Saltelli design. The Sobol estimators need a specific pattern of
   rows, not free-form random points, so the design comes from
   `jaxgsa.sobol.sample` and not from your own sampler.
2. Run your model on every row. This is the only expensive step, which is why
   the sampler returns unique rows only.
3. Pass the design object and the outputs to `jaxgsa.sobol.analyze`. The design
   carries the layout metadata, so the analyzer knows which row is which.
4. Read `S1`, `ST` and `S2` off the result.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

design = jaxgsa.sobol.sample(
    PROBLEM,
    n_samples=4096,
    seed=42,
    calc_second_order=True,
)

Y = evaluate(design.samples)

result = jaxgsa.sobol.analyze(design, Y)
```

Both calls print, because `verbose=True` is the default in 1.0:

```text
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=512, requested_runs>=4096, n_runs=4096, n_expanded=4096, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4096 runs, T=1 x K=1 output slice
    invalid: none found in 512 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.4485 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.6266
    2. x2  ST=0.44
    3. x3  ST=0.2423
```

The line worth stopping on is `base_n=512`. You asked for 4096 model runs, and
a second-order Saltelli design spends $2D + 2 = 8$ rows per base point, so those
4096 runs buy only 512 independent base points. Every index below is a Monte
Carlo mean over 512 samples. Pass `verbose=False` to silence all of this.

## The numbers against the exact answer

```python
import numpy as np

from jaxgsa.benchmarks.ishigami import ANALYTICAL_S1, ANALYTICAL_ST, ANALYTICAL_S2

np.set_printoptions(precision=4, suppress=True)
print("S1     ", np.asarray(result.S1))
print("exact  ", ANALYTICAL_S1)
print("ST     ", np.asarray(result.ST))
print("exact  ", ANALYTICAL_ST)
print("S2")
print(np.asarray(result.S2))
```

```text
S1      [0.3387 0.4421 0.0155]
exact   [0.3139 0.4424 0.    ]
ST      [0.6266 0.44   0.2423]
exact   [0.5576 0.4424 0.2437]
S2
[[    nan -0.0356  0.2128]
 [-0.0356     nan  0.0054]
 [ 0.2128  0.0054     nan]]
```

Read it one input at a time. The array order is `x1`, `x2`, `x3`, the order
`PROBLEM` declares.

`x2` lands almost exactly: `S1` 0.4421 against 0.4424, and `ST` 0.4400 against
the same 0.4424. `S1` and `ST` agree, so `x2` acts alone and joins nothing. That
is right. The $A\sin^2(x_2)$ term never touches another input.

`x1` gets `S1` 0.3387 against 0.3139 and `ST` 0.6266 against 0.5576. Both are
too high by roughly 0.03 and 0.07. The gap between `S1` and `ST` is the real
signal. About half of what `x1` does to the output, it does through an
interaction.

`x3` is the one that decides whether a method is worth using. Its exact `S1` is
zero, and the estimate is 0.0155. Its exact `ST` is 0.2437, and the estimate is
0.2423. Freeze `x3` at a nominal value because its main effect is nil and you
throw away a quarter of the output variance. Any screening method that reads
main effects alone would tell you to do exactly that.

`S2` is a symmetric matrix with NaN down the diagonal, because a parameter
paired with itself is not a second-order term. The only nonzero entry in the
exact answer is the `x1`-`x3` pair at 0.2437, estimated here as 0.2128. The two
pairs that should be zero come out at -0.0356 and 0.0054. A negative Sobol index
is impossible, so that -0.0356 is pure estimator noise, and it tells you the
size of the noise floor. Anything under about 0.04 in this run means nothing.

## Watching the error shrink

`SobolSamples.downsample()` prefix-slices a design to a smaller power-of-two
`base_n` and slices the matching output rows with it. So one expensive
evaluation gives you the whole convergence curve, with no re-simulation.

```python
big = jaxgsa.sobol.sample(PROBLEM, n_samples=327_680, calc_second_order=False, seed=42)
Y_big = np.asarray(evaluate(big.samples))

print(f"{'base_n':>7} {'model runs':>11} {'max |S1 err|':>13} {'max |ST err|':>13}")
for base_n in (64, 256, 1024, 4096, 16384, 65536):
    design_n, Y_n = big.downsample(base_n, Y_big)
    r = jaxgsa.sobol.analyze(design_n, Y_n, verbose=False)
    s1 = np.max(np.abs(np.asarray(r.S1) - ANALYTICAL_S1))
    st = np.max(np.abs(np.asarray(r.ST) - ANALYTICAL_ST))
    print(f"{base_n:>7} {design_n.n_runs:>11} {s1:>13.5f} {st:>13.5f}")
```

```text
 base_n  model runs  max |S1 err|  max |ST err|
     64         320       0.07518       0.04392
    256        1280       0.04077       0.07091
   1024        5120       0.01309       0.00257
   4096       20480       0.00152       0.00152
  16384       81920       0.00111       0.00037
  65536      327680       0.00000       0.00001
```

Four things in that table are worth arguing about.

The error is not monotone. `base_n` 256 has a worse `ST` error than 64 does.
Monte Carlo error is a random variable, and on a single seed a step can go the
wrong way. Never conclude anything from one convergence step on one seed.

Between `base_n` 1024 and 4096 the error drops by roughly a factor of eight for
a factor of four in cost, which beats the $1/\sqrt{n}$ you would get from plain
random sampling. That is the scrambled Sobol' sequence doing its job. Between
16384 and 65536 the `S1` error falls from 1.1e-3 to 4.5e-6, a factor of 250,
which is luck rather than a rate.

Useful rankings arrive early. At `base_n=64`, 320 model runs, the largest error
is 0.075, and yet both `S1` and `ST` already order the three inputs correctly.
If all you need is which knob matters, you can stop far sooner than if you need
the index value itself.

The run at the top of this page used `base_n=512`, which sits between the first
two rows. Its errors of 0.02 to 0.07 are exactly what the table predicts. Nothing
was wrong with it. It was small.

## Free Morris screening from the same run

A Saltelli design already contains a Morris radial design. Inside each base
point, row `A` and each row `AB_j` differ in exactly one parameter, which is the
definition of an elementary effect. So you can read screening measures off a
design you have already paid for.

```python
morris = jaxgsa.morris.analyze(design.to_morris(), Y)
```

```text
jaxgsa.sobol.SobolSamples.to_morris: D=3, mode=second-order, base_n=512, blocks=512, effects=1536, reusing n_runs=4096 existing evaluations (0 new model runs)
jaxgsa.morris.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=2048 runs, T=1 x K=1 output slice
    invalid: none found in 512 trajectories (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1195 s
  results: top 3 of 3 parameters by mu_star
    1. x2  mu_star=15
    2. x1  mu_star=8.681
    3. x3  mu_star=6.595
```

`mu_star` is the mean absolute elementary effect and it ranks inputs by
importance. `sigma` is the spread of those effects for one input.

```python
print("mu_star:", np.asarray(morris.mu_star))
print("sigma:  ", np.asarray(morris.sigma))
```

```text
mu_star: [ 8.681  14.9952  6.5952]
sigma:   [12.5216 19.9888 11.3866]
```

Morris ranks `x2` first, then `x1`, then `x3`. Sobol's `ST` ranks `x1` first.
The two disagree because `mu_star` is a mean absolute derivative in raw output
units and `ST` is a variance share, and those are different questions. Do not
treat the Morris numbers as a confirmation of the Sobol numbers either. They
come from the same model outputs, so their errors are correlated.

What `sigma` adds is the ratio. Here `sigma` exceeds `mu_star` for all three
inputs, between 1.3 and 1.7 times. An input whose elementary effects vary more
than they average is nonlinear, interacting, or both. On a model with no
analytic answer that ratio is often the first warning that a linear screening
argument will not hold. [Morris](/examples/morris) has the derivation and the
caveats, under "Morris measures from a Saltelli design, for free".

## Practical caveats

- Evaluate `design.samples`, not an expanded Saltelli matrix. `jaxgsa` rebuilds
  the expanded layout internally.
- `calc_second_order=False` cuts the rows per base point from $2D+2$ to $D+2$,
  so the same budget buys more base points. Use it whenever you do not need
  `S2`. `result.S2` is then `None`.
- `sample()` may raise `base_n` internally until the deduplicated matrix holds
  at least `n_samples` unique rows. The verbose line reports the `base_n` it
  settled on, and that is the number that governs your error.
- `downsample()` needs a power-of-two target at or below the design's own
  `base_n`. It cannot go up.
- `design.samples` is a plain NumPy array, so `np.savetxt("samples.csv",
  design.samples, delimiter=",")` hands the design to another process.

## Next examples

- [Non-Uniform Inputs](/examples/non-uniform-inputs) when your inputs are not
  all uniform, and what goes wrong when you pretend they are.
- [Save and Reload Samples](/examples/save-load) for persisting `SobolSamples`
  and its reconstruction metadata.
- [Bootstrap Confidence Intervals](/examples/bootstrap) to get the noise floor
  from the data instead of from an analytic answer you will not have.
- [Multi-Output & Time-Series](/examples/multi-output) for `(N, T, K)` outputs.
- [xarray Labeled Output](/examples/xarray) for selecting results by name.
- [RS-HDMR Example](/examples/hdmr) for the surrogate route, which works on any
  `(X, Y)` pair rather than a structured design.
- [Screen first, then quantify](/examples/advanced-workflow) for cutting 20 inputs to 4 with Morris,
  then spending the Sobol budget on the survivors.

For the theory behind the estimators, read [Methods](/guide/methods).
