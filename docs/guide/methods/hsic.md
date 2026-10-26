# HSIC: kernel dependence from a sample you already have

HSIC scores how strongly each input and the output depend on each other. It
does not split the output variance into shares. It maps inputs and outputs
into a reproducing kernel Hilbert space, a space of functions where a kernel
(a smooth similarity between two points) plays the role of an inner product,
and measures how far the joint distribution sits from the product of the
marginals. Dependence that a correlation coefficient reads as zero shows up
there: nonlinear, non-monotone, and heteroscedastic effects alike.
Heteroscedastic means the spread of the output changes with the input, not
only its average.

HSIC is a given-data method. You pass whatever (X, Y) pairs you already have.
No Saltelli design, no extra model runs, no re-running the simulator. If you
have a 5000-row log from a Monte Carlo campaign that was run for some other
purpose, that log is a valid HSIC input.

Two indices come back:

- **R2-HSIC**, the first-order view. `HSIC(x_i, Y)` divided by
  `sqrt(HSIC(x_i, x_i) * HSIC(Y, Y))`, so it lands in [0, 1] and reads like a
  kernel correlation coefficient.
- **T**, the total-order view. The fraction of the joint dependence that
  is lost when x_i is removed, so it also counts influence carried through
  interactions with other inputs.

Neither of these is a variance fraction. See
[R2-HSIC does not sum to 1](#r2-hsic-does-not-sum-to-1) before you read the
numbers as percentages.

## Import style

```python
from jaxgsa import hsic
# hsic.analyze(...)
```

`monte_carlo` lives in `jaxgsa.sampling`, not in `jaxgsa.hsic`. Call it as
`jaxgsa.sampling.monte_carlo()`.

## A first run on Ishigami

Ishigami is a three-input test function with a published answer, which makes
it a good place to learn to read a new index. The run below draws a plain
Monte Carlo sample, evaluates the model on it, and analyzes the pairs. Any
sampler works here. That is the whole point of a given-data method.

`analyze` needs a `key`. The permutation test draws random shuffles, so there
is no sensible default, and omitting it raises
`ValueError: key is required for the permutation test`. Pass
`jax.random.key(0)` when all you want is reproducibility.

```python
import jax
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=2048, seed=42))
Y = evaluate(X)

result = jaxgsa.hsic.analyze(PROBLEM, X, Y, key=jax.random.key(0))

print("R2: ", result.R2)
print("T:  ", result.T)
print("p_values:", result.p_values)
print("hsic_raw:", result.hsic_raw)
print("sum of R2:", float(result.R2.sum()))
```

```
jaxgsa.hsic.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=2048 runs, T=1 x K=1 output slice
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 6.441 s
    n_perms: 200
    bandwidth: 1.0 (median-heuristic multiplier)
  results: top 3 of 3 parameters by T
    1. x1  T=0.8192
    2. x3  T=0.1662
    3. x2  T=0.05517
R2:  [0.17024454 0.01001067 0.02882455]
T:   [0.8192088  0.05517024 0.16619332]
p_values: [0.00497512 0.00497512 0.00497512]
hsic_raw: [0.01507932 0.00089368 0.00253559]
sum of R2: 0.20907975733280182
```

The block above the arrays is the verbose summary. `verbose=True` is the 1.0
default on every `analyze`, so you get it without asking. Read the top three
lines as a receipt of what was analyzed: three uniform inputs, no declared
correlation, 2048 rows, no non-finite values. The `bandwidth: 1.0` line is
the kernel-width multiplier, covered
[below](#bandwidth-is-a-multiplier-not-a-width). Pass `verbose=False` for a
silent run.

Each of the four arrays has one entry per input, in the order the parameters
appear in `PROBLEM`. Read them together.

`R2` ranks x1 first, then x3, then x2. That ranking disagrees with the
published first-order Sobol indices for Ishigami, which put x2 first (0.442),
then x1 (0.314), then x3 (0.000). The disagreement is real and it is not a
bug. Sobol asks how much variance an input explains; x2 enters Ishigami
through a large `7 sin^2(x2)` term, so it owns a lot of variance. HSIC asks
how much a kernel can tell x2 apart from noise at the chosen bandwidth, and
`sin^2` is an even, oscillating function that a wide Gaussian kernel smooths
away. Change the bandwidth and x2 comes back, as the next section shows.

`T[i]` sits above `R2[i]` for every input here. The gap is the
dependence carried through interactions. On x1 it is enormous (0.170 to
0.819), which is correct: x1 multiplies the `x3^4` term, so almost every
interaction in this model runs through it.

`hsic_raw` is the unnormalized statistic on the kernels' own scale. Use it to
check that two runs did the same thing, not to rank inputs.

## R2-HSIC does not sum to 1

The last printed line is 0.209. That is not an error and there is nothing
missing. R2-HSIC indices are individual dependence measures, one per input,
each independently normalized to [0, 1]. They are not shares of anything, so
they do not add to 1, and the residual is not "unexplained variance". If you
need numbers that partition the output variance, use
[Sobol](/examples/basic) or [PCE](/guide/methods/pce). If you want a number that
partitions the output *distribution*, no such thing exists here either;
[Borgonovo delta](/guide/methods/borgonovo) and [PAWN](/guide/methods/pawn) are also
per-input measures.

The practical consequence: compare R2-HSIC values against each other, and
against the p-values, and never against 1.

## Bandwidth is a multiplier, not a width

`bandwidth` scales the median heuristic. It does not replace it. The
Gaussian's standard deviation is `bandwidth * sqrt(m)`, where `m` is the
median of the off-diagonal squared pairwise distances of that variable. So
`1.0` is the plain heuristic, `0.5` halves every width, and `2.0` doubles
them. There is deliberately no way to pass an absolute width: one fixed
number cannot be right for an input mapped to [0, 1] and an output measured
in megapascals at the same time.

The bandwidth decides which effects HSIC can see, so sweep it when a ranking
surprises you.

```python
for b in (0.5, 1.0, 2.0):
    r = jaxgsa.hsic.analyze(PROBLEM, X, Y, key=jax.random.key(0),
                            bandwidth=b, verbose=False)
    print(f"bandwidth={b}: R2={r.R2}  T={r.T}")
```

```
bandwidth=0.5: R2=[0.10485408 0.0712513  0.03178859]  T=[0.5324838  0.38032433 0.1935405 ]
bandwidth=1.0: R2=[0.17024454 0.01001067 0.02882455]  T=[0.8192088  0.05517024 0.16619332]
bandwidth=2.0: R2=[0.21804838 0.00577381 0.0124179 ]  T=[0.9236818  0.02538207 0.06159621]
```

This is the failure mode to know about. At `bandwidth=2.0` the kernel is wide
enough that x2's oscillation averages out inside it, and x2's total index
falls to 0.025. Halve the width and x2 jumps to 0.380, second place, close to
where a variance-based method puts it. Nothing about the model changed. The
resolution of the measuring instrument did.

The rule I follow: if two bandwidths half an order of magnitude apart give
the same ranking, believe the ranking. If they do not, the model has
structure at a scale the default kernel cannot resolve, and the narrower
bandwidth is usually the informative one. Keep the bandwidth pinned when you
compare runs at different N, or the width moves with the sample and the runs
stop being comparable.

## What the p-value tests

The p-value comes from a permutation test. `analyze` shuffles one input
column against the output many times, which destroys the pairing and makes
the two independent by construction, and recomputes HSIC on each shuffle.
The p-value is the fraction of shuffles whose statistic reaches or beats the
real one, with the Phipson-Smyth correction so it is never reported as
exactly 0.

The null hypothesis is "this input and the output are independent". A small
p-value says the measured dependence is bigger than shuffling produces by
chance. It says nothing about how big the effect is. Read the p-value to
decide whether an input belongs in the analysis at all, then read R2-HSIC to
decide how much it matters.

HSIC takes no `n_bootstrap` and returns no confidence interval, unlike almost
every other method here. That is deliberate. HSIC is a V-statistic, a double
sum over all `N^2` pairs of rows, so a row bootstrap repeats rows, the
repeats land on the kernel diagonal where the kernel equals 1, and the
resampled index is biased upward by construction. The interval would mix that
bias into the sampling spread it was supposed to show. The permutation test
has no such problem, so it is the uncertainty statement HSIC gives you.

In the Ishigami run above all three p-values came back at 0.004975. That is
the floor, `1 / (n_perms + 1)` with the correction applied, and it means only
"smaller than the test can resolve". To see a p-value that carries
information, add an input the model ignores:

```python
problem = jaxgsa.Problem.from_dict(
    {"x1": (-3.14159, 3.14159), "x2": (-3.14159, 3.14159),
     "x3": (-3.14159, 3.14159), "inert": (-3.14159, 3.14159)}
)
X4 = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n_samples=2048, seed=42))
Y4 = evaluate(X4[:, :3])          # the model never reads `inert`

r = jaxgsa.hsic.analyze(problem, X4, Y4, key=jax.random.key(0))
print("R2: ", r.R2)
print("p_values:", r.p_values)
```

```
jaxgsa.hsic.analyze
  problem: D=4 (x1, x2, x3, inert)
    marginals: uniform=4
    dependence: independent
    output: N=2048 runs, T=1 x K=1 output slice
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 5.263 s
    n_perms: 200
    bandwidth: 1.0 (median-heuristic multiplier)
  results: top 4 of 4 parameters by T
    1. x1     T=0.8092
    2. x3     T=0.1708
    3. x2     T=0.05837
    4. inert  T=0.01559
R2:  [0.16443375 0.00859426 0.02879651 0.00051297]
p_values: [0.00497512 0.00497512 0.00497512 0.80597013]
```

`inert` scores p = 0.806: four shuffles in five produced a statistic at least
as large as the real one, which is exactly what an independent input should
do. Its R2-HSIC of 0.00051 is not zero, and it never will be at finite N.
The p-value is what tells you 0.00051 is noise. Meanwhile x2's R2-HSIC of
0.0086 is also small, and its p-value says that one is real. Two similar-looking
small numbers, opposite verdicts. This is the reason to print the p-values
next to the indices rather than eyeballing a cutoff on the indices.

`n_perms` sets the resolution. The default 200 resolves down to about 0.005
and costs 200 extra kernel evaluations per input. Raise it if you need to
separate p = 0.001 from p = 0.004; there is rarely a reason to.

## Multiple outputs

When the model returns K outputs, every index array gains a leading output
axis and comes back shaped `(K, D)`. The output order is the order in
`output_names`, so name your outputs and you never have to count columns.

```python
import jax
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
    output_names=("linear", "quadratic"),
)
X = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, n_samples=2048, seed=42))
Y = jnp.column_stack([
    X @ jnp.array([1.0, 2.0, 3.0]),
    jnp.sum(X**2, axis=1),
])

result = jaxgsa.hsic.analyze(problem, X, Y, key=jax.random.key(0))
print("R2 shape:", result.R2.shape)
print(result.R2)
```

```
jaxgsa.hsic.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    dependence: independent
    output: N=2048 runs, T=1 x K=2 output slices
    invalid: none found in 2048 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 7.946 s
    n_perms: 200
    bandwidth: 1.0 (median-heuristic multiplier)
  results: top 3 of 3 parameters by T, mean over 2 output slices
    1. x3  T=0.5441
    2. x2  T=0.3587
    3. x1  T=0.2289
R2 shape: (2, 3)
[[0.05313455 0.18776193 0.5290061 ]
 [0.22435622 0.23674537 0.22942361]]
```

One `analyze` call covered both outputs, and both outputs cost one kernel
build over the same X. Row 0 is `"linear"` and row 1 is `"quadratic"`.

Row 0 ranks the inputs 3 > 2 > 1, matching the coefficients (1, 2, 3) in the
linear combination. Row 1 is almost flat at 0.23, which is correct for a sum
of squares that treats all three inputs alike.

Note the header line: `top 3 of 3 parameters by T, mean over 2 output
slices`. The verbose table averages across outputs to produce a single
ranking. That average is a convenience for a quick look. When two outputs
disagree, as they do here, read the full array instead.

## Selecting by name with xarray

`to_dataset()` attaches the parameter and output names as coordinates, so you
can index by name instead of counting positions.

```python
ds = result.to_dataset()
print(ds)
print(ds.R2.sel(param="x1"))
print(ds.p_values.sel(output="linear"))
```

```
<xarray.Dataset> Size: 192B
Dimensions:   (output: 2, param: 3)
Coordinates:
  * output    (output) <U9 72B 'linear' 'quadratic'
  * param     (param) <U2 24B 'x1' 'x2' 'x3'
Data variables:
    R2        (output, param) float32 24B 0.05313 0.1878 0.529 ... 0.2367 0.2294
    T         (output, param) float32 24B 0.08565 0.3215 0.717 ... 0.396 0.3711
    p_values  (output, param) float32 24B 0.004975 0.004975 ... 0.004975
    hsic_raw  (output, param) float32 24B 0.004868 0.01734 ... 0.02192 0.02094
Attributes:
    bandwidth:  1.0
    n_perms:    200

<xarray.DataArray 'R2' (output: 2)> Size: 8B
array([0.05313455, 0.22435622], dtype=float32)
Coordinates:
  * output   (output) <U9 72B 'linear' 'quadratic'
    param    <U2 8B 'x1'

<xarray.DataArray 'p_values' (param: 3)> Size: 12B
array([0.00497512, 0.00497512, 0.00497512], dtype=float32)
Coordinates:
  * param    (param) <U2 24B 'x1' 'x2' 'x3'
    output   <U9 36B 'linear'
```

The first `sel` picks x1 across both outputs, the second picks all inputs for
the `"linear"` output. For a scalar output the dataset has a `param`
dimension only.

## Shape rules

N is the number of samples, T the number of time steps, K the number of
outputs, and D the number of inputs.

| `Y` shape | `R2`, `T`, `p_values`, `hsic_raw` |
|---|---|
| `(N,)` | `(D,)` |
| `(N, K)` | `(K, D)` |
| `(N, T, K)` | `(T, K, D)` |

D is always the last axis.

## Memory: reduce N, or screen first

HSIC works on full `N x N` kernels. It holds only as many input kernels at
once as the [memory budget](/guide/configuration#memory-budget) allows and
builds the rest again when it needs them. It always needs about six `N x N`
float arrays. At N = 10000 in float32 one such array is 400 MB, so a small
budget cannot go below about 2.4 GB.

**There is no `batch_size` keyword on `hsic.analyze`.** That is not an oversight. A row-blocked kernel build was
implemented, measured, and removed: it bounded a transient of the build while
the resident stacks stayed exactly as large, so it bought nothing. If a
sample does not fit, you have two honest options.

Thin N. HSIC converges quickly in N, and a ranking that is stable between
N = 2000 and N = 4000 will not change at N = 20000. Take a random subset and
check that the ranking holds.

Or lower D by screening first. Run [Morris](/guide/methods/morris) or
[PAWN](/guide/methods/pawn), both cheap and linear in memory, drop the inputs they
find inert, and give HSIC the survivors. The cost is linear in D and quadratic
in N, so halving D halves it while halving N quarters it. Thin the
sample first if you only need one of the two.

## Other things worth knowing

`analyze` maps each input through its marginal CDF to [0, 1] before building
the kernel, so bandwidths are comparable across inputs of different units.
The marginal CDF of an input maps its values to the probability of drawing
something smaller.

The indices do not depend on the output units, because the median heuristic
carries the output's own scale: `Y` and `a*Y + b` give the same answer. An
output of extreme magnitude is still worth rescaling by hand, because the
squared distances the kernel builds can overflow float32 long before the
index would care. `(Y - Y.mean(0)) / Y.std(0)` changes nothing else.

float32 is safe for HSIC. jaxgsa computes the index as one sum over the
product of two centred kernels, so no large sums cancel. On Ishigami up to
N = 4096, float32 stays within 3e-6 relative of float64 for every parameter.
This includes a dummy input with R2 near 6e-4. The p-values are the same in
both precisions.

Correlated inputs are fine. HSIC is a dependence measure and assumes nothing
about input independence, so a declared `problem.correlation` does not
invalidate the indices. Each index then reads as total association with the
output, which includes influence borrowed from correlated partners. An input
the model never reads can score above 0 when it correlates with one the model
does read. That reading is correct. If you need the direct effect separated
from the borrowed one, use [VKOGA](/guide/methods/vkoga) or
[Kucherenko](/guide/methods/kucherenko).

Above about D = 15 the total-order product kernel can underflow in float32.
Enable float64 or screen the input list down.

## See also

- [Basic example](/examples/basic) for the Sobol variance decomposition and
  its Saltelli design.
- [PAWN](/guide/methods/pawn) and [Borgonovo delta](/guide/methods/borgonovo), the
  other two given-data methods that read the whole output distribution.
- [Optimal transport](/guide/methods/optimal-transport) for a given-data index
  that splits into a location shift and a reshape.
- [Methods](/guide/methods) for the theory and a side-by-side comparison.
- [API reference](/api/#given-data-methods) for every parameter.
