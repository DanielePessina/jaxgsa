# Benchmarks

This page gives measured analysis times for every jaxgsa method. The numbers
show how time changes with the sample count, the number of time points, the
number of outputs and the number of inputs. They apply to the machine and
software below. They are not package limits.

## What is timed

Each case times one call to `analyze`. Sampling and model evaluation are not
in the timed region. The end-to-end table is the only exception: it times
sampling, model evaluation and analysis separately.

The warm time is the median of five calls after the first call. The first
call includes tracing and compilation, so it is reported separately. Each call
waits until all its result arrays are ready before the clock stops.

| Component | Value |
| --- | --- |
| Date | 25 October 2026 |
| Machine | MacBook Pro, Apple M1 Pro, 8 CPU cores, 16 GiB memory, on mains power |
| Operating system | macOS 15.6.1, arm64 |
| Python | 3.12.13 |
| jaxgsa | 1.3.1 development tree, commit `0750177` |
| JAX | 0.11.2, CPU backend |
| NumPy | 2.5.3 |
| Precision | float32 and float64, each in a separate run |
| Memory budget | 2048 MiB, the default of `jaxgsa.config.set_memory_budget` |
| Isolation | One fresh subprocess per case, killed above 3 GiB resident memory or after 600 s |

No GPU or TPU result follows from these CPU measurements.

## Results at a glance

All figures are warm float32 times unless the text says otherwise.

- Sobol analysis at $D=8$ and base $N=4096$ (40,960 runs, scalar output)
  takes 0.7 ms.
- Sobol analysis of 1000 time points at the same size takes 78 ms in float32
  and 163 ms in float64.
- A full Sobol study of 81,920 runs with 1000 outputs per run takes 0.26 s in
  float32 and 0.44 s in float64. This time includes sampling, model evaluation
  and analysis. The first run, with compilation, takes 0.44 s and 0.61 s.
- With 5000 output slices per run ($N=4096$, $T=100$, $K=50$), the warm
  analysis times are:

| Method | Time for 5000 slices |
| --- | ---: |
| morris | 21 ms |
| rank | 97 ms |
| pce, shapley | 135 ms |
| dgsm | 0.65 s |
| efast | 0.74 s |
| sobol | 1.03 s |
| kucherenko | 1.10 s |
| optimal_transport | 1.30 s |
| pawn | 2.13 s |
| borgonovo | 7.21 s |
| vkoga | 42.5 s |
| hdmr | killed above the 3 GiB cap |
| hsic | timed out after 600 s |

- The first call of a new shape takes between 0.06 s (kucherenko) and 3.8 s
  (vkoga) at $N=4096$. Later calls with the same shapes reuse the compiled
  code.

## The model

All cases use the Sobol G-function from `jaxgsa.benchmarks.sobol_g` with $D$
inputs, each uniform on $[0, 1]$. The coefficients are
$a = (0, 1, 4.5, 9, 99, 99, \ldots)$, so four inputs are important and the
other inputs are almost inert.

A wrapper adds $T$ time points and $K$ outputs per run:

$$
y(x, t, k) = g(x)\, e^{-t\,(0.2 + x_2)}\,(1 + 0.25\,k) + x_3 \sin(t + 0.5\,k)
$$

Here $t$ takes $T$ equally spaced values on $[0, 10]$ ($t = 1$ when $T = 1$),
and $k = 0, 1, \ldots, K-1$. Each slice has its own indices, so no slice is a
copy of another. The output has shape $(N,)$ for one slice, $(N, T)$ for
$K = 1$, and $(N, T, K)$ otherwise.

## Method settings

$N$ has a different meaning for each family of methods:

- **sobol, kucherenko:** $N$ is the base sample size. The design has
  $N(D+2)$ runs. Sobol does not compute second-order indices.
- **efast:** $N$ is the number of points per curve.
- **morris:** the design has $N/16$ trajectories.
- **Given-data methods:** $N$ is the number of Monte Carlo rows.

Other settings are the defaults, except these:

- hsic: 50 permutations.
- pce: order 3.
- hdmr: order 2.
- shapley: PCE backend, order 3.
- dgsm: the call receives the model function. The time includes the
  derivatives that JAX computes from it.

## Results

All values are warm median analysis times. Values are in milliseconds unless
they show a unit. The last column gives the float64 time at the largest size.

These words mark cells that did not give a time:

- **skipped:** the case did not start. HSIC and VKOGA build $N \times N$
  matrices, so the sweep skips them above $N = 8192$. Other skipped cases
  had input and output arrays too large for the 3 GiB cap.
- **timed out:** the case ran for more than 600 s. Each HSIC slice is a full
  kernel analysis, so HSIC with many slices does not finish in this limit.
- **over cap:** the case used more than 3 GiB of resident memory and was
  stopped.

### Sample count

$D=8$, scalar output.

| Method | $N=1024$ | $N=4096$ | $N=16384$ | $N=65536$ | float64, $N=65536$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| sobol | 0.6 | 0.7 | 1.4 | 3.4 | 4.6 |
| kucherenko | 0.6 | 0.8 | 2.0 | 6.2 | 6.1 |
| efast | 0.5 | 0.6 | 1.4 | 4.2 | 5.3 |
| morris | 0.6 | 0.5 | 0.6 | 0.6 | 0.7 |
| dgsm | 1.4 | 1.4 | 1.9 | 3.4 | 4.8 |
| borgonovo | 2.0 | 5.2 | 9.0 | 38.7 | 58.5 |
| pawn | 1.8 | 3.0 | 7.0 | 27.1 | 27.1 |
| hsic | 166.1 | 1.32 s | skipped | skipped | skipped |
| optimal_transport | 1.5 | 5.3 | 8.1 | 28.9 | 30.1 |
| rank | 0.9 | 3.7 | 5.0 | 16.1 | 16.7 |
| pce | 3.7 | 9.1 | 27.8 | 108.1 | 229.7 |
| hdmr | 22.1 | 43.6 | 198.0 | 801.7 | 937.2 |
| shapley | 3.8 | 9.3 | 27.8 | 109.0 | 239.0 |
| vkoga | 819.4 | 1.54 s | skipped | skipped | skipped |

At $N=4096$ in float64, hsic takes 2.25 s and vkoga takes 4.30 s.

### Time points

$D=8$, $N=4096$, $K=1$.

| Method | $T=1$ | $T=10$ | $T=100$ | $T=1000$ | float64, $T=1000$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| sobol | 0.8 | 1.3 | 6.7 | 78.3 | 163.0 |
| kucherenko | 0.8 | 2.5 | 17.9 | 171.2 | 121.1 |
| efast | 0.6 | 1.5 | 7.4 | 94.7 | 162.5 |
| morris | 0.4 | 0.5 | 1.0 | 4.4 | 7.3 |
| dgsm | 1.5 | 2.3 | 7.0 | 54.7 | 133.6 |
| borgonovo | 5.2 | 17.9 | 117.2 | 1.29 s | 2.56 s |
| pawn | 3.0 | 9.4 | 55.8 | 674.6 | 648.4 |
| hsic | 1.32 s | 7.68 s | 71.3 s | timed out | timed out |
| optimal_transport | 5.3 | 9.3 | 27.0 | 234.6 | 258.8 |
| rank | 3.7 | 4.0 | 5.5 | 20.9 | 35.7 |
| pce | 9.3 | 10.2 | 19.0 | 35.8 | 81.3 |
| hdmr | 41.3 | 119.7 | 403.6 | 2.50 s | 496.5 |
| shapley | 9.1 | 9.8 | 19.5 | 36.6 | 83.0 |
| vkoga | 1.52 s | 1.62 s | 2.05 s | 7.34 s | 12.38 s |

PCE, Shapley and VKOGA fit one surrogate for all slices, so their time grows
slowly with $T$. In float64, HSIC also timed out at $T=100$.

### Outputs per time point

$D=8$, $N=4096$, $T=100$. $K=50$ gives 5000 slices per run.

| Method | $K=1$ | $K=10$ | $K=50$ | float64, $K=50$ |
| --- | ---: | ---: | ---: | ---: |
| sobol | 6.4 | 103.4 | 1.03 s | skipped |
| kucherenko | 17.0 | 190.3 | 1.10 s | skipped |
| efast | 7.5 | 101.3 | 738.2 | skipped |
| morris | 0.9 | 5.1 | 21.1 | 31.8 |
| dgsm | 7.2 | 89.5 | 647.2 | 934.1 |
| borgonovo | 114.0 | 1.29 s | 7.21 s | 13.30 s |
| pawn | 53.6 | 663.5 | 2.13 s | 2.61 s |
| hsic | 70.9 s | timed out | timed out | timed out |
| optimal_transport | 27.1 | 243.6 | 1.30 s | 1.36 s |
| rank | 5.4 | 22.0 | 97.0 | 180.4 |
| pce | 19.8 | 38.8 | 134.6 | 293.1 |
| hdmr | 399.7 | 2.52 s | over cap | over cap |
| shapley | 19.6 | 38.6 | 135.4 | 282.8 |
| vkoga | 2.05 s | 7.25 s | 42.5 s | 53.5 s |

In float64 at $K=50$, the estimated design, output and temporary arrays need
about 4.6 GiB for Sobol and Kucherenko and 3.7 GiB for eFAST. These cases did
not start. In float64 at $K=10$, sobol takes
189.1 ms and hdmr takes 517.0 ms.

### Number of inputs

$N=4096$, scalar output. At $D=128$, a Sobol design has 532,480 runs.

| Method | $D=4$ | $D=8$ | $D=16$ | $D=32$ | $D=64$ | $D=128$ | float64, $D=128$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sobol | 0.6 | 0.8 | 1.0 | 1.2 | 1.7 | 2.9 | 3.6 |
| kucherenko | 0.6 | 0.8 | 1.3 | 2.1 | 3.9 | 6.0 | over cap |
| efast | 0.5 | 0.6 | 1.0 | 1.2 | 2.0 | 9.6 | 10.9 |
| morris | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.7 | 0.8 |
| dgsm | 1.4 | 1.4 | 1.7 | 1.9 | 2.7 | 3.6 | 4.7 |
| borgonovo | 3.4 | 5.3 | 6.1 | 9.9 | 16.2 | 27.6 | 41.1 |
| pawn | 2.2 | 3.1 | 4.8 | 5.3 | 9.0 | 15.9 | 20.0 |
| hsic | 897.5 | 1.31 s | 2.45 s | 7.68 s | 12.00 s | 21.43 s | 45.49 s |
| optimal_transport | 3.1 | 5.2 | 5.5 | 6.1 | 8.3 | 15.2 | 15.3 |
| rank | 2.0 | 3.7 | 3.9 | 4.3 | 6.7 | 12.6 | 12.4 |
| pce | 3.1 | 9.0 | 119.7 | 48.0 | 10.4 | 20.9 | 27.0 |
| hdmr | 14.7 | 42.1 | 214.1 | 1.15 s | 16.74 s | 93.26 s | over cap |
| shapley | 3.5 | 9.2 | 120.1 | 48.1 | 10.7 | 21.5 | 26.9 |
| vkoga | 945.5 | 1.55 s | 2.61 s | 6.11 s | 19.66 s | 62.57 s | 69.07 s |

Read the PCE, Shapley and HDMR rows with these points:

- PCE and Shapley lower the polynomial order when the term count is too
  large for $N$. The order falls from 3 to 2 at $D=32$ and to 1 at $D=64$.
  This is why $D=16$ is the slowest case. At $D \geq 64$ both methods warn
  that the fit explains less than half of the output variance.
- HDMR of order 2 has more coefficients than samples at $D \geq 32$, and it
  warns about this. The times at large $D$ measure a fit that $N=4096$
  cannot support.
- VKOGA warns at $D \geq 32$ that its surrogate misses most of the output
  variation.

### First call

$D=8$, $N=4096$, scalar output. The first call includes tracing,
compilation and one analysis.

| Method | float32 | float64 |
| --- | ---: | ---: |
| sobol | 167 ms | 143 ms |
| kucherenko | 58 ms | 56 ms |
| efast | 127 ms | 134 ms |
| morris | 191 ms | 168 ms |
| dgsm | 445 ms | 471 ms |
| borgonovo | 324 ms | 316 ms |
| pawn | 373 ms | 376 ms |
| hsic | 2.08 s | 3.12 s |
| optimal_transport | 275 ms | 263 ms |
| rank | 214 ms | 117 ms |
| pce | 946 ms | 1.01 s |
| hdmr | 2.13 s | 2.02 s |
| shapley | 1.02 s | 1.09 s |
| vkoga | 3.78 s | 6.52 s |

For most methods, the compilation part of the first call changes little
across the sizes in this sweep. When the analysis itself is slow, the first
call is approximately the warm time plus the compilation time.

### End to end

Sobol, $D=8$, base $N=8192$ (81,920 runs), $T=1000$. The output has shape
$(81920, 1000)$. The model is a jitted JAX function.

| Precision | Sample | Evaluate | Analyze | Total | First run total |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 13.5 ms | 39.8 ms | 205.0 ms | 258.4 ms | 436.3 ms |
| float64 | 13.8 ms | 91.2 ms | 330.4 ms | 436.0 ms | 614.4 ms |

### Memory

The memory budget controls how jaxgsa splits work into chunks. It is not a
hard limit on process memory. Some HDMR, VKOGA and HSIC cases used more
resident memory than the 2 GiB budget plus the input data, up to 2.9 GiB.
The cases marked "over cap" used more than 3 GiB.

## What the benchmarks do not show

- Model evaluation and sampling are excluded. For an expensive simulator,
  estimator speed may have little effect on total study time.
- First calls include tracing and compilation. The warm figures apply after
  compiled programs are reusable for the same shapes.
- The sweep uses one model and fixed method settings. It does not rank the
  scientific value or accuracy of different sensitivity measures.
- Surrogate timings do not establish surrogate quality. Check the fit before
  interpreting PCE, HDMR, Shapley, or VKOGA indices.
- Results on other hardware and software versions will differ.

## Reproduce the results

The sweep script is not part of the package. To repeat the measurement, use
the model and method settings above. Sweep one size at a time from the base
case $D=8$, $N=4096$, $T=1$, $K=1$. Run each case in a new Python process,
so that the first call includes compilation and memory belongs to one case
only. Time one call to `analyze`, then the median of five more calls, and
block on every result array before you stop the clock. Use an idle machine on
mains power. Record the commit, hardware, dependency versions, backend,
precision and memory budget with the results.

## Analytical benchmark functions

`jaxgsa.benchmarks` ships five models with analytical or published reference
indices. Use them to test estimator accuracy separately from runtime.

| Module | Parameters | Marginals | Useful check |
| --- | ---: | --- | --- |
| `ishigami` | 3 | Uniform $[-\pi, \pi]$ | Separates a zero first-order effect from a non-zero total effect |
| `sobol_g` | 8 | Uniform $[0,1]$ | Provides clear influential and nearly inert parameter tiers |
| `linear` | 3 | Uniform $[0,1]$ | Has equal first/total indices and zero interactions |
| `gaussian_linear` | 3 | Gaussian | Includes analytical Borgonovo and optimal-transport indices |
| `oakley_ohagan` | 15 | Gaussian | Tests a flatter, interaction-rich sensitivity profile |

Each module provides a `PROBLEM`, batched `evaluate(X)`, and analytical or
published arrays. Read the module API for non-default parameters and the
[method comparison example](/examples/method-comparison) for a worked
accuracy comparison.
