# Scaling to large problems

Plan a large sensitivity study around three costs: model evaluations, analysis
work, and memory. The limiting cost differs by method, so a parameter count on
its own does not determine what is practical.

Here, $D$ is the number of parameters and $N$ is a method-specific sample
count. Output size also matters: $K$ denotes outputs and $T$ time points.

## Start with the model-run budget

For an expensive model, the sampling design usually costs more than the
sensitivity estimator.

| Method | Model evaluations | Planning consequence |
| --- | --- | --- |
| Sobol | $N(D+2)$, or $N(2D+2)$ with second order | Cost grows linearly in $D$; the stored design grows quadratically because each row also has $D$ columns. |
| Morris | $r(D+1)$ | A low-cost screening pass before a more precise analysis. |
| eFAST | At least $4M^2D(D-1)+D$ | The design grows quadratically in $D$. |
| Kucherenko | $N(2D+1)$ | Similar model-run scaling to second-order Sobol, for dependent inputs. |
| DGSM | $N$ model Jacobians | Cost depends on both input and output dimensions and the autodiff direction. |
| Given-data methods | No new model runs | Analysis time, memory, and sample adequacy become the constraints. |

The `n_samples` argument does not mean the same thing for every sampler.
`sobol.sample(problem, n_samples)` treats it as a total evaluation budget and
chooses a base count. eFAST and Morris expose their curve and trajectory sizes
directly. Inspect `samples.samples.shape[0]` before launching the model.

## Choose a scaling strategy

### Expensive model, many parameters

Screen first. Morris uses a small number of one-at-a-time trajectories. DGSM
is an alternative when the model is written in JAX and its derivatives are
meaningful.

Fix only parameters that are clearly negligible, then use Sobol or another
quantitative method on the reduced problem. See
[Screen first, then quantify](/examples/advanced-workflow).

### Existing simulation data

Use a given-data method, but choose the quantity before choosing the fastest
implementation:

- PCE and HDMR provide variance indices through fitted surrogates.
- Shapley provides one variance allocation per parameter.
- HSIC, PAWN, Borgonovo, and optimal transport measure distributional or
  dependence effects.
- VKOGA provides dependent-input variance quantities through a kernel
  surrogate.

More rows do not automatically make a surrogate suitable. Check fit quality,
especially as $D$ or interaction order increases.

### Many outputs or time points

Pass the full `(N, T, K)` output array instead of calling an estimator in a
Python loop. jaxgsa vectorizes estimator work across output slices and returns
one labelled result.

Compilation has a fixed cost. Repeated calls with the same shapes can reuse
compiled code; a single small scalar analysis may not benefit. Use the
[persistent compilation cache](/guide/configuration#persistent-compilation-cache)
for workloads that run in separate processes.

## Know what grows in memory

### Sampling designs

A first- and total-order Saltelli design has $N(D+2)$ rows and $D$ columns.
The design array therefore grows as $O(ND^2)$ even though the number of model
runs is linear in $D$. Second-order designs use $N(2D+2)$ rows.

The global jaxgsa memory budget controls transient analysis arrays. It does
not shrink a sampling design already returned by `sample()`, or the model
outputs you keep beside it. Estimate those arrays before generating them.

### Pairwise and kernel methods

Some estimators have additional structural costs:

- HSIC constructs sample-by-sample kernels, so its core storage grows with
  $N^2$ and is repeated across parameters and permutations.
- HDMR at `maxorder=2` fits pairwise components, whose count grows with $D^2$.
- PCE basis size grows combinatorially with dimension and polynomial order;
  the implementation reduces the order when the sample budget cannot support
  the requested fit.
- VKOGA performs greedy centre selection and repeated surrogate validation.
- Optimal transport solves one or more transport problems for each parameter
  and output slice.

Reducing a setting changes the analysis, not only its runtime. For example,
lowering HDMR's `maxorder` removes interactions, and lowering HSIC's
`n_perms` coarsens the permutation p-values.

## Measure the workload you will run

Wall time depends on hardware, JAX version, dtype, compilation state, and array
shape. Benchmark the same $N$, $D$, $T$, $K$, bootstrap count, and method
settings that the study will use.

Separate three measurements:

1. **Model evaluation time**, including the sampling design.
2. **First analysis call**, which may include JAX compilation.
3. **Warm analysis calls**, which reuse compiled code for the same shapes.

Report peak memory as well as time. A fast configuration that exceeds device
or host memory is not a usable configuration.

The [benchmarks guide](/guide/benchmarks) records the repository's measured
workloads and reproduction commands. Treat those results as comparisons on
the stated machine, not as fixed package limits.

## Practical checklist

Before starting a large run:

1. Choose the sensitivity quantity with the [method guide](/guide/methods).
2. Calculate the number of model evaluations.
3. Estimate the sampling, output, and estimator arrays.
4. Run a smaller problem once to include compilation and validate shapes.
5. Run it again to measure the warm path.
6. Check uncertainty or surrogate-fit diagnostics before increasing scale.
