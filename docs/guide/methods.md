# Methods

jaxgsa implements thirteen methods for global sensitivity analysis (GSA). All of them answer the same broad question: which parameters actually drive my model's output? They differ in three ways. They measure different quantities, they cost different numbers of model evaluations, and some need a dedicated sampling design while others work with data you already have.

If you are new to the package, start with [Choosing a Method](#choosing-a-method), then jump to the section for the method you picked. Every method section opens with what it measures, when to pick it, and what data it needs. The estimator details follow.

Throughout this page, $D$ is the number of parameters and $N$ is a sample count.

## Choosing a Method

Three questions narrow the field quickly.

1. Can you still choose where to run the model? Four methods need their own sampling design, which jaxgsa generates for you: Sobol' (Saltelli matrices), eFAST (search curves), Morris (trajectories), and Kucherenko (conditional-copula blocks for dependent parameters). The other nine are given-data methods: HDMR, PCE, Shapley effects, DGSM, HSIC, PAWN, Borgonovo delta, optimal transport, and VKOGA. They accept any set of $(X, Y)$ pairs, including simulation runs you already have. DGSM has no sampler of its own: draw plain Monte Carlo points with `jaxgsa.sampling.monte_carlo` and let autodiff do the rest.

2. What should the number mean? Variance-based methods (Sobol', HDMR, PCE, eFAST, Shapley) report fractions of output variance, as in "parameter 3 explains 40% of the output's spread". Screening methods (Morris, DGSM) trade that precision for cheap, reliable rankings. Moment-independent methods (HSIC, PAWN, Borgonovo delta, optimal transport) measure how strongly a parameter affects the whole output distribution. Use them when your output is skewed or heavy-tailed and variance is the wrong summary. Optimal transport also splits its index into a mean-shift part and a shape-change part.

3. What is your evaluation budget? Sobol' needs $N(2D+2)$ model runs by default, with $N$ typically 1024 or more. Morris needs only $r(D+1)$, with $r \approx 10\text{–}50$ trajectories. DGSM gets the whole gradient for roughly the price of one evaluation per sample point, for JAX-differentiable models only. The given-data methods cost nothing beyond the runs you already have.

Common situations:

- "I can run the model freely and want the standard variance decomposition." Use [Sobol' via Saltelli sampling](#sobol-indices-via-saltelli-sampling), the reference method, with first-order, total-order, and second-order indices.
- "My model is expensive and has many parameters." Screen first with [Morris](#morris-elementary-effects-screening) at $r(D+1)$ runs, or with [DGSM](#dgsm-derivative-based-global-sensitivity-measures) if the model is JAX-differentiable. Fix the negligible parameters, then spend the remaining budget on Sobol' for the survivors.
- "I only have existing simulation data." Any given-data method works. Use [HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) or [PCE](#pce-polynomial-chaos-expansion) for variance-based indices via a surrogate, or [VKOGA](#vkoga-correlated-input-variance-indices) when the parameters are dependent. Use [HSIC](#hsic-hilbert–schmidt-independence-criterion), [PAWN](#pawn-cdf-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), or [optimal transport](#optimal-transport-wasserstein-based-sensitivity) for distribution-based indices.
- "My parameters are correlated." Sobol', PCE, eFAST, DGSM, Morris, and PCE-backed Shapley all assume independent parameters, and they refuse to run when `problem.correlation` is declared. The full menu has three routes. To generate correlated samples, declare a Gaussian-copula matrix on the `Problem` (`correlation=`, or `problem.with_correlation(R)`) and draw with `jaxgsa.sampling.monte_carlo`. To analyze data you already have, use [VKOGA](#vkoga-correlated-input-variance-indices) for variance fractions split into correlated and uncorrelated parts via a kernel surrogate, HDMR for the ANCOVA separation of structural and correlative variance, or [optimal transport](#optimal-transport-wasserstein-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), HSIC, and PAWN, which make no independence assumption. `shapley.analyze(backend="hdmr", include_correlative=True)` allocates the HDMR decomposition, but HDMR's `ST` is not a total-effect index under dependence — see the [HDMR section](#rs-hdmr-random-sampling-high-dimensional-model-representation). To run your model on a dedicated design, use [Kucherenko](#kucherenko-dependent-input-sobol-indices): conditional-copula sampling that evaluates the actual model and returns $S_1$/$S_T$ under the declared dependence. See [Correlated Inputs](/examples/correlated-inputs).
- "Some of my parameters are categorical." Declare them with `{"dist": "categorical", "probs": [...]}`, and samples then carry integer level codes. Four methods handle unordered levels correctly: [Sobol'](#sobol-indices-via-saltelli-sampling), because the Saltelli column-swap scheme is distribution-agnostic, plus [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), [optimal transport](#optimal-transport-wasserstein-based-sensitivity), and [PAWN](#pawn-cdf-based-sensitivity), which all condition on one class per level. Every other method refuses with a `ValueError`, because its indices would depend on the arbitrary code order. See [Categorical Inputs](/examples/categorical-inputs).
- "I need to decide what to measure more accurately, or what to hold fixed." Use [VKOGA](#vkoga-correlated-input-variance-indices): $S_{TC}$ is the prioritisation measure and $S_{TU}$ the fixing measure. Under dependence they can rank parameters very differently.
- "My output distribution is skewed or heavy-tailed." Use [PAWN](#pawn-cdf-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), or [optimal transport](#optimal-transport-wasserstein-based-sensitivity). All three compare whole output distributions rather than variances.
- "I want to know how a parameter matters: shift or shape?" Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity). Its index decomposes exactly into an advective (mean-shift, $= S_1/2$) and a diffusive (spread/shape) component.
- "I want one number per parameter for a whole trajectory." Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) with `mode="trajectory"`. Point-cloud transport scores each parameter against the entire time course jointly.
- "I want one fair importance number per parameter that sums to 1." Use [Shapley effects](#shapley-effects).
- "I also want a fast surrogate of my model." Use HDMR or PCE and call `result.predict(...)`.

### Method capabilities

This table is the one place that records what each method accepts. The other
pages link here instead of repeating it. `tests/test_docs_matrix.py` checks
every cell against the method registry, so the table cannot fall behind the
code.

| Method | Reports | Own design | Correlated | Categorical | Bootstrap CI |
|---|---|:--:|:--:|:--:|---|
| [`borgonovo`](#borgonovo-delta-density-based-sensitivity) | $\delta$, $S_1$ | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`dgsm`](#dgsm-derivative-based-global-sensitivity-measures) | bounds on $S_T$ | ✗ | ✗ | ✗ | — |
| [`efast`](#efast-extended-fourier-amplitude-sensitivity-test) | $S_1$, $S_T$ | ✓ | ✗ | ✗ | — |
| [`hdmr`](#rs-hdmr-random-sampling-high-dimensional-model-representation) | $S_a$ / $S_b$ / $S$ per term, surrogate | ✗ | ✓ † | ✗ | `n_bootstrap` |
| [`hsic`](#hsic-hilbert–schmidt-independence-criterion) | dependence measure | ✗ | ✓ § | ✗ | — |
| [`kucherenko`](#kucherenko-dependent-input-sobol-indices) | $S_1$, $S_T$ under dependence | ✓ | ✓ | ✗ | — |
| [`morris`](#morris-elementary-effects-screening) | $\mu^*$, $\sigma$ | ✓ | ✗ | ✗ | `n_bootstrap` |
| [`optimal_transport`](#optimal-transport-wasserstein-based-sensitivity) | $W_2^2$ index, advective + diffusive | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pawn`](#pawn-cdf-based-sensitivity) | KS distance | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pce`](#pce-polynomial-chaos-expansion) | $S_1$, $S_2$, $S_T$, surrogate | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`shapley`](#shapley-effects) | allocation summing to 1 | ✗ | ✗ ‡ | ✗ | `n_bootstrap` |
| [`sobol`](#sobol-indices-via-saltelli-sampling) | $S_1$, $S_2$, $S_T$ | ✓ | ✗ | ✓ | `n_bootstrap` |
| [`vkoga`](#vkoga-correlated-input-variance-indices) | $S_{TC}$, $S_{TU}$, $S_U$, $S_C$, $S_{IU}$, surrogate | ✗ | ✓ | ✗ | — |

**Own design** means the method builds its own sample matrix, so you must be
able to run the model at the points it chooses. The other nine are given-data
methods. They accept any $(X, Y)$ pairs you already have.

**Correlated** and **Categorical** say what the method does with a problem
that declares a Gaussian-copula correlation, or that declares a categorical
parameter. A ✗ is a refusal, not a silent approximation. The method raises a
`ValueError` that names the parameters and the alternatives.

**Bootstrap CI** gives the keyword that asks for bootstrap confidence
intervals. Two spellings are in use. Eight methods report no intervals at all
and show a —. See [Confidence intervals](/api/#confidence-intervals) for the
`result.ci` record that comes back with them.

† HDMR handles dependence through its ANCOVA decomposition: $S_a$ is the
structural share and $S_b$ the correlative share. Its $S_T$ is the SCSA
convention and is not a total-effect index under dependence. See the
[HDMR section](#rs-hdmr-random-sampling-high-dimensional-model-representation).

‡ The default `backend="pce"` assumes independent parameters and refuses. The
table records that default. `shapley.analyze(backend="hdmr")` does accept a
correlated problem, and with `include_correlative=True` it allocates the
ANCOVA decomposition.

§ Correlation-inclusive: a parameter that does not enter the model, but that
correlates with one that does, scores above zero. That is the correct reading
of these indices, not an estimation error. Use HDMR ($S_a$ and $S_b$), VKOGA,
or Kucherenko when you must separate the structural effect from the
correlation-induced one.

### Comparison table

The rest of the differences, method by method. The capability columns above are not repeated here.

| Consideration | Sobol' | HDMR | PCE | Shapley | eFAST | DGSM | Morris | HSIC | PAWN | Borgonovo delta | Optimal transport | VKOGA | Kucherenko |
|---------------|--------|------|-----|---------|-------|------|--------|------|------|-----------------|------------------|-------|------------|
| Sampling requirement | Structured Saltelli design, $N(2D+2)$ evaluations (default) | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Search curves, $N \times D$ evaluations | Plain MC, $N$ evaluations + autodiff | Trajectory or radial design, $r(D+1)$ evaluations (deduplicated) | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Conditional-copula blocks, $N(2D+1)$ evaluations |
| Parameter distributions | Uniform + Gaussian | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian | Uniform + Gaussian (both backends) | Uniform + Gaussian | Uniform + Gaussian (+ truncated Normal) | Uniform + Gaussian (truncated-quantile grid) | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian (via CDF mapping) | Any (rank-based classes; marginals not used) | Any (rank-based classes; marginals not used) | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian (latent-copula inverse CDF) |
| Output shapes | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series (both backends) | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series; joint point-cloud modes over outputs/time | Scalar, multi-output, time-series | Scalar, multi-output, time-series |
| What the numbers mean | Exact variance fractions (given enough samples) | Variance fractions from a B-spline surrogate (fit-dependent) | Variance fractions from a polynomial surrogate (fit-dependent) | Exact allocation within the fitted surrogate; depends on fit quality | Exact variance fractions (given enough samples) | Bounds on $S_T$, not exact indices | Screening ranks ($\mu^*$ as $S_T$ proxy), not variance fractions | Dependence measure, not variance fractions | Distributional (KS) distance, not variance fractions | Distributional (L1) distance, not variance fractions | Distributional ($W_2^2$) distance in $[0,1]$, split into mean-shift + shape parts | Correlated and uncorrelated variance fractions from a kernel surrogate (fit-dependent) | Exact conditional-variance fractions under the declared dependence (given enough samples) |
| Second-order indices | Direct estimation from cross-matrices | From interaction component functions | Analytical from coefficients | Not available (interaction variance folded into $\mathrm{Sh}$) | Not available | Not available | Not available | Not available | Not available | Not available | Not available | Not available | Not available |
| Interaction detection | Via $S_2$ and the gap $S_T - S_1$ | Via explicit interaction component functions | Via $S_2$ from coefficients | Via the gaps $\mathrm{Sh} - S_1$ and $S_T - \mathrm{Sh}$ | Via the gap $S_T - S_1$ only | Not available (bounds only) | Via large $\sigma$ relative to $\mu^*$ (not pair-attributable) | Via the Total HSIC − R2-HSIC gap | Not available (first-order only) | Not available (the $\delta - S_1$ gap flags influence beyond first-order variance) | Not available (the diffusive component flags influence beyond mean shift) | Via $S_{IU}$, the independent-interaction index | Via the gap $S_T - S_1$ under independence (under correlation the gap mixes interactions and coupling) |
| Reusable surrogate | No | Yes (`result.predict`) | Yes (`result.predict`) | Derived from either fitted result | No | No | No | No | No | No | No | Yes (`result.predict`) | No |

## Background: Variance-Based Sensitivity Analysis

### Why Global Sensitivity Analysis?

Local sensitivity methods, such as partial derivatives at a nominal point, describe the model at one location. Global sensitivity analysis explores the entire parameter space instead. This matters for non-linear models, where interactions and non-monotonic responses mean a gradient at one point can be misleading. GSA quantifies each parameter's contribution to output uncertainty across the whole parameter domain.

In practice, GSA serves several roles:

- **Parameter identifiability**: parameters with near-zero sensitivity across all outputs are effectively unidentifiable from data and may need to be fixed rather than estimated; high-sensitivity parameters are the ones data can constrain.
- **Experimental design**: for time-series outputs, watching sensitivity indices evolve over time helps pick measurement times when outputs are most informative about the parameters of interest.
- **Model simplification**: if interaction indices are negligible, the model response is approximately additive, and simpler surrogate models may suffice.

### The Hoeffding–Sobol' Decomposition

The theoretical foundation of variance-based GSA is the Hoeffding (ANOVA) decomposition. Any square-integrable function $f(\mathbf{X})$ of $D$ independent parameters can be uniquely decomposed into summands of increasing dimensionality:

$$
f(\mathbf{X}) = f_0 + \sum_{i=1}^{D} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots + f_{1,2,\ldots,D}(X_1, \ldots, X_D)
$$

where $f_0 = \mathbb{E}[f(\mathbf{X})]$ is the overall mean, each $f_i$ captures the main effect of parameter $i$, each $f_{ij}$ captures the pairwise interaction between $i$ and $j$, and so on. Because these component functions are mutually orthogonal, the total output variance decomposes additively:

$$
\mathrm{Var}(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,D}
$$

where $V_i = \mathrm{Var}[f_i(X_i)]$, $V_{ij} = \mathrm{Var}[f_{ij}(X_i, X_j)]$, etc.

### Sobol' Sensitivity Indices

Dividing each variance component by $\mathrm{Var}(Y)$ yields the Sobol' sensitivity indices.

The first-order index $S_i$ is the fraction of output variance you could remove by fixing parameter $i$ at its true value. It is the main effect of parameter $i$, ignoring interactions:

$$
S_i = \frac{V_i}{\mathrm{Var}(Y)} = \frac{\mathrm{Var}_{X_i}[\mathbb{E}_{\mathbf{X}_{\sim i}}(Y \mid X_i)]}{\mathrm{Var}(Y)}
$$

The second-order index $S_{ij}$ is the additional variance from the pairwise interaction between $i$ and $j$, beyond their individual main effects:

$$
S_{ij} = \frac{V_{ij}}{\mathrm{Var}(Y)}
$$

The total-order index $S_{T_i}$ is the fraction of output variance parameter $i$ is involved in at all, counting every interaction it participates in. A parameter with $S_{T_i} \approx 0$ can safely be fixed:

$$
S_{T_i} = \frac{\mathbb{E}_{\mathbf{X}_{\sim i}}[\mathrm{Var}_{X_i}(Y \mid \mathbf{X}_{\sim i})]}{\mathrm{Var}(Y)} = 1 - \frac{\mathrm{Var}_{\mathbf{X}_{\sim i}}[\mathbb{E}_{X_i}(Y \mid \mathbf{X}_{\sim i})]}{\mathrm{Var}(Y)}
$$

where $\mathbf{X}_{\sim i}$ denotes all parameters except $X_i$. By construction, $S_{T_i} \geq S_i$ always holds, with equality when parameter $i$ has no interactions. The gap $S_{T_i} - S_i$ quantifies how much of parameter $i$'s influence comes through interactions.

## Sobol' Indices via Saltelli Sampling

Sobol' indices split the output variance into the share each parameter owns alone and the share it owns through interactions. This is the reference method and jaxgsa's default workflow: an exact, model-free variance decomposition with well-understood convergence.

Pick it when you can afford a dedicated sampling design and your parameters are independent. The method needs its own design, so you must be able to run the model at points jaxgsa chooses. jaxgsa uses the Saltelli sampling scheme (Saltelli 2002, 2010), which arranges quasi-random sample matrices so that first-order ($S_1$), total-order ($S_T$), and second-order ($S_2$) indices can all be estimated from a single batch of model evaluations.

### The Saltelli Column-Swap Scheme

The method generates two independent $N \times D$ quasi-random sample matrices $\mathbf{A}$ and $\mathbf{B}$ using a Sobol' low-discrepancy sequence (via `scipy.stats.qmc.Sobol`). For each parameter $j$, a cross-matrix $\mathbf{AB}^{(j)}$ is constructed by taking all columns from $\mathbf{A}$ except column $j$, which is replaced by column $j$ from $\mathbf{B}$. This column-swap construction allows conditional expectations to be estimated via sample averages.

The cost is $N(D + 2)$ model evaluations for all first-order and total-order indices, or $N(2D + 2)$ when second-order indices are included (`calc_second_order=True`, the default).

### Estimators

The default estimator pair is `estimator="saltelli-jansen"`.

First-order, the improved form of Sobol' et al. (2007), tabulated by Saltelli et al. (2010):

$$
\hat{S}_i = \frac{\frac{1}{N}\sum_{n=1}^{N} f(\mathbf{B})_n \cdot \left(f(\mathbf{AB}^{(i)})_n - f(\mathbf{A})_n\right)}{\mathrm{Var}(Y)}
$$

Total-order, from Jansen (1999):

$$
\hat{S}_{T_i} = \frac{\frac{1}{2N}\sum_{n=1}^{N}\left(f(\mathbf{A})_n - f(\mathbf{AB}^{(i)})_n\right)^2}{\mathrm{Var}(Y)}
$$

These two normalise by a pooled output variance computed over the concatenation of $\mathbf{A}$ and $\mathbf{B}$ outputs, that is $\mathrm{Var}([\mathbf{A}; \mathbf{B}])$ over $2N$ points. Pooling both base-sample vectors doubles the effective sample size and gives a more robust variance estimate.

#### Choosing a different estimator

`jaxgsa.sobol.analyze(..., estimator=...)` and `jaxgsa.sobol.indices(..., estimator=...)` accept six named pairs. All six converge to the same indices. They differ in how much sampling noise they carry at a small $N$, and in what they can report.

| `estimator` | First order | Total order | Design |
|---|---|---|---|
| `"saltelli-jansen"` (default) | Sobol' et al. (2007) | Jansen (1999) | $N(D+2)$ |
| `"jansen"` | Jansen (1999) | Jansen (1999) | $N(D+2)$ |
| `"janon-monod"` | Monod et al. (2006), Janon et al. (2014) | same | $N(D+2)$ |
| `"martinez"` | Martinez (2011) | Martinez (2011) | $N(D+2)$ |
| `"mauntz-kucherenko"` | Sobol' et al. (2007) | Sobol' et al. (2007) | $N(D+2)$ |
| `"azzini-rosati"` | Azzini, Mara & Rosati (2021) | same | $N(2D+2)$ |

`"azzini-rosati"` reads the $\mathbf{BA}^{(j)}$ blocks, so it needs a design drawn with `calc_second_order=True`. Asking for it on a first-order-only design raises a `ValueError`.

Second-order indices always use the Saltelli (2002) pairwise formula. Only the $S_1$ terms it subtracts come from the estimator you chose.

#### Which one to use

The defaults were measured, not assumed. On Ishigami and Sobol-G against their analytical indices, over 100 seeds per point, with each estimator given the design it actually needs so the model-run budget is comparable:

- For the $N(D+2)$ design, `"saltelli-jansen"` is the best or joint-best choice at every budget tested. Its first-order formula ties with `"mauntz-kucherenko"` and beats `"jansen"`, `"janon-monod"` and `"martinez"` by about a factor of two on Sobol-G, where four parameters are nearly inert. Its total-order formula is the best or joint-best everywhere; `"mauntz-kucherenko"`'s total order is the worst of the menu, by up to a factor of three at a small $N$.
- `"azzini-rosati"` is the better choice when your budget is tight, when you have many parameters, or when you are already paying for the $N(2D+2)$ design. On Sobol-G at 640 model runs its $S_1$ error is 0.029 against 0.087 for the default, and it is the only estimator that cannot report $S_1 > S_T$. Its advantage narrows as the budget grows, and on Ishigami above roughly 5000 runs the default overtakes it, because the cheaper design buys twice as many base points and the quasi-random design converges faster than $1/\sqrt{N}$.
- `"janon-monod"` and `"martinez"` give nearly identical numbers. Pick `"martinez"` if you want the point estimate that OpenTURNS' `MartinezSensitivityAlgorithm` computes, which jaxgsa reproduces to machine precision.

#### Negative index estimates

A first-order estimate can come out below zero, whichever estimator you choose. That is expected, and it is not a sign that the sample is too small.

Every first-order formula here is a difference of two correlated Monte Carlo estimates. The difference is unbiased but noisy, so when the true index is near zero the sampling error is bigger than the index and about half the estimates land below it. Owen (2013) states the mechanism: the cross-moment form "has very large variance when $\tau^2_u \ll \mu^2$", whereas a squared-difference form "is a sum of squares, hence nonnegative".

Note the limit of that last point. Only the *total* order of `"saltelli-jansen"` and `"jansen"` is a bare sum of squares, and only it is guaranteed non-negative. A Jansen *first-order* estimate is one minus such a term, so it is bounded above by 1 and free to go below zero. In jaxgsa's measurements on Sobol-G, every estimator except `"azzini-rosati"` returns a negative first-order value somewhere in 15% to 35% of runs, and that fraction does not fall away as $N$ grows.

So read a negative value as "the interval covers zero", and turn on the bootstrap (`num_resamples`) to see that directly. Investigate only if the value is large, if it appears for a parameter whose index is demonstrably not near zero, or if it grows with $N$.

jaxgsa does not clip. Clipping to zero is a display choice, and it must never be done before ranking: it biases upward in exactly the near-zero regime where the ranking decision is being made.

### How to use it

1. `jaxgsa.sobol.sample()` generates the Sobol' quasi-random sequence and builds the Saltelli cross-matrices. Duplicate rows are removed so your model only evaluates unique sample points.
2. You evaluate your model on `sampling_result.samples`.
3. `jaxgsa.sobol.analyze()` reconstructs the Saltelli layout internally and computes all indices in a single `jit(vmap(...))` pass.

`jaxgsa.sobol.analyze()` always standardizes each output slice over the sample axis before it computes the estimators. The Saltelli/Sobol'-Mauntz $S_1$ estimator and every $S_2$ estimator are uncentred products. A non-zero output mean therefore biases them. The standardization removes that bias. SALib standardizes in the same way. When bootstrapping (`num_resamples > 0`), `ci_method="quantile"` reports percentile bootstrap bounds and `ci_method="gaussian"` reports symmetric bounds from the bootstrap standard deviation. Either way, jaxgsa returns explicit lower/upper endpoint arrays rather than SALib's symmetric confidence widths.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | Fraction of output variance due to parameter $i$ alone (main effect). |
| $S_T(i)$ | Fraction of output variance due to parameter $i$ including all its interactions. $S_T \geq S_1$ always. |
| $S_2(i,j)$ | Fraction of output variance due to the pairwise interaction between $i$ and $j$, beyond their individual effects. |

### When to use it

- You can afford the structured Saltelli design: $N(D+2)$ evaluations for first-order and total-order only, or $N(2D+2)$ with second-order (the default)
- You want an exact, model-free variance decomposition
- Your parameters are independent

## RS-HDMR (Random Sampling High-Dimensional Model Representation)

RS-HDMR is a variance-based method that works from data you already have. It fits a B-spline surrogate to any set of $(X, Y)$ pairs. It then derives sensitivity indices analytically from the surrogate's variance decomposition.

Pick it in three situations. Model runs are expensive and you want to reuse existing data. Your parameters may be correlated. Or you also want a fast emulator of the model. No sampling design is required.

### Theoretical Background

High-Dimensional Model Representation (HDMR) exploits the observation that, for many practical problems, only the low-order interactions among parameters significantly influence the output. The RS-HDMR variant constructs component functions from randomly sampled parameter and output data, rather than requiring structured grids. The model is decomposed as:

$$
f(\mathbf{X}) \approx f_0 + \sum_{i} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \sum_{i<j<k} f_{ijk}(X_i, X_j, X_k)
$$

where each component function is expanded in a B-spline basis and fitted via backfitting with Tikhonov regularisation.

### ANCOVA Decomposition

The classical Sobol' decomposition assumes independent parameters. RS-HDMR instead uses an ANCOVA (analysis of covariance) decomposition, which separates each component's variance into two parts:

- **Structural variance ($S_a$)**: the contribution that would remain if all parameters were independent — analogous to the classical Sobol' index.
- **Correlative variance ($S_b$)**: the additional contribution arising from correlations between parameters.

This distinction matters because many real-world models have correlated parameters, for example coupled physical parameters. Conflating structural and correlative contributions can produce misleading sensitivity rankings.

### How to use it

1. You provide any set of $(X, Y)$ pairs — no sampling design required.
2. `jaxgsa.hdmr.analyze()` maps parameters to $[0, 1]$ via their marginal CDFs, builds B-spline basis matrices, and fits component functions via backfitting with Tikhonov regularisation.
3. The ANCOVA decomposition splits each component's variance into structural ($S_a$) and correlative ($S_b$) parts. Total-order indices ($S_T$) sum contributions from all terms involving a given parameter.

The surrogate is trained on the outputs you supply. `result.predict(...)` and
`result.rmse` are on that same scale. There is no inverse transform.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_a(t)$ | Structural (uncorrelated) variance contribution of term $t$. For first-order terms with independent parameters, equivalent to Sobol' $S_1$. |
| $S_b(t)$ | Correlative variance contribution of term $t$ (due to parameter correlations). |
| $S(t)$ | Total contribution per term: $S_a + S_b$. |
| $S_T(i)$ | SCSA total per parameter: $S_T(i) = \sum_{u \ni i} (S_a(u) + S_b(u))$, the sum of $S$ over every term that contains parameter $i$ (Li et al., 2010, Section 2.2.3). Equal to the Sobol' total-order index when the parameters are independent. Read the warning below when they are not. |

::: warning HDMR's total under correlated inputs
With correlated parameters, HDMR's $S_T$ is not a Sobol' total-order index. It is the SCSA total that Li et al. (2010) define in Section 2.2.3, from the per-term indices of their Eqs. (19)-(22): the sum of $S_a + S_b$ over every term that contains the parameter. It is the same convention SALib uses. Read it as a term-membership sum and nothing more.

It can be negative, because $S_b$ can be. It is not bounded in $[0, 1]$. Sarazin, Viaud & Cournède (2017), who restate the total as their Eq. (8), say so explicitly. It does not measure the expected variance reduction $\mathrm{E}[\mathrm{Var}(Y \mid X_{\sim i})] / \mathrm{Var}(Y)$ that a total-order index normally reports, so it does not answer the parameter-fixing question. The bias runs toward "cannot be fixed", and it can be large. On a linear model at $\rho = 0.95$, HDMR reports 0.502 for a parameter whose true total effect is 0.025. A parameter the model ignores can outrank one that scored negative.

The source paper invites the confusion. Its Eq. (4) uses the symbol $S_{Ti}$ for the classical conditional-variance total, and Section 2.2.3 reuses the same symbol for the term-membership sum. Only the second is what HDMR reports.

Li et al. also attach a precondition to the totals. They are reliable only when the per-term $S$ values sum to about 1 (Eq. 24). The shortfall is the variance the surrogate leaves unexplained. Check `result.S.sum()` before you rank parameters from a correlated fit.

$S_1$ has the matching caveat: it is the structural share $S_a$ of the first-order term, not the Sobol' first-order index.

When you need a conditional-variance total under dependence, use [Kucherenko](#kucherenko-dependent-input-sobol-indices) ($S_T$) or [VKOGA](#vkoga-correlated-input-variance-indices) ($S_{TU}$, the parameter-fixing measure). HDMR's own contribution under dependence is the per-term $S_a$ versus $S_b$ split, which neither of those provides. `jaxgsa.hdmr.analyze()` emits one `JaxgsaWarning` on a correlated problem to say all of this.
:::

### When to use it

- Model evaluations are expensive and you want to reuse existing runs
- Parameters may be correlated, and you want the per-term structural ($S_a$) versus correlative ($S_b$) split. Read $S_T$ with care under dependence — see the note above
- You need a surrogate for fast prediction at new parameter values (`result.predict`)

## PCE (Polynomial Chaos Expansion)

PCE is the second surrogate-based route to Sobol indices that works from data you already have. It fits an orthogonal polynomial surrogate to $(X, Y)$ data and reads the indices directly from the expansion coefficients (Sudret, 2008), with no Monte Carlo estimation noise.

Pick it when your model is smooth. Any set of $(X, Y)$ pairs works, so no sampling design is required. The polynomial basis follows the Wiener-Askey scheme: Legendre polynomials for uniform parameters, and Hermite polynomials for unbounded Gaussian parameters. Truncated Gaussian parameters use Legendre polynomials after CDF mapping to $[-1, 1]$.

### How to use it

1. You provide any set of $(X, Y)$ pairs; `Y` may be scalar `(N,)`, multi-output `(N, K)`, or time-series `(N, T, K)`. All output slices share one polynomial basis and are fitted in a single solve.
2. `jaxgsa.pce.analyze()` maps parameters to the appropriate reference domain, builds the design matrix from a total-degree multi-index, and fits coefficients via regularized least squares.
3. Sobol indices ($S_1$, $S_T$, $S_2$) are computed analytically from the squared coefficients.
4. Leave-one-out cross-validation RMSE quantifies surrogate accuracy.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | First-order Sobol index for parameter $i$, computed analytically from the squared coefficients. |
| $S_T(i)$ | Total-order Sobol index for parameter $i$, computed analytically from the squared coefficients. |
| $S_2(i,j)$ | Second-order Sobol index for the pair $(i, j)$, computed analytically from the squared coefficients. |
| Leave-one-out RMSE | Cross-validation error of the fitted surrogate — a fit-quality diagnostic, not a per-parameter index. |

### When to use it

- You want analytical Sobol indices without Monte Carlo sampling noise
- Your model is smooth enough to be well-approximated by low-order polynomials
- You have mixed uniform and Gaussian parameters (the Wiener-Askey scheme selects the appropriate basis automatically)
- You need a fast surrogate (`result.predict` mirrors the training output layout)

## Shapley Effects

The Shapley effect $\mathrm{Sh}_i$ is a single, fairly allocated importance score per parameter. It gives each parameter its share of the output variance, with every interaction split evenly among its participants, so the scores sum to exactly 1. The method applies the Shapley value from cooperative game theory to variance-based sensitivity analysis. It treats the output variance as a payout divided among the parameters, viewed as players whose coalition worths are the partial variances of the ANOVA decomposition (Owen, 2014; Song, Nelson & Staum, 2016).

Pick it when you need one defensible number per parameter — for ranking, reporting, or budget allocation — rather than the two-sided $S_1$/$S_T$ view. Like HDMR and PCE, it works from data you already have: any set of $(X, Y)$ pairs, with no sampling design.

### Theoretical Background

For independent parameters, the Hoeffding–Sobol' decomposition splits the output variance into partial variances $V_u$ indexed by subsets $u \subseteq \{1, \ldots, D\}$ of the parameters. The Shapley effect of parameter $i$ allocates each interaction's variance equally among its participants:

$$
\mathrm{Sh}_i = \sum_{u \ni i} \frac{V_u}{|u|}
$$

so a main-effect variance $V_i$ is attributed entirely to parameter $i$, a pairwise interaction variance $V_{ij}$ is split half-and-half between $i$ and $j$, and so on. Under independent parameters this yields two properties:

- **Bracketing**: $S_{1,i} \leq \mathrm{Sh}_i \leq S_{T,i}$ — the Shapley effect always lies between the first-order and total-order Sobol indices.
- **Exact partition**: $S_1$ omits interactions, so $\sum_i S_{1,i} \leq 1$, and $S_T$ counts each interaction once per participant, so $\sum_i S_{T,i} \geq 1$. Shapley effects split every interaction fairly and sum to exactly 1 with no gaps or double counting.

The Hoeffding decomposition above defines the `backend="pce"` allocation, and it holds only for independent parameters. `jaxgsa.shapley.analyze(backend="pce")` refuses to run when `problem.correlation` declares a dependence structure. For correlated parameters, use `backend="hdmr"` with `include_correlative=True`. It folds HDMR's ANCOVA decomposition into the allocation: each term's structural plus correlation-induced variance ($S_a + S_b$) is split among its participants. Be clear about what that gives you. It is an ANCOVA-based attribution, and correlative shares can be negative. It is not the conditional-variance Shapley effects of Song et al. (2016), which remain future work. See [Correlated Inputs](/examples/correlated-inputs).

### How jaxgsa computes them

jaxgsa computes Shapley effects analytically from a fitted surrogate's variance decomposition. There is no permutation Monte Carlo, no conditional-variance sampling, and no external `shap` dependency:

- `backend="pce"` (default) fits a polynomial chaos expansion and groups the squared orthonormal coefficients by the support of their multi-index (Sudret, 2008) — exact within the fitted polynomial.
- `backend="hdmr"` fits the RS-HDMR B-spline surrogate and uses the structural ($S_a$) variances of its component functions as the partial variances $V_u$, truncated at `maxorder`.

Both backends accept scalar `(N,)`, multi-output `(N, K)`, and time-series `(N, T, K)` `Y`.

Normalization is by the surrogate's total decomposed variance $\sum_u V_u$, so $\sum_i \mathrm{Sh}_i = 1$ exactly — the Shapley efficiency property (Owen, 2014). $S_1$ and $S_T$ from the same surrogate use the same denominator. For `backend="pce"` they therefore match `jaxgsa.pce.analyze` exactly. For `backend="hdmr"` they differ from `jaxgsa.hdmr.analyze`, which normalizes by $\mathrm{Var}(Y)$, by a factor of `explained_variance`.

How much of the output variance the surrogate actually captured is reported separately in the `explained_variance` field, $\sum_u V_u / \mathrm{Var}(Y)$. It is close to 1 for a good fit, below 1 when truncation or fit error leaves variance unexplained, and above 1 when an overfit surrogate over-counts shared variance. It is an honest diagnostic rather than a silently renormalized result. A `JaxgsaWarning` is emitted when it strays far from 1. Interactions above `maxorder` (HDMR) or the polynomial order (PCE) are absent from the allocation.

### How to use it

1. You provide any set of $(X, Y)$ pairs — no sampling design required.
2. Call `.shapley()` on a fitted PCE or HDMR result. Each partial variance is
   allocated equally among the parameters in its interaction set.
3. The result carries `Sh` alongside `S1` and `ST` computed from the same surrogate, so the three indices are directly comparable and the ordering $S_1 \leq \mathrm{Sh} \leq S_T$ is visible at a glance.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2000, seed=42)
Y = evaluate(jnp.asarray(X))

# PCE backend (default) — exact within the fitted polynomial
result = jaxgsa.pce.analyze(PROBLEM, jnp.asarray(X), Y).shapley()
print("Sh:", result.Sh)              # (D,) Shapley effects
print("sum:", result.Sh.sum())       # == 1 (Shapley efficiency property)
print("explained:", result.explained_variance)  # sum_u V_u / Var(Y) — fit quality
print("order:", result.order)        # effective surrogate order used
print("S1:", result.S1)              # first-order, same surrogate
print("ST:", result.ST)              # total-order, same surrogate

# HDMR backend — B-spline surrogate; HDMR-only knobs
result_hdmr = jaxgsa.hdmr.analyze(
    PROBLEM,
    jnp.asarray(X),
    Y,
    maxorder=2,
).shapley()
```

Backend-specific keyword arguments are validated: explicitly setting a knob that belongs to the non-selected backend (e.g. `backend="pce"` with `maxorder=3`) raises `ValueError`.

### Index summary

| Index | Meaning |
|-------|---------|
| $\mathrm{Sh}(i)$ | Shapley effect: parameter $i$'s fair share of decomposed variance, including an equal split of every interaction it participates in. $\sum_i \mathrm{Sh}_i = 1$ exactly (Shapley efficiency). |
| $S_1(i)$ | First-order index from the same surrogate (main effect only). |
| $S_T(i)$ | Total-order index from the same surrogate (main effect plus all interactions counted in full). |
| `explained_variance` | Fraction of $\mathrm{Var}(Y)$ the surrogate captured, $\sum_u V_u / \mathrm{Var}(Y)$ — a separate fit-quality diagnostic, not a per-parameter index. |

### When to use it

- You want a single, fairly allocated importance score per parameter that sums to exactly 1, for example for ranking, reporting, or budget allocation
- Interactions matter and you want them attributed to their participants rather than omitted ($S_1$) or double-counted ($S_T$)
- You have existing $(X, Y)$ pairs and want analytical indices without permutation Monte Carlo noise
- Your parameters are independent (required for `backend="pce"`; under a declared correlation use `backend="hdmr"` with `include_correlative=True` for the ANCOVA-based allocation)

### References

- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245-251.
- Song, E., Nelson, B.L. & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060-1083.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7), 964-979.

## eFAST (Extended Fourier Amplitude Sensitivity Test)

eFAST computes the same first-order and total-order Sobol indices as the Saltelli workflow, but through a frequency-based decomposition. Instead of column-swapped sample matrices, eFAST evaluates the model along sinusoidal search curves in the parameter space. It then applies the discrete Fourier transform to extract variance contributions from the spectral content of the output.

Pick it when you need $S_1$ and $S_T$ but not second-order indices. It needs its own sampling design, and that design is simpler than Saltelli's: $N \times D$ evaluations.

### How it works

For each parameter $i$, eFAST constructs a search curve by assigning the highest frequency $\omega_0$ to parameter $i$ (the "focal" parameter) and lower complementary frequencies $\omega_j$ to all other parameters. The model is evaluated at $N$ points along each curve, yielding one output vector per parameter.

The Fourier power spectrum of the output along each curve is then decomposed. The first-order index is the fraction of total variance captured by harmonics of $\omega_0$:

$$
S_i = \frac{D_1}{V} = \frac{\sum_{p=1}^{M} |F_{p\omega_0}|^2}{V}
$$

where $V$ is the total variance (via Parseval's theorem) and $M$ is the interference factor controlling how many harmonics are summed.

The total-order index is the complement of the low-frequency (non-focal) variance:

$$
S_{T_i} = 1 - \frac{D_t}{V} = 1 - \frac{\sum_{k \leq \lfloor\omega_0/2\rfloor} |F_k|^2}{V}
$$

The low-frequency content at or below $\lfloor\omega_0/2\rfloor$ is driven entirely by the complementary parameters' slower oscillations, so subtracting it from unity gives the total effect of the focal parameter including all its interactions.

### How to use it

1. `jaxgsa.efast.sample(problem, n_per_curve, ...)` returns an `EFASTSamples` design whose `samples` array has shape `(n_per_curve * D, D)`, where each contiguous block of `n_per_curve` rows corresponds to one parameter's search curve.
2. You evaluate your model on all `n_per_curve * D` rows of `samples.samples`, in order.
3. `jaxgsa.efast.analyze(samples, Y)` splits the output by curve, computes the Fourier spectrum for each, and extracts $S_1$ and $S_T$ indices. The interference factor `M` and the problem travel inside the `EFASTSamples` object, so they can never be mismatched between sampling and analysis.

eFAST does not produce second-order ($S_2$) interaction indices. If pairwise interactions are needed, use the Sobol workflow instead.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | Fraction of output variance from the focal parameter's harmonics (main effect). |
| $S_T(i)$ | Total effect including interactions, computed as $1 - D_t / V$. |

### When to use it

- You only need $S_1$ and $S_T$ (no $S_2$ required)
- You want a simpler sampling design without the Saltelli cross-matrix structure
- You are screening a large number of parameters
- The total cost is $N \times D$ evaluations, which can be lower than Saltelli's $N(D+2)$ (first/total only) or $N(2D+2)$ (with second-order, the default) when $N$ is chosen smaller than the Saltelli base count

### Reference

Saltelli, A., Tarantola, S. & Chan, K.P.-S. (1999). A quantitative model-independent method for global sensitivity analysis of model output. *Technometrics*, 41(1), 39-56.

## DGSM (Derivative-based Global Sensitivity Measures)

DGSM uses exact gradients from automatic differentiation to compute bounds on the total Sobol index $S_T$. It is the cheapest quantitative method when your model is JAX-differentiable, costing roughly one model evaluation per sample point.

Pick it as a fast screening or sanity-check step before committing to a full Sobol' analysis. Use Morris (below) instead if your model is a black box. DGSM needs its own design, but only a plain Monte Carlo sample of $N$ points.

### The DGSM Moments

For a model $f(\mathbf{X})$ with $D$ parameters, DGSM computes two statistics for each parameter $i$. The first is the mean squared derivative, which is the importance measure:

$$
\nu_i = \mathbb{E}\left[\left(\frac{\partial f}{\partial X_i}\right)^2\right]
$$

The second is the mean derivative:

$$
\sigma_i = \mathbb{E}\left[\frac{\partial f}{\partial X_i}\right]
$$

These moments are estimated from $N$ i.i.d. Monte Carlo samples. DGSM uses `jax.jacrev` (reverse-mode autodiff), so the cost of computing the full Jacobian for all $D$ parameters in a single pass is comparable to a single model evaluation. That makes DGSM particularly efficient for high-dimensional problems.

### Bounds on the Total Sobol Index

DGSM does not compute Sobol indices directly. Instead, it provides an upper bound and a lower bound on the total-order index $S_{T_i}$.

The Poincaré upper bound (Sobol' & Kucherenko, 2009):

$$
S_{T_i} \leq \frac{C(p_i) \cdot \nu_i}{\mathrm{Var}(Y)}
$$

where $C(p_i)$ is the Poincaré constant of the $i$-th parameter's marginal distribution.

The Kucherenko–Song lower bound (Kucherenko & Song, 2016):

$$
S_{T_i} \geq \frac{\mathrm{Var}(X_i) \cdot \sigma_i^2}{\mathrm{Var}(Y)}
$$

When the upper and lower bounds are close, DGSM gives a tight bracket on $S_T$ without the cost of a full Sobol analysis.

### Poincaré Constants by Distribution

The Poincaré constant depends on the marginal distribution of each parameter:

| Distribution | Poincaré Constant $C$ |
|---|---|
| Uniform $[a, b]$ | $(b - a)^2 / \pi^2$ |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\sigma^2$ |
| Truncated Normal | Spectral solve (P1 finite-element Neumann eigenproblem) |

For truncated normal parameters, the constant is computed numerically by solving a weighted eigenproblem on a finite-element grid. jaxgsa handles this automatically when the parameter spec declares truncation bounds.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples from the declared parameter distributions.
2. You pass your JAX-differentiable function and the samples to `jaxgsa.dgsm.analyze()`.
3. Internally, `jax.jacrev` computes the Jacobian via reverse-mode autodiff, and the DGSM moments and bounds are derived.
4. The returned `DGSMResult` contains `nu`, `sigma`, `upper_bound`, `lower_bound`, and `var_y`.

Alternatively, if the Jacobian has been computed externally (e.g. for non-JAX models), you can pass pre-computed `Y` and `dfdx` arrays directly.

### Index summary

| Field | Meaning |
|-------|---------|
| $\nu_i$ | Mean squared derivative: $\mathbb{E}[(\partial f / \partial X_i)^2]$. Higher values indicate stronger influence. |
| $\sigma_i$ | Mean derivative: $\mathbb{E}[\partial f / \partial X_i]$. Non-zero when the effect is non-symmetric. |
| Upper bound | Poincaré bound: $C_i \cdot \nu_i / \mathrm{Var}(Y)$. Conservative upper bound on $S_T$. |
| Lower bound | Kucherenko–Song bound: $\mathrm{Var}(X_i) \cdot \sigma_i^2 / \mathrm{Var}(Y)$. Guaranteed lower bound on $S_T$. |

### When to use it

- You have a JAX-differentiable model and want fast screening without the cost of Saltelli or eFAST sampling
- You want bounds on $S_T$ rather than exact indices
- You are screening a large number of parameters where autodiff is cheaper than structured designs
- You want a quick sanity check before running a full Sobol analysis

### References

- Sobol', I.M. & Kucherenko, S. (2009). Derivative based global sensitivity measures and their link with global sensitivity indices. *Mathematics and Computers in Simulation*, 79(10), 3009-3017.
- Kucherenko, S. & Song, S. (2016). Derivative-based global sensitivity measures and their link with Sobol' sensitivity indices. *Reliability Engineering & System Safety*, 148, 81-95.
- Lamboni, M., Iooss, B., Popelin, A.-L. & Gamboa, F. (2013). Derivative-based global sensitivity measures: General links with Sobol' indices and numerical tests. *Mathematics and Computers in Simulation*, 87, 44-54.

## Morris (Elementary Effects Screening)

Morris is a global screening method. With only $r(D+1)$ model evaluations, where $r$ is typically 10–50 trajectories, it ranks parameters and flags which ones are negligible. Technically it is a globalized one-at-a-time (OAT) design: it measures coarse finite-difference effects of each parameter at many locations spread across the parameter domain, then summarises them into robust importance measures.

Pick it as a triage step for expensive black-box models. Fix the parameters Morris rules out, then spend your remaining budget on an exact method like Sobol' for the survivors. Morris needs its own design, which jaxgsa generates.

### How it works

The design consists of $r$ trajectories, each a path of $D + 1$ points where consecutive points differ in exactly one coordinate. Each trajectory contributes one elementary effect per parameter — a finite-difference slope:

$$
EE_i = \frac{f(\mathbf{x} + \Delta \mathbf{e}_i) - f(\mathbf{x})}{\Delta}
$$

where $\mathbf{e}_i$ is the unit vector along parameter $i$ and $\Delta$ is the step in unit-cube coordinates. jaxgsa implements two designs:

- **Trajectory design** (Morris 1991, default): each trajectory is a random walk on a $p$-level grid (`num_levels`, default 4) with the canonical step $\Delta = p / (2(p-1))$, visiting parameters in a random order.
- **Radial design** (Campolongo et al. 2011, `method="radial"`): star designs around scrambled-Sobol' base points, where each elementary effect compares a one-coordinate swap against the shared base point with a per-step $\Delta_i = b_i - a_i$.

Both uniform and Gaussian marginals are supported. The design touches the unit-cube boundaries, and an unbounded inverse CDF maps 0 and 1 to infinity. Each open side of a Gaussian marginal is therefore pulled in by $q$ (`truncation_quantile`, default $q = 10^{-4}$ — the 0.01%–99.99% quantile range) before the inverse-CDF transform. A side the problem already bounds with an explicit `low` or `high` is left exactly where the user put it, so a two-sided truncated Gaussian is sampled as declared. Uniform marginals are untouched, and deduplication and prefix-nesting are unaffected. The elementary-effect divisor is the step the design really takes, so this rescaling does not bias $\mu^*$.

On an unbounded marginal there is no $q \to 0$ limit for $\mu^*$. The design always includes unit levels 0 and 1 exactly, so a smaller $q$ always reaches further into the tail and the effects grow with it. $\mu^*$ magnitudes on an unbounded marginal are therefore scale-dependent by construction, and only rankings are comparable across truncation settings. If you want one bounded parameter model that every method shares, declare it once:

```python
problem = jaxgsa.Problem.from_dict(
    {"x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}},
    truncate_gaussians=1e-4,   # fills low/high at this marginal's own quantiles
)
```

The $r$ elementary effects per parameter are reduced to three screening measures:

- $\mu_i$ — the mean elementary effect. Sign cancellation can mask non-monotonic influence, which is why $\mu$ alone is unreliable.
- $\mu^*_i$ — the mean absolute elementary effect (Campolongo et al. 2007). This is the headline importance measure. Read it as "how strongly does the output respond, on average, when this parameter moves?". It is a good proxy for the total-order index $S_T$ ranking.
- $\sigma_i$ — the standard deviation of the elementary effects (ddof=1). A large $\sigma_i$ relative to $\mu^*_i$ means the effect of parameter $i$ changes across the domain, indicating nonlinearity or interactions with other parameters.

The canonical output is the $\mu^*$–$\sigma$ scatter plot. Parameters near the origin are negligible. Parameters far along the $\mu^*$ axis are influential. Parameters high above the diagonal act mainly through nonlinearity or interactions.

Morris is closely related to DGSM. As $\Delta \to 0$, $\mu^*_i \to \mathbb{E}|\partial f / \partial x_i|$, so Morris is the black-box, macro-step analog of jaxgsa's DGSM. Use DGSM when the model is JAX-differentiable, and Morris when it is not.

### How to use it

1. `jaxgsa.morris.sample()` builds the trajectories, removes exact duplicate rows, and returns only the unique rows. Grid designs collide often in low dimensions, so this saves real model evaluations, just like Saltelli sampling.
2. You evaluate your model on `sampling_result.samples`.
3. `jaxgsa.morris.analyze()` reconstructs the expanded design internally, applies the `on_invalid` policy at trajectory granularity (see [Failed model runs](#failed-model-runs)), and reduces one elementary effect per trajectory and parameter to $\mu$, $\mu^*$, and $\sigma$. Pass `num_resamples > 0` (with a JAX PRNG `key`) for bootstrap confidence intervals over trajectories.

Elementary effects are computed in unit-cube coordinates, so $\mu^*$ is directly comparable across parameters regardless of their physical ranges. `MorrisResult.to_physical_units()` rescales to derivative-scale values in the problem's native units. That rescaling covers uniform-marginal problems only: for Gaussian marginals the inverse-CDF transform is nonlinear, so the measures stay in grid coordinates. `MorrisSamples.downsample()` prefix-slices to fewer trajectories without re-simulation, mirroring `SobolSamples.downsample()`.

Compared to SALib's Morris implementation, jaxgsa adds unique-row deduplication, vectorized multi-output and time-series analysis (SALib's Morris is scalar-only), bootstrap confidence intervals, the radial design, and prefix-nested downsampling.

### Free screening from a Sobol' design

A Saltelli design is already a radial Morris design. Within each base point it holds a row $A$ and $D$ rows $A_B^{(j)}$ that differ from $A$ in exactly one parameter, which is precisely what an elementary effect needs. This is not a coincidence: Campolongo et al. (2011) build the radial design from a $2D$-dimensional Sobol' sequence split into halves $(a, b)$, and `jaxgsa.sobol.sample` draws the same sequence the same way.

Write the step as $\Delta_j = B_j - A_j$, so that $EE_j = \left(f(A_B^{(j)}) - f(A)\right) / \Delta_j$. Substituting $f(A_B^{(j)}) - f(A) = \Delta_j \cdot EE_j$ into the estimators jaxgsa uses for Sobol' indices gives

$$S_{T_j} = \frac{\mathbb{E}\left[(f(A) - f(A_B^{(j)}))^2\right]}{2\,\mathrm{Var}(Y)} = \frac{\mathbb{E}\left[\Delta_j^2\, EE_j^2\right]}{2\,\mathrm{Var}(Y)} \quad \text{(Jansen 1999)}$$

$$S_{1_j} = \frac{\mathbb{E}\left[f(B)\left(f(A_B^{(j)}) - f(A)\right)\right]}{\mathrm{Var}(Y)} = \frac{\mathbb{E}\left[f(B)\, \Delta_j\, EE_j\right]}{\mathrm{Var}(Y)} \quad \text{(Saltelli 2010)}$$

against Morris's $\mu^*_j = \mathbb{E}|EE_j|$. Same increments, different weighting: Morris divides by $\Delta$ and takes a first absolute moment, Jansen keeps $\Delta$ and takes a second moment. Campolongo et al. (2011) call this the unified approach — one design serving both screening and quantitative indices. The chain closes at DGSM: as $\Delta_j \to 0$ the effect tends to $\partial f / \partial x_j$, so $\mathbb{E}[EE_j^2] \to \nu_j$, the quantity that bounds $S_{T_j}$ through the Poincaré inequality.

`SobolSamples.to_morris()` performs this reinterpretation, so screening measures cost no extra model evaluations:

```python
samples = jaxgsa.sobol.sample(problem, 1024)
Y = model(samples.samples)

sobol_result = jaxgsa.sobol.analyze(samples, Y)          # S1, ST, S2
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)  # mu*, sigma
```

You get one radial block per base point, so `n_trajectories == base_n` for both design variants. A second-order design also contains a block based at $B$ ($B$ with its $B_A^{(j)}$ rows), and it is tempting to harvest as a free doubling. `to_morris()` does not use it. The reason is not that it is a duplicate. That equality holds only for additive contributions: whenever parameter $j$'s contribution is additive,

$$\frac{f(B_A^{(j)}) - f(B)}{A_j - B_j} = \frac{g_j(A_j) - g_j(B_j)}{A_j - B_j} = \frac{f(A_B^{(j)}) - f(A)}{B_j - A_j}$$

but in general it does not. Measured on Ishigami the paired effects correlate 0.50 / 1.00 / −0.06, so only $x_2$ — from the purely additive $7\sin^2(x_2)$ term — is a genuine duplicate. The real reason is that pooling buys nothing: over 150 seeds at `base_n=128` the pooled estimator's variance ratio against the $A$-only estimator is $[1.07, 1.00, 1.59]$, so it reduces no variance and is worse on $x_3$. Pooling would also need a cluster bootstrap over base points to keep confidence intervals honest, because the two blocks in a base point share their sampling unit. That is real machinery for no gain.

Take care over which estimand you get. The derived design is a radial design, so it estimates $\mathbb{E}\left|f(A \text{ with } B_j) - f(A)\right| / |B_j - A_j|$, in which the step varies from block to block. That is not the classical Morris quantity with one fixed grid step $\Delta$. `jaxgsa.morris.sample` defaults to `method="trajectory"`, so compare against `morris.sample(..., method="radial")`, never against the default. On Ishigami at $r = 8192$ the derived $\mu^*$ is $[8.68, 15.01, 6.62]$ against $[8.69, 15.02, 6.64]$ for the native radial design, but $[7.59, 7.88, 6.39]$ for the native trajectory design — a factor 1.9 on $x_2$, and 2.5 on its $\sigma$.

Three further caveats:

- The derived measures reuse the same model outputs as the Sobol' indices, so agreement between $\mu^*$ and $S_T$ is not an independent check of either. They may also legitimately rank parameters differently, because $\mu^*$ is a mean absolute derivative, not a variance share.
- Saltelli takes $A$ and $B$ from the same Sobol' row, whereas `jaxgsa.morris.sample`'s radial design offsets them by four draws precisely to keep $\Delta$ away from zero. Blocks whose step is unmeasurable are dropped with a warning. At the default `scramble=True` this is a non-issue: 0 of 65536 blocks were dropped across 8 seeds at $D = 3$. With `scramble=False` the drop rate is real but falls off with `base_n` — 21.9% at `base_n=64`, 9.4% at 256, 2.3% at 1024, 1.2% at 4096. The survivors are a biased subsequence, giving $\mu^* = [8.34, 14.88, 5.55]$ at `base_n=64` against $[8.68, 15.01, 6.62]$ scrambled, so $x_3$ reads 16% low. Keep `scramble=True`.
- For unbounded Gaussian marginals, $\mu^*$ has no fixed scale. How far a design reaches into the tail sets the magnitude, and the Saltelli design (bounded only by the library's own $\pm 7.03\sigma$ support clip) and `morris.sample` reach different distances. Only rankings are comparable. Bound the marginals once if magnitudes must match:

  ```python
  problem = jaxgsa.Problem.from_dict(params, truncate_gaussians=1e-4)
  ```

  Both sides are then genuinely bounded, `morris.sample` does not squash them again, and the derived and native radial measures agree — measured ratios 0.999 (linear), 0.997 ($x^2$), 0.988 ($x^4$), 0.987 ($\exp(x^2/3)$), each within its own seed-to-seed spread. `to_morris()` warns when unbounded Gaussians are present.

The reverse derivation is impossible: a radial Morris design never evaluates the $B$ rows, so $S_1$ and $S_T$ cannot be recovered from it.

### Index summary

| Measure | Meaning |
|-------|---------|
| $\mu(i)$ | Mean elementary effect. Sign cancellation can hide non-monotonic influence. |
| $\mu^*(i)$ | Mean absolute elementary effect. Headline importance measure; proxy for the $S_T$ ranking. |
| $\sigma(i)$ | Standard deviation of the elementary effects. Large $\sigma / \mu^*$ indicates nonlinearity or interactions. |

### When to use it

- You want a cheap screening pass before committing to a full Sobol' run
- Your model is a black box (not JAX-differentiable — otherwise consider DGSM)
- You have many parameters and a tight evaluation budget — the cost is $r(D+1)$ with $r$ typically 10-50
- You only need a ranking and an interaction flag, not exact variance fractions

### References

- Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161-174.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large models. *Environmental Modelling & Software*, 22(10), 1509-1518.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2011). From screening to quantitative sensitivity analysis. A unified approach. *Computer Physics Communications*, 182(4), 978-988.
- Jansen, M.J.W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1-2), 35-43.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*, ch. 3. Wiley.

## HSIC (Hilbert–Schmidt Independence Criterion)

HSIC measures the statistical dependence between each parameter and the output. It captures any dependence, including nonlinear, non-monotone, and heteroscedastic effects that variance-based indices can underweight. It works in a reproducing kernel Hilbert space (RKHS), mapping parameters and outputs through Gaussian RBF kernels.

Pick it when you suspect your model's behaviour is not well summarised by variance, when your parameters may be correlated, or when you want statistical significance tests attached to the indices. Like HDMR, it works from data you already have: any set of $(X, Y)$ pairs, with no independence assumption on the parameters and no sampling design.

### The HSIC Dependence Measure

Each parameter $X_i$ and the output $Y$ are passed through a characteristic kernel — a Gaussian RBF whose bandwidth is set automatically by the median heuristic (the median pairwise distance between sample points). Writing $\mathbf{K}$ and $\mathbf{L}$ for the two $N \times N$ kernel matrices, jaxgsa uses the biased V-statistic estimator

$$
\widehat{\mathrm{HSIC}}(X_i, Y) = \frac{1}{N^2}\,\mathrm{tr}(\mathbf{K}\mathbf{H}\mathbf{L}\mathbf{H}), \qquad \mathbf{H} = \mathbf{I} - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top
$$

where $\mathbf{H}$ is the centering matrix. For characteristic kernels, $\mathrm{HSIC}(X_i, Y) = 0$ if and only if $X_i$ and $Y$ are independent, so a larger value signals stronger dependence.

### First-Order and Total Indices

jaxgsa reports two normalised indices per parameter.

R2-HSIC is the first-order index: the normalised dependence between parameter $i$ and the output, in $[0, 1]$. Read it as a kernel analogue of a squared correlation coefficient (centred kernel alignment):

$$
R^2_{\mathrm{HSIC}, i} = \frac{\widehat{\mathrm{HSIC}}(X_i, Y)}{\sqrt{\widehat{\mathrm{HSIC}}(X_i, X_i)\,\widehat{\mathrm{HSIC}}(Y, Y)}}
$$

Total HSIC is the analogue of a total-order index, capturing dependence carried through interactions with the other parameters. It is built from augmented product kernels $k^*_d = 1 + k_{c,d}$ (Larsen & Alexanderian, 2026), where $k_{c,d}$ is the centred kernel for parameter $d$. The constant term makes the product of augmented kernels capture all interaction orders rather than only the highest, which yields correct total indices even for purely additive models. The total index for parameter $i$ then follows from comparing the complement product kernel (all parameters except $i$) with the full product kernel.

Unlike Sobol indices, R2-HSIC values are individual dependence measures and do not sum to 1.

### Permutation p-values

HSIC is a dependence measure rather than a variance fraction, so jaxgsa attaches a permutation test to each first-order index. The output labels are randomly shuffled `n_perms` times to build a null distribution of HSIC values. The p-value uses the Phipson–Smyth correction $(c + 1)/(M + 1)$, where $M$ is the number of permutations (`n_perms`) and $c$ counts permuted HSIC values at least as large as the observed one. A small p-value (< 0.05) indicates a statistically significant dependence between the parameter and the output.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples — any sampling strategy works, since no structured design is required.
2. You evaluate your model on the samples.
3. `jaxgsa.hsic.analyze()` transforms each parameter to $[0, 1]$ via its marginal CDF, builds the kernel matrices with the median heuristic, and computes all indices and p-values in a single JIT-compiled pass.

HSIC is $O(N^2)$ in time and memory because it forms $N \times N$ kernel matrices. For large $N$, pass `batch_size` to build them in row blocks. That bounds the working memory of the build, not the kernel matrix, which is one full $N \times N$ array in every case. For outputs of large magnitude, set `prenormalize=True` to standardise $Y$ before kernel construction.

### Index summary

| Index | Meaning |
|-------|---------|
| $R^2_{\mathrm{HSIC}}(i)$ | Normalised first-order kernel dependence between parameter $i$ and the output, in $[0, 1]$. |
| Total HSIC $(i)$ | Total dependence of parameter $i$ including interactions, via augmented complement product kernels. |
| p-value $(i)$ | Permutation p-value for the first-order dependence (Phipson–Smyth corrected). |

### When to use it

- You want a measure that captures any dependence — nonlinear, non-monotone, or heteroscedastic — not just variance contributions
- Your parameters may be correlated (HSIC makes no independence assumption)
- You have existing $(X, Y)$ pairs and want indices without additional model runs
- You want statistical significance testing via permutation p-values

### References

- Gretton, A., Herbrich, R., Smola, A., Bousquet, O. & Schölkopf, B. (2005). Kernel methods for measuring independence. *Journal of Machine Learning Research*, 6, 2075-2129.
- Da Veiga, S. (2015). Global sensitivity analysis with dependence measures. *Reliability Engineering & System Safety*, 142, 346-362.
- Larsen and Alexanderian (2026). Total HSIC sensitivity indices via augmented product kernels. *arXiv preprint* arXiv:2603.00849.

## PAWN (CDF-Based Sensitivity)

PAWN asks a different question from the variance-based methods. Not "how much variance does this parameter explain?", but "how much does the entire output distribution shift when this parameter is held fixed?". It compares the unconditional output CDF against conditional CDFs obtained by fixing each parameter within a bin, using the Kolmogorov–Smirnov (KS) distance as the measure of separation (Pianosi & Wagener, 2015).

Pick it when you care about tails, skewness, or other distributional features that variance misses. Like HSIC and HDMR, it works from data you already have: any $(X, Y)$ pairs, with no independence assumption on the parameters and no sampling design.

### The KS Distance

For parameter $i$, its range is partitioned into `n_bins` equal-width bins. Within each bin $b$, PAWN forms the conditional output CDF $F_{Y \mid X_i \in b}$ from the samples whose $i$-th parameter falls in that bin, and compares it with the unconditional CDF $F_Y$ (built from all samples) via the Kolmogorov–Smirnov statistic — the largest absolute gap between the two CDFs:

$$
\mathrm{KS}_{i,b} = \sup_{y}\left| F_Y(y) - F_{Y \mid X_i \in b}(y) \right|
$$

A large KS value in a bin means fixing $X_i$ there substantially changes the output distribution. A value near zero means the output is insensitive to that parameter over that region.

### Aggregating Across Bins

Each parameter yields one KS value per bin. The PAWN index reduces these to a single number per parameter using one of three statistics:

- **median** (default) — robust to a single anomalous bin.
- **max** — the worst-case shift across the parameter range.
- **mean** — the average shift.

The PAWN index is built on CDFs rather than moments, so it is moment-independent and invariant under monotone transformations of the output. It captures tail and skewness changes that variance-based indices miss.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples (Monte Carlo, Latin Hypercube, or Sobol sequences all work — no structured design required).
2. You evaluate your model on the samples.
3. `jaxgsa.pawn.analyze()` maps each parameter to $[0, 1]$, assigns samples to bins, and computes the per-bin KS distances and their aggregate in a single JIT-compiled pass. Pass `n_bootstrap > 0` for bootstrap confidence intervals.

The number of bins (`n_bins`, default 10) trades conditioning resolution against sample density per bin. With very few samples per bin the KS statistic becomes noisy, so increase $N$ or decrease `n_bins`.

Categorical parameters are supported. A categorical parameter needs no binning: its level code already names the conditioning class, so PAWN uses one bin per level and `n_bins` does not apply to it. Bins with too few samples yield `NaN`, and the median, max, and mean over bins all drop them. The index is therefore unchanged by the order of the level codes — relabel the levels and you get the same number.

### Index summary

| Index | Meaning |
|-------|---------|
| PAWN $(i)$ | Aggregated (median / max / mean) KS distance between the unconditional and conditional output CDFs for parameter $i$, in $[0, 1]$. Higher means stronger influence on the output distribution. |

### When to use it

- You care about distributional changes beyond variance, such as tail behaviour or skewness shifts
- You want a moment-independent index, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy
- Your parameters may be correlated (no independence assumption or structured design)
- Some of your parameters are categorical, or your output is discrete — PAWN needs neither an ordering on the parameters nor a density on the output

### Reference

Pianosi, F. & Wagener, T. (2015). A simple and efficient method for global sensitivity analysis based on cumulative distribution functions. *Environmental Modelling & Software*, 67, 1-11.

## Borgonovo Delta (Density-Based Sensitivity)

Borgonovo's $\delta$ index measures the expected L1 distance between the whole output density and the output density conditional on a parameter (Borgonovo, 2007):

$$
\delta_i = \frac{1}{2}\,\mathbb{E}_{X_i}\!\left[\int \left| f_Y(y) - f_{Y \mid X_i}(y) \right| \mathrm{d}y \right]
$$

It is the second moment-independent method in jaxgsa, and the natural companion to PAWN. PAWN summarises a distributional shift by the largest gap between CDFs; $\delta$ compares the densities themselves. Pick it when you want a distribution-based index on a fixed $[0, 1]$ scale, or as a faster drop-in replacement for `SALib.analyze.delta`.

The index is $0$ when fixing $X_i$ never changes the output distribution, and $1$ when the output is a deterministic function of $X_i$ alone. Because it compares whole densities rather than variances, $\delta$ captures influence carried through tails, skewness, or multimodality that variance-based indices underweight. It is also invariant under monotone transformations of the output. Like HSIC and PAWN, it works from data you already have: any $(X, Y)$ pairs, with no independence assumption on the parameters and no sampling design.

### How it works

jaxgsa implements the given-data estimator of Plischke, Borgonovo & Smith (2013):

1. For each parameter, the samples are ordered by that parameter's rank and split into $M$ equal-frequency classes. By default $M$ follows the Plischke sample-size heuristic (roughly $N^{2/7}$, at most 48 classes); override it with `n_classes`. Categorical parameters instead get one class per level (`n_classes` does not apply to them), so the index never depends on the arbitrary code order.
2. The unconditional density $f_Y$ and each class-conditional density $f_{Y \mid X_i \in \mathcal{C}_m}$ are estimated by Gaussian KDE with Silverman bandwidths on a fixed grid of `grid_size` points spanning $[\min Y, \max Y]$.
3. The L1 distances are integrated with the trapezoid rule and averaged with class weights, giving the plug-in estimate

$$
\hat{\delta}_i = \sum_{m=1}^{M} \frac{n_m}{2N} \int \left| \hat{f}_Y(y) - \hat{f}_{Y \mid X_i \in \mathcal{C}_m}(y) \right| \mathrm{d}y
$$

The plug-in estimate is biased upward at finite $N$. By default jaxgsa therefore applies Plischke's bootstrap bias reduction $2\hat{\delta}_i - \overline{\hat{\delta}_i^{(b)}}$ over `n_bootstrap` resamples, with percentile confidence intervals from the same replicates. This correction subtracts a bootstrap mean from twice the plug-in estimate. The reported $\delta$ and its percentile-interval bounds can therefore fall marginally below $0$ for weak or near-noninfluential parameters at small $N$, even though the true index and the plug-in estimate both lie in $[0, 1]$.

The same class partition also yields the given-data first-order Sobol index (variance of the class means over the total variance) at negligible extra cost, so every analysis returns both $\delta$ and $S_1$.

The estimator matches `SALib.analyze.delta` on the equal-frequency rank partition, the class-count heuristic, the Silverman KDE factors, and the 100-point output grid. It differs in three ways. The central estimate is computed on the original sample, so it is deterministic given the data, where SALib evaluates it on a random resample. A constant output column yields $\delta = S_1 = 0$ instead of an error. A bootstrap replicate that happens to be constant, which is reachable for rare-event outputs, contributes the point estimate rather than a spurious zero, where SALib raises `LinAlgError`.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples (any sampling strategy works — no structured design is required).
2. You evaluate your model on the samples.
3. `jaxgsa.borgonovo.analyze()` partitions each parameter into rank classes and computes $\delta$, $S_1$, and their bootstrap intervals in a single JIT-compiled kernel, vmapped over output columns and scanned over bootstrap replicates.

Set `n_bootstrap=0` to skip bias correction and confidence intervals (raw plug-in estimate), or `bias_correct=False` to keep the intervals but report the uncorrected estimate. For large time-series outputs, lower `slice_chunk_size` to bound peak memory, which scales with `slice_chunk_size * D * N * grid_size`.

::: warning Continuous outputs only
The $\delta$ estimator supports a continuous output distribution only. It compares kernel density estimates on a shared output grid, and a discrete output has atoms that no grid resolves. `borgonovo.analyze` checks the output first and raises `ValueError` when a column takes at most 20 distinct values and those values are fewer than 1% of the samples. Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) or [PAWN](#pawn-cdf-based-sensitivity) for a discrete output: both compare empirical distributions and need no density. A constant column is exempt, because its exact answer is $\delta = S_1 = 0$. Categorical parameters stay supported. The limit applies to the output only.
:::

$\delta$ is a half L1 distance between densities, so it lies in $[0, 1]$. If the returned estimate leaves that range by more than 0.05, the computation failed and `analyze` raises `ValueError`. The message names the parameter, reports what the kernel did to the offending class, and points at the knob that applies to that case. The value is never clipped, because a clipped value looks plausible and is still wrong. A confidence bound outside the range only warns: the point estimate is the contract, and the interval is a diagnostic.

Two settings control how a near-degenerate conditioning class is treated. `degenerate_tol` says when a class counts as degenerate. `degenerate_bandwidth` says how wide a kernel such a class is given.

`degenerate_bandwidth="auto"`, the default, floors the kernel at `max(0.1 * h_full, grid_step)`, so it never goes below what the output grid can integrate. A float is a fraction of the full-sample bandwidth and is applied exactly.

`analyze` does not refuse a `degenerate_bandwidth` on the setting alone, because the setting alone does not say whether the run works. Two conditions have to hold first. The floor only ever reaches a class the estimator already called degenerate, so on data with no such class the setting changes nothing at any value. And even on a degenerate class, a kernel narrower than one grid step only aliases if a grid point lands on the narrow peak. On one test problem with a genuine point mass, a floor of 0.01 of the full-sample bandwidth — a tenth of one grid step — still returns a $\delta$ inside $[0, 1]$, and moving the same point mass off the grid boundary keeps the answer stable down to $10^{-5}$.

`analyze` therefore checks the returned $\delta$, not the setting. When the estimate does leave $[0, 1]$, the error message reads what the run actually did and names the knob that applies to it. Every case names `grid_size`, because a finer grid always shortens the step. A run where no class was floored also names `degenerate_tol`, because the floor changed nothing and the tolerance is what kept it away. A run floored by an explicit `degenerate_bandwidth` adds that floor width, one grid step, and the fraction of the full-sample bandwidth that equals one grid step. Under the `"auto"` default there is no such fraction to give: the floor is already at least one grid step wide, so `grid_size` is the whole of the advice. The value is not clipped, for the same reason as above.

Raising `degenerate_tol` also does not raise. A higher tolerance calls more classes degenerate, and each of those is then given the floor. When the floor is narrower than a class's own bandwidth, that class gets a *narrower* kernel than it had, which inflates $\delta$ for the classes the higher tolerance said to distrust. The answer is still a valid computation, only biased, so `analyze` neither raises nor warns: whether the bias is real depends on the data inside the kernel, and a warning based on the settings alone would be a false alarm in most runs.

### Index summary

| Index | Meaning |
|-------|---------|
| $\delta(i)$ | Expected L1 distance between the unconditional and conditional output densities for parameter $i$, in $[0, 1]$. Higher means stronger influence on the output distribution; $0$ means no influence at all. |
| $S_1(i)$ | Given-data first-order Sobol index from the same class partition — the variance-based view of the same conditioning, for comparison at no extra cost. |

### When to use it

- You care about influence on the whole output distribution — tails, skewness, multimodality — not just variance
- You want a moment-independent index with a fixed $[0, 1]$ scale, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy, possibly with correlated parameters
- You use `SALib.analyze.delta` and want a deterministic, JIT-compiled equivalent that also handles multi-output and time-series `Y`

### Reference

- Borgonovo, E. (2007). A new uncertainty importance measure. *Reliability Engineering & System Safety*, 92(6), 771-784.
- Plischke, E., Borgonovo, E. & Smith, C.L. (2013). Global sensitivity measures from given data. *European Journal of Operational Research*, 226(3), 536-550.

## Optimal Transport (Wasserstein-Based Sensitivity)

The optimal-transport index (Borgonovo, Figalli, Plischke & Savaré, 2024) measures how far knowing a parameter moves the whole output distribution. It uses the squared 2-Wasserstein distance, which is the minimal quadratic work needed to transport the unconditional output distribution onto the conditional one:

$$
\iota_i = \frac{\mathbb{E}_{X_i}\!\left[ W_2^2\!\left(P_{Y \mid X_i},\, P_Y\right) \right]}{2\,\mathrm{Var}(Y)}
$$

The denominator is the theoretical maximum of the numerator, so $\iota_i \in [0, 1]$. A value of $0$ means the output distribution never reacts to $X_i$, and $1$ means it is fully determined by it. The defining feature is the exact decomposition of every index into two parts:

- **advective** — the class-averaged squared shift of the conditional mean, which equals exactly half the given-data first-order Sobol index ($2 \cdot \mathrm{advective} = S_1$), and
- **diffusive** — the remainder: changes in spread, tails, and shape.

So the OT index subsumes the variance-based first-order view and quantifies what lies beyond it, on one scale. It works from data you already have: any $(X, Y)$ pairs, with no sampling design.

### How it works

1. For each parameter, samples are split into `n_partitions` equal-frequency classes by the parameter's rank (default `min(25, N // 2)`). Rank-based conditioning is distribution-free: uniform, Gaussian, or mixed marginals work unchanged, and monotone parameter transforms change nothing. Correlated parameters are supported — the index then measures total, correlation-inclusive influence. Categorical parameters instead get one class per level (`n_partitions` does not apply to them), so the index never depends on the arbitrary code order.
2. Per class, $W_2^2$ between the conditional and unconditional output samples is computed. In the default `mode="univariate"` (per output column) this uses the closed form of 1-D optimal transport — both empirical quantile functions evaluated at the $N$ uniform mass points via sorting, no iterative solver. The `"multivariate"` and `"trajectory"` modes treat the output vector as a point cloud and solve entropic transport with a pure-JAX log-domain Sinkhorn solver (regularization `epsilon`, reported cost is the unregularized $\langle P, C\rangle$).
3. Class results are averaged with class-size weights and divided by $2\,\mathrm{Var}(Y)$ (point-cloud modes: $2\,\mathrm{Tr}\,\mathrm{Cov}(Y)$, with per-column standardization on by default so no output dominates through its units).

Entropic and finite-sample bias keep point-cloud-mode indices of irrelevant parameters strictly positive. Pass `dummy=True` to run a synthetic, provably independent parameter through the same estimator: its index (`ot_dummy`) is the irrelevance floor to compare against.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` or any existing $(X, Y)$ data — no structured design required.
2. `jaxgsa.optimal_transport.analyze()` computes `ot`, `advective`, and `diffusive` per parameter (and per output column in `"univariate"` mode), with optional stratified bootstrap confidence intervals.

Pick the mode by the question: `"univariate"` for per-column indices across `(N,)`/`(N, K)`/`(N, T, K)` outputs, `"multivariate"` for one index per parameter over the flattened joint output, `"trajectory"` for one index per parameter per output over the whole time course. Bootstrap in the point-cloud modes costs `n_bootstrap * D * n_partitions` Sinkhorn solves, so keep it modest.

### Index summary

| Index | Meaning |
|-------|---------|
| $\iota(i)$ (`ot`) | Normalized expected $W_2^2$ between conditional and unconditional output distributions, in $[0, 1]$. |
| `advective` | Mean-shift component; $2 \cdot \mathrm{advective}$ is the given-data first-order Sobol index. |
| `diffusive` | Spread/shape component, `ot - advective`; flags influence invisible to the conditional mean. |
| `ot_dummy` | Index of a synthetic independent parameter (with `dummy=True`) — the irrelevance floor. |

### Valid under correlated inputs

The OT index is valid under correlated parameters, and jaxgsa certifies it. `optimal_transport.analyze` accepts a problem with a declared `problem.correlation`, because it is exempt from the correlated-input error. The definition $\mathbb{E}_{X_i}[W_2^2(P_{Y|X_i}, P_Y)]$ conditions on one parameter at a time and never requires an independence decomposition. The estimator conditions on rank classes of the observed sample and never reads the declared matrix. The test suite asserts bit-equality between a correlated problem and the same $(X, Y)$ with the correlation stripped.

Read the index as total, correlation-inclusive influence. A parameter the model never uses still gets a clearly non-zero index when it is correlated with one the model does use (tested at $\rho = 0.8$). To separate direct from correlation-borne influence, use [VKOGA](#vkoga-correlated-input-variance-indices) or [Kucherenko](#kucherenko-dependent-input-sobol-indices).

### When to use it

- You want a moment-independent index that still ties exactly to the variance-based world
- You want to distinguish parameters that move the output from parameters that reshape it
- You want one index per parameter for a whole trajectory or multivariate output (`multivariate` / `trajectory` modes)
- Your parameters have mixed marginals or are correlated

### Reference

- Borgonovo, E., Figalli, A., Plischke, E. & Savaré, G. (2024). Global sensitivity analysis via optimal transport. *Management Science*. doi:10.1287/mnsc.2023.01796

## VKOGA (Correlated-Input Variance Indices)

VKOGA reports variance-based sensitivity indices for parameters that are genuinely dependent. It separates what a parameter explains by itself from what it explains through its correlations. Apart from VKOGA and [Kucherenko](#kucherenko-dependent-input-sobol-indices), every variance-based method on this page assumes independent parameters, or sidesteps the question by measuring something other than variance.

VKOGA is the given-data route of that pair. It is the surrogate-based sensitivity analysis (SSA) of Hilhorst, Quicken, van de Vosse & Huberts (2024), which computes the correlated variance-based indices of Li et al. (2010) — five of them. Pick it when your parameters are dependent and you still want variance fractions, not a distributional distance. Any set of $(X, Y)$ pairs works, with no sampling design.

The method runs in two stages, and the split is the whole point. The indices need nested conditional sampling. That is hopeless against an expensive model, but trivial against a cheap emulator:

1. Fit a VKOGA surrogate (Vectorial Kernel Orthogonal Greedy Algorithm; Wirtz & Haasdonk, 2013) to the given $(X, Y)$ data. It uses a Gaussian RBF kernel, with centres chosen one at a time at the maximiser of the power function (P-greedy) expressed in a nested Newton basis, and coefficients from an RKHS-regularised least-squares solve. Centre selection depends only on $X$, so all output slices share one basis and one set of centres. That is the "vectorial" part. `gamma` and `ridge` are chosen by k-fold cross validation.
2. Estimate the indices against that surrogate by quasi-Monte-Carlo, with a Gaussian copula supplying the dependency structure. The copula keeps each parameter's declared marginal exactly as written and adds a rank-correlation structure on top. All conditioning happens in the latent standard-normal space, where the conditionals are closed-form, so no iterative sampler is involved.

### The five indices

When parameters are dependent the Sobol' decomposition no longer holds. The conditional-variance quantities remain perfectly well defined, though. They simply change connotation, and split into correlated and uncorrelated halves:

$$
S_{TC,i} = \frac{\mathrm{Var}\left[\mathbb{E}(Y \mid X_i)\right]}{\mathrm{Var}(Y)}, \qquad
S_{TU,i} = \frac{\mathbb{E}\left[\mathrm{Var}(Y \mid \mathbf{X}_{\sim i})\right]}{\mathrm{Var}(Y)}
$$

These are the same two formulas as $S_1$ and $S_T$. What changes is what they now mean, and the distinction is the single most important thing to get right:

- **$S_{TC}$ (total correlated)** answers "what should I measure more accurately?", the parameter prioritisation setting. It counts everything $X_i$ explains, including variance it only explains because it moves together with something else. Learning $X_i$ exactly removes that variance whatever put it there.
- **$S_{TU}$ (total uncorrelated)** answers "what can I freeze?", the parameter fixing setting. It counts only what nothing else can account for. A parameter whose $S_{TU}$ is near zero can be fixed at a nominal value, even if its $S_{TC}$ is large, because its apparent influence is carried by its correlates.

Two strongly correlated parameters will both show a large $S_{TC}$ and a small $S_{TU}$. Either one is worth measuring, but you cannot fix one and keep the other free without changing the answer.

The remaining three split $S_{TC}$ and $S_{TU}$ into their independent and correlation-borne parts:

| Index | Definition | Meaning |
|-------|------------|---------|
| $S_{TC}(i)$ | $\mathrm{Var}[\mathbb{E}(Y \mid X_i)] / \mathrm{Var}(Y)$ | Total correlated: what $X_i$ explains through itself, plus what it explains through its correlation with the rest. Use for parameter prioritisation. |
| $S_{TU}(i)$ | $\mathbb{E}[\mathrm{Var}(Y \mid \mathbf{X}_{\sim i})] / \mathrm{Var}(Y)$ | Total uncorrelated: what only $X_i$ can explain. Use for parameter fixing. |
| $S_U(i)$ | $\mathbb{E}[\mathrm{Var}(f_i \mid \mathbf{X}_{\sim i})] / \mathrm{Var}(Y)$ | The contribution of $X_i$ alone, with the part of it that $\mathbf{X}_{\sim i}$ already determines removed. $f_i$ is the fitted additive component of the output. |
| $S_C(i)$ | $S_{TC} - S_U$ | The correlation-borne contribution. It can be negative, when a correlation works against a direct effect. |
| $S_{IU}(i)$ | $S_{TU} - S_U$ | Independent interactions — zero for an additive model, non-negative always. |

The name $S_{TC}$ says "total", but the formula is a first-order conditional variance. "Total" names the pathways it counts, direct and correlated, not the interaction order. It is not a total-order Sobol' index.

$S_U$ uses an additive projection $f_i$, and no additive function of $X_i$ can represent an interaction. On a model with interactions under a correlated measure, the raw $S_U$ can therefore come out above $S_{TU}$. jaxgsa clips $S_U$ to $S_{TU}$, which keeps $S_{IU}$ non-negative, and warns when the clip is wider than 1% of the output variance. Read that warning as a statement about the model: the additive component functions are not enough for it. Trust $S_{TC}$ and $S_{TU}$ in that case, and treat $S_U$, $S_C$ and $S_{IU}$ as indicative. $S_C$ is never clipped, because a negative $S_C$ is a real reading.

Under independent parameters the whole structure collapses back to the familiar one. $S_{TC}$ becomes the first-order Sobol' index $S_1$, $S_{TU}$ becomes the total index $S_T$, $S_U$ equals $S_{TC}$, and $S_C$ goes to zero. Running it on an uncorrelated problem is therefore a legitimate, if roundabout, way to get $S_1$ and $S_T$ from a kernel surrogate.

### VKOGA or HDMR's ANCOVA split?

Both handle correlated given data, and both report a decomposition. They decompose different things, so they answer different questions.

[HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) fits an explicit additive expansion $f_0 + \sum_i f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots$. It splits each term's variance into a structural part $S_a$ and a correlative part $S_b$, where $S_b$ collects the covariance that term shares with the others. The decomposition is term-wise. You get per-interaction attribution and dense $S_2$/$S_3$ arrays. HDMR also knows which parameters each term involves, so it can produce correlation-aware Shapley effects via `shapley(include_correlative=True)`.

VKOGA fits a kernel expansion, which is a sum over centres rather than over parameter subsets, so it has no term-wise structure at all. What it has instead is direct access to the conditional-variance definitions. The surrogate is cheap enough to sample $\mathbb{E}(Y \mid X_i)$ and $\mathrm{Var}(Y \mid \mathbf{X}_{\sim i})$ by brute force under an explicit copula. The decomposition is per parameter, and it is the one that maps onto the prioritisation-versus-fixing decision.

Practical guidance:

- Want the prioritise / fix distinction under dependence, with an explicit and auditable dependency structure? Use VKOGA.
- Want to know which interaction carries the variance, or a fair per-parameter allocation summing to 1? Use HDMR — its terms are labelled, and only it can produce Shapley effects. `VKOGAResult.shapley()` deliberately raises `NotImplementedError`.
- Want to declare a dependency structure rather than infer one from the data (a copula from expert knowledge, a sensitivity sweep over $\rho$, or the same data analysed under several correlation assumptions)? Only VKOGA takes a correlation matrix as an argument; HDMR reads correlation implicitly out of whatever $X$ you hand it.
- The two are complementary, not redundant: HDMR's $S_b$ tells you that correlation matters, and VKOGA's $S_{TC} - S_{TU}$ gap tells you what to do about it.

### How to use it

1. You provide any set of $(X, Y)$ pairs — no sampling design required.
2. `jaxgsa.vkoga.analyze()` maps parameters to $[0, 1]$ through their marginal CDFs (the RBF kernel is isotropic, so every column must share a scale), centres the outputs, cross-validates `gamma` and `ridge`, and fits the greedy kernel surrogate.
3. The same call then draws the nested conditional samples in latent copula space and returns the five indices, along with the surrogate's `n_centers`, `gamma`, `ridge`, and per-slice training `rmse`.
4. `result.predict(X_new)` reuses the fitted surrogate; `result.to_dataset()` exports everything, including the correlation matrix, as a labeled `xarray.Dataset`.

The dependency structure comes from the problem. `analyze` reads `problem.correlation` by default and falls back to independent parameters when the problem declares none. A `(D, D)` matrix passed as `correlation=` overrides the declaration for one call. To fit a matrix from observed data, use `jaxgsa.sampling.fit_correlation(problem, X_data)` and attach it with `problem.with_correlation(...)`. Whichever route you choose, the matrix actually used is returned on `result.correlation`.

Cost is dominated by the hyperparameter search, a 10×10 grid of k-fold refits, so pass `gamma` and `ridge` explicitly to skip it once you know good values. The estimator sample sizes (`n_outer`, `n_inner`, `n_variance`) only ever touch the surrogate, so they are cheap to raise.

### Two caveats

1. Train on an independent, space-filling design, even when the analysis is correlated. This is the easy way to get wrong answers. A correlated sample concentrates on a ridge through the parameter space. But $S_{TU}$ conditions on $\mathbf{X}_{\sim i}$ and then resamples $X_i$ across its whole marginal, which is precisely the off-ridge region a correlated training set never visited. A surrogate fitted there is extrapolating exactly where the estimator queries it hardest. If your data is observational and correlated, you can still fit the copula from it (`problem.with_correlation(jaxgsa.sampling.fit_correlation(problem, X))`). Read $S_{TU}$, and hence $S_U$, $S_C$ and $S_{IU}$, as carrying the surrogate's extrapolation error.

2. Use float64. The coefficient step forms the normal matrix $A^\top A$, which squares the condition number of the cross kernel. For small `gamma` that exceeds what single precision can carry, and the surrogate can come out an order of magnitude worse than the same equations solved in double.

```python
import jax
jax.config.update("jax_enable_x64", True)  # before fitting
```

`jaxgsa.vkoga.analyze()` emits a `JaxgsaWarning` when x64 is off. Cross validation partly self-corrects, because the scores are computed in the same arithmetic and so penalise the blown-up corner of the grid, but the ceiling is real.

### When to use it

- Your parameters are correlated and you want variance fractions, not a distributional distance
- You need to separate "worth measuring" ($S_{TC}$) from "safe to fix" ($S_{TU}$)
- You want to state the dependency structure explicitly, or sweep over several
- You have existing $(X, Y)$ pairs and also want a fast surrogate (`result.predict`)

### References

- Hilhorst, G., Quicken, S., van de Vosse, F.N. & Huberts, W. (2024). Efficient sensitivity analysis for biomechanical models with correlated inputs. *International Journal for Numerical Methods in Biomedical Engineering*, 40(2), e3797.
- Li, G., Rabitz, H., Yelvington, P.E., Oluwole, O.O., Bacon, F., Kolb, C.E. & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *Journal of Physical Chemistry A*, 114(19), 6022-6032.
- Wirtz, D. & Haasdonk, B. (2013). A vectorial kernel orthogonal greedy algorithm. *Dolomites Research Notes on Approximation*, 6, 83-100.
- Santin, G. & Haasdonk, B. (2021). Kernel methods for surrogate modeling. In *Model Order Reduction, Volume 2*, De Gruyter, 311-354.

## Kucherenko (Dependent-Input Sobol' Indices)

Kucherenko, Tarantola & Annoni (2012) generalise the Sobol' indices to dependent parameters. They keep the two defining quantities and estimate them by direct model evaluation:

$$
S_i = \frac{V\!\left(\mathbb{E}(Y \mid X_i)\right)}{V(Y)}, \qquad
S_{T_i} = \frac{\mathbb{E}\!\left(V(Y \mid \mathbf{X}_{\sim i})\right)}{V(Y)}
$$

Under independent parameters these are the classic first-order and total-order Sobol' indices. Under a declared `problem.correlation` they keep their exact conditional-variance meaning. $S_i$ becomes correlation-inclusive: what $X_i$ explains through itself, plus what it explains through its coupling, which is VKOGA's $S_{TC}$. $S_{T_i}$ becomes correlation-exclusive: what only $X_i$ can explain, which is VKOGA's $S_{TU}$. $S_{T_i} \ge S_i$ no longer holds in general. For strongly coupled parameters the total index drops below the first-order one, and the gap is the correlation-borne share.

This is the design-based counterpart to [VKOGA](#vkoga-correlated-input-variance-indices): the same two quantities, but estimated on your actual model instead of a fitted surrogate. Kucherenko needs its own design, so pick it when you can still run the model and want estimates free of surrogate error. Choose VKOGA when all you have is existing $(X, Y)$ data. The test suite pins both routes to the same closed-form linear-Gaussian reference, and to each other.

### How it works

1. `jaxgsa.kucherenko.sample(problem, n)` builds a design of $N(2D+1)$ rows, where $N$ is `n` rounded up to a power of two. It contains one joint block drawn from the full copula, then per parameter one block where $X_i$ is kept and the rest is redrawn from $p(\mathbf{X}_{\sim i} \mid X_i)$ (for $S_i$), and one block where the rest is kept and $X_i$ is redrawn from $p(X_i \mid \mathbf{X}_{\sim i})$ (for $S_{T_i}$). Both conditionals are closed-form Gaussians in the latent copula space, so no iterative sampler is involved. Under an identity correlation the design reduces exactly to the Saltelli column-swap scheme.
2. You evaluate your model on `samples.samples`. This is the whole model cost.
3. `jaxgsa.kucherenko.analyze(samples, Y)` applies the single-loop estimators: the paired product over the shared-$X_i$ rows for $S_i$, and the Jansen squared difference over the shared-$\mathbf{X}_{\sim i}$ rows for $S_{T_i}$. The exact formulas are stated in the `jaxgsa.kucherenko._analyze` module docstring.

`kucherenko.sample` reads `problem.correlation` and is deliberately exempt from the correlated-design error on `sobol` / `morris` / `efast`, because conditioning on the declared copula is the method's purpose. Categorical problems raise, since the conditional copula needs continuous marginals, as do problems with fewer than two parameters.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_i$ | $V(\mathbb{E}(Y \mid X_i)) / V(Y)$. The classic first-order Sobol' index under independence; correlation-inclusive under a declared correlation (VKOGA's $S_{TC}$). |
| $S_{T_i}$ | $\mathbb{E}(V(Y \mid \mathbf{X}_{\sim i})) / V(Y)$. The classic total-order Sobol' index under independence; correlation-exclusive under a declared correlation (VKOGA's $S_{TU}$). $S_{T_i} \ge S_i$ no longer holds in general. |

### When to use it

- Your parameters are correlated, you want $S_1$/$S_T$ with their exact conditional-variance meaning, and you can still run the model
- You want a design-based cross-check of a VKOGA (surrogate) analysis
- Your parameters are independent and you want the classic Sobol' indices from a conditional design (it reduces to them exactly)

### Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Computer Physics Communications*, 183(4), 937-946.

## Output Shapes

All thirteen methods share the same output contract: scalar, multi-output, and time-series outputs. The shape of `Y` determines the shape of all returned index arrays. Read `S1 / ST` as the method's per-parameter measures — `mu / mu_star / sigma` for Morris, and `nu / sigma` and the bounds for DGSM. Only Sobol and PCE produce S2.

| Y shape | S1 / ST shape | S2 shape |
|---------|---------------|----------|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. Confidence interval arrays (when using bootstrap) prepend a leading dimension of 2 for `[lower, upper]`.

How a 2-D `Y` is read depends on `problem.output_names`. Without it, a 2-D `Y` is always `(N, K)`: multiple outputs, no time dimension. With exactly one entry in `output_names` and more than one column, a 2-D `(N, M)` `Y` is read as `M` timepoints of that single labeled output and flows through as `(N, M, 1)`, keeping the labeled output axis in results. A lone column `(N, 1)` stays a scalar output `(N, K=1)`; pass `(N, 1, 1)` explicitly for a genuine 1-timepoint series. With several entries, the column count must equal `len(output_names)`. A 1-D `(N,)` `Y` is one output regardless of how many names are declared.

You need not pass exactly the canonical layout, because every public entry point resolves `Y` through the same inference ladder. Exact canonical shapes pass silently. Unambiguously recoverable layouts — a transposed `(K, N)` array, or a 3-D `(N, K, T)` array whose middle axis matches `len(output_names)` — are fixed with a `JaxgsaWarning` naming the transformation. Ambiguous layouts raise. jaxgsa never guesses.

Every warning that jaxgsa raises uses the `JaxgsaWarning` category. The class is a subclass of `UserWarning`, so a filter on `UserWarning` still catches it. Filter on `JaxgsaWarning` to select the jaxgsa warnings alone:

```python
import warnings
from jaxgsa import JaxgsaWarning

warnings.filterwarnings("ignore", category=JaxgsaWarning)
```

Time-series outputs are particularly useful for dynamic models. Watching the sensitivity indices evolve over time reveals which parameters dominate at different stages of a process — for example, a parameter that is highly influential early in a batch but negligible later.

## Failed model runs

A model that fails on some of its runs returns `NaN` or `Inf`. Every `analyze()` function takes the same `on_invalid` keyword to say what should happen then.

| Value | What it does |
| --- | --- |
| `"raise"` | Refuse the analysis. This is the default. |
| `"propagate"` | Compute anyway, and let the non-finite value reach the indices. |
| `"drop"` | Remove the affected data, and use what is left. |

The default refuses, because an index computed from part of a sample is a different quantity from the one you asked for, and `analyze()` is cheap to run again once you know which runs failed.

What `"drop"` removes depends on the design. A Saltelli group, a Morris trajectory and a Kucherenko base point are each read as one block, so a single bad value removes the whole block. Keeping part of a block would leave the estimator reading rows that no longer line up, and nothing would report an error. For the methods that take any `(X, Y)` sample, one bad value removes one row. A bad input row always takes its matching output row with it.

`jaxgsa.efast.analyze()` accepts only `"raise"` and `"propagate"`. Its design is an ordered sweep read by a Fourier transform, so removing a point does not shrink the sample; it changes what the estimator computes. Asking for `"drop"` there raises and says so.

Whatever you choose, the result carries an `invalid` report:

```python
result = jaxgsa.sobol.analyze(samples, Y, on_invalid="drop")

result.invalid.n_invalid        # how many blocks held a bad value
result.invalid.unit_indices     # which blocks
result.invalid.bad_row_indices  # the rows that actually failed
result.invalid.row_indices      # every row those blocks cover
result.invalid.sources          # whether the bad values were in X, in Y, or both
```

The positions are the useful part: they name the model runs to investigate.

`bad_row_indices` and `row_indices` answer different questions, and for a block design the difference is large. One failed run inside an eFAST search curve gives one entry in `bad_row_indices` and 257 in `row_indices`. The first tells you which model run to look at. The second tells you what `"drop"` would remove.

Both always refer to the array as you passed it. A Saltelli or Morris design is analysed in an expanded form that repeats rows, but you evaluated the model once per unique row, so the report is translated back to the numbering you hold.

## References

- Sobol', I.M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.
- Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*, 181(2), 259-270.
- Jansen, M.J.W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1-2), 35-43.
- Li, G., Rabitz, H., Yelvington, P. E., Oluwole, O. O., Bacon, F., Kolb, C. E. & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *The Journal of Physical Chemistry A*, 114(19), 6022-6032. (Defines the SCSA method and its per-term indices $S$, $S^a$, $S^b$ in Eqs. 19-22, the per-input totals in Section 2.2.3, and the $\sum_j S_{p_j} \approx 1$ reliability criterion in Eq. 24.)
- Sarazin, G., Viaud, C. & Cournède, P.-H. (2017). Analyse de sensibilité globale pour les modèles à entrées corrélées. *Journal de la Société Française de Statistique*, 158(1), 68-89. (Restates the SCSA total as $S_T(u) = \sum_{v \supseteq u} S(v)$ in Eq. 8, and states explicitly that it is no longer confined to $[0, 1]$ in the correlated case.)
- Rabitz, H. & Alis, O. (1999). General foundations of high-dimensional model representations. *Journal of Mathematical Chemistry*, 25(2-3), 197-233.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7), 964-979.
- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245-251.
- Song, E., Nelson, B.L. & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060-1083.
- Saltelli, A., Tarantola, S. & Chan, K.P.-S. (1999). A quantitative model-independent method for global sensitivity analysis of model output. *Technometrics*, 41(1), 39-56.
- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Computer Physics Communications*, 183(4), 937-946.
- Borgonovo, E., Figalli, A., Plischke, E. & Savaré, G. (2024). Global sensitivity analysis via optimal transport. *Management Science*. doi:10.1287/mnsc.2023.01796
