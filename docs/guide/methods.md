# Methods

jaxgsa implements thirteen methods for global sensitivity analysis (GSA). All of them answer the same broad question: which parameters actually drive my model's output? They differ in three ways. They measure different quantities, they cost different numbers of model evaluations, and some need a dedicated sampling design while others work with data you already have.

If you are new to the package, start with [Choosing a method](#choosing-a-method), then jump to the section for the method you picked. Every method section opens with what it measures, when to pick it, and what data it needs. The estimator details follow.

Throughout this page, $D$ is the number of parameters and $N$ is a sample count.

Two conventions for the code on this page. `verbose` defaults to `True` on every `analyze()` and every `sample()`, so a plain call prints a problem summary, timings and a top-k table to stdout. Every example here passes `verbose=False` so the printed output is only what the example asks for; drop it and you get the report as well. And the examples share one setup, which is Ishigami unless the text says otherwise:

```python
import jax
import jax.numpy as jnp
import numpy as np
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=4000, seed=0))
Y = evaluate(X)
```

## Choosing a method

Three questions narrow the field quickly.

1. Can you still choose where to run the model? Four methods need their own sampling design, which jaxgsa generates for you: Sobol' (Saltelli matrices), eFAST (search curves), Morris (trajectories), and Kucherenko (conditional-copula blocks for dependent parameters). The other nine are given-data methods: HDMR, PCE, Shapley effects, DGSM, HSIC, PAWN, Borgonovo delta, optimal transport, and VKOGA. They accept any set of $(X, Y)$ pairs, including simulation runs you already have. DGSM has no sampler of its own: draw plain Monte Carlo points with `jaxgsa.sampling.monte_carlo` and let autodiff do the rest.

2. What should the number mean? Variance-based methods (Sobol', HDMR, PCE, eFAST, Shapley) report fractions of output variance, as in "parameter 3 explains 40% of the output's spread". Screening methods (Morris, DGSM) trade that precision for a cheap answer to a narrower question: which parameters can I stop worrying about? They are good at finding the ones that do nothing and less good at ordering the ones that do. Both misrank Ishigami's top two; see their sections. Moment-independent methods (HSIC, PAWN, Borgonovo delta, optimal transport) measure how strongly a parameter affects the whole output distribution. Use them when your output is skewed or heavy-tailed and variance is the wrong summary. Optimal transport also splits its index into a mean-shift part and a shape-change part.

3. What is your evaluation budget? Sobol' needs $N(2D+2)$ model runs by default, where $N$ is the base sample count and is typically 128 or more. Morris needs only $r(D+1)$, with $r \approx 10\text{–}50$ trajectories. DGSM costs $N$ Jacobians, each about $\min(D,\,T K)$ evaluations; `has_aux=True` returns the primal output from the same forward/reverse pass, so the output itself costs nothing extra. It is cheap for a scalar output and stops being cheap for a long time series. The given-data methods cost nothing beyond the runs you already have.

One thing to get straight before you read any cost formula on this page. The $N$ in $N(2D+2)$ is the base sample count, but `jaxgsa.sobol.sample(problem, n_samples)` takes the **total** evaluation budget and picks the base count for you. `sample(problem, 8192)` on a 3-parameter problem gives 8192 model runs from a base count of 1024, not 8192 × 8. `jaxgsa.efast.sample` and `jaxgsa.morris.sample` take per-curve and per-trajectory counts instead, so they do multiply. Check `samples.samples.shape[0]` if you are not sure.

Common situations:

- "I can run the model freely and want the standard variance decomposition." Use [Sobol' via Saltelli sampling](#sobol-indices-via-saltelli-sampling), the reference method, with first-order, total-order, and second-order indices.
- "My model is expensive and has many parameters." Screen first with [Morris](#morris-elementary-effects-screening) at $r(D+1)$ runs, or with [DGSM](#dgsm-derivative-based-global-sensitivity-measures) if the model is JAX-differentiable. Fix the negligible parameters, then spend the remaining budget on Sobol' for the survivors.
- "I only have existing simulation data." Any given-data method works. Use [HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) or [PCE](#pce-polynomial-chaos-expansion) for variance-based indices via a surrogate, or [VKOGA](#vkoga-correlated-input-variance-indices) when the parameters are dependent. Use [HSIC](#hsic-hilbert–schmidt-independence-criterion), [PAWN](#pawn-cdf-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), or [optimal transport](#optimal-transport-wasserstein-based-sensitivity) for distribution-based indices.
- "My parameters are correlated." Sobol', PCE, eFAST, DGSM, Morris, and PCE-backed Shapley all assume independent parameters. They refuse to run when `problem.correlation` is declared. Three routes remain, depending on what you have:
  - **Declare the dependence and sample it.** Put a Gaussian-copula matrix on the `Problem` (`correlation=`, or `problem.with_correlation(R)`), then draw with `jaxgsa.sampling.monte_carlo`. A copula is a way to build correlated samples that still keep each parameter's own declared marginal distribution exactly.
  - **Analyze data you already have.** Use [VKOGA](#vkoga-correlated-input-variance-indices) for variance fractions split into a correlated part and an uncorrelated part, fitted through a kernel surrogate. Use HDMR for the ANCOVA separation, which splits each interaction term's variance into a structural share and a correlation-driven share the same way. Or use [optimal transport](#optimal-transport-wasserstein-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), HSIC, or PAWN, none of which assume independence in the first place. `shapley.analyze(backend="hdmr", include_correlative=True)` turns the HDMR split into one allocation per parameter, but read the [HDMR section](#rs-hdmr-random-sampling-high-dimensional-model-representation) first: its `ST` is not a total-effect index under dependence.
  - **Run your model on a dedicated design.** Use [Kucherenko](#kucherenko-dependent-input-sobol-indices): it samples conditionally on the declared copula, evaluates your actual model, and returns $S_1$/$S_T$ under the declared dependence. See [Correlated Inputs](/examples/correlated-inputs) for a worked example of all three routes.

  The four variance-based routes measure different things and disagree on the same data. [Four indices under dependence](#four-indices-under-dependence) puts them side by side.
- "Some of my parameters are categorical." Declare them with `{"dist": "categorical", "probs": [...]}`, and samples then carry integer level codes. Four methods handle unordered levels correctly: [Sobol'](#sobol-indices-via-saltelli-sampling), because the Saltelli column-swap scheme is distribution-agnostic, plus [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), [optimal transport](#optimal-transport-wasserstein-based-sensitivity), and [PAWN](#pawn-cdf-based-sensitivity), which all condition on one class per level. Every other method refuses with a `ValueError`, because its indices would depend on the arbitrary code order. See [Categorical Inputs](/examples/categorical-inputs).
- "I need to decide what to measure more accurately, or what to hold fixed." Use [VKOGA](#vkoga-correlated-input-variance-indices): $S_{TC}$ is the prioritisation measure and $S_{TU}$ the fixing measure. Under dependence they can rank parameters very differently.
- "My output distribution is skewed or heavy-tailed." Use [PAWN](#pawn-cdf-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), or [optimal transport](#optimal-transport-wasserstein-based-sensitivity). All three compare whole output distributions rather than variances.
- "I want to know how a parameter matters: shift or shape?" Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity). Its index decomposes exactly into an advective (mean-shift, close to $S_1/2$) and a diffusive (spread/shape) component.
- "I want one number per parameter for a whole trajectory." Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) with `mode="trajectory"`. Point-cloud transport scores each parameter against the entire time course jointly.
- "I want one fair importance number per parameter that sums to 1." Use [Shapley effects](#shapley-effects).
- "I also want a fast surrogate of my model." Use HDMR or PCE and call `result.predict(...)`.

### One model, four answers

The methods disagree, and the disagreement is the point. Here is Ishigami, $f = \sin x_1 + 7\sin^2 x_2 + 0.1 x_3^4 \sin x_1$, under four methods. Parameter $x_3$ is the interesting one: on its own it does nothing, and it only acts through its product with $x_1$.

```python
samples = jaxgsa.sobol.sample(PROBLEM, 8192, seed=0, verbose=False)
Ys = evaluate(samples.samples)

sobol = jaxgsa.sobol.analyze(samples, Ys, verbose=False)
morris = jaxgsa.morris.analyze(samples.to_morris(verbose=False), Ys, verbose=False)
ot = jaxgsa.optimal_transport.analyze(PROBLEM, X, Y, verbose=False)

print("S1     ", sobol.S1)
print("ST     ", sobol.ST)
print("mu*    ", morris.mu_star)
print("advect ", ot.advective)
print("diffuse", ot.diffusive)
```

```
S1      [0.32232326 0.43612355 0.00139014]
ST      [0.55598414 0.44165453 0.24129711]
mu*     [ 8.70476  15.02531   6.620432]
advect  [0.15357937 0.21982558 0.00371554]
diffuse [0.04772939 0.0577217  0.09400754]
```

Four readings of the same model:

- Sobol' says $x_3$ owns 0.1% of the variance alone and 24% once you count its interaction with $x_1$. The 24-point gap is the whole story about $x_3$, and only a method with a total-order index tells you it exists.
- Morris ranks $x_2 > x_1 > x_3$, which is not the $S_T$ ranking $x_1 > x_2 > x_3$. $\mu^*$ is a mean absolute slope, not a variance share, and the two top parameters swap. If you screen with Morris and then drop everything but the top parameter, you drop the wrong one here. Drop only what is near the origin of the $\mu^*$–$\sigma$ plot.
- Optimal transport splits $x_3$'s influence into 0.004 of mean shift and 0.094 of shape change. That is a quantitative statement that $x_3$ changes the spread of the output without moving its mean. No variance-based index says that.
- The Morris measures cost zero extra model runs here: `to_morris()` reinterprets the Saltelli design you already paid for. See [Free screening from a Sobol' design](#free-screening-from-a-sobol-design).

### Method capabilities

This table is the one place that records what each method accepts. The other
pages link here instead of repeating it. `tests/test_docs_matrix.py` checks
the Own design, Correlated, Categorical, and Bootstrap CI columns against the
method registry, and checks that every Reports cell holds prose rather than a
stray capability mark. It does not check the wording inside Reports, and it
does not check the Comparison table below, so those stay a human's
responsibility to keep in step with the code.

| Method | Reports | Own design | Correlated | Categorical | Bootstrap CI |
|---|---|:--:|:--:|:--:|---|
| [`borgonovo`](#borgonovo-delta-density-based-sensitivity) | $\delta$, $S_1$ | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`dgsm`](#dgsm-derivative-based-global-sensitivity-measures) | bounds on $S_T$ | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`efast`](#efast-extended-fourier-amplitude-sensitivity-test) | $S_1$, $S_T$ | ✓ | ✗ | ✗ | — |
| [`hdmr`](#rs-hdmr-random-sampling-high-dimensional-model-representation) | $S_a$ / $S_b$ / $S$ per term, surrogate | ✗ | ✓ † | ✗ | `n_bootstrap` |
| [`hsic`](#hsic-hilbert–schmidt-independence-criterion) | dependence measure | ✗ | ✓ § | ✗ | — |
| [`kucherenko`](#kucherenko-dependent-input-sobol-indices) | $S_1$, $S_T$ under dependence | ✓ | ✓ | ✗ | `n_bootstrap` |
| [`morris`](#morris-elementary-effects-screening) | $\mu^*$, $\sigma$ | ✓ | ✗ | ✗ | `n_bootstrap` |
| [`optimal_transport`](#optimal-transport-wasserstein-based-sensitivity) | $W_2^2$ index, advective + diffusive | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pawn`](#pawn-cdf-based-sensitivity) | KS distance | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pce`](#pce-polynomial-chaos-expansion) | $S_1$, $S_2$, $S_T$, surrogate | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`shapley`](#shapley-effects) | allocation summing to 1 | ✗ | ✗ ‡ | ✗ | `n_bootstrap` |
| [`sobol`](#sobol-indices-via-saltelli-sampling) | $S_1$, $S_2$, $S_T$ | ✓ | ✗ | ✓ | `n_bootstrap` |
| [`vkoga`](#vkoga-correlated-input-variance-indices) | $S_{TC}$, $S_{TU}$, $S_U$, $S_C$, $S_{IU}$, surrogate | ✗ | ✓ | ✗ | `n_bootstrap` |

**Own design** means the method builds its own sample matrix, so you must be
able to run the model at the points it chooses. The other nine are given-data
methods. They accept any $(X, Y)$ pairs you already have.

**Correlated** and **Categorical** say what the method does with a problem
that declares a Gaussian-copula correlation, or that declares a categorical
parameter. A ✗ is a refusal, not a silent approximation. The method raises a
`ValueError` that names the parameters and the alternatives.

**Bootstrap CI** gives the keyword that asks for bootstrap confidence
intervals. There is one spelling, `n_bootstrap`, and it defaults to `0`
everywhere, so you never pay for an interval you did not ask for. Passing
`n_bootstrap > 0` without a `key` raises `ValueError: key is required when
n_bootstrap > 0`; pass `key=jax.random.key(0)` so the interval is
reproducible. See [Confidence intervals](/api/#confidence-intervals) for the
`result.ci` record that comes back with them.

Two methods have no entry in that column, and the gap is deliberate. eFAST has
one search curve per parameter, so there is nothing to resample: removing a
point does not shrink the sample, it changes what the estimator computes. An
eFAST interval would need replicated designs with different random phase
shifts, which is a change to `sample()`, not a keyword on `analyze()`.
**HSIC** already reports permutation `p_values`, which is the uncertainty
statement for a V-statistic; a row bootstrap would repeat rows onto the kernel
diagonal, where the kernel is exactly 1, so the resampled index is biased
upward by construction.

The four surrogate-backed methods, `pce`, `hdmr`, `vkoga` and `shapley`,
refit their surrogate on every replicate, so an interval there costs an order
of magnitude more than a row resample on a direct estimator. That is why the
default is `0` and not why it is unavailable.

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
| Parameter distributions ‖ | Uniform + Gaussian | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian | Uniform + Gaussian (both backends) | Uniform + Gaussian | Uniform + Gaussian | Uniform + Gaussian (truncated-quantile grid) | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian (via CDF mapping) | Any (rank-based classes; marginals not used) | Any (rank-based classes; marginals not used) | Uniform + Gaussian (via CDF mapping) | Uniform + Gaussian (latent-copula inverse CDF) |
| Output shapes | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series (both backends) | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series; joint point-cloud modes over outputs/time | Scalar, multi-output, time-series | Scalar, multi-output, time-series |
| What the numbers mean | Exact variance fractions (given enough samples) | Variance fractions from a B-spline surrogate (fit-dependent) | Variance fractions from a polynomial surrogate (fit-dependent) | Exact allocation within the fitted surrogate; depends on fit quality | Exact variance fractions (given enough samples) | Bounds on $S_T$, not exact indices | Screening ranks ($\mu^*$ as $S_T$ proxy), not variance fractions | Dependence measure, not variance fractions | Distributional (KS) distance, not variance fractions | Distributional (L1) distance, not variance fractions | Distributional ($W_2^2$) distance in $[0,1]$, split into mean-shift + shape parts | Correlated and uncorrelated variance fractions from a kernel surrogate (fit-dependent) | Exact conditional-variance fractions under the declared dependence (given enough samples) |
| Second-order indices | Direct estimation from cross-matrices | From interaction component functions | Analytical from coefficients | Not available (interaction variance folded into $\mathrm{Sh}$) | Not available | Not available | Not available | Not available | Not available | Not available | Not available | Not available | Not available |
| Interaction detection | Via $S_2$ and the gap $S_T - S_1$ | Via explicit interaction component functions | Via $S_2$ from coefficients | Via the gaps $\mathrm{Sh} - S_1$ and $S_T - \mathrm{Sh}$ | Via the gap $S_T - S_1$ only | Not available (bounds only) | Via large $\sigma$ relative to $\mu^*$ (not pair-attributable) | Via the Total HSIC − R2-HSIC gap | Not available (first-order only) | Not available (the $\delta - S_1$ gap flags influence beyond first-order variance) | Not available (the diffusive component flags influence beyond mean shift) | Via $S_{IU}$, the independent-interaction index | Via the gap $S_T - S_1$ under independence (under correlation the gap mixes interactions and coupling) |
| Reusable surrogate | No | Yes (`result.predict`) | Yes (`result.predict`) | Derived from either fitted result | No | No | No | No | No | No | No | Yes (`result.predict`) | No |

‖ A truncated Gaussian marginal is not special to any one row here; every method that accepts a Gaussian marginal accepts a truncated one the same way it accepts the untruncated case. DGSM is the exception worth naming: it needs the Poincaré constant of the truncated marginal, computed by a finite-element spectral solve rather than read off a closed form, so it earns its own paragraph in [Poincaré constants by distribution](#poincare-constants-by-distribution) even though the cell above says the same "Uniform + Gaussian" as its neighbours.

### Four indices under dependence

There is no single generalisation of the Sobol' indices to dependent inputs. There are several, they measure different things, and they disagree on the same data. jaxgsa ships four variance-based routes. Pick by the question you are asking, not by which one is closest to hand.

| Route | What it estimates | What it needs | Ask for it when |
|-------|-------------------|---------------|-----------------|
| [`kucherenko`](#kucherenko-dependent-input-sobol-indices) | $S_1 = V(\mathbb{E}(Y \mid X_i))/V(Y)$ and $S_T = \mathbb{E}(V(Y \mid \mathbf{X}_{\sim i}))/V(Y)$, exactly, under the declared copula. $S_1$ is correlation-inclusive, $S_T$ correlation-exclusive. | Its own design, $N(2D+1)$ model runs, and a declared correlation matrix. | You can still run the model and you want the conditional-variance quantities with no surrogate in the chain. |
| [`vkoga`](#vkoga-correlated-input-variance-indices) | The same two quantities as $S_{TC}$ and $S_{TU}$, plus $S_U$, $S_C$ and $S_{IU}$, sampled from a fitted kernel surrogate. | Any $(X, Y)$ pairs and a declared correlation matrix. | You cannot run the model again, or you want to sweep the same data under several correlation assumptions. |
| [`hdmr`](#rs-hdmr-random-sampling-high-dimensional-model-representation) ANCOVA split | Per **term**, not per parameter: each component function's variance split into a structural share $S_a$ and a correlation-driven share $S_b$. | Any $(X, Y)$ pairs. The correlation is read implicitly out of $X$. | You want to know which interaction carries the variance, and how much of it is coupling rather than structure. |
| [`shapley(backend="hdmr", include_correlative=True)`](#shapley-effects) | One allocation per parameter, summing to 1, by splitting each term's $S_a + S_b$ among its participants. | Any $(X, Y)$ pairs. | You want a single fair-share number per parameter and you accept an ANCOVA attribution. |

Three things to hold on to.

**They are different estimands.** A disagreement between them is not a bug in one of them. `kucherenko` and `vkoga` estimate the same pair of conditional-variance quantities and should agree up to surrogate and Monte Carlo error; the test suite pins both to the same closed-form linear-Gaussian reference. The HDMR split and the ANCOVA Shapley allocation estimate something else entirely and have no reason to match.

**Only two of the four are conditional-variance indices.** HDMR's $S_T$ under dependence is a term-membership sum, not a total-effect index, and the ANCOVA Shapley allocation is not the conditional-variance Shapley effects of Song et al. (2016); its correlative shares can be negative. Both are documented in their own sections and both warn at runtime.

**None of them is comparable to `jaxgsa.sobol`.** `sobol` refuses a correlated problem, and it is right to. Under dependence a first-order index that includes coupling is a different number from one that does not, so a `kucherenko` $S_1$ and a `sobol` $S_1$ do not belong in the same table.

For the distribution-based alternatives, which never assumed independence in the first place, see [optimal transport](#optimal-transport-wasserstein-based-sensitivity), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), [HSIC](#hsic-hilbert–schmidt-independence-criterion), and [PAWN](#pawn-cdf-based-sensitivity). [Correlated Inputs](/examples/correlated-inputs) works one model through several of these routes side by side.

## Background: variance-based sensitivity analysis

### Why global sensitivity analysis?

Local sensitivity methods, such as partial derivatives at a nominal point, describe the model at one location. Global sensitivity analysis explores the entire parameter space instead. This matters for non-linear models, where interactions and non-monotonic responses mean a gradient at one point can be misleading. GSA quantifies each parameter's contribution to output uncertainty across the whole parameter domain.

In practice, GSA serves several roles:

- **Parameter identifiability**: parameters with near-zero sensitivity across all outputs are effectively unidentifiable from data and may need to be fixed rather than estimated; high-sensitivity parameters are the ones data can constrain.
- **Experimental design**: for time-series outputs, watching sensitivity indices evolve over time helps pick measurement times when outputs are most informative about the parameters of interest.
- **Model simplification**: if interaction indices are negligible, the model response is approximately additive, and simpler surrogate models may suffice.

### The Hoeffding–Sobol' decomposition

The theoretical foundation of variance-based GSA is the Hoeffding (ANOVA) decomposition. Any square-integrable function $f(\mathbf{X})$ of $D$ independent parameters can be uniquely decomposed into summands of increasing dimensionality:

$$
f(\mathbf{X}) = f_0 + \sum_{i=1}^{D} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots + f_{1,2,\ldots,D}(X_1, \ldots, X_D)
$$

where $f_0 = \mathbb{E}[f(\mathbf{X})]$ is the overall mean, each $f_i$ captures the main effect of parameter $i$, each $f_{ij}$ captures the pairwise interaction between $i$ and $j$, and so on. Because these component functions are mutually orthogonal, the total output variance decomposes additively:

$$
\mathrm{Var}(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,D}
$$

where $V_i = \mathrm{Var}[f_i(X_i)]$, $V_{ij} = \mathrm{Var}[f_{ij}(X_i, X_j)]$, etc.

### Sobol' sensitivity indices

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

## Sobol' indices via Saltelli sampling

Sobol' indices split the output variance into the share each parameter owns alone and the share it owns through interactions. This is the reference method and jaxgsa's default workflow: an exact, model-free variance decomposition with well-understood convergence.

Pick it when you can afford a dedicated sampling design and your parameters are independent. The method needs its own design, so you must be able to run the model at points jaxgsa chooses. jaxgsa uses the Saltelli sampling scheme (Saltelli 2002, 2010), which arranges quasi-random sample matrices so that first-order ($S_1$), total-order ($S_T$), and second-order ($S_2$) indices can all be estimated from a single batch of model evaluations.

### The Saltelli column-swap scheme

The method generates two independent $N \times D$ quasi-random sample matrices $\mathbf{A}$ and $\mathbf{B}$ using a Sobol' low-discrepancy sequence (via `scipy.stats.qmc.Sobol`). For each parameter $j$, a cross-matrix $\mathbf{AB}^{(j)}$ is constructed by taking all columns from $\mathbf{A}$ except column $j$, which is replaced by column $j$ from $\mathbf{B}$. This column-swap construction allows conditional expectations to be estimated via sample averages.

The cost is $N(D + 2)$ model evaluations for all first-order and total-order indices, or $N(2D + 2)$ when second-order indices are included (`calc_second_order=True`, the default).

### Estimators

The default estimator pair is `estimator="saltelli-jansen"`: Sobol'-Mauntz for the first order, Jansen (1999) for the total order. Two reasons pick it. Jansen's total-order estimator is a mean of squares, so it can never come out negative, and users screen on $S_T$. A negative $S_T$ invites the clipping that jaxgsa refuses to do. And SALib computes the same pairing by default, so moving between the two libraries needs no keyword.

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

Note the limit of that last point. Only the *total* order of `"saltelli-jansen"` and `"jansen"` is a bare sum of squares, and only it is guaranteed non-negative. A Jansen *first-order* estimate is one minus such a term, so it is bounded above by 1 and free to go below zero. Measured on Sobol-G over 40 seeds at `base_n` 64/256/1024/4096, the default estimator's negative-$S_1$ rate is 100%/88%/65%/30%, the Jansen family's is 50%/58%/30%/45%, and `"azzini-rosati"`'s is 0% throughout. The rate falls as $N$ grows for the default estimator, does not fall monotonically for the Jansen family, and is exactly zero for `"azzini-rosati"` by construction (see the table above).

So read a negative value as "the interval covers zero", and turn on the bootstrap (`n_bootstrap`, with a `key`) to see that directly. Investigate only if the value is large, if it appears for a parameter whose index is demonstrably not near zero, or if it grows with $N$.

jaxgsa does not clip. Clipping to zero is a display choice, and it must never be done before ranking: it biases upward in exactly the near-zero regime where the ranking decision is being made.

### How to use it

1. `jaxgsa.sobol.sample()` generates the Sobol' quasi-random sequence and builds the Saltelli cross-matrices. Duplicate rows are removed so your model only evaluates unique sample points.
2. You evaluate your model on `sampling_result.samples`.
3. `jaxgsa.sobol.analyze()` reconstructs the Saltelli layout internally and computes all indices in a single `jit(vmap(...))` pass.

`jaxgsa.sobol.analyze()` always standardizes each output slice over the sample axis before it computes the estimators. The Saltelli/Sobol'-Mauntz $S_1$ estimator and every $S_2$ estimator are uncentred products. A non-zero output mean therefore biases them. The standardization removes that bias. SALib standardizes in the same way. When bootstrapping (`n_bootstrap > 0`, with a `key`), `ci_method="quantile"` reports percentile bootstrap bounds and `ci_method="gaussian"` reports symmetric bounds from the bootstrap standard deviation. Either way, jaxgsa returns explicit lower/upper endpoint arrays rather than SALib's symmetric confidence widths.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | Fraction of output variance due to parameter $i$ alone (main effect). |
| $S_T(i)$ | Fraction of output variance due to parameter $i$ including all its interactions. $S_T \geq S_1$ holds for the true population values, but not for every estimator's finite-sample output: `"azzini-rosati"` enforces it sample-wise, the others do not and can print $S_1 > S_T$ on noisy data. |
| $S_2(i,j)$ | Fraction of output variance due to the pairwise interaction between $i$ and $j$, beyond their individual effects. |

### When to use it

- You can afford the structured Saltelli design: $N(D+2)$ evaluations for first-order and total-order only, or $N(2D+2)$ with second-order (the default)
- You want an exact, model-free variance decomposition
- Your parameters are independent

If you can run the model and your parameters are independent, this is the first thing to try. It is the only method here that gives you $S_1$, $S_2$ and $S_T$ from one design with no surrogate in between.

### When it is the wrong choice

- **Your parameters are correlated.** `analyze` refuses, and it is right to. Use [Kucherenko](#kucherenko-dependent-input-sobol-indices) if you can still run the model, [VKOGA](#vkoga-correlated-input-variance-indices) if you cannot.
- **You cannot choose the sample points.** No amount of existing $(X, Y)$ data can be reshaped into a Saltelli design. Go to [PCE](#pce-polynomial-chaos-expansion), [HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) or [Borgonovo delta](#borgonovo-delta-density-based-sensitivity).
- **Your budget is under about $100 \times D$ runs.** At that size the estimates carry more sampling noise than signal, and you are better off screening with Morris at $r(D+1)$ and coming back once you have fixed the inert parameters.
- **Variance is the wrong summary.** A bimodal or heavy-tailed output makes $\mathrm{Var}(Y)$ a poor denominator. The indices are still correct; they just answer a question you did not mean to ask. Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) or [Borgonovo delta](#borgonovo-delta-density-based-sensitivity).
- **You only need a ranking.** Second-order indices cost you $D$ extra columns of design. Pass `calc_second_order=False` and nearly halve the bill.

## RS-HDMR (Random Sampling High-Dimensional Model Representation)

RS-HDMR is a variance-based method that works from data you already have. It fits a B-spline surrogate to any set of $(X, Y)$ pairs. It then derives sensitivity indices analytically from the surrogate's variance decomposition.

Pick it in three situations. Model runs are expensive and you want to reuse existing data. Your parameters may be correlated. Or you also want a fast emulator of the model. No sampling design is required.

### Theoretical background

High-Dimensional Model Representation (HDMR) exploits the observation that, for many practical problems, only the low-order interactions among parameters significantly influence the output. The RS-HDMR variant constructs component functions from randomly sampled parameter and output data, rather than requiring structured grids. The model is decomposed as:

$$
f(\mathbf{X}) \approx f_0 + \sum_{i} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \sum_{i<j<k} f_{ijk}(X_i, X_j, X_k)
$$

where each component function is expanded in a B-spline basis with Tikhonov regularisation. Only the first-order terms $f_i$ are fitted by backfitting, which is needed there because the terms are coupled; the second- and third-order terms are each a single ridge solve.

### ANCOVA decomposition

The classical Sobol' decomposition assumes independent parameters. RS-HDMR instead uses an ANCOVA (analysis of covariance) decomposition, which separates each component's variance into two parts:

- **Structural variance ($S_a$)**: the contribution that would remain if all parameters were independent. It is the analogue of the classical Sobol' index.
- **Correlative variance ($S_b$)**: the additional contribution arising from correlations between parameters.

This distinction matters because many real-world models have correlated parameters, for example coupled physical parameters. Conflating structural and correlative contributions can produce misleading sensitivity rankings.

### How to use it

1. You provide any set of $(X, Y)$ pairs. No sampling design required.
2. `jaxgsa.hdmr.analyze()` maps parameters to $[0, 1]$ via their marginal CDFs, builds B-spline basis matrices, backfits the first-order component functions with Tikhonov regularisation, and fits every higher-order component in one ridge solve each.
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

It can be negative, because $S_b$ can be. It is not bounded in $[0, 1]$. Sarazin, Viaud & Cournède (2017), who restate the total as their Eq. (8), say so explicitly. It does not measure the expected variance reduction $\mathrm{E}[\mathrm{Var}(Y \mid X_{\sim i})] / \mathrm{Var}(Y)$ that a total-order index normally reports, so it does not answer the parameter-fixing question. The bias runs toward "cannot be fixed", and it can be an order of magnitude. On $Y = X_1 + X_2 + X_3$ with standard normal marginals and $\mathrm{corr}(X_1, X_2) = 0.95$, 8192 samples, HDMR reports `ST = [0.398, 0.397, 0.207]`. The true conditional-variance totals are $[0.020, 0.020, 0.204]$. HDMR is right about the independent parameter and 20 times too high on the two coupled ones, which are the two it was asked about.

The source paper invites the confusion. Its Eq. (4) uses the symbol $S_{Ti}$ for the classical conditional-variance total, and Section 2.2.3 reuses the same symbol for the term-membership sum. Only the second is what HDMR reports.

Li et al. also attach a precondition to the totals. They are reliable only when the per-term $S$ values sum to about 1 (Eq. 24). The shortfall is the variance the surrogate leaves unexplained. `analyze` reads that sum and warns when it falls below 0.5 or rises above 1.3 on any output slice, so a decomposition that never captured the model says so before you rank anything.

$S_1$ has the matching caveat: it is the structural share $S_a$ of the first-order term, not the Sobol' first-order index.

When you need a conditional-variance total under dependence, use [Kucherenko](#kucherenko-dependent-input-sobol-indices) ($S_T$) or [VKOGA](#vkoga-correlated-input-variance-indices) ($S_{TU}$, the parameter-fixing measure). HDMR's own contribution under dependence is the per-term $S_a$ versus $S_b$ split, which neither of those provides. `jaxgsa.hdmr.analyze()` emits one `JaxgsaWarning` on a correlated problem to say all of this.
:::

### When to use it

- Model evaluations are expensive and you want to reuse existing runs
- Parameters may be correlated, and you want the per-term structural ($S_a$) versus correlative ($S_b$) split. Read $S_T$ with care under dependence. See the note above
- You need a surrogate for fast prediction at new parameter values (`result.predict`)

### When it is the wrong choice

- **You want a total-order index under dependence.** Read the warning above; HDMR's $S_T$ is a different quantity. Use [Kucherenko](#kucherenko-dependent-input-sobol-indices) or [VKOGA](#vkoga-correlated-input-variance-indices).
- **Your model is smooth and you only want $S_1$/$S_2$/$S_T$.** [PCE](#pce-polynomial-chaos-expansion) fits in one linear solve and reads the indices off the coefficients. HDMR backfits only its first-order components, up to `maxiter=100` sweeps with an early stop once the coefficients settle (relative to their own scale); every higher-order component is a single ridge solve, no backfitting. That gives you the same numbers with more knobs to get wrong. Reach for HDMR when you specifically want the per-term $S_a$/$S_b$ split, or when the response has kinks a polynomial cannot follow.
- **`result.S.sum()` is far from 1.** That is unexplained variance, and every index derived from the fit inherits it. `analyze` warns about it. Raise `maxorder` or `m`, or accept that this model does not decompose into low-order terms.
- **Any of your parameters is categorical.** HDMR raises. Use [Sobol'](#sobol-indices-via-saltelli-sampling), [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), [optimal transport](#optimal-transport-wasserstein-based-sensitivity) or [PAWN](#pawn-cdf-based-sensitivity).

## PCE (Polynomial Chaos Expansion)

PCE is the second surrogate-based route to Sobol indices that works from data you already have. It fits an orthogonal polynomial surrogate to $(X, Y)$ data and reads the indices directly from the expansion coefficients (Sudret, 2008), with no Monte Carlo estimation noise.

Pick it when your model is smooth. Any set of $(X, Y)$ pairs works, so no sampling design is required. The polynomial basis follows the Wiener-Askey scheme: Legendre polynomials for uniform parameters, and Hermite polynomials for unbounded Gaussian parameters. A truncated Gaussian parameter keeps the Hermite basis when both truncation bounds sit at least 7 standard deviations out and the order is at most 7, because Hermite is still close enough to orthonormal there; inside that width it switches to Legendre polynomials after CDF mapping to $[-1, 1]$, which is exactly orthonormal at any truncation.

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
| Leave-one-out RMSE | Cross-validation error of the fitted surrogate. A fit-quality diagnostic, not a per-parameter index. |
| `explained_variance` | Coefficient of determination of the fit: the sample variance of the fitted values over the sample variance of $Y$. In $[0, 1]$. An in-sample fit-quality diagnostic, not a per-parameter index. |

### Check the fit before you read the indices

The indices are exact within the fitted polynomial. If the polynomial is wrong, they are exactly wrong. `order` defaults to 3, and 3 is not enough for anything with a strong nonlinearity. Ishigami makes this concrete:

```python
for order in (3, 6, 10):
    r = jaxgsa.pce.analyze(PROBLEM, X, Y, order=order, verbose=False)
    print(order, round(float(r.explained_variance), 3),
          np.round(np.asarray(r.loo_rmse), 3), np.round(np.asarray(r.S1), 3))
```

```
3 0.463 2.72 [0.662 0.054 0.001]
6 0.982 0.512 [0.32  0.442 0.   ]
10 1.0 0.006 [0.314 0.442 0.   ]
```

Ishigami's analytical $S_1$ is $[0.3139, 0.4424, 0]$. At `order=3` the surrogate captured 46% of the variance and reported $S_1(x_1) = 0.662$, twice the truth, and $S_1(x_2) = 0.054$ against a truth of 0.442. The indices are not noisy. They are the correct indices of a cubic that is not this model.

Two numbers decide whether to trust a PCE result, and both come back on the result. `explained_variance` should sit near 1. `loo_rmse` should be small next to `Y.std()`, which is 3.69 here. At `order=10` both pass and $S_1$ matches the analytical values to three decimals.

For PCE, `explained_variance` is a coefficient of determination: the sample variance of the fitted values over the sample variance of `Y`. It measures the fit in sample, so it lies in $[0, 1]$ and cannot rise above 1. It also keeps climbing as you add basis terms, even when the surrogate is getting worse out of sample, so it cannot detect an overfit on its own.

Raising `order` costs basis terms, not model runs, so raise it until `loo_rmse` stops falling. Watch for it turning back up: that is overfitting. `loo_rmse` is the signal, because it is the out-of-sample number. A high `explained_variance` next to a `loo_rmse` that approaches or passes `Y.std()` is the overfit signature. `pce.analyze` warns about both failures: it fires when `explained_variance` drops below 0.5 on any output slice, and separately when `loo_rmse` passes 0.71 times `std(Y)`, which is the same line read out of sample. A silent run means both diagnostics passed. `jaxgsa.pce.indices` warns about neither, because it has to stay traceable.

### When to use it

- You want analytical Sobol indices without Monte Carlo sampling noise
- Your model is smooth enough to be well-approximated by low-order polynomials
- You have mixed uniform and Gaussian parameters (the Wiener-Askey scheme selects the appropriate basis automatically)
- You need a fast surrogate (`result.predict` mirrors the training output layout)

### When it is the wrong choice

- **Your response has a discontinuity, a threshold, or a hard saturation.** Polynomials ring around a step and no `order` fixes it. Use [HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation)'s B-splines or a non-surrogate method like [Borgonovo delta](#borgonovo-delta-density-based-sensitivity).
- **You have fewer samples than basis terms.** A total-degree basis at order $p$ in $D$ parameters has $\binom{D+p}{p}$ terms: 286 at $D=3, p=10$, but 3003 at $D=10, p=5$. `jaxgsa.pce.effective_order(problem, n_samples, order=...)` tells you the order the data can actually support, capped at the `order` you pass, and `analyze` drops to it the same way. `effective_order` never exceeds `order`, whose own default is 3, so a call with no `order=` argument returns 3 whenever the data support at least that much, whatever `n_samples` is. On Ishigami, `effective_order(problem, 2000, order=10)` returns 10 and `effective_order(problem, 100, order=10)` returns 4. If it comes back at 1 or 2, PCE is not the method for this dataset.
- **Your parameters are correlated.** The orthogonality that makes the coefficients readable as variances is orthogonality under the independent product measure. `analyze` refuses. Use [VKOGA](#vkoga-correlated-input-variance-indices) or [Kucherenko](#kucherenko-dependent-input-sobol-indices).
- **You can run the model freely and want a guarantee.** A converged Saltelli estimate is model-free. PCE is only ever as good as its fit, and the fit is the thing you have to defend.

## Shapley effects

The Shapley effect $\mathrm{Sh}_i$ is a single, fairly allocated importance score per parameter. It gives each parameter its share of the output variance, with every interaction split evenly among its participants, so the scores sum to exactly 1. The method applies the Shapley value from cooperative game theory to variance-based sensitivity analysis. It treats the output variance as a payout divided among the parameters, viewed as players whose coalition worths are the partial variances of the ANOVA decomposition (Owen, 2014; Song, Nelson & Staum, 2016).

Pick it when you need one defensible number per parameter, for ranking, reporting, or budget allocation, rather than the two-sided $S_1$/$S_T$ view. Like HDMR and PCE, it works from data you already have: any set of $(X, Y)$ pairs, with no sampling design.

### Theoretical background

For independent parameters, the Hoeffding–Sobol' decomposition splits the output variance into partial variances $V_u$ indexed by subsets $u \subseteq \{1, \ldots, D\}$ of the parameters. The Shapley effect of parameter $i$ allocates each interaction's variance equally among its participants:

$$
\mathrm{Sh}_i = \sum_{u \ni i} \frac{V_u}{|u|}
$$

so a main-effect variance $V_i$ is attributed entirely to parameter $i$, a pairwise interaction variance $V_{ij}$ is split half-and-half between $i$ and $j$, and so on. Under independent parameters this yields two properties:

- **Bracketing**: $S_{1,i} \leq \mathrm{Sh}_i \leq S_{T,i}$. The Shapley effect always lies between the first-order and total-order Sobol indices.
- **Exact partition**: $S_1$ omits interactions, so $\sum_i S_{1,i} \leq 1$, and $S_T$ counts each interaction once per participant, so $\sum_i S_{T,i} \geq 1$. Shapley effects split every interaction fairly and sum to exactly 1 with no gaps or double counting.

The Hoeffding decomposition above defines the `backend="pce"` allocation, and it holds only for independent parameters. `jaxgsa.shapley.analyze(backend="pce")` refuses to run when `problem.correlation` declares a dependence structure. For correlated parameters, use `backend="hdmr"` with `include_correlative=True`. It folds HDMR's ANCOVA decomposition into the allocation: each term's structural plus correlation-induced variance ($S_a + S_b$) is split among its participants. Be clear about what that gives you. It is an ANCOVA-based attribution, and correlative shares can be negative. It is not the conditional-variance Shapley effects of Song et al. (2016), which remain future work. See [Correlated Inputs](/examples/correlated-inputs).

### How jaxgsa computes them

jaxgsa computes Shapley effects analytically from a fitted surrogate's variance decomposition. There is no permutation Monte Carlo, no conditional-variance sampling, and no external `shap` dependency:

- `backend="pce"` (default) fits a polynomial chaos expansion and groups the squared orthonormal coefficients by the support of their multi-index (Sudret, 2008), exact within the fitted polynomial.
- `backend="hdmr"` fits the RS-HDMR B-spline surrogate and uses the structural ($S_a$) variances of its component functions as the partial variances $V_u$, truncated at `maxorder`.

Both backends accept scalar `(N,)`, multi-output `(N, K)`, and time-series `(N, T, K)` `Y`.

Normalization is by the surrogate's total decomposed variance $\sum_u V_u$, so $\sum_i \mathrm{Sh}_i = 1$ exactly, the Shapley efficiency property (Owen, 2014). $S_1$ and $S_T$ from the same surrogate use the same denominator. For `backend="pce"` they therefore match `jaxgsa.pce.analyze` exactly. For `backend="hdmr"` they differ from `jaxgsa.hdmr.analyze`, which normalizes $S_1$ by $\mathrm{Var}(Y)$: `shapley`'s $S_1$ is exactly `hdmr.S1` divided by `explained_variance`, one shared factor per output slice. `ST` does not divide out as cleanly, because HDMR's $S_T$ sums structural and correlative shares per term (see the HDMR section below); the ratio to `hdmr.ST` is close to `1 / explained_variance` but varies a little per parameter.

How much of the output variance the surrogate actually captured is reported separately in the `explained_variance` field. It is an honest diagnostic rather than a silently renormalized result, but the two backends put a different quantity in it, so read it against the backend you ran.

- `backend="pce"`: a coefficient of determination. It is the sample variance of the fitted values over the sample variance of $Y$. It lies in $[0, 1]$ and cannot exceed 1. Close to 1 means the polynomial reproduces the sample; well below 1 means it misses variance, and a `JaxgsaWarning` fires. It is an in-sample number, so it does not flag an overfit. For that, read `loo_rmse` on the PCE result: the surrogate is unreliable once `loo_rmse` approaches or passes `Y.std()`, and jaxgsa warns on that ratio.
- `backend="hdmr"`: the decomposed fraction $\sum_u V_u / \mathrm{Var}(Y)$. It is close to 1 for a good fit, below 1 when truncation or fit error leaves variance unexplained, and above 1 when an overfit surrogate over-counts shared variance. A `JaxgsaWarning` is emitted when it strays far from 1 in either direction.

Interactions above `maxorder` (HDMR) or the polynomial order (PCE) are absent from the allocation.

### How to use it

1. You provide any set of $(X, Y)$ pairs. No sampling design required.
2. Call `.shapley()` on a fitted PCE or HDMR result. Each partial variance is
   allocated equally among the parameters in its interaction set.
3. The result carries `Sh` alongside `S1` and `ST` computed from the same surrogate, so the three indices are directly comparable and the ordering $S_1 \leq \mathrm{Sh} \leq S_T$ is visible at a glance.

```python
result = jaxgsa.pce.analyze(PROBLEM, X, Y, order=10, verbose=False).shapley()
print("Sh        ", result.Sh)
print("sum       ", result.Sh.sum())
print("S1        ", result.S1)
print("ST        ", result.ST)
print("explained ", result.explained_variance)
```

```
Sh         [0.4357345  0.44241282 0.12185249]
sum        0.9999999
S1         [3.1388211e-01 4.4241282e-01 1.8721769e-08]
ST         [0.5575869  0.44241285 0.24370503]
explained  0.9999987
```

Read it left to right. $x_3$ has $S_1 \approx 0$ and $S_T = 0.244$, so everything it does is an interaction. Shapley splits that interaction with $x_1$ and hands $x_3$ half of it, 0.122. $S_1$ sums to 0.756, $S_T$ sums to 1.244, and $\mathrm{Sh}$ sums to 1. The analytical Shapley effects for Ishigami are $[0.4357, 0.4424, 0.1218]$.

That `order=10` is not decoration. At the default `order=3` the same call gives `Sh = [0.803, 0.055, 0.141]` and `explained_variance` drops to 0.463. The Shapley effects are exact within the surrogate, and a bad surrogate gives you exact nonsense. Check `explained_variance` first, every time. It sits on the result for that reason, and a `JaxgsaWarning` fires when the fit is too poor to trust.

The HDMR backend takes its own knobs:

```python
result_hdmr = jaxgsa.hdmr.analyze(PROBLEM, X, Y, maxorder=2, verbose=False).shapley()
```

Backend-specific keyword arguments are not validated against the selected backend: setting a knob that belongs to the other backend (for example `backend="pce"` with `maxorder=3`) is forwarded unchanged to `pce.analyze`, which does not accept it, so you get its plain `TypeError: analyze() got an unexpected keyword argument 'maxorder'`, not a `jaxgsa`-specific message.

### Index summary

| Index | Meaning |
|-------|---------|
| $\mathrm{Sh}(i)$ | Shapley effect: parameter $i$'s fair share of decomposed variance, including an equal split of every interaction it participates in. $\sum_i \mathrm{Sh}_i = 1$ exactly (Shapley efficiency). |
| $S_1(i)$ | First-order index from the same surrogate (main effect only). |
| $S_T(i)$ | Total-order index from the same surrogate (main effect plus all interactions counted in full). |
| `explained_variance` | How much of the output variance the surrogate captured. A separate fit-quality diagnostic, not a per-parameter index. For `backend="pce"` it is the coefficient of determination of the fit, in $[0, 1]$. For `backend="hdmr"` it is the decomposed fraction $\sum_u V_u / \mathrm{Var}(Y)$, which can exceed 1. |

### When to use it

- You want a single, fairly allocated importance score per parameter that sums to exactly 1, for example for ranking, reporting, or budget allocation
- Interactions matter and you want them attributed to their participants rather than omitted ($S_1$) or double-counted ($S_T$)
- You have existing $(X, Y)$ pairs and want analytical indices without permutation Monte Carlo noise
- Your parameters are independent (required for `backend="pce"`; under a declared correlation use `backend="hdmr"` with `include_correlative=True` for the ANCOVA-based allocation)

### When it is the wrong choice

- **You want to know which parameters you can fix.** Shapley gives every parameter a positive share, because splitting an interaction gives a share to both participants. A parameter can be safe to fix and still carry a visible $\mathrm{Sh}$. $S_T$ is the fixing measure; use it.
- **You want the interaction itself.** Shapley dissolves interactions into the participants by design. If you need to know that $x_1$ and $x_3$ interact rather than that they both matter, read $S_2$ from [Sobol'](#sobol-indices-via-saltelli-sampling) or [PCE](#pce-polynomial-chaos-expansion), or the gap $S_T - S_1$.
- **You cannot get the surrogate to fit.** Everything here rides on `explained_variance`. There is no surrogate-free route to Shapley effects in jaxgsa; the conditional-variance estimator of Song et al. (2016) is not implemented.
- **Your parameters are correlated and you want a rigorous answer.** `backend="hdmr", include_correlative=True` gives an ANCOVA-based allocation whose correlative shares can go negative. It is a defensible reading, not the Song et al. Shapley effects, and it no longer has the "fair split" interpretation that makes the method attractive in the first place.

### References

- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245-251.
- Song, E., Nelson, B.L. & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060-1083.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7), 964-979.

## eFAST (extended Fourier amplitude sensitivity test)

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

The interference factor `M` (default `4`) sets how many harmonics of $\omega_0$ the first-order sum $D_1$ credits to the focal parameter, and it also sizes the frequency plan: the focal frequency is $\omega_0 = \lfloor(n\_per\_curve - 1) / (2M)\rfloor$, and the $D-1$ complementary parameters need distinct integer frequencies in $[1, \omega_0 / (2M)]$. That requirement sets the smallest usable `n_per_curve`, $4 M^2 (D - 1) + 1$; below it there are not enough distinct frequencies to hand out, and `sample` raises `ValueError`. For Ishigami ($D = 3$) at the default $M = 4$, that floor is 129.

### How to use it

1. `jaxgsa.efast.sample(problem, n_per_curve, ...)` returns an `EFASTSamples` design whose `samples` array has shape `(n_per_curve * D, D)`, where each contiguous block of `n_per_curve` rows corresponds to one parameter's search curve.
2. You evaluate your model on all `n_per_curve * D` rows of `samples.samples`, in order.
3. `jaxgsa.efast.analyze(samples, Y)` splits the output by curve, computes the Fourier spectrum for each, and extracts $S_1$ and $S_T$ indices. The interference factor `M` and the problem travel inside the `EFASTSamples` object, so they can never be mismatched between sampling and analysis.

eFAST does not produce second-order ($S_2$) interaction indices. If pairwise interactions are needed, use the Sobol workflow instead.

`EFASTSamples.save(path)` writes the design to an NPZ file and `EFASTSamples.load(path)` reads it back, including the problem, `M` and `n_per_curve`. Use it when the model runs somewhere else: save the design, ship the CSV of `samples.samples`, and load the design back to analyze the outputs weeks later. The other three design classes have the same pair.

Ishigami at `n_per_curve=2048`, so 6144 model runs:

```python
samples = jaxgsa.efast.sample(PROBLEM, 2048, seed=0, verbose=False)
result = jaxgsa.efast.analyze(samples, evaluate(jnp.asarray(samples.samples)), verbose=False)
print(result.S1, result.ST)
```

```
[3.0759403e-01 4.4230729e-01 7.8303506e-09] [0.5507463  0.46289188 0.23926514]
```

The analytical values are $S_1 = [0.3139, 0.4424, 0]$ and $S_T = [0.5576, 0.4424, 0.2437]$. Compare a Saltelli run at 8192 runs, $S_1 = [0.322, 0.436, 0.001]$: eFAST matched it on 25% fewer evaluations, and its $S_1(x_3)$ came back at $8 \times 10^{-9}$ rather than the small negative a Saltelli estimator can produce. The error sits in $S_T(x_2)$, 0.463 against 0.442, which is eFAST's known upward bias on the total order from harmonic interference.

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

### When it is the wrong choice

Honestly, most of the time. Saltelli gives you $S_2$ as well, takes `on_invalid="drop"`, and supports bootstrap intervals, none of which eFAST does. Pick eFAST when the run budget is the binding constraint and you have measured that $N \times D$ beats $N'(D+2)$ at the accuracy you need. Specifically, it is the wrong choice when:

- **Any model run can fail.** eFAST's design is an ordered sweep read by a Fourier transform. One `NaN` and you have `"raise"` or `"propagate"` and nothing else. `on_invalid="drop"` raises, and says why.
- **You need a confidence interval.** There is nothing to resample inside one search curve. An eFAST interval needs replicated designs at different random phases, which is a change to `sample()`, not a keyword on `analyze()`.
- **You need $S_2$.** It cannot produce them at all.
- **Your parameters are correlated or categorical.** eFAST refuses both.

### Reference

Saltelli, A., Tarantola, S. & Chan, K.P.-S. (1999). A quantitative model-independent method for global sensitivity analysis of model output. *Technometrics*, 41(1), 39-56.

## DGSM (derivative-based global sensitivity measures)

DGSM uses exact gradients from automatic differentiation to compute bounds on the total Sobol index $S_T$. For a **scalar output** it is the cheapest quantitative method when your model is JAX-differentiable: one reverse-mode Jacobian costs about 3 model evaluations regardless of $D$, against $D+2$ for the Saltelli design.

That advantage is scalar-only. Reverse mode costs one pass per output slice, so a model with $T \times K$ output slices costs $T K$ passes. jaxgsa therefore picks the mode from the shapes: `jax.jacfwd` when $T K > D$, `jax.jacrev` otherwise. A Jacobian costs about $\min(D,\, T K)$ evaluations, and the saving against Saltelli's $D+2$ disappears once your output is a long time series. There is no keyword for
the mode: the two shape numbers already decide it, so exporting the choice
would only invite a wrong answer.

Pick it as a fast screening or sanity-check step before committing to a full Sobol' analysis. Use Morris (below) instead if your model is a black box. DGSM has no sampler of its own; it is a given-data method that happens to need a JAX-differentiable `fn` too. A plain Monte Carlo sample of $N$ points, from `jaxgsa.sampling.monte_carlo`, is the usual choice, but any $(X, Y)$-and-Jacobian data works.

### The DGSM moments

For a model $f(\mathbf{X})$ with $D$ parameters, DGSM computes two statistics for each parameter $i$. The first is the mean squared derivative, which is the importance measure:

$$
\nu_i = \mathbb{E}\left[\left(\frac{\partial f}{\partial X_i}\right)^2\right]
$$

The second is the mean derivative:

$$
\sigma_i = \mathbb{E}\left[\frac{\partial f}{\partial X_i}\right]
$$

These moments are estimated from $N$ i.i.d. Monte Carlo samples. For a scalar output, one reverse-mode pass returns the whole $D$-vector of derivatives, so DGSM stays cheap however many parameters you add. That is the case it is built for.

### Bounds on the total Sobol index

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

When the upper and lower bounds are close, DGSM gives a tight bracket on $S_T$ without the cost of a full Sobol analysis. They are often not close. Measure the width before you rely on the bracket; the next section shows what a useless one looks like.

### Poincaré constants by distribution

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
3. jaxgsa differentiates it, choosing forward or reverse mode from the shapes, and derives the moments and bounds.
4. The returned `DGSMResult` contains `nu`, `sigma`, `upper_bound`, `lower_bound`, and `var_y`.

`fn` takes **one sample row**, shape `(D,)`, and returns a scalar, a `(K,)` vector, or a `(T, K)` array. jaxgsa vmaps it for you. A batch model that expects `(N, D)` must be wrapped:

```python
result = jaxgsa.dgsm.analyze(PROBLEM, lambda x: model(x[None, :])[0], X, verbose=False)
```

Passing a batch model unwrapped raises a `ValueError` that spells out this exact fix, so you will not be left guessing.

Alternatively, if the Jacobian has been computed externally (for a non-JAX model, say), you can pass pre-computed `Y=` and `dfdx=` arrays directly and skip `fn` entirely.

### The bounds can be far too loose to rank with

DGSM on Ishigami, 1024 Monte Carlo points:

```python
X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=1024, seed=0))
d = jaxgsa.dgsm.analyze(PROBLEM, lambda x: evaluate(x[None, :])[0], X, verbose=False)
print("lower", d.lower_bound)
print("upper", d.upper_bound)
```

```
JaxgsaWarning: jaxgsa.dgsm: lower_bound is a valid lower bound on the total
Sobol index only for untruncated Gaussian marginals (Kucherenko & Song 2016,
Theorem 6, Section 4.1, eq. 31). These marginals do not meet that condition:
x1, x2, x3. For them lower_bound is an estimate, not a bound: it is exact
when the response is linear in that input, and it can exceed the true total
index when the response is curved. Confirm anything that rests on it with
jaxgsa.sobol. upper_bound is unaffected: the Poincare bound holds for every
supported marginal.

lower [0.00099489 0.0073687  0.00224933]
upper [2.3450265 7.3845625 3.10674  ]
```

The true $S_T$ is $[0.5576, 0.4424, 0.2437]$. Both bounds hold, and neither is worth anything.

The lower bound is near zero for all three parameters, and raising $N$ to 131072 pushes it to $10^{-4}$, not up. It is built from $\mathbb{E}[\partial f/\partial X_i]^2$, and every Ishigami term is symmetric about the middle of its range, so the mean derivative cancels to zero. Any model that is not monotone in a parameter will do the same. Treat the Kucherenko–Song lower bound as informative only for monotone responses.

The upper bound is above 1 on every parameter, which tells you nothing, since $S_T \le 1$ by definition. `analyze` warns when that happens on an output slice, because an array of plausible positive numbers reads like a ranking whether or not it constrains anything. Worse, it ranks the parameters $x_2 > x_3 > x_1$ while the true $S_T$ ranks them $x_1 > x_2 > x_3$. All three marginals are uniform on $[-\pi, \pi]$, so they share the Poincaré constant $C = (2\pi)^2/\pi^2 = 4$ and the ranking is the ranking of $\nu$ alone. At $N = 1024$, $\nu = [7.75, 24.41, 10.27]$; the ranking does not settle down with more samples, either: at $N = 131072$, $\nu = [7.61, 24.52, 11.00]$. $\nu$ is a mean **squared** derivative, so a steep slope over a small part of the range dominates it. $x_3$'s derivative is $0.4 x_3^3 \sin x_1$, which reaches 12 at the ends of the range and is near zero over most of it. That gives $x_3$ a large $\nu$ and a small variance share. Raising $N$ does not fix it: at 131072 points the bounds settle at $[2.19, 7.06, 3.17]$ and the ranking is unchanged.

So DGSM is a fast way to find parameters that do nothing at all. It is not a reliable ranking of the ones that do.

### Index summary

| Field | Meaning |
|-------|---------|
| $\nu_i$ | Mean squared derivative: $\mathbb{E}[(\partial f / \partial X_i)^2]$. Higher values indicate stronger influence. |
| $\sigma_i$ | Mean derivative: $\mathbb{E}[\partial f / \partial X_i]$. Non-zero when the effect is non-symmetric. |
| Upper bound | Poincaré bound: $C_i \cdot \nu_i / \mathrm{Var}(Y)$. Conservative upper bound on $S_T$. |
| Lower bound | Kucherenko–Song bound: $\mathrm{Var}(X_i) \cdot \sigma_i^2 / \mathrm{Var}(Y)$. A proven lower bound on $S_T$ only for an untruncated Gaussian marginal; otherwise an estimate that can exceed the true $S_T$. |

### When to use it

- You have a JAX-differentiable model, a scalar or short output, and want fast screening without the cost of Saltelli or eFAST sampling
- You want to find the parameters with no effect at all, cheaply
- You are screening many parameters where one Jacobian beats $D+2$ model runs
- You want a quick sanity check before running a full Sobol analysis

### When it is the wrong choice

- **You need a ranking you can act on.** See above. The Poincaré bound is a bound, not an index, and on Ishigami it ranks the parameters wrong.
- **Your output is a long time series.** At $T K \gg D$ the Jacobian costs $D$ forward passes and the cost argument for DGSM evaporates. Run Sobol'.
- **Your model is not monotone in the parameter.** The lower bound goes to zero and only the upper bound is left, which by itself brackets $[0, \text{something}]$.
- **Your model is not JAX-differentiable.** Use [Morris](#morris-elementary-effects-screening), which is the same idea at a finite step size and needs no gradient.
- **Your parameters are correlated or categorical.** DGSM refuses both. A derivative with respect to an unordered level code is meaningless.

### References

- Sobol', I.M. & Kucherenko, S. (2009). Derivative based global sensitivity measures and their link with global sensitivity indices. *Mathematics and Computers in Simulation*, 79(10), 3009-3017.
- Kucherenko, S. & Song, S. (2016). Derivative-based global sensitivity measures and their link with Sobol' sensitivity indices. In *Monte Carlo and Quasi-Monte Carlo Methods* (MCQMC 2014), Springer Proceedings in Mathematics & Statistics 163, 455-469. doi:10.1007/978-3-319-33507-0_23.
- Lamboni, M., Iooss, B., Popelin, A.-L. & Gamboa, F. (2013). Derivative-based global sensitivity measures: General links with Sobol' indices and numerical tests. *Mathematics and Computers in Simulation*, 87, 45-54.

## Morris (elementary effects screening)

Morris is a global screening method. With only $r(D+1)$ model evaluations, where $r$ is typically 10–50 trajectories, it ranks parameters and flags which ones are negligible. Technically it is a globalized one-at-a-time (OAT) design: it measures coarse finite-difference effects of each parameter at many locations spread across the parameter domain, then summarises them into robust importance measures.

Pick it as a triage step for expensive black-box models. Fix the parameters Morris rules out, then spend your remaining budget on an exact method like Sobol' for the survivors. Morris needs its own design, which jaxgsa generates.

### How it works

The design consists of $r$ trajectories, each a path of $D + 1$ points where consecutive points differ in exactly one coordinate. Each trajectory contributes one elementary effect per parameter, a finite-difference slope:

$$
EE_i = \frac{f(\mathbf{x} + \Delta \mathbf{e}_i) - f(\mathbf{x})}{\Delta}
$$

where $\mathbf{e}_i$ is the unit vector along parameter $i$ and $\Delta$ is the step in unit-cube coordinates. jaxgsa implements two designs:

- **Trajectory design** (Morris 1991, default): each trajectory is a random walk on a $p$-level grid (`num_levels`, default 4) with the canonical step $\Delta = p / (2(p-1))$, visiting parameters in a random order.
- **Radial design** (Campolongo et al. 2011, `method="radial"`): star designs around scrambled-Sobol' base points, where each elementary effect compares a one-coordinate swap against the shared base point with a per-step $\Delta_i = b_i - a_i$.

Both uniform and Gaussian marginals are supported. The design touches the unit-cube boundaries, and an unbounded inverse CDF maps 0 and 1 to infinity. Each open side of a Gaussian marginal is therefore pulled in by $q$ (`truncation_quantile`, default $q = 10^{-4}$, the 0.01%–99.99% quantile range) before the inverse-CDF transform. A side the problem already bounds with an explicit `low` or `high` is left exactly where the user put it, so a two-sided truncated Gaussian is sampled as declared. Uniform marginals are untouched, and deduplication and prefix-nesting are unaffected. The elementary-effect divisor is the step the design really takes, so this rescaling does not bias $\mu^*$.

On an unbounded marginal there is no $q \to 0$ limit for $\mu^*$. The design always includes unit levels 0 and 1 exactly, so a smaller $q$ always reaches further into the tail and the effects grow with it. $\mu^*$ magnitudes on an unbounded marginal are therefore scale-dependent by construction, and only rankings are comparable across truncation settings. If you want one bounded parameter model that every method shares, declare it once:

```python
problem = jaxgsa.Problem.from_dict(
    {"x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0}},
    truncate_gaussians=1e-4,   # fills low/high at this marginal's own quantiles
)
```

The $r$ elementary effects per parameter are reduced to three screening measures:

- $\mu_i$ is the mean elementary effect. Sign cancellation can mask non-monotonic influence, which is why $\mu$ alone is unreliable.
- $\mu^*_i$ is the mean absolute elementary effect (Campolongo et al. 2007). This is the headline importance measure. Read it as "how strongly does the output respond, on average, when this parameter moves?". It is a good proxy for the total-order index $S_T$ ranking.
- $\sigma_i$ is the standard deviation of the elementary effects (ddof=1). A large $\sigma_i$ relative to $\mu^*_i$ means the effect of parameter $i$ changes across the domain, indicating nonlinearity or interactions with other parameters.

The canonical output is the $\mu^*$–$\sigma$ scatter plot. Parameters near the origin are negligible. Parameters far along the $\mu^*$ axis are influential. Parameters high above the diagonal act mainly through nonlinearity or interactions.

Morris is closely related to DGSM. As $\Delta \to 0$, $\mu^*_i \to \mathbb{E}|\partial f / \partial x_i|$, so Morris is the black-box, macro-step analog of jaxgsa's DGSM. Use DGSM when the model is JAX-differentiable, and Morris when it is not.

### How to use it

1. `jaxgsa.morris.sample()` builds the trajectories, removes exact duplicate rows, and returns only the unique rows. Grid designs collide often in low dimensions, so this saves real model evaluations, just like Saltelli sampling.
2. You evaluate your model on `sampling_result.samples`.
3. `jaxgsa.morris.analyze()` reconstructs the expanded design internally, applies the `on_invalid` policy at trajectory granularity (see [Failed model runs](#failed-model-runs)), and reduces one elementary effect per trajectory and parameter to $\mu$, $\mu^*$, and $\sigma$. Pass `n_bootstrap > 0` with a `key` for bootstrap confidence intervals over trajectories. Use `resample_chunk_size` to bound the peak memory of that resampling; Morris spells it that way, not `slice_chunk_size`.

Elementary effects are computed in unit-cube coordinates, so $\mu^*$ is directly comparable across parameters regardless of their physical ranges. `MorrisResult.to_physical_units()` rescales to derivative-scale values in the problem's native units. That rescaling covers uniform-marginal problems only: for Gaussian marginals the inverse-CDF transform is nonlinear, so there is no single linear rescaling to fall back to, and `to_physical_units()` raises `ValueError` rather than return a number on the wrong scale. `MorrisSamples.downsample()` prefix-slices to fewer trajectories without re-simulation, mirroring `SobolSamples.downsample()`.

Compared to SALib's Morris implementation, jaxgsa adds unique-row deduplication, vectorized multi-output and time-series analysis (SALib's Morris is scalar-only, so its own `num_resamples` bootstrap does not extend to that case), the radial design, and prefix-nested downsampling.

### Free screening from a Sobol' design

A Saltelli design is already a radial Morris design. Within each base point it holds a row $A$ and $D$ rows $A_B^{(j)}$ that differ from $A$ in exactly one parameter, which is precisely what an elementary effect needs. This is not a coincidence: Campolongo et al. (2011) build the radial design from a $2D$-dimensional Sobol' sequence split into halves $(a, b)$, and `jaxgsa.sobol.sample` draws the same sequence the same way.

Write the step as $\Delta_j = B_j - A_j$, so that $EE_j = \left(f(A_B^{(j)}) - f(A)\right) / \Delta_j$. Substituting $f(A_B^{(j)}) - f(A) = \Delta_j \cdot EE_j$ into the estimators jaxgsa uses for Sobol' indices gives

$$S_{T_j} = \frac{\mathbb{E}\left[(f(A) - f(A_B^{(j)}))^2\right]}{2\,\mathrm{Var}(Y)} = \frac{\mathbb{E}\left[\Delta_j^2\, EE_j^2\right]}{2\,\mathrm{Var}(Y)} \quad \text{(Jansen 1999)}$$

$$S_{1_j} = \frac{\mathbb{E}\left[f(B)\left(f(A_B^{(j)}) - f(A)\right)\right]}{\mathrm{Var}(Y)} = \frac{\mathbb{E}\left[f(B)\, \Delta_j\, EE_j\right]}{\mathrm{Var}(Y)} \quad \text{(Saltelli 2010)}$$

against Morris's $\mu^*_j = \mathbb{E}|EE_j|$. Same increments, different weighting: Morris divides by $\Delta$ and takes a first absolute moment, Jansen keeps $\Delta$ and takes a second moment. Campolongo et al. (2011) call this the unified approach, one design serving both screening and quantitative indices. The chain closes at DGSM: as $\Delta_j \to 0$ the effect tends to $\partial f / \partial x_j$, so $\mathbb{E}[EE_j^2] \to \nu_j$, the quantity that bounds $S_{T_j}$ through the Poincaré inequality.

`SobolSamples.to_morris()` performs this reinterpretation, so screening measures cost no extra model evaluations:

```python
samples = jaxgsa.sobol.sample(PROBLEM, 8192, seed=0, verbose=False)
Y = evaluate(samples.samples)

sobol_result = jaxgsa.sobol.analyze(samples, Y, verbose=False)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y, verbose=False)

print(sobol_result.ST)
print(morris_result.mu_star, morris_result.sigma)
```

```
jaxgsa.sobol.SobolSamples.to_morris: D=3, mode=second-order, base_n=1024, blocks=1024, effects=3072, reusing n_runs=8192 existing evaluations (0 new model runs)
[0.55598414 0.44165453 0.24129711]
[ 8.70476  15.02531   6.620432] [12.5912485 20.024467  11.469142 ]
```

`to_morris()` prints that line because it is worth knowing what it reused; pass `verbose=False` to silence it. The 3072 elementary effects came out of model runs you had already paid for.

You get one radial block per base point, so `n_trajectories == base_n` for both design variants. A second-order design also contains a block based at $B$ ($B$ with its $B_A^{(j)}$ rows), and it is tempting to harvest as a free doubling. `to_morris()` does not use it. The reason is not that it is a duplicate. That equality holds only for additive contributions: whenever parameter $j$'s contribution is additive,

$$\frac{f(B_A^{(j)}) - f(B)}{A_j - B_j} = \frac{g_j(A_j) - g_j(B_j)}{A_j - B_j} = \frac{f(A_B^{(j)}) - f(A)}{B_j - A_j}$$

but in general it does not. Measured on Ishigami the paired effects correlate 0.50 / 1.00 / −0.06, so only $x_2$, from the purely additive $7\sin^2(x_2)$ term, is a genuine duplicate. The real reason is that pooling buys nothing: over 150 seeds at `base_n=128` the pooled estimator's variance ratio against the $A$-only estimator is $[1.07, 1.00, 1.59]$, so it reduces no variance and is worse on $x_3$. Pooling would also need a cluster bootstrap over base points to keep confidence intervals honest, because the two blocks in a base point share their sampling unit. That is real machinery for no gain.

Take care over which estimand you get. The derived design is a radial design, so it estimates $\mathbb{E}\left|f(A \text{ with } B_j) - f(A)\right| / |B_j - A_j|$, in which the step varies from block to block. That is not the classical Morris quantity with one fixed grid step $\Delta$. `jaxgsa.morris.sample` defaults to `method="trajectory"`, so compare against `morris.sample(..., method="radial")`, never against the default. On Ishigami at $r = 8192$ the derived $\mu^*$ is $[8.68, 15.01, 6.62]$ against $[8.69, 15.02, 6.64]$ for the native radial design, but $[7.59, 7.88, 6.39]$ for the native trajectory design, a factor 1.9 on $x_2$, and 2.5 on its $\sigma$.

Three further caveats:

- The derived measures reuse the same model outputs as the Sobol' indices, so agreement between $\mu^*$ and $S_T$ is not an independent check of either. They may also legitimately rank parameters differently, because $\mu^*$ is a mean absolute derivative, not a variance share.
- Saltelli takes $A$ and $B$ from the same Sobol' row, whereas `jaxgsa.morris.sample`'s own radial design offsets them by four draws precisely to keep $\Delta$ away from zero, and it raises `ValueError` outright if a step still comes out numerically zero. `to_morris()`, which reinterprets an *existing* Saltelli design instead of building a fresh one, cannot raise its way out of a bad block without discarding model runs you already paid for, so there it drops the block and warns. At the default `scramble=True` this is a non-issue: 0 of 65536 blocks were dropped across 8 seeds at $D = 3$. With `scramble=False` the drop rate is real but falls off with `base_n`: 21.9% at `base_n=64`, 9.4% at 256, 2.3% at 1024, 1.2% at 4096. The survivors are a biased subsequence, giving $\mu^* = [8.34, 14.88, 5.55]$ at `base_n=64` against $[8.68, 15.01, 6.62]$ scrambled, so $x_3$ reads 16% low. Keep `scramble=True`.
- For unbounded Gaussian marginals, $\mu^*$ has no fixed scale. How far a design reaches into the tail sets the magnitude, and the Saltelli design (bounded only by the library's own $\pm 7.03\sigma$ support clip) and `morris.sample` reach different distances. Only rankings are comparable. Bound the marginals once if magnitudes must match:

  ```python
  problem = jaxgsa.Problem.from_dict(params, truncate_gaussians=1e-4)
  ```

  Both sides are then genuinely bounded, `morris.sample` does not squash them again, and the derived and native radial measures agree. The measured ratios are 0.999 (linear), 0.997 ($x^2$), 0.988 ($x^4$), 0.987 ($\exp(x^2/3)$), each within its own seed-to-seed spread. `to_morris()` warns when unbounded Gaussians are present.

The reverse derivation is impossible: a radial Morris design never evaluates the $B$ rows, so $S_1$ and $S_T$ cannot be recovered from it.

### Index summary

| Measure | Meaning |
|-------|---------|
| $\mu(i)$ | Mean elementary effect. Sign cancellation can hide non-monotonic influence. |
| $\mu^*(i)$ | Mean absolute elementary effect. Headline importance measure; proxy for the $S_T$ ranking. |
| $\sigma(i)$ | Standard deviation of the elementary effects. Large $\sigma / \mu^*$ indicates nonlinearity or interactions. |

### When to use it

- You want a cheap screening pass before committing to a full Sobol' run
- Your model is a black box (not JAX-differentiable; otherwise consider DGSM)
- You have many parameters and a tight evaluation budget. The cost is $r(D+1)$ with $r$ typically 10-50
- You only need a ranking and an interaction flag, not exact variance fractions

### When it is the wrong choice

- **You want to trust the ranking of the parameters that matter.** $\mu^*$ is a mean absolute slope, and it is a proxy for the $S_T$ ranking, not a substitute. On Ishigami it swaps the top two: $\mu^* = [8.70, 15.03, 6.62]$ ranks $x_2$ first, while $S_T = [0.556, 0.442, 0.241]$ ranks $x_1$ first. Use Morris to decide what to **drop**, which is what it is good at, and let Sobol' rank what is left.
- **You want a number that means something.** $\mu^*$ is on the scale of $\partial f / \partial x$ in unit-cube coordinates. It is not a variance fraction and it does not sum to anything. `to_physical_units()` puts it on a derivative scale, and only for uniform marginals; it raises `ValueError` on a Gaussian one rather than return something on the wrong scale.
- **You have unbounded Gaussian marginals.** $\mu^*$ has no fixed magnitude then: how far the design reaches into the tail sets it, and `truncation_quantile` sets that. Only rankings survive a change of setting. Declare `truncate_gaussians=` once on the `Problem` if the magnitudes have to mean anything.
- **You already have a Saltelli design.** Then Morris is free rather than cheap, via `to_morris()` above, and there is no reason to run a separate design.
- **Your model is JAX-differentiable and the output is scalar.** [DGSM](#dgsm-derivative-based-global-sensitivity-measures) is the same measure at $\Delta \to 0$ and costs less. Take its warnings above with it.
- **Your parameters are correlated or categorical.** Morris refuses both. A one-at-a-time step off a correlation ridge lands somewhere the model never sees.

### References

- Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161-174.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large models. *Environmental Modelling & Software*, 22(10), 1509-1518.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2011). From screening to quantitative sensitivity analysis. A unified approach. *Computer Physics Communications*, 182(4), 978-988.
- Jansen, M.J.W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1-2), 35-43.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*, ch. 3. Wiley.

## HSIC (Hilbert–Schmidt Independence Criterion)

HSIC measures the statistical dependence between each parameter and the output. It captures any dependence, including nonlinear, non-monotone, and heteroscedastic effects that variance-based indices can underweight. It works in a reproducing kernel Hilbert space (RKHS), mapping parameters and outputs through Gaussian RBF kernels.

Pick it when you suspect your model's behaviour is not well summarised by variance, when your parameters may be correlated, or when you want statistical significance tests attached to the indices. Like HDMR, it works from data you already have: any set of $(X, Y)$ pairs, with no independence assumption on the parameters and no sampling design.

### The HSIC dependence measure

Each parameter $X_i$ and the output $Y$ are passed through a characteristic kernel, a Gaussian RBF whose bandwidth is set automatically by the median heuristic (the median pairwise distance between sample points). Writing $\mathbf{K}$ and $\mathbf{L}$ for the two $N \times N$ kernel matrices, jaxgsa uses the biased V-statistic estimator

$$
\widehat{\mathrm{HSIC}}(X_i, Y) = \frac{1}{N^2}\,\mathrm{tr}(\mathbf{K}\mathbf{H}\mathbf{L}\mathbf{H}), \qquad \mathbf{H} = \mathbf{I} - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top
$$

where $\mathbf{H}$ is the centering matrix. For characteristic kernels, $\mathrm{HSIC}(X_i, Y) = 0$ if and only if $X_i$ and $Y$ are independent, so a larger value signals stronger dependence.

### First-order and total indices

jaxgsa reports two normalised indices per parameter.

R2-HSIC is the first-order index: the normalised dependence between parameter $i$ and the output, in $[0, 1]$. Read it as a kernel analogue of a squared correlation coefficient (centred kernel alignment):

$$
R^2_{\mathrm{HSIC}, i} = \frac{\widehat{\mathrm{HSIC}}(X_i, Y)}{\sqrt{\widehat{\mathrm{HSIC}}(X_i, X_i)\,\widehat{\mathrm{HSIC}}(Y, Y)}}
$$

Total HSIC is the analogue of a total-order index, capturing dependence carried through interactions with the other parameters:

$$
T_i = 1 - \frac{\mathrm{HSIC}(X_{-i}, Y)}{\mathrm{HSIC}(X, Y)}
$$

where $X_{-i}$ is every input except $i$. Dropping input $i$ can only lose dependence, never add it, so $\mathrm{HSIC}(X_{-i}, Y) \le \mathrm{HSIC}(X, Y)$ and $T_i \in [0, 1]$: a $T_i$ near 0 means input $i$ carries no dependence the others do not already carry, and near 1 means removing it collapses the measured dependence almost entirely. Computing it needs one further trick: a naive product kernel over $X_{-i}$ only captures the highest-order interaction, which would silently miss a purely additive effect. jaxgsa instead builds $\mathrm{HSIC}(X, Y)$ from augmented product kernels $k^*_d = 1 + k_{c,d}$ (Larsen & Alexanderian, 2026), where $k_{c,d}$ is the centred kernel for parameter $d$; the added constant term makes the product capture every interaction order, not only the highest, so $T_i$ comes out correct even for a model with no interactions at all.

Unlike Sobol indices, R2-HSIC values are individual dependence measures and do not sum to 1.

### Permutation p-values

HSIC is a dependence measure rather than a variance fraction, so jaxgsa attaches a permutation test to each first-order index. The output labels are randomly shuffled `n_perms` times to build a null distribution of HSIC values. The p-value uses the Phipson–Smyth correction $(c + 1)/(M + 1)$, where $M$ is the number of permutations (`n_perms`) and $c$ counts permuted HSIC values at least as large as the observed one. A small p-value (< 0.05) indicates a statistically significant dependence between the parameter and the output.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples. Any sampling strategy works, since no structured design is required.
2. You evaluate your model on the samples.
3. `jaxgsa.hsic.analyze()` transforms each parameter to $[0, 1]$ via its marginal CDF and builds the input kernel matrices with the median heuristic eagerly, then maps the indices and permutation p-values over output columns in one JIT-compiled pass. `key` is required; `hsic.analyze(problem, X, Y)` without one raises `ValueError`, because the permutation test always runs and there is no `n_bootstrap=0` equivalent that skips it.

HSIC is $O(N^2)$ in time and memory because it forms $N \times N$ kernel matrices, about $2D + 1$ of them resident at once. No option bounds this. `hsic.analyze` has no `batch_size` and no `slice_chunk_size`, because there is no axis to chunk along: the kernel matrices are the computation. Reduce $N$ if memory is the limit, or screen with a cheaper method first. The indices do not depend on the output units, but outputs of extreme magnitude can overflow float32 in the squared distances; rescale by hand with `(Y - Y.mean(0)) / Y.std(0)`, which changes nothing else.

Turn on float64 before you run HSIC. The V-statistic cancels three large sums against each other, so float32 leaves about three or four correct digits, and the index changes with the order of the sample rows. `analyze` warns about this. Small indices and close rankings are not reliable without it.

```python
import jax
jax.config.update("jax_enable_x64", True)  # before the analysis
```

### The bandwidth is a real choice

`bandwidth` (default `1.0`) multiplies the median-heuristic length scale. It is not a tuning detail. On Ishigami with 2000 samples in float64, sweeping it changes which parameter comes first:

| `bandwidth` | R2-HSIC |
|---|---|
| 0.25 | `[0.058, 0.111, 0.025]` |
| 0.5 | `[0.085, 0.070, 0.028]` |
| 1.0 | `[0.135, 0.008, 0.025]` |
| 2.0 | `[0.177, 0.002, 0.009]` |

At 0.25 the ranking is $x_2 > x_1 > x_3$, which agrees with $S_1 = [0.314, 0.442, 0]$. At the default 1.0 it is $x_1 > x_3 > x_2$, and $x_2$ has dropped to 0.008 despite owning 44% of the output variance. A wide kernel smooths $7\sin^2 x_2$, which oscillates twice across the range, into a near-constant, and the dependence disappears from the estimator.

So sweep `bandwidth` before you report an HSIC ranking, and say which value you used. A single HSIC number without its bandwidth is not reproducible. The result carries `bandwidth` and `n_perms` for that reason, and `to_dataset()` writes both into the dataset attributes.

### Index summary

| Index | Meaning |
|-------|---------|
| $R^2_{\mathrm{HSIC}}(i)$ | Normalised first-order kernel dependence between parameter $i$ and the output, in $[0, 1]$. |
| Total HSIC $(i)$ | Total dependence of parameter $i$ including interactions, via augmented complement product kernels. |
| p-value $(i)$ | Permutation p-value for the first-order dependence (Phipson–Smyth corrected). |

### When to use it

- You want a measure that captures any dependence, nonlinear, non-monotone or heteroscedastic, and not only variance contributions
- Your parameters may be correlated (HSIC makes no independence assumption)
- You have existing $(X, Y)$ pairs and want indices without additional model runs
- You want statistical significance testing via permutation p-values

### When it is the wrong choice

- **You want a number to report.** R2-HSIC has no units, does not sum to 1, and moves with the bandwidth. What it answers well is "is this parameter doing anything at all", via the p-value. For a magnitude, use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) or [Borgonovo delta](#borgonovo-delta-density-based-sensitivity), which are both on a fixed $[0, 1]$ scale.
- **$N$ is above about 20000.** The kernel matrices are $N \times N$ and there are $2D+1$ of them. At $N = 20000$ and $D = 5$ that is 11 matrices of 3.2 GB each in float64, so 35 GB. Nothing chunks it.
- **You want a confidence interval.** There is none, deliberately. A row bootstrap repeats rows onto the kernel diagonal where the kernel is exactly 1, which biases the resampled index upward by construction. The permutation p-values are the uncertainty statement.
- **Any of your parameters is categorical.** HSIC refuses: the RBF kernel would read the level codes as distances.
- **You want interaction attribution.** The Total HSIC minus R2-HSIC gap says interactions exist, not which pairs. No $S_2$.

### References

- Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. In *Algorithmic Learning Theory* (ALT 2005), LNCS 3734, 63-77. This is the source of the `tr(KHLH)/n^2` estimator jaxgsa implements.
- Da Veiga, S. (2015). Global sensitivity analysis with dependence measures. *Journal of Statistical Computation and Simulation*, 85(7), 1283-1305. doi:10.1080/00949655.2014.945932.
- Larsen, K. & Alexanderian, A. (2026). A new kernel-based approach for the global sensitivity analysis of models with correlated inputs. *arXiv preprint* arXiv:2603.00849. Definition 9 gives $k^* = 1 + k_c$; Eq. 32 gives the total index $T_A = 1 - \mathrm{HSIC}(X_{\sim A}, Y)/\mathrm{HSIC}(X, Y)$ used above.

## PAWN (CDF-based sensitivity)

PAWN asks a different question from the variance-based methods. Not "how much variance does this parameter explain?", but "how much does the entire output distribution shift when this parameter is held fixed?". It compares the unconditional output CDF against conditional CDFs obtained by fixing each parameter within a bin, using the Kolmogorov–Smirnov (KS) distance as the measure of separation (Pianosi & Wagener, 2015).

Pick it when you care about tails, skewness, or other distributional features that variance misses. Like HSIC and HDMR, it works from data you already have: any $(X, Y)$ pairs, with no independence assumption on the parameters and no sampling design.

### The KS distance

For parameter $i$, its values are first mapped through their own marginal CDF onto $[0, 1]$, then that image is partitioned into `n_bins` equal-width bins. Because the mapping is the marginal's own CDF, the bins are equal-probability on the parameter's original scale, whatever its marginal shape. Within each bin $b$, PAWN forms the conditional output CDF $F_{Y \mid X_i \in b}$ from the samples whose $i$-th parameter falls in that bin, and compares it with the unconditional CDF $F_Y$ (built from all samples) via the Kolmogorov–Smirnov statistic, the largest absolute gap between the two CDFs:

$$
\mathrm{KS}_{i,b} = \sup_{y}\left| F_Y(y) - F_{Y \mid X_i \in b}(y) \right|
$$

A large KS value in a bin means fixing $X_i$ there substantially changes the output distribution. A value near zero means the output is insensitive to that parameter over that region.

### Aggregating across bins

Each parameter yields one KS value per bin. The PAWN index reduces these to a single number per parameter using one of three statistics:

- **median** (default). Robust to a single anomalous bin.
- **max**. The worst-case shift across the parameter range.
- **mean**. The average shift.

The PAWN index is built on CDFs rather than moments, so it is moment-independent and invariant under monotone transformations of the output. It captures tail and skewness changes that variance-based indices miss.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples (Monte Carlo, Latin Hypercube, or Sobol sequences all work; no structured design required).
2. You evaluate your model on the samples.
3. `jaxgsa.pawn.analyze()` maps each parameter to $[0, 1]$, assigns samples to bins, and computes the per-bin KS distances and their aggregate in a single JIT-compiled pass. Pass `n_bootstrap > 0` for bootstrap confidence intervals.

The number of bins (`n_bins`, default 10) trades conditioning resolution against sample density per bin. With very few samples per bin the KS statistic becomes noisy, so increase $N$ or decrease `n_bins`.

`result.n_valid_bins` tells you whether that happened. It counts, per parameter, the bins that held at least 2 samples. Bins below that are dropped, and the median, max and mean run over what is left. When a parameter keeps fewer than half its bins, `analyze` warns. Because the bins are equal-probability on the marginal's own CDF, a skewed marginal does not by itself starve a tail bin; `jaxgsa` has no built-in lognormal marginal, and every continuous marginal it does support (uniform, Gaussian, truncated Gaussian) gets the same equal-probability treatment. What empties a bin is a small `N`, a large `n_bins`, or samples that land outside the declared marginal (dropped with a `-1` sentinel, the same as a `NaN`).

```python
result = jaxgsa.pawn.analyze(PROBLEM, X, Y, verbose=False)
print(result.pawn, result.n_valid_bins)
```

```
[0.2484047  0.402167   0.08681974] [10 10 10]
```

All 10 bins survived for all three Ishigami parameters, so those indices stand on the full sample.

Categorical parameters are supported. A categorical parameter needs no binning: its level code already names the conditioning class, so PAWN uses one bin per level and `n_bins` does not apply to it. Bins with too few samples yield `NaN`, and the median, max, and mean over bins all drop them. The index is therefore unchanged by the order of the level codes. Relabel the levels and you get the same number.

### Index summary

| Index | Meaning |
|-------|---------|
| PAWN $(i)$ | Aggregated (median / max / mean) KS distance between the unconditional and conditional output CDFs for parameter $i$, in $[0, 1]$. Higher means stronger influence on the output distribution. |

### When to use it

- You care about distributional changes beyond variance, such as tail behaviour or skewness shifts
- You want a moment-independent index, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy
- Your parameters may be correlated (no independence assumption or structured design)
- Some of your parameters are categorical, or your output is discrete. PAWN needs neither an ordering on the parameters nor a density on the output

### When it is the wrong choice

- **You want to compare parameters on a meaningful scale.** The KS distance is bounded in $[0, 1]$, but the aggregate over bins is a summary statistic and not a share of anything. On Ishigami, PAWN gives $[0.248, 0.402, 0.087]$ where $\delta$ gives $[0.211, 0.334, 0.156]$ and OT gives $[0.201, 0.278, 0.098]$. They rank the same but the spacings differ, and none of them is "the" answer.
- **You want interactions.** PAWN conditions on one parameter at a time and stops there. No total-order equivalent, no $S_2$.
- **Your marginals are skewed and $N$ is small.** Equal-width bins go empty in the tail. `n_valid_bins` tells you; a parameter down to 3 or 4 bins has a median over 3 or 4 numbers.
- **The KS statistic is the wrong summary for your question.** It is a supremum, so it reacts to the single largest gap between two CDFs and ignores everything else. If a parameter shifts the whole distribution a little, $\delta$ or the OT index sees more of it. If a parameter moves one part of the range a lot, PAWN is the sharper instrument.

### Reference

Pianosi, F. & Wagener, T. (2015). A simple and efficient method for global sensitivity analysis based on cumulative distribution functions. *Environmental Modelling & Software*, 67, 1-11.

Pianosi, F. & Wagener, T. (2018). Distribution-based sensitivity analysis from a generic input-output sample. *Environmental Modelling & Software*, 108, 197-207. The 2015 paper introduces the index; the estimator implemented here — bins built on a generic sample and each bin's KS distance measured against the whole-sample unconditional CDF — is theirs.

## Borgonovo delta (density-based sensitivity)

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

The plug-in estimate is biased upward at finite $N$. `bias_correct` defaults to `None`, which means "correct if you are bootstrapping anyway": with `n_bootstrap > 0` jaxgsa applies Plischke's bias reduction $2\hat{\delta}_i - \overline{\hat{\delta}_i^{(b)}}$ over the same replicates that give the percentile intervals, and it warns once per process to say so. The warning exists because the reported $\delta$ is then not the plug-in estimate, and a reader comparing against another library needs to know which one they are looking at. Pass `bias_correct=True` to keep the correction and silence the warning, or `bias_correct=False` to keep the intervals and report the uncorrected estimate.

The correction subtracts a bootstrap mean from twice the plug-in estimate, so the reported $\delta$ and its interval bounds can fall marginally below $0$ for weak parameters at small $N$, even though the true index and the plug-in estimate both lie in $[0, 1]$.

The same class partition also yields the given-data first-order Sobol index (variance of the class means over the total variance) at negligible extra cost, so every analysis returns both $\delta$ and $S_1$. Reading them side by side is the point of the method. Ishigami, 4000 samples, 100 bootstrap replicates:

```python
result = jaxgsa.borgonovo.analyze(
    PROBLEM, X, Y, n_bootstrap=100, key=jax.random.key(0), verbose=False
)
print(result.delta)
print(result.S1)
```

```
[0.21102615 0.33395138 0.15578218]
[0.30567423 0.42081362 0.00262259]
```

$x_3$ has $S_1 = 0.003$ and $\delta = 0.156$. Fixing $x_3$ does not move the mean of the output at all, which is why the variance-based first-order index is zero, and it visibly reshapes the output density, which is why $\delta$ is not. That gap is the entire argument for a moment-independent index, and here it is in two lines of output.

The estimator matches `SALib.analyze.delta` on the equal-frequency rank partition, the class-count heuristic, the Silverman KDE factors, and the 100-point output grid. It differs in three ways. The central estimate is computed on the original sample, so it is deterministic given the data, where SALib evaluates it on a random resample. A constant output column yields $\delta = S_1 = 0$ instead of an error. A bootstrap replicate that happens to be constant, which is reachable for rare-event outputs, contributes the point estimate rather than a spurious zero, where SALib raises `LinAlgError`.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` generates plain Monte Carlo samples (any sampling strategy works; no structured design is required).
2. You evaluate your model on the samples.
3. `jaxgsa.borgonovo.analyze()` partitions each parameter into rank classes and computes $\delta$, $S_1$, and their bootstrap intervals in a single JIT-compiled kernel, vmapped over output columns and scanned over bootstrap replicates.

Set `n_bootstrap=0` to skip bias correction and confidence intervals (raw plug-in estimate), or `bias_correct=False` to keep the intervals but report the uncorrected estimate. Peak memory is dominated by the class layout, about `slice_chunk_size * D * N` for continuous parameters (more for an imbalanced categorical one, which pads every level up to the largest). The output grid itself is evaluated in tiles rather than held whole, so it does not multiply into that figure, and `slice_chunk_size` is rarely the knob that saves memory for a large `grid_size`.

::: warning Continuous outputs only
The $\delta$ estimator supports a continuous output distribution only. It compares kernel density estimates on a shared output grid, and a discrete output has atoms that no grid resolves. `borgonovo.analyze` checks the output first and raises `ValueError` when a column takes at most 20 distinct values and each value repeats at least 5 times on average (equivalently, at most 20% of the samples are distinct). Use [optimal transport](#optimal-transport-wasserstein-based-sensitivity) or [PAWN](#pawn-cdf-based-sensitivity) for a discrete output: both compare empirical distributions and need no density. A constant column is exempt, because its exact answer is $\delta = S_1 = 0$. Categorical parameters stay supported. The limit applies to the output only.
:::

$\delta$ is a half L1 distance between densities, so it lies in $[0, 1]$. If the returned estimate leaves that range by more than 0.05, the computation failed and `analyze` raises `ValueError`. The message names the parameter, reports what the kernel did to the offending class, and points at the knob that applies to that case. The value is never clipped, because a clipped value looks plausible and is still wrong. A confidence bound outside the range only warns: the point estimate is the contract, and the interval is a diagnostic.

Two settings control how a near-degenerate conditioning class is treated. `degenerate_tol` says when a class counts as degenerate. `degenerate_bandwidth` says how wide a kernel such a class is given.

`degenerate_bandwidth="auto"`, the default, floors the kernel at `max(0.1 * h_full, grid_step)`, so it never goes below what the output grid can integrate. A float is a fraction of the full-sample bandwidth and is applied exactly.

`analyze` does not refuse a `degenerate_bandwidth` on the setting alone, because the setting alone does not say whether the run works. Two conditions have to hold first. The floor only ever reaches a class the estimator already called degenerate, so on data with no such class the setting changes nothing at any value. And even on a degenerate class, a kernel narrower than one grid step only aliases if a grid point lands on the narrow peak. On one test problem with a genuine point mass, a floor of 0.01 of the full-sample bandwidth, a tenth of one grid step, still returns a $\delta$ inside $[0, 1]$, and moving the same point mass off the grid boundary keeps the answer stable down to $10^{-5}$.

`analyze` therefore checks the returned $\delta$, not the setting. When the estimate does leave $[0, 1]$, the error message reads what the run actually did and names the knob that fixes it. Every message names `grid_size`, because a finer grid always shortens the step. What else it names depends on what happened during the run:

| What happened | The message also names |
|---|---|
| No class was floored | `degenerate_tol`. The floor changed nothing, so the tolerance is what kept it away. |
| A class was floored by an explicit `degenerate_bandwidth` | That floor width, one grid step, and the fraction of the full-sample bandwidth equal to one grid step. |
| A class was floored by the `"auto"` default | Nothing further. The `"auto"` floor is already at least one grid step wide by construction, so `grid_size` is the whole of the advice. |

The value itself is never clipped, for the same reason as above.

Raising `degenerate_tol` does not raise. A higher tolerance calls more classes degenerate, and each of those is then given the floor. When the floor is narrower than a class's own bandwidth, that class gets a *narrower* kernel than it had, which biases $\delta$ for the classes the higher tolerance said to distrust. `analyze` does warn when this happens: the floor-width warning quoted above fires on the run that triggered the floor, whatever set `degenerate_tol` to call that class degenerate. On $Y = x_1 + 0.01 x_2$, `degenerate_tol=0.5` floors a class of `x2` that the default tolerance left alone, warns, and moves `delta[0]` from 0.901 to 0.884. What `analyze` does not do is judge the size of the bias from the setting alone: the warning names the mechanism, not the resulting error, because that depends on the data inside the kernel.

### Index summary

| Index | Meaning |
|-------|---------|
| $\delta(i)$ | Expected L1 distance between the unconditional and conditional output densities for parameter $i$, in $[0, 1]$. Higher means stronger influence on the output distribution; $0$ means no influence at all. |
| $S_1(i)$ | Given-data first-order Sobol index from the same class partition. The variance-based view of the same conditioning, for comparison at no extra cost. |

### When to use it

- You care about influence on the whole output distribution, so tails, skewness and multimodality, not only variance
- You want a moment-independent index with a fixed $[0, 1]$ scale, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy, possibly with correlated parameters
- You use `SALib.analyze.delta` and want a deterministic, JIT-compiled equivalent that also handles multi-output and time-series `Y`

### When it is the wrong choice

- **Your output is discrete.** `analyze` raises. It compares kernel density estimates, and a discrete output has atoms no grid resolves. The check fires when a column takes at most 20 distinct values and each value repeats at least 5 times on average. Use [PAWN](#pawn-cdf-based-sensitivity) or [optimal transport](#optimal-transport-wasserstein-based-sensitivity), which compare empirical distributions and need no density.
- **You need interactions or a total-order index.** $\delta$ conditions on one parameter. The $\delta - S_1$ gap tells you influence exists beyond the first-order variance, and nothing more.
- **You want to separate direct influence from correlation-borne influence.** $\delta$ is correlation-inclusive: a parameter the model never reads scores above zero when it correlates with one the model does read. That is the correct reading of the index, not an error. Use [VKOGA](#vkoga-correlated-input-variance-indices) or [Kucherenko](#kucherenko-dependent-input-sobol-indices) for the split.
- **Your output has a point mass or a hard bound.** The KDE has to be told what to do with a near-degenerate conditioning class; see `degenerate_tol` and `degenerate_bandwidth` below. OT handles the same data with no bandwidth at all.
- **$N$ is small.** The plug-in estimate is biased upward and the bias correction can push weak parameters below zero. Both are visible, neither is comfortable. Below about 500 samples, read the ranking and ignore the magnitudes.

### Reference

- Borgonovo, E. (2007). A new uncertainty importance measure. *Reliability Engineering & System Safety*, 92(6), 771-784.
- Plischke, E., Borgonovo, E. & Smith, C.L. (2013). Global sensitivity measures from given data. *European Journal of Operational Research*, 226(3), 536-550.

## Optimal transport (Wasserstein-based sensitivity)

The optimal-transport index (Borgonovo, Figalli, Plischke & Savaré, 2024) measures how far knowing a parameter moves the whole output distribution. It uses the squared 2-Wasserstein distance, which is the minimal quadratic work needed to transport the unconditional output distribution onto the conditional one:

$$
\iota_i = \frac{\mathbb{E}_{X_i}\!\left[ W_2^2\!\left(P_{Y \mid X_i},\, P_Y\right) \right]}{2\,\mathrm{Var}(Y)}
$$

The denominator is the theoretical maximum of the numerator, so $\iota_i \in [0, 1]$. A value of $0$ means the output distribution never reacts to $X_i$, and $1$ means it is fully determined by it. The defining feature is the exact decomposition of every index into two parts:

- **advective**, the class-averaged squared shift of the conditional mean, which is half the given-data first-order Sobol index up to a finite-sample factor ($2 \cdot \mathrm{advective} \cdot N/(N-1) = S_1$), and
- **diffusive**, the remainder: changes in spread, tails, and shape.

So the OT index subsumes the variance-based first-order view and quantifies what lies beyond it, on one scale. It works from data you already have: any $(X, Y)$ pairs, with no sampling design.

### How it works

1. For each parameter, samples are split into `n_partitions` equal-frequency classes by the parameter's rank (default `min(25, N // 2)`). Rank-based conditioning is distribution-free: uniform, Gaussian, or mixed marginals work unchanged, and monotone parameter transforms change nothing. Correlated parameters are supported, and the index then measures total, correlation-inclusive influence. Categorical parameters instead get one class per level (`n_partitions` does not apply to them), so the index never depends on the arbitrary code order.
2. Per class, $W_2^2$ between the conditional and unconditional output samples is computed. In the default `mode="univariate"` (per output column) this uses the closed form of 1-D optimal transport: both empirical quantile functions evaluated at the $N$ uniform mass points via sorting, no iterative solver. The `"multivariate"` and `"trajectory"` modes treat the output vector as a point cloud and solve entropic transport with a pure-JAX log-domain Sinkhorn solver (regularization `epsilon`, reported cost is the unregularized $\langle P, C\rangle$).
3. Class results are averaged with class-size weights and divided by $2\,\mathrm{Var}(Y)$ (point-cloud modes: $2\,\mathrm{Tr}\,\mathrm{Cov}(Y)$, with per-column standardization on by default so no output dominates through its units).

Entropic and finite-sample bias keep point-cloud-mode indices of irrelevant parameters strictly positive. Pass `dummy=True` (with a `key`) to run a synthetic, provably independent parameter through the same estimator. Its index comes back as `ot_dummy`, the irrelevance floor, and `above_dummy` is `max(ot - ot_dummy, 0)` computed for you.

### The split, on real numbers

Ishigami, 4000 samples:

```python
result = jaxgsa.optimal_transport.analyze(
    PROBLEM, X, Y, dummy=True, key=jax.random.key(0), verbose=False
)
print("ot         ", result.ot)
print("advective  ", result.advective)
print("diffusive  ", result.diffusive)
print("S1         ", result.S1)
print("ot_dummy   ", result.ot_dummy)
print("above_dummy", result.above_dummy)
```

```
ot          [0.20130877 0.27754727 0.09772307]
advective   [0.15357937 0.21982558 0.00371554]
diffusive   [0.04772939 0.0577217  0.09400754]
S1          [0.30723557 0.43976113 0.00743294]
ot_dummy    [0.00946424 0.00946424 0.00946424]
above_dummy [0.19184452 0.26808304 0.08825883]
```

Three things to read off it.

`S1` is $2 \times$ `advective` $\times N/(N-1)$: $2 \times 0.00371554 \times 4000/3999 = 0.00743294$. The $N/(N-1)$ factor is the usual finite-sample correction and is close to 1 for any sample worth analyzing, so `S1` and `2 * advective` agree to within that correction, not bit for bit.

$x_1$ and $x_2$ are mean-shift parameters: 76% and 79% of their index is advective. $x_3$ is the opposite, 96% diffusive. Knowing $x_3$ tells you almost nothing about where the output will land and a lot about how far it will spread. No variance-based index distinguishes those two situations.

`ot_dummy` is a per-parameter array, one permutation floor per column. All three read 0.0095 here because every Ishigami parameter shares the same continuous marginal and class count. The 0.098 for $x_3$ is about ten times its own floor and is real. In `"univariate"` mode the floor is small; in the point-cloud modes it is not, and `dummy=True` stops being optional there.

### How to use it

1. `jaxgsa.sampling.monte_carlo()` or any existing $(X, Y)$ data. No structured design required.
2. `jaxgsa.optimal_transport.analyze()` computes `ot`, `advective`, and `diffusive` per parameter (and per output column in `"univariate"` mode), with an optional row bootstrap for confidence intervals.

Pick the mode by the question: `"univariate"` for per-column indices across `(N,)`/`(N, K)`/`(N, T, K)` outputs, `"multivariate"` for one index per parameter over the flattened joint output, `"trajectory"` for one index per parameter per output over the whole time course. The point-cloud modes solve one entropic transport problem per parameter, per class, per replicate, and per point cloud. For continuous parameters that is `(n_bootstrap + 1) * D * n_partitions` solves, and `dummy=True` adds one more single-replicate pass of `n_partitions` solves. Both figures multiply by the output count `K` in `"trajectory"` mode, which builds one cloud per output; `"multivariate"` mode builds one cloud in total. A categorical parameter costs its own level count instead of `n_partitions`, and adds its own dummy pass. Keep it modest.

### Index summary

| Index | Meaning |
|-------|---------|
| $\iota(i)$ (`ot`) | Normalized expected $W_2^2$ between conditional and unconditional output distributions, in $[0, 1]$. |
| `advective` | Mean-shift component; $2 \cdot \mathrm{advective} \cdot N/(N-1)$ is the given-data first-order Sobol index. |
| `diffusive` | Spread/shape component, `ot - advective`; flags influence invisible to the conditional mean. |
| `S1` | The given-data first-order Sobol index, $2 \times$ `advective` up to the $N/(N-1)$ finite-sample factor, returned for convenience. |
| `ot_dummy` | Per-parameter index of a synthetic independent column, permuted against that parameter's own partition (with `dummy=True`). The irrelevance floor. `None` otherwise. |
| `above_dummy` | `max(ot - ot_dummy, 0)`, the index with the floor subtracted. `None` unless you passed `dummy=True`. |

### Valid under correlated inputs

The OT index is valid under correlated parameters, and jaxgsa certifies it. `optimal_transport.analyze` accepts a problem with a declared `problem.correlation`, because it is exempt from the correlated-input error. The definition $\mathbb{E}_{X_i}[W_2^2(P_{Y|X_i}, P_Y)]$ conditions on one parameter at a time and never requires an independence decomposition. The estimator conditions on rank classes of the observed sample and never reads the declared matrix. The test suite asserts bit-equality between a correlated problem and the same $(X, Y)$ with the correlation stripped.

Read the index as total, correlation-inclusive influence. A parameter the model never uses still gets a clearly non-zero index when it is correlated with one the model does use (tested at $\rho = 0.8$). To separate direct from correlation-borne influence, use [VKOGA](#vkoga-correlated-input-variance-indices) or [Kucherenko](#kucherenko-dependent-input-sobol-indices).

### When to use it

- You want a moment-independent index that still ties exactly to the variance-based world
- You want to distinguish parameters that move the output from parameters that reshape it
- You want one index per parameter for a whole trajectory or multivariate output (`multivariate` / `trajectory` modes)
- Your parameters have mixed marginals or are correlated

If you have $(X, Y)$ data and no strong reason to prefer another method, this is where I would start. It is on a fixed $[0, 1]$ scale, it needs no bandwidth, it accepts correlated and categorical parameters, and it hands you the given-data $S_1$ for free so you can compare against the variance-based world without a second run.

### When it is the wrong choice

- **You need interactions.** OT conditions on one parameter at a time. The diffusive part says influence exists beyond the mean shift; it does not say which parameter it is shared with. No $S_2$, no total order.
- **You need to separate direct from correlation-borne influence.** The index is correlation-inclusive by construction. Use [VKOGA](#vkoga-correlated-input-variance-indices) or [Kucherenko](#kucherenko-dependent-input-sobol-indices).
- **You are in a point-cloud mode without a dummy.** `"multivariate"` and `"trajectory"` solve entropic transport, and the entropic bias keeps irrelevant parameters visibly above zero. Reading those indices without `dummy=True` will make you believe in parameters that do nothing, which is why `analyze` warns when either mode runs without one.
- **You are bootstrapping a point-cloud mode.** The bill is `(n_bootstrap + 1) * D * n_partitions` Sinkhorn solves. At 100 replicates, 10 parameters and 25 partitions that is just over 25000 solves.
- **You want a surrogate too.** OT gives you indices and nothing else. Use [PCE](#pce-polynomial-chaos-expansion), [HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) or [VKOGA](#vkoga-correlated-input-variance-indices).

### Reference

- Borgonovo, E., Figalli, A., Plischke, E. & Savaré, G. (2024). Global sensitivity analysis via optimal transport. *Management Science*, 71(5), 3809-3828 (in print 2025; online-first 2024). doi:10.1287/mnsc.2023.01796

## VKOGA (correlated-input variance indices)

VKOGA reports variance-based sensitivity indices for parameters that are genuinely dependent. It separates what a parameter explains by itself from what it explains through its correlations. Apart from VKOGA and [Kucherenko](#kucherenko-dependent-input-sobol-indices), every variance-based method on this page assumes independent parameters, or sidesteps the question by measuring something other than variance.

VKOGA is the given-data route of that pair. It is the surrogate-based sensitivity analysis (SSA) of Hilhorst, Quicken, van de Vosse & Huberts (2024), which computes the correlated variance-based indices of Li et al. (2010), five of them. Pick it when your parameters are dependent and you still want variance fractions, not a distributional distance. Any set of $(X, Y)$ pairs works, with no sampling design.

The method runs in two stages, and the split is the whole point. The indices need nested conditional sampling. That is hopeless against an expensive model, but trivial against a cheap emulator:

1. Fit a VKOGA surrogate (Vectorial Kernel Orthogonal Greedy Algorithm; the greedy fit itself follows De Marchi, Schaback & Wendland's P-greedy, 2005) to the given $(X, Y)$ data. It uses a Gaussian RBF kernel, with centres chosen one at a time at the maximiser of the power function (P-greedy) expressed in a nested Newton basis, and coefficients from an RKHS-regularised least-squares solve. *Which* point is picked next depends only on $X$, so all output slices share one basis. *How many* get picked does not: the greedy loop stops on a residual tolerance measured against $Y$, so the final centre count also depends on the fit quality. That is the "vectorial" part: one shared basis, not a target-independent one. `gamma` and `ridge` are chosen by k-fold cross validation.
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

Here it is on a model simple enough to check by hand: $Y = X_1 + X_2 + X_3$, standard normal marginals, $\mathrm{corr}(X_1, X_2) = 0.9$. The surrogate trains on 1024 points from an **independent** design, and the analysis then applies the declared correlation.

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxgsa

spec = {n: {"dist": "gaussian", "mean": 0.0, "variance": 1.0} for n in ("x1", "x2", "x3")}
R = np.eye(3)
R[0, 1] = R[1, 0] = 0.9
problem = jaxgsa.Problem.from_dict(spec).with_correlation(R)

X = jnp.asarray(jaxgsa.sampling.monte_carlo(jaxgsa.Problem.from_dict(spec), n=1024, seed=0))
result = jaxgsa.vkoga.analyze(
    problem, X, X.sum(axis=1), gamma=0.5, ridge=1e-8, key=jax.random.key(0), verbose=False
)
print("S_TC", np.round(np.asarray(result.S_TC), 3))
print("S_TU", np.round(np.asarray(result.S_TU), 3))
print("S_U ", np.round(np.asarray(result.S_U), 3))
print("S_C ", np.round(np.asarray(result.S_C), 3))
```

```
S_TC [0.74 0.74 0.22]
S_TU [0.041 0.041 0.222]
S_U  [0.039 0.039 0.219]
S_C  [0.7   0.701 0.001]
```

The exact values are $S_{TC} = [0.752, 0.752, 0.208]$ and $S_{TU} = [0.040, 0.040, 0.208]$, so the kernel surrogate is within about 0.015 everywhere.

Read the decision off it. $x_1$ and $x_2$ each explain 74% of the output variance, and each explains 4% that nothing else can account for. So both are worth measuring accurately, and either one can be fixed provided you keep the other free. $x_3$ has $S_{TC} = S_{TU} = 0.22$ and $S_C = 0.001$, which is zero to the accuracy of the surrogate: it is uncorrelated, so its two answers coincide and it must be neither fixed nor ignored. A ranking on $S_{TC}$ alone would have told you $x_3$ was the least important parameter. On $S_{TU}$ it is the most important. Both readings are correct, and they answer different questions.

The remaining three split $S_{TC}$ and $S_{TU}$ into their independent and correlation-borne parts:

| Index | Definition | Meaning |
|-------|------------|---------|
| $S_{TC}(i)$ | $\mathrm{Var}[\mathbb{E}(Y \mid X_i)] / \mathrm{Var}(Y)$ | Total correlated: what $X_i$ explains through itself, plus what it explains through its correlation with the rest. Use for parameter prioritisation. |
| $S_{TU}(i)$ | $\mathbb{E}[\mathrm{Var}(Y \mid \mathbf{X}_{\sim i})] / \mathrm{Var}(Y)$ | Total uncorrelated: what only $X_i$ can explain. Use for parameter fixing. |
| $S_U(i)$ | $\mathbb{E}[\mathrm{Var}(f_i \mid \mathbf{X}_{\sim i})] / \mathrm{Var}(Y)$ | The contribution of $X_i$ alone, with the part of it that $\mathbf{X}_{\sim i}$ already determines removed. $f_i$ is the fitted additive component of the output. |
| $S_C(i)$ | $S_{TC} - S_U$ | The correlation-borne contribution. It can be negative, when a correlation works against a direct effect. |
| $S_{IU}(i)$ | $S_{TU} - S_U$ | Independent interactions. Zero for an additive model, non-negative always. |

The name $S_{TC}$ says "total", but the formula is a first-order conditional variance. "Total" names the pathways it counts, direct and correlated, not the interaction order. It is not a total-order Sobol' index.

$S_U$ uses an additive projection $f_i$, and no additive function of $X_i$ can represent an interaction. On a model with interactions under a correlated measure, the raw $S_U$ can therefore come out above $S_{TU}$. jaxgsa clips $S_U$ to $S_{TU}$, which keeps $S_{IU}$ non-negative, and warns when the clip is wider than 1% of the output variance. Read that warning as a statement about the model: the additive component functions are not enough for it. Trust $S_{TC}$ and $S_{TU}$ in that case, and treat $S_U$, $S_C$ and $S_{IU}$ as indicative. $S_C$ is never clipped, because a negative $S_C$ is a real reading.

Under independent parameters the whole structure collapses back to the familiar one. $S_{TC}$ becomes the first-order Sobol' index $S_1$, $S_{TU}$ becomes the total index $S_T$, $S_U$ equals $S_{TC}$, and $S_C$ goes to zero. Running it on an uncorrelated problem is therefore a legitimate, if roundabout, way to get $S_1$ and $S_T$ from a kernel surrogate.

### VKOGA or HDMR's ANCOVA split?

Both handle correlated given data, and both report a decomposition. They decompose different things, so they answer different questions.

[HDMR](#rs-hdmr-random-sampling-high-dimensional-model-representation) fits an explicit additive expansion $f_0 + \sum_i f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots$. It splits each term's variance into a structural part $S_a$ and a correlative part $S_b$, where $S_b$ collects the covariance that term shares with the others. The decomposition is term-wise. You get per-interaction attribution and dense $S_2$/$S_3$ arrays. HDMR also knows which parameters each term involves, so it can produce correlation-aware Shapley effects via `shapley(include_correlative=True)`.

VKOGA fits a kernel expansion, which is a sum over centres rather than over parameter subsets, so it has no term-wise structure at all. What it has instead is direct access to the conditional-variance definitions. The surrogate is cheap enough to sample $\mathbb{E}(Y \mid X_i)$ and $\mathrm{Var}(Y \mid \mathbf{X}_{\sim i})$ by brute force under an explicit copula. The decomposition is per parameter, and it is the one that maps onto the prioritisation-versus-fixing decision.

Practical guidance:

- Want the prioritise / fix distinction under dependence, with an explicit and auditable dependency structure? Use VKOGA.
- Want to know which interaction carries the variance, or a fair per-parameter allocation summing to 1? Use HDMR: its terms are labelled, and only it can produce Shapley effects. `VKOGAResult.shapley()` deliberately raises `NotImplementedError`.
- Want to declare a dependency structure rather than infer one from the data (a copula from expert knowledge, a sensitivity sweep over $\rho$, or the same data analysed under several correlation assumptions)? Only VKOGA takes a correlation matrix as an argument; HDMR reads correlation implicitly out of whatever $X$ you hand it.
- The two are complementary, not redundant: HDMR's $S_b$ tells you that correlation matters, and VKOGA's $S_{TC} - S_{TU}$ gap tells you what to do about it.

### How to use it

1. You provide any set of $(X, Y)$ pairs. No sampling design required.
2. `jaxgsa.vkoga.analyze()` maps parameters to $[0, 1]$ through their marginal CDFs (the RBF kernel is isotropic, so every column must share a scale), centres the outputs, cross-validates `gamma` and `ridge`, and fits the greedy kernel surrogate.
3. The same call then draws the nested conditional samples in latent copula space and returns the five indices, along with the surrogate's `n_centers`, `gamma`, `ridge`, and per-slice training `rmse`. `key` is required, because that draw is always a Monte-Carlo estimate: `vkoga.analyze(problem, X, Y)` without one raises `ValueError`.
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

### When it is the wrong choice

- **You can still run the model.** Then run [Kucherenko](#kucherenko-dependent-input-sobol-indices) and get the same two quantities with no surrogate error in between. VKOGA is the given-data fallback, not the better estimator.
- **Your training data is correlated.** Caveat 1 above is the one that bites. $S_{TU}$ resamples $X_i$ across its whole marginal while holding the rest fixed, which is exactly the region off the correlation ridge that your training set never visited. The surrogate is extrapolating where the estimator leans on it hardest.
- **You want to know which interaction carries the variance.** A kernel expansion sums over centres, not over parameter subsets, so there is no term to point at. No $S_2$, and `VKOGAResult.shapley()` raises `NotImplementedError` on purpose. Use HDMR.
- **You cannot enable float64.** The coefficient step forms $A^\top A$ and squares the condition number. In float32 the surrogate can come out an order of magnitude worse. `analyze` warns; take the warning seriously.
- **Any of your parameters is categorical.** VKOGA refuses. The isotropic RBF kernel would read level codes as distances.
- **$D$ is large and you left `gamma` and `ridge` unset.** The default is a 10×10 grid of 10-fold refits, so up to 1000 solves before a single index is computed. Most of that is cheap: the greedy centre search, the expensive step, runs once per `(fold, gamma)` pair (100 sweeps), and only the ridge solve on top of it repeats for each of the 10 `ridge` values. Set `gamma` and `ridge` once you know good values, to skip the search entirely.

### References

- Hilhorst, G., Quicken, S., van de Vosse, F.N. & Huberts, W. (2024). Efficient sensitivity analysis for biomechanical models with correlated inputs. *International Journal for Numerical Methods in Biomedical Engineering*, 40(2), e3797.
- Li, G., Rabitz, H., Yelvington, P.E., Oluwole, O.O., Bacon, F., Kolb, C.E. & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *Journal of Physical Chemistry A*, 114(19), 6022-6032.
- De Marchi, S., Schaback, R. & Wendland, H. (2005). Near-optimal data-independent point locations for radial basis function interpolation. *Advances in Computational Mathematics*, 23(3), 317-330. The centre-selection rule this module implements, P-greedy.
- Wirtz, D. & Haasdonk, B. (2013). A vectorial kernel orthogonal greedy algorithm. *Dolomites Research Notes on Approximation*, 6, 83-100. Their VKOGA is target-dependent (f-greedy); the "VKOGA" name is borrowed here for a target-independent P-greedy fit, correctly, with residual-based stopping only.
- Santin, G. & Haasdonk, B. (2021). Kernel methods for surrogate modeling. In *Model Order Reduction, Volume 1: System- and Data-Driven Methods and Algorithms*, De Gruyter, 311-354.

## Kucherenko (dependent-input Sobol' indices)

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

`kucherenko.sample` reads `problem.correlation` and is deliberately exempt from the correlated-design error on `sobol` / `morris` / `efast`, because conditioning on the declared copula is the method's purpose. Categorical problems raise, since the conditional copula needs continuous marginals, as do problems with fewer than two parameters. Like the other samplers it takes `seed: int | np.random.Generator | None`; `scramble=False` together with a seed raises `ValueError`.

The same model as the VKOGA section, $Y = X_1 + X_2 + X_3$ at $\mathrm{corr}(X_1, X_2) = 0.9$:

```python
samples = jaxgsa.kucherenko.sample(problem, 4096, seed=0, verbose=False)
print("model runs:", samples.samples.shape[0])
result = jaxgsa.kucherenko.analyze(
    samples, jnp.asarray(samples.samples).sum(axis=1), verbose=False
)
print("S1", np.round(np.asarray(result.S1), 3))
print("ST", np.round(np.asarray(result.ST), 3))
```

```
model runs: 28672
S1 [0.752 0.752 0.208]
ST [0.04  0.04  0.208]
```

Both match the closed-form values to three decimals. The cost is the point: $4096 \times (2 \times 3 + 1) = 28672$ model runs for a 3-parameter problem, against 1024 for the VKOGA surrogate. You are paying 28 times as much to remove the surrogate from the chain.

Note $S_T < S_1$ for $x_1$ and $x_2$. Under independence that is impossible. Under dependence it is the normal case for coupled parameters, and the gap 0.752 − 0.040 is the share of $x_1$'s apparent influence that is carried by $x_2$.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_i$ | $V(\mathbb{E}(Y \mid X_i)) / V(Y)$. The classic first-order Sobol' index under independence; correlation-inclusive under a declared correlation (VKOGA's $S_{TC}$). |
| $S_{T_i}$ | $\mathbb{E}(V(Y \mid \mathbf{X}_{\sim i})) / V(Y)$. The classic total-order Sobol' index under independence; correlation-exclusive under a declared correlation (VKOGA's $S_{TU}$). $S_{T_i} \ge S_i$ no longer holds in general. |

### When to use it

- Your parameters are correlated, you want $S_1$/$S_T$ with their exact conditional-variance meaning, and you can still run the model
- You want a design-based cross-check of a VKOGA (surrogate) analysis
- Your parameters are independent and you want the classic Sobol' indices from a conditional design (it reduces to them exactly)

### When it is the wrong choice

- **Your parameters are independent.** The design reduces to the Saltelli column-swap scheme, so `kucherenko.S1`/`ST` estimate the same quantities as `sobol.S1`/`ST` for $N(2D+1)$ runs, but not with the same formula: Kucherenko's $S_1$ is the Homma–Saltelli estimator ($\mathrm{mean}(f_A f_{BA}) - \hat f_{0,A} \hat f_{0,B}$), not the Sobol'-Mauntz form `sobol` uses by default, so the two numbers agree only up to Monte-Carlo noise, not bit for bit. `sobol` also gives you $S_2$ for $N(2D+2)$. Just run `sobol`.
- **You cannot run the model.** The whole method is a design. Use [VKOGA](#vkoga-correlated-input-variance-indices).
- **You do not know the correlation matrix.** The design is built from the declared copula, and a wrong copula gives you clean estimates of the wrong quantity. Fit one with `jaxgsa.sampling.fit_correlation` if your data supports it, but understand that you are then assuming a Gaussian copula.
- **Your dependence is not a Gaussian copula.** Conditioning is closed-form only in the latent normal space. A tail-dependent or non-monotone dependence is not representable here.
- **Any of your parameters is categorical.** It raises: the conditional copula needs continuous marginals.
- **You want $S_2$ or a surrogate.** Neither is available.

### Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Computer Physics Communications*, 183(4), 937-946.

## Output shapes

All thirteen methods share the same output contract: scalar, multi-output, and time-series outputs. The shape of `Y` determines the shape of all returned index arrays. Read `S1 / ST` as the method's per-parameter measures: `mu / mu_star / sigma` for Morris, and `nu / sigma` and the bounds for DGSM. Sobol, PCE, and HDMR (`S2` and `S3`, from its explicit interaction terms) produce second-order indices; no other method does.

| Y shape | S1 / ST shape | S2 shape |
|---------|---------------|----------|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. Confidence interval arrays (when using bootstrap) prepend a leading dimension of 2 for `[lower, upper]`.

### How a 2-D Y is read

Shapes are taken as given. A 2-D `Y` is always `(N, K)`. There is no heuristic that might read it as `(N, T)` instead, `problem.output_names` does not change the reading, and there is no shape jaxgsa will quietly transpose for you. A time series is `(N, T, K)`, so a single time-varying output is written explicitly as `(N, T, 1)`.

```python
result = jaxgsa.sobol.analyze(samples, Y_2d)     # Y_2d is (8192, 5)
result.S1.shape                                  # (5, 3)  ->  (K, D)

result = jaxgsa.sobol.analyze(samples, Y_2d[:, :, None])
result.S1.shape                                  # (5, 1, 3)  ->  (T, K, D)
```

The index shape is the tell. `(K, D)` means jaxgsa read 5 separate outputs at one time step. `(T, K, D)` means it read 5 time steps of one output. Check it once on the first run and a transposed array cannot reach your plots.

A transposed array is caught by the row count, not repaired. Passing `(5, 8192)` where `(8192, 5)` was meant raises `ValueError: Y has 5 sample rows but 8192 were expected; pass Y as (N,), (N, K), or (N, T, K)`.

Setting `problem.output_names` is the guard rail worth having. When it is present, its length must equal the trailing axis, and the mismatch is caught before any array work: `output_names` of length 1 against a `(8192, 5)` `Y` raises `ValueError: output_names length 1 does not match the output axis K=5`. A 1-D `(N,)` `Y` is one output whatever the names say.

Every warning that jaxgsa raises uses the `JaxgsaWarning` category. The class is a subclass of `UserWarning`, so a filter on `UserWarning` still catches it. Filter on `JaxgsaWarning` to select the jaxgsa warnings alone:

```python
import warnings
from jaxgsa import JaxgsaWarning

warnings.filterwarnings("ignore", category=JaxgsaWarning)
```

Time-series outputs are particularly useful for dynamic models. Watching the sensitivity indices evolve over time reveals which parameters dominate at different stages of a process. For example, a parameter that is highly influential early in a batch but negligible later.

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
- Borgonovo, E., Figalli, A., Plischke, E. & Savaré, G. (2024). Global sensitivity analysis via optimal transport. *Management Science*, 71(5), 3809-3828 (in print 2025; online-first 2024). doi:10.1287/mnsc.2023.01796
