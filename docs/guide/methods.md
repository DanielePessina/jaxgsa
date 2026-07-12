# Methods

gsax implements ten complementary approaches to global sensitivity analysis (GSA). The variance-based methods (Sobol', HDMR, PCE, eFAST) decompose the variance of a model's output into contributions attributable to individual input parameters and their interactions; Shapley effects reallocate that variance decomposition into a single fair share per parameter using the Shapley value from game theory; the derivative-based method (DGSM) uses autodiff to bound total Sobol indices directly from partial derivatives; the screening method (Morris) ranks parameters cheaply from finite-difference elementary effects; and the dependence-based methods (HSIC, PAWN, Borgonovo delta) quantify sensitivity without decomposing variance — HSIC through kernel-based dependence measures, PAWN through shifts in the output's cumulative distribution, and Borgonovo delta through shifts in the output's probability density. Together, these enable practitioners to identify which parameters drive model behaviour and which are effectively unidentifiable from available measurements.

## Background: Variance-Based Sensitivity Analysis

### Why Global Sensitivity Analysis?

Unlike local sensitivity methods (e.g. partial derivatives at a nominal point), global sensitivity analysis explores the **entire parameter space** simultaneously. This is essential for non-linear models where parameter interactions and non-monotonic responses mean that local gradients can be misleading. GSA quantifies the contribution of each parameter to output uncertainty and reveals which parameters have the strongest influence on observable outputs. Parameters with negligible sensitivity indices are effectively unidentifiable from available measurements, while high sensitivity indices indicate parameters that are likely to be well-constrained by data.

In practice, GSA serves multiple roles:

- **Parameter identifiability**: Ranking parameters by their sensitivity indices before fitting reveals which parameters can realistically be estimated. Parameters with near-zero indices across all outputs are practically unidentifiable and may need to be fixed rather than estimated.
- **Experimental design**: For time-series outputs, the evolution of sensitivity indices over time can guide the selection of optimal measurement times, ensuring data is collected when outputs are most sensitive to the parameters of interest.
- **Model simplification**: If interaction indices are negligible, the model response is approximately additive, and simpler surrogate models may suffice.

### The Hoeffding–Sobol' Decomposition

The theoretical foundation of variance-based GSA is the **Hoeffding (ANOVA) decomposition**. Any square-integrable function $f(\mathbf{X})$ of $D$ independent inputs can be uniquely decomposed into summands of increasing dimensionality:

$$
f(\mathbf{X}) = f_0 + \sum_{i=1}^{D} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots + f_{1,2,\ldots,D}(X_1, \ldots, X_D)
$$

where $f_0 = \mathbb{E}[f(\mathbf{X})]$ is the overall mean, each $f_i$ captures the main effect of parameter $i$, each $f_{ij}$ captures the pairwise interaction between $i$ and $j$, and so on. Because these component functions are mutually orthogonal, the total output variance decomposes additively:

$$
\mathrm{Var}(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,D}
$$

where $V_i = \mathrm{Var}[f_i(X_i)]$, $V_{ij} = \mathrm{Var}[f_{ij}(X_i, X_j)]$, etc.

### Sobol' Sensitivity Indices

Dividing each variance component by $\mathrm{Var}(Y)$ yields the **Sobol' sensitivity indices**:

**First-order index** — the fraction of output variance explained by parameter $i$ alone (main effect):

$$
S_i = \frac{V_i}{\mathrm{Var}(Y)} = \frac{\mathrm{Var}_{X_i}[\mathbb{E}_{\mathbf{X}_{\sim i}}(Y \mid X_i)]}{\mathrm{Var}(Y)}
$$

**Second-order index** — the additional variance from the pairwise interaction between $i$ and $j$, beyond their individual main effects:

$$
S_{ij} = \frac{V_{ij}}{\mathrm{Var}(Y)}
$$

**Total-order index** — the total contribution of parameter $i$, including all interactions of any order with any other parameters:

$$
S_{T_i} = \frac{\mathbb{E}_{\mathbf{X}_{\sim i}}[\mathrm{Var}_{X_i}(Y \mid \mathbf{X}_{\sim i})]}{\mathrm{Var}(Y)} = 1 - \frac{\mathrm{Var}_{\mathbf{X}_{\sim i}}[\mathbb{E}_{X_i}(Y \mid \mathbf{X}_{\sim i})]}{\mathrm{Var}(Y)}
$$

where $\mathbf{X}_{\sim i}$ denotes all inputs except $X_i$. By construction, $S_{T_i} \geq S_i$ always holds, with equality when parameter $i$ has no interactions with other parameters. The gap $S_{T_i} - S_i$ quantifies the total interaction contribution of parameter $i$.

## Sobol' Indices via Saltelli Sampling

gsax uses the **Saltelli sampling scheme** (Saltelli 2002, 2010), which constructs a structured set of quasi-random sample matrices so that first-order ($S_1$), total-order ($S_T$), and second-order ($S_2$) indices can all be estimated from a single batch of model evaluations.

### The Pick-Freeze Sampling Scheme

The method generates two independent $N \times D$ quasi-random sample matrices $\mathbf{A}$ and $\mathbf{B}$ using a Sobol' low-discrepancy sequence (via `scipy.stats.qmc.Sobol`). For each parameter $j$, a **cross-matrix** $\mathbf{AB}^{(j)}$ is constructed by taking all columns from $\mathbf{A}$ except column $j$, which is replaced by column $j$ from $\mathbf{B}$. This "pick-and-freeze" construction allows conditional expectations to be estimated via sample averages.

The total computational cost is $N(D + 2)$ model evaluations for all $D$ first-order and $D$ total-order indices.

### Estimators

gsax implements the following estimators:

**First-order** — Saltelli (2010):

$$
\hat{S}_i = \frac{\frac{1}{N}\sum_{n=1}^{N} f(\mathbf{B})_n \cdot \left(f(\mathbf{AB}^{(i)})_n - f(\mathbf{A})_n\right)}{\mathrm{Var}(Y)}
$$

**Total-order** — Jansen (1999):

$$
\hat{S}_{T_i} = \frac{\frac{1}{2N}\sum_{n=1}^{N}\left(f(\mathbf{A})_n - f(\mathbf{AB}^{(i)})_n\right)^2}{\mathrm{Var}(Y)}
$$

**Variance normalisation**: All estimators normalise by a pooled output variance computed over the concatenation of $\mathbf{A}$ and $\mathbf{B}$ outputs (i.e. $\mathrm{Var}([\mathbf{A}; \mathbf{B}])$ over $2N$ points). Pooling both base-sample vectors doubles the effective sample size and gives a more robust variance estimate.

### How to use it

1. `gsax.sample()` generates the Sobol' quasi-random sequence and builds the Saltelli cross-matrices. Duplicate rows are removed so your model only evaluates unique input points.
2. You evaluate your model on `sampling_result.samples`.
3. `gsax.analyze()` reconstructs the Saltelli layout internally and computes all indices in a single `jit(vmap(...))` pass.

Optional: `gsax.analyze(..., prenormalize=True)` applies SALib-style output
standardization once per output slice over the sample axis before computing the
Sobol estimators. This changes the point-estimate path to align more closely
with SALib. When bootstrapping, `ci_method="quantile"` reports percentile
bootstrap lower/upper bounds, while `ci_method="gaussian"` reports symmetric
gaussian lower/upper bounds from the bootstrap standard deviation. `gsax` still
returns endpoint arrays rather than SALib's symmetric confidence widths.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | Fraction of output variance due to parameter $i$ alone (main effect). |
| $S_T(i)$ | Fraction of output variance due to parameter $i$ including all its interactions. $S_T \geq S_1$ always. |
| $S_2(i,j)$ | Fraction of output variance due to the pairwise interaction between $i$ and $j$, beyond their individual effects. |

**When to use Sobol':** You can afford the structured Saltelli sampling design ($N(D+2)$ for first/total only, $N(2D+2)$ with second-order (default)) and want exact, model-free variance decomposition with independent inputs.

## RS-HDMR (Random Sampling High-Dimensional Model Representation)

RS-HDMR takes a fundamentally different approach: instead of requiring a structured sampling design, it constructs a **B-spline surrogate** from any set of input–output pairs and then derives sensitivity indices analytically from the surrogate's variance decomposition.

### Theoretical Background

High-Dimensional Model Representation (HDMR) exploits the observation that, for many practical problems, only the low-order interactions among input variables significantly influence the model output. The RS-HDMR variant constructs component functions from randomly sampled input–output data, rather than requiring structured grids. The model is decomposed as:

$$
f(\mathbf{X}) \approx f_0 + \sum_{i} f_i(X_i) + \sum_{i<j} f_{ij}(X_i, X_j) + \sum_{i<j<k} f_{ijk}(X_i, X_j, X_k)
$$

where each component function is expanded in a B-spline basis and fitted via backfitting with Tikhonov regularisation.

### ANCOVA Decomposition

Unlike the classical Sobol' decomposition which assumes independent inputs, RS-HDMR uses an **ANCOVA (analysis of covariance) decomposition** that separates each component's variance into:

- **Structural variance ($S_a$)**: the contribution that would remain if all inputs were independent — analogous to the classical Sobol' index.
- **Correlative variance ($S_b$)**: the additional contribution arising from correlations between inputs.

This distinction is important in practice because many real-world models have correlated inputs (e.g. coupled physical parameters), and conflating structural and correlative contributions can lead to misleading sensitivity rankings.

### How to use it

1. You provide any set of $(X, Y)$ pairs — no structured sampling design required.
2. `gsax.analyze_hdmr()` normalises inputs to $[0, 1]$, optionally
   standardises outputs once over the sample axis via
   `prenormalize=True`, builds B-spline basis matrices, and fits component
   functions via backfitting with Tikhonov regularisation.
3. The ANCOVA decomposition splits each component's variance into structural ($S_a$) and correlative ($S_b$) parts. Total-order indices ($S_T$) sum contributions from all terms involving a given parameter.

When HDMR prenormalization is enabled, the fitted surrogate is trained on the
standardized outputs but `gsax.hdmr.emulate()` (or the top-level alias
`gsax.emulate_hdmr()`) maps predictions back to the original output scale
before returning them.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_a(t)$ | Structural (uncorrelated) variance contribution of term $t$. For first-order terms with independent inputs, equivalent to Sobol' $S_1$. |
| $S_b(t)$ | Correlative variance contribution of term $t$ (due to input correlations). |
| $S(t)$ | Total contribution per term: $S_a + S_b$. |
| $S_T(i)$ | Total-order per parameter: sum of $S$ for all terms involving parameter $i$. |

**When to use HDMR:**
- Model evaluations are expensive and you want to reuse existing runs (no structured design needed)
- Inputs may be correlated (Sobol' assumes independent inputs)
- You need a surrogate/emulator for fast prediction at new inputs (`gsax.hdmr.emulate`)

## PCE (Polynomial Chaos Expansion)

PCE takes a spectral approach: it fits an orthogonal polynomial surrogate to `(X, Y)` data and then reads Sobol indices directly from the expansion coefficients (Sudret, 2008). The polynomial basis follows the Wiener-Askey scheme: Legendre polynomials for uniform inputs, Hermite polynomials for unbounded Gaussian inputs; truncated Gaussian inputs use Legendre polynomials after CDF mapping to [-1, 1].

### How to use it

1. You provide any set of $(X, Y)$ pairs (scalar output only in this version).
2. `gsax.analyze_pce()` maps inputs to the appropriate reference domain, builds the design matrix from a total-degree multi-index, and fits coefficients via regularized least squares.
3. Sobol indices ($S_1$, $S_T$, $S_2$) are computed analytically from the squared coefficients.
4. Leave-one-out cross-validation RMSE quantifies surrogate accuracy.

**When to use PCE:**
- You want analytical Sobol indices without Monte Carlo sampling noise
- Your model is smooth enough to be well-approximated by low-order polynomials
- You have mixed uniform and Gaussian inputs (the Wiener-Askey scheme selects the optimal basis automatically)
- You need a fast emulator for scalar-output models

## Shapley Effects

Shapley effects apply the **Shapley value** from cooperative game theory to variance-based sensitivity analysis: the output variance is treated as a payout to be divided fairly among the input parameters, viewed as players whose coalition worths are the partial variances of the ANOVA decomposition (Owen, 2014; Song, Nelson & Staum, 2016). Like RS-HDMR and PCE, this is a **given-data** method: it works with any set of $(X, Y)$ pairs and requires no structured sampling design.

### Theoretical Background

For independent inputs, the Hoeffding–Sobol' decomposition splits the output variance into partial variances $V_u$ indexed by subsets $u \subseteq \{1, \ldots, D\}$ of the parameters. The Shapley effect of parameter $i$ allocates each interaction's variance equally among its participants:

$$
\mathrm{Sh}_i = \sum_{u \ni i} \frac{V_u}{|u|}
$$

so a main-effect variance $V_i$ is attributed entirely to parameter $i$, a pairwise interaction variance $V_{ij}$ is split half-and-half between $i$ and $j$, and so on. Under independent inputs this yields:

- **Bracketing**: $S_{1,i} \leq \mathrm{Sh}_i \leq S_{T,i}$ — the Shapley effect always lies between the first-order and total-order Sobol indices.
- **Exact partition**: unlike $S_1$ (which omits interactions, so $\sum_i S_{1,i} \leq 1$) and $S_T$ (which counts each interaction once per participant, so $\sum_i S_{T,i} \geq 1$), Shapley effects split every interaction fairly and sum to exactly 1 with no gaps or double counting.

**Independence assumption (v1 limitation)**: gsax currently assumes **independent inputs**. The Shapley value is particularly attractive for dependent inputs — where Sobol indices lose their clean interpretation — but the dependent-input formulation requires conditional-variance estimation and is future work. Do not rely on the indices when inputs are strongly correlated.

### How gsax computes them

gsax computes Shapley effects **analytically** from a fitted surrogate's variance decomposition — no permutation Monte Carlo, no conditional-variance sampling, and no external `shap` dependency:

- **`backend="hdmr"`** (default) fits the RS-HDMR B-spline surrogate and uses the structural ($S_a$) variances of its component functions as the partial variances $V_u$, truncated at `maxorder`.
- **`backend="pce"`** fits a polynomial chaos expansion and groups the squared orthonormal coefficients by the support of their multi-index (Sudret, 2008) — exact within the fitted polynomial.

Normalization is by the surrogate's **total decomposed variance** $\sum_u V_u$, so $\sum_i \mathrm{Sh}_i = 1$ exactly — the Shapley efficiency property (Owen, 2014). $S_1$ and $S_T$ from the same surrogate use the same denominator, so for `backend="pce"` they match `analyze_pce` exactly, while for `backend="hdmr"` they differ from `analyze_hdmr` (which normalizes by $\mathrm{Var}(Y)$) by a factor of `explained_variance`. How much of the *output* variance the surrogate actually captured is reported separately in the `explained_variance` field, $\sum_u V_u / \mathrm{Var}(Y)$: close to 1 for a good fit, below 1 when truncation or fit error leaves variance unexplained, and above 1 when an overfit surrogate over-counts shared variance — an honest diagnostic rather than a silently renormalized result. A `UserWarning` is emitted when it strays far from 1. Interactions above `maxorder` (HDMR) or the polynomial order (PCE) are absent from the allocation.

### How to use it

1. You provide any set of $(X, Y)$ pairs — no structured sampling design required.
2. `gsax.analyze_shapley()` fits the selected surrogate backend (`"hdmr"` or `"pce"`), extracts its variance decomposition, and allocates each partial variance equally among the parameters in its interaction set.
3. The result carries `Sh` alongside `S1` and `ST` computed from the **same surrogate**, so the three indices are directly comparable and the ordering $S_1 \leq \mathrm{Sh} \leq S_T$ is visible at a glance.

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

X = gsax.sample_mc(PROBLEM, N=2000, seed=42)
Y = evaluate(jnp.asarray(X))

# HDMR backend (default) — supports scalar, multi-output, time-series Y
result = gsax.analyze_shapley(PROBLEM, jnp.asarray(X), Y)
print("Sh:", result.Sh)              # (D,) Shapley effects
print("sum:", result.Sh.sum())       # == 1 (Shapley efficiency property)
print("explained:", result.explained_variance)  # sum_u V_u / Var(Y) — fit quality
print("order:", result.order)        # effective surrogate order used
print("S1:", result.S1)              # first-order, same surrogate
print("ST:", result.ST)              # total-order, same surrogate

# PCE backend — scalar Y only, PCE-only knobs
result_pce = gsax.analyze_shapley(PROBLEM, jnp.asarray(X), Y, backend="pce", order=4)
```

Backend-specific keyword arguments are validated: explicitly setting a knob that belongs to the non-selected backend (e.g. `backend="pce"` with `maxorder=3`) raises `ValueError`.

### Index summary

| Index | Meaning |
|-------|---------|
| $\mathrm{Sh}(i)$ | Shapley effect: parameter $i$'s fair share of decomposed variance, including an equal split of every interaction it participates in. $\sum_i \mathrm{Sh}_i = 1$ exactly (Shapley efficiency). |
| $S_1(i)$ | First-order index from the same surrogate (main effect only). |
| $S_T(i)$ | Total-order index from the same surrogate (main effect plus all interactions counted in full). |
| `explained_variance` | Fraction of $\mathrm{Var}(Y)$ the surrogate captured, $\sum_u V_u / \mathrm{Var}(Y)$ — a separate fit-quality diagnostic, not a per-parameter index. |

**When to use Shapley effects:**
- You want a single, fairly allocated importance score per parameter that sums to exactly 1 (e.g. for ranking, reporting, or budget allocation), with a separate `explained_variance` diagnostic reporting how much output variance the surrogate captured
- Interactions matter and you want them attributed to their participants rather than omitted ($S_1$) or double-counted ($S_T$)
- You have existing $(X, Y)$ pairs and want analytical indices without permutation Monte Carlo noise
- Your inputs are independent (required in this version)

### References

- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245-251.
- Song, E., Nelson, B.L. & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060-1083.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7), 964-979.

## eFAST (Extended Fourier Amplitude Sensitivity Test)

eFAST uses a frequency-based variance decomposition to compute first-order and total-order Sobol indices. Instead of the pick-and-freeze design used by Saltelli sampling, eFAST evaluates the model along **sinusoidal search curves** in the input space, then applies the discrete Fourier transform to extract variance contributions from the spectral content of the model output.

### How it works

For each parameter $i$, eFAST constructs a search curve by assigning the highest frequency $\omega_0$ to parameter $i$ (the "focal" parameter) and lower complementary frequencies $\omega_j$ to all other parameters. The model is evaluated at $N$ points along each curve, yielding one output vector per parameter.

The Fourier power spectrum of the output along each curve is then decomposed:

**First-order index** — the fraction of total variance captured by harmonics of $\omega_0$:

$$
S_i = \frac{D_1}{V} = \frac{\sum_{p=1}^{M} |F_{p\omega_0}|^2}{V}
$$

where $V$ is the total variance (via Parseval's theorem) and $M$ is the interference factor controlling how many harmonics are summed.

**Total-order index** — the complement of the low-frequency (non-focal) variance:

$$
S_{T_i} = 1 - \frac{D_t}{V} = 1 - \frac{\sum_{k \leq \lfloor\omega_0/2\rfloor} |F_k|^2}{V}
$$

The low-frequency content at or below $\lfloor\omega_0/2\rfloor$ is driven entirely by the complementary parameters' slower oscillations, so subtracting it from unity gives the total effect of the focal parameter including all its interactions.

### How to use it

1. `gsax.sample_efast()` (or `efast.sample()`) generates search-curve samples with shape `(N * D, D)`, where each contiguous block of $N$ rows corresponds to one parameter's search curve.
2. You evaluate your model on all `N * D` rows.
3. `gsax.analyze_efast()` (or `efast.analyze()`) splits the output by curve, computes the Fourier spectrum for each, and extracts S1 and ST indices.

### Index summary

| Index | Meaning |
|-------|---------|
| $S_1(i)$ | Fraction of output variance from the focal parameter's harmonics (main effect). |
| $S_T(i)$ | Total effect including interactions, computed as $1 - D_t / V$. |

eFAST does **not** produce second-order ($S_2$) interaction indices. If pairwise interactions are needed, use the Sobol workflow instead.

**When to use eFAST:**
- You only need S1 and ST (no S2 required)
- You want a simpler sampling design without the Saltelli cross-matrix structure
- You are screening a large number of parameters
- The total cost is $N \times D$ evaluations, which can be lower than Saltelli's $N(D+2)$ for first/total only, $N(2D+2)$ with second-order (default), when $N$ is chosen smaller than the Saltelli base count

### Reference

Saltelli, A., Tarantola, S. & Chan, K.P.-S. (1999). A quantitative model-independent method for global sensitivity analysis of model output. *Technometrics*, 41(1), 39-56.

## DGSM (Derivative-based Global Sensitivity Measures)

DGSM computes sensitivity information directly from the **partial derivatives** of the model output with respect to each input parameter. Unlike Sobol' or eFAST, which decompose output variance via structured sampling, DGSM exploits JAX's reverse-mode automatic differentiation to obtain exact gradients and then derives **bounds** on the total Sobol index $S_T$.

### The DGSM Moments

For a model $f(\mathbf{X})$ with $D$ inputs, DGSM computes two statistics for each parameter $i$:

**Mean squared derivative** (importance measure):

$$
\nu_i = \mathbb{E}\left[\left(\frac{\partial f}{\partial X_i}\right)^2\right]
$$

**Mean derivative**:

$$
\sigma_i = \mathbb{E}\left[\frac{\partial f}{\partial X_i}\right]
$$

These moments are estimated from $N$ i.i.d. Monte Carlo samples. Because DGSM uses `jax.jacrev` (reverse-mode autodiff), the cost of computing the full Jacobian for all $D$ inputs in a single pass is comparable to a single model evaluation, making DGSM particularly efficient for high-dimensional problems.

### Bounds on the Total Sobol Index

DGSM does not compute Sobol indices directly. Instead, it provides an **upper bound** and a **lower bound** on the total-order index $S_{T_i}$.

**Poincaré upper bound** (Sobol' & Kucherenko, 2009):

$$
S_{T_i} \leq \frac{C(p_i) \cdot \nu_i}{\mathrm{Var}(Y)}
$$

where $C(p_i)$ is the **Poincaré constant** of the $i$-th input's marginal distribution.

**Kucherenko–Song lower bound** (Kucherenko & Song, 2016):

$$
S_{T_i} \geq \frac{\mathrm{Var}(X_i) \cdot \sigma_i^2}{\mathrm{Var}(Y)}
$$

When the upper and lower bounds are close, DGSM gives a tight bracket on $S_T$ without the cost of a full Sobol analysis.

### Poincaré Constants by Distribution

The Poincaré constant depends on the marginal distribution of each input:

| Distribution | Poincaré Constant $C$ |
|---|---|
| Uniform $[a, b]$ | $(b - a)^2 / \pi^2$ |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\sigma^2$ |
| Truncated Normal | Spectral solve (P1 finite-element Neumann eigenproblem) |

For truncated normal inputs, the constant is computed numerically by solving a weighted eigenproblem on a finite-element grid. gsax handles this automatically when the input spec declares truncation bounds.

### How to use it

1. `gsax.sample_mc()` generates plain Monte Carlo samples from the declared input distributions.
2. You pass your JAX-differentiable function and the samples to `gsax.analyze_dgsm()`.
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

**When to use DGSM:**
- You have a JAX-differentiable model and want fast screening without the cost of Saltelli or eFAST sampling
- You want bounds on $S_T$ rather than exact indices
- You are screening a large number of parameters where autodiff is cheaper than structured designs
- You want a quick sanity check before running a full Sobol analysis

### References

- Sobol', I.M. & Kucherenko, S. (2009). Derivative based global sensitivity measures and their link with global sensitivity indices. *Mathematics and Computers in Simulation*, 79(10), 3009-3017.
- Kucherenko, S. & Song, S. (2016). Derivative-based global sensitivity measures and their link with Sobol' sensitivity indices. *Reliability Engineering & System Safety*, 148, 81-95.
- Lamboni, M., Iooss, B., Popelin, A.-L. & Gamboa, F. (2013). Derivative-based global sensitivity measures: General links with Sobol' indices and numerical tests. *Mathematics and Computers in Simulation*, 87, 44-54.

## Morris (Elementary Effects Screening)

Morris is a global **screening** method — a globalized one-at-a-time (OAT) design. Instead of decomposing variance, it measures coarse finite-difference effects of each input at many locations spread across the whole input domain, then summarises them into cheap, robust importance measures. Its typical role is triage: with a fraction of the budget of a Sobol' analysis, Morris identifies which parameters are negligible and can be fixed before spending model evaluations on an exact variance decomposition.

### How it works

The design consists of $r$ **trajectories**, each a path of $D + 1$ points where consecutive points differ in exactly one coordinate. The total cost is therefore $r(D+1)$ model evaluations, and each trajectory contributes one **elementary effect** per input:

$$
EE_i = \frac{f(\mathbf{x} + \Delta \mathbf{e}_i) - f(\mathbf{x})}{\Delta}
$$

where $\mathbf{e}_i$ is the unit vector along input $i$ and $\Delta$ is the step in unit-cube coordinates. gsax implements two designs:

- **Trajectory design** (Morris 1991, default): each trajectory is a random walk on a $p$-level grid (`num_levels`, default 4) with the canonical step $\Delta = p / (2(p-1))$, visiting inputs in a random order.
- **Radial design** (Campolongo et al. 2011, `method="radial"`): star designs around scrambled-Sobol' base points, where each elementary effect compares a one-coordinate swap against the shared base point with a per-step $\Delta_i = b_i - a_i$.

Both uniform and Gaussian marginals are supported. The design includes the unit-cube boundaries, which an unbounded inverse CDF would map to infinity, so each Gaussian coordinate is confined to $[q, 1-q]$ (`truncation_quantile`, default $q = 0.005$ — the 0.5%–99.5% quantile range) before the inverse-CDF transform; uniform marginals are untouched, and deduplication and prefix-nesting are unaffected.

The $r$ elementary effects per input are reduced to three screening measures:

- $\mu_i$ — the **mean** elementary effect. Sign cancellation can mask non-monotonic influence, which is why $\mu$ alone is unreliable.
- $\mu^*_i$ — the mean **absolute** elementary effect (Campolongo et al. 2007), the headline importance measure and a good proxy for the total-order index $S_T$ ranking.
- $\sigma_i$ — the standard deviation of the elementary effects (ddof=1). A large $\sigma_i$ relative to $\mu^*_i$ indicates nonlinearity or interactions with other inputs.

The canonical output is the **$\mu^*$–$\sigma$ scatter plot**: parameters near the origin are negligible, parameters far along the $\mu^*$ axis are influential, and parameters high above the diagonal act mainly through nonlinearity or interactions.

Morris is closely related to DGSM: as $\Delta \to 0$, $\mu^*_i \to \mathbb{E}|\partial f / \partial x_i|$, so Morris is the black-box, macro-step analog of gsax's DGSM — use DGSM when the model is JAX-differentiable, Morris when it is not.

### How to use it

1. `gsax.sample_morris()` builds the trajectories, removes exact duplicate rows (grid designs collide often in low dimensions, so this saves real model evaluations, just like Saltelli sampling), and returns only the unique rows.
2. You evaluate your model on `sampling_result.samples`.
3. `gsax.analyze_morris()` reconstructs the expanded design internally, drops trajectories containing non-finite values with a warning, and reduces one elementary effect per trajectory and parameter to $\mu$, $\mu^*$, and $\sigma$. Pass `num_resamples > 0` (with a JAX PRNG `key`) for bootstrap confidence intervals over trajectories.

Elementary effects are computed in unit-cube coordinates, so $\mu^*$ is directly comparable across parameters regardless of their physical ranges; `MorrisResult.to_physical_units()` rescales to derivative-scale values in the problem's native units (uniform-marginal problems only — for Gaussian marginals the inverse-CDF transform is nonlinear, so the measures stay in grid coordinates). `MorrisSamplingResult.downsample()` prefix-slices to fewer trajectories without re-simulation, mirroring `SamplingResult.downsample()`.

Compared to SALib's Morris implementation, gsax adds unique-row deduplication, vectorized multi-output and time-series analysis (SALib's Morris is scalar-only), bootstrap confidence intervals, the radial design, and prefix-nested downsampling.

### Index summary

| Measure | Meaning |
|-------|---------|
| $\mu(i)$ | Mean elementary effect. Sign cancellation can hide non-monotonic influence. |
| $\mu^*(i)$ | Mean absolute elementary effect. Headline importance measure; proxy for the $S_T$ ranking. |
| $\sigma(i)$ | Standard deviation of the elementary effects. Large $\sigma / \mu^*$ indicates nonlinearity or interactions. |

**When to use Morris:**
- You want a cheap screening pass before committing to a full Sobol' run
- Your model is a black box (not JAX-differentiable, otherwise consider DGSM)
- You have many parameters and a tight evaluation budget — the cost is $r(D+1)$ with $r$ typically 10-50
- You only need a ranking and an interaction flag, not exact variance fractions

### References

- Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161-174.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large models. *Environmental Modelling & Software*, 22(10), 1509-1518.
- Campolongo, F., Cariboni, J. & Saltelli, A. (2011). From screening to quantitative sensitivity analysis. A unified approach. *Computer Physics Communications*, 182(4), 978-988.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*, ch. 3. Wiley.

## HSIC (Hilbert–Schmidt Independence Criterion)

HSIC takes a **kernel-based** view of sensitivity: instead of decomposing output variance, it measures the statistical *dependence* between each input and the output in a reproducing kernel Hilbert space (RKHS). By mapping inputs and outputs through Gaussian RBF kernels, HSIC detects **any** form of dependence — nonlinear, non-monotone, or heteroscedastic — that variance-based indices may underweight. Like RS-HDMR, it is a **given-data** method: it works with any set of $(X, Y)$ pairs and makes no independence assumption about the inputs.

### The HSIC Dependence Measure

Each input $X_i$ and the output $Y$ are passed through a characteristic kernel — a Gaussian RBF whose bandwidth is set automatically by the **median heuristic** (the median pairwise distance between sample points). Writing $\mathbf{K}$ and $\mathbf{L}$ for the two $N \times N$ kernel matrices, gsax uses the biased V-statistic estimator

$$
\widehat{\mathrm{HSIC}}(X_i, Y) = \frac{1}{N^2}\,\mathrm{tr}(\mathbf{K}\mathbf{H}\mathbf{L}\mathbf{H}), \qquad \mathbf{H} = \mathbf{I} - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top
$$

where $\mathbf{H}$ is the centering matrix. For characteristic kernels, $\mathrm{HSIC}(X_i, Y) = 0$ if and only if $X_i$ and $Y$ are independent, so a larger value signals stronger dependence.

### First-Order and Total Indices

gsax reports two normalised indices per parameter.

**R2-HSIC** (first-order) — the normalised dependence between input $i$ and the output, analogous to a kernel correlation coefficient (centred kernel alignment):

$$
R^2_{\mathrm{HSIC}, i} = \frac{\widehat{\mathrm{HSIC}}(X_i, Y)}{\sqrt{\widehat{\mathrm{HSIC}}(X_i, X_i)\,\widehat{\mathrm{HSIC}}(Y, Y)}}
$$

**Total HSIC** — the analogue of a total-order index, capturing dependence carried through interactions with the other inputs. It is built from **augmented product kernels** $k^*_d = 1 + k_{c,d}$ (Larsen & Alexanderian, 2026), where $k_{c,d}$ is the centred kernel for input $d$. The constant term makes the product of augmented kernels capture *all* interaction orders rather than only the highest, which yields correct total indices even for purely additive models. The total index for input $i$ then follows from comparing the complement product kernel (all inputs except $i$) with the full product kernel.

Unlike Sobol indices, R2-HSIC values are individual dependence measures and do **not** sum to 1.

### Permutation p-values

Because HSIC is a dependence measure rather than a variance fraction, gsax attaches a **permutation test** to each first-order index: the output labels are randomly shuffled `n_perms` times to build a null distribution of HSIC values, and the p-value uses the Phipson–Smyth correction $(c + 1)/(M + 1)$, where $M$ is the number of permutations (`n_perms`) and $c$ counts permuted HSIC values at least as large as the observed one. A small p-value (< 0.05) indicates a statistically significant dependence between the input and the output.

### How to use it

1. `gsax.sample_mc()` generates plain Monte Carlo samples — any sampling strategy works, since no structured design is required.
2. You evaluate your model on the samples.
3. `gsax.analyze_hsic()` transforms each input to $[0, 1]$ via its marginal CDF, builds the kernel matrices with the median heuristic, and computes all indices and p-values in a single JIT-compiled pass.

HSIC is $O(N^2)$ in time and memory because it forms $N \times N$ kernel matrices; for large $N$, pass `chunk_size` to build them in row blocks and limit peak memory. For outputs of large magnitude, set `prenormalize=True` to standardise $Y$ before kernel construction.

### Index summary

| Index | Meaning |
|-------|---------|
| $R^2_{\mathrm{HSIC}}(i)$ | Normalised first-order kernel dependence between input $i$ and the output, in $[0, 1]$. |
| Total HSIC $(i)$ | Total dependence of input $i$ including interactions, via augmented complement product kernels. |
| p-value $(i)$ | Permutation p-value for the first-order dependence (Phipson–Smyth corrected). |

**When to use HSIC:**
- You want a measure that captures **any** dependence — nonlinear, non-monotone, or heteroscedastic — not just variance contributions
- Your inputs may be correlated (HSIC makes no independence assumption)
- You have existing $(X, Y)$ pairs and want indices without additional model runs
- You want statistical significance testing via permutation p-values

### References

- Gretton, A., Herbrich, R., Smola, A., Bousquet, O. & Schölkopf, B. (2005). Kernel methods for measuring independence. *Journal of Machine Learning Research*, 6, 2075-2129.
- Da Veiga, S. (2015). Global sensitivity analysis with dependence measures. *Reliability Engineering & System Safety*, 142, 346-362.
- Larsen and Alexanderian (2026). Total HSIC sensitivity indices via augmented product kernels. *arXiv preprint* arXiv:2603.00849.

## PAWN (CDF-Based Sensitivity)

PAWN takes a **distribution-based** (moment-independent) view: rather than asking how much of the output *variance* an input explains, it asks how much the **entire output distribution** shifts when that input is held fixed. It compares the unconditional output CDF against conditional CDFs obtained by fixing each input within a bin, using the **Kolmogorov–Smirnov (KS) distance** as the measure of separation (Pianosi & Wagener, 2015). Like HSIC and RS-HDMR, it is a given-data method that works with any $(X, Y)$ pairs and assumes no particular sampling design.

### The KS Distance

For parameter $i$, its range is partitioned into `n_bins` equal-width bins. Within each bin $b$, PAWN forms the conditional output CDF $F_{Y \mid X_i \in b}$ from the samples whose $i$-th input falls in that bin, and compares it with the unconditional CDF $F_Y$ (built from all samples) via the Kolmogorov–Smirnov statistic — the largest absolute gap between the two CDFs:

$$
\mathrm{KS}_{i,b} = \sup_{y}\left| F_Y(y) - F_{Y \mid X_i \in b}(y) \right|
$$

A large KS value in a bin means fixing $X_i$ there substantially changes the output distribution; a value near zero means the output is insensitive to that input over that region.

### Aggregating Across Bins

Each parameter yields one KS value per bin. The PAWN index reduces these to a single number per input using one of three statistics:

- **median** (default) — robust to a single anomalous bin.
- **max** — the worst-case shift across the input range.
- **mean** — the average shift.

Because it is built on CDFs rather than moments, the PAWN index is **moment-independent** and invariant under monotone transformations of the output — it captures tail and skewness changes that variance-based indices miss.

### How to use it

1. `gsax.sample_mc()` generates plain Monte Carlo samples (Monte Carlo, Latin Hypercube, or Sobol sequences all work — no structured design required).
2. You evaluate your model on the samples.
3. `gsax.analyze_pawn()` maps each input to $[0, 1]$, assigns samples to bins, and computes the per-bin KS distances and their aggregate in a single JIT-compiled pass. Pass `n_bootstrap > 0` for bootstrap confidence intervals.

The number of bins (`n_bins`, default 10) trades conditioning resolution against sample density per bin; with very few samples per bin the KS statistic becomes noisy, so increase $N$ or decrease `n_bins`.

### Index summary

| Index | Meaning |
|-------|---------|
| PAWN $(i)$ | Aggregated (median / max / mean) KS distance between the unconditional and conditional output CDFs for input $i$, in $[0, 1]$. Higher means stronger influence on the output distribution. |

**When to use PAWN:**
- You care about distributional changes beyond variance, such as tail behaviour or skewness shifts
- You want a moment-independent index, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy
- Your inputs may be correlated (no independence assumption or structured design)

### Reference

Pianosi, F. & Wagener, T. (2015). A simple and efficient method for global sensitivity analysis based on cumulative distribution functions. *Environmental Modelling & Software*, 67, 1-11.

## Borgonovo Delta (Density-Based Sensitivity)

Borgonovo's $\delta$ index is the second **moment-independent** method in gsax. Where PAWN summarises the distributional shift by the largest gap between CDFs, $\delta$ measures the expected **L1 distance between the entire output density** and the output density conditional on an input (Borgonovo, 2007):

$$
\delta_i = \frac{1}{2}\,\mathbb{E}_{X_i}\!\left[\int \left| f_Y(y) - f_{Y \mid X_i}(y) \right| \mathrm{d}y \right]
$$

The index lies in $[0, 1]$: it is $0$ when fixing $X_i$ never changes the output distribution, and $1$ when the output is a deterministic function of $X_i$ alone. Because it compares whole densities rather than variances, $\delta$ captures influence carried through tails, skewness, or multimodality that variance-based indices underweight, and it is invariant under monotone transformations of the output. Like HSIC and PAWN, it is a **given-data** method: any $(X, Y)$ pairs work, with no independence assumption on the inputs.

### How it works

gsax implements the given-data estimator of Plischke, Borgonovo & Smith (2013):

1. For each input, the samples are ordered by that input's rank and split into $M$ **equal-frequency classes**. By default $M$ follows the Plischke sample-size heuristic (roughly $N^{2/7}$, at most 48 classes); override it with `n_classes`.
2. The unconditional density $f_Y$ and each class-conditional density $f_{Y \mid X_i \in \mathcal{C}_m}$ are estimated by **Gaussian KDE** with Silverman bandwidths on a fixed grid of `grid_size` points spanning $[\min Y, \max Y]$.
3. The L1 distances are integrated with the trapezoid rule and averaged with class weights, giving the plug-in estimate

$$
\hat{\delta}_i = \sum_{m=1}^{M} \frac{n_m}{2N} \int \left| \hat{f}_Y(y) - \hat{f}_{Y \mid X_i \in \mathcal{C}_m}(y) \right| \mathrm{d}y
$$

The plug-in estimate is **biased upward** at finite $N$, so by default gsax applies Plischke's bootstrap bias reduction $2\hat{\delta}_i - \overline{\hat{\delta}_i^{(b)}}$ over `n_bootstrap` resamples, with percentile confidence intervals from the same replicates. Because this correction subtracts a bootstrap mean from twice the plug-in estimate, the reported $\delta$ (and its percentile-interval bounds) can fall marginally below $0$ for weak or near-noninfluential inputs at small $N$, even though the true index and the plug-in estimate both lie in $[0, 1]$. The same class partition also yields the **given-data first-order Sobol index** (variance of the class means over the total variance) at negligible extra cost, so every analysis returns both $\delta$ and $S_1$.

The estimator matches `SALib.analyze.delta` (same equal-frequency rank partition, class-count heuristic, Silverman KDE factors, and 100-point output grid) with two differences: the central estimate is computed on the original sample — deterministic given the data, where SALib evaluates it on a random resample — and a constant output column yields $\delta = S_1 = 0$ instead of an error.

### How to use it

1. `gsax.sample_mc()` generates plain Monte Carlo samples (any sampling strategy works — no structured design is required).
2. You evaluate your model on the samples.
3. `gsax.analyze_borgonovo()` partitions each input into rank classes and computes $\delta$, $S_1$, and their bootstrap intervals in a single JIT-compiled kernel, vmapped over output columns and scanned over bootstrap replicates.

Set `n_bootstrap=0` to skip bias correction and confidence intervals (raw plug-in estimate), or `bias_correct=False` to keep the intervals but report the uncorrected estimate. For large time-series outputs, lower `chunk_size` to bound peak memory, which scales with `chunk_size * D * N * grid_size`.

### Index summary

| Index | Meaning |
|-------|---------|
| $\delta(i)$ | Expected L1 distance between the unconditional and conditional output densities for input $i$, in $[0, 1]$. Higher means stronger influence on the output distribution; $0$ means no influence at all. |
| $S_1(i)$ | Given-data first-order Sobol index from the same class partition — the variance-based view of the same conditioning, for comparison at no extra cost. |

**When to use Borgonovo delta:**
- You care about influence on the whole output distribution — tails, skewness, multimodality — not just variance
- You want a moment-independent index with a fixed $[0, 1]$ scale, invariant under monotone output transforms
- You have existing $(X, Y)$ pairs from any sampling strategy, possibly with correlated inputs
- You use `SALib.analyze.delta` and want a deterministic, JIT-compiled equivalent that also handles multi-output and time-series `Y`

### Reference

- Borgonovo, E. (2007). A new uncertainty importance measure. *Reliability Engineering & System Safety*, 92(6), 771-784.
- Plischke, E., Borgonovo, E. & Smith, C.L. (2013). Global sensitivity measures from given data. *European Journal of Operational Research*, 226(3), 536-550.

## Choosing Between Them

| Consideration | Sobol' | HDMR | PCE | Shapley | eFAST | DGSM | Morris | HSIC | PAWN | Borgonovo delta |
|---------------|--------|------|-----|---------|-------|------|--------|------|------|-----------------|
| Sampling requirement | Structured Saltelli design, $N(2D+2)$ evaluations (default) | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Search curves, $N \times D$ evaluations | Plain MC, $N$ evaluations + autodiff | Trajectory or radial design, $r(D+1)$ evaluations (deduplicated) | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs |
| Input independence | Assumed | Handled via ANCOVA decomposition | Assumed | Assumed (dependent-input Shapley is future work) | Assumed | Assumed | Assumed | Not assumed | Not assumed | Not assumed |
| Surrogate/emulator | No | Yes (`emulate_hdmr`) | Yes (`emulate_pce`) | Fits HDMR or PCE internally (no emulator returned) | No | No | No | No | No | No |
| Accuracy | Exact (given enough samples) | Depends on B-spline fit quality | Depends on polynomial fit quality | Exact allocation within the fitted surrogate; depends on fit quality | Exact (given enough samples) | Bounds on $S_T$, not exact indices | Screening ranks ($\mu^*$ as $S_T$ proxy), not variance fractions | Dependence measure, not variance fractions | Distributional (KS) distance, not variance fractions | Distributional (L1) distance, not variance fractions |
| Second-order indices | Direct estimation from cross-matrices | From interaction component functions | Analytical from coefficients | Not available (interaction variance folded into $\mathrm{Sh}$) | Not available | Not available | Not available | Not available | Not available | Not available |
| Interaction detection | Via $S_2$ and the gap $S_T - S_1$ | Via explicit interaction component functions | Via $S_2$ from coefficients | Via the gaps $\mathrm{Sh} - S_1$ and $S_T - \mathrm{Sh}$ | Via the gap $S_T - S_1$ only | Not available (bounds only) | Via large $\sigma$ relative to $\mu^*$ (not pair-attributable) | Via the Total HSIC − R2-HSIC gap | Not available (first-order only) | Not available (the $\delta - S_1$ gap flags influence beyond first-order variance) |
| Output shapes | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar only | Scalar, multi-output, time-series (`hdmr` backend); scalar only (`pce`) | Scalar, multi-output, time-series | Scalar, multi-output | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar, multi-output, time-series |
| Input distributions | Uniform + Gaussian | Any (via CDF mapping) | Uniform + Gaussian | Follows backend: any (`hdmr`), uniform + Gaussian (`pce`) | Uniform + Gaussian | Uniform + Gaussian (+ truncated Normal) | Uniform + Gaussian (truncated-quantile grid) | Any (via CDF mapping) | Any (via CDF mapping) | Any (rank-based classes) |

## Output Shapes

Sobol, HDMR, eFAST, Shapley (with the default HDMR backend), and Morris all support scalar, multi-output, and time-series outputs. The shape of `Y` determines the shape of all returned index arrays (for Morris, read `S1 / ST` as `mu / mu_star / sigma`; Morris has no S2):

| Y shape | S1 / ST shape | S2 shape |
|---------|---------------|----------|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. Confidence interval arrays (when using bootstrap) prepend a leading dimension of 2 for `[lower, upper]`.

Time-series outputs are particularly useful for dynamic models, where the evolution of sensitivity indices over time can reveal which parameters dominate at different stages of a process — for example, a parameter that is highly influential early in a batch but negligible later.

DGSM returns arrays of shape `(T, D)` where `T` is the number of output components. For scalar-output models, `T = 1`. DGSM does not use the `(N, T, K)` convention; instead, multi-output functions produce `(T,)` outputs per sample and the result arrays are always `(T, D)`.

## Data Cleaning

`gsax.analyze()` automatically drops sample groups that contain non-finite values (NaN, Inf). The Saltelli layout requires groups of rows to stay together, so if any row in a group is non-finite, the entire group is removed. A message is printed when this happens. The `nan_counts` field on the result reports how many NaN values remain in the computed indices.

## References

- Sobol', I.M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.
- Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*, 181(2), 259-270.
- Jansen, M.J.W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1-2), 35-43.
- Li, G., Rabitz, H., Yelvington, P.E., Oluwole, O.O., Bacon, F., Kolb, C.E., & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *Journal of Physical Chemistry A*, 114(19), 6022-6032.
- Rabitz, H. & Alis, O. (1999). General foundations of high-dimensional model representations. *Journal of Mathematical Chemistry*, 25(2-3), 197-233.
- Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions. *Reliability Engineering & System Safety*, 93(7), 964-979.
- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245-251.
- Song, E., Nelson, B.L. & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060-1083.
- Saltelli, A., Tarantola, S. & Chan, K.P.-S. (1999). A quantitative model-independent method for global sensitivity analysis of model output. *Technometrics*, 41(1), 39-56.
