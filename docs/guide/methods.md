# Methods

gsax implements three complementary approaches to variance-based global sensitivity analysis (GSA). All methods decompose the variance of a model's output into contributions attributable to individual input parameters and their interactions, enabling practitioners to identify which parameters drive model behaviour and which are effectively unidentifiable from available measurements.

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

**When to use Sobol':** You can afford the structured Saltelli sampling design ($N(D+2)$ evaluations) and want exact, model-free variance decomposition with independent inputs.

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

PCE takes a spectral approach: it fits an orthogonal polynomial surrogate to `(X, Y)` data and then reads Sobol indices directly from the expansion coefficients (Sudret, 2008). The polynomial basis follows the Wiener-Askey scheme: Legendre polynomials for uniform inputs, Hermite polynomials for Gaussian inputs.

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

## Choosing Between Them

| Consideration | Sobol' | HDMR | PCE |
|---------------|--------|------|-----|
| Sampling requirement | Structured Saltelli design, $N(D+2)$ evaluations | Any $(X, Y)$ pairs | Any $(X, Y)$ pairs |
| Input independence | Assumed | Handled via ANCOVA decomposition | Assumed |
| Surrogate/emulator | No | Yes (`emulate_hdmr`) | Yes (`emulate_pce`) |
| Accuracy | Exact (given enough samples) | Depends on B-spline fit quality | Depends on polynomial fit quality |
| Second-order indices | Direct estimation from cross-matrices | From interaction component functions | Analytical from coefficients |
| Interaction detection | Via $S_2$ and the gap $S_T - S_1$ | Via explicit interaction component functions | Via $S_2$ from coefficients |
| Output shapes | Scalar, multi-output, time-series | Scalar, multi-output, time-series | Scalar only |
| Input distributions | Uniform + Gaussian | Uniform only | Uniform + Gaussian |

## Output Shapes

Both methods support scalar, multi-output, and time-series outputs. The shape of `Y` determines the shape of all returned index arrays:

| Y shape | S1 / ST shape | S2 shape |
|---------|---------------|----------|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. Confidence interval arrays (when using bootstrap) prepend a leading dimension of 2 for `[lower, upper]`.

Time-series outputs are particularly useful for dynamic models, where the evolution of sensitivity indices over time can reveal which parameters dominate at different stages of a process — for example, a parameter that is highly influential early in a batch but negligible later.

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
