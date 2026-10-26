# Choosing a method

Start with the question you want answered, then check whether you can run your
model at new input points. If this is your first analysis with independent
inputs, [run the Sobol' quickstart](/guide/getting-started) before comparing all
fourteen methods. [GSA concepts](/guide/concepts) explains what the results mean.

## Decide in three questions

### 1. Can you choose new model inputs?

[Sobol'](/guide/methods/sobol), [eFAST](/guide/methods/efast), [Morris](/guide/methods/morris), and
[Kucherenko](/guide/methods/kucherenko) create their own sample designs. Evaluate your
model at the returned points, then pass the outputs to `analyze()`. Ordinary
input-output pairs cannot replace those designs.

The other methods analyze an existing input matrix `X` and output `Y`. DGSM
also needs the JAX model so it can calculate derivatives; it has no `sample()`
method. If model evaluations are already fixed, start with a method that uses
existing data.

### 2. What do you need to know?

- **Which inputs explain output variance?** Start with Sobol' if inputs are
  independent and you can run a new design. Use PCE or HDMR if you need to fit
  existing data. Their indices depend on the quality of the fitted model.
- **Which inputs are probably unimportant?** Use Morris for a relatively cheap
  designed first pass, or DGSM if derivatives of your JAX model are meaningful.
- **Does an input change more than output variance?** Use PAWN, Borgonovo, or
  optimal transport for changes to the output distribution. Use HSIC for a
  general dependence measure and a permutation test.
- **Are inputs dependent, or do you want one allocation per input?** Read
  [Sensitivity with dependent inputs](/guide/dependent-inputs) before choosing
  an index. Shapley divides variance into one share per input, including a
  share of interactions.

A screening score, a distributional distance, and a variance fraction answer
different questions. Their numerical values cannot be compared as if they were
the same index.

### 3. How many model runs can you afford?

For $D$ inputs, a Sobol' design uses $N(D+2)$ runs for first- and total-order
indices, or $N(2D+2)$ with second-order indices. Morris uses $r(D+1)$ runs,
where $r$ is the number of trajectories. Kucherenko uses $N(2D+1)$ runs.
eFAST uses one search curve per input, and its minimum design grows
quadratically with $D$. Here $N$ is a method-specific base sample count.

Methods that use existing data need no new model runs, though fitting or
permutation work can still be costly. DGSM evaluates model Jacobians; its cost
depends on model dimensions and differentiation direction. For larger models,
see [Scaling to large problems](/guide/scale).

## Compare the methods

Use this table to narrow the choice. Each method name links to its guide
page, which has a worked example. The
[method capabilities](#method-capabilities) table links to each API page,
which lists the assumptions, sampling instructions, result fields, and
uncertainty options.
"Own design" means you must evaluate the model at points produced by
`sample()`.

Each method's API page includes a **References** section with the original
method and the estimator used by jaxgsa. When reporting an analysis, cite
those sources alongside the software. For Sobol', use the
[estimator-specific citation table](/api/sobol#choosing-an-estimator).

| Method | Main question and result | Data needed | Watch for |
| --- | --- | --- | --- |
| [Sobol'](/guide/methods/sobol) | How much variance comes from each independent input alone and with interactions? $S_1$, $S_T$, optional $S_2$ | Own design | Requires independent inputs. |
| [eFAST](/guide/methods/efast) | How much variance is due to each independent input? $S_1$, $S_T$ | Own design | Design size grows quickly with input count. |
| [Morris](/guide/methods/morris) | Which inputs are worth a closer look? $\mu^*$, $\sigma$ | Own design | Screening statistics are not variance fractions. |
| [Kucherenko](/guide/methods/kucherenko) | What are conditional variance effects with dependent inputs? $S_1$, $S_T$, and with `subsets="all"` exact Shapley effects and PME | Own design | Interpretation differs from independent-input Sobol'. The `"all"` design costs $N\,2^D$ runs and needs $D \le 12$. |
| [PCE](/guide/methods/pce) | Can a polynomial model provide variance indices? $S_1$, $S_T$, $S_2$ | Existing `X`, `Y` | Check fit; basis size grows with dimension and order. |
| [HDMR](/guide/methods/hdmr) | Which fitted component contributes, including under dependence? ANCOVA terms | Existing `X`, `Y` | Check fit; interaction order increases cost. |
| [VKOGA](/guide/methods/vkoga) | How do correlated and uncorrelated effects differ? Several variance quantities | Existing `X`, `Y` | Results depend on a kernel surrogate. |
| [Shapley](/guide/methods/shapley) | How can variance be allocated to one share per input? Shapley effects | Existing `X`, `Y` | Interactions are shared, not reported separately; check surrogate fit. |
| [DGSM](/guide/methods/dgsm) | Are derivative-based bounds useful for screening? Bounds on $S_T$ | Existing `X` and JAX model | Bounds can be loose; Jacobians can be costly. |
| [PAWN](/guide/methods/pawn) | Does an input change the output CDF? KS distance | Existing `X`, `Y` | Conditional binning needs a tuning choice. |
| [Rank](/guide/methods/rank) | How much variance does each input explain alone? $S_1$ from one sample, by rank statistics | Existing `X`, `Y` | First order only; many tied inputs bias it. |
| [Borgonovo delta](/guide/methods/borgonovo) | Does an input change the output density? $\delta$ | Existing `X`, `Y` | Density estimates need enough data. |
| [Optimal transport](/guide/methods/optimal-transport) | How far does the output distribution move? Transport distance | Existing `X`, `Y` | Transport calculations cost more than simple univariate statistics. |
| [HSIC](/guide/methods/hsic) | Are input and output statistically dependent? Kernel dependence and p-values | Existing `X`, `Y` | Kernel and permutation cost grow with sample size. |

### Special cases

- **Dependent inputs:** Kucherenko (including its exact Shapley and PME
  output), VKOGA, HDMR, and the HDMR-backed Shapley route answer different
  questions. Distribution-based methods include
  correlation-borne influence. Use the [dependent-input guide](/guide/dependent-inputs)
  to choose a definition before comparing results.
- **Categorical inputs:** Sobol', PAWN, Borgonovo, and optimal transport accept
  unordered categories. See the [categorical example](/examples/categorical-inputs).
- **Lognormal, Weibull, triangular, or beta inputs:** All methods except DGSM
  accept them. eFAST, HDMR and the HDMR-backed Shapley analysis accept only
  the triangular and beta families. See
  [Non-uniform inputs](/examples/non-uniform-inputs#lognormal-weibull-triangular-and-beta-inputs).
- **Gradients through an analysis:** This differs from DGSM's use of model
  derivatives. See [Differentiating sensitivity analyses](/guide/differentiation)
  for method-specific support.
- **Output channels on different time grids:** Most methods accept irregular
  output grids. See [Irregular output grids](/examples/irregular-outputs) for
  the supported-method matrix and result shape.

## Method capabilities

This is the compatibility reference for input types and confidence intervals.
A cross means the method refuses that input type.

| Method | Reports | Own design | Correlated | Categorical | Lognormal, Weibull, triangular, beta | Bootstrap CI |
|---|---|:--:|:--:|:--:|:--:|---|
| [`borgonovo`](/api/borgonovo) | $\delta$, $S_1$ | ✗ | ✓ § | ✓ | ✓ | `n_bootstrap` |
| [`dgsm`](/api/dgsm) | bounds on $S_T$ | ✗ | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`efast`](/api/efast) | $S_1$, $S_T$ | ✓ | ✗ | ✗ | triangular, beta ¶ | — |
| [`hdmr`](/api/hdmr) | $S_a$ / $S_b$ / $S$ per term, surrogate | ✗ | ✓ † | ✗ | triangular, beta ¶ | `n_bootstrap` |
| [`hsic`](/api/hsic) | dependence measure | ✗ | ✓ § | ✗ | ✓ | — |
| [`kucherenko`](/api/kucherenko) | $S_1$, $S_T$ under dependence; Shapley and PME with `subsets="all"` | ✓ | ✓ | ✗ | ✓ | `n_bootstrap` |
| [`morris`](/api/morris) | $\mu^*$, $\sigma$ | ✓ | ✗ | ✗ | ✓ | `n_bootstrap` |
| [`optimal_transport`](/api/optimal-transport) | $W_2^2$ index, advective + diffusive | ✗ | ✓ § | ✓ | ✓ | `n_bootstrap` |
| [`pawn`](/api/pawn) | KS distance | ✗ | ✓ § | ✓ | ✓ | `n_bootstrap` |
| [`pce`](/api/pce) | $S_1$, $S_2$, $S_T$, surrogate | ✗ | ✗ | ✗ | ✓ ‖ | `n_bootstrap` |
| [`rank`](/api/rank) | $S_1$ by rank statistics | ✗ | ✓ § | ✗ | ✓ | `n_bootstrap` |
| [`shapley`](/api/shapley) | allocation summing to 1 | ✗ | ✗ ‡ | ✗ | ✓ ‖ | `n_bootstrap` |
| [`sobol`](/api/sobol) | $S_1$, $S_2$, $S_T$ | ✓ | ✗ | ✓ | ✓ | `n_bootstrap` |
| [`vkoga`](/api/vkoga) | $S_{TC}$, $S_{TU}$, $S_U$, $S_C$, $S_{IU}$, surrogate | ✗ | ✓ | ✗ | ✓ | `n_bootstrap` |

† HDMR accepts dependent inputs through its ANCOVA decomposition. Its $S_T$
is not a classical total-effect index under dependence.

‡ The default PCE-backed Shapley analysis refuses dependent inputs. The HDMR
backend accepts them, and `include_correlative=True` includes the ANCOVA
correlative contribution in the allocation.

§ These measures include correlation-borne influence. A parameter may score
above zero because it is correlated with a parameter used by the model.

¶ eFAST and HDMR accept triangular and beta inputs. They refuse lognormal and
Weibull inputs: a long upper tail biases $S_1$ low, and more samples do not
remove the bias. DGSM refuses all four families, because it has no Poincare
constant for them.

‖ PCE, and Shapley with its default PCE backend, map these inputs through
their CDF onto Legendre polynomials. This is correct but converges slower
than a basis matched to the distribution. Shapley with `backend="hdmr"`
refuses lognormal and Weibull inputs, as HDMR does.

All methods support scalar, multi-output, and time-series outputs. Twelve
methods offer bootstrap confidence intervals through `n_bootstrap`; eFAST has
no row-bootstrap interval, while HSIC reports permutation p-values. The
[API overview](/api/#confidence-intervals) documents the shared result shape.

For a complete run and interpretation, continue with the
[basic example](/examples/basic). The [API overview](/api/) describes shared
inputs and results.

## Supplying your own input distribution

Kucherenko and VKOGA accept a custom sampler for dependence beyond Gaussian
copulas. Declare it on the problem with
`problem.with_sampler(f, sampler_dimension=k)`. The sampler maps uniform points to
joint and conditional rows. Kucherenko assembles the design. VKOGA integrates
its fitted surrogate under the supplied distribution. A joint sampler alone is
not enough. For `subsets="all"` the sampler must accept any subset of fixed
inputs. It must be a deterministic inverse Rosenblatt (conditional-quantile)
map. See [the sampler contract](/guide/dependent-inputs#what-a-custom-sampler-must-do). Draw `X` for given-data methods with `jaxgsa.sampling.monte_carlo`.
See the
[custom-dependence example](/examples/correlated-inputs#custom-dependence-without-a-copula).
PAWN, Borgonovo, optimal transport and HSIC retain their direct `X, Y` APIs.
