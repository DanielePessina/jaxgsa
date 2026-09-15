# Choosing a method

jaxgsa provides thirteen global sensitivity analysis methods. Choose one by
the question you want to answer, the data you have, and the model evaluations
you can afford.

If this is your first analysis, start with [Sobol](#start-with-sobol) when you
can run the model at new points. Start with [a given-data method](#when-you-already-have-data)
when the model runs already exist. For definitions of the reported quantities,
read [GSA concepts](/guide/concepts).

## Decide in three questions

### 1. Can you run the model at new points?

Four methods build a dedicated design: Sobol, eFAST, Morris, and Kucherenko.
They require model evaluations at the points returned by `sample()`.

The other nine methods use ordinary input samples. Pass the input matrix `X`
and model output `Y` to `analyze()`. DGSM also takes the model itself so it can
compute derivatives, but it does not require a dedicated design.

### 2. What should the result mean?

| Goal | Use | Result |
| --- | --- | --- |
| Attribute output variance | Sobol, eFAST, PCE, HDMR | First-, total-, or higher-order variance contributions |
| Allocate variance to one number per parameter | Shapley | Shares that sum to one |
| Screen many parameters cheaply | Morris, DGSM | Elementary effects or derivative-based bounds |
| Measure changes to the full output distribution | PAWN, Borgonovo, optimal transport | CDF, density, or transport distances |
| Test general dependence | HSIC | Dependence indices and permutation p-values |
| Separate effects under dependent inputs | Kucherenko, VKOGA, HDMR | Conditional-variance or ANCOVA quantities |

These results are not interchangeable. A Morris score is not a variance
fraction, and a distributional distance need not rank parameters like a Sobol
index. Choose the quantity before comparing the numbers.

### 3. What is the evaluation budget?

- **Sobol** uses $N(D+2)$ model runs for first- and total-order indices, or
  $N(2D+2)$ when second-order indices are requested.
- **Morris** uses $r(D+1)$ runs, where $r$ is the number of trajectories.
- **Kucherenko** uses $N(2D+1)$ runs.
- **eFAST** uses one search curve per parameter; its minimum design grows
  quadratically with the number of parameters.
- **Given-data methods** require no new model evaluations, although their
  analysis time and memory requirements differ.
- **DGSM** evaluates model Jacobians. Its cost depends on the input and output
  dimensions and the automatic-differentiation direction.

Here $D$ is the number of parameters and $N$ is a method-specific base sample
count. For high-dimensional planning, see [Scaling to large problems](/guide/scale).

## Common choices

### Start with Sobol

Use Sobol when the inputs are independent, you can choose the evaluation
points, and you want first-order, total-order, or pairwise interaction indices.
It is the most direct general-purpose variance decomposition in the package.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

samples = jaxgsa.sobol.sample(PROBLEM, 8192, seed=0)
Y = evaluate(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)

print(result.S1)
print(result.ST)
```

Read the [Sobol API](/api/sobol) for estimator choices, confidence intervals,
and second-order indices. See the [basic example](/examples/basic) for a full
workflow.

### Screen before a costly analysis

Use Morris when model evaluations dominate the budget. Use DGSM when the model
is written in JAX and its derivatives are meaningful. Both methods are most
useful for identifying parameters that have little influence; neither reports
the same quantity as a Sobol index.

After screening, fix negligible parameters and run a variance-based analysis
on the remaining set. The [screen-first workflow](/examples/advanced-workflow)
shows this sequence.

### When you already have data

Choose by the result you need:

- **PCE** returns variance indices and a reusable polynomial surrogate.
- **HDMR** returns component-wise variance contributions and a reusable spline
  surrogate.
- **Shapley** returns one variance allocation per parameter.
- **VKOGA** returns dependent-input variance quantities through a kernel
  surrogate.
- **PAWN**, **Borgonovo**, and **optimal transport** compare output
  distributions rather than variance alone.
- **HSIC** measures dependence and supplies permutation p-values.
- **DGSM** evaluates derivatives of a JAX model at ordinary sample points.

Surrogate-based indices are only as reliable as the fitted surrogate. Check
the fit diagnostics before interpreting PCE, HDMR, VKOGA, or Shapley results.

### When inputs are dependent

There is no single extension of Sobol indices to dependent inputs. Kucherenko,
VKOGA, HDMR, and the HDMR-backed Shapley route answer different questions.
Distribution-based methods accept dependence but include both direct and
correlation-borne influence in their scores.

Use [Sensitivity with dependent inputs](/guide/dependent-inputs) to choose
among these definitions. Do not compare their values directly with classical
Sobol indices for independent inputs.

### When inputs are categorical

Sobol, PAWN, Borgonovo, and optimal transport accept unordered categorical
parameters. The other methods refuse them because their results would depend
on the arbitrary numeric coding of the levels.

See [Categorical inputs](/examples/categorical-inputs) for a worked comparison.

### When you need gradients through the analysis

Most methods expose a pure `indices()` computation that can participate in JAX
transformations. This is different from DGSM, which differentiates the model
to define its sensitivity measure.

Support varies by method and transformation. Read
[Differentiating sensitivity analyses](/guide/differentiation) before building
an end-to-end differentiable workflow.

## Compare the methods

### Direct variance methods

| Method | Choose it when | Main trade-off |
| --- | --- | --- |
| [Sobol](/api/sobol) | You want $S_1$, $S_T$, and optional $S_2$ for independent inputs | Requires a Saltelli design |
| [eFAST](/api/efast) | You want $S_1$ and $S_T$ from Fourier search curves | Design size grows quickly with $D$ |
| [Kucherenko](/api/kucherenko) | You can run a dedicated design under declared input dependence | Indices have dependence-specific interpretations |

### Surrogate variance and allocation methods

| Method | Choose it when | Main trade-off |
| --- | --- | --- |
| [PCE](/api/pce) | A polynomial surrogate fits the response and you want $S_1$, $S_T$, or $S_2$ | Expansion size grows with dimension and order |
| [HDMR](/api/hdmr) | You want component functions and an ANCOVA decomposition | Fit cost grows with interaction order |
| [VKOGA](/api/vkoga) | Inputs are dependent and you want several conditional-variance quantities | Results depend on a kernel surrogate |
| [Shapley](/api/shapley) | You want one allocation per parameter that sums to one | Allocation depends on the PCE or HDMR fit |

### Screening and bounds

| Method | Choose it when | Main trade-off |
| --- | --- | --- |
| [Morris](/api/morris) | You need a low-cost first pass over many parameters | Reports screening statistics, not variance fractions |
| [DGSM](/api/dgsm) | The JAX model is differentiable and derivative-based bounds are useful | Bounds may be loose; long outputs increase Jacobian cost |

### Distribution and dependence measures

| Method | Choose it when | Main trade-off |
| --- | --- | --- |
| [HSIC](/api/hsic) | You want a general dependence measure and permutation test | Kernel and permutation cost grow with sample size |
| [PAWN](/api/pawn) | You want a CDF-based measure | Conditional binning introduces a tuning choice |
| [Borgonovo](/api/borgonovo) | You want a density-based measure of distributional change | Conditional density estimation needs enough data |
| [Optimal transport](/api/optimal-transport) | You want mean-shift and shape-change contributions or a joint trajectory score | Transport solves cost more than simple univariate statistics |

### Method capabilities

This is the canonical compatibility table. “Own design” means the method
provides `sample()` and must evaluate the model on that design. A cross means
the method refuses the declared input type rather than silently approximating
it.

| Method | Reports | Own design | Correlated | Categorical | Bootstrap CI |
|---|---|:--:|:--:|:--:|---|
| [`borgonovo`](/api/borgonovo) | $\delta$, $S_1$ | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`dgsm`](/api/dgsm) | bounds on $S_T$ | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`efast`](/api/efast) | $S_1$, $S_T$ | ✓ | ✗ | ✗ | — |
| [`hdmr`](/api/hdmr) | $S_a$ / $S_b$ / $S$ per term, surrogate | ✗ | ✓ † | ✗ | `n_bootstrap` |
| [`hsic`](/api/hsic) | dependence measure | ✗ | ✓ § | ✗ | — |
| [`kucherenko`](/api/kucherenko) | $S_1$, $S_T$ under dependence | ✓ | ✓ | ✗ | `n_bootstrap` |
| [`morris`](/api/morris) | $\mu^*$, $\sigma$ | ✓ | ✗ | ✗ | `n_bootstrap` |
| [`optimal_transport`](/api/optimal-transport) | $W_2^2$ index, advective + diffusive | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pawn`](/api/pawn) | KS distance | ✗ | ✓ § | ✓ | `n_bootstrap` |
| [`pce`](/api/pce) | $S_1$, $S_2$, $S_T$, surrogate | ✗ | ✗ | ✗ | `n_bootstrap` |
| [`shapley`](/api/shapley) | allocation summing to 1 | ✗ | ✗ ‡ | ✗ | `n_bootstrap` |
| [`sobol`](/api/sobol) | $S_1$, $S_2$, $S_T$ | ✓ | ✗ | ✓ | `n_bootstrap` |
| [`vkoga`](/api/vkoga) | $S_{TC}$, $S_{TU}$, $S_U$, $S_C$, $S_{IU}$, surrogate | ✗ | ✓ | ✗ | `n_bootstrap` |

† HDMR accepts dependent inputs through its ANCOVA decomposition. Its $S_T$
is not a classical total-effect index under dependence.

‡ The default PCE-backed Shapley analysis refuses dependent inputs. The HDMR
backend accepts them, and `include_correlative=True` includes the ANCOVA
correlative contribution in the allocation.

§ These measures include correlation-borne influence. A parameter may score
above zero because it is correlated with a parameter used by the model.

All methods support scalar, multi-output, and time-series outputs. Eleven
methods offer bootstrap confidence intervals through `n_bootstrap`; eFAST has
no row-bootstrap interval, while HSIC reports permutation p-values. The
[API overview](/api/#confidence-intervals) documents the shared result shape.

## Where to go next

- Read [GSA concepts](/guide/concepts) to understand the reported quantities.
- Read [Sensitivity with dependent inputs](/guide/dependent-inputs) when the
  parameters are correlated.
- Read [Scaling to large problems](/guide/scale) before committing a large
  evaluation budget.
- Open a method's [API reference](/api/) for its complete signature and
  estimator details.
- Browse the [examples](/examples/basic) for complete analyses and output.
