# Global sensitivity analysis concepts

Global sensitivity analysis (GSA) asks how uncertainty in a model's inputs
relates to uncertainty in its outputs. Unlike a local derivative, it considers
the full input distributions rather than behaviour at one nominal point.

The practical questions are usually:

- Which parameters account for most of the output variation?
- Which parameters can be fixed with little effect?
- Which parameters matter mainly through interactions?
- Does a parameter change only the mean, or the whole output distribution?
- How does dependence between parameters affect the answer?

There is no single sensitivity index that answers all of these questions.
Choose the quantity you want to interpret before choosing an estimator. See
[Choosing a method](/guide/methods) for the package-wide comparison.

## Variance-based indices

Variance-based methods express importance as a fraction of output variance.
For independent inputs, a square-integrable model can be decomposed into main
effects and interactions:

$$
f(\mathbf{X}) = f_0 + \sum_i f_i(X_i)
  + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots
$$

The corresponding variance components add to the total output variance:

$$
\operatorname{Var}(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + \cdots
$$

This gives three commonly reported Sobol' indices.

| Index | Meaning |
| --- | --- |
| $S_1(i)$ | Variance attributed to parameter $i$ alone. |
| $S_2(i,j)$ | Additional variance attributed to the interaction between $i$ and $j$. |
| $S_T(i)$ | Variance involving parameter $i$, including all of its interactions. |

More formally,

$$
S_1(i) =
\frac{\operatorname{Var}_{X_i}(\mathbb{E}[Y \mid X_i])}
{\operatorname{Var}(Y)},
$$

$$
S_2(i,j) = \frac{V_{ij}}{\operatorname{Var}(Y)},
$$

and

$$
S_T(i) =
\frac{\mathbb{E}_{\mathbf{X}_{\sim i}}
[\operatorname{Var}(Y \mid \mathbf{X}_{\sim i})]}
{\operatorname{Var}(Y)}.
$$

Here, $\mathbf{X}_{\sim i}$ means every input except $X_i$. For the population
indices, $S_T(i) \geq S_1(i)$. The gap $S_T(i)-S_1(i)$ shows how much of the
parameter's influence is associated with interactions. A value of
$S_T(i) \approx 0$ supports fixing that parameter within the distributions and
model configuration that were analysed.

Finite-sample estimates can be noisy. Small negative estimates, or an estimated
$S_1$ slightly above $S_T$, do not change the mathematical definitions. Use
confidence intervals and larger designs before interpreting small differences.
See the [Sobol' API](/api/sobol) for estimators and sampling requirements.

[Sobol'](/api/sobol) and [eFAST](/api/efast) estimate variance indices from
dedicated model evaluations. [PCE](/api/pce) and [HDMR](/api/hdmr) derive them
from fitted surrogates. [Shapley effects](/api/shapley) instead divide the
variance into one allocation per parameter.

### Dependent inputs

The usual Sobol' decomposition assumes independent inputs. Under dependence,
"importance" can include the model's structural response, information carried
through correlation, or both. These are different estimands and need not give
the same ranking.

jaxgsa provides several routes for dependent inputs:

- [Kucherenko](/api/kucherenko) estimates conditional-variance indices from a
  dedicated design.
- [VKOGA](/api/vkoga) estimates correlated and uncorrelated variance
  contributions through a kernel surrogate.
- [HDMR](/api/hdmr) separates each fitted component into structural and
  correlative contributions.
- Distribution-based methods can measure dependence-inclusive influence
  without constructing a Sobol' decomposition.

Do not compare these values as though they were interchangeable versions of
the same index. The [methods guide](/guide/methods) explains which dependent-
input question each route answers.

## Screening measures

Screening methods look for inputs that are negligible or worth investigating
further. They are useful when the parameter count is large or the model-run
budget is too small for a precise variance decomposition.

[Morris](/api/morris) summarizes elementary effects with $\mu^*$ and $\sigma$.
A small $\mu^*$ suggests little overall influence; a large $\sigma$ suggests a
nonlinear response or interactions. [DGSM](/api/dgsm) uses derivatives across
the input domain and can provide bounds on total effects.

These quantities are not variance fractions. Use them to remove clearly inert
parameters or to plan a second analysis, not to read off percentages of
explained variance. A common workflow is to screen first, then apply a
variance-based method to the remaining parameters.

## Distribution-based measures

Variance can be an incomplete description of skewed, multimodal, or
heavy-tailed outputs. Distribution-based methods compare the unconditional
output distribution with the output conditioned on an input. They detect
changes in location, spread, and shape, but their scores are not Sobol' indices.

- [HSIC](/api/hsic) measures statistical dependence between an input and the
  output using kernels.
- [PAWN](/api/pawn) compares conditional and unconditional cumulative
  distributions with the Kolmogorov--Smirnov distance.
- [Borgonovo delta](/api/borgonovo) compares output densities.
- [Optimal transport](/api/optimal-transport) uses Wasserstein distance and
  separates mean-shift and shape-change contributions.

Use these methods when the question is "does this input change the output
distribution?" rather than "what fraction of variance does it explain?"

## Allocation and surrogate-based methods

[Shapley effects](/api/shapley) allocate importance across parameters so the
reported shares sum to one. Interactions are divided among their participants,
which gives one number per parameter but no separate pairwise interaction
index.

[PCE](/api/pce), [HDMR](/api/hdmr), and [VKOGA](/api/vkoga) fit a surrogate and
derive sensitivity measures from that fitted model. This is valuable when you
have an existing set of input-output pairs or also need a fast emulator.
However, the reported sensitivity is sensitivity of the fitted surrogate.
Poor fit, sparse coverage, or a restrictive basis can produce confident-looking
indices that do not describe the original model.

Check predictive error on held-out data where possible, inspect diagnostics,
and include surrogate refitting in uncertainty estimates. A bootstrap that
only resamples a fixed surrogate understates fit uncertainty.

## Interpreting an analysis

Keep four points attached to every result:

1. **The estimand.** A screening score, variance fraction, dependence measure,
   and distributional distance answer different questions.
2. **The input distribution.** Sensitivity is defined over the ranges,
   marginals, and correlations supplied to the analysis. Change them and the
   result can change.
3. **The output.** Indices may differ across scalar outputs or across time. A
   single aggregation can hide that structure.
4. **The uncertainty.** Sampling error and, where applicable, surrogate error
   matter most when indices are small or rankings are close.

Once the question and estimand are clear, use [Choosing a method](/guide/methods)
to select an implementation and follow its API page for sampling, assumptions,
and result fields.
