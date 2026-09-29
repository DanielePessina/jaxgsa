# Global sensitivity analysis concepts

Global sensitivity analysis (GSA) asks which uncertain inputs matter to a model's
output. It looks across the input ranges and distributions you specify, rather
than at one chosen set of input values.

A result is always relative to that choice of inputs. If you change their ranges,
probabilities, or correlations, you may change the ranking of important inputs.

## What does “important” mean?

Imagine a model with two uncertain inputs. If the output is `x1 + x2`, each input
contributes on its own: changing `x1` has the same effect whatever `x2` is. If
the output is `x1 * x2`, the effect of `x1` depends on `x2`. That is an
**interaction**.

For independent inputs, Sobol' indices separate these effects:

| Index | Read it as |
| --- | --- |
| $S_1(i)$, first order | The share of output variance due to input $i$ on its own. |
| $S_T(i)$, total order | The share involving input $i$, on its own or through interactions. |
| $S_2(i,j)$, second order | The share due to the interaction between inputs $i$ and $j$, beyond their separate effects. |

For an additive model, $S_1$ and $S_T$ agree. A gap between them points to
interactions. For example, an input with $S_1$ near zero and $S_T$ around 0.25
matters through interactions even though it has little effect alone. A small
$S_T$ supports fixing that input **within the model and input distributions you
analysed**.

These are shares of **variance**, not shares of the mean output or probabilities
that an input matters. [Sobol'](/api/sobol) and [eFAST](/api/efast) estimate them
from planned model runs; [PCE](/api/pce) and [HDMR](/api/hdmr) can derive them
from a fitted surrogate.

Estimates from a finite number of runs can be noisy. Small negative estimates,
or an estimated $S_1$ slightly above $S_T$, do not change the definitions.
Check confidence intervals or increase the design before interpreting small
differences. See the [Sobol' API](/api/sobol) for sampling requirements.

## Other questions need other measures

A variance share may not answer your question. Use [Choosing a method](/guide/methods)
to match the result to the decision you need to make.

### Which inputs can I screen out?

[Morris](/api/morris) measures how much the output changes in short moves across
the input space. Its $\mu^*$ summarizes the size of those changes; a large
$\sigma$ suggests a nonlinear effect or interactions. [DGSM](/api/dgsm) measures
derivatives across the input domain and can give bounds on total effects.

These are screening measures, **not variance percentages**. They are useful for
finding clearly inactive inputs before a more detailed analysis. A large
$\sigma$ alone does not tell you whether nonlinearity or interactions caused it.

### Does an input change the whole output distribution?

Two inputs can have similar variance effects but change different parts of an
output distribution. This matters for skewed, multimodal, or heavy-tailed
outputs. Distribution-based methods compare outputs across input values:

- [PAWN](/api/pawn) compares cumulative distributions.
- [Borgonovo delta](/api/borgonovo) compares densities.
- [Optimal transport](/api/optimal-transport) measures how far distributions
  move and separates changes in mean from changes in shape.
- [HSIC](/api/hsic) measures statistical dependence with kernels and can report
  permutation p-values.

Their scores are **not Sobol' indices**. They can rank inputs differently because
they measure different changes.

### What if inputs are dependent?

If inputs are correlated, one input can carry information about another even
when the model uses only the latter. The usual Sobol' variance decomposition
assumes independent inputs, so there is no single replacement index under
dependence.

[Kucherenko](/api/kucherenko), [VKOGA](/api/vkoga), and [HDMR](/api/hdmr) report
different conditional or structural and correlative variance quantities.
Distribution-based scores can include influence carried through correlation.
Do not compare these values as though they were interchangeable. See
[Sensitivity with dependent inputs](/guide/dependent-inputs) for a worked choice.

### Can I have one share per input?

[Shapley effects](/api/shapley) allocate variance across inputs so that the
shares sum to one. They divide interactions among participating inputs; they
do not show a separate pairwise interaction index.

## Before trusting a result

1. **Check the input distribution.** Indices refer to the specified ranges,
   marginal distributions, and correlations.
2. **Check the output.** Different outputs or time points can have different
   rankings; averaging them can hide that difference.
3. **Check uncertainty.** Sampling error matters when estimates are small or
   close together.
4. **Check surrogate fit.** PCE, HDMR, and VKOGA report sensitivity of a fitted
   model. Poor fit or sparse coverage can make its indices misleading. Check
   predictions on held-out data where possible. A bootstrap that keeps the
   fitted surrogate fixed does not include uncertainty from refitting it.

## Formal definitions of variance indices

For independent inputs and a square-integrable model, the output can be
decomposed into main effects and interactions:

$$
f(\mathbf{X}) = f_0 + \sum_i f_i(X_i)
  + \sum_{i<j} f_{ij}(X_i, X_j) + \cdots
$$

The variance components then add to the total output variance:

$$
\operatorname{Var}(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + \cdots
$$

The first-order, second-order, and total-order indices are

$$
S_1(i) =
\frac{\operatorname{Var}_{X_i}(\mathbb{E}[Y \mid X_i])}
{\operatorname{Var}(Y)},
\qquad
S_2(i,j) = \frac{V_{ij}}{\operatorname{Var}(Y)},
$$

$$
S_T(i) =
\frac{\mathbb{E}_{\mathbf{X}_{\sim i}}
[\operatorname{Var}(Y \mid \mathbf{X}_{\sim i})]}
{\operatorname{Var}(Y)}.
$$

Here, $\mathbf{X}_{\sim i}$ contains every input except $X_i$. For population
indices, $S_T(i) \geq S_1(i)$, and their difference is the contribution involving
interactions with input $i$. For a model with nonzero output variance, an input's
indices lie between zero and one. Finite-sample estimates may fall outside
these bounds.
