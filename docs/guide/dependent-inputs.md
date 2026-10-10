# Choosing a method for dependent inputs

When inputs are correlated, sensitivity is no longer one question. An input
can matter because the model uses it directly, because it moves with another
input, or both. Different methods separate those effects differently.

This guide helps you choose among jaxgsa's methods. For a complete worked
analysis, see [Correlated Inputs](/examples/correlated-inputs).

## Start with the decision

Choose the method from the question you need to answer:

| Question | Method |
| --- | --- |
| What does each input explain, including what it carries through correlation? | Kucherenko $S_1$ or VKOGA $S_{TC}$ |
| What can only this input explain, after accounting for the others? | Kucherenko $S_T$ or VKOGA $S_{TU}$ |
| Which model terms carry variance, and how much comes from structure or correlation? | HDMR |
| How can I split the output variance into one share per input, with the right conditional-variance meaning? | Kucherenko Shapley effects (`subsets="all"`) |
| Which inputs does the model actually use, with zero for an input it never reads? | Kucherenko PME (`subsets="all"`) |
| How can I split the fitted variance into one number per input from existing data? | Shapley with the HDMR backend |
| Does an input change the output distribution, rather than only its variance? | HSIC, PAWN, Borgonovo delta, or optimal transport |

Then choose between a dedicated design and existing data:

- If you can run the model at new points, use **Kucherenko** for
  conditional-variance indices without a surrogate.
- If you already have $(X, Y)$ data, use **VKOGA** for conditional-variance
  indices, **HDMR** for a term-level ANCOVA split, or a distribution-based
  method when variance is not the quantity you care about.
- If you need one share per parameter and you can run the model, use
  **Kucherenko Shapley effects or PME**. They are exact conditional-variance
  quantities. The design needs $D \le 12$.
- If you need one allocation per parameter from existing data, use **Shapley
  with the HDMR backend**, but read its interpretation carefully.

## Why familiar indices split

Suppose $X_1$ and $X_2$ are correlated, while the model reads only $X_1$.
Observing $X_2$ still tells you something about the output because it carries
information about $X_1$. But once $X_1$ is known, changing $X_2$ does nothing.

That creates two useful views:

- A **correlation-inclusive** measure asks what an input explains through
  itself and through everything it moves with. This is useful when deciding
  what to measure more accurately.
- A **correlation-exclusive** measure asks what only that input can explain.
  This is useful when deciding what can be fixed at a nominal value.

Under independence these views coincide. Under dependence they can rank inputs
very differently.

## The four variance-based routes

There is no single generalisation of Sobol' indices to dependent inputs.
jaxgsa provides five routes, each with a different estimand.

| Route | What it estimates | What it needs |
| --- | --- | --- |
| [Kucherenko](/api/kucherenko) | $S_1 = V(\mathbb{E}(Y \mid X_i))/V(Y)$ and $S_T = \mathbb{E}(V(Y \mid \mathbf{X}_{\sim i}))/V(Y)$ under the supplied joint distribution | A dedicated design, $N(2D+1)$ model runs, and either a Gaussian-copula correlation matrix or a custom joint/conditional sampler |
| [VKOGA](/api/vkoga) | The same conditional-variance quantities as $S_{TC}$ and $S_{TU}$, plus $S_U$, $S_C$, and $S_{IU}$ | Existing $(X, Y)$ pairs and either a Gaussian-copula correlation matrix or a custom joint/conditional sampler |
| [Kucherenko Shapley and PME](/api/kucherenko#shapley-effects-and-pme) | The exact Shapley effect and the proportional marginal effect of each input, from conditional variances, with $\sum_i = 1$ | A dedicated `subsets="all"` design, $N\,2^D$ model runs, $D \le 12$, and a Gaussian-copula matrix or a custom sampler that supports every subset |
| [HDMR](/api/hdmr) | For each fitted component, a structural share $S_a$ and a correlation-driven share $S_b$ | Existing $(X, Y)$ pairs; dependence is inferred from $X$ |
| [Shapley with HDMR](/api/shapley) | One ANCOVA allocation per parameter, formed by splitting each term's $S_a + S_b$ among its participants | Existing $(X, Y)$ pairs |

### Kucherenko

Use Kucherenko when you can still evaluate the model and want the
conditional-variance quantities directly. It samples conditionally from either
the Gaussian copula declared on `Problem` or a custom sampler declared on the
`Problem` with `with_sampler`. A custom sampler transforms uniform points into joint and conditional physical inputs,
so it can express dependence beyond a pairwise correlation matrix. See the
[custom-dependence example](/examples/correlated-inputs#custom-dependence-without-a-copula).

The conditional-variance estimators follow
[Kucherenko, Tarantola & Annoni (2012)](/api/kucherenko#references).

- $S_1$ is correlation-inclusive.
- $S_T$ is correlation-exclusive.
- There is no fitted surrogate between the model and the estimator.
- The design costs $N(2D+1)$ model evaluations.

For one share per input, draw the design with `subsets="all"` and call
`result.shapley()` or `result.pme()`. See
[Shapley effects and PME](/api/kucherenko#shapley-effects-and-pme).
The design costs $N\,2^D$ runs and needs $D \le 12$.

- Shapley effects ([Owen, 2014](#references)) split the output variance among
  the inputs. A correlated input that the model never reads still gets a share,
  because it carries information about an input that the model reads.
- PME ([Herin et al., 2024](#references)) gives zero to an input that the model
  does not use, even when that input is correlated with one the model uses.
- Both sum to 1 and are nonnegative in expectation. Choose Shapley to credit
  correlation. Choose PME to find the inputs the model uses.

See the [Kucherenko example](/examples/kucherenko) and the broader
[correlated-inputs example](/examples/correlated-inputs).

### VKOGA

Use VKOGA when you have existing runs or want to reuse the same fitted model
under several input-distribution assumptions. It fits a kernel surrogate, then
samples that surrogate to estimate dependent-input variance indices. Declare
the integration distribution on the `Problem`, with a Gaussian-copula matrix or
the same custom sampler used by Kucherenko. Neither is inferred from the
training rows. The declared `Problem` marginals must still match the custom
distribution. The training data must cover the region the analysis puts its
mass on.

This surrogate workflow follows [Hilhorst et al. (2024)](/api/vkoga#references).

Its two main decision measures match the Kucherenko quantities:

- $S_{TC}$ is the correlation-inclusive prioritisation measure.
- $S_{TU}$ is the correlation-exclusive fixing measure.

VKOGA also reports $S_U$, $S_C$, and $S_{IU}$ to separate uncorrelated,
correlated, and independent-interaction contributions. This full and
independent index family follows [Mara, Tarantola & Annoni (2015)](#references). Because these values
come from a surrogate, their quality depends on the fit.

See the [VKOGA example](/examples/vkoga).

### HDMR

Use HDMR when you care about *where* variance appears in the fitted model. Its
ANCOVA decomposition follows [Li et al. (2010)](/api/hdmr#references) and
separates every component function into:

- $S_a$: the structural contribution that would remain without correlation;
- $S_b$: the contribution driven by correlation.

This split is per term, so it can describe both main effects and interactions.
That is different from the conditional-variance question answered by
Kucherenko and VKOGA.

HDMR's $S_T$ needs special care under dependence. It sums every structural and
correlative term containing the parameter. It is not the expected residual
variance after fixing that parameter, so do not use it as a fixing measure.
Use Kucherenko $S_T$ or VKOGA $S_{TU}$ for that question.

See the [HDMR example](/examples/hdmr).

### Shapley with the HDMR backend

If you can run the model, prefer the exact route above. Use
`shapley.analyze(backend="hdmr", include_correlative=True)` when you only have
existing $(X, Y)$ data and want one allocation per parameter from the HDMR
decomposition. It splits each term's
$S_a + S_b$ among the parameters in that term, and the allocations sum to 1.

This is an ANCOVA-based attribution. It is not the conditional-variance
Shapley effect of Song et al. (2016), and its correlative shares can be
negative. The Kucherenko route computes that conditional-variance Shapley
effect. It should not be interpreted as the same quantity as Kucherenko or
VKOGA.

For the conditional-variance definition under dependence, see
[Song, Nelson & Staum (2016)](https://doi.org/10.1137/15M1048070).
The [Shapley API references](/api/shapley#references) cover the independent-input
allocation and the ANCOVA source used by jaxgsa.

See the [Shapley example](/examples/shapley).

## What a custom sampler must do

A `subsets="all"` design asks the sampler for any subset of the inputs. The
default design asks only for one input, or for all inputs but one. The
`InputSampler` contract is now stronger. For any sorted set `fixed_indices` of
inputs, the sampler must keep those columns exactly. It must draw the other
columns from their joint law given the fixed ones.

The sampler must be a deterministic inverse Rosenblatt map
([Rosenblatt, 1952](#references)). This is a conditional-quantile map. It
turns the uniform points $u$ into the free columns by inverting the
conditional distribution function given the fixed columns. In the chain-rule
form, the first free column uses $F_1^{-1}(u_1)$. The next uses
$F_2^{-1}(u_2 \mid x_1)$, and so on. Two rules follow.

- Do not use rejection sampling or MCMC. Their output is not a deterministic
  function of the uniform points. The package then cannot control the
  randomness, and the Sobol' balance is lost.
- Do not use hidden random state. The same arguments must give the same rows.

The built-in Gaussian copula meets this contract. Its conditional law is
Gaussian in latent space, so the map is closed form. This is the Nataf
transform ([Lebrun & Dutfoy, 2009](#references)): a Gaussian copula with the
declared marginals. See also
[supplying your own input distribution](/guide/methods#supplying-your-own-input-distribution).

## Distribution-based alternatives

Variance may be the wrong summary when the output is skewed, heavy-tailed,
multimodal, or otherwise changes shape. The following given-data methods do
not assume independent inputs:

| Method | Use it when |
| --- | --- |
| [HSIC](/api/hsic) | You want a kernel dependence measure and permutation significance tests |
| [PAWN](/api/pawn) | You want a CDF-based, moment-independent measure |
| [Borgonovo delta](/api/borgonovo) | You want a density-based measure on a fixed $[0, 1]$ scale |
| [Optimal transport](/api/optimal-transport) | You want a Wasserstein measure split into mean-shift and shape-change components |

These methods are correlation-inclusive: an input can score highly because it
contains information about another input that drives the output. That is a
meaningful result, not evidence that the model uses the input directly.

Worked examples are available for [HSIC](/examples/hsic),
[PAWN](/examples/pawn), [Borgonovo delta](/examples/borgonovo), and
[optimal transport](/examples/optimal-transport).

## Comparisons to avoid

Keep these boundaries clear when interpreting results:

- Kucherenko and VKOGA estimate the same pair of conditional-variance
  quantities. They should agree up to surrogate and numerical integration error.
- HDMR and HDMR-backed Shapley estimate different quantities. They do not need
  to agree with Kucherenko or VKOGA.
- HDMR's $S_T$ under dependence is not a conditional-variance total-effect
  index.
- HDMR-backed Shapley is not conditional-variance Shapley. Kucherenko Shapley
  is.
- None of these dependent-input indices should be placed beside
  `jaxgsa.sobol` as if they had the same meaning. Sobol' analysis assumes
  independent inputs and refuses a problem with a declared correlation or a
  custom sampler.

## A practical workflow

1. Declare the marginal distributions on `Problem`. For Kucherenko or VKOGA,
   specify dependence through either a Gaussian-copula correlation matrix or
   a custom sampler implementing joint and conditional transformations. For
   given-data methods, use rows representative of the target joint distribution.
2. Decide whether your question is correlation-inclusive,
   correlation-exclusive, term-level, allocative, or distribution-based.
3. Decide whether you can run a dedicated design or must use existing data.
4. Select the method from the table above.
5. If the method uses a surrogate, check its fit before interpreting the
   indices.
6. Report the method and the meaning of its index, not only the numeric value.

For setup, code, and a side-by-side comparison, continue to
[Correlated Inputs](/examples/correlated-inputs).

## References

- Owen, A. B. (2014).
  [Sobol' indices and Shapley value](https://doi.org/10.1137/130936233).
  *SIAM/ASA Journal on Uncertainty Quantification*, 2(1), 245–251.
- Song, E., Nelson, B. L., & Staum, J. (2016).
  [Shapley effects for global sensitivity analysis: theory and computation](https://doi.org/10.1137/15M1048070).
  *SIAM/ASA Journal on Uncertainty Quantification*, 4(1), 1060–1083.
- Herin, M., Il Idrissi, M., Chabridon, V., & Iooss, B. (2024).
  [Proportional marginal effects for global sensitivity analysis](https://doi.org/10.1137/22M153032X).
  *SIAM/ASA Journal on Uncertainty Quantification*, 12(2), 667–692.
- Rosenblatt, M. (1952).
  [Remarks on a multivariate transformation](https://doi.org/10.1214/aoms/1177729394).
  *Annals of Mathematical Statistics*, 23(3), 470–472.
- Lebrun, R., & Dutfoy, A. (2009).
  [An innovating analysis of the Nataf transformation from the copula viewpoint](https://doi.org/10.1016/j.probengmech.2008.08.001).
  *Probabilistic Engineering Mechanics*, 24(3), 312–320.
- Mara, T. A., Tarantola, S., & Annoni, P. (2015).
  [Non-parametric methods for global sensitivity analysis of model output with dependent inputs](https://doi.org/10.1016/j.envsoft.2015.07.010).
  *Environmental Modelling & Software*, 72, 173–183.
