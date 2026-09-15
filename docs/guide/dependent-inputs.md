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
| How can I allocate the fitted variance to one number per input? | Shapley with the HDMR backend |
| Does an input change the output distribution, rather than only its variance? | HSIC, PAWN, Borgonovo delta, or optimal transport |

Then choose between a dedicated design and existing data:

- If you can run the model at new points, use **Kucherenko** for
  conditional-variance indices without a surrogate.
- If you already have $(X, Y)$ data, use **VKOGA** for conditional-variance
  indices, **HDMR** for a term-level ANCOVA split, or a distribution-based
  method when variance is not the quantity you care about.
- If you need one allocation per parameter, use **Shapley with the HDMR
  backend**, but read its interpretation carefully.

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
jaxgsa provides four routes, each with a different estimand.

| Route | What it estimates | What it needs |
| --- | --- | --- |
| [Kucherenko](/api/kucherenko) | $S_1 = V(\mathbb{E}(Y \mid X_i))/V(Y)$ and $S_T = \mathbb{E}(V(Y \mid \mathbf{X}_{\sim i}))/V(Y)$ under the declared copula | A dedicated design, $N(2D+1)$ model runs, and a declared correlation matrix |
| [VKOGA](/api/vkoga) | The same conditional-variance quantities as $S_{TC}$ and $S_{TU}$, plus $S_U$, $S_C$, and $S_{IU}$ | Existing $(X, Y)$ pairs and a declared correlation matrix |
| [HDMR](/api/hdmr) | For each fitted component, a structural share $S_a$ and a correlation-driven share $S_b$ | Existing $(X, Y)$ pairs; dependence is inferred from $X$ |
| [Shapley with HDMR](/api/shapley) | One ANCOVA allocation per parameter, formed by splitting each term's $S_a + S_b$ among its participants | Existing $(X, Y)$ pairs |

### Kucherenko

Use Kucherenko when you can still evaluate the model and want the
conditional-variance quantities directly. It samples conditionally on the
correlation declared on the `Problem`.

- $S_1$ is correlation-inclusive.
- $S_T$ is correlation-exclusive.
- There is no fitted surrogate between the model and the estimator.
- The design costs $N(2D+1)$ model evaluations.

See the [Kucherenko example](/examples/kucherenko) and the broader
[correlated-inputs example](/examples/correlated-inputs).

### VKOGA

Use VKOGA when you have existing runs or want to reuse the same fitted model
under several correlation assumptions. It fits a kernel surrogate, then
samples that surrogate to estimate correlated-input variance indices.

Its two main decision measures match the Kucherenko quantities:

- $S_{TC}$ is the correlation-inclusive prioritisation measure.
- $S_{TU}$ is the correlation-exclusive fixing measure.

VKOGA also reports $S_U$, $S_C$, and $S_{IU}$ to separate uncorrelated,
correlated, and independent-interaction contributions. Because these values
come from a surrogate, their quality depends on the fit.

See the [VKOGA example](/examples/vkoga).

### HDMR

Use HDMR when you care about *where* variance appears in the fitted model. Its
ANCOVA decomposition separates every component function into:

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

Use
`shapley.analyze(backend="hdmr", include_correlative=True)` when you want one
allocation per parameter from the HDMR decomposition. It splits each term's
$S_a + S_b$ among the parameters in that term, and the allocations sum to 1.

This is an ANCOVA-based attribution. It is not the conditional-variance
Shapley effect of Song et al. (2016), and its correlative shares can be
negative. It should not be interpreted as the same quantity as Kucherenko or
VKOGA.

See the [Shapley example](/examples/shapley).

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
  quantities. They should agree up to surrogate and Monte Carlo error.
- HDMR and HDMR-backed Shapley estimate different quantities. They do not need
  to agree with Kucherenko or VKOGA.
- HDMR's $S_T$ under dependence is not a conditional-variance total-effect
  index.
- HDMR-backed Shapley is not conditional-variance Shapley.
- None of these dependent-input indices should be placed beside
  `jaxgsa.sobol` as if they had the same meaning. Sobol' analysis assumes
  independent inputs and refuses a problem with declared correlation.

## A practical workflow

1. Declare the input distributions and their correlation on the `Problem`.
2. Decide whether your question is correlation-inclusive,
   correlation-exclusive, term-level, allocative, or distribution-based.
3. Decide whether you can run a dedicated design or must use existing data.
4. Select the method from the table above.
5. If the method uses a surrogate, check its fit before interpreting the
   indices.
6. Report the method and the meaning of its index, not only the numeric value.

For setup, code, and a side-by-side comparison, continue to
[Correlated Inputs](/examples/correlated-inputs).
