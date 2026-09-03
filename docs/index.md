---
layout: home

hero:
  name: jaxgsa
  text: Global Sensitivity Analysis in JAX
  tagline: Thirteen methods, one interface, eleven of them JIT-compiled. Scalar, multi-output, and time-series models.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Choose a Method
      link: /guide/methods#choosing-a-method
    - theme: alt
      text: API Reference
      link: /api/problem
features:
  - title: Thirteen methods, one interface
    details: Every method has analyze(). The design-based ones also have sample(). Results share one output contract and convert to labeled xarray with to_dataset().
  - title: Bring your own design, or your old runs
    details: Sobol, eFAST, Morris, and Kucherenko build the sample matrix for you. The other nine read indices off any (X, Y) pairs you already have, including a sweep you ran last year.
  - title: Time series in one pass
    details: Pass Y as (N,), (N, K), or (N, T, K). One compiled call returns indices for every timestep and output, so you can watch a ranking change along the trajectory instead of averaging it away.
  - title: Gradients are cheap
    details: DGSM bounds the total Sobol index from model derivatives, taken by JAX autodiff. It picks forward or reverse mode from the output shape, so time-series models do not pay T reverse passes.
  - title: Correlated inputs
    details: Declare a Gaussian-copula matrix on the Problem. VKOGA and Kucherenko return variance-based indices under that dependence, split into correlated and uncorrelated parts.
  - title: Categorical inputs
    details: Declare unordered levels with probs and labels. Sobol, Borgonovo delta, optimal transport, and PAWN handle them. The methods whose indices would depend on code order refuse instead.
  - title: It refuses rather than approximates
    details: A method that cannot handle your problem raises a ValueError naming the parameters and the alternatives. A zero-variance output slice returns NaN with a warning that names the slice.
  - title: Fast where output size is large
    details: Vectorized estimators replace SALib's per-slice Python loop. RS-HDMR on 50 timesteps by 6 outputs runs 1060x faster than single-process SALib on one M1 Pro core, and 10.9x faster on a single output slice. On a scalar output there is no gain.
---

## Which method should I use?

The methods measure different quantities, cost different numbers of model
runs, and do not all accept the same problems. Four build their own sampling
design, and the other nine work on $(X, Y)$ pairs you already have.

Start at [Choosing a Method](/guide/methods#choosing-a-method). It walks three
questions: can you still choose where to run the model, what should the number
mean, and what is your evaluation budget.

For a high-dimensional study, read [Scale and limits](/guide/scale) before
choosing a design. It records the main cost drivers and the settings that move
each method's practical limit.

The [method capability table](/guide/methods#method-capabilities) is the one
place that records which methods accept correlated parameters, which accept
categorical parameters, and which report bootstrap confidence intervals.
`tests/test_docs_matrix.py` checks those three columns, plus Own design,
against the code.

## A note on the speed numbers

The gain is vectorization over output slices, so it scales with $T \times K$
and vanishes at $T \times K = 1$. Any speedup quoted without those two numbers
says nothing. The [benchmarks guide](/guide/benchmarks) gives the full tables,
the hardware, and the baseline.

jaxgsa's Sobol sampling and analysis workflow follows
[SALib](https://salib.readthedocs.io/), reimplemented for JAX.
