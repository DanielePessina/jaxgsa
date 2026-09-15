---
layout: home

hero:
  name: jaxgsa
  text: Global Sensitivity Analysis in JAX
  tagline: Thirteen methods for screening, variance decomposition, and distribution-based sensitivity—with a common JAX interface.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Choose a Method
      link: /guide/methods
    - theme: alt
      text: API Reference
      link: /api/problem
features:
  - title: Choose among thirteen methods
    details: Compare the question each method answers, its sampling cost, and its support for dependent or categorical inputs before running it.
  - title: Reuse data or build a design
    details: Four methods create dedicated sample designs. Nine work with ordinary input samples, including simulation runs you already have.
  - title: Analyze every output together
    details: Use scalar, multi-output, or time-series model outputs. The same result contract carries parameter and output labels into xarray.
  - title: Differentiate the analysis
    details: Eleven methods expose JAX-transformable estimator cores. DGSM separately uses automatic differentiation of the model to construct derivative-based measures.
  - title: Work with dependent inputs
    details: Kucherenko, VKOGA, HDMR, and distribution-based methods offer distinct dependence-aware quantities. The guide explains which question each one answers.
  - title: Scale with JAX
    details: Vectorized estimator cores avoid per-output Python loops and can be compiled for repeated analyses. Reproducible benchmarks document where this matters.
---

## Which method should I use?

The methods measure different quantities, cost different numbers of model
runs, and do not all accept the same problems. Four build their own sampling
design, and the other nine work on $(X, Y)$ pairs you already have.

Start at [Choosing a method](/guide/methods). It walks through three
questions: can you still choose where to run the model, what should the number
mean, and what is your evaluation budget.

For a high-dimensional study, read [Scaling to large problems](/guide/scale) before
choosing a design. It records the main cost drivers and the settings that move
each method's practical limit.

The [method capability table](/guide/methods#method-capabilities) records which
methods accept dependent or categorical parameters and which report bootstrap
confidence intervals.

## Performance

Performance depends on the method, sample count, parameter count, and output
shape. The [benchmarks guide](/guide/benchmarks) reports complete workloads,
hardware, and baselines rather than a single headline number.

jaxgsa's Sobol sampling and analysis workflow follows
[SALib](https://salib.readthedocs.io/), reimplemented for JAX.
