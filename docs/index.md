---
layout: home

hero:
  name: jaxgsa
  text: Global Sensitivity Analysis in JAX
  tagline: Thirteen global sensitivity analysis methods in JAX, JIT-compiled and vectorized, for scalar, multi-output, and time-series models.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/problem
features:
  - title: Sobol Indices
    details: The standard variance-based method. It splits output variance into first-order (S1), total-order (ST), and second-order (S2) shares. Pick it when you can choose where to evaluate your model.
  - title: RS-HDMR
    details: Fits a spline surrogate to any existing (X, Y) data, then reads Sobol-compatible indices off it and keeps the surrogate as an emulator. Pick it when your model runs already exist and a structured design is not an option.
  - title: PCE
    details: Fits an orthogonal-polynomial surrogate and computes Sobol indices exactly from its coefficients. Pick it for smooth models, where it needs few samples.
  - title: Shapley Effects
    details: Shares each interaction equally among the inputs that take part in it, so the effects sum to exactly 1. jaxgsa computes them analytically from a fitted PCE (default) or RS-HDMR surrogate, with no extra model runs.
  - title: eFAST
    details: Estimates S1 and ST from the Fourier spectrum of the output along sinusoidal search curves. Pick it instead of Saltelli sampling when you want a plain N × D design.
  - title: DGSM
    details: Bounds the total Sobol index from model derivatives, obtained through JAX autodiff. A cheap screening option when your model is differentiable.
  - title: Morris Screening
    details: Changes one input at a time to rank inputs (mu_star) and flag nonlinearity or interactions (sigma). It needs very few model runs, so use it to drop unimportant inputs before a full Sobol study.
  - title: HSIC
    details: Kernel measure that detects any statistical dependence, including nonlinear, non-monotone, and heteroscedastic effects. It runs on any (X, Y) pairs, reports permutation p-values, and works with correlated inputs.
  - title: PAWN
    details: Measures how much the whole output distribution (its CDF) shifts when an input is fixed, so it catches effects that variance-based indices miss. Pick it when tails and extremes matter, not just spread.
  - title: Borgonovo Delta
    details: Measures how much the whole output density shifts when an input is fixed. Like PAWN it is moment-independent, but it works on the density instead of the CDF. The given-data estimator runs on any (X, Y) pairs and reports bootstrap confidence intervals.
  - title: VKOGA
    details: Fits a greedy kernel surrogate to any (X, Y) pairs, then splits each input's effect into a correlated and an uncorrelated part under a Gaussian copula. Pick it when your inputs are correlated and you still want variance-based indices.
  - title: Kucherenko Indices
    details: Sobol indices for dependent inputs, with no surrogate. It reads the dependence from problem.correlation and evaluates your actual model on a conditional-copula design. With independent inputs it reduces to the classic Saltelli scheme.
  - title: Save & Reload Samples
    details: Save a sample set with SobolSamples.save(), evaluate your model elsewhere, then reload it with SobolSamples.load(). The analysis metadata travels with the samples.
  - title: Multi-Output & Time-Series
    details: Pass scalar, (N, K), or (N, T, K) outputs to any of the thirteen methods. One vectorized pass returns indices for every output and timestep. Set output_names and jaxgsa reads 2-D layouts correctly, fixing obvious transposes with a warning.
  - title: Up to 668× Faster than SALib
    details: Fused JIT kernels and vectorized execution replace Python loops. Sobol up to 15.8× faster, HDMR up to 668× on multi-output workloads.
---

## Method capabilities

The methods do not all accept the same problems. Four of the thirteen build
their own sampling design, and the rest work on $(X, Y)$ pairs you already
have. Some accept correlated parameters, some accept categorical parameters,
and five report bootstrap confidence intervals.

The [method capability table](/guide/methods#method-capabilities) records all
of that, one row per method. A method that does not accept your problem raises
a `ValueError` that names the parameters and the alternatives. It never
returns a silent approximation.

`jaxgsa`'s Sobol sampling and analysis workflow draws heavily on [SALib](https://salib.readthedocs.io/). jaxgsa adapts it into a JAX-first implementation focused on JIT compilation, accelerator execution, and multi-output workloads.
