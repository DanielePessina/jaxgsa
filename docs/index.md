---
layout: home

hero:
  name: jaxgsa
  text: Global Sensitivity Analysis in JAX
  tagline: GPU-accelerated Sobol indices, RS-HDMR, PCE, Shapley effects, eFAST, DGSM, Morris screening, HSIC, PAWN, Borgonovo delta, optimal transport, and VKOGA with JIT compilation, vectorized bootstrap, and multi-output support.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/problem
features:
  - title: Sobol Indices
    details: The standard variance-based method — first-order (S1), total-order (ST), and second-order (S2) shares of output variance. Pick it when you can choose where to evaluate your model.
  - title: RS-HDMR
    details: Fits a spline surrogate to any existing (X, Y) data and reads Sobol-compatible indices off it, with a built-in emulator. Pick it when your model runs already exist and a structured design isn't an option.
  - title: PCE
    details: Fits an orthogonal-polynomial surrogate and computes Sobol indices exactly from its coefficients. Very sample-efficient for smooth models.
  - title: Shapley Effects
    details: Splits output variance fairly among inputs — each interaction shared equally by its participants — so the effects sum to exactly 1. Computed analytically from a fitted PCE (default) or RS-HDMR surrogate, with no extra model runs.
  - title: eFAST
    details: Estimates S1 and ST from the Fourier spectrum of the output along sinusoidal search curves. A classic alternative to Saltelli sampling that needs only a plain N × D design.
  - title: DGSM
    details: Bounds the total Sobol index from model derivatives, obtained through JAX autodiff. A cheap screening option when your model is differentiable.
  - title: Morris Screening
    details: Coarse one-at-a-time screening that ranks inputs (mu_star) and flags nonlinearity or interactions (sigma) from very few model runs. Use it to discard unimportant inputs before a full Sobol study.
  - title: HSIC
    details: Kernel-based measure that detects any statistical dependence — nonlinear, non-monotone, or heteroscedastic — from any (X, Y) pairs, with permutation p-values. Works even with correlated inputs.
  - title: PAWN
    details: Measures how much the whole output distribution (its CDF) shifts when an input is fixed, capturing effects that variance-based indices miss. Useful when tails and extremes matter, not just spread.
  - title: Borgonovo Delta
    details: Measures the shift of the entire output density when an input is fixed — moment-independent like PAWN, but density-based. The given-data estimator works on any (X, Y) pairs and includes bootstrap confidence intervals.
  - title: VKOGA
    details: Variance-based indices that stay meaningful under correlated inputs. Fits a greedy kernel surrogate to any (X, Y) pairs and splits each input's effect into a correlated and an uncorrelated part under a Gaussian copula.
  - title: Save & Reload Samples
    details: Persist a sample set with SobolSamples.save(), evaluate your model elsewhere, and reload with SobolSamples.load() — the analysis metadata travels with the samples.
  - title: Multi-Output & Time-Series
    details: Pass scalar, (N, K), or (N, T, K) outputs to any of the twelve methods and get indices for every output and timestep in a single vectorized pass. Set output_names and jaxgsa disambiguates 2-D layouts, fixing obvious transposes with a warning.
  - title: Up to 668× Faster than SALib
    details: Fused JIT kernels and vectorized execution replace Python loops. Sobol up to 15.8× faster, HDMR up to 668× on multi-output workloads.
---

`jaxgsa`'s Sobol sampling and analysis workflow is heavily drawn from [SALib](https://salib.readthedocs.io/), adapted here into a JAX-first implementation focused on JIT compilation, accelerator execution, and multi-output workloads.
