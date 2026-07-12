---
layout: home

hero:
  name: gsax
  text: Global Sensitivity Analysis in JAX
  tagline: GPU-accelerated Sobol indices, RS-HDMR, PCE, Shapley effects, eFAST, DGSM, Morris screening, HSIC, PAWN, and Borgonovo delta with JIT compilation, vectorized bootstrap, and multi-output support.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/problem

features:
  - title: Sobol Indices
    details: First-order, total-order, and second-order indices via Saltelli sampling with Sobol quasi-random sequences.
  - title: RS-HDMR
    details: Surrogate-based sensitivity analysis that works with any (X, Y) pairs. Includes a built-in emulator for prediction.
  - title: PCE
    details: Polynomial Chaos Expansion with analytical Sobol indices from expansion coefficients. Wiener-Askey optimal basis for uniform and Gaussian inputs.
  - title: Shapley Effects
    details: Fair, game-theoretic allocation of output variance — each interaction's variance split equally among its participants. Computed analytically from a fitted PCE (default) or RS-HDMR surrogate, with Sh summing to 1 alongside S1 and ST from the same fit.
  - title: eFAST
    details: Extended Fourier Amplitude Sensitivity Test. Frequency-based S1 and ST via sinusoidal search curves. Supports scalar, multi-output, and time-series outputs.
  - title: DGSM
    details: Derivative-based Global Sensitivity Measures via JAX autodiff. Computes Poincare upper bounds and Kucherenko-Song lower bounds on total Sobol index ST.
  - title: Morris Screening
    details: Globalized one-at-a-time elementary-effects screening — mu_star importance ranking and sigma interaction flag at r * (D + 1) cost. Trajectory and radial designs with unique-row deduplication and bootstrap confidence intervals.
  - title: HSIC
    details: Kernel-based dependence via the Hilbert–Schmidt Independence Criterion. Detects nonlinear, non-monotone, and heteroscedastic dependence with permutation p-values; works with any (X, Y) pairs.
  - title: PAWN
    details: Moment-independent, CDF-based sensitivity via Kolmogorov–Smirnov distances between the unconditional and conditional output distributions.
  - title: Borgonovo Delta
    details: Moment-independent, density-based importance measure (Borgonovo, 2007). The Plischke et al. (2013) given-data estimator returns the delta index plus the given-data first-order Sobol S1 from any (X, Y) pairs, with bootstrap bias correction and confidence intervals.
  - title: Save & Reload Samples
    details: Persist unique sample matrices plus Saltelli reconstruction metadata with `SamplingResult.save()` and reload them later with `gsax.load()`.
  - title: Multi-Output & Time-Series
    details: Pass scalar, (N, K), or (N, T, K) outputs. All indices are computed in a single vectorized pass.
  - title: Up to 668× Faster than SALib
    details: Fused JIT kernels and vectorized execution replace Python loops. Sobol up to 15.8× faster, HDMR up to 668× on multi-output workloads.
---

`gsax`'s Sobol sampling and analysis workflow is heavily drawn from [SALib](https://salib.readthedocs.io/), adapted here into a JAX-first implementation focused on JIT compilation, accelerator execution, and multi-output workloads.
