---
layout: home

hero:
  name: gsax
  text: Global Sensitivity Analysis in JAX
  tagline: GPU-accelerated Sobol indices, RS-HDMR, and PCE with JIT compilation, vectorized bootstrap, and multi-output support.
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
  - title: Save & Reload Samples
    details: Persist unique sample matrices plus Saltelli reconstruction metadata with `SamplingResult.save()` and reload them later with `gsax.load()`.
  - title: Multi-Output & Time-Series
    details: Pass scalar, (N, K), or (N, T, K) outputs. All indices are computed in a single vectorized pass.
  - title: Up to 929× Faster than SALib
    details: Fused JIT kernels and vectorized execution replace Python loops. 4.7× faster even on scalar outputs, up to 929× on multi-output workloads.
---

`gsax`'s Sobol sampling and analysis workflow is heavily drawn from [SALib](https://salib.readthedocs.io/), adapted here into a JAX-first implementation focused on JIT compilation, accelerator execution, and multi-output workloads.
