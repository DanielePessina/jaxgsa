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

| Method | Reports | Own design | Categorical | Correlated |
|---|---|:--:|:--:|:--:|
| **Variance-based** | | | | |
| [`sobol`](/guide/methods#sobol-indices-via-saltelli-sampling) | $S_1$, $S_2$, $S_T$ | ✓ | ✓ | ✗ |
| [`efast`](/guide/methods#efast-extended-fourier-amplitude-sensitivity-test) | $S_1$, $S_T$ | ✓ | ✗ | ✗ |
| [`kucherenko`](/guide/methods#kucherenko-dependent-input-sobol-indices) | $S_1$, $S_T$ under dependence | ✓ | ✗ | ✓ |
| [`pce`](/guide/methods#pce-polynomial-chaos-expansion) | $S_1$, $S_2$, $S_T$, surrogate | — | ✗ | ✗ |
| [`hdmr`](/guide/methods#rs-hdmr-random-sampling-high-dimensional-model-representation) | $S_a$ / $S_b$ / $S$ per term, surrogate | — | ✗ | ✓ † |
| [`shapley`](/guide/methods#shapley-effects) | allocation summing to 1 | — | ✗ | ✓ ‡ |
| [`vkoga`](/guide/methods#vkoga-correlated-input-variance-indices) | $S_{TC}$, $S_{TU}$, $S_U$, $S_C$, $S_{IU}$, surrogate | — | ✗ | ✓ |
| **Screening** | | | | |
| [`morris`](/guide/methods#morris-elementary-effects-screening) | $\mu^*$, $\sigma$ | ✓ | ✗ | ✗ |
| [`dgsm`](/guide/methods#dgsm-derivative-based-global-sensitivity-measures) | bounds on $S_T$ | ✓ | ✗ | ✗ |
| **Moment-independent** | | | | |
| [`borgonovo`](/guide/methods#borgonovo-delta-density-based-sensitivity) | $\delta$, $S_1$ | — | ✓ | ✓ § |
| [`optimal_transport`](/guide/methods#optimal-transport-wasserstein-based-sensitivity) | $W_2^2$ index, advective + diffusive | — | ✓ | ✓ § |
| [`pawn`](/guide/methods#pawn-cdf-based-sensitivity) | KS distance | — | ✓ | ✓ § |
| [`hsic`](/guide/methods#hsic-hilbert–schmidt-independence-criterion) | dependence measure | — | ✗ | ✓ § |

**Own design** means the method builds its own sample matrix, so you must be able to run the model at points it chooses. The rest are given-data methods: they accept any $(X, Y)$ pairs you already have.

A ✗ is a refusal, not a silent approximation. The method raises a `ValueError` that names the parameters and the alternatives.

† HDMR's per-term structural ($S_a$) and correlative ($S_b$) split is valid under dependence, but its $S_T$ is the SCSA convention and not a total-effect index. See the [HDMR section](/guide/methods#rs-hdmr-random-sampling-high-dimensional-model-representation).

‡ Requires `backend="hdmr"`. The PCE backend assumes independent inputs and refuses.

§ Correlation-inclusive: an input that does not enter the model but correlates with one that does scores non-zero. That is the correct reading of these indices, not an error.

`jaxgsa`'s Sobol sampling and analysis workflow draws heavily on [SALib](https://salib.readthedocs.io/). jaxgsa adapts it into a JAX-first implementation focused on JIT compilation, accelerator execution, and multi-output workloads.
