# Optimal Transport

Wasserstein-based sensitivity indices measuring how far fixing an input
moves the full output distribution, decomposed into advective
(mean-shift) and diffusive (spread/shape) components. Works on arbitrary
`(X, Y)` pairs; no structured design, mixed marginals and correlated
inputs supported.

The canonical API reference now lives at [API Reference](/api/).

Jump directly to:

- [`sample_mc()`](/api/#sample-mc)
- [`analyze_optimal_transport()`](/api/#analyze-optimal-transport)
- [`OTResult`](/api/#otresult)
- [`OTResult.to_dataset()`](/api/#otresult-to_dataset)

Related docs:

- [Optimal Transport Example](/examples/optimal-transport)
- [Methods](/guide/methods)
