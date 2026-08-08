# Optimal Transport

`jaxgsa.optimal_transport.analyze()` computes Wasserstein-based sensitivity
indices. Each index measures how far fixing an input moves the full output
distribution. Each index splits into an advective (mean-shift) component and a
diffusive (spread and shape) component. The method works on arbitrary
`(X, Y)` pairs and needs no structured design. It supports mixed marginals and
correlated inputs.

The canonical API reference lives at [API Reference](/api/).

Jump directly to:

- [`jaxgsa.sampling.monte_carlo()`](/api/#given-data-methods)
- [`jaxgsa.optimal_transport.analyze()`](/api/#given-data-methods)
- [`jaxgsa.optimal_transport.OTResult`](/api/#given-data-methods)

Related docs:

- [Optimal Transport Example](/examples/optimal-transport)
- [Methods](/guide/methods)
