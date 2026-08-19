# HSIC

`jaxgsa.hsic.analyze()` computes kernel-based dependence indices with
permutation p-values. These indices detect input influence beyond variance. It
works on arbitrary `(X, Y)` pairs.

The canonical API reference lives at [API Reference](/api/).

## Memory

Each kernel matrix is one full `N x N` array, and about `2D + 1` of them are
resident at once. No option bounds this: peak memory is of order `N^2` in
every case. Reduce `N` if memory is the limit, or screen with a cheaper
method first.

Jump directly to:

- [`jaxgsa.sampling.monte_carlo()`](/api/#given-data-methods)
- [`jaxgsa.hsic.analyze()`](/api/#given-data-methods)
- [`jaxgsa.hsic.HSICResult`](/api/#given-data-methods)

Related docs:

- [HSIC Example](/examples/hsic)
- [Methods](/guide/methods)
