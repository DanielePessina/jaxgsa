# HSIC

`jaxgsa.hsic.analyze()` computes kernel-based dependence indices with
permutation p-values. These indices detect input influence beyond variance. It
works on arbitrary `(X, Y)` pairs.

The canonical API reference lives at [API Reference](/api/).

## Memory

Each kernel matrix is `N x N`. The `batch_size` option builds a matrix in row
blocks, then joins the blocks. It bounds the working memory of the build. It
does **not** bound the kernel matrix, because the result is one full `N x N`
array in every case. Peak memory therefore stays of order `N^2`. Reduce `N` if
memory is the limit.

Jump directly to:

- [`jaxgsa.sampling.monte_carlo()`](/api/#given-data-methods)
- [`jaxgsa.hsic.analyze()`](/api/#given-data-methods)
- [`jaxgsa.hsic.HSICResult`](/api/#given-data-methods)

Related docs:

- [HSIC Example](/examples/hsic)
- [Methods](/guide/methods)
