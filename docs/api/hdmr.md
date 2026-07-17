# HDMR

`gsax.hdmr.analyze(problem, X, Y, ...)` fits an RS-HDMR surrogate and returns
`HDMRResult`.

Important result operations:

- `result.predict(X_new, batch_size=None)`
- `result.shapley(include_correlative=False)`
- `result.to_dataset(time_coords=None)`

The result exposes ANCOVA term arrays `Sa`, `Sb`, and `S`, parameter totals
`ST`, structural interaction properties `S1`, `S2`, and `S3`, fit selection
counts, and RMSE.

See the [HDMR example](/examples/hdmr) and [API overview](/api/).
