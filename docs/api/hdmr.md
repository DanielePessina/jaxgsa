# HDMR

`jaxgsa.hdmr.analyze(problem, X, Y, ...)` fits an RS-HDMR surrogate and returns
`HDMRResult`.

Result operations:

- `result.predict(X_new, batch_size=None)` — evaluate the fitted surrogate on
  new inputs.
- `result.shapley(include_correlative=False)` — derive Shapley effects from
  the fit.
- `result.to_dataset(time_coords=None)` — labeled xarray view of the indices.

Result fields:

- `Sa`, `Sb`, `S` — the ANCOVA term arrays.
- `ST` — the total index per parameter.
- `S1`, `S2`, `S3` — the structural first-, second-, and third-order indices.
- Fit selection counts and RMSE.

See the [HDMR example](/examples/hdmr) and [API overview](/api/).
