# PCE

`jaxgsa.pce.analyze(problem, X, Y, ...)` fits a polynomial chaos expansion and
returns `PCEResult`.

Result operations:

- `result.predict(X_new, batch_size=None)` — evaluate the fitted expansion on
  new inputs.
- `result.shapley()` — derive Shapley effects from the fit.
- `result.to_dataset(time_coords=None)` — labeled xarray view of the indices.

Result fields:

- `S1`, `ST`, `S2` — the first-order, total, and second-order indices.
- The fitted coefficients, the multi-index, and the effective order.
- The leave-one-out RMSE and the explained variance.

See the [PCE example](/examples/pce) and [API overview](/api/).
