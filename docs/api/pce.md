# PCE

`jaxgsa.pce.analyze(problem, X, Y, ...)` fits a polynomial chaos expansion and
returns `PCEResult`.

Important result operations:

- `result.predict(X_new, batch_size=None)`
- `result.shapley()`
- `result.to_dataset(time_coords=None)`

The result exposes `S1`, `ST`, `S2`, fitted coefficients, the multi-index,
effective order, leave-one-out RMSE, and explained variance.

See the [PCE example](/examples/pce) and [API overview](/api/).
