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
- `streamed` — `True` when the fit read the data in row batches, `False` when
  it did the whole fit in one pass.

## The two fit paths

The fit has two paths. The one-pass path builds the full design matrix. The
streamed path reads the rows in batches, so it holds much less memory. Both
paths solve the same equations and report the same numbers. They differ only in
the order of the float32 sums.

The streamed path starts when you give `batch_size` an integer, or when jaxgsa
estimates that the one-pass fit would go over the memory budget. Set that
budget with `jaxgsa.config.set_memory_budget`.

Read `streamed` when a fit takes much longer than you expect. `True` means the
memory budget engaged.

The estimate of the one-pass memory charged for one array more than the fit
needs. Version 0.9 corrects it, so the point at which the streamed path starts
has moved: some fits that streamed before now run in one pass. The results do
not change.

See the [PCE example](/examples/pce) and [API overview](/api/).
