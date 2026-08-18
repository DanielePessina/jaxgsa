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
- `streamed` — `True` when the fit read the data in row batches, `False` when
  it held the full basis matrices in memory.

## The two fit paths

The fit has two paths. The in-memory path builds the full B-spline basis
matrices. The streamed path reads the rows in batches, so it holds much less
memory. Both paths fit the same components, keep the same terms after the
F-test, and report the same indices. They differ only in the order of the
float32 sums.

The streamed path starts when you give `batch_size` an integer, or when jaxgsa
estimates that the in-memory fit would go over the memory budget. Set that
budget with `jaxgsa.config.set_memory_budget`.

Read `streamed` when a fit takes much longer than you expect. `True` means the
memory budget engaged.

Note that the PCE memory estimate changed in version 0.9, so a PCE fit can now
start streaming at a different size than before. The HDMR estimate is
unchanged.

See the [HDMR example](/examples/hdmr) and [API overview](/api/).
