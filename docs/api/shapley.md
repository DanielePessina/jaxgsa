# Shapley Effects

`jaxgsa.shapley` exposes `analyze` and the `ShapleyResult` type.

The canonical form is a result method. Fit a PCE or HDMR surrogate, then
derive Shapley effects from the fitted result:

```python
effects = jaxgsa.pce.analyze(problem, X, Y).shapley()
effects = jaxgsa.hdmr.analyze(problem, X, Y).shapley()
effects = jaxgsa.hdmr.analyze(problem, X, Y).shapley(
    include_correlative=True,
)
```

`jaxgsa.shapley.analyze(problem, X, Y, backend="pce" | "hdmr", ...)` is a thin
convenience over the same two steps. It is literally
`jaxgsa.pce.analyze(problem, X, Y, **kw).shapley()`, or the HDMR equivalent
with `include_correlative=...`. There is no separate Shapley pipeline behind
it. Extra keyword arguments pass through to the selected backend's `analyze`,
such as `order` for PCE and `maxorder` for HDMR:

```python
effects = jaxgsa.shapley.analyze(problem, X, Y, backend="pce", order=4)
effects = jaxgsa.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True
)
```

Prefer the result-method form when you also want the fitted surrogate for
prediction, Sobol-style indices, or fit diagnostics. Use the wrapper when only
the Shapley effects are needed.

Result fields:

- `Sh` — the Shapley effect per parameter.
- `S1`, `ST` — the first-order and total indices.
- `explained_variance` — the share of output variance the fit explains.
- Fit provenance.
- `to_dataset(...)` — labeled xarray view of the indices.

See the [Shapley example](/examples/shapley) and [API overview](/api/).
