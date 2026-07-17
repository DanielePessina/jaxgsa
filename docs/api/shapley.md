# Shapley Effects

`gsax.shapley` exposes `analyze` and the `ShapleyResult` type.

The canonical form is a result method: fit a PCE or HDMR surrogate, then
derive Shapley effects from the fitted result:

```python
effects = gsax.pce.analyze(problem, X, Y).shapley()
effects = gsax.hdmr.analyze(problem, X, Y).shapley()
effects = gsax.hdmr.analyze(problem, X, Y).shapley(
    include_correlative=True,
)
```

`gsax.shapley.analyze(problem, X, Y, backend="pce" | "hdmr", ...)` is a thin
convenience over the same two steps — it is literally
`gsax.pce.analyze(problem, X, Y, **kw).shapley()` (or the HDMR equivalent
with `include_correlative=...`); there is no separate Shapley pipeline.
Extra keyword arguments pass through to the selected backend's `analyze`
(e.g. `order` for PCE, `maxorder` for HDMR):

```python
effects = gsax.shapley.analyze(problem, X, Y, backend="pce", order=4)
effects = gsax.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True
)
```

Prefer the result-method form when you also want the fitted surrogate
(prediction, Sobol-style indices, fit diagnostics); use the wrapper when only
the Shapley effects are needed.

The result exposes `Sh`, `S1`, `ST`, `explained_variance`, fit provenance, and
`to_dataset(...)`.

See the [Shapley example](/examples/shapley) and [API overview](/api/).
