# Shapley Effects

`gsax.shapley` exposes the `ShapleyResult` type. Effects are derived from a
fitted surrogate:

```python
effects = gsax.pce.analyze(problem, X, Y).shapley()
effects = gsax.hdmr.analyze(problem, X, Y).shapley()
effects = gsax.hdmr.analyze(problem, X, Y).shapley(
    include_correlative=True,
)
```

The result exposes `Sh`, `S1`, `ST`, `explained_variance`, fit provenance, and
`to_dataset(...)`.

See the [Shapley example](/examples/shapley) and [API overview](/api/).
