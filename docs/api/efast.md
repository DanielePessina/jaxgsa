# eFAST

eFAST reads first- and total-order sensitivity indices off the Fourier
spectrum of the model output. Instead of a random design it sweeps one
sinusoidal search curve per parameter. On parameter `i`'s curve, parameter `i`
oscillates at the high frequency `omega_0` while every other parameter
oscillates slowly. Power at `omega_0` and its first `M` harmonics is `i`'s
own variance contribution; power below `omega_0 / 2` belongs to the others.

It gives `S1` and `ST` from `n_per_curve * D` model runs, and no `S2`.

## sample

```python
sample(
    problem: Problem,
    n_per_curve: int,
    *,
    M: int = 4,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> EFASTSamples
```

`n_per_curve` is the number of points along each curve, and there is one curve
per parameter, so the total cost is `n_per_curve * D` runs. It must satisfy
`n_per_curve >= 4*M^2*(D-1) + 1`, so that all `D` parameters can get distinct
frequencies. Raising it raises `omega_0 = (n_per_curve - 1) // (2M)`, which
pushes the focal parameter's harmonics further from the complementary
frequencies and sharpens the indices, at proportionally more model runs.

`M` is the interference factor: how many harmonics of `omega_0` are credited
to the focal parameter during analysis. The default 4 is the standard choice
and rarely needs changing. It travels inside `EFASTSamples`, so `analyze`
cannot use a different one than `sample` did.

`seed` sets the random phase shift of each curve, which is the only randomness
in the design. There is no `scramble` keyword here.

`sample` raises on a correlated problem, and on a categorical one, because a
search curve sweeps each input continuously and an unordered level code has no
continuous sweep.

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=1024, seed=0)
```

```
jaxgsa.efast.sample: D=3, n_curves=3, n_per_curve=1024, n_runs=3072, M=4, omega_0=127
```

## EFASTSamples

- `samples` — `(n_per_curve * D, D)` in physical units. Rows
  `i*n_per_curve : (i+1)*n_per_curve` are the search curve for parameter `i`.
- `n_per_curve`, `M`, `problem` — the design metadata that `analyze` reads
  back.
- `n_runs` — `n_per_curve * D`. A search curve is an ordered sweep, so nothing
  is deduplicated and `n_runs` is exact.

### save and load

New in 1.0. `EFASTSamples` now persists like every other design class:

```python
samples.save("run.npz")
back = jaxgsa.efast.EFASTSamples.load("run.npz")
back.n_per_curve, back.M            # 1024, 4
np.array_equal(back.samples, samples.samples)   # True
```

One compressed NPZ holds the sample matrix plus a JSON metadata blob with the
problem definition, `n_per_curve`, `M`, and the jaxgsa version that wrote it.
eFAST does no deduplication, so there is no expansion map to store. A path
without an `.npz` suffix gets one appended, matching NumPy's `savez`
convention, so `save("run.A")` writes `run.A.npz` and `load("run.A")` reads it
back.

## analyze

```python
analyze(
    sampling_result: EFASTSamples,
    Y: Array,
    *,
    slice_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
) -> EFASTResult
```

`Y` holds the model output at each row of `sampling_result.samples`, in that
row order, as `(n_runs,)`, `(n_runs, K)` or `(n_runs, T, K)`.

```python
result = jaxgsa.efast.analyze(samples, ishigami(samples.samples), verbose=False)

result.S1       # Array([0.3077, 0.4414, 0.    ], dtype=float32)
result.ST       # Array([0.5507, 0.4625, 0.2393], dtype=float32)
result.omega_0  # 127
result.M        # 4
```

Ishigami's analytical answer is `S1 = [0.3139, 0.4424, 0.0]` and
`ST = [0.5576, 0.4424, 0.2437]`, from 3072 runs here.

`slice_chunk_size` caps how many `(T, K)` output slices go into one vmapped
batch. It bounds peak device memory when `T * K` is large, and trades speed
for memory. `None` derives a width from the memory budget. It changes no index
beyond float summation order.

`on_invalid` accepts `"raise"` (the default), `"propagate"` and `"none"`
here. `"drop"` raises a `ValueError`. Dropping a point from a search curve
does not shrink the sample, it changes what the discrete Fourier transform
computes,
so there is no honest way to analyze the survivors. The non-finite report
still names the curve to investigate.

`analyze` also runs a per-curve zero-variance check, on top of the global one.
A parameter with no effect at all gives `V = 0` on its own curve while the
other curves vary, which the global check cannot see and which would otherwise
produce a silent NaN.

## EFASTResult

| Field | Meaning |
| --- | --- |
| `S1`, `ST` | `(D,)` / `(K, D)` / `(T, K, D)`, mirroring `Y` |
| `omega_0` | the focal frequency the design used |
| `M` | the interference factor |
| `problem`, `invalid` | the problem, and the non-finite report |

There is no `ci`, no `*_conf`, and no `n_bootstrap`. That follows from the
design rather than from missing work. The resampling unit would be the search
curve, and the design holds exactly one curve per parameter, so there is
nothing to resample. For an interval, draw several designs at different random
phase shifts and compare the indices across them. That is a change to
`sample`, not a keyword on `analyze`.

## indices

```python
indices(sampling_result, Y, *, slice_chunk_size=None) -> tuple[Array, Array]
```

`S1` and `ST` as plain arrays, with no checks and no result object, so it runs
inside `jax.jit`, `jax.vmap` and `jax.jacrev`.

Related docs:

- [eFAST Example](/examples/efast)
- [Methods](/guide/methods)
- [API reference](/api/)
