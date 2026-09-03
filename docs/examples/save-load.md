# Save and reload a design

Sampling and analysis are two separate jobs, and nothing says they have to
happen in the same process. You draw a design, hand the rows to a solver that
runs for two days on a cluster, and come back later with a column of outputs.
By then the Python process that drew the rows is long gone.

Every design object in jaxgsa writes itself to one compressed NPZ file, and
reads back as an object that analyzes to the same numbers. This page shows the
round trip, checks it digit for digit, and points at the one mistake the file
cannot catch for you.

## Draw the design and save it

`save()` writes a file. It does not create the parent directory, so make it
first. This holds for all four design classes. Save into a directory that does
not exist and `save()` checks the parent up front and tells you exactly what
is missing:

```
FileNotFoundError: cannot save to runs/sobol.npz: the directory runs does not exist. Make it first, for example with Path("runs").mkdir(parents=True, exist_ok=True), or save to a directory that is already there.
```

Every path in this page writes into `runs/`. Delete that directory when you are
done; nothing in jaxgsa reads it back except the code below.

```python
from pathlib import Path

import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

Path("runs").mkdir(exist_ok=True)

samples = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42)
samples.save("runs/ishigami")

print("rows to evaluate:", samples.n_runs)
print("file size (bytes):", Path("runs/ishigami.npz").stat().st_size)
```

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=512, requested_runs>=4096, n_runs=4096, n_expanded=4096, duplicates_removed=0 (0.0%), scramble=True
rows to evaluate: 4096
file size (bytes): 68501
```

The first line is the sampler's verbose summary, on by default in 1.0. Read
`n_runs=4096` as the number of times your model has to run. 67 KB buys you the
right to lose the process.

The `.npz` suffix is added when you leave it off, so `"runs/ishigami"` and
`"runs/ishigami.npz"` name the same file. Save before you run the model, not
after. A design you have evaluated but not stored is a set of outputs with no
inputs attached.

## Reload and analyze

In the later session, load the file and pass the loaded object to `analyze`.

```python
import numpy as np

import jaxgsa
from jaxgsa.benchmarks.ishigami import evaluate

restored = jaxgsa.sobol.SobolSamples.load("runs/ishigami")
Y = evaluate(restored.samples)
result = jaxgsa.sobol.analyze(restored, Y)

print("problem came back from the file:", restored.problem.names)
print("S1:", np.asarray(result.S1))
```

```
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=4096 runs, T=1 x K=1 output slice
    invalid: none found in 512 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.4473 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.6266
    2. x2  ST=0.44
    3. x3  ST=0.2423
problem came back from the file: ('x1', 'x2', 'x3')
S1: [0.3387249  0.4420562  0.01550918]
```

Nothing in that script mentions `PROBLEM`, the seed, or `calc_second_order`.
The file carries all three. The summary block confirms it: the problem line
reads `D=3 (x1, x2, x3)` with three uniform marginals and no correlation,
which is the problem definition travelling inside the NPZ.

The `0.4473 s` timing is machine-specific and includes the JIT compile of the
estimators, so your number will differ.

## Same design, same numbers

Worth checking once, so you can stop worrying about it.

```python
samples = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42, verbose=False)
samples.save("runs/ishigami")
restored = jaxgsa.sobol.SobolSamples.load("runs/ishigami")

Y = evaluate(samples.samples)

before = jaxgsa.sobol.analyze(samples, Y, verbose=False)
after = jaxgsa.sobol.analyze(restored, Y, verbose=False)

print("same rows:      ", np.array_equal(samples.samples, restored.samples))
print("same expansion: ", np.array_equal(samples.expanded_to_unique, restored.expanded_to_unique))
print("same problem:   ", samples.problem == restored.problem)
print("S1 before:", np.asarray(before.S1))
print("S1 after: ", np.asarray(after.S1))
print("bitwise identical S1/ST/S2:",
      np.array_equal(before.S1, after.S1),
      np.array_equal(before.ST, after.ST),
      np.array_equal(before.S2, after.S2, equal_nan=True))
```

```
same rows:       True
same expansion:  True
same problem:    True
S1 before: [0.3387249  0.4420562  0.01550918]
S1 after:  [0.3387249  0.4420562  0.01550918]
bitwise identical S1/ST/S2: True True True
```

Bitwise, not merely close. The NPZ stores the float64 sample matrix and the
integer index arrays exactly, so the reloaded design feeds the estimators the
same bits. `verbose=False` here because the interesting output is the
comparison, not two more summary blocks.

## What is in the file

```python
import json

with np.load("runs/ishigami.npz", allow_pickle=False) as data:
    print("arrays:  ", data.files)
    meta = json.loads(data["metadata"].item())

print("metadata:", sorted(meta))
print("design:  ", {k: meta[k] for k in ("base_n", "calc_second_order", "n_expanded")})
print("names:   ", meta["problem"]["names"])
print("spec[0]: ", meta["problem"]["input_specs"][0])
```

```
arrays:   ['samples', 'unit', 'expanded_to_unit', 'metadata']
metadata: ['base_n', 'calc_second_order', 'identity_mapping', 'jaxgsa_version', 'n_expanded', 'problem']
design:   {'base_n': 512, 'calc_second_order': True, 'n_expanded': 4096}
names:    ['x1', 'x2', 'x3']
spec[0]:  {'dist': 'uniform', 'low': -3.141592653589793, 'high': 3.141592653589793}
```

The arrays are the design. The `metadata` entry is a JSON string holding the
problem definition, the design settings, and the jaxgsa version that wrote the
file. `allow_pickle=False` on both the write and the read side, so the file
holds arrays and text and cannot execute anything when you open it.

`identity_mapping` is a size optimization. When the design has no duplicate
rows, `expanded_to_unique` is `arange(n_expanded)`, so the flag is stored
instead of the array and `load()` rebuilds it.

Notice what the file does **not** contain: your model outputs. Store `Y`
yourself, next to the design, with a name that ties the two together.

```python
np.savez_compressed("runs/ishigami_Y.npz", Y=np.asarray(Y))
```

## Every design class round-trips

Four samplers, four design classes, one save/load contract. `EFASTSamples`
gained `save()` and `.load()` in 1.0 and now matches the other three.

```python
designs = {
    "sobol": jaxgsa.sobol.sample(PROBLEM, n_samples=1024, seed=42, verbose=False),
    "morris": jaxgsa.morris.sample(PROBLEM, n_trajectories=64, seed=42, verbose=False),
    "efast": jaxgsa.efast.sample(PROBLEM, n_per_curve=257, seed=42, verbose=False),
    "kucherenko": jaxgsa.kucherenko.sample(PROBLEM, n_samples=1024, seed=42, verbose=False),
}

for name, design in designs.items():
    design.save(f"runs/{name}")
    back = type(design).load(f"runs/{name}")
    same = np.array_equal(design.samples, back.samples)
    print(f"{name:11s} {type(design).__name__:17s} rows={design.n_runs:5d} round-trip={same}")
```

```
sobol       SobolSamples      rows= 1024 round-trip=True
morris      MorrisSamples     rows=   64 round-trip=True
efast       EFASTSamples      rows=  771 round-trip=True
kucherenko  KucherenkoSamples rows= 7168 round-trip=True
```

Load with the class that wrote the file: `jaxgsa.sobol.SobolSamples.load`,
`jaxgsa.morris.MorrisSamples.load`, `jaxgsa.efast.EFASTSamples.load`,
`jaxgsa.kucherenko.KucherenkoSamples.load`.

The eFAST file is the simplest of the four. eFAST evaluates one search curve
per parameter and never removes duplicate rows, so its NPZ carries the sample
matrix, `n_per_curve`, `M`, and the problem, with no expansion map. Here is
the full eFAST round trip:

```python
design = jaxgsa.efast.sample(PROBLEM, n_per_curve=257, M=4, seed=0)
design.save("runs/ishigami_efast")

restored = jaxgsa.efast.EFASTSamples.load("runs/ishigami_efast")
Y = evaluate(restored.samples)
result = jaxgsa.efast.analyze(restored, Y)

print("n_per_curve:", restored.n_per_curve, " M:", restored.M)
print("S1:", np.asarray(result.S1))
```

```
jaxgsa.efast.sample: D=3, n_curves=3, n_per_curve=257, n_runs=771, M=4, omega_0=32
jaxgsa.efast.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=771 runs, T=1 x K=1 output slice
    invalid: none found in 3 search curves (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.1108 s
    slice_chunk_size: auto (resolved from the memory budget)
    omega_0: 32, M: 4
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5372
    2. x2  ST=0.4876
    3. x3  ST=0.2446
n_per_curve: 257  M: 4
S1: [0.31269768 0.44250646 0.02491138]
```

`M=4` is not a free choice on reload. The analysis credits `M` harmonics of
the drive frequency to the focal parameter, and using a different `M` than
the design was built with reads the wrong harmonics. The file stores it, which
is the point.

## The row count that surprises people

Morris in the table above shows 64 rows for 64 trajectories, and 64 * (3 + 1)
is 256. The design object explains itself:

```python
m = jaxgsa.morris.sample(PROBLEM, n_trajectories=64, seed=42)
print("unique rows to evaluate:", m.n_runs)
print("rows in the raw design: ", m.n_expanded)
print("expansion map length:   ", m.expanded_to_unique.shape)
```

```
jaxgsa.morris.sample: D=3, method=trajectory, n_trajectories=64, num_levels=4, n_expanded=256, n_runs=64, duplicates_removed=192 (75.0%)
unique rows to evaluate: 64
rows in the raw design:  256
expansion map length:    (256,)
```

Morris walks a grid of `num_levels=4` values per parameter, and in three
dimensions that grid has only 4^3 = 64 points. The 64 trajectories keep
revisiting them, so 192 of the 256 rows are repeats. jaxgsa evaluates each
distinct row once and rebuilds the full 256-row design inside `analyze`
through `expanded_to_unique`. You paid for 64 model runs instead of 256.

This is why the expansion map has to be in the file. Without it, the outputs
you computed for 64 rows cannot be put back into the 256-row order the
elementary-effect differences are taken in.

## What the file cannot save you from

Redrawing a design instead of loading one is the mistake to avoid. A size
mismatch raises:

```python
design = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42, verbose=False)
Y = evaluate(design.samples)

smaller = jaxgsa.sobol.sample(PROBLEM, n_samples=2048, seed=42, verbose=False)
try:
    jaxgsa.sobol.analyze(smaller, Y, verbose=False)
except Exception as exc:
    print(type(exc).__name__, exc)
```

```
ValueError Y has 4096 sample rows but 2048 were expected; pass Y as (N,), (N, K), or (N, T, K)
```

A same-size mismatch does not. If you redraw with a different seed and get a
design of the same shape, `analyze` accepts it without complaint, because the
Sobol estimators read `Y` and the design's structure and never look at the
sample values. Your indices then describe a pairing of outputs to input rows
that never happened. There is no check that can catch this, because there is
nothing wrong with the arrays. Load the file.

## Handing rows to another program

The design object holds a plain NumPy matrix, so exporting for a solver that
reads text is one call:

```python
np.savetxt("runs/ishigami.csv", samples.samples, delimiter=",")
```

One row per model run, one column per parameter, no header, physical units.
Keep the row order. The CSV is an input table and nothing more, so keep the
NPZ as well. Analysis needs the metadata, and a CSV does not carry it.

## See also

- [Bootstrap confidence intervals](/examples/bootstrap) to put error bars on a
  reloaded design's indices.
- [Advanced workflow](/examples/advanced-workflow) for the screen-then-quantify
  study these files are usually part of.
