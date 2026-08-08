# Save and Reload Sobol Samples

By the end of this page you will have written one Sobol design to disk, read it
back in a later session, and run the analysis on it. This is what you need when
the model evaluation happens somewhere else: on a cluster, in another language,
or days after the samples were drawn.

Sobol analysis is a two-part job. First `jaxgsa.sobol.sample()` builds a
structured set of input rows called a Saltelli design. Then
`jaxgsa.sobol.analyze()` reads the model outputs for those rows in exactly that
structure. The row order and the design settings are therefore part of the
result, not incidental. `SobolSamples` stores both the unique model-evaluation
rows and the metadata needed to reconstruct the internal Saltelli design, so a
saved file carries everything the analysis step needs.

The example runs in four steps.

1. Draw the design. `sample()` returns the unique rows to evaluate.
2. Save it, before running the model. The design is the thing you cannot
   recreate by accident: a different seed or a different `n_samples` gives
   rows that no longer match your outputs.
3. Reload it in the later session, using the same path.
4. Evaluate and analyze. Pass the restored object, not a fresh one, so the
   internal design matches the row order of `Y`.

```python
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

samples = jaxgsa.sobol.sample(PROBLEM, n_samples=4096, seed=42)
samples.save("runs/ishigami")

restored = jaxgsa.sobol.SobolSamples.load("runs/ishigami")
Y = my_model(restored.samples)
result = jaxgsa.sobol.analyze(restored, Y)
```

The `.npz` suffix is optional. Both calls above address
`runs/ishigami.npz`.

`restored` is a `SobolSamples` equal in content to `samples`, so `result` is
the same as if the two halves had run in one process. The reload works because
one compressed file contains:

- the unique sample matrix;
- stable sample identifiers;
- the expanded-to-unique Saltelli mapping;
- the input problem and output names;
- sampling settings such as `base_n` and second-order mode.

The last three entries are why reloading is enough on its own. You do not have
to remember the seed, the second-order setting, or the problem definition, and
you cannot silently pair the outputs with a design that was built differently.

Use NumPy directly when a separate CSV or table is needed:

```python
import numpy as np

np.savetxt("runs/ishigami.csv", samples.samples, delimiter=",")
```

That call writes the sample matrix and nothing else: one row per model
evaluation, one column per input parameter, and no header. It is the right
format to hand to a solver that reads plain text. The CSV is only a
model-input table; keep the NPZ file for later jaxgsa analysis.
