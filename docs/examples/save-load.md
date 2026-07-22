# Save and Reload Sobol Samples

`SobolSamples` stores both the unique model-evaluation rows and the metadata
needed to reconstruct the internal Saltelli design.

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

One compressed file contains:

- the unique sample matrix;
- stable sample identifiers;
- the expanded-to-unique Saltelli mapping;
- the input problem and output names;
- sampling settings such as `base_n` and second-order mode.

Use NumPy directly when a separate CSV or table is needed:

```python
import numpy as np

np.savetxt("runs/ishigami.csv", samples.samples, delimiter=",")
```

The CSV is only a model-input table; keep the NPZ file for later jaxgsa analysis.
