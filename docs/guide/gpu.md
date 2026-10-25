# Sobol indices on GPU

The same jaxgsa code runs on a CPU, a GPU or a TPU. JAX picks the device. You
change the JAX installation, not your code.

## Install JAX with GPU support

Follow the JAX installation guide for your hardware. For an NVIDIA GPU, the
command is usually:

```bash
uv add "jax[cuda13]"
```

Check the extra name against the JAX guide, because it changes between JAX
releases. Then check the device:

```python
import jax

print(jax.devices())
```

If the list shows a CUDA device, JAX will use it.

## Run the analysis

The code is the same as on CPU. This example runs on whichever device JAX
found, including a CPU-only machine:

```python
import jax
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

print("device:", jax.devices()[0].platform)

design = jaxgsa.sobol.sample(PROBLEM, n_samples=1024, seed=0, verbose=False)
Y = evaluate(design.samples)
result = jaxgsa.sobol.analyze(design, Y, verbose=False)
print("S1:", result.S1)
```

## What to know about GPUs

- Sampling runs on the host. The index computation runs on the device.
- Float64 is slow on consumer GPUs. TPUs have no float64. See
  [configuration](/guide/configuration).
- A large design can exceed GPU memory. See
  [scaling to large problems](/guide/scale).

## Performance

The published [benchmarks](/guide/benchmarks) are CPU only. They ran on an
Apple M1 Pro in float32. jaxgsa publishes no GPU timings. Measure your own
workload before you plan around a speed-up.
