"""Generate ``colab_gpu.ipynb``, the GPU-against-CPU benchmark notebook.

The notebook is self-contained: it installs JAX with CUDA and jaxgsa, then
times the analysis step of several jaxgsa methods on the GPU and on the CPU of
the same Colab session. It imports nothing from ``scripts/``.

Run with::

    uv run --with nbformat scripts/benchmarks/build_colab_gpu.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).with_name("colab_gpu.ipynb")

INTRO = """\
# jaxgsa: GPU against CPU

This notebook measures how long jaxgsa takes to **analyze** model outputs on
an NVIDIA GPU and on the CPU of the same Colab session. It does not time model
evaluation or sampling.

The method follows the [benchmark guide](https://danielepessina.github.io/jaxgsa/guide/benchmarks):

- Only the `analyze` call is timed.
- The first call (tracing and compilation) is reported separately.
- The warm time is the median of `N_WARM` later calls.
- Every call blocks until all result arrays are ready.
- Verbose reports and expected `JaxgsaWarning`s are off in the timed region.
- Arithmetic is float32, as in the published CPU benchmarks.

**How to run**

1. *Runtime > Change runtime type > T4 GPU* (any NVIDIA GPU works).
2. *Runtime > Run all*.

The install cell picks the CUDA 12 or CUDA 13 JAX wheels from the driver
version. The full run takes about 5 minutes on a free T4, install included. At the end,
the notebook prints a Markdown table and `results.json`. Copy both back.
"""

PARAMS = """\
# Parameters. Change JAXGSA_REF to a release tag (for example "v1.3.0") to test a release.
JAXGSA_REF = "feat/1.3-marginals"
QUICK = False  # True shrinks every workload (smoke test, not for publication)
REQUIRE_GPU = True  # False lets the notebook run on CPU only (local testing)
N_WARM = 9  # warm calls per workload and device; the median is reported
SEED = 20260819
"""

INSTALL = """\
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules


def _driver_major() -> int | None:
    \"\"\"Return the NVIDIA driver major version, or None without a driver.\"\"\"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(out.strip().splitlines()[0].split(".")[0])


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


DRIVER_MAJOR = _driver_major()
# JAX CUDA 13 wheels need driver >= 580; CUDA 12 wheels need >= 525.
CUDA_EXTRA = "cuda13" if (DRIVER_MAJOR or 0) >= 580 else "cuda12"
if IN_COLAB:
    print(f"NVIDIA driver major: {DRIVER_MAJOR}; installing jax[{CUDA_EXTRA}]")
    _pip(f"jax[{CUDA_EXTRA}]>=0.11.2")
    _pip(f"git+https://github.com/DanielePessina/jaxgsa.git@{JAXGSA_REF}")
else:
    print("Not on Colab: using the jax and jaxgsa already installed.")
"""

ENV = """\
import dataclasses
import importlib.metadata
import json
import platform
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa import JaxgsaWarning, borgonovo, hdmr, morris, optimal_transport, pce, sobol
from jaxgsa.benchmarks import ishigami, sobol_g


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "none"
    return out.strip()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


CPU = jax.devices("cpu")[0]
try:
    GPUS = jax.devices("gpu")
except RuntimeError:
    GPUS = []
if not GPUS and REQUIRE_GPU:
    raise RuntimeError(
        "No GPU device found by JAX. Select Runtime > Change runtime type > T4 GPU, "
        "then Runtime > Restart and run all. (Set REQUIRE_GPU = False to run on CPU only.)"
    )
DEVICES = [CPU, *GPUS[:1]]

ENVIRONMENT = {
    "jaxgsa": _version("jaxgsa"),
    "jax": jax.__version__,
    "jaxlib": _version("jaxlib"),
    "jax_cuda_plugin": _version(f"jax-{CUDA_EXTRA}-plugin"),
    "numpy": np.__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_model": _cpu_model(),
    "cpu_count": __import__("os").cpu_count(),
    "gpu": _gpu_name(),
    "jax_devices": [str(d) for d in jax.devices()],
    "cpu_devices": [str(d) for d in jax.devices("cpu")],
    "benchmark_devices": [str(d) for d in DEVICES],
    "x64_enabled": bool(getattr(jax.config, "jax_enable_x64")),
    "jaxgsa_ref": JAXGSA_REF,
    "quick": QUICK,
    "n_warm": N_WARM,
    "seed": SEED,
}
for key, value in ENVIRONMENT.items():
    print(f"{key:>18}: {value}")
assert not ENVIRONMENT["x64_enabled"], "These benchmarks are float32; leave jax_enable_x64 off."
"""

HELPERS = """\
def result_arrays(value, depth: int = 0) -> list:
    \"\"\"Collect every JAX array reachable from a jaxgsa result object.\"\"\"
    if depth > 6:
        return []
    if isinstance(value, jax.Array):
        return [value]
    if isinstance(value, jaxgsa.Problem | str | bytes):
        return []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = []
        for field in dataclasses.fields(value):
            out.extend(result_arrays(getattr(value, field.name, None), depth + 1))
        return out
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list | tuple):
        out = []
        for item in value:
            out.extend(result_arrays(item, depth + 1))
        return out
    return []


def timed(call) -> tuple[float, list]:
    \"\"\"Run ``call`` once; return wall seconds and the result arrays (all blocked on).\"\"\"
    start = time.perf_counter()
    arrays = result_arrays(call())
    jax.block_until_ready(arrays)
    return time.perf_counter() - start, arrays


def widen(y: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    \"\"\"Widen a scalar output to ``(N, *dims)``; every slice is a distinct affine copy.\"\"\"
    if not dims:
        return y
    n = int(np.prod(dims))
    scales = 1.0 + 0.5 * np.arange(n, dtype=y.dtype) / n
    offsets = np.arange(n, dtype=y.dtype)
    return (y[:, None] * scales[None, :] + offsets[None, :]).reshape((y.shape[0], *dims))


def host_eval(model, X) -> np.ndarray:
    \"\"\"Evaluate a jaxgsa benchmark model on the CPU and return a float32 NumPy array.\"\"\"
    with jax.default_device(CPU):
        return np.asarray(model(jnp.asarray(X, dtype=jnp.float32)), dtype=np.float32)


def uniform_problem(d: int) -> jaxgsa.Problem:
    return jaxgsa.Problem.from_dict({f"x{i + 1}": (0.0, 1.0) for i in range(d)})


def sobol_g_model(d: int):
    a = tuple(0.5 * i for i in range(d))
    return lambda x: sobol_g.evaluate(x, a=a)


def shape_label(n_rows: int, d: int, dims: tuple[int, ...]) -> str:
    out = "(N" + "".join(f", {k}" for k in dims) + ")"
    return f"D={d}, rows={n_rows}, Y={out}"


@dataclasses.dataclass
class Workload:
    \"\"\"One analysis call. ``bind(device)`` moves the host data and returns the timed call.\"\"\"

    name: str
    shape: str
    n_slices: int
    bind: object


RECORDS: list[dict] = []


def bench(work: Workload, group: str = "workloads") -> None:
    \"\"\"Time ``work`` on every benchmark device and append one record per device.\"\"\"
    for dev in DEVICES:
        with jax.default_device(dev), warnings.catch_warnings():
            warnings.simplefilter("ignore", JaxgsaWarning)
            call = work.bind(dev)
            first, arrays = timed(call)
            warm = [timed(call)[0] for _ in range(N_WARM)]
        placed = sorted({str(d) for a in arrays for d in a.devices()})
        rec = {
            "group": group,
            "workload": work.name,
            "shape": work.shape,
            "n_slices": work.n_slices,
            "device": dev.platform,
            "device_name": str(dev),
            "result_on": placed,
            "first_call_s": first,
            "warm_median_s": float(np.median(warm)),
            "warm_min_s": float(np.min(warm)),
            "warm_max_s": float(np.max(warm)),
        }
        RECORDS.append(rec)
        print(
            f"{work.name:<28} {dev.platform:>4}  first {first * 1e3:9.1f} ms   "
            f"warm median {rec['warm_median_s'] * 1e3:9.2f} ms   result on {placed}"
        )
"""

WORKLOADS = """\
def q(full: int, quick: int) -> int:
    return quick if QUICK else full


def sobol_workload(name, d, dims, n=None, base_n=None, s2=False, boot=0):
    problem = ishigami.PROBLEM if d == 3 else uniform_problem(d)
    model = ishigami.evaluate if d == 3 else sobol_g_model(d)
    sr = sobol.sample(
        problem, n_samples=n, base_n=base_n, calc_second_order=s2, seed=SEED, verbose=False
    )
    y = widen(host_eval(model, sr.samples), dims)

    def bind(dev):
        y_dev = jax.device_put(y, dev)
        key = jax.random.key(SEED) if boot else None
        return lambda: sobol.analyze(sr, y_dev, n_bootstrap=boot, key=key, verbose=False)

    n_slices = int(np.prod(dims)) if dims else 1
    return Workload(name, shape_label(y.shape[0], d, dims), n_slices, bind)


def given_data(n, dims, d=3):
    problem = ishigami.PROBLEM if d == 3 else uniform_problem(d)
    model = ishigami.evaluate if d == 3 else sobol_g_model(d)
    X = np.asarray(jaxgsa.sampling.monte_carlo(problem, n, seed=SEED), dtype=np.float32)
    Y = widen(host_eval(model, X), dims)
    return problem, X, Y


def given_workload(name, n, dims, analyze, **kwargs):
    problem, X, Y = given_data(n, dims)
    boot = kwargs.get("n_bootstrap", 0)

    def bind(dev):
        X_dev, Y_dev = jax.device_put(X, dev), jax.device_put(Y, dev)
        extra = {"key": jax.random.key(SEED)} if boot else {}
        return lambda: analyze(problem, X_dev, Y_dev, verbose=False, **kwargs, **extra)

    n_slices = int(np.prod(dims)) if dims else 1
    return Workload(name, shape_label(n, 3, dims), n_slices, bind)


def morris_workload(name, d, n_traj):
    problem = ishigami.PROBLEM if d == 3 else uniform_problem(d)
    model = ishigami.evaluate if d == 3 else sobol_g_model(d)
    sr = morris.sample(problem, n_trajectories=n_traj, seed=SEED, verbose=False)
    y = host_eval(model, sr.samples)

    def bind(dev):
        y_dev = jax.device_put(y, dev)
        return lambda: morris.analyze(sr, y_dev, verbose=False)

    return Workload(name, shape_label(y.shape[0], d, ()), 1, bind)


# Shapes follow docs/guide/benchmarks.md where a published CPU row exists.
WORKLOADS = [
    lambda: sobol_workload("sobol small", 3, (), n=q(1024, 256)),
    lambda: sobol_workload("sobol high N", 3, (), n=q(65536, 2048)),
    lambda: sobol_workload("sobol D=30 with S2", 30, (), base_n=q(8192, 256), s2=True),
    lambda: sobol_workload("sobol time series", 3, (50, 6), n=q(8192, 512)),
    lambda: sobol_workload("sobol bootstrap", 3, (32,), n=q(4096, 512), boot=q(100, 10)),
    lambda: given_workload(
        "hdmr multi-output", q(1024, 256), (50, 6), hdmr.analyze, order=2, max_iter=50
    ),
    lambda: given_workload("pce scalar", q(1024, 256), (), pce.analyze, order=3),
    lambda: given_workload("pce multi-output", q(1024, 256), (50, 6), pce.analyze, order=3),
    lambda: given_workload(
        "borgonovo multi-output", q(1024, 256), (32,), borgonovo.analyze, n_bootstrap=q(10, 2)
    ),
    lambda: given_workload(
        "optimal_transport multi-output",
        q(1024, 256),
        (32,),
        optimal_transport.analyze,
        n_classes=8,
        n_bootstrap=q(10, 2),
        dummy=False,
    ),
    lambda: morris_workload("morris small", 3, q(32, 8)),
    lambda: morris_workload("morris D=30", 30, q(40, 8)),
]

t0 = time.perf_counter()
for make in WORKLOADS:
    work = make()
    bench(work)
    del work
print(f"workloads took {time.perf_counter() - t0:.0f} s")
"""

SCALING_MD = """\
## Scaling with the number of output slices

Sobol first- and total-order indices, Ishigami ($D=3$), `n_samples=32768`
(40,960 rows), no $S_2$, no bootstrap. Only the number of output slices
changes. One call analyzes all slices, so this is where a GPU should help.
"""

SCALING = """\
SLICE_COUNTS = [1, 10, 100, 1000] if QUICK else [1, 10, 100, 1000, 3000]
N_SCALING = q(32768, 512)
_sr = sobol.sample(
    ishigami.PROBLEM, n_samples=N_SCALING, calc_second_order=False, seed=SEED, verbose=False
)
_y0 = host_eval(ishigami.evaluate, _sr.samples)


def scaling_workload(n_slices: int) -> Workload:
    y = widen(_y0, (n_slices,)) if n_slices > 1 else _y0

    def bind(dev):
        y_dev = jax.device_put(y, dev)
        return lambda: sobol.analyze(_sr, y_dev, verbose=False)

    dims = (n_slices,) if n_slices > 1 else ()
    return Workload(f"sobol {n_slices} slices", shape_label(y.shape[0], 3, dims), n_slices, bind)


t0 = time.perf_counter()
for k in SLICE_COUNTS:
    bench(scaling_workload(k), group="scaling")
print(f"scaling took {time.perf_counter() - t0:.0f} s")
"""

RESULTS = """\
import pandas as pd

df = pd.DataFrame(RECORDS)
wide = df.pivot_table(
    index=["group", "workload", "shape"],
    columns="device",
    values=["first_call_s", "warm_median_s"],
    sort=False,
)
wide.columns = [f"{dev}_{metric}" for metric, dev in wide.columns]
wide = wide.reset_index()
if "gpu_warm_median_s" in wide:
    wide["speedup_cpu_over_gpu"] = wide["cpu_warm_median_s"] / wide["gpu_warm_median_s"]
display(wide) if "display" in globals() else print(wide.to_string())


def fmt(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "n/a"
    return f"{seconds * 1e3:.2f} ms" if seconds < 1.0 else f"{seconds:.2f} s"


has_gpu = "gpu_warm_median_s" in wide
head = "| Workload | Shape | CPU first | CPU warm |"
rule = "| --- | --- | ---: | ---: |"
if has_gpu:
    head += " GPU first | GPU warm | CPU/GPU (warm) |"
    rule += " ---: | ---: | ---: |"
lines = [head, rule]
for _, row in wide.iterrows():
    line = (
        f"| {row['workload']} | {row['shape']} | {fmt(row['cpu_first_call_s'])} "
        f"| {fmt(row['cpu_warm_median_s'])} |"
    )
    if has_gpu:
        line += (
            f" {fmt(row['gpu_first_call_s'])} | {fmt(row['gpu_warm_median_s'])} "
            f"| {row['speedup_cpu_over_gpu']:.2f}x |"
        )
    lines.append(line)
MARKDOWN = "\\n".join(lines)
env = ENVIRONMENT
print(f"GPU: {env['gpu']}; CPU: {env['cpu_model']} ({env['cpu_count']} threads)")
print(f"jax {env['jax']}, jaxgsa {env['jaxgsa']}, float32, median of {N_WARM} warm calls")
print("CPU/GPU above 1 means the GPU was faster.\\n")
print(MARKDOWN)
"""

SAVE = """\
payload = {"environment": ENVIRONMENT, "records": RECORDS, "markdown": MARKDOWN}
with open("results.json", "w") as fh:
    json.dump(payload, fh, indent=2)
print(json.dumps(payload, indent=2))
"""


def build() -> nbformat.NotebookNode:
    """Assemble the notebook.

    Returns:
        The notebook object, with a GPU accelerator hint for Colab.
    """
    cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PARAMS),
        new_markdown_cell("## Install\n\nOn Colab only. Locally the cell does nothing."),
        new_code_cell(INSTALL),
        new_markdown_cell("## Environment"),
        new_code_cell(ENV),
        new_markdown_cell("## Timing helpers"),
        new_code_cell(HELPERS),
        new_markdown_cell(
            "## Workloads\n\nHost data is made once with NumPy. Each device gets its own "
            "copy through `jax.device_put`, and the call runs under "
            "`jax.default_device`. The `result on` column confirms where the result arrays live."
        ),
        new_code_cell(WORKLOADS),
        new_markdown_cell(SCALING_MD),
        new_code_cell(SCALING),
        new_markdown_cell("## Results\n\nCopy the Markdown table and the JSON below."),
        new_code_cell(RESULTS),
        new_code_cell(SAVE),
    ]
    for cell in cells:
        cell.source = cell.source.rstrip("\n")
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    """Write the notebook next to this script and validate it."""
    nb = build()
    nbformat.validate(nb)
    OUT.write_text(nbformat.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
