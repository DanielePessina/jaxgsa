"""Regenerate the example figures embedded in the docs pages.

Runs every example script headless (Agg backend), intercepts ``plt.show()``
to save the current figure into ``docs/examples/figures/``, and names the
files ``{script}_{title-slug}.png``. The docs pages reference these files
with relative links, so re-run this script whenever an example's plotting
code changes.

Run: ``uv run python scripts/build_example_figures.py``
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
OUT = REPO / "docs" / "examples" / "figures"

SCRIPTS = [
    "batch_reactor_gsa.py",
    "efast_gsa.py",
    "morris_gsa.py",
    "shapley_gsa.py",
    "dgsm_benchmark.py",
    "oakley_ohagan_15d.py",
    "dynamic_gsa.py",
    "method_comparison.py",
    "benchmark_all.py",
]


def _slug(text: str) -> str:
    """Turn a figure title into a filesystem-safe name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48].rstrip("-")


def _figure_title(fig) -> str:
    """First axes title, falling back to the suptitle, then a default."""
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            return title
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        return suptitle.get_text()
    return "figure"


def _run_script(script: str) -> None:
    path = EXAMPLES / script
    module_name = f"_docs_figures_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()

    original_show = plt.show
    seen: set[str] = set()
    current_script = ""

    def save_and_show(*args, **kwargs):
        fig = plt.gcf()
        name = f"{current_script}_{_slug(_figure_title(fig))}.png"
        if name in seen:
            base, ext = name.rsplit(".", 1)
            counter = 2
            while f"{base}-{counter}.{ext}" in seen:
                counter += 1
            name = f"{base}-{counter}.{ext}"
        seen.add(name)
        fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return original_show(*args, **kwargs)

    plt.show = cast(Any, save_and_show)

    for script in SCRIPTS:
        plt.close("all")
        current_script = Path(script).stem
        _run_script(script)
        saved = sorted(p.name for p in OUT.glob(f"{current_script}_*.png"))
        print(f"{script}: {len(saved)} figures")

    plt.show = original_show


if __name__ == "__main__":
    sys.exit(main())
