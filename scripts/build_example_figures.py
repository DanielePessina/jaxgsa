"""Regenerate the example figures embedded in the docs pages.

Runs every example script headless (Agg backend), intercepts ``plt.show()``
to save the current figure into ``docs/examples/figures/``, and names the
files ``{script}_{title-slug}.png``. Applies ``examples/styles/paper.mplstyle``
and matching palettes, minor ticks, and grids to every exported figure.
The inverse-bounds example saves directly to its stable documentation filename.
Re-run this script whenever an example's plotting code or shared style changes.

Run every example: ``uv run scripts/build_example_figures.py``
Run selected examples: ``uv run scripts/build_example_figures.py benchmark_all.py``
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import shutil
import sys
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
OUT = REPO / "docs" / "examples" / "figures"
STYLE = EXAMPLES / "styles" / "paper.mplstyle"

# Match explicit legacy palettes to the shared cycle, retaining tier shades
# and the alpha channel used by confidence bands. Neutral references stay neutral.
COLOURS = {
    "#1f77b4": "#1b4f9c",
    "#ff7f0e": "#c85200",
    "#2ca02c": "#117a3d",
    "#d62728": "#a52c3c",
    "#9467bd": "#7570b3",
    "#e377c2": "#c04f87",
    "#7f7f7f": "#666666",
    "#bcbd22": "#9a861b",
    "#17becf": "#17868b",
    "#2196f3": "#1b4f9c",
    "#4caf50": "#117a3d",
    "#ff9800": "#c85200",
    "#e91e63": "#c04f87",
    "#1e88e5": "#1b4f9c",
    "#43a047": "#117a3d",
    "#e53935": "#a52c3c",
    "#fb8c00": "#c85200",
    "#42a5f5": "#5c85bd",
    "#90caf9": "#afc4df",
}
RGBA_COLOURS = {to_rgba(old)[:3]: to_rgba(new)[:3] for old, new in COLOURS.items()}

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
    "inverse_sobol_bounds.py",
]


def _recolour(colour):
    """Map a legacy colour to the paper palette, preserving transparency."""
    if isinstance(colour, str) and colour.lower() in {"auto", "none"}:
        return colour
    rgba = to_rgba(colour)
    return (*RGBA_COLOURS.get(rgba[:3], rgba[:3]), rgba[3])


def _style_figure(fig: Figure) -> None:
    """Apply paper colours and numeric-axis grids to an existing figure.

    Categorical bar axes, heatmaps, and colourbars do not receive minor grids.
    Continuous colourmaps and neutral analytical references remain unchanged.
    """
    for artist in fig.findobj():
        if isinstance(artist, Line2D):
            artist.set_color(_recolour(artist.get_color()))
            artist.set_markerfacecolor(_recolour(artist.get_markerfacecolor()))
            artist.set_markeredgecolor(_recolour(artist.get_markeredgecolor()))
        elif isinstance(artist, Patch):
            artist.set_facecolor(_recolour(artist.get_facecolor()))
            artist.set_edgecolor(_recolour(artist.get_edgecolor()))
        elif isinstance(artist, Collection) and artist.get_array() is None:
            for getter, setter in (
                (artist.get_facecolor, artist.set_facecolor),
                (artist.get_edgecolor, artist.set_edgecolor),
            ):
                colours = getter()
                if len(colours):
                    setter([_recolour(colour) for colour in colours])

    for ax in fig.axes:
        # Monospace titles are wider than the former default font. Wrap them
        # within each panel so adjacent axes and colourbars cannot cover them.
        panel_width = fig.get_size_inches()[0] * ax.get_position().width * 72
        title_width = max(20, int(panel_width / (float(ax.title.get_fontsize()) * 0.65)))
        ax.set_title(
            "\n".join(
                textwrap.fill(
                    line, width=title_width, break_long_words=False, break_on_hyphens=False
                )
                for line in ax.get_title().splitlines()
            )
        )
        ax.grid(False, which="both")
        if ax.images or ax.get_label() == "<colorbar>":
            ax.minorticks_off()
            continue
        categorical = {
            "x" if container.orientation == "vertical" else "y"
            for container in ax.containers
            if isinstance(container, BarContainer)
        }
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            if name in categorical:
                axis.set_minor_locator(NullLocator())
                continue
            if axis.get_scale() == "log":
                axis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=200))
            elif axis.get_scale() == "linear":
                axis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(which="major", axis=name, alpha=0.28, linewidth=0.5)
            ax.grid(which="minor", axis=name, alpha=0.10, linewidth=0.35)
        ax.set_axisbelow(True)
    fig.tight_layout()


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
    original_argv = sys.argv
    try:
        with TemporaryDirectory(prefix="jaxgsa-doc-figures-") as temporary:
            sys.argv = [str(path)]
            if script == "inverse_sobol_bounds.py":
                sys.argv.extend(["--output-dir", temporary])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                spec.loader.exec_module(module)
                entrypoint = getattr(module, "main", None)
                if callable(entrypoint):
                    entrypoint()
            if script == "inverse_sobol_bounds.py":
                name = "inverse_sobol_bounds.png"
                shutil.copy2(Path(temporary) / name, OUT / name)
    finally:
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scripts",
        nargs="*",
        choices=SCRIPTS,
        help="example scripts to render; omit to render every configured script",
    )
    args = parser.parse_args()
    scripts = args.scripts or SCRIPTS

    OUT.mkdir(parents=True, exist_ok=True)
    original_show = plt.show
    seen: set[str] = set()
    current_script = ""

    def save_and_show(*args, **kwargs):
        fig = plt.gcf()
        _style_figure(fig)
        name = f"{current_script}_{_slug(_figure_title(fig))}.png"
        if name in seen:
            base, ext = name.rsplit(".", 1)
            counter = 2
            while f"{base}-{counter}.{ext}" in seen:
                counter += 1
            name = f"{base}-{counter}.{ext}"
        seen.add(name)
        fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return original_show(*args, **kwargs)

    plt.show = cast(Any, save_and_show)

    try:
        for script in scripts:
            plt.close("all")
            current_script = Path(script).stem
            with plt.style.context(str(STYLE)):
                _run_script(script)
            pattern = f"{current_script}*.png"
            saved = sorted(p.name for p in OUT.glob(pattern))
            print(f"{script}: {len(saved)} figures", flush=True)
    finally:
        plt.show = original_show


if __name__ == "__main__":
    sys.exit(main())
