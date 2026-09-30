"""Check documentation styling without running numerical examples."""

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator


def load_builder() -> ModuleType:
    """Load the standalone generator without assuming pytest adds the repo root."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_example_figures.py"
    spec = importlib.util.spec_from_file_location("docs_figure_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Release every small test figure, including after an assertion fails."""
    yield
    plt.close("all")


def test_paper_cycle_keeps_multiple_series_distinct() -> None:
    """Legacy C3 and C4 series must not wrap onto the first three colours."""
    with plt.style.context(str(builder.STYLE)):
        colours = [to_rgba(f"C{i}") for i in range(10)]
        assert len(set(colours)) == 10
        assert colours[0] == to_rgba("#1b4f9c")
        assert colours[3] != colours[0]
        assert colours[4] != colours[1]
        assert plt.rcParams["font.family"] == ["monospace"]


def test_artist_and_legend_colours_preserve_transparency() -> None:
    """Recolour lines, scatter markers, confidence bands, and legend copies."""
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [1, 2], color="#2196f3", marker="o", label="Line")
    scatter = ax.scatter([0.5], [1.5], color="#ff9800", alpha=0.4, label="Scatter")
    band = ax.fill_between([0, 1], [0.5, 1.5], [1.5, 2.5], color="#4caf50", alpha=0.18)
    reference = ax.axhline(1, color="black", alpha=0.5)
    legend = ax.legend()

    builder._style_figure(fig)

    np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba("#1b4f9c"))
    np.testing.assert_allclose(to_rgba(line.get_markerfacecolor()), to_rgba("#1b4f9c"))
    np.testing.assert_allclose(
        np.asarray(scatter.get_facecolor(), dtype=float)[0], to_rgba("#c85200", alpha=0.4)
    )
    np.testing.assert_allclose(
        np.asarray(band.get_facecolor(), dtype=float)[0], to_rgba("#117a3d", alpha=0.18)
    )
    assert scatter.get_alpha() == 0.4
    assert band.get_alpha() == 0.18
    assert reference.get_alpha() == 0.5
    assert to_rgba(reference.get_color()) == to_rgba("black")
    np.testing.assert_allclose(to_rgba(legend.get_lines()[0].get_color()), to_rgba("#1b4f9c"))
    legend_scatter = legend.legend_handles[1]
    assert isinstance(legend_scatter, PathCollection)
    np.testing.assert_allclose(
        np.asarray(legend_scatter.get_facecolor(), dtype=float)[0], to_rgba("#c85200", alpha=0.4)
    )


@pytest.mark.parametrize("horizontal", [False, True])
def test_bar_charts_only_grid_the_numeric_axis(horizontal: bool) -> None:
    """Categorical bar labels must not gain misleading intermediate ticks."""
    fig, ax = plt.subplots()
    if horizontal:
        ax.barh([0, 1], [0.2, 0.6])
        ax.set_yticks([0, 1], ["first", "second"])
        categorical, numeric = ax.yaxis, ax.xaxis
    else:
        ax.bar([0, 1], [0.2, 0.6])
        ax.set_xticks([0, 1], ["first", "second"])
        categorical, numeric = ax.xaxis, ax.yaxis
    ax.grid(True, which="both")

    builder._style_figure(fig)
    fig.canvas.draw()

    assert isinstance(categorical.get_minor_locator(), NullLocator)
    assert isinstance(numeric.get_minor_locator(), AutoMinorLocator)
    assert not any(tick.gridline.get_visible() for tick in categorical.get_major_ticks())
    assert any(tick.gridline.get_visible() for tick in numeric.get_major_ticks())
    assert any(tick.gridline.get_visible() for tick in numeric.get_minor_ticks())


def test_heatmap_and_colourbar_keep_their_scalar_colour_mapping() -> None:
    """Continuous maps retain their colours and avoid overlaid tick grids."""
    fig, ax = plt.subplots()
    image = ax.imshow([[0, 0.3], [0.6, 1]], cmap="viridis", vmin=0, vmax=1)
    colourbar = fig.colorbar(image, ax=ax)
    before = image.to_rgba(np.array([0, 0.3, 0.6, 1]))

    builder._style_figure(fig)
    fig.canvas.draw()

    np.testing.assert_array_equal(image.to_rgba(np.array([0, 0.3, 0.6, 1])), before)
    assert colourbar.mappable is image
    assert image.get_clim() == (0, 1)
    for panel in (ax, colourbar.ax):
        for axis in (panel.xaxis, panel.yaxis):
            assert isinstance(axis.get_minor_locator(), NullLocator)
            assert not any(tick.gridline.get_visible() for tick in axis.get_major_ticks())


def test_log_axes_have_positive_minor_ticks_and_faint_grids() -> None:
    """Logarithmic plots need log locators rather than linear subdivisions."""
    fig, ax = plt.subplots()
    ax.loglog([1, 10, 100], [0.01, 0.1, 1])

    builder._style_figure(fig)
    fig.canvas.draw()

    for axis in (ax.xaxis, ax.yaxis):
        assert isinstance(axis.get_minor_locator(), LogLocator)
        locations = axis.get_minorticklocs()
        assert len(locations) > 0
        assert np.all(locations > 0)
        major = next(
            tick.gridline for tick in axis.get_major_ticks() if tick.gridline.get_visible()
        )
        minor = next(
            tick.gridline for tick in axis.get_minor_ticks() if tick.gridline.get_visible()
        )
        assert major.get_alpha() == 0.28
        assert minor.get_alpha() == 0.10
