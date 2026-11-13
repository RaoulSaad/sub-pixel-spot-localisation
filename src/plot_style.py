"""
Plot style shared by every experiment: the palette, one line style per estimator,
and the figure frame (title block, axes, legend) that the charts are drawn on.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Plot styling: matplotlib's own default colours, one per estimator. The
# weighted centroid is also dashed with an open marker, because its curve
# lies on top of the least-squares fit and must not rely on colour alone.
STYLE = {
    "windowed_centroid": dict(
        label="Windowed centroid", color="#1f77b4", linestyle="-", marker="o", open=False
    ),
    "iteratively_weighted_centroid": dict(
        label="Weighted centroid (iterative)", color="#d62728", linestyle="--", marker="s", open=True
    ),
    "gaussian_fit_lsq": dict(
        label="Gaussian fit, least squares", color="#ff7f0e", linestyle="-", marker="^", open=False
    ),
    "gaussian_fit_mle": dict(
        label="Gaussian fit, max likelihood", color="#2ca02c", linestyle="-", marker="D", open=False
    ),
}
BAND_ALPHA = 0.16  # opacity of the sensor-to-sensor range band
SURFACE = "#ffffff"
TEXT = "#000000"
TEXT_SECONDARY = "#000000"
GRID = "#d0d0d0"
NEUTRAL = "#7f7f7f"  # for series that are not one of the four shipped estimators
TITLE_BLOCK_IN = 0.0   # the title is the axes title, so nothing is kept clear above it


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def series_kwargs(name: str) -> dict:
    s = STYLE[name]
    return dict(
        color=s["color"], linestyle=s["linestyle"], linewidth=2, marker=s["marker"],
        markersize=6.5, markerfacecolor=SURFACE if s["open"] else s["color"],
        markeredgecolor=s["color"], markeredgewidth=1.5, label=s["label"],
        capsize=2, elinewidth=1,
    )


def marker_only(name: str) -> dict:
    """Marker style of an estimator without the connecting line or error-bar settings."""
    kw = {k: v for k, v in series_kwargs(name).items() if k not in ("capsize", "elinewidth")}
    return kw | {"linestyle": "none"}


def new_figure(height: float, title: str, caption: str):
    """
    One chart with the title used across the project's figures.

    A chart carries a title, a legend and axis labels, and no other text.
    `caption` is therefore not drawn: it stays in each call as the
    description of what the chart shows (sample sizes, what a bar or a band
    means), which the report gives beside the chart.
    """
    fig, ax = plt.subplots(figsize=(7.5, height), facecolor=SURFACE)
    ax.set_title(title)
    return fig, ax, 1 - TITLE_BLOCK_IN / height


def save_figure(fig, ax, top: float, path: Path, legend_loc: str | None = "best") -> None:
    style_axes(ax)
    if legend_loc and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, loc=legend_loc)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plain_log_axis(axis) -> None:
    """Logarithmic axis with ticks at 1, 2, 5 per decade, written as plain numbers."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())
