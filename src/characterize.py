"""
Characterization: position error of each estimator against SNR, by Monte Carlo.

Two kinds of randomness are sampled. Frame noise and the spot position
change on every frame. The sensor's fixed pattern (each pixel's gain, and
which pixels are hot) belongs to one physical camera and never changes, so
each experiment is repeated on several sensors. The results are reported
pooled over the sensors, with the spread between sensors beside them.

Two experiments per camera preset and sensor:

  1. SNR sweep. For each SNR, simulate many frames with the spot at a random
     sub-pixel position, run every estimator, and record the error against
     the known truth. Compared with the Cramer-Rao floor.
  2. Sub-pixel bias. At the preset's own signal level, the mean error as a
     function of where the spot sits inside a pixel. A random position hides
     this bias in the sweep (it averages to zero), so it is measured on its
     own.

Conventions are those of simulator.py: positions are (x, y) in pixels, pixel
centres have integer coordinates.
"""

import os

# One thread for the linear-algebra library, set before numpy loads. The
# arrays here are tiny (32 x 32), so extra threads give no speed; and with
# one worker process per core already, a thread pool inside each of them
# makes the processes fight over the cores.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

import estimators as est
from plot_style import (
    BAND_ALPHA, STYLE, SURFACE, TEXT, TEXT_SECONDARY, TITLE_BLOCK_IN, series_kwargs, style_axes,
)
from precision_floor import cramer_rao_floor_px
from simulator import (
    load_config,
    make_rngs,
    make_sensor_maps,
    signal_for_target_snr,
    simulate_frame,
)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "characterization"

# One seed per simulated sensor. 42 is the sensor every other experiment uses. The
# others are spaced so that the seeds derived from each one (seed + 1 to
# seed + 3, for the position and sub-pixel streams) never coincide.
SENSOR_SEEDS = (42, 142, 242, 342, 442)
SNR_RANGE = (5.0, 1000.0)
DEFAULT_FRAMES = 2000
DEFAULT_POINTS = 12
SUBPIXEL_BINS = 10
GROSS_ERROR_PX = 1.0  # an accepted estimate further than this from the truth

# No estimator refuses a spot for being saturated. Each reports how many
# saturated pixels its window held, and the mean of that count is written
# beside the error, so the growth of the error with saturation can be read.
METHODS = {
    "windowed_centroid": est.windowed_centroid,
    "iteratively_weighted_centroid": est.iteratively_weighted_centroid,
    "gaussian_fit_lsq": est.gaussian_fit_lsq,
    "gaussian_fit_mle": est.gaussian_fit_mle,
}


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

def run_frames(camera, spot, background, maps, hot, total_signal_e, n_frames, frame_rng, position_rng):
    """
    Simulate n_frames with the spot at a random position inside one pixel
    and run every estimator on each. Returns the true sub-pixel offsets, the
    mean nominal SNR, and per method the errors (NaN where it failed), the
    success flags and the saturated-pixel counts.
    """
    centre_x, centre_y = (round(c) for c in spot.center_xy)
    offsets = position_rng.uniform(-0.5, 0.5, size=(n_frames, 2))
    errors = {name: np.full((n_frames, 2), np.nan) for name in METHODS}
    ok = {name: np.zeros(n_frames, dtype=bool) for name in METHODS}
    n_saturated = {name: np.zeros(n_frames, dtype=int) for name in METHODS}
    snr = np.empty(n_frames)

    for i, (dx, dy) in enumerate(offsets):
        truth = (centre_x + dx, centre_y + dy)
        frame_spot = replace(spot, center_xy=truth, total_signal_e=total_signal_e)
        frame = simulate_frame(frame_spot, camera, background, maps, frame_rng)
        snr[i] = frame.nominal_snr
        for name, estimator in METHODS.items():
            result = estimator(frame.image_dn, camera, hot)
            if result.ok:
                ok[name][i] = True
                errors[name][i] = (result.x - truth[0], result.y - truth[1])
                n_saturated[name][i] = result.n_saturated

    return offsets, float(snr.mean()), errors, ok, n_saturated


def summarise(errors, ok, floor_px):
    """
    Statistics of one estimator at one SNR point, over the frames it
    accepted. x and y are pooled where it helps: the spot and the estimators
    are symmetric, so both axes measure the same thing and pooling halves
    the statistical error.
    """
    n_frames = len(ok)
    n_ok = int(ok.sum())
    row = {
        "n_frames": n_frames,
        "n_ok": n_ok,
        "failure_rate": 1 - n_ok / n_frames,
    }
    if n_ok < 2:
        return row | {key: float("nan") for key in (
            "bias_x_px", "bias_y_px", "bias_se_px", "std_px", "std_se_px",
            "rms_px", "rms_se_px", "rms_over_floor", "gross_error_rate",
        )}

    e = errors[ok]
    std = float(np.sqrt(e.var(axis=0, ddof=1).mean()))
    rms = float(np.sqrt((e**2).mean()))
    # Standard error of a standard deviation from m samples is std / sqrt(2m);
    # here m = 2 * n_ok because both axes are pooled.
    row |= {
        "bias_x_px": float(e[:, 0].mean()),
        "bias_y_px": float(e[:, 1].mean()),
        "bias_se_px": std / np.sqrt(n_ok),
        "std_px": std,
        "std_se_px": std / (2 * np.sqrt(n_ok)),
        "rms_px": rms,
        "rms_se_px": rms / (2 * np.sqrt(n_ok)),
        "rms_over_floor": rms / floor_px,
        "gross_error_rate": float((np.abs(e).max(axis=1) > GROSS_ERROR_PX).mean()),
    }
    return row



# --------------------------------------------------------------------------
# One sensor (each call runs in its own worker process)
# --------------------------------------------------------------------------

def build_sensor(preset: str, sensor_seed: int):
    """A camera preset with the fixed pattern (pixel gains, hot pixels) of one sensor."""
    camera, spot, background = load_config(preset)
    sensor_rng, frame_rng = make_rngs(sensor_seed)
    maps = make_sensor_maps(camera, sensor_rng)
    return camera, spot, background, maps, est.hot_pixel_mask(maps, camera), frame_rng


def sweep_one_sensor(preset: str, sensor_seed: int, n_frames: int, n_points: int):
    """
    Experiment 1 on one sensor: every estimator at each SNR. Returns the raw
    errors per SNR point (so they can be pooled with other sensors) and the
    SNR at which this sensor starts to saturate.
    """
    camera, spot, background, maps, hot, frame_rng = build_sensor(preset, sensor_seed)
    position_rng = np.random.default_rng(sensor_seed + 1)

    points = []
    for target_snr in np.geomspace(*SNR_RANGE, n_points):
        with warnings.catch_warnings():
            # The simulator warns when a target SNR needs a near-saturated
            # peak. The sweep goes there on purpose.
            warnings.simplefilter("ignore")
            total = signal_for_target_snr(target_snr, spot, camera, background, maps)
        _, snr, errors, ok, n_saturated = run_frames(
            camera, spot, background, maps, hot, total, n_frames, frame_rng, position_rng
        )
        points.append({
            "snr": snr,
            "signal_e": total,
            "floor_px": cramer_rao_floor_px(camera, spot, background, total),
            "errors": errors,
            "ok": ok,
            "n_saturated": n_saturated,
        })
    return points, saturation_onset_snr(camera, spot, background, maps)


def saturation_onset_snr(camera, spot, background, maps) -> float:
    """Nominal SNR at which the brightest pixel of a pixel-centred spot reaches full well."""
    centred = replace(spot, center_xy=tuple(float(round(c)) for c in spot.center_xy))
    unit = simulate_frame(replace(centred, total_signal_e=1.0), camera, background, maps,
                          np.random.default_rng(0))
    headroom = camera.full_well_e - unit.expected_background_e.max() - unit.expected_dark_e.min()
    total_at_onset = headroom / unit.expected_spot_e.max()
    onset = simulate_frame(replace(centred, total_signal_e=total_at_onset), camera, background,
                           maps, np.random.default_rng(0))
    return onset.nominal_snr


def subpixel_one_sensor(preset: str, sensor_seed: int, n_frames: int):
    """
    Experiment 2 on one sensor: mean error against the spot's offset from the
    nearest pixel centre, at the preset's own signal level. The x error is
    binned by the x offset and the y error by the y offset, then both are
    pooled. Also fits a straight line through error against offset; its
    slope is the bias in pixels per pixel of offset.
    """
    camera, spot, background, maps, hot, _ = build_sensor(preset, sensor_seed)
    _, frame_rng = make_rngs(sensor_seed + 2)
    position_rng = np.random.default_rng(sensor_seed + 3)

    offsets, snr, errors, ok, _ = run_frames(
        camera, spot, background, maps, hot, spot.total_signal_e, n_frames, frame_rng, position_rng
    )
    edges = np.linspace(-0.5, 0.5, SUBPIXEL_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_rows, slope_rows = [], []
    for name in METHODS:
        offset = offsets[ok[name]].ravel()   # x offsets and y offsets together
        error = errors[name][ok[name]].ravel()
        which = np.clip(np.digitize(offset, edges) - 1, 0, SUBPIXEL_BINS - 1)
        for b, centre in enumerate(centres):
            e = error[which == b]
            bin_rows.append({
                "preset": preset, "sensor_seed": sensor_seed, "snr": snr, "estimator": name,
                "subpixel_offset_px": float(centre),
                "n": int(e.size),
                "mean_error_px": float(e.mean()),
                "mean_error_se_px": float(e.std(ddof=1) / np.sqrt(e.size)),
            })
        slope, intercept = np.polyfit(offset, error, 1)
        scatter = (error - (slope * offset + intercept)).std(ddof=2)
        slope_rows.append({
            "preset": preset, "sensor_seed": sensor_seed, "snr": snr, "estimator": name,
            "slope_px_per_px": float(slope),
            "slope_se_px_per_px": float(scatter / (offset.std() * np.sqrt(offset.size))),
            "mean_error_px": float(error.mean()),
        })
    return bin_rows, slope_rows


def run_job(job):
    """Entry point of a worker process: one experiment on one sensor."""
    kind, preset, sensor_seed, n_frames, n_points = job
    if kind == "sweep":
        return job, sweep_one_sensor(preset, sensor_seed, n_frames, n_points)
    return job, subpixel_one_sensor(preset, sensor_seed, 2 * n_frames)


# --------------------------------------------------------------------------
# Pooling the sensors
# --------------------------------------------------------------------------

def value_range(values) -> tuple[float, float]:
    """Lowest and highest finite value, or NaN if there is none."""
    finite = [v for v in values if np.isfinite(v)]
    return (min(finite), max(finite)) if finite else (float("nan"), float("nan"))


def pool_sweep(preset: str, by_sensor: dict):
    """
    Combine the sweeps of several sensors. Returns (pooled rows, per-sensor
    rows, mean saturation onset).

    A pooled row treats the frames of all sensors as one sample, so its RMS
    error is what a camera drawn at random would give on average. Frame
    noise averages down with more frames; a sensor's fixed pattern does not,
    so the lowest and highest per-sensor values are kept beside it. That
    range, not the pooled standard error, is the honest uncertainty wherever
    the fixed pattern matters.
    """
    seeds = list(by_sensor)
    n_points = len(by_sensor[seeds[0]][0])
    pooled_rows, sensor_rows = [], []

    for point in range(n_points):
        at = {seed: by_sensor[seed][0][point] for seed in seeds}
        saturated = {
            seed: float(np.mean([p["n_saturated"][name][p["ok"][name]].mean()
                                 for name in METHODS if p["ok"][name].any()] or [0.0]))
            for seed, p in at.items()
        }
        own = {}
        for seed, p in at.items():
            for name in METHODS:
                own[seed, name] = {
                    "preset": preset, "sensor_seed": seed, "point": point,
                    "snr": p["snr"], "signal_e": p["signal_e"], "floor_px": p["floor_px"],
                    "mean_saturated_px": saturated[seed], "estimator": name,
                } | summarise(p["errors"][name], p["ok"][name], p["floor_px"])
                sensor_rows.append(own[seed, name])

        floor = float(np.mean([p["floor_px"] for p in at.values()]))
        for name in METHODS:
            row = {
                "preset": preset, "n_sensors": len(seeds), "point": point,
                "snr": float(np.mean([p["snr"] for p in at.values()])),
                "signal_e": float(np.mean([p["signal_e"] for p in at.values()])),
                "floor_px": floor,
                "mean_saturated_px": float(np.mean(list(saturated.values()))),
                "estimator": name,
            } | summarise(
                np.concatenate([p["errors"][name] for p in at.values()]),
                np.concatenate([p["ok"][name] for p in at.values()]),
                floor,
            )
            mine = [own[seed, name] for seed in seeds]
            for key in ("rms_px", "rms_over_floor", "failure_rate"):
                row[f"{key}_min"], row[f"{key}_max"] = value_range(r[key] for r in mine)
            row["bias_min_px"], row["bias_max_px"] = value_range(
                [r["bias_x_px"] for r in mine] + [r["bias_y_px"] for r in mine]
            )
            pooled_rows.append(row)

    onset = float(np.mean([by_sensor[seed][1] for seed in seeds]))
    return pooled_rows, sensor_rows, onset


def pool_subpixel(preset: str, by_sensor: dict):
    """Combine experiment 2 across sensors. Returns (pooled rows, per-sensor rows, slope rows)."""
    sensor_rows = [row for bins, _ in by_sensor.values() for row in bins]
    slope_rows = [row for _, slopes in by_sensor.values() for row in slopes]
    snr = float(np.mean([row["snr"] for row in sensor_rows]))

    pooled_rows = []
    for name in METHODS:
        for offset in sorted({row["subpixel_offset_px"] for row in sensor_rows}):
            mine = [row for row in sensor_rows
                    if row["estimator"] == name and row["subpixel_offset_px"] == offset]
            n = np.array([row["n"] for row in mine])
            mean = np.array([row["mean_error_px"] for row in mine])
            se = np.array([row["mean_error_se_px"] for row in mine])
            low, high = value_range(mean)
            pooled_rows.append({
                "preset": preset, "n_sensors": len(by_sensor), "snr": snr, "estimator": name,
                "subpixel_offset_px": offset,
                "n": int(n.sum()),
                "mean_error_px": float((n * mean).sum() / n.sum()),
                "mean_error_se_px": float(np.sqrt(((n * se) ** 2).sum()) / n.sum()),
                "mean_error_px_min": low,
                "mean_error_px_max": high,
            })
    return pooled_rows, sensor_rows, slope_rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_series(ax, rows: list[dict], x_key: str, y_key: str, se=None, scale: float = 1.0) -> None:
    """
    One line per estimator: the value pooled over all sensors, with a band
    from the lowest to the highest single sensor. `se` maps a row to its
    standard error; leave it out for no error bars.

    The band is shaded for the solid-line methods. The weighted centroid,
    drawn dashed with open markers, gets a dashed outline instead: its band
    mostly coincides with the least-squares one, and a fourth shaded layer
    in the plain centroid's colour would make the overlap unreadable.
    """
    for name in METHODS:
        r = [row for row in rows if row["estimator"] == name]
        x = [row[x_key] for row in r]
        kw = series_kwargs(name)
        low = [scale * row[f"{y_key}_min"] for row in r]
        high = [scale * row[f"{y_key}_max"] for row in r]
        if STYLE[name]["open"]:
            for edge in (low, high):
                ax.plot(x, edge, color=kw["color"], linestyle=(0, (2, 2)), linewidth=1,
                        alpha=0.8, zorder=2)
        else:
            ax.fill_between(x, low, high, color=kw["color"], alpha=BAND_ALPHA,
                            linewidth=0, zorder=2)
        if se is None:
            kw = {k: v for k, v in kw.items() if k not in ("capsize", "elinewidth")}
            ax.plot(x, [scale * row[y_key] for row in r], zorder=3, **kw)
        else:
            ax.errorbar(x, [scale * row[y_key] for row in r],
                        yerr=[scale * se(row) for row in r], zorder=3, **kw)


def sweep_figure(height: float, title: str, caption: str):
    """
    A single chart on an SNR axis, with the title the three sweep charts
    share. `caption` is not drawn (see plot_style.new_figure).
    """
    fig, ax = plt.subplots(figsize=(7.5, height), facecolor=SURFACE)
    top = 1 - TITLE_BLOCK_IN / height
    ax.set_title(title)
    return fig, ax, top


def finish_sweep_figure(fig, ax, top: float, path: Path, onset_snr: float, legend_loc: str) -> None:
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Nominal SNR")
    if SNR_RANGE[0] < onset_snr < SNR_RANGE[1]:
        ax.axvline(onset_snr, color=TEXT_SECONDARY, linewidth=1.2, linestyle=":",
                   label="Saturation begins")
    ax.legend(fontsize=9, loc=legend_loc)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def sweep_caption(rows: list[dict], n_frames: int, error_bars: bool) -> str:
    return (
        f"{rows[0]['n_sensors']} sensors x {n_frames} frames per point, "
        "spot at a random sub-pixel position."
        + (" Bars: one standard error." if error_bars else "")
        + "\nLine: all sensors pooled. Band: lowest to highest sensor "
        "(dashed outline: weighted centroid)."
    )


def plot_error(path: Path, preset: str, rows: list[dict], onset_snr: float, n_frames: int) -> None:
    """Chart 1: RMS position error against SNR, with the theoretical floor."""
    fig, ax, top = sweep_figure(5.4, f"Position error against SNR, {preset} preset",
                                sweep_caption(rows, n_frames, error_bars=True))
    first = [r for r in rows if r["estimator"] == next(iter(METHODS))]
    ax.plot([r["snr"] for r in first], [r["floor_px"] for r in first], color=TEXT, linewidth=2,
            label="Cramér-Rao floor (ideal sensor)", zorder=1)
    draw_series(ax, rows, "snr", "rms_px", se=lambda row: row["rms_se_px"])
    ax.set_yscale("log")
    ax.set_ylabel("RMS position error per axis (px)")
    finish_sweep_figure(fig, ax, top, path, onset_snr, "lower left")


def plot_ratio(path: Path, preset: str, rows: list[dict], onset_snr: float, n_frames: int) -> None:
    """Chart 2: the same error divided by the floor, so 1 means as good as theory allows."""
    fig, ax, top = sweep_figure(5.0, f"Position error relative to the floor, {preset} preset",
                                sweep_caption(rows, n_frames, error_bars=True))
    ax.axhline(1.0, color=TEXT, linewidth=2, zorder=1, label="Cramér-Rao floor (ideal sensor)")
    draw_series(ax, rows, "snr", "rms_over_floor", se=lambda row: row["rms_se_px"] / row["floor_px"])
    ax.set_yscale("log")
    ax.set_ylabel("RMS error / floor")
    ticks = [t for t in (1, 2, 3, 5, 10, 20) if t <= max(2.0, ax.get_ylim()[1])]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    finish_sweep_figure(fig, ax, top, path, onset_snr, "upper left")


def plot_refused(path: Path, preset: str, rows: list[dict], onset_snr: float, n_frames: int) -> None:
    """Chart 3: share of frames each estimator refused to answer."""
    fig, ax, top = sweep_figure(3.9, f"Frames refused against SNR, {preset} preset",
                                sweep_caption(rows, n_frames, error_bars=False))
    draw_series(ax, rows, "snr", "failure_rate", scale=100.0)
    ax.set_ylabel("Frames refused (%)")
    ax.set_ylim(-3, 103)
    finish_sweep_figure(fig, ax, top, path, onset_snr, "center")


def plot_subpixel(path: Path, preset: str, rows: list[dict], n_frames: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.9), facecolor=SURFACE)
    ax.axhline(0, color=TEXT, linewidth=1)
    draw_series(ax, rows, "subpixel_offset_px", "mean_error_px",
                se=lambda row: row["mean_error_se_px"])
    style_axes(ax)
    ax.set_xlabel("Spot offset from the nearest pixel centre (px)")
    ax.set_ylabel("Mean position error (px)")
    ax.legend(fontsize=9)
    ax.set_title(f"Bias against sub-pixel position, {preset} preset")
    fig.tight_layout(rect=(0, 0, 1, 1 - TITLE_BLOCK_IN / 4.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_tables(path: Path, sweeps: dict, sensor_sweeps: dict, biases: dict, slopes: dict,
                 onsets: dict, n_frames: int, seeds: list[int]) -> None:
    """The numbers behind the plots, as readable Markdown."""
    labels = " | ".join(STYLE[name]["label"] for name in METHODS)
    lines = [
        "# Characterization results",
        "",
        "Generated by `src/characterize.py`; do not edit by hand. "
        f"{len(seeds)} sensors (seeds {', '.join(str(s) for s in seeds)}), "
        f"{n_frames} frames per sensor per SNR point.",
        "",
        "A sensor is one draw of the fixed pattern: pixel gains and hot pixels. "
        "Pooled values treat the frames of all sensors as one sample. "
        "Ranges are the lowest and highest single sensor.",
        "",
        "RMS error is per axis, in pixels, over the frames each estimator accepted. "
        "The number in brackets is the RMS error divided by the Cramér-Rao floor. "
        "No estimator refuses a spot for being saturated; Saturated px is the mean "
        "number of saturated pixels in the window.",
        "",
    ]
    for preset, rows in sweeps.items():
        points = sorted({r["point"] for r in rows})
        at = {(r["point"], r["estimator"]): r for r in rows}
        first = next(iter(METHODS))

        lines += [f"## {preset}: error against SNR, all sensors pooled", "",
                  f"Saturation begins at nominal SNR {onsets[preset]:.0f}.", "",
                  f"| SNR | Signal (e) | Saturated px | Floor | {labels} |",
                  "|---|---|---|---|" + "---|" * len(METHODS)]
        for point in points:
            any_row = at[point, first]
            cells = [f"{at[point, name]['rms_px']:.4f} ({at[point, name]['rms_over_floor']:.2f})"
                     for name in METHODS]
            lines.append(f"| {any_row['snr']:.1f} | {any_row['signal_e']:,.0f} "
                         f"| {any_row['mean_saturated_px']:.1f} | {any_row['floor_px']:.4f} | "
                         + " | ".join(cells) + " |")

        lines += ["", f"### {preset}: RMS error ÷ floor, lowest to highest sensor", "",
                  f"| SNR | {labels} |", "|---|" + "---|" * len(METHODS)]
        for point in points:
            cells = [f"{at[point, name]['rms_over_floor_min']:.2f} to "
                     f"{at[point, name]['rms_over_floor_max']:.2f}" for name in METHODS]
            lines.append(f"| {at[point, first]['snr']:.1f} | " + " | ".join(cells) + " |")

        lines += ["", f"### {preset}: RMS error ÷ floor per sensor, four brightest points", "",
                  f"| SNR | Sensor | {labels} |", "|---|---|" + "---|" * len(METHODS)]
        own = {(r["point"], r["sensor_seed"], r["estimator"]): r for r in sensor_sweeps[preset]}
        for point in points[-4:]:
            for seed in seeds:
                cells = [f"{own[point, seed, name]['rms_over_floor']:.2f}" for name in METHODS]
                lines.append(f"| {own[point, seed, first]['snr']:.1f} | {seed} | "
                             + " | ".join(cells) + " |")

        lines += ["", f"### {preset}: frames refused, gross errors, bias", "",
                  "Refused: the estimator reported failure. Gross: accepted, but more than "
                  f"{GROSS_ERROR_PX:.0f} px from the truth. Bias: mean error (x / y) "
                  "over all sensors, with its standard error from frame noise. Bias range: lowest "
                  "to highest mean error of a single sensor, either axis.", "",
                  "| SNR | Estimator | Refused | Gross | Bias x / y (px) | ± "
                  "| Bias range over sensors |",
                  "|---|---|---|---|---|---|---|"]
        for point in points:
            for name in METHODS:
                r = at[point, name]
                lines.append(
                    f"| {r['snr']:.1f} | {STYLE[name]['label']} | {100 * r['failure_rate']:.1f}% "
                    f"| {100 * r['gross_error_rate']:.1f}% "
                    f"| {r['bias_x_px']:+.4f} / {r['bias_y_px']:+.4f} | {r['bias_se_px']:.4f} "
                    f"| {r['bias_min_px']:+.4f} to {r['bias_max_px']:+.4f} |")
        lines.append("")

    for preset, rows in biases.items():
        lines += [f"## {preset}: mean error against sub-pixel offset (SNR {rows[0]['snr']:.0f})", "",
                  "Pooled mean ± standard error, then the lowest and highest single sensor in brackets.",
                  "", f"| Offset (px) | {labels} |", "|---|" + "---|" * len(METHODS)]
        at = {(r["subpixel_offset_px"], r["estimator"]): r for r in rows}
        for offset in sorted({r["subpixel_offset_px"] for r in rows}):
            cells = [f"{at[offset, name]['mean_error_px']:+.4f} ± {at[offset, name]['mean_error_se_px']:.4f} "
                     f"({at[offset, name]['mean_error_px_min']:+.4f} to "
                     f"{at[offset, name]['mean_error_px_max']:+.4f})" for name in METHODS]
            lines.append(f"| {offset:+.2f} | " + " | ".join(cells) + " |")

        lines += ["", f"### {preset}: slope of error against offset, per sensor", "",
                  "Straight-line fit through error against offset, in pixels of error per pixel "
                  "of offset, ± one standard error. Zero means no dependence on sub-pixel position.",
                  "", f"| Sensor | {labels} |", "|---|" + "---|" * len(METHODS)]
        own = {(r["sensor_seed"], r["estimator"]): r for r in slopes[preset]}
        for seed in seeds:
            cells = [f"{own[seed, name]['slope_px_per_px']:+.4f} ± "
                     f"{own[seed, name]['slope_se_px_per_px']:.4f}" for name in METHODS]
            lines.append(f"| {seed} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames per sensor per SNR point")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="SNR points in the sweep")
    parser.add_argument("--presets", nargs="+", default=["silicon", "ingaas"])
    parser.add_argument("--sensors", nargs="+", type=int, default=list(SENSOR_SEEDS),
                        help="sensor seeds; each is one draw of the fixed pattern")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--quick", action="store_true",
                        help="100 frames, 4 SNR points, 2 sensors: a smoke test, not a result")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.quick:
        args.frames, args.points, args.sensors = 100, 4, args.sensors[:2]
    args.output.mkdir(parents=True, exist_ok=True)

    # One job per (experiment, preset, sensor). Each seeds its own random
    # streams from the sensor seed, so the results do not depend on how many
    # workers there are or on the order in which they finish.
    jobs = [(kind, preset, seed, args.frames, args.points)
            for preset in args.presets for seed in args.sensors
            for kind in ("sweep", "subpixel")]
    results = {}
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(run_job, job) for job in jobs]), 1):
            (kind, preset, seed, *_), result = future.result()
            results[kind, preset, seed] = result
            print(f"  [{done}/{len(jobs)}] {preset} sensor {seed}: {kind}", flush=True)

    sweeps, sensor_sweeps, biases, slopes, onsets = {}, {}, {}, {}, {}
    for preset in args.presets:
        sweeps[preset], sensor_sweeps[preset], onsets[preset] = pool_sweep(
            preset, {seed: results["sweep", preset, seed] for seed in args.sensors}
        )
        biases[preset], sensor_biases, slopes[preset] = pool_subpixel(
            preset, {seed: results["subpixel", preset, seed] for seed in args.sensors}
        )
        write_csv(args.output / f"results_{preset}.csv", sweeps[preset])
        write_csv(args.output / f"results_by_sensor_{preset}.csv", sensor_sweeps[preset])
        write_csv(args.output / f"subpixel_bias_{preset}.csv", biases[preset])
        write_csv(args.output / f"subpixel_bias_by_sensor_{preset}.csv", sensor_biases)
        write_csv(args.output / f"subpixel_slope_by_sensor_{preset}.csv", slopes[preset])

        plot_error(args.output / f"error_vs_snr_{preset}.png", preset, sweeps[preset],
                   onsets[preset], args.frames)
        plot_ratio(args.output / f"error_over_floor_vs_snr_{preset}.png", preset, sweeps[preset],
                   onsets[preset], args.frames)
        plot_refused(args.output / f"frames_refused_vs_snr_{preset}.png", preset, sweeps[preset],
                     onsets[preset], args.frames)
        plot_subpixel(args.output / f"subpixel_bias_{preset}.png", preset, biases[preset],
                      2 * args.frames)

    write_tables(args.output / "tables.md", sweeps, sensor_sweeps, biases, slopes, onsets,
                 args.frames, args.sensors)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
