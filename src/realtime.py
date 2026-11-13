"""
Real-time cost: time per frame of each estimator, against a 1 kHz frame rate.

Three measurements per camera preset:

  1. Each estimator exactly as shipped in estimators.py, on single frames,
     at several SNRs (the iterative methods take more steps on dim spots).
  2. The shared front end (`preprocess`) on its own, and its parts, to show
     where the time goes.
  3. A tracking-mode weighted centroid (defined in this file): the same
     estimator with everything removed that a running tracker does not need
     to redo on every frame. Its accuracy is measured against the shipped
     version on the same frames, so the saving has a stated price.

Everything is written to results/realtime/.

The numbers are for this machine, this Python and one core. They rank the
methods and show where the time goes; they are not what an embedded
implementation would take.
"""

import os

# One thread for the linear-algebra library, set before numpy loads. The
# arrays here are tiny (32 x 32); extra threads add scheduling jitter, not
# speed, and a real-time loop would pin the estimator to one core anyway.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import csv
import gc
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import scipy

import estimators as est
from plot_style import NEUTRAL, STYLE, SURFACE, TEXT_SECONDARY, TITLE_BLOCK_IN, style_axes
from simulator import (
    load_config,
    make_rngs,
    make_sensor_maps,
    signal_for_target_snr,
    simulate_frame,
)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "realtime"

SEED = 42
FRAME_RATE_HZ = 1000.0
PERIOD_US = 1e6 / FRAME_RATE_HZ
# Share of the frame period given to the estimator. An assumption, not a
# stated requirement: exposure takes 500 us of the 1,000 us period, and
# readout, transfer and the control output need the rest of the other half.
BUDGET_US = 200.0
SNRS = (10.0, 30.0, 100.0)   # plus each preset's own signal level
DEFAULT_FRAMES = 300
DEFAULT_REPEATS = 3
WARMUP_CALLS = 30

TRACKING_ITERATIONS = 3      # fixed, so the tracking-mode cost does not vary
PRIOR_ERROR_PX = 0.3         # how far the previous position may be from the truth
CALIBRATION_FRAMES = 20      # frames averaged for the tracking-mode background

SHIPPED = {
    "windowed_centroid": est.windowed_centroid,
    "iteratively_weighted_centroid": est.iteratively_weighted_centroid,
    "gaussian_fit_lsq": est.gaussian_fit_lsq,
    "gaussian_fit_mle": est.gaussian_fit_mle,
}
LABELS = {name: STYLE[name]["label"] for name in SHIPPED} | {
    "tracking_weighted_centroid": "Weighted centroid, tracking mode",
    "preprocess": "Shared front end only (preprocess)",
}


# --------------------------------------------------------------------------
# Tracking-mode weighted centroid
# --------------------------------------------------------------------------

def calibrate_background(frames, camera, hot_mask):
    """
    Background plane and its scatter, averaged over a few frames, using the
    same plane fit as the shipped front end. A running tracker would do this
    at acquisition and refresh it slowly; the background changes over
    seconds, not from one frame to the next.

    A frame in which the front end finds no spot is left out: with a dim
    spot (SNR 6 to 8) that happens to a few frames in twenty, and one such
    frame must not stop the calibration. Raises EstimationFailure only if
    no frame at all shows a spot.
    """
    planes, scatters = [], []
    for frame in frames:
        try:
            window = est.preprocess(frame.image_dn, camera, hot_mask)
        except est.EstimationFailure:
            continue
        image_e = est.dn_to_electrons(frame.image_dn, camera)
        centre = (int(window.xs[est.HALF_WIDTH]), int(window.ys[est.HALF_WIDTH]))
        plane, scatter = est.fit_background_plane(
            image_e, ~hot_mask, centre, est.BACKGROUND_EXCLUDE_RADIUS
        )
        planes.append(plane)
        scatters.append(scatter)
    if not planes:
        raise est.EstimationFailure("no calibration frame shows a spot")
    return np.mean(planes, axis=0), float(np.mean(scatters))


def tracking_weighted_centroid(
    image_dn, camera, hot_mask, background_e, background_std_e, last_xy,
    half_width: int = est.HALF_WIDTH,
    sigma_weight: float = est.SIGMA_GUESS,
    n_iter: int = TRACKING_ITERATIONS,
) -> est.Estimate:
    """
    The weighted centroid for a tracker that already knows roughly where the
    spot is. It reaches the same answer as est.iteratively_weighted_centroid
    (same weight, same fixed point) with three things removed:

      - no search: the window is cut around `last_xy`, the previous frame's
        estimate, instead of looking for the peak in the whole frame;
      - no background fit: `background_e` is a plane measured earlier
        (calibrate_background), not refitted twice per frame;
      - a fixed number of steps, so the time per frame does not depend on
        the image.

    Each step is twice as long as the shipped version's. With the weight as
    wide as the spot, one weighted-centroid pass lands halfway between where
    the weight sits and where the spot is.
    So the spot is at  weight + 2 * (centroid - weight). The shipped version
    takes the half step and repeats until it stops moving; this one takes
    the full step. If the spot is wider or narrower than the weight the full
    step is slightly off, and the next step corrects it.

    Fails (ok=False) when the window would leave the frame or when there is
    no spot under it. The caller must then fall back to a full-frame
    estimator to find the spot again.
    """
    cx, cy = int(round(last_xy[0])), int(round(last_xy[1]))
    height, width = image_dn.shape
    if not (half_width <= cx < width - half_width and half_width <= cy < height - half_width):
        return est.Estimate(float("nan"), float("nan"), ok=False, reason="window off frame")

    rows = slice(cy - half_width, cy + half_width + 1)
    cols = slice(cx - half_width, cx + half_width + 1)
    signal = (
        (image_dn[rows, cols].astype(np.float64) - camera.bias_dn) * camera.electrons_per_dn
        - background_e[rows, cols]
    )
    hot = hot_mask[rows, cols]
    if hot.any():
        signal = est.fill_invalid(signal, ~hot)

    core = signal[half_width - 1 : half_width + 2, half_width - 1 : half_width + 2]
    if core.mean() < est.DETECTION_THRESHOLD * background_std_e / 3:
        return est.Estimate(float("nan"), float("nan"), ok=False, reason="no spot under the window")

    xs = np.arange(cols.start, cols.stop, dtype=np.float64)
    ys = np.arange(rows.start, rows.stop, dtype=np.float64)
    x, y = float(last_xy[0]), float(last_xy[1])
    for _ in range(n_iter):
        # The weight is separable, w = outer(wy, wx), so the weighted sums
        # are three small matrix products instead of a full weighted image.
        wx = est._pixel_fractions(xs, x, sigma_weight)
        wy = est._pixel_fractions(ys, y, sigma_weight)
        row_sums = wy @ signal          # weighted sum down each column
        col_sums = signal @ wx          # weighted sum along each row
        total = row_sums @ wx
        if total <= 0:
            return est.Estimate(float("nan"), float("nan"), ok=False, reason="no net weighted signal")
        x += 2 * ((row_sums @ (wx * xs)) / total - x)
        y += 2 * (((wy * ys) @ col_sums) / total - y)
        if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
            return est.Estimate(float("nan"), float("nan"), ok=False, reason="estimate left the window")

    return est.Estimate(x=float(x), y=float(y), ok=True, n_iter=n_iter)


def track_with_fallback(
    image_dn, camera, hot_mask, background_e, background_std_e, last_xy,
    full_frame: est.Estimate | None = None,
) -> tuple[est.Estimate, bool]:
    """
    One frame of the loop: tracking mode
    around the previous estimate, and the shipped full-frame weighted
    centroid when tracking mode refuses the frame. Returns (estimate,
    whether the full-frame estimator had to be used).

    `last_xy` is None before the first estimate exists; the full-frame
    estimator then finds the spot (acquisition), and that is not counted as
    a fallback. `background_e` is None when no calibration is available;
    every frame is then a full-frame one.

    `full_frame` is the full-frame estimate of this same frame, if the
    caller has already computed it; otherwise it is computed here when
    needed.

    The returned estimate can itself be a failure (ok=False): both
    estimators refused. The caller keeps its previous position and tries
    again on the next frame.
    """
    if last_xy is not None and background_e is not None:
        tracked = tracking_weighted_centroid(
            image_dn, camera, hot_mask, background_e, background_std_e, last_xy
        )
        if tracked.ok:
            return tracked, False
    if full_frame is None:
        full_frame = est.iteratively_weighted_centroid(image_dn, camera, hot_mask)
    return full_frame, last_xy is not None


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def time_calls(call, n_frames: int, repeats: int) -> np.ndarray:
    """
    Time `call(i)` for every frame index, `repeats` times over. Returns the
    microseconds of every single call, shape (repeats, n_frames).

    The garbage collector is switched off while timing: it would add a pause
    to a random call that has nothing to do with the estimator. The clock is
    time.perf_counter_ns, the highest-resolution monotonic clock Python has.
    """
    for i in range(min(WARMUP_CALLS, n_frames)):
        call(i)
    out = np.empty((repeats, n_frames))
    gc.collect()
    gc.disable()
    try:
        for r in range(repeats):
            for i in range(n_frames):
                start = time.perf_counter_ns()
                call(i)
                out[r, i] = (time.perf_counter_ns() - start) / 1e3
    finally:
        gc.enable()
    return out


def summarise_times(times_us: np.ndarray) -> dict:
    """
    Median, tail and worst case over every call. The 99th percentile is the
    number a real-time loop cares about: a frame that arrives late is a
    frame the loop did not get. `floor_us` is the median of each frame's
    fastest repeat: the time with operating-system interruptions mostly
    removed.
    """
    flat = times_us.ravel()
    return {
        "median_us": float(np.median(flat)),
        "p99_us": float(np.percentile(flat, 99)),
        "max_us": float(flat.max()),
        "floor_us": float(np.median(times_us.min(axis=0))),
    }


def verdict(p99_us: float) -> str:
    if p99_us <= BUDGET_US:
        return "fits the budget"
    if p99_us <= PERIOD_US:
        return "fits the period, not the budget"
    return "too slow"


def make_frames(preset: str, target_snr: float | None, n_frames: int, seed_offset: int):
    """Frames with the spot at a random sub-pixel position, one camera."""
    camera, spot, background = load_config(preset)
    sensor_rng, _ = make_rngs(SEED)
    maps = make_sensor_maps(camera, sensor_rng)
    hot = est.hot_pixel_mask(maps, camera)
    _, frame_rng = make_rngs(SEED + 10 + seed_offset)
    position_rng = np.random.default_rng(SEED + 20 + seed_offset)

    if target_snr is None:
        total = spot.total_signal_e
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total = signal_for_target_snr(target_snr, spot, camera, background, maps)
    centre = np.round(spot.center_xy)
    truths = centre + position_rng.uniform(-0.5, 0.5, size=(n_frames, 2))
    frames = [
        simulate_frame(replace(spot, center_xy=tuple(t), total_signal_e=total),
                       camera, background, maps, frame_rng)
        for t in truths
    ]
    priors = truths + position_rng.normal(0.0, PRIOR_ERROR_PX, size=truths.shape)
    return camera, hot, frames, truths, priors


def measure_condition(preset: str, target_snr: float | None, n_frames: int, repeats: int,
                      seed_offset: int) -> list[dict]:
    """Time and score every method at one preset and signal level."""
    camera, hot, frames, truths, priors = make_frames(preset, target_snr, n_frames, seed_offset)
    snr = float(np.mean([f.nominal_snr for f in frames]))
    images = [f.image_dn for f in frames]
    background_e, background_std = calibrate_background(frames[:CALIBRATION_FRAMES], camera, hot)

    calls = {name: (lambda i, f=func: f(images[i], camera, hot)) for name, func in SHIPPED.items()}
    calls["tracking_weighted_centroid"] = lambda i: tracking_weighted_centroid(
        images[i], camera, hot, background_e, background_std, priors[i]
    )

    rows = []
    results = {}
    for name, call in calls.items():
        results[name] = [call(i) for i in range(n_frames)]
        ok = np.array([r.ok for r in results[name]])
        xy = np.array([[r.x, r.y] for r in results[name]])
        error = (xy - truths)[ok]
        iterations = np.array([r.n_iter for r in results[name]])[ok]
        rows.append({
            "preset": preset, "snr": snr, "method": name,
            "n_frames": n_frames, "repeats": repeats,
        } | summarise_times(time_calls(call, n_frames, repeats)) | {
            "refused": float(1 - ok.mean()),
            "rms_error_px": float(np.sqrt((error**2).mean())) if ok.any() else float("nan"),
            "iterations_mean": float(iterations.mean()) if ok.any() else float("nan"),
            "iterations_max": int(iterations.max()) if ok.any() else 0,
        })

    # How far the tracking-mode answer is from the shipped weighted centroid
    # on frames both accepted: the price of the shortcuts, apart from noise.
    shipped, tracking = results["iteratively_weighted_centroid"], results["tracking_weighted_centroid"]
    both = [(a, b) for a, b in zip(shipped, tracking) if a.ok and b.ok]
    difference = np.array([[a.x - b.x, a.y - b.y] for a, b in both])
    rows[-1]["rms_difference_from_shipped_px"] = (
        float(np.sqrt((difference**2).mean())) if len(both) else float("nan")
    )

    # The front end on its own; it has no position output to score.
    front = lambda i: est.preprocess(images[i], camera, hot)
    rows.append({
        "preset": preset, "snr": snr, "method": "preprocess",
        "n_frames": n_frames, "repeats": repeats,
    } | summarise_times(time_calls(front, n_frames, repeats)))

    for row in rows:
        row["verdict"] = verdict(row["p99_us"])
    return rows


def measure_front_end_parts(preset: str, n_frames: int, repeats: int) -> list[dict]:
    """Median time of each step inside `preprocess`, at the preset's signal."""
    camera, hot, frames, _, _ = make_frames(preset, None, n_frames, seed_offset=99)
    images = [f.image_dn for f in frames]
    electrons = [est.dn_to_electrons(image, camera) for image in images]
    valid = ~hot
    rough = [est.fit_background_plane(e, valid, None, 0.0)[0] for e in electrons]
    peaks = [est.find_peak(e - r, valid) for e, r in zip(electrons, rough)]
    planes = [est.fit_background_plane(e, valid, p, est.BACKGROUND_EXCLUDE_RADIUS)
              for e, p in zip(electrons, peaks)]
    saturated = np.zeros(camera.shape, dtype=bool)
    windows = [est.cut_window(e, plane, valid, saturated, p, est.HALF_WIDTH, camera.read_noise_e, std)
               for e, p, (plane, std) in zip(electrons, peaks, planes)]

    parts = {
        "DN to electrons": lambda i: est.dn_to_electrons(images[i], camera),
        "Background plane, rough (all pixels)":
            lambda i: est.fit_background_plane(electrons[i], valid, None, 0.0),
        "Peak search (3 x 3 box, maximum)": lambda i: est.find_peak(electrons[i] - rough[i], valid),
        "Background plane, final (spot excluded)":
            lambda i: est.fit_background_plane(electrons[i], valid, peaks[i],
                                               est.BACKGROUND_EXCLUDE_RADIUS),
        "Cut window": lambda i: est.cut_window(electrons[i], planes[i][0], valid, saturated, peaks[i],
                                               est.HALF_WIDTH, camera.read_noise_e, planes[i][1]),
        "Centroid of window (re-centring check)": lambda i: est.centroid_of(windows[i]),
        "Whole front end": lambda i: est.preprocess(images[i], camera, hot),
    }
    return [{"preset": preset, "step": step} | summarise_times(time_calls(call, n_frames, repeats))
            for step, call in parts.items()]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def machine_description() -> str:
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return (f"{chip}, {platform.system()} {platform.release()}, Python {platform.python_version()}, "
            f"NumPy {np.__version__}, SciPy {scipy.__version__}, one linear-algebra thread")


def plot_times(path: Path, preset: str, rows: list[dict]) -> None:
    """
    One row per method: a marker at the median time per frame and a line out
    to the 99th percentile. Markers, not bars, because the axis is
    logarithmic and a bar's length means nothing without a zero to start from.
    """
    order = ["preprocess", "windowed_centroid", "iteratively_weighted_centroid",
             "tracking_weighted_centroid", "gaussian_fit_lsq", "gaussian_fit_mle"]
    by_method = {r["method"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    for position, name in enumerate(order):
        r = by_method[name]
        style = STYLE.get(name, {"color": NEUTRAL, "marker": "o", "open": False})
        colour = style["color"]
        ax.plot([r["median_us"], r["p99_us"]], [position, position], color=colour, linewidth=2,
                zorder=3, solid_capstyle="butt")
        ax.plot([r["p99_us"]] * 2, [position - 0.2, position + 0.2], color=colour, linewidth=2,
                zorder=3, solid_capstyle="butt")
        ax.plot(r["median_us"], position, marker=style["marker"], markersize=9, linestyle="none",
                markerfacecolor=SURFACE if style["open"] else colour, markeredgecolor=colour,
                markeredgewidth=1.5, zorder=4)
    # What the marks mean goes in the legend: the charts carry no other text.
    ax.plot([], [], marker="o", markersize=9, linestyle="none", color=NEUTRAL, label="Median")
    ax.plot([], [], marker="|", markersize=11, markeredgewidth=2, linestyle="none", color=NEUTRAL,
            label="99th percentile")
    for limit, label, dash in ((BUDGET_US, f"Budget, {BUDGET_US:.0f} µs", ":"),
                               (PERIOD_US, f"Frame period, {PERIOD_US:,.0f} µs", "--")):
        ax.axvline(limit, color=TEXT_SECONDARY, linewidth=1.2, linestyle=dash, zorder=2, label=label)
    style_axes(ax)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(False, axis="y")
    ax.set_xscale("log")
    ax.set_xlim(20, 20_000)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[name] for name in order])
    ax.set_xlabel("Time per frame (µs)")
    ax.set_title(f"Time per frame, {preset} preset")
    fig.tight_layout(rect=(0, 0, 1, 1 - TITLE_BLOCK_IN / 4.4))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(path: Path, timings: dict, parts: dict) -> None:
    lines = [
        "# Real-time cost",
        "",
        "Generated by `src/realtime.py`; do not edit by hand.",
        "",
        f"Machine: {machine_description()}.",
        "",
        f"Frame period {PERIOD_US:,.0f} µs (1 kHz). Budget for the estimator: {BUDGET_US:.0f} µs "
        "(an assumption). Times are per frame, in microseconds, over every call. "
        "Floor: median of each frame's fastest repeat. The verdict uses the 99th percentile.",
        "",
    ]
    for preset, rows in timings.items():
        lines += [f"## {preset}: time per frame", "",
                  "| SNR | Method | Median | 99% | Max | Floor | Steps (mean / max) | Refused | "
                  "RMS error (px) | Verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            scored = "rms_error_px" in r
            lines.append(
                f"| {r['snr']:.0f} | {LABELS[r['method']]} | {r['median_us']:,.0f} | {r['p99_us']:,.0f} "
                f"| {r['max_us']:,.0f} | {r['floor_us']:,.0f} | "
                + (f"{r['iterations_mean']:.1f} / {r['iterations_max']}" if scored and r['iterations_max'] else "")
                + " | " + (f"{100 * r['refused']:.1f}%" if scored else "")
                + " | " + (f"{r['rms_error_px']:.4f}" if scored else "")
                + f" | {r['verdict']} |")
        lines += ["", f"### {preset}: tracking mode against the shipped weighted centroid", "",
                  "Same frames. The tracking-mode window is cut around a previous position that is "
                  f"wrong by {PRIOR_ERROR_PX} px RMS per axis. Difference: RMS distance between the two "
                  "answers, per axis.", "",
                  "| SNR | RMS error, shipped (px) | RMS error, tracking mode (px) | Difference (px) "
                  "| Median time, shipped | Median time, tracking mode | Speed-up |",
                  "|---|---|---|---|---|---|---|"]
        for snr in sorted({r["snr"] for r in rows}):
            at = {r["method"]: r for r in rows if r["snr"] == snr}
            shipped, tracking = at["iteratively_weighted_centroid"], at["tracking_weighted_centroid"]
            lines.append(
                f"| {snr:.0f} | {shipped['rms_error_px']:.4f} | {tracking['rms_error_px']:.4f} "
                f"| {tracking['rms_difference_from_shipped_px']:.4f} | {shipped['median_us']:,.0f} "
                f"| {tracking['median_us']:,.0f} | {shipped['median_us'] / tracking['median_us']:.0f}x |")
        whole = next(r for r in parts[preset] if r["step"] == "Whole front end")
        lines += ["", f"### {preset}: where the front end's time goes", "",
                  "Each step timed on its own, at the preset's signal. The steps do not sum exactly "
                  "to the whole: the whole also re-centres the window when needed.", "",
                  "| Step | Median (µs) | Share of the whole |", "|---|---|---|"]
        for r in parts[preset]:
            lines.append(f"| {r['step']} | {r['median_us']:.1f} "
                         f"| {100 * r['median_us'] / whole['median_us']:.0f}% |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="distinct frames per condition")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="timing passes over the frames")
    parser.add_argument("--presets", nargs="+", default=["silicon", "ingaas"])
    parser.add_argument("--quick", action="store_true",
                        help="40 frames, 1 timing pass: a smoke test, not a result")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.quick:
        args.frames, args.repeats = 40, 1
    args.output.mkdir(parents=True, exist_ok=True)

    timings, parts = {}, {}
    for preset in args.presets:
        timings[preset] = []
        for k, snr in enumerate((*SNRS, None)):
            rows = measure_condition(preset, snr, args.frames, args.repeats, seed_offset=k)
            timings[preset] += rows
            print(f"{preset} SNR {rows[0]['snr']:6.0f}: "
                  + "  ".join(f"{r['median_us']:.0f}" for r in rows) + "  (median µs)", flush=True)
        timings[preset].sort(key=lambda r: r["snr"])
        parts[preset] = measure_front_end_parts(preset, args.frames, args.repeats)
        own_snr = max(r["snr"] for r in rows)  # the last condition is the preset's own signal
        plot_times(args.output / f"time_per_frame_{preset}.png", preset,
                   [r for r in timings[preset] if r["snr"] == own_snr])
        write_csv(args.output / f"timing_{preset}.csv", timings[preset])
        write_csv(args.output / f"front_end_parts_{preset}.csv", parts[preset])

    write_tables(args.output / "timing.md", timings, parts)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
