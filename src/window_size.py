"""
Window size: position error of each estimator against the size of its window.

The window is the square of pixels around the spot that an estimator works
on. Every estimator uses 11 x 11 pixels (estimators.HALF_WIDTH = 5). This
script is the evidence for that choice. A small window cuts the tails of the
spot, which biases the centroid; a large one adds pixels that hold no spot
light but still add noise. The best size is where the two meet.

For each camera preset and each of three signal levels, the same frames are
given to every estimator at every window size, so a difference between two
sizes is not sampling noise. The spot sits at a random position inside one
pixel. One sensor (seed 42), the one the later experiments use.

Everything is written to results/estimators/:

  window_size.csv   one row per preset, SNR, window size and estimator
  window_size.md    the same numbers as readable tables (generated)

Conventions are those of simulator.py: positions are (x, y) in pixels, pixel
centres have integer coordinates.
"""

import os

# One thread for the linear-algebra library, set before numpy loads (see the
# same lines in characterize.py).
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

import estimators as est
from plot_style import STYLE
from precision_floor import cramer_rao_floor_px
from simulator import (
    load_config,
    make_rngs,
    make_sensor_maps,
    signal_for_target_snr,
    simulate_frame,
)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "estimators"

SEED = 42                        # the sensor, and the root of the frame streams
HALF_WIDTHS = (3, 4, 5, 6, 7)    # windows of 7, 9, 11, 13 and 15 pixels
TARGET_SNRS = (20.0, 90.0, 300.0)  # dim, near the silicon preset's own signal, bright; none saturates
DEFAULT_FRAMES = 2000
GROSS_ERROR_PX = 1.0             # an accepted estimate further than this from the truth

METHODS = {
    "windowed_centroid": est.windowed_centroid,
    "iteratively_weighted_centroid": est.iteratively_weighted_centroid,
    "gaussian_fit_lsq": est.gaussian_fit_lsq,
    "gaussian_fit_mle": est.gaussian_fit_mle,
}


def run_job(job: tuple) -> list[dict]:
    """
    One preset at one SNR: simulate the frames once and run every estimator
    at every window size on each of them. Returns one row per window size
    and estimator.
    """
    preset, snr_index, target_snr, n_frames = job
    camera, spot, background = load_config(preset)
    sensor_rng, _ = make_rngs(SEED)
    maps = make_sensor_maps(camera, sensor_rng)
    hot = est.hot_pixel_mask(maps, camera)

    # Its own stream per job, so the numbers do not depend on which presets
    # are run, on the number of workers or on the order they finish in.
    rng = np.random.default_rng([SEED, sum(ord(c) for c in preset), snr_index])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the near-saturation warning; none of these levels saturates
        total_signal_e = signal_for_target_snr(target_snr, spot, camera, background, maps)
    floor_px = cramer_rao_floor_px(camera, spot, background, total_signal_e)

    centre_x, centre_y = (round(c) for c in spot.center_xy)
    offsets = rng.uniform(-0.5, 0.5, size=(n_frames, 2))
    keys = [(h, name) for h in HALF_WIDTHS for name in METHODS]
    errors = {key: np.full((n_frames, 2), np.nan) for key in keys}
    snr = np.empty(n_frames)

    for i, (dx, dy) in enumerate(offsets):
        truth = (centre_x + dx, centre_y + dy)
        frame = simulate_frame(replace(spot, center_xy=truth, total_signal_e=total_signal_e),
                               camera, background, maps, rng)
        snr[i] = frame.nominal_snr
        for half_width, name in keys:
            result = METHODS[name](frame.image_dn, camera, hot, half_width=half_width)
            if result.ok:
                errors[half_width, name][i] = (result.x - truth[0], result.y - truth[1])

    rows = []
    for half_width, name in keys:
        e = errors[half_width, name]
        e = e[np.isfinite(e[:, 0])]
        n_ok = len(e)
        row = {
            "preset": preset, "target_snr": target_snr, "snr": float(snr.mean()),
            "signal_e": total_signal_e, "floor_px": floor_px,
            "window_px": 2 * half_width + 1, "estimator": name,
            "n_frames": n_frames, "n_ok": n_ok, "failure_rate": 1 - n_ok / n_frames,
        }
        if n_ok >= 2:
            rms = float(np.sqrt((e**2).mean()))
            # Standard error of an RMS over m samples is rms / sqrt(2m); here
            # m = 2 * n_ok because both axes are pooled.
            row |= {
                "rms_px": rms, "rms_se_px": rms / (2 * np.sqrt(n_ok)),
                "rms_over_floor": rms / floor_px,
                "bias_x_px": float(e[:, 0].mean()), "bias_y_px": float(e[:, 1].mean()),
                "gross_error_rate": float((np.abs(e).max(axis=1) > GROSS_ERROR_PX).mean()),
            }
        else:
            row |= {key: float("nan") for key in (
                "rms_px", "rms_se_px", "rms_over_floor", "bias_x_px", "bias_y_px", "gross_error_rate")}
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(path: Path, rows: list[dict], n_frames: int) -> None:
    """The numbers of window_size.csv as readable Markdown."""
    lines = [
        "# Window size",
        "",
        "Generated by `src/window_size.py`; do not edit by hand. "
        f"Sensor {SEED}, {n_frames} frames per SNR, spot at a random position inside one pixel. "
        "The same frames are used for every window size and estimator.",
        "",
        "RMS error per axis in pixels, over the frames the estimator accepted, ± one standard "
        "error. In brackets: the RMS error divided by the Cramér-Rao floor. "
        "The window every estimator uses is 11 x 11.",
        "",
    ]
    presets = list(dict.fromkeys(r["preset"] for r in rows))
    for preset in presets:
        mine = [r for r in rows if r["preset"] == preset]
        snrs = sorted({r["target_snr"] for r in mine})
        at = {(r["target_snr"], r["window_px"], r["estimator"]): r for r in mine}
        windows = sorted({r["window_px"] for r in mine})
        for name in METHODS:
            lines += [f"## {preset}: {STYLE[name]['label']}", "",
                      "| Window | " + " | ".join(f"SNR {s:g}" for s in snrs) + " |",
                      "|---|" + "---|" * len(snrs)]
            for window in windows:
                cells = []
                for s in snrs:
                    r = at[s, window, name]
                    cell = f"{r['rms_px']:.4f} ± {r['rms_se_px']:.4f} ({r['rms_over_floor']:.2f})"
                    if r["failure_rate"] > 0:
                        cell += f", {100 * r['failure_rate']:.1f}% refused"
                    cells.append(cell)
                lines.append(f"| {window} x {window} | " + " | ".join(cells) + " |")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="frames per SNR")
    parser.add_argument("--presets", nargs="+", default=["silicon", "ingaas"])
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--quick", action="store_true",
                        help="60 frames per SNR: a smoke test, not a result")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.quick:
        args.frames = 60
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [(preset, k, snr, args.frames)
            for preset in args.presets for k, snr in enumerate(TARGET_SNRS)]
    results = {}
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            preset, k, snr, _ = futures[future]
            results[preset, k] = future.result()
            print(f"  [{done}/{len(jobs)}] {preset} SNR {snr:g}", flush=True)

    # Job order, not completion order, so the files are the same on every run.
    rows = [row for preset, k, *_ in jobs for row in results[preset, k]]
    write_csv(args.output / "window_size.csv", rows)
    write_tables(args.output / "window_size.md", rows, args.frames)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
