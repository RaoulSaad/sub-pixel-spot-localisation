"""
Dynamic tracking: follow a moving spot and identify a periodic disturbance.

The spot moves along a known trajectory made of three parts: a slow drift,
random jitter, and one sinusoidal disturbance. Each frame is rendered with
the spot moving during the exposure, then every estimator from
estimators.py measures it. So does the loop of realtime.py: the weighted
centroid in tracking mode, which looks only around its own previous
estimate, with the full-frame estimator as fallback (realtime.py). From the
measured positions alone the script

  1. scores the recovered trajectory against the truth,
  2. searches the spectrum for a periodic line, and
  3. measures that line's frequency, amplitude and direction,

and compares all three with what was injected. Sweeps over the disturbance
amplitude, its frequency, the SNR and the record length show where this
stops working. Two further cases change the disturbance itself: a weak
tone at 100 Hz, where camera noise and not jitter is the floor, and a tone
whose frequency changes steadily during the record. A separate test moves
the spot suddenly between two frames (a knock) to find how far tracking
mode can follow.

Conventions are those of simulator.py: positions are (x, y) in pixels.
Spectra are one-sided power spectral densities in px^2/Hz: the area under
the curve between two frequencies is the variance of the motion in that
band.
"""

import os

# One thread for the linear-algebra library, set before numpy loads: the
# records run one per core already (see the same lines in characterize.py).
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from plot_style import STYLE
from spectral import NULL_TRIALS, null_threshold
from tracking_experiment import (
    CAMERA_LIMITED_SNRS, KNOCK_JUMPS_PX, KNOCK_SNRS, KNOCK_TRIALS, LOOP, METHODS, SEED,
    TRACKER,
    build_jobs, run_knock, run_record, select, self_check, summarise_group, write_csv,
)
from tracking_plots import (
    plot_amplitude_sweep, plot_detection, plot_error_over_time, plot_folded, plot_ratio_sweep,
    plot_snr_sweep, plot_spectrum, plot_trajectory, plot_zoom, write_tables,
)
from trajectory import Scenario

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "dynamic_tracking"

DEFAULT_REALISATIONS = 8
# Record lengths of the record-length experiment. Kept beside main(), which
# shortens the list in --quick mode.
RECORD_LENGTHS_S = (1.0, 2.0, 5.0, 10.0)


def draw_charts(output: Path, presets: list[str], scenario: Scenario, n_realisations: int,
                thresholds: dict, main_summary: dict, length_summary: dict, sweep_summary: dict,
                by_job: dict) -> None:
    """
    Every chart of this experiment. `by_job` holds the main-scenario records, keyed
    by ("main", preset, record number), with their series and spectra.
    """
    first = by_job["main", presets[0], 0]["series"]
    plot_trajectory(output / "scenario_trajectory.png", first, scenario)
    for preset in presets:
        series = by_job["main", preset, 0]["series"]
        spectra = [by_job["main", preset, r]["spectra"] for r in range(n_realisations)]
        floor = main_summary[preset][TRACKER]["static_floor_px"]
        plot_zoom(output / f"tracking_zoom_{preset}.png", preset, series, scenario)
        plot_error_over_time(output / f"tracking_error_{preset}.png", preset, series, floor,
                             scenario)
        plot_spectrum(output / f"spectrum_{preset}.png", preset, spectra, scenario)
        plot_detection(output / f"line_search_{preset}.png", preset, series, scenario,
                       thresholds[scenario.n_frames])
        plot_folded(output / f"disturbance_folded_{preset}.png", preset, series, scenario,
                    thresholds[scenario.n_frames])

    preset = presets[0]
    tracker_only = lambda tag: [s for s in sweep_summary[tag] if s["method"] == TRACKER]
    plot_amplitude_sweep(output / "sweep_amplitude.png", tracker_only("amplitude"), preset)
    plot_ratio_sweep(
        output / "sweep_frequency.png", tracker_only("frequency"), "injected_hz",
        "Injected frequency (Hz)", "Where the disturbance sits in the spectrum",
        f"{STYLE[TRACKER]['label']}, {preset} preset, {scenario.disturbance_amplitude_px:g} px "
        f"disturbance, {n_realisations} records of {scenario.duration_s:g} s.\nPoint: mean. Bars: "
        "lowest to highest record.")
    plot_ratio_sweep(
        output / "sweep_record_length.png",
        [s for s in length_summary[preset] if s["method"] == TRACKER], "duration_s",
        "Record length (s)", "How long a record is needed",
        f"{STYLE[TRACKER]['label']}, {preset} preset, the main scenario cut short, "
        f"{n_realisations} records.\nPoint: mean. Bars: lowest to highest record.")
    plot_snr_sweep(output / "sweep_snr.png", tracker_only("snr"), preset)
    plot_ratio_sweep(
        output / "sweep_unsteady.png", tracker_only("unsteady"), "injected_drift_hz",
        "Total change of the disturbance frequency over the record (Hz)",
        "A disturbance whose frequency is not steady",
        f"{STYLE[TRACKER]['label']}, {preset} preset, {scenario.disturbance_amplitude_px:g} px at "
        f"{scenario.disturbance_hz:g} Hz, the same {n_realisations} records of "
        f"{scenario.duration_s:g} s at every value.\nPoint: mean. Bars: lowest to highest record. "
        "One spectrum bin is "
        f"{1 / scenario.duration_s:g} Hz.", log_x=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--realisations", type=int, default=DEFAULT_REALISATIONS,
                        help="records per condition")
    parser.add_argument("--presets", nargs="+", default=["silicon", "ingaas"])
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--quick", action="store_true",
                        help="2 s records and 2 realisations: a smoke test, not a result")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    scenario = Scenario()
    n_realisations, n_checks, n_null, n_knock = args.realisations, 1000, NULL_TRIALS, KNOCK_TRIALS
    if args.quick:
        global RECORD_LENGTHS_S
        scenario = replace(scenario, duration_s=2.0)
        RECORD_LENGTHS_S = (1.0, 2.0)
        n_realisations, n_checks, n_null, n_knock = 2, 20, 200, 20

    # Detection threshold for every record length that will be analysed,
    # keyed by the number of frames.
    lengths = {int(round(s * scenario.frame_rate_hz)) for s in RECORD_LENGTHS_S}
    lengths.add(scenario.n_frames)
    print("setting detection thresholds from noise-only records", flush=True)
    thresholds = {
        n: null_threshold(n, scenario.frame_rate_hz, np.random.default_rng([SEED, 11, n]), n_null)
        for n in sorted(lengths)
    }

    jobs = build_jobs(args.presets, n_realisations, scenario, thresholds)
    print(f"{len(jobs)} records on {args.workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        check = pool.submit(self_check, scenario, n_checks, np.random.default_rng([SEED, 7]),
                            thresholds)
        knock_futures = [
            pool.submit(run_knock, {"preset": preset, "snr": snr, "trials": n_knock,
                                    "jumps": KNOCK_JUMPS_PX, "start_xy": scenario.start_xy})
            for preset in args.presets for snr in KNOCK_SNRS
        ]
        futures = [pool.submit(run_record, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            job = results[-1]["job"]
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] last: {job['tag']} {job['preset']} "
                      f"record {job['realisation']}", flush=True)
        checks = check.result()
        knocks = [row for future in knock_futures for row in future.result()]
    write_csv(args.output / "knock.csv", knocks)

    # Workers finish in any order; sort so the files do not depend on it.
    results.sort(key=lambda r: (r["job"]["tag"], r["job"]["preset"], r["job"].get("point", 0),
                                r["job"]["realisation"]))
    rows = [row for r in results for row in r["rows"]]
    length_rows = [row for r in results for row in r["length_rows"]]
    write_csv(args.output / "records.csv", rows)
    write_csv(args.output / "record_lengths.csv", length_rows)

    main_summary, length_summary = {}, {}
    for preset in args.presets:
        main_summary[preset] = {
            name: summarise_group(select(rows, tag="main", preset=preset, method=name))
            for name in ("truth", *METHODS, LOOP)
        }
        length_summary[preset] = [
            {"duration_s": length, "method": name} | summarise_group(
                select(length_rows, preset=preset, method=name, duration_s=length))
            for length in RECORD_LENGTHS_S for name in ("truth", TRACKER)
        ]
    sweep_summary = {}
    for tag in ("amplitude", "frequency", "snr", "unsteady"):
        sweep_summary[tag] = []
        for point in sorted({r["job"].get("point", 0) for r in results if r["job"]["tag"] == tag}):
            at_point = [row for r in results
                        if r["job"]["tag"] == tag and r["job"].get("point") == point
                        for row in r["rows"]]
            for name in ("truth", TRACKER, LOOP):
                group = select(at_point, method=name)
                summary = summarise_group(group)
                sweep_summary[tag].append(
                    {"method": name, "injected_amplitude_px": group[0]["injected_amplitude_px"],
                     "injected_hz": group[0]["injected_hz"],
                     "injected_drift_hz": group[0]["injected_drift_hz"]} | summary)
    camera_summary = {
        preset: [
            {"target_snr": snr, "method": name} | summarise_group(
                select(rows, tag="camera", preset=preset, method=name, target_snr=snr))
            for snr in CAMERA_LIMITED_SNRS for name in ("truth", *METHODS, LOOP)
        ]
        for preset in args.presets
    }

    by_job = {(r["job"]["tag"], r["job"]["preset"], r["job"]["realisation"]): r for r in results
              if r["job"]["tag"] == "main"}
    draw_charts(args.output, args.presets, scenario, n_realisations, thresholds, main_summary,
                length_summary, sweep_summary, by_job)

    write_tables(args.output / "tables.md", scenario, args.presets, main_summary, length_summary,
                 sweep_summary, checks, n_realisations, knocks, camera_summary)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
