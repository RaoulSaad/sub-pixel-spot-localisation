"""
The moving-spot experiments: what is simulated, measured and summarised.

`run_record` simulates one record of the moving spot, measures every frame
with each estimator and with the loop of realtime.py (tracking mode with the
full-frame estimator as fallback), and analyses each measured series.
`run_knock` moves the spot suddenly between two frames. `self_check` tests
the analysis on synthetic series with no camera. `build_jobs` lists the
records of the main experiment and of every sweep; `summarise_group` turns
the records of one condition into one line of results.

The first three run in worker processes. The command line is dynamic.py.
"""

import csv
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import estimators as est
from precision_floor import cramer_rao_floor_px
from realtime import (
    CALIBRATION_FRAMES, calibrate_background, track_with_fallback, tracking_weighted_centroid,
)
from simulator import (
    load_config,
    make_rngs,
    make_sensor_maps,
    signal_for_target_snr,
    simulate_frame,
)
from spectral import (
    FALSE_ALARM, alias_frequency, analyse, averaged_spectrum, clean_series, detect_line, fit_tone,
    jitter_psd, measurement_psd, spectrum, tone_precision, unsteady_fit_factor, wander_psd,
)
from trajectory import Scenario, exposure_samples, make_trajectory, per_frame

SEED = 42                    # also the sensor the other experiments use
TRACKER = "iteratively_weighted_centroid"  # full-frame form of the estimator realtime.py picks
# The loop of realtime.py: the same estimator in tracking mode, fed its own
# previous estimate, with TRACKER as fallback when it refuses a frame.
LOOP = "tracking_weighted_centroid"
LOOP_LABEL = "Weighted centroid, tracking mode + fallback"
METHODS = {
    "windowed_centroid": est.windowed_centroid,
    "iteratively_weighted_centroid": est.iteratively_weighted_centroid,
    "gaussian_fit_lsq": est.gaussian_fit_lsq,
    "gaussian_fit_mle": est.gaussian_fit_mle,
}
GROSS_ERROR_PX = 1.0         # same meaning as in characterize.py

# Sudden jump between two frames (tracking mode only).
KNOCK_JUMPS_PX = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
KNOCK_SNRS = (20.0, 200.0)
KNOCK_TRIALS = 300
KNOCK_MATCH_PX = 0.5         # an estimate further than this from the truth, on either axis, is wrong
KNOCK_FOLLOW_FRAMES = 5      # frames after the jump given to tracking mode to recover alone

SWEEP_AMPLITUDES_PX = (0.0, 0.03, 0.05, 0.07, 0.10, 0.15)
SWEEP_FREQUENCIES_HZ = (0.4, 3.0, 24.37, 150.0, 400.0, 620.0)
SWEEP_SNRS = (6.0, 8.0, 12.0, 20.0, 50.0)
# Total change of the disturbance frequency over the record. 0.1 Hz is one
# spectrum bin of a 10 s record.
SWEEP_DRIFTS_HZ = (0.0, 0.1, 0.2, 0.5)

# A disturbance above the jitter band, where the camera sets the floor.
CAMERA_LIMITED_HZ = 100.0             # twice the 50 Hz mains
CAMERA_LIMITED_AMPLITUDE_PX = 0.02
CAMERA_LIMITED_SNRS = (20.0, 12.0)


# --------------------------------------------------------------------------
# One record
# --------------------------------------------------------------------------

def run_record(job: dict) -> dict:
    """
    Simulate one record, measure every frame with each requested estimator,
    and analyse the result. `job` holds: tag, preset, scenario, realisation,
    methods (names), keep_series (return the raw series for plotting).

    Random streams. The trajectory depends on (tag, realisation) only, so
    both camera presets watch the same motion. The frame noise also depends
    on the preset. The sensor is the fixed one of the other experiments. The
    frames that calibrate the tracking-mode background have a stream of
    their own, so adding them did not change any frame of the record.

    The loop (LOOP). Before the record starts, the background plane is
    measured once from CALIBRATION_FRAMES frames with the spot at its first
    position. The first frame is found by the full-frame estimator. From
    then on each frame goes to tracking mode around the previous estimate,
    and to the full-frame estimator only if tracking mode refuses it. If
    both refuse, the previous position is kept for the next frame.
    """
    scenario: Scenario = job["scenario"]
    preset, realisation = job["preset"], job["realisation"]
    camera, spot, background = load_config(preset)
    maps = make_sensor_maps(camera, make_rngs(SEED)[0])
    hot = est.hot_pixel_mask(maps, camera)
    tag_key = sum(ord(c) for c in job["tag"])
    # Sweep points normally draw their own trajectories. A job may name a
    # shared `seed_point` instead, so that every point of that experiment
    # watches the same drift, jitter and noise and differs only in what is
    # being varied.
    seed_point = job.get("seed_point", job.get("point", 0))
    trajectory_rng = np.random.default_rng([SEED, tag_key, realisation, seed_point])
    frame_rng = np.random.default_rng(
        [SEED, tag_key, realisation, seed_point, sum(ord(c) for c in preset)]
    )

    trajectory = make_trajectory(scenario, trajectory_rng)
    n_exposed = exposure_samples(scenario, camera.exposure_s)
    paths = trajectory["total"].reshape(scenario.n_frames, scenario.sub_steps, 2)[:, :n_exposed]
    t = per_frame(trajectory["t"], scenario, n_exposed)   # mid-exposure times
    truth = paths.mean(axis=1)

    start = replace(spot, center_xy=scenario.start_xy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = signal_for_target_snr(scenario.target_snr, start, camera, background, maps)
    moving = replace(spot, total_signal_e=total)
    static_floor = cramer_rao_floor_px(camera, start, background, total)

    calibration_rng = np.random.default_rng(
        [SEED, tag_key, realisation, seed_point, sum(ord(c) for c in preset), 1]
    )
    calibration = [
        simulate_frame(moving, camera, background, maps, calibration_rng, path_xy=paths[0])
        for _ in range(CALIBRATION_FRAMES)
    ]
    try:
        plane, plane_scatter = calibrate_background(calibration, camera, hot)
    except est.EstimationFailure:
        plane, plane_scatter = None, None   # no spot seen: every frame goes full-frame

    estimates = {name: np.full((scenario.n_frames, 2), np.nan) for name in (*job["methods"], LOOP)}
    snr = np.empty(scenario.n_frames)
    last_xy, n_fallback = None, 0
    for k in range(scenario.n_frames):
        frame = simulate_frame(moving, camera, background, maps, frame_rng, path_xy=paths[k])
        snr[k] = frame.nominal_snr
        full_frame = None
        for name in job["methods"]:
            result = METHODS[name](frame.image_dn, camera, hot)
            if result.ok:
                estimates[name][k] = (result.x, result.y)
            if name == TRACKER:
                full_frame = result
        result, fell_back = track_with_fallback(
            frame.image_dn, camera, hot, plane, plane_scatter, last_xy, full_frame)
        n_fallback += fell_back
        if result.ok:
            estimates[LOOP][k] = last_xy = (result.x, result.y)

    base = {
        "tag": job["tag"], "preset": preset, "realisation": realisation,
        "snr": float(snr.mean()), "signal_e": total,
        "injected_hz": scenario.disturbance_hz,
        "injected_amplitude_px": scenario.disturbance_amplitude_px,
        "injected_angle_deg": scenario.disturbance_angle_deg,
        "injected_drift_hz": scenario.disturbance_drift_hz,
        "target_snr": scenario.target_snr,
        "static_floor_px": static_floor,
    }
    apparent = alias_frequency(scenario.disturbance_hz, scenario.frame_rate_hz)
    model_floor = float(jitter_psd(apparent, scenario)[0] + wander_psd(apparent, scenario)[0])

    rows, length_rows = [], []
    # The truth itself, analysed the same way: what a perfect camera would
    # conclude. The gap between this and an estimator is the camera's share.
    series = {"truth": truth} | estimates
    for name, xy in series.items():
        ok = np.isfinite(xy[:, 0])
        error = (xy - truth)[ok]
        gross = np.abs(error).max(axis=1) > GROSS_ERROR_PX if ok.any() else np.array([])
        rms = float(np.sqrt((error**2).mean())) if ok.any() else float("nan")
        if name != "truth" and ok.sum() > 2:
            # Error against where the spot sat inside a pixel, x and y together:
            # the centroid's window-truncation bias should show as a slope of
            # about -0.02.
            offset = (truth - np.round(truth))[ok].ravel()
            slope = float(np.polyfit(offset, error.ravel(), 1)[0])
        else:
            slope = float("nan")
        recovery = {
            "method": name,
            "subpixel_slope": slope,
            "refused": float(1 - ok.mean()),
            # Frames that tracking mode refused and the full-frame estimator took over.
            "fallback": n_fallback / scenario.n_frames if name == LOOP else float("nan"),
            "gross": float(gross.mean()) if ok.any() else float("nan"),
            "rms_error_px": rms,
            "rms_over_floor": rms / static_floor,
        }
        if name == LOOP and ok.any() and TRACKER in estimates:
            both = ok & np.isfinite(estimates[TRACKER][:, 0])
            apart = (xy - estimates[TRACKER])[both]
            # How far the loop's answer is from the full-frame one on the same frames.
            recovery["rms_from_full_frame_px"] = float(np.sqrt((apart**2).mean()))
        measured = 0.0 if name == "truth" else (rms if np.isfinite(rms) else 0.0)
        floor = model_floor + measurement_psd(measured, scenario)
        try:
            rows.append(base | recovery | analyse(t, xy, scenario, n_exposed, floor,
                                                  job["thresholds"]))
            if job.get("record_lengths") and name in ("truth", TRACKER):
                # The record lengths come from the thresholds, which the main
                # process prepared one per length to be analysed.
                for n in sorted(job["thresholds"]):
                    if n <= scenario.n_frames:
                        length_rows.append(base | {"method": name} | analyse(
                            t[:n], xy[:n], scenario, n_exposed, floor, job["thresholds"]))
        except ValueError as exc:
            rows.append(base | recovery | {"analysis_failed": str(exc), "detected": False,
                                           "correct": False})

    out = {"job": {k: v for k, v in job.items() if k not in ("scenario", "thresholds")}, "rows": rows,
           "length_rows": length_rows}
    if job.get("keep_spectra"):
        out["spectra"] = {}
        for name, xy in series.items():
            try:
                out["spectra"][name] = averaged_spectrum(clean_series(t, xy)[1],
                                                         scenario.frame_rate_hz)
            except ValueError:
                pass
    if job.get("keep_series"):
        out["series"] = {
            "t": t, "truth": truth, "estimates": estimates, "phase": trajectory["phase"],
            "drift": per_frame(trajectory["drift"], scenario, n_exposed),
            "jitter": per_frame(trajectory["jitter"], scenario, n_exposed),
            "disturbance": per_frame(trajectory["disturbance"], scenario, n_exposed),
        }
    return out


# --------------------------------------------------------------------------
# A sudden jump between two frames
# --------------------------------------------------------------------------

def run_knock(job: dict) -> list[dict]:
    """
    How far can the spot jump between two frames before tracking mode loses
    it? `job` holds: preset, snr, trials, start_xy.

    One trial: the spot sits at a random position within half a pixel of
    `start_xy`. The full-frame estimator measures one frame there; that is
    the position the tracker holds. The spot then jumps by a set distance in
    a random direction, between two exposures (no blur across the jump), and
    stays. Tracking mode alone, with no fallback, measures the next frames,
    each from its own previous answer.

    Counted on the first frame after the jump:
      followed  accepted, and within KNOCK_MATCH_PX of the truth on both axes
      refused   tracking mode reported a failure (the loop would fall back)
      wrong     accepted, but further than KNOCK_MATCH_PX from the truth
    and two more:
      alone     still (or again) following KNOCK_FOLLOW_FRAMES frames later,
                with no fallback
      with_fallback  the loop's answer on the first frame (tracking mode if
                it accepted, the full-frame estimator if not) is within
                KNOCK_MATCH_PX
    """
    preset, target_snr = job["preset"], job["snr"]
    camera, spot, background = load_config(preset)
    maps = make_sensor_maps(camera, make_rngs(SEED)[0])
    hot = est.hot_pixel_mask(maps, camera)
    rng = np.random.default_rng([SEED, 5, sum(ord(c) for c in preset), int(target_snr)])
    start = np.asarray(job["start_xy"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = signal_for_target_snr(target_snr, replace(spot, center_xy=tuple(start)), camera,
                                      background, maps)
    at = lambda xy: replace(spot, total_signal_e=total, center_xy=(float(xy[0]), float(xy[1])))
    plane, scatter = calibrate_background(
        [simulate_frame(at(start), camera, background, maps, rng) for _ in range(CALIBRATION_FRAMES)],
        camera, hot)

    def close(estimate, truth) -> bool:
        return bool(estimate.ok and abs(estimate.x - truth[0]) < KNOCK_MATCH_PX
                    and abs(estimate.y - truth[1]) < KNOCK_MATCH_PX)

    rows = []
    for jump in job["jumps"]:
        counts = dict.fromkeys(("followed", "refused", "wrong", "alone", "with_fallback"), 0)
        wrong_by = []
        for _ in range(job["trials"]):
            before = start + rng.uniform(-0.5, 0.5, size=2)
            angle = rng.uniform(0.0, 2 * np.pi)
            after = before + jump * np.array([np.cos(angle), np.sin(angle)])
            held = est.iteratively_weighted_centroid(
                simulate_frame(at(before), camera, background, maps, rng).image_dn, camera, hot)
            last = (held.x, held.y) if held.ok else tuple(before)
            for step in range(KNOCK_FOLLOW_FRAMES + 1):
                image = simulate_frame(at(after), camera, background, maps, rng).image_dn
                tracked = tracking_weighted_centroid(image, camera, hot, plane, scatter, last)
                if step == 0:
                    counts["followed"] += close(tracked, after)
                    counts["refused"] += not tracked.ok
                    if tracked.ok and not close(tracked, after):
                        counts["wrong"] += 1
                        wrong_by.append(np.hypot(tracked.x - after[0], tracked.y - after[1]))
                    answer = tracked if tracked.ok else est.iteratively_weighted_centroid(
                        image, camera, hot)
                    counts["with_fallback"] += close(answer, after)
                if tracked.ok:
                    last = (tracked.x, tracked.y)
            counts["alone"] += close(tracked, after)
        rows.append({"preset": preset, "snr": target_snr, "jump_px": jump, "trials": job["trials"]}
                    | counts
                    | {"wrong_mean_distance_px": float(np.mean(wrong_by)) if wrong_by else float("nan")})
    return rows


# --------------------------------------------------------------------------
# Checks on the analysis itself, with no camera involved
# --------------------------------------------------------------------------

def self_check(scenario: Scenario, n_trials: int, rng: np.random.Generator,
               thresholds: dict) -> list[dict]:
    """
    Two tests of the analysis on synthetic series (positions drawn directly,
    no images), where the right answer is known in advance:

      1. No disturbance at all: how often is a line announced? It should be
         about FALSE_ALARM. This tests the floor estimate and the threshold
         on the scenario's own drift and jitter.
      2. A tone in plain white noise: do the scatter of the fitted amplitude
         and frequency match tone_precision()? This tests the fit and the
         formulae together.

    And a third with no noise at all: a tone whose frequency ramps, fitted
    the way analyse() does it (start from the spectrum's peak). Does the
    fitted amplitude match unsteady_fit_factor()?
    """
    quiet = replace(scenario, disturbance_amplitude_px=0.0)
    n_exposed = scenario.sub_steps // 2
    measurement_rms = 0.1
    lengths = [n for n in sorted(thresholds) if n <= quiet.n_frames]
    false_alarms = dict.fromkeys(lengths, 0)
    for _ in range(n_trials):
        trajectory = make_trajectory(quiet, rng)
        xy = per_frame(trajectory["total"], quiet, n_exposed)
        xy = xy + rng.normal(0.0, measurement_rms, size=xy.shape)
        for n in lengths:
            false_alarms[n] += detect_line(xy[:n], quiet.frame_rate_hz, thresholds[n])["detected"]

    t = (np.arange(scenario.n_frames) + 0.5) / scenario.frame_rate_hz
    noise_rms = 0.3
    floor = measurement_psd(noise_rms, scenario)
    amplitudes, frequencies = [], []
    for _ in range(n_trials):
        wave = scenario.disturbance_amplitude_px * np.sin(
            2 * np.pi * scenario.disturbance_hz * t + rng.uniform(0, 2 * np.pi))
        xy = np.outer(wave, scenario.direction) + rng.normal(0.0, noise_rms, size=(len(t), 2))
        tone = fit_tone(t, xy, scenario.disturbance_hz + rng.uniform(-0.4, 0.4) / scenario.duration_s)
        amplitudes.append(tone["amplitude_px"])
        frequencies.append(tone["frequency_hz"])
    sigma_a, sigma_f = tone_precision(scenario.disturbance_amplitude_px, floor, scenario.duration_s)

    ramp_checks = []
    for drift in SWEEP_DRIFTS_HZ:
        if drift == 0:
            continue
        ramp = drift * (t**2 / (2 * scenario.duration_s) - t / 2)
        wave = scenario.disturbance_amplitude_px * np.sin(
            2 * np.pi * scenario.disturbance_hz * t + 1.0 + 2 * np.pi * ramp)
        xy = np.outer(wave, scenario.direction)
        f, psd = spectrum(xy, scenario.frame_rate_hz)
        tone = fit_tone(t, xy, float(f[np.argmax(psd)]))
        ramp_checks.append({
            "check": f"Fitted share of the amplitude, noise-free tone ramping by {drift:g} Hz",
            "expected": f"{unsteady_fit_factor(drift, scenario.duration_s):.3f}",
            "measured": f"{tone['amplitude_px'] / scenario.disturbance_amplitude_px:.3f}"})
    return [
        {"check": f"False detections with no disturbance, {n / quiet.frame_rate_hz:g} s record "
                  f"(threshold {thresholds[n]:.2f})",
         "expected": f"{100 * FALSE_ALARM:.1f}%",
         "measured": f"{100 * count / n_trials:.1f}% ({count} of {n_trials})"}
        for n, count in false_alarms.items()
    ] + [
        {"check": "Scatter of fitted amplitude, tone in white noise (px)",
         "expected": f"{sigma_a:.5f}", "measured": f"{np.std(amplitudes, ddof=1):.5f}"},
        {"check": "Mean fitted amplitude minus true, tone in white noise (px)",
         "expected": "small and positive", "measured":
             f"{np.mean(amplitudes) - scenario.disturbance_amplitude_px:+.5f}"},
        {"check": "Scatter of fitted frequency, tone in white noise (Hz)",
         "expected": f"{sigma_f:.5f}", "measured": f"{np.std(frequencies, ddof=1):.5f}"},
    ] + ramp_checks


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def build_jobs(presets: list[str], n_realisations: int, scenario: Scenario,
               thresholds: dict) -> list[dict]:
    """
    The main experiment (every estimator, each preset), three sweeps
    (tracker and loop, first preset), the unsteady-frequency sweep (same)
    and the camera-limited disturbance (every estimator, each preset, two
    SNRs), one job per record.
    """
    jobs = [
        {"tag": "main", "preset": preset, "scenario": scenario, "realisation": r,
         "methods": list(METHODS), "keep_series": r == 0, "keep_spectra": True,
         "record_lengths": True, "thresholds": thresholds}
        for preset in presets for r in range(n_realisations)
    ]
    sweeps = (
        ("amplitude", "disturbance_amplitude_px", SWEEP_AMPLITUDES_PX),
        ("frequency", "disturbance_hz", SWEEP_FREQUENCIES_HZ),
        ("snr", "target_snr", SWEEP_SNRS),
    )
    for tag, field, values in sweeps:
        for point, value in enumerate(values):
            for r in range(n_realisations):
                jobs.append({"tag": tag, "preset": presets[0], "realisation": r, "point": point,
                             "scenario": replace(scenario, **{field: value}),
                             "methods": [TRACKER], "thresholds": thresholds})
    # The same eight trajectories at every drift value (seed_point), so the
    # steady row is the direct reference for the others.
    for point, drift in enumerate(SWEEP_DRIFTS_HZ):
        for r in range(n_realisations):
            jobs.append({"tag": "unsteady", "preset": presets[0], "realisation": r, "point": point,
                         "seed_point": 0,
                         "scenario": replace(scenario, disturbance_drift_hz=drift),
                         "methods": [TRACKER], "thresholds": thresholds})
    # Same trajectories for both presets and both SNRs.
    for preset in presets:
        for point, snr in enumerate(CAMERA_LIMITED_SNRS):
            for r in range(n_realisations):
                jobs.append({"tag": "camera", "preset": preset, "realisation": r, "point": point,
                             "seed_point": 0,
                             "scenario": replace(scenario, disturbance_hz=CAMERA_LIMITED_HZ,
                                                 disturbance_amplitude_px=CAMERA_LIMITED_AMPLITUDE_PX,
                                                 target_snr=snr),
                             "methods": list(METHODS), "thresholds": thresholds})
    return jobs


def mean_std(values) -> tuple[float, float, int]:
    """Mean and standard deviation of the finite values, and how many there were."""
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else float("nan"), finite.size


def summarise_group(rows: list[dict]) -> dict:
    """
    One line of results from the records of one condition and one method.
    Frequency, amplitude and direction are averaged over the records where
    the line was found at the right frequency; the count says how many.
    """
    right = [r for r in rows if r.get("correct")]
    out = {
        "records": len(rows),
        "detected": sum(bool(r.get("detected")) for r in rows),
        "correct": len(right),
        "refused": mean_std(r.get("refused") for r in rows)[0],
        "fallback": mean_std(r.get("fallback") for r in rows)[0],
        "rms_from_full_frame_px": mean_std(r.get("rms_from_full_frame_px") for r in rows)[0],
        "gross": mean_std(r.get("gross") for r in rows)[0],
        "gated": mean_std(r.get("gated") for r in rows)[0],
    }
    for key in ("snr", "rms_error_px", "rms_over_floor", "static_floor_px", "subpixel_slope",
                "line_ratio", "threshold",
                "seen_amplitude_px", "apparent_hz", "predicted_sigma_amplitude_px",
                "predicted_sigma_frequency_hz", "expected_line_ratio"):
        out[key], out[f"{key}_std"], _ = mean_std(r.get(key) for r in rows)
    for key in ("frequency_hz", "amplitude_px", "cross_amplitude_px", "angle_deg",
                "frequency_error_hz", "amplitude_error_px", "angle_error_deg"):
        out[key], out[f"{key}_std"], _ = mean_std(r.get(key) for r in right)
    ratios = [r["line_ratio"] for r in rows if np.isfinite(r.get("line_ratio", np.nan))]
    out["line_ratio_min"] = min(ratios) if ratios else float("nan")
    out["line_ratio_max"] = max(ratios) if ratios else float("nan")
    return out


def select(rows: list[dict], **match) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
