"""
Real-world conditions: what outdoor operation does to the position estimate.

Five conditions are simulated. Two are run on the moving scenario of
dynamic.py, frame after frame, because they change over time:

  1. Scintillation: the spot's brightness fades and flares from frame to
     frame, while its position follows the moving-spot trajectory.
  2. Changing daylight: the background brightens through the record.

Three are run on single frames with a static spot, because they are a
change of the image and not of its history:

  3. Spot shape: the spot is wider, narrower or elliptical.
  4. A second light source: a reflection or another beam in the frame.
  5. Sensor temperature: more dark current than at calibration.

Each is compared with a baseline that is identical except for the
condition. Everything is written to results/real_world/.

The estimators, the loop (tracking mode with fallback) and the spectral
analysis are those of the other experiments, unchanged.
"""

import os

# One thread for the linear-algebra library, set before numpy loads: the
# records run one per core already.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import signal

import estimators as est
import plot_style
import spectral
import tracking_experiment as experiment
import tracking_plots
import trajectory as traj   # `trajectory` is a local variable in run_sequence
from plot_style import NEUTRAL, STYLE, TEXT, TEXT_SECONDARY, series_kwargs
from precision_floor import cramer_rao_floor_px
from realtime import (
    CALIBRATION_FRAMES, PRIOR_ERROR_PX,
    calibrate_background, track_with_fallback, tracking_weighted_centroid,
)
from simulator import (
    SensorMaps,
    load_config,
    make_rngs,
    make_sensor_maps,
    signal_for_target_snr,
    simulate_frame,
)
from tracking_experiment import LOOP, LOOP_LABEL, METHODS, TRACKER

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "real_world"

SEED = 42                    # also the sensor the other experiments use
DEFAULT_REALISATIONS = 8
DEFAULT_STATIC_FRAMES = 2000
GROSS_ERROR_PX = experiment.GROSS_ERROR_PX

# Scintillation. Every value is an assumption.
SCINTILLATION_INDICES = (0.0, 0.1, 0.5)   # variance of brightness / mean^2
CORRELATION_S = 0.020                     # how long a fade lasts
CORRELATION_SWEEP_S = (0.002, 0.2)        # also tried, at the largest index
# Brightness bins (multiples of the mean) for "error against brightness".
BRIGHTNESS_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, np.inf])

# Changing daylight: the background level and its gradient grow steadily to
# this multiple of their starting values by the end of the record.
DAYLIGHT_GAINS = (1.0, 3.0, 10.0, 30.0)
REFRESH_FRAMES = 100         # the refreshed loop re-measures its background this often
LOOP_REFRESH = "tracking_refreshed"
LOOP_REFRESH_LABEL = "Tracking mode + fallback, background refreshed"

# Single-frame conditions.
STATIC_SNRS = (20.0, 100.0)
SHAPE_SIGMAS = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0)   # circular spot, px; 1.75 is nominal
SHAPE_ELLIPSES = (1.25, 1.5)                      # sigma_x = 1.75 * r, sigma_y = 1.75 / r
SECOND_SPOT_OFFSETS_PX = (3.0, 6.0, 10.0)
SECOND_SPOT_BRIGHTNESS = (0.1, 0.5, 2.0)          # relative to the tracked spot
SECOND_SPOT_ANGLE_DEG = 30.0
DARK_MULTIPLIERS = (1.0, 10.0, 100.0, 1000.0)
TRACKING = "tracking_mode_alone"   # no fallback; distinct from the loop's series name
TRACKING_LABEL = "Weighted centroid, tracking mode"


def label(name: str) -> str:
    return {LOOP: LOOP_LABEL, LOOP_REFRESH: LOOP_REFRESH_LABEL, TRACKING: TRACKING_LABEL,
            "truth": "Perfect camera (true positions)"}.get(name) or STYLE[name]["label"]


# --------------------------------------------------------------------------
# Models of the conditions
# --------------------------------------------------------------------------

def brightness_sequence(n: int, index: float, correlation_s: float, frame_rate_hz: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Brightness of the spot in each frame, as a multiple of its mean.

    The logarithm of the brightness is a Gaussian series that remembers its
    past: each value is `rho` times the previous one plus fresh noise, with
    rho = exp(-frame period / correlation time). Exponentiating gives a
    lognormal series, which is what weak atmospheric turbulence produces:

        z[k] = rho * z[k-1] + sqrt(1 - rho^2) * e[k],   e ~ Normal(0, 1)
        brightness[k] = exp(s * z[k] - s^2 / 2),        s^2 = log(1 + index)

    Its mean is 1 and its variance is `index`, the scintillation index.
    The noise e is drawn whatever the index, so records that differ only in
    index fade at the same moments, more or less deeply.
    """
    noise = rng.standard_normal(n)
    if index <= 0:
        return np.ones(n)
    rho = np.exp(-1.0 / (frame_rate_hz * correlation_s))
    rest, _ = signal.lfilter([np.sqrt(1 - rho**2)], [1.0, -rho], noise[1:], zi=[rho * noise[0]])
    z = np.concatenate([noise[:1], rest])
    s2 = np.log1p(index)
    return np.exp(np.sqrt(s2) * z - s2 / 2)


def scaled_background(background, gain: float):
    """The same background shape, `gain` times brighter."""
    return replace(background, level_e=background.level_e * gain,
                   gradient_x_e_per_px=background.gradient_x_e_per_px * gain,
                   gradient_y_e_per_px=background.gradient_y_e_per_px * gain)


def floor_per_frame(camera, start, background, total: float, brightness: np.ndarray,
                    gain: np.ndarray) -> np.ndarray:
    """
    The Cramer-Rao floor of each frame, given its brightness and background.
    Evaluated on a grid of the quantity that varies and interpolated, since
    the floor changes smoothly with either. Only one of the two varies in
    any one experiment.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.ptp(gain) > 0:
            grid = np.geomspace(gain.min(), gain.max(), 24)
            floors = [cramer_rao_floor_px(camera, start, scaled_background(background, g), total)
                      for g in grid]
            return np.exp(np.interp(np.log(gain), np.log(grid), np.log(floors)))
        if np.ptp(brightness) > 0:
            grid = np.geomspace(brightness.min(), brightness.max(), 60)
            floors = [cramer_rao_floor_px(camera, start, background, total * b) for b in grid]
            return np.exp(np.interp(np.log(brightness), np.log(grid), np.log(floors)))
        return np.full(len(brightness), cramer_rao_floor_px(camera, start, background, total))


def gap_lengths(ok: np.ndarray) -> np.ndarray:
    """Lengths, in frames, of every run of consecutive frames with no estimate."""
    edges = np.diff(np.concatenate([[1], ok.astype(int), [1]]))
    return np.flatnonzero(edges == 1) - np.flatnonzero(edges == -1)


# --------------------------------------------------------------------------
# One record of the moving scenario under a condition
# --------------------------------------------------------------------------

def run_sequence(job: dict) -> dict:
    """
    One record of the moving-spot scenario with the spot's brightness and the
    background changing from frame to frame. `job` holds: case (name),
    preset, realisation, scenario, index, correlation_s, gain_end, methods,
    refresh (also run the loop with a refreshed background), thresholds,
    keep_series.

    Random streams. The trajectory and the fading depend on the realisation
    only, so every preset, method, index and gain watches the same motion
    and the same fades. The camera noise depends also on the preset.
    """
    scenario: traj.Scenario = job["scenario"]
    preset, realisation = job["preset"], job["realisation"]
    camera, spot, background = load_config(preset)
    maps = make_sensor_maps(camera, make_rngs(SEED)[0])
    hot = est.hot_pixel_mask(maps, camera)
    preset_key = sum(ord(c) for c in preset)
    trajectory = traj.make_trajectory(scenario, np.random.default_rng([SEED, 6, realisation]))
    fading_rng = np.random.default_rng([SEED, 6, realisation, 1])
    frame_rng = np.random.default_rng([SEED, 6, realisation, 2, preset_key])
    calibration_rng = np.random.default_rng([SEED, 6, realisation, 3, preset_key])

    n = scenario.n_frames
    n_exposed = traj.exposure_samples(scenario, camera.exposure_s)
    paths = trajectory["total"].reshape(n, scenario.sub_steps, 2)[:, :n_exposed]
    t = traj.per_frame(trajectory["t"], scenario, n_exposed)
    truth = paths.mean(axis=1)

    start = replace(spot, center_xy=scenario.start_xy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = signal_for_target_snr(scenario.target_snr, start, camera, background, maps)
    moving = replace(spot, total_signal_e=total)
    # The brightness is held for the whole exposure, so the frame's true
    # position stays the plain mean over the exposure.
    brightness = brightness_sequence(n, job["index"], job["correlation_s"],
                                     scenario.frame_rate_hz, fading_rng)
    gain = 1.0 + (job["gain_end"] - 1.0) * np.arange(n) / (n - 1)
    floors = floor_per_frame(camera, start, background, total, brightness, gain)

    # The loop's background, measured before the record under the starting
    # conditions (mean brightness, starting background).
    calibration = [simulate_frame(moving, camera, background, maps, calibration_rng,
                                  path_xy=paths[0]) for _ in range(CALIBRATION_FRAMES)]
    try:
        plane, scatter = calibrate_background(calibration, camera, hot)
    except est.EstimationFailure:
        plane, scatter = None, None

    names = [*job["methods"], LOOP] + ([LOOP_REFRESH] if job.get("refresh") else [])
    estimates = {name: np.full((n, 2), np.nan) for name in names}
    flux = np.full(n, np.nan)          # fitted spot signal, likelihood fit only
    snr = np.empty(n)
    last, fallbacks = None, 0
    last_r, fallbacks_r, plane_r, scatter_r = None, 0, plane, scatter
    recent = deque(maxlen=CALIBRATION_FRAMES)
    for k in range(n):
        frame = simulate_frame(
            replace(moving, total_signal_e=total * brightness[k]), camera,
            background if gain[k] == 1.0 else scaled_background(background, gain[k]),
            maps, frame_rng, path_xy=paths[k])
        snr[k] = frame.nominal_snr
        full_frame = None
        for name in job["methods"]:
            result = METHODS[name](frame.image_dn, camera, hot)
            if result.ok:
                estimates[name][k] = (result.x, result.y)
                if name == "gaussian_fit_mle":
                    flux[k] = result.amplitude_e
            if name == TRACKER:
                full_frame = result
        result, fell_back = track_with_fallback(frame.image_dn, camera, hot, plane, scatter,
                                                last, full_frame)
        fallbacks += fell_back
        if result.ok:
            estimates[LOOP][k] = last = (result.x, result.y)
        if job.get("refresh"):
            result, fell_back = track_with_fallback(frame.image_dn, camera, hot, plane_r,
                                                    scatter_r, last_r, full_frame)
            fallbacks_r += fell_back
            if result.ok:
                estimates[LOOP_REFRESH][k] = last_r = (result.x, result.y)
            recent.append(frame)
            if k % REFRESH_FRAMES == REFRESH_FRAMES - 1:
                try:
                    plane_r, scatter_r = calibrate_background(list(recent), camera, hot)
                except est.EstimationFailure:
                    pass   # no spot in any recent frame: keep the old background

    base = {
        "case": job["case"], "preset": preset, "realisation": realisation,
        "index": job["index"], "correlation_s": job["correlation_s"], "gain_end": job["gain_end"],
        "injected_amplitude_px": scenario.disturbance_amplitude_px,
        "signal_e": total,
        "brightness_mean": float(brightness.mean()),
        "brightness_index": float(brightness.var() / brightness.mean() ** 2),
        "brightness_min": float(brightness.min()),
        "snr_mean": float(snr.mean()), "snr_min": float(snr.min()),
        "snr_p05": float(np.quantile(snr, 0.05)), "snr_p50": float(np.quantile(snr, 0.5)),
        "snr_p95": float(np.quantile(snr, 0.95)),
        "floor_constant_px": float(np.sqrt((floors**2).mean())) if job["index"] == 0
        and job["gain_end"] == 1 else float(cramer_rao_floor_px(camera, start, background, total)),
        "floor_all_frames_px": float(np.sqrt((floors**2).mean())),
    }
    apparent = spectral.alias_frequency(scenario.disturbance_hz, scenario.frame_rate_hz)
    model_floor = float(spectral.jitter_psd(apparent, scenario)[0] + spectral.wander_psd(apparent, scenario)[0])
    which_bin = np.digitize(brightness, BRIGHTNESS_EDGES) - 1

    rows, bins = [], []
    for name, xy in estimates.items():
        ok = np.isfinite(xy[:, 0])
        error = xy - truth
        gross = ok & (np.abs(np.nan_to_num(error)).max(axis=1) > GROSS_ERROR_PX)
        gaps = gap_lengths(ok)
        rms = float(np.sqrt((error[ok] ** 2).mean())) if ok.any() else float("nan")
        # What a loop that simply keeps its last position would hand on
        # during the frames with no estimate.
        held = xy.copy()
        for k in np.flatnonzero(~ok):
            held[k] = held[k - 1] if k else np.nan
        held_error = (held - truth)[~ok]
        floor_accepted = float(np.sqrt((floors[ok] ** 2).mean())) if ok.any() else float("nan")
        row = base | {
            "method": name,
            "refused": float(1 - ok.mean()),
            "fallback": {LOOP: fallbacks, LOOP_REFRESH: fallbacks_r}.get(name, np.nan) / n,
            "gross": float(gross.sum() / ok.sum()) if ok.any() else float("nan"),
            "bias_x_px": float(error[ok, 0].mean()) if ok.any() else float("nan"),
            "bias_y_px": float(error[ok, 1].mean()) if ok.any() else float("nan"),
            "rms_x_px": float(np.sqrt((error[ok, 0] ** 2).mean())) if ok.any() else float("nan"),
            "rms_y_px": float(np.sqrt((error[ok, 1] ** 2).mean())) if ok.any() else float("nan"),
            "rms_error_px": rms,
            "floor_accepted_px": floor_accepted,
            "rms_over_floor": rms / floor_accepted,
            "gaps": int(gaps.size),
            "longest_gap_frames": int(gaps.max()) if gaps.size else 0,
            "mean_gap_frames": float(gaps.mean()) if gaps.size else 0.0,
            "held_rms_px": float(np.sqrt(np.nanmean(held_error**2))) if (~ok).any() else float("nan"),
        }
        if name == "gaussian_fit_mle":
            both = ok & np.isfinite(flux)
            # Does the fitted signal follow the true brightness? If so the
            # fit itself is a brightness monitor.
            ratio = flux[both] / (total * brightness[both])
            row |= {"flux_ratio_mean": float(ratio.mean()), "flux_ratio_std": float(ratio.std()),
                    "flux_correlation": float(np.corrcoef(flux[both], brightness[both])[0, 1])
                    if np.ptp(brightness) > 0 else float("nan")}
        measured = rms if np.isfinite(rms) else 0.0
        try:
            row |= spectral.analyse(t, xy, scenario, n_exposed,
                              model_floor + spectral.measurement_psd(measured, scenario),
                              job["thresholds"])
        except ValueError as exc:
            row |= {"analysis_failed": str(exc), "detected": False, "correct": False}
        rows.append(row)
        if job["index"] > 0:
            for b in range(len(BRIGHTNESS_EDGES) - 1):
                inside = which_bin == b
                if inside.any():
                    bins.append({
                        "case": job["case"], "preset": preset, "method": name, "index": job["index"],
                        "correlation_s": job["correlation_s"], "bin": b,
                        "frames": int(inside.sum()), "accepted": int((inside & ok).sum()),
                        "gross": int((inside & gross).sum()),
                        "sum_square_error": float((error[inside & ok] ** 2).sum()),
                        "sum_brightness": float(brightness[inside].sum()),
                        "sum_snr": float(snr[inside].sum()),
                        "sum_square_floor": float((floors[inside] ** 2).sum()),
                    })

    out = {"job": {k: v for k, v in job.items() if k not in ("scenario", "thresholds")},
           "rows": rows, "bins": bins}
    if job.get("keep_series"):
        out["series"] = {"t": t, "truth": truth, "brightness": brightness, "snr": snr,
                         "drift": traj.per_frame(trajectory["drift"], scenario, n_exposed),
                         "estimates": estimates}
    return out


# --------------------------------------------------------------------------
# Single frames under a condition
# --------------------------------------------------------------------------

def run_static(job: dict) -> list[dict]:
    """
    Many independent frames of a static spot under one condition, measured
    by the four shipped estimators and by tracking mode. `job` holds:
    preset, snr, kind ("shape", "second" or "dark"), value, point, frames.

    The signal is the one that gives `snr` for the nominal spot, and stays
    fixed when the condition changes: a wider spot has the same number of
    photons, spread thinner.

    The truth is the centre of the tracked spot. For a second source this
    is a choice: the centre of all the light in the frame is elsewhere.

    Tracking mode starts from a previous position wrong by 0.3 px RMS, as in
    realtime.py, with a background measured beforehand under nominal
    conditions (no second source, calibration temperature).
    """
    preset, snr, kind, value = job["preset"], job["snr"], job["kind"], job["value"]
    camera, spot, background = load_config(preset)
    maps = make_sensor_maps(camera, make_rngs(SEED)[0])
    hot = est.hot_pixel_mask(maps, camera)   # the calibration-time defect map
    rng = np.random.default_rng(
        [SEED, 7, sum(ord(c) for c in preset), int(snr), sum(ord(c) for c in kind), job["point"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = signal_for_target_snr(snr, spot, camera, background, maps)
    nominal = replace(spot, total_signal_e=total)
    plane, scatter = calibrate_background(
        [simulate_frame(nominal, camera, background, maps, rng) for _ in range(CALIBRATION_FRAMES)],
        camera, hot)

    sigma_x, sigma_y, sim_maps = spot.sigma_x, spot.sigma_y, maps
    if kind == "shape":
        sigma_x, sigma_y = value
    if kind == "dark":
        sim_maps = SensorMaps(maps.relative_sensitivity, maps.dark_current_e_per_s * value)
    angle = np.radians(SECOND_SPOT_ANGLE_DEG)
    direction = np.array([np.cos(angle), np.sin(angle)])
    centre = np.round(spot.center_xy)

    names = [*METHODS, TRACKING]
    n = job["frames"]
    estimates = {name: np.full((n, 2), np.nan) for name in names}
    fitted_sigma, fitted_flux = np.full(n, np.nan), np.full(n, np.nan)
    truths, others, snrs = np.empty((n, 2)), np.full((n, 2), np.nan), np.empty(n)
    for i in range(n):
        truths[i] = centre + rng.uniform(-0.5, 0.5, size=2)
        prior = truths[i] + rng.normal(0.0, PRIOR_ERROR_PX, size=2)
        tracked = replace(nominal, center_xy=tuple(truths[i]), sigma_x=sigma_x, sigma_y=sigma_y)
        extra = ()
        if kind == "second":
            offset, ratio = value
            others[i] = truths[i] + offset * direction
            extra = (replace(nominal, center_xy=tuple(others[i]), total_signal_e=total * ratio),)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = simulate_frame(tracked, camera, background, sim_maps, rng, extra_spots=extra)
        snrs[i] = frame.nominal_snr
        for name, estimator in METHODS.items():
            result = estimator(frame.image_dn, camera, hot)
            if result.ok:
                estimates[name][i] = (result.x, result.y)
                if name == "gaussian_fit_mle":
                    fitted_sigma[i], fitted_flux[i] = result.sigma_px, result.amplitude_e
        result = tracking_weighted_centroid(frame.image_dn, camera, hot, plane, scatter, prior)
        if result.ok:
            estimates[TRACKING][i] = (result.x, result.y)

    rows = []
    for name, xy in estimates.items():
        ok = np.isfinite(xy[:, 0])
        error = (xy - truths)[ok]
        gross = np.abs(error).max(axis=1) > GROSS_ERROR_PX if ok.any() else np.array([])
        row = {
            "preset": preset, "target_snr": snr, "kind": kind, "point": job["point"],
            "value": job["value_label"], "method": name, "frames": n,
            "snr": float(snrs.mean()),
            "refused": float(1 - ok.mean()),
            "gross": float(gross.mean()) if ok.any() else float("nan"),
            "bias_x_px": float(error[:, 0].mean()) if ok.any() else float("nan"),
            "bias_y_px": float(error[:, 1].mean()) if ok.any() else float("nan"),
            "std_x_px": float(error[:, 0].std()) if ok.any() else float("nan"),
            "std_y_px": float(error[:, 1].std()) if ok.any() else float("nan"),
            "rms_x_px": float(np.sqrt((error[:, 0] ** 2).mean())) if ok.any() else float("nan"),
            "rms_y_px": float(np.sqrt((error[:, 1] ** 2).mean())) if ok.any() else float("nan"),
        }
        if kind == "second" and ok.any():
            to_other = np.hypot(*(xy[ok] - others[ok]).T)
            to_spot = np.hypot(*error.T)
            # The estimate lies nearer the other source than the tracked spot.
            row["on_other_source"] = float((to_other < to_spot).mean())
            # Error with the wrong locks left out: what is left is the pull.
            near = to_other >= to_spot
            along = error[near] @ direction
            row["pull_toward_other_px"] = float(along.mean()) if near.any() else float("nan")
        if name == "gaussian_fit_mle" and ok.any():
            row["fitted_sigma_px"] = float(np.nanmean(fitted_sigma))
            row["fitted_sigma_std_px"] = float(np.nanstd(fitted_sigma))
            row["fitted_flux_ratio"] = float(np.nanmean(fitted_flux) / total)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Checks of the models, with no camera involved
# --------------------------------------------------------------------------

def self_check(scenario: traj.Scenario, preset: str) -> list[dict]:
    """
    1. Does brightness_sequence() have mean 1, the requested index and the
       requested memory? Measured on one long series.
    2. Does scaling the brightness move the estimate when there is no
       noise? It should not: the same shape, brighter or dimmer, has the
       same centre. What does move it is the rounding to whole digital
       numbers, which is coarser relative to a dim spot.
    """
    rng = np.random.default_rng([SEED, 8])
    checks = []
    for index in SCINTILLATION_INDICES[1:]:
        series = brightness_sequence(2_000_000, index, CORRELATION_S, scenario.frame_rate_hz, rng)
        log = np.log(series)
        lag = int(round(CORRELATION_S * scenario.frame_rate_hz))
        memory = float(np.corrcoef(log[:-lag], log[lag:])[0, 1])
        memory_linear = float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
        checks += [
            {"check": f"Mean brightness, index {index:g}", "expected": "1.000",
             "measured": f"{series.mean():.3f}"},
            {"check": f"Variance / mean², index {index:g}", "expected": f"{index:.3f}",
             "measured": f"{series.var() / series.mean() ** 2:.3f}"},
            {"check": f"Correlation of log brightness one correlation time apart, index {index:g}",
             "expected": f"{np.exp(-1):.3f}", "measured": f"{memory:.3f}"},
            {"check": f"Correlation of brightness itself at the same lag, index {index:g}",
             "expected": f"{(np.exp(np.log1p(index) * np.exp(-1)) - 1) / index:.3f}",
             "measured": f"{memory_linear:.3f}"},
        ]

    camera, spot, background = load_config(preset)
    maps = make_sensor_maps(camera, make_rngs(SEED)[0])
    hot = est.hot_pixel_mask(maps, camera)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = signal_for_target_snr(scenario.target_snr, spot, camera, background, maps)
    from simulator import digitize
    positions = np.round(spot.center_xy) + np.random.default_rng([SEED, 9]).uniform(-0.5, 0.5, (200, 2))
    for factor in (0.3, 1.0, 3.0):
        shifts = []
        for xy in positions:
            frame = simulate_frame(replace(spot, center_xy=tuple(xy), total_signal_e=total * factor),
                                   camera, background, maps, rng)
            clean = digitize(frame.expected_spot_e + frame.expected_background_e
                             + frame.expected_dark_e, camera)
            result = est.gaussian_fit_mle(clean, camera, hot)
            if result.ok:
                shifts.append((result.x - xy[0], result.y - xy[1]))
        shifts = np.array(shifts)
        checks.append({
            "check": f"Noise-free image at {factor:g} x brightness ({preset}, likelihood fit): "
                     "RMS error per axis (px)",
            "expected": "small; set by rounding to digital numbers",
            "measured": f"{np.sqrt((shifts**2).mean()):.4f} ({len(shifts)} of {len(positions)} accepted)"})
    return checks


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def build_sequence_jobs(presets: list[str], n_realisations: int, scenario: traj.Scenario,
                        thresholds: dict) -> list[dict]:
    jobs = []

    def add(case, preset, r, methods, **kw):
        jobs.append({"case": case, "preset": preset, "realisation": r, "scenario": scenario,
                     "index": 0.0, "correlation_s": CORRELATION_S, "gain_end": 1.0,
                     "methods": methods, "thresholds": thresholds} | kw)

    for preset in presets:
        for r in range(n_realisations):
            for index in SCINTILLATION_INDICES:
                add("scintillation", preset, r, list(METHODS), index=index,
                    keep_series=r == 0 and index == SCINTILLATION_INDICES[-1])
            for correlation_s in CORRELATION_SWEEP_S:
                add("correlation", preset, r, [TRACKER], index=SCINTILLATION_INDICES[-1],
                    correlation_s=correlation_s)
            # No disturbance at all: does fading make the detector see one?
            add("quiet", preset, r, [TRACKER], index=SCINTILLATION_INDICES[-1],
                scenario=replace(scenario, disturbance_amplitude_px=0.0))
            for gain_end in DAYLIGHT_GAINS:
                add("daylight", preset, r, [TRACKER], gain_end=gain_end, refresh=True)
    return jobs


def build_static_jobs(presets: list[str], n_frames: int) -> list[dict]:
    conditions = []
    nominal = 1.75
    for point, sigma in enumerate(SHAPE_SIGMAS):
        conditions.append(("shape", point, (sigma, sigma), f"circular, sigma {sigma:g} px"))
    for point, r in enumerate(SHAPE_ELLIPSES, start=len(SHAPE_SIGMAS)):
        conditions.append(("shape", point, (nominal * r, nominal / r),
                           f"elliptical, sigma {nominal * r:.2f} x {nominal / r:.2f} px"))
    point = 0
    for ratio in SECOND_SPOT_BRIGHTNESS:
        for offset in SECOND_SPOT_OFFSETS_PX:
            conditions.append(("second", point, (offset, ratio),
                               f"{ratio:g} x as bright, {offset:g} px away"))
            point += 1
    for point, multiplier in enumerate(DARK_MULTIPLIERS):
        conditions.append(("dark", point, multiplier, f"dark current x {multiplier:g}"))
    return [{"preset": preset, "snr": snr, "kind": kind, "point": point, "value": value,
             "value_label": text, "frames": n_frames}
            for preset in presets for snr in STATIC_SNRS
            for kind, point, value, text in conditions]


def mean_of(rows: list[dict], key: str) -> float:
    return experiment.mean_std(r.get(key) for r in rows)[0]


def summarise_sequence(rows: list[dict]) -> dict:
    """One line from the records of one condition, preset and method."""
    right = [r for r in rows if r.get("correct")]
    out = {"records": len(rows), "detected": sum(bool(r.get("detected")) for r in rows),
           "correct": len(right),
           "longest_gap_frames": max((r["longest_gap_frames"] for r in rows), default=0)}
    for key in ("refused", "fallback", "gross", "bias_x_px", "bias_y_px", "rms_x_px", "rms_y_px",
                "rms_error_px", "rms_over_floor", "floor_constant_px", "floor_all_frames_px",
                "floor_accepted_px", "gaps", "mean_gap_frames", "held_rms_px", "brightness_mean",
                "brightness_index", "brightness_min", "snr_mean", "snr_min", "snr_p05", "snr_p50",
                "snr_p95", "line_ratio", "threshold", "seen_amplitude_px", "flux_ratio_mean",
                "flux_ratio_std", "flux_correlation"):
        out[key], out[f"{key}_std"], _ = experiment.mean_std(r.get(key) for r in rows)
    for key in ("frequency_error_hz", "amplitude_px"):
        out[key], out[f"{key}_std"], _ = experiment.mean_std(r.get(key) for r in right)
    return out


def select(rows: list[dict], **match) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


def pool_bins(bins: list[dict], **match) -> list[dict]:
    """Add up the brightness bins of every record of one condition and method."""
    pooled = []
    for b in range(len(BRIGHTNESS_EDGES) - 1):
        rows = select(bins, bin=b, **match)
        frames = sum(r["frames"] for r in rows)
        accepted = sum(r["accepted"] for r in rows)
        if frames == 0:
            continue
        pooled.append({
            **match, "bin": b,
            "brightness_low": float(BRIGHTNESS_EDGES[b]), "brightness_high": float(BRIGHTNESS_EDGES[b + 1]),
            "brightness": sum(r["sum_brightness"] for r in rows) / frames,
            "snr": sum(r["sum_snr"] for r in rows) / frames,
            "frames": frames, "accepted": accepted, "refused": 1 - accepted / frames,
            "gross": sum(r["gross"] for r in rows) / accepted if accepted else float("nan"),
            "rms_error_px": float(np.sqrt(sum(r["sum_square_error"] for r in rows) / (2 * accepted)))
            if accepted else float("nan"),
            "floor_px": float(np.sqrt(sum(r["sum_square_floor"] for r in rows) / frames)),
        })
    return pooled


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def line_only(name: str) -> dict:
    return {k: v for k, v in series_kwargs(name).items() if k not in ("capsize", "elinewidth")}


LOOP_STYLE = dict(color=NEUTRAL, linestyle="-", linewidth=2, marker="v", markersize=6.5,
                  markerfacecolor=NEUTRAL, markeredgecolor=NEUTRAL, markeredgewidth=1.5)
REFRESH_STYLE = LOOP_STYLE | dict(linestyle=(0, (1, 1.5)), marker="P",
                                  markerfacecolor=plot_style.SURFACE)


def plot_error_vs_brightness(path: Path, preset: str, pooled: dict, index: float) -> None:
    """Chart: position error of each estimator against the brightness of the frame."""
    fig, ax, top = plot_style.new_figure(
        4.8, f"Position error against the brightness of the frame, {preset} preset",
        f"Scintillation index {index:g}, fades lasting about {1000 * CORRELATION_S:g} ms, mean "
        "brightness equal to SNR 20. Frames grouped by brightness.\nRMS error per axis over the "
        "frames each method accepted. Black: the Cramér-Rao floor at that brightness.")
    reference = pooled[TRACKER]
    ax.plot([b["brightness"] for b in reference], [b["floor_px"] for b in reference], color=TEXT,
            linewidth=2, label="Cramér-Rao floor", zorder=2)
    for name in METHODS:
        rows = [b for b in pooled[name] if b["accepted"] >= 50]
        ax.plot([b["brightness"] for b in rows], [b["rms_error_px"] for b in rows], zorder=3,
                **line_only(name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style.plain_log_axis(ax.xaxis)
    plot_style.plain_log_axis(ax.yaxis)
    ax.set_xlabel("Brightness of the frame (multiple of the mean)")
    ax.set_ylabel("RMS position error per axis (px)")
    plot_style.save_figure(fig, ax, top, path, "upper right")


def plot_refused_vs_brightness(path: Path, preset: str, pooled: dict, index: float) -> None:
    """Chart: share of frames each estimator refuses, against brightness."""
    fig, ax, top = plot_style.new_figure(
        4.8, f"Frames with no estimate against brightness, {preset} preset",
        f"Scintillation index {index:g}, same records as the error chart. A method refuses a "
        "frame when it finds no spot it trusts.\nGrey: the loop (tracking mode, then the "
        "full-frame weighted centroid) has no estimate from either.")
    for name in METHODS:
        rows = pooled[name]
        ax.plot([b["brightness"] for b in rows], [100 * b["refused"] for b in rows], zorder=3,
                **line_only(name))
    rows = pooled[LOOP]
    ax.plot([b["brightness"] for b in rows], [100 * b["refused"] for b in rows], zorder=4,
            label=LOOP_LABEL, **LOOP_STYLE)
    ax.set_xscale("log")
    plot_style.plain_log_axis(ax.xaxis)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Brightness of the frame (multiple of the mean)")
    ax.set_ylabel("Frames with no estimate (%)")
    plot_style.save_figure(fig, ax, top, path, "upper right")


def deepest_window(series: dict, length_s: float = 1.0) -> np.ndarray:
    """The stretch of the record, `length_s` long, centred on its dimmest frame."""
    t = series["t"]
    centre = t[int(np.argmin(series["brightness"]))]
    low = min(max(centre - length_s / 2, t[0]), t[-1] - length_s)
    return (t >= low) & (t < low + length_s)


def plot_fading_brightness(path: Path, preset: str, series: dict, index: float) -> None:
    """Chart: brightness through one second of a record, with the refused frames marked."""
    window = deepest_window(series)
    t, brightness = series["t"][window], series["brightness"][window]
    refused = ~np.isfinite(series["estimates"][TRACKER][window, 0])
    fig, ax, top = plot_style.new_figure(
        4.6, f"One second of fading, {preset} preset",
        f"First record, scintillation index {index:g}: the second around its deepest fade. "
        "Dotted line: the mean (SNR 20).\nMarks along the bottom: frames the "
        f"full-frame weighted centroid refused ({int(refused.sum())} of {len(t)}).")
    ax.plot(t, brightness, color=TEXT, linewidth=1.2, label="Brightness of the spot", zorder=3)
    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=1)
    ax.set_yscale("log")
    plot_style.plain_log_axis(ax.yaxis)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 5)   # room for the legend above the brightest frame
    ax.plot(t[refused], np.full(int(refused.sum()), low * 1.15), linestyle="none", marker="|",
            markersize=9, markeredgewidth=1.2, color=STYLE[TRACKER]["color"],
            label="Frame refused", zorder=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Brightness (multiple of the mean)")
    plot_style.save_figure(fig, ax, top, path, "upper left")


def plot_fading_tracking(path: Path, preset: str, series: dict, index: float) -> None:
    """Chart: truth and estimates through the same second, drift removed."""
    window = deepest_window(series)
    t = series["t"][window]
    drift = series["drift"][window, 0]
    estimate = series["estimates"][TRACKER][window, 0]
    fig, ax, top = plot_style.new_figure(
        4.8, f"Tracking through a fade, {preset} preset",
        f"The same second as the brightness chart, scintillation index {index:g}. x position "
        "with the drift subtracted.\nOne estimate per accepted frame; refused frames leave a gap. "
        "The scatter widens as the spot dims.")
    ax.plot(t, series["truth"][window, 0] - drift, color=TEXT, linewidth=2, label="True position",
            zorder=3)
    kw = plot_style.marker_only(TRACKER) | {"markersize": 3.5, "markeredgewidth": 0.9,
                                    "label": f"{STYLE[TRACKER]['label']}, per frame"}
    ax.plot(t, estimate - drift, zorder=4, **kw)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.2 * (high - low))   # room for the legend above the data
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x minus drift (px)")
    plot_style.style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper center", ncol=2,
              columnspacing=1.5, handletextpad=0.5)
    plot_style.save_figure(fig, ax, top, path, None)


def plot_daylight(path: Path, preset: str, summaries: dict) -> None:
    """Chart: error against how much the background brightened, three ways of tracking."""
    fig, ax, top = plot_style.new_figure(
        4.8, f"A background that brightens during the record, {preset} preset",
        "Background level and gradient grow steadily to the stated multiple over the record; "
        "spot signal fixed.\nRMS error per axis over accepted frames, mean of the records. "
        "Refreshed: re-measured every 0.1 s.")
    x = list(DAYLIGHT_GAINS)
    ax.plot(x, [summaries[g][TRACKER]["floor_all_frames_px"] for g in x], color=TEXT, linewidth=2,
            label="Cramér-Rao floor", zorder=2)
    ax.plot(x, [summaries[g][TRACKER]["rms_error_px"] for g in x], zorder=3, **line_only(TRACKER))
    ax.plot(x, [summaries[g][LOOP]["rms_error_px"] for g in x], zorder=4,
            label="Tracking mode + fallback, background measured once", **LOOP_STYLE)
    ax.plot(x, [summaries[g][LOOP_REFRESH]["rms_error_px"] for g in x], zorder=5,
            label=LOOP_REFRESH_LABEL, **REFRESH_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style.plain_log_axis(ax.xaxis)
    plot_style.plain_log_axis(ax.yaxis)
    ax.set_xlabel("Background at the end of the record (multiple of its start)")
    ax.set_ylabel("RMS position error per axis (px)")
    plot_style.save_figure(fig, ax, top, path, "upper left")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%" if np.isfinite(value) else "-"


def num(value: float, digits: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def write_tables(path: Path, scenario: traj.Scenario, presets: list[str], sequence: dict,
                 pooled: dict, static: list[dict], checks: list[dict], n_realisations: int,
                 n_static: int) -> None:
    index_max = SCINTILLATION_INDICES[-1]
    lines = [
        "# Real-world conditions: results", "",
        "Generated by `src/real_world.py`; do not edit by hand. "
        f"{n_realisations} records of {scenario.duration_s:g} s per moving condition, "
        f"{n_static} frames per single-frame condition, sensor seed {SEED}.", "",
        "RMS error is per axis over the frames a method accepted. Gross: accepted, but more than "
        f"{GROSS_ERROR_PX:g} px from the truth. Values are means over the records unless a "
        "standard deviation is given.", "",
        "## Checks of the models (no camera, except the last three)", "",
        "| Check | Expected | Measured |", "|---|---|---|",
    ]
    lines += [f"| {c['check']} | {c['expected']} | {c['measured']} |" for c in checks]

    for preset in presets:
        lines += ["", f"## Scintillation, {preset}", "",
                  f"Moving-spot scenario (SNR {scenario.target_snr:g} on average), brightness changing "
                  f"from frame to frame, fades lasting about {1000 * CORRELATION_S:g} ms. The same "
                  "motion and the same fades at every index.", "",
                  "### Brightness actually delivered", "",
                  "| Index | Mean brightness | Variance / mean² | Dimmest frame | SNR: 5% of frames "
                  "below | SNR: median | SNR: 5% above |",
                  "|---|---|---|---|---|---|---|"]
        for index in SCINTILLATION_INDICES:
            s = sequence[preset, "scintillation", index][TRACKER]
            lines.append(f"| {index:g} | {num(s['brightness_mean'], 3)} ± "
                         f"{num(s['brightness_mean_std'], 3)} | {num(s['brightness_index'], 3)} "
                         f"| {num(s['brightness_min'], 3)} | {num(s['snr_p05'], 1)} "
                         f"| {num(s['snr_p50'], 1)} | {num(s['snr_p95'], 1)} |")
        lines += ["", "### Recovered trajectory", "",
                  "Floor, accepted frames: the Cramér-Rao floor of each accepted frame at its own "
                  "brightness, combined as an RMS. Held: the error of keeping the last estimate "
                  "through the frames with none.", "",
                  "| Index | Method | No estimate | Fallback | Gross | Bias x / y (px) "
                  "| RMS x / y (px) | Floor, accepted frames (px) | RMS ÷ floor | Gaps per record "
                  "| Longest gap (frames) | Held through gaps, RMS (px) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for index in SCINTILLATION_INDICES:
            for name in (*METHODS, LOOP):
                s = sequence[preset, "scintillation", index][name]
                lines.append(
                    f"| {index:g} | {label(name)} | {pct(s['refused'])} | {pct(s['fallback'])} "
                    f"| {pct(s['gross'])} | {num(s['bias_x_px'], 4, True)} / "
                    f"{num(s['bias_y_px'], 4, True)} | {num(s['rms_x_px'])} / {num(s['rms_y_px'])} "
                    f"| {num(s['floor_accepted_px'])} | {num(s['rms_over_floor'], 2)} "
                    f"| {num(s['gaps'], 1)} | {s['longest_gap_frames']} | {num(s['held_rms_px'])} |")
        lines += ["", "### The periodic disturbance", "",
                  f"Injected: {scenario.disturbance_hz:g} Hz, {scenario.disturbance_amplitude_px:g} "
                  "px. Frequency and amplitude are over the records where the line was found.", "",
                  "| Index | Method | Found | Line ÷ floor (threshold) | Frequency error (Hz) "
                  "| Amplitude (px) |", "|---|---|---|---|---|---|"]
        for index in SCINTILLATION_INDICES:
            for name in (*METHODS, LOOP):
                s = sequence[preset, "scintillation", index][name]
                lines.append(
                    f"| {index:g} | {label(name)} | {s['correct']}/{s['records']} "
                    f"| {num(s['line_ratio'], 1)} ({num(s['threshold'], 1)}) "
                    f"| {tracking_plots.pm(s['frequency_error_hz'], s['frequency_error_hz_std'])} "
                    f"| {tracking_plots.pm(s['amplitude_px'], s['amplitude_px_std'])} |")
        quiet = sequence[preset, "quiet", index_max]
        lines += ["", f"With no disturbance injected and index {index_max:g}: a line was announced "
                      f"in {quiet[TRACKER]['detected']} of {quiet[TRACKER]['records']} records "
                      f"(weighted centroid) and {quiet[LOOP]['detected']} of "
                      f"{quiet[LOOP]['records']} (the loop). Strongest ratio, mean: "
                      f"{num(quiet[TRACKER]['line_ratio'], 1)} against a threshold of "
                      f"{num(quiet[TRACKER]['threshold'], 1)}.",
                  "", "### The fitted signal as a brightness monitor (likelihood fit)", "",
                  "| Index | Fitted signal ÷ true, mean | Scatter of that ratio | Correlation with "
                  "the true brightness |", "|---|---|---|---|"]
        for index in SCINTILLATION_INDICES:
            s = sequence[preset, "scintillation", index]["gaussian_fit_mle"]
            lines.append(f"| {index:g} | {num(s['flux_ratio_mean'], 3)} "
                         f"| {num(s['flux_ratio_std'], 3)} | {num(s['flux_correlation'], 3)} |")
        lines += ["", f"### Error against brightness, index {index_max:g}", "",
                  "| Brightness (× mean) | Mean SNR | Frames | Floor (px) | "
                  + " | ".join(f"{label(n)}: RMS (px), no estimate" for n in (*METHODS, LOOP)) + " |",
                  "|---|---|---|---|" + "---|" * (len(METHODS) + 1)]
        for row in pooled[preset][TRACKER]:
            cells = []
            for name in (*METHODS, LOOP):
                b = next(x for x in pooled[preset][name] if x["bin"] == row["bin"])
                cells.append(f"{num(b['rms_error_px'])}, {pct(b['refused'], 1)}")
            high = "∞" if not np.isfinite(row["brightness_high"]) else f"{row['brightness_high']:g}"
            lines.append(f"| {row['brightness_low']:g} to {high} | {num(row['snr'], 1)} "
                         f"| {row['frames']} | {num(row['floor_px'])} | " + " | ".join(cells) + " |")
        lines += ["", f"### How long the fades last, index {index_max:g} (weighted centroid and "
                      "the loop)", "",
                  "| Correlation time | Method | No estimate | Fallback | RMS (px) | Gaps per record "
                  "| Mean gap (frames) | Longest gap (frames) | Held through gaps, RMS (px) "
                  "| Disturbance found |", "|---|---|---|---|---|---|---|---|---|---|"]
        for correlation_s in sorted({*CORRELATION_SWEEP_S, CORRELATION_S}):
            case = "scintillation" if correlation_s == CORRELATION_S else "correlation"
            group = sequence[preset, case, index_max if case == "scintillation" else correlation_s]
            for name in (TRACKER, LOOP):
                s = group[name]
                lines.append(
                    f"| {1000 * correlation_s:g} ms | {label(name)} | {pct(s['refused'])} "
                    f"| {pct(s['fallback'])} | {num(s['rms_error_px'])} | {num(s['gaps'], 1)} "
                    f"| {num(s['mean_gap_frames'], 2)} | {s['longest_gap_frames']} "
                    f"| {num(s['held_rms_px'])} | {s['correct']}/{s['records']} |")

        lines += ["", f"## Changing daylight, {preset}", "",
                  "Moving-spot scenario with constant spot brightness. The background level and its "
                  "gradient grow steadily to the stated multiple by the end of the record.", "",
                  "| Background at the end | SNR at the end | Method | No estimate | Fallback | Gross "
                  "| Bias x / y (px) | RMS x / y (px) | Floor (px) | RMS ÷ floor | Disturbance found |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for gain in DAYLIGHT_GAINS:
            for name in (TRACKER, LOOP, LOOP_REFRESH):
                s = sequence[preset, "daylight", gain][name]
                lines.append(
                    f"| × {gain:g} | {num(s['snr_min'], 1)} | {label(name)} | {pct(s['refused'])} "
                    f"| {pct(s['fallback'])} | {pct(s['gross'])} | {num(s['bias_x_px'], 4, True)} / "
                    f"{num(s['bias_y_px'], 4, True)} | {num(s['rms_x_px'])} / {num(s['rms_y_px'])} "
                    f"| {num(s['floor_accepted_px'])} | {num(s['rms_over_floor'], 2)} "
                    f"| {s['correct']}/{s['records']} |")

    titles = {
        "shape": ("Spot shape", "Same number of photons as the nominal spot at the stated SNR, "
                  "spread into a different shape. The estimators assume a circular spot of "
                  "sigma 1.75 px."),
        "second": ("A second light source", "A second spot of the nominal shape, in the direction "
                   f"{SECOND_SPOT_ANGLE_DEG:g} degrees from the x axis. Truth is the centre of the "
                   "tracked spot. On other: accepted estimates nearer the second source than the "
                   "tracked spot. Pull: mean error toward the second source, over the rest."),
        "dark": ("Sensor temperature", "Dark current of every pixel multiplied by the stated "
                 "factor after calibration. The defect map and the loop's background are from "
                 "before the change."),
    }
    for kind, (title, text) in titles.items():
        for preset in presets:
            for snr in STATIC_SNRS:
                rows = select(static, kind=kind, preset=preset, target_snr=snr)
                lines += ["", f"## {title}, {preset}, nominal SNR {snr:g}", "", text, ""]
                extra_head = extra_rule = ""
                if kind == "second":
                    extra_head, extra_rule = " On other | Pull (px) |", "---|---|"
                if kind == "shape":
                    extra_head, extra_rule = " Fitted sigma (px) |", "---|"
                lines += ["| Condition | SNR | Method | Refused | Gross | Bias x / y (px) "
                          "| Scatter x / y (px) | RMS x / y (px) |" + extra_head,
                          "|---|---|---|---|---|---|---|---|" + extra_rule]
                for r in rows:
                    extra = ""
                    if kind == "second":
                        extra = (f" {pct(r.get('on_other_source', np.nan), 1)} "
                                 f"| {num(r.get('pull_toward_other_px', np.nan), 4, True)} |")
                    if kind == "shape":
                        extra = (f" {num(r.get('fitted_sigma_px', np.nan), 3)} |"
                                 if r["method"] == "gaussian_fit_mle" else " - |")
                    lines.append(
                        f"| {r['value']} | {num(r['snr'], 1)} | {label(r['method'])} "
                        f"| {pct(r['refused'])} | {pct(r['gross'])} | {num(r['bias_x_px'], 4, True)} "
                        f"/ {num(r['bias_y_px'], 4, True)} | {num(r['std_x_px'])} / "
                        f"{num(r['std_y_px'])} | {num(r['rms_x_px'])} / {num(r['rms_y_px'])} |" + extra)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_charts(output: Path, presets: list[str], pooled: dict, sequence: dict, series: dict) -> None:
    """
    Every chart of this experiment. `series` holds, per preset, the kept series of
    the first fading record.
    """
    index_max = SCINTILLATION_INDICES[-1]
    for preset in presets:
        plot_error_vs_brightness(output / f"error_vs_brightness_{preset}.png", preset,
                                 pooled[preset], index_max)
        plot_refused_vs_brightness(output / f"no_estimate_vs_brightness_{preset}.png", preset,
                                   pooled[preset], index_max)
        plot_daylight(output / f"daylight_{preset}.png", preset,
                      {g: sequence[preset, "daylight", g] for g in DAYLIGHT_GAINS})
        plot_fading_brightness(output / f"fading_brightness_{preset}.png", preset, series[preset],
                               index_max)
        plot_fading_tracking(output / f"fading_tracking_{preset}.png", preset, series[preset],
                             index_max)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--realisations", type=int, default=DEFAULT_REALISATIONS,
                        help="records per moving condition")
    parser.add_argument("--frames", type=int, default=DEFAULT_STATIC_FRAMES,
                        help="frames per single-frame condition")
    parser.add_argument("--presets", nargs="+", default=["silicon", "ingaas"])
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--quick", action="store_true",
                        help="2 s records, 2 realisations, 100 frames: a smoke test, not a result")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    scenario = traj.Scenario()
    n_realisations, n_static, n_null = args.realisations, args.frames, spectral.NULL_TRIALS
    if args.quick:
        scenario = replace(scenario, duration_s=2.0)
        n_realisations, n_static, n_null = 2, 100, 200

    # The same detection threshold as dynamic.py (same seed, same procedure).
    print("setting the detection threshold from noise-only records", flush=True)
    n = scenario.n_frames
    thresholds = {n: spectral.null_threshold(n, scenario.frame_rate_hz,
                                       np.random.default_rng([experiment.SEED, 11, n]), n_null)}

    sequence_jobs = build_sequence_jobs(args.presets, n_realisations, scenario, thresholds)
    static_jobs = build_static_jobs(args.presets, n_static)
    print(f"{len(sequence_jobs)} records and {len(static_jobs)} single-frame conditions "
          f"on {args.workers} workers", flush=True)
    results, static = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        check = pool.submit(self_check, scenario, args.presets[0])
        static_futures = [pool.submit(run_static, job) for job in static_jobs]
        futures = [pool.submit(run_sequence, job) for job in sequence_jobs]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 20 == 0 or done == len(futures):
                job = results[-1]["job"]
                print(f"  [{done}/{len(futures)}] last: {job['case']} {job['preset']} "
                      f"record {job['realisation']}", flush=True)
        static = [row for future in static_futures for row in future.result()]
        checks = check.result()

    # Workers finish in any order; sort so the files do not depend on it.
    results.sort(key=lambda r: (r["job"]["case"], r["job"]["preset"], r["job"]["index"],
                                r["job"]["correlation_s"], r["job"]["gain_end"],
                                r["job"]["realisation"]))
    rows = [row for r in results for row in r["rows"]]
    bins = [row for r in results for row in r["bins"]]
    experiment.write_csv(args.output / "records.csv", rows)
    experiment.write_csv(args.output / "single_frames.csv", static)

    def group(preset, case, **match):
        found = select(rows, preset=preset, case=case, **match)
        return {name: summarise_sequence(select(found, method=name))
                for name in dict.fromkeys(r["method"] for r in found)}

    sequence, pooled = {}, {}
    index_max = SCINTILLATION_INDICES[-1]
    for preset in args.presets:
        for index in SCINTILLATION_INDICES:
            sequence[preset, "scintillation", index] = group(preset, "scintillation", index=index)
        for correlation_s in CORRELATION_SWEEP_S:
            sequence[preset, "correlation", correlation_s] = group(
                preset, "correlation", correlation_s=correlation_s)
        sequence[preset, "quiet", index_max] = group(preset, "quiet")
        for gain in DAYLIGHT_GAINS:
            sequence[preset, "daylight", gain] = group(preset, "daylight", gain_end=gain)
        pooled[preset] = {
            name: pool_bins(bins, case="scintillation", preset=preset, method=name, index=index_max)
            for name in (*METHODS, LOOP)}
        experiment.write_csv(args.output / f"error_vs_brightness_{preset}.csv",
                     [row for name in pooled[preset] for row in pooled[preset][name]])

    series = {preset: next(r["series"] for r in results
                           if r["job"].get("keep_series") and r["job"]["preset"] == preset)
              for preset in args.presets}
    draw_charts(args.output, args.presets, pooled, sequence, series)

    write_tables(args.output / "tables.md", scenario, args.presets, sequence, pooled, static,
                 checks, n_realisations, n_static)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
