"""
Charts and the Markdown tables of the moving-spot experiment. One chart per function, one
chart per file. The numbers come from tracking_experiment.py; nothing here
simulates or measures.
"""

import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from plot_style import (
    GRID, STYLE, TEXT, TEXT_SECONDARY,
    marker_only, new_figure, plain_log_axis, save_figure, series_kwargs, style_axes,
)
from spectral import (
    clean_series, detect_line, fit_tone, fold, jitter_psd, unsteady_fit_factor, wander_psd,
)
from tracking_experiment import (
    CAMERA_LIMITED_AMPLITUDE_PX, CAMERA_LIMITED_HZ, GROSS_ERROR_PX, KNOCK_FOLLOW_FRAMES,
    KNOCK_MATCH_PX, LOOP, LOOP_LABEL, METHODS, SEED, TRACKER,
)
from trajectory import Scenario


def mark_frequency(ax, frequency_hz: float, label: str) -> None:
    ax.axvline(frequency_hz, color=TEXT_SECONDARY, linewidth=1.2, linestyle=":", zorder=1,
               label=label)


def plot_trajectory(path: Path, series: dict, scenario: Scenario) -> None:
    """Chart: the true motion along x over the whole record, and its drift part."""
    t, x0 = series["t"], scenario.start_xy[0]
    travel = series["drift"][-1, 0] - series["drift"][0, 0]
    fig, ax, top = new_figure(
        4.6, "The scenario: true spot position along x",
        f"First record. Drift {travel:+.1f} px in {scenario.duration_s:g} s; jitter "
        f"{scenario.jitter_rms_px:.2f} px RMS; disturbance {scenario.disturbance_amplitude_px:.2f} px "
        f"at {scenario.disturbance_hz:g} Hz,\ntoo small to pick out by eye. "
        "Dotted lines: pixel boundaries.")
    ax.plot(t, series["truth"][:, 0], color="#8a8984", linewidth=0.6,
            label="True position (drift + jitter + disturbance)")
    ax.plot(t, series["drift"][:, 0], color=TEXT, linewidth=2, label="Drift alone")
    low, high = series["truth"][:, 0].min(), series["truth"][:, 0].max()
    for boundary in np.arange(np.floor(low) + 0.5, high, 1.0):
        if boundary > low:
            ax.axhline(boundary, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x position (px)")
    save_figure(fig, ax, top, path, "upper left")


def plot_zoom(path: Path, preset: str, series: dict, scenario: Scenario) -> None:
    """Chart: a quarter of a second of truth against the tracker's estimates."""
    t = series["t"]
    middle = scenario.duration_s / 2
    window = (t >= middle) & (t < middle + 0.25)
    drift = series["drift"][:, 0]
    fig, ax, top = new_figure(
        4.8, f"A quarter second of tracking, {preset} preset",
        f"x axis with the drift subtracted, first record, SNR {scenario.target_snr:g}. "
        "One estimate per frame (1 ms apart).\nThe disturbance is drawn alone to show its size "
        "against the jitter and the measurement noise.")
    ax.plot(t[window], (series["truth"][:, 0] - drift)[window], color=TEXT, linewidth=2,
            label="True position", zorder=3)
    ax.plot(t[window], series["disturbance"][window, 0], color="#8a8984", linewidth=1.5,
            label="Disturbance alone", zorder=2)
    kw = marker_only(TRACKER) | {"markersize": 4, "markeredgewidth": 1}
    ax.plot(t[window], (series["estimates"][TRACKER][:, 0] - drift)[window], zorder=4,
            **(kw | {"label": f"{STYLE[TRACKER]['label']}, per frame"}))
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.22 * (high - low))   # room for the legend above the data
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x minus drift (px)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper center", ncol=3,
              columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, ax, top, path, None)


def plot_error_over_time(path: Path, preset: str, series: dict, floor_px: float,
                         scenario: Scenario) -> None:
    """Chart: RMS error in half-second blocks, to show whether it grows or wanders."""
    block = int(0.5 * scenario.frame_rate_hz)
    n_blocks = scenario.n_frames // block
    centres = (np.arange(n_blocks) + 0.5) * block / scenario.frame_rate_hz
    fig, ax, top = new_figure(
        4.8, f"Tracking error through the record, {preset} preset",
        f"First record, SNR {scenario.target_snr:g}. RMS error per axis in blocks of 0.5 s "
        "(500 frames, both axes).\nFlat lines mean the error does not build up as the spot moves.")
    ax.axhline(floor_px, color=TEXT, linewidth=2, label="Cramér-Rao floor, static spot", zorder=1)
    for name in METHODS:
        error = series["estimates"][name] - series["truth"]
        blocks = error[: n_blocks * block].reshape(n_blocks, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = np.sqrt(np.nanmean(blocks**2, axis=1))
        kw = {k: v for k, v in series_kwargs(name).items() if k not in ("capsize", "elinewidth")}
        ax.plot(centres, rms, zorder=3, **kw)
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS position error per axis (px)")
    save_figure(fig, ax, top, path, "lower left")


def plot_spectrum(path: Path, preset: str, spectra: list[dict], scenario: Scenario) -> None:
    """Chart: averaged spectrum of the true motion and of each estimator's output."""
    f = spectra[0]["truth"][0]
    show = (f >= 1.0) & (f <= 500.0)
    fig, ax, top = new_figure(
        5.4, f"Spectrum of the recovered motion, {preset} preset",
        f"Mean of {len(spectra)} records and both axes, SNR {scenario.target_snr:g}. "
        "Welch method, 2 s segments (0.5 Hz resolution).\nWhere a method's curve leaves the true "
        "motion, its own measurement noise has taken over.")
    model = jitter_psd(f[show], scenario) + wander_psd(f[show], scenario)
    ax.plot(f[show], model, color="#8a8984", linewidth=1.5, linestyle=(0, (1, 1.5)),
            label="Model: jitter + drift, no disturbance", zorder=2)
    mean_of = lambda name: np.mean([s[name][1] for s in spectra if name in s], axis=0)
    ax.plot(f[show], mean_of("truth")[show], color=TEXT, linewidth=2, label="True motion", zorder=3)
    for name in METHODS:
        kw = {k: v for k, v in series_kwargs(name).items()
              if k not in ("capsize", "elinewidth", "marker", "markersize", "markerfacecolor",
                           "markeredgecolor", "markeredgewidth")}
        ax.plot(f[show], mean_of(name)[show], zorder=4, **(kw | {"linewidth": 1.5}))
    ax.set_xscale("log")
    ax.set_yscale("log")
    mark_frequency(ax, scenario.disturbance_hz, f"Injected disturbance, {scenario.disturbance_hz:g} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density per axis (px²/Hz)")
    save_figure(fig, ax, top, path, "lower left")


def plot_detection(path: Path, preset: str, series: dict, scenario: Scenario,
                   threshold: float) -> None:
    """Chart: the detection statistic over the whole search band, one record."""
    _, filled, _ = clean_series(series["t"], series["estimates"][TRACKER])
    line = detect_line(filled, scenario.frame_rate_hz, threshold)
    band = np.isfinite(line["ratio"])
    fig, ax, top = new_figure(
        4.8, f"Searching the spectrum for a line, {preset} preset",
        f"{STYLE[TRACKER]['label']}, first record, {scenario.duration_s:g} s. Each bin's power "
        "divided by the mean of its neighbours.\nThreshold: noise alone crosses it somewhere in "
        "the band in 1% of records (set by simulation).")
    ax.plot(line["f"][band], line["ratio"][band], color=STYLE[TRACKER]["color"], linewidth=0.8,
            zorder=3, label=STYLE[TRACKER]["label"])
    ax.axhline(threshold, color=TEXT, linewidth=1.5, linestyle="--", zorder=2,
               label="Detection threshold")
    ax.set_xscale("log")
    ax.set_ylim(0, max(line["peak_ratio"], line["threshold"]) * 1.15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power / local floor")
    save_figure(fig, ax, top, path, "upper right")


def plot_folded(path: Path, preset: str, series: dict, scenario: Scenario,
                threshold: float) -> bool:
    """Chart: the motion averaged over one cycle of the fitted frequency."""
    t = series["t"]
    kept, filled, _ = clean_series(t, series["estimates"][TRACKER])
    line = detect_line(filled, scenario.frame_rate_hz, threshold)
    if not line["detected"]:
        return False
    tone = fit_tone(t, kept, line["peak_hz"])
    centres, mean, error = fold(t, kept, tone)
    fig, ax, top = new_figure(
        4.8, f"The disturbance, folded into one cycle, {preset} preset",
        f"{STYLE[TRACKER]['label']}, first record. Motion along the fitted direction, drift removed, "
        f"averaged\nover {tone['frequency_hz'] * scenario.duration_s:.0f} cycles of the fitted "
        f"frequency ({tone['frequency_hz']:.3f} Hz). Bars: one standard error.")
    cycle = np.linspace(0, 1, 200)
    # The injected wave on the same axis. Folding uses the fitted frequency,
    # so the injected one is drawn with the phase it has mid-record.
    slip = 2 * np.pi * (scenario.disturbance_hz - tone["frequency_hz"]) * t.mean()
    injected = (scenario.disturbance_amplitude_px * (scenario.direction @ tone["direction"])
                * np.sin(2 * np.pi * cycle + series["phase"] + slip))
    ax.plot(cycle, injected, color=TEXT, linewidth=2, label="Injected", zorder=2)
    kw = series_kwargs(TRACKER) | {"linestyle": "none", "label": "Measured, folded"}
    ax.errorbar(centres, mean, yerr=error, zorder=3, **kw)
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.set_xlabel("Phase (cycles)")
    ax.set_ylabel("Displacement along the vibration (px)")
    save_figure(fig, ax, top, path, "upper right")
    return True


def plot_amplitude_sweep(path: Path, summaries: list[dict], preset: str) -> None:
    """Chart: recovered against injected amplitude."""
    shown = [s for s in summaries if s["injected_amplitude_px"] > 0]
    x = [s["injected_amplitude_px"] for s in shown]
    fig, ax, top = new_figure(
        4.8, "How small a disturbance can be recovered",
        f"{STYLE[TRACKER]['label']}, {preset} preset, {shown[0]['records']} records of 10 s per "
        "point.\nPoint: mean of the records where the line was found. Bars: their standard "
        "deviation.")
    limit = max(x) * 1.15
    ax.plot([0, limit], [0, limit], color=TEXT, linewidth=2, label="Recovered = injected", zorder=2)
    y = [s["amplitude_px"] for s in shown]
    kw = series_kwargs(TRACKER) | {"linestyle": "none", "label": "Recovered amplitude"}
    ax.errorbar(x, y, yerr=[s["amplitude_px_std"] if np.isfinite(s["amplitude_px_std"]) else 0
                            for s in shown], zorder=3, **kw)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Injected amplitude (px)")
    ax.set_ylabel("Recovered amplitude (px)")
    save_figure(fig, ax, top, path, "upper left")


def plot_ratio_sweep(path: Path, summaries: list[dict], x_key: str, x_label: str, title: str,
                     caption: str, log_x: bool = True) -> None:
    """Chart: line strength against one swept quantity, with the detection threshold."""
    x = [s[x_key] for s in summaries]
    y = [s["line_ratio"] for s in summaries]
    fig, ax, top = new_figure(4.8, title, caption)
    ax.plot(x, [s["threshold"] for s in summaries], color=TEXT, linewidth=1.5, linestyle="--",
            label="Detection threshold", zorder=2)
    kw = series_kwargs(TRACKER) | {"label": "Strongest line / local floor"}
    ax.errorbar(x, y, yerr=[[a - s["line_ratio_min"] for a, s in zip(y, summaries)],
                            [s["line_ratio_max"] - a for a, s in zip(y, summaries)]],
                zorder=3, **kw)
    ax.set_yscale("log")
    plain_log_axis(ax.yaxis)
    if log_x:
        ax.set_xscale("log")
        plain_log_axis(ax.xaxis)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power / local floor")
    save_figure(fig, ax, top, path, "upper left" if log_x else "upper center")


def plot_snr_sweep(path: Path, summaries: list[dict], preset: str) -> None:
    """Chart: tracking error against SNR in the moving scenario."""
    x = [s["snr"] for s in summaries]
    fig, ax, top = new_figure(
        4.8, "Tracking error against SNR, moving spot",
        f"{STYLE[TRACKER]['label']}, {preset} preset, {summaries[0]['records']} records of 10 s per "
        "point.\nRMS error per axis over accepted frames.")
    ax.plot(x, [s["static_floor_px"] for s in summaries], color=TEXT, linewidth=2,
            label="Cramér-Rao floor, static spot", zorder=2)
    y = [s["rms_error_px"] for s in summaries]
    ax.errorbar(x, y, yerr=[s["rms_error_px_std"] for s in summaries], zorder=3,
                **series_kwargs(TRACKER))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis)
    plain_log_axis(ax.yaxis)
    ax.set_xlabel("Nominal SNR")
    ax.set_ylabel("RMS position error per axis (px)")
    save_figure(fig, ax, top, path, "lower left")


def pm(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "none"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def method_label(name: str) -> str:
    if name == "truth":
        return "Perfect camera (true positions)"
    return LOOP_LABEL if name == LOOP else STYLE[name]["label"]


def percent_or_dash(value: float) -> str:
    return f"{100 * value:.2f}%" if np.isfinite(value) else "-"


def write_tables(path: Path, scenario: Scenario, presets: list[str], main: dict, lengths: dict,
                 sweeps: dict, checks: list[dict], n_realisations: int, knocks: list[dict],
                 camera: dict) -> None:
    """The numbers behind the plots, as readable Markdown."""
    lines = [
        "# Dynamic tracking results",
        "",
        "Generated by `src/dynamic.py`; do not edit by hand. "
        f"{n_realisations} records per condition, sensor seed {SEED}.",
        "",
        "Values are mean ± standard deviation over the records. Frequency, amplitude and "
        "direction are over the records where the line was found at the right frequency.",
        "",
        "## Scenario",
        "",
        "| Parameter | Value |", "|---|---|",
    ]
    for key, value in asdict(scenario).items():
        lines.append(f"| `{key}` | {value} |")
    lines += ["", "## Checks of the analysis on synthetic series (no camera)", "",
              "| Check | Expected | Measured |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['expected']} | {c['measured']} |" for c in checks]

    for preset in presets:
        rows = main[preset]
        lines += ["", f"## {preset}: recovered trajectory", "",
                  "RMS error is per axis over accepted frames. Gross: accepted, but more than "
                  f"{GROSS_ERROR_PX:g} px from the truth. Floor: Cramér-Rao bound for a static spot "
                  "at the same signal. Slope: error against the spot's position inside a pixel "
                  "(px per px). Fallback (last row only): frames that tracking mode refused and "
                  "the full-frame weighted centroid took over; Refused is then frames with no "
                  "estimate from either.", "",
                  "| Method | SNR | Refused | Fallback | Gross | RMS error (px) | Floor (px) "
                  "| Error ÷ floor | Slope against sub-pixel position |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in (*METHODS, LOOP):
            s = rows[name]
            lines.append(f"| {method_label(name)} | {s['snr']:.1f} | {100 * s['refused']:.2f}% "
                         f"| {percent_or_dash(s['fallback'])} "
                         f"| {100 * s['gross']:.2f}% | {pm(s['rms_error_px'], s['rms_error_px_std'])} "
                         f"| {s['static_floor_px']:.4f} | {s['rms_over_floor']:.2f} "
                         f"| {pm(s['subpixel_slope'], s['subpixel_slope_std'])} |")
        lines += ["", f"RMS distance between the loop's answer and the full-frame weighted "
                      f"centroid's on the same frames: {rows[LOOP]['rms_from_full_frame_px']:.4f} px."]
        first = rows["truth"]
        lines += ["", f"### {preset}: the periodic disturbance", "",
                  f"Injected: {scenario.disturbance_hz:g} Hz, {scenario.disturbance_amplitude_px:g} px, "
                  f"{scenario.disturbance_angle_deg:g} degrees. Theory for one record "
                  f"(perfect camera): amplitude ± {first['predicted_sigma_amplitude_px']:.4f} px, "
                  f"frequency ± {first['predicted_sigma_frequency_hz']:.4f} Hz.", "",
                  "| Method | Found | Line ÷ floor (threshold) | Frequency (Hz) | Amplitude (px) "
                  "| Direction (deg) | Across (px) |",
                  "|---|---|---|---|---|---|---|"]
        for name in ("truth", *METHODS, LOOP):
            s = rows[name]
            lines.append(
                f"| {method_label(name)} | {s['correct']}/{s['records']} "
                f"| {s['line_ratio']:.1f} ({s['threshold']:.1f}) "
                f"| {pm(s['frequency_hz'], s['frequency_hz_std'])} "
                f"| {pm(s['amplitude_px'], s['amplitude_px_std'])} "
                f"| {pm(s['angle_deg'], s['angle_deg_std'], 1)} "
                f"| {pm(s['cross_amplitude_px'], s['cross_amplitude_px_std'])} |")
        lines += ["", f"### {preset}: record length", "",
                  "The same records, cut short. Theory columns: best possible scatter for one "
                  "record of that length.", "",
                  "| Length (s) | Series | Found | Line ÷ floor (threshold) | Frequency error (Hz) "
                  "| Theory (Hz) | Amplitude (px) | Theory (px) |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in lengths[preset]:
            lines.append(
                f"| {s['duration_s']:g} | {method_label(s['method'])} | {s['correct']}/{s['records']} "
                f"| {s['line_ratio']:.1f} ({s['threshold']:.1f}) "
                f"| {pm(s['frequency_error_hz'], s['frequency_error_hz_std'])} "
                f"| ± {s['predicted_sigma_frequency_hz']:.4f} "
                f"| {pm(s['amplitude_px'], s['amplitude_px_std'])} "
                f"| ± {s['predicted_sigma_amplitude_px']:.4f} |")

    sweep_preset = presets[0]
    titles = {
        "amplitude": ("Sweep of the disturbance amplitude", "injected_amplitude_px", "Injected (px)"),
        "frequency": ("Sweep of the disturbance frequency", "injected_hz", "Injected (Hz)"),
        "snr": ("Sweep of the SNR", "snr", "SNR"),
    }
    for tag, (title, key, heading) in titles.items():
        lines += ["", f"## {title} ({sweep_preset}, {STYLE[TRACKER]['label'].lower()})", "",
                  f"| {heading} | Series | Found | Detected at all | Line ÷ floor (threshold) "
                  "| Expected at (Hz) | Frequency (Hz) | Amplitude seen (px) | Amplitude (px) "
                  "| Refused | Fallback | Gated per record | RMS error (px) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in sweeps[tag]:
            lines.append(
                f"| {s[key]:.3g} | " if key == "snr" else f"| {s[key]:g} | ")
            lines[-1] += (
                f"{method_label(s['method'])} | {s['correct']}/{s['records']} "
                f"| {s['detected']}/{s['records']} | {s['line_ratio']:.1f} ({s['threshold']:.1f}) "
                f"| {s['apparent_hz']:g} | {pm(s['frequency_hz'], s['frequency_hz_std'])} "
                f"| {s['seen_amplitude_px']:.4f} | {pm(s['amplitude_px'], s['amplitude_px_std'])} "
                f"| {100 * s['refused']:.2f}% | {percent_or_dash(s['fallback'])} | {s['gated']:.1f} "
                f"| {pm(s['rms_error_px'], s['rms_error_px_std'])} |")

    lines += ["", f"## Unsteady disturbance frequency ({sweep_preset})", "",
              "The main scenario, with the disturbance frequency ramping steadily by the stated "
              "total over the record, centred on the nominal value. The same records at every "
              "value, so the first rows are the direct reference. Found: fitted frequency within "
              "2 bins of the band the tone occupies. Best fixed-frequency fit: the share of the "
              "amplitude any single-frequency fit could recover, from `unsteady_fit_factor`.", "",
              "| Total change (Hz) | Series | Found | Detected at all | Line ÷ floor (threshold) "
              "| Frequency (Hz) | Amplitude (px) | Amplitude ÷ injected | Best fixed-frequency fit |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in sweeps["unsteady"]:
        share = (f"{s['amplitude_px'] / s['seen_amplitude_px']:.2f}"
                 if np.isfinite(s["amplitude_px"]) else "none")
        lines.append(
            f"| {s['injected_drift_hz']:g} | {method_label(s['method'])} "
            f"| {s['correct']}/{s['records']} | {s['detected']}/{s['records']} "
            f"| {s['line_ratio']:.1f} ({s['threshold']:.1f}) "
            f"| {pm(s['frequency_hz'], s['frequency_hz_std'])} "
            f"| {pm(s['amplitude_px'], s['amplitude_px_std'])} | {share} "
            f"| {unsteady_fit_factor(s['injected_drift_hz'], scenario.duration_s):.2f} |")

    lines += ["", f"## Camera-limited disturbance: {CAMERA_LIMITED_HZ:g} Hz, "
                  f"{CAMERA_LIMITED_AMPLITUDE_PX:g} px", "",
              "The main scenario with only the disturbance changed, and the same records for "
              "both presets and both SNRs. Predicted: A² T / (6 S) + 1, with S the jitter and "
              "drift at that frequency plus the series' own measurement noise (2 e² / frame "
              "rate, e its RMS error)."]
    for preset in presets:
        lines += ["", f"### {preset}", "",
                  "| SNR | Series | RMS error (px) | Found | Line ÷ floor (threshold) | Predicted "
                  "| Frequency (Hz) | Amplitude (px) |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in camera[preset]:
            error = "0" if s["method"] == "truth" else f"{s['rms_error_px']:.4f}"
            lines.append(
                f"| {s['target_snr']:g} | {method_label(s['method'])} | {error} "
                f"| {s['correct']}/{s['records']} | {s['line_ratio']:.1f} ({s['threshold']:.1f}) "
                f"| {s['expected_line_ratio']:.1f} "
                f"| {pm(s['frequency_hz'], s['frequency_hz_std'])} "
                f"| {pm(s['amplitude_px'], s['amplitude_px_std'], 5)} |")

    lines += ["", "## Sudden jump between two frames (tracking mode alone, no fallback)", "",
              f"{knocks[0]['trials']} trials per row, jump in a random direction. First frame after "
              f"the jump: followed (accepted, within {KNOCK_MATCH_PX:g} px of the truth on both "
              "axes), refused (the loop would fall back), wrong (accepted, further off). "
              f"Alone: following {KNOCK_FOLLOW_FRAMES} frames later with no fallback. "
              "With fallback: the loop's answer on the first frame is within "
              f"{KNOCK_MATCH_PX:g} px.", "",
              "| Preset | SNR | Jump (px) | Followed | Refused | Wrong | Mean distance when wrong (px) "
              "| Alone, 5 frames later | With fallback |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k in knocks:
        share = lambda key: f"{100 * k[key] / k['trials']:.0f}%"
        distance = (f"{k['wrong_mean_distance_px']:.2f}"
                    if np.isfinite(k["wrong_mean_distance_px"]) else "-")
        lines.append(f"| {k['preset']} | {k['snr']:g} | {k['jump_px']:g} | {share('followed')} "
                     f"| {share('refused')} | {share('wrong')} | {distance} | {share('alone')} "
                     f"| {share('with_fallback')} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
