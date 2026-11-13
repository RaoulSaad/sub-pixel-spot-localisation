"""
Spectral analysis of a measured position series: is there a periodic line,
and if so what are its frequency, amplitude and direction?

In the order the work is done: what the motion should look like in a
spectrum (models, for comparison only), cleaning a measured series, the
spectrum and its local floor, the detection threshold, the tone fit, and
`analyse`, which runs all of it on one series.

Spectra are one-sided power spectral densities in px^2/Hz: the area under
the curve between two frequencies is the variance of the motion in that
band.
"""

import warnings

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from trajectory import Scenario, jitter_filter

# Cleaning the measured series before spectral analysis.
GATE_SIGMAS = 5.0            # an estimate this many scatters from its neighbours is thrown out,
GATE_MIN_PX = 1.0            # but never one closer than this
GATE_FRAMES = 21             # length of the running median the gate compares against

# Line search.
SEARCH_BAND_HZ = (1.0, 450.0)
FLOOR_HALF_WIDTH_HZ = 4.0    # the local floor is estimated from this far on each side
MIN_FLOOR_BINS = 8           # but never from fewer bins than this per side
GUARD_BINS = 3               # bins next to the candidate left out of the floor
FALSE_ALARM = 0.01           # chance of announcing a line in a record that has none
NULL_TRIALS = 4000           # simulated noise-only records used to set the threshold
MATCH_BINS = 2.0             # a detection counts as correct within this many bins of the truth


# --------------------------------------------------------------------------
# What the motion should look like in a spectrum (for comparison only)
# --------------------------------------------------------------------------

def jitter_psd(f: np.ndarray, scenario: Scenario) -> np.ndarray:
    """
    Spectrum of the jitter on one axis. White noise of variance s^2 sampled
    at rate R has a flat spectrum of 2 * s^2 / R; the filter multiplies it
    by its squared gain |H(f)|^2.
    """
    sos, scale, _ = jitter_filter(scenario)
    _, response = signal.sosfreqz(sos, worN=np.atleast_1d(f), fs=scenario.fine_rate_hz)
    return 2 * scale**2 * np.abs(response) ** 2 / scenario.fine_rate_hz


def wander_psd(f: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Spectrum of the random walk on one axis: q^2 / (2 pi^2 f^2), falling as 1/f^2."""
    return scenario.wander_px_per_sqrt_s**2 / (2 * np.pi**2 * np.atleast_1d(f) ** 2)


def measurement_psd(rms_error_px: float, scenario: Scenario) -> float:
    """Frame-to-frame independent measurement error has a flat spectrum, 2 s^2 / frame rate."""
    return 2 * rms_error_px**2 / scenario.frame_rate_hz


def tone_precision(amplitude_px: float, floor_psd: float, duration_s: float) -> tuple[float, float]:
    """
    Best achievable standard deviation of a tone's amplitude and frequency,
    from the Cramer-Rao bound for a sinusoid in white noise (Rife1974) with
    the spectrum's level at the tone, `floor_psd` per axis, standing in for
    the white level:

        sigma_A = sqrt(S / T)
        sigma_f = sqrt(12 S) / (2 pi A T^1.5)

    Checked numerically by self_check().
    """
    sigma_a = np.sqrt(floor_psd / duration_s)
    if amplitude_px <= 0:
        return float(sigma_a), float("nan")
    sigma_f = np.sqrt(12 * floor_psd) / (2 * np.pi * amplitude_px * duration_s**1.5)
    return float(sigma_a), float(sigma_f)


def expected_line_ratio(amplitude_px: float, floor_psd: float, duration_s: float) -> float:
    """
    Expected height of the line over its local floor, for a steady tone on
    a bin centre: A^2 T / (6 S) + 1, with S the floor per axis. A tone
    between two bins gives up to 1.42 dB less.
    """
    return float(amplitude_px**2 * duration_s / (6 * floor_psd) + 1)


def unsteady_fit_factor(drift_hz: float, duration_s: float) -> float:
    """
    Share of a tone's amplitude that a fit at one fixed frequency can
    recover when the tone's frequency ramps by `drift_hz` over the record.

    A fixed-frequency sinusoid stays in step with the tone only part of the
    time. The fitted amplitude is the true one times |mean(exp(i * d(t)))|,
    with d(t) the phase by which the tone leads or lags the fitted
    sinusoid. The fit is free to choose its frequency, so the best fixed
    frequency within the ramp is taken.
    """
    if drift_hz == 0:
        return 1.0
    t = np.linspace(0.0, duration_s, 4001)
    ramp = 2 * np.pi * drift_hz * (t**2 / (2 * duration_s) - t / 2)
    offsets = np.linspace(-abs(drift_hz) / 2, abs(drift_hz) / 2, 201)
    return float(max(abs(np.exp(1j * (ramp - 2 * np.pi * offset * t)).mean())
                     for offset in offsets))


# --------------------------------------------------------------------------
# Cleaning and spectral analysis of a measured series
# --------------------------------------------------------------------------

def interpolate_gaps(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Fill NaN samples by straight lines between their accepted neighbours."""
    filled = xy.copy()
    for axis in range(xy.shape[1]):
        good = np.isfinite(xy[:, axis])
        filled[:, axis] = np.interp(t, t[good], xy[good, axis])
    return filled


def clean_series(t: np.ndarray, xy: np.ndarray):
    """
    Prepare a measured series for the spectrum. Returns (the series with
    refused and gated samples set to NaN, the same with gaps filled by
    interpolation, number gated).

    Gate: an accepted estimate far from the running median of its
    neighbours is not a small error, it is a lock on something that is not
    the spot. One such sample puts a spike in the series, and a spike raises
    the whole spectrum.

    "Far" is the larger of two distances. GATE_MIN_PX, because the true
    motion here never moves a pixel within 10 ms. And GATE_SIGMAS times the
    series' own scatter about that running median, so that a noisy camera is
    not punished for its ordinary noise: with a fixed 1 px gate, 2% of good
    frames were thrown out at SNR 6. The scatter is measured robustly
    (median absolute deviation times 1.4826, which equals the standard
    deviation for Gaussian data and ignores the outliers it is there to
    catch).

    The scatter rule alone is not safe. Smooth motion is not Gaussian about
    its running median, and with the scatter rule alone 2.5% of perfectly
    good samples were gated at high SNR, preferentially at the crests of
    the motion, which shaved 4% off the fitted amplitude.
    """
    if np.isfinite(xy[:, 0]).sum() < GATE_FRAMES:
        raise ValueError("too few accepted estimates to analyse")
    filled = interpolate_gaps(t, xy)
    residual = filled - median_filter(filled, size=(GATE_FRAMES, 1), mode="nearest")
    accepted = np.isfinite(xy[:, 0])
    scatter = 1.4826 * np.median(np.abs(residual[accepted]), axis=0)
    limit = np.maximum(GATE_SIGMAS * scatter, GATE_MIN_PX)
    gated = accepted & (np.abs(residual) > limit).any(axis=1)
    kept = xy.copy()
    kept[gated] = np.nan
    return kept, interpolate_gaps(t, kept), int(gated.sum())


def spectrum(xy: np.ndarray, frame_rate_hz: float):
    """
    Spectrum of x plus spectrum of y. Each axis has its straight-line trend
    removed and is multiplied by a Hann window before the Fourier transform.

    Why the window: a record that does not hold a whole number of cycles of
    some motion smears that motion's power over all frequencies. The drift
    is strong and slow, and without a window its smear would bury a weak
    line at 24 Hz. The Hann window tapers the record to zero at both ends,
    which confines the smear to a few neighbouring bins (Harris1978).
    """
    f, psd = signal.periodogram(xy, fs=frame_rate_hz, window="hann", detrend="linear",
                                scaling="density", axis=0)
    return f, psd.sum(axis=1)


def averaged_spectrum(xy: np.ndarray, frame_rate_hz: float, segment_s: float = 2.0):
    """
    Welch's method (Welch1967): cut the record into overlapping segments and
    average their spectra. Smoother than one long spectrum, at the cost of
    resolution. Used for display, not for detection. Returns the mean of the
    two axes, so it compares directly with a per-axis model.
    """
    f, psd = signal.welch(xy, fs=frame_rate_hz, window="hann", detrend="linear",
                          nperseg=int(segment_s * frame_rate_hz), axis=0)
    return f, psd.mean(axis=1)


def local_floor(psd: np.ndarray, f: np.ndarray):
    """
    The noise level around each bin: the mean of the neighbours within
    FLOOR_HALF_WIDTH_HZ on each side, skipping the GUARD_BINS nearest (a
    line spreads into them) and anything below the search band (the drift
    lives there and is far stronger than the floor). Returns (floor, number
    of neighbours used).

    The neighbourhood is set in hertz because the spectrum's shape is: over
    a few hertz the jitter spectrum is nearly level, over tens it is not. A
    short record has wide bins, so few neighbours fall inside that span and
    the floor is known less well; null_threshold() allows for that.
    """
    bin_width = f[1] - f[0]
    side = max(MIN_FLOOR_BINS, int(round(FLOOR_HALF_WIDTH_HZ / bin_width)))
    reach = side + GUARD_BINS
    usable = np.where(f >= SEARCH_BAND_HZ[0], psd, np.nan)
    padded = np.pad(usable, reach, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
    keep = np.ones(2 * reach + 1, dtype=bool)
    keep[side : side + 2 * GUARD_BINS + 1] = False
    neighbours = windows[:, keep]
    count = np.isfinite(neighbours).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # a bin with no usable neighbour gives NaN
        return np.nanmean(neighbours, axis=1), count


def line_statistic(xy: np.ndarray, frame_rate_hz: float):
    """
    Each bin's power divided by its local floor, NaN outside the search band.
    Returns (frequencies, ratio). This is the cell-averaging detector of
    radar (Finn1968) applied along the frequency axis: it adapts to a floor
    whose level changes with frequency.
    """
    f, psd = spectrum(xy, frame_rate_hz)
    floor, count = local_floor(psd, f)
    in_band = ((f >= SEARCH_BAND_HZ[0]) & (f <= SEARCH_BAND_HZ[1])
               & np.isfinite(floor) & (count >= MIN_FLOOR_BINS))
    with np.errstate(divide="ignore", invalid="ignore"):
        return f, np.where(in_band, psd / floor, np.nan)


def null_threshold(n_frames: int, frame_rate_hz: float, rng: np.random.Generator,
                   n_trials: int = NULL_TRIALS) -> float:
    """
    How strong the strongest bin gets when there is no line at all: the
    level that the largest ratio in the band exceeds in only FALSE_ALARM of
    noise-only records of this length.

    Found by simulation: make records of plain white noise, run them through
    the same spectrum and floor estimate, note the largest ratio of each,
    and take the 99th percentile. The noise level does not matter, because
    the statistic is a ratio.

    Why simulate. In theory a bin's power over the mean of n independent
    neighbours follows an F distribution, F(4, 4n) for two axes. But the
    Hann window makes neighbouring bins partly repeat each other, so the n
    neighbours are worth fewer than n, and the thousands of bins searched
    are not thousands of independent tries. Correcting both by the window's
    noise bandwidth (1.5) gives 2% false detections where 1% is intended.
    Simulating the whole procedure has no such constant to get wrong. It also allows for short records automatically:
    fewer neighbours make a rougher floor and a higher threshold.

    The simulated noise is white; the real floor slopes. self_check()
    measures the false-detection rate on the scenario's own drift and jitter
    to confirm the threshold carries over.
    """
    strongest = [
        np.nanmax(line_statistic(rng.standard_normal((n_frames, 2)), frame_rate_hz)[1])
        for _ in range(n_trials)
    ]
    return float(np.quantile(strongest, 1 - FALSE_ALARM))


def detect_line(xy: np.ndarray, frame_rate_hz: float, threshold: float) -> dict:
    """Strongest line in the search band, and whether it clears the threshold."""
    f, ratio = line_statistic(xy, frame_rate_hz)
    peak = int(np.nanargmax(ratio))
    return {
        "f": f, "ratio": ratio, "threshold": threshold,
        "peak_hz": float(f[peak]), "peak_ratio": float(ratio[peak]),
        "detected": bool(ratio[peak] > threshold),
    }


def tone_design(t: np.ndarray, frequency_hz: float) -> np.ndarray:
    """Columns: constant, straight line, cosine, sine."""
    phase = 2 * np.pi * frequency_hz * t
    return np.column_stack([np.ones_like(t), t - t.mean(), np.cos(phase), np.sin(phase)])


def fit_tone(t: np.ndarray, xy: np.ndarray, guess_hz: float) -> dict:
    """
    Least-squares fit of  offset + slope * t + c * cos + s * sin  to each
    axis, at one frequency shared by both. NaN samples are left out, so gaps
    need no interpolation here.

    The frequency is found by trying a grid within one bin of `guess_hz`,
    then refining around the best grid point. Two stages because the
    residual has side dips next to the main one, and a one-stage search
    started from a spectrum bin can settle in a side dip.

    The four tone coefficients form a 2 x 2 matrix (rows: x and y; columns:
    cosine and sine). For a straight-line vibration that matrix has only one
    non-zero singular value, the amplitude, and its singular vector is the
    direction. The second singular value measures any motion at that
    frequency across the main direction: zero for a straight line, noise in
    practice.
    """
    good = np.isfinite(xy).all(axis=1)
    t_good, xy_good = t[good], xy[good]
    duration = t_good[-1] - t_good[0]

    def residual(frequency_hz: float) -> float:
        design = tone_design(t_good, frequency_hz)
        coefficients, *_ = np.linalg.lstsq(design, xy_good, rcond=None)
        return float(((xy_good - design @ coefficients) ** 2).sum())

    grid = guess_hz + np.linspace(-1.0, 1.0, 41) / duration
    best = grid[int(np.argmin([residual(f) for f in grid]))]
    step = grid[1] - grid[0]
    # Golden-section search inside the two grid cells around the best point.
    low, high = best - step, best + step
    golden = (np.sqrt(5) - 1) / 2
    a, b = high - golden * (high - low), low + golden * (high - low)
    ra, rb = residual(a), residual(b)
    for _ in range(40):
        if ra < rb:
            high, b, rb = b, a, ra
            a = high - golden * (high - low)
            ra = residual(a)
        else:
            low, a, ra = a, b, rb
            b = low + golden * (high - low)
            rb = residual(b)
    frequency = (low + high) / 2

    coefficients, *_ = np.linalg.lstsq(tone_design(t_good, frequency), xy_good, rcond=None)
    tone = coefficients[2:].T   # rows: x, y; columns: cosine, sine
    u, singular, vt = np.linalg.svd(tone)
    direction = u[:, 0] if u[0, 0] >= 0 else -u[:, 0]
    return {
        "frequency_hz": float(frequency),
        "amplitude_px": float(singular[0]),
        "cross_amplitude_px": float(singular[1]),
        "angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
        "direction": direction,
        "tone": tone,
    }


def angle_difference_deg(a: float, b: float) -> float:
    """Difference between two line directions; a line at 30 degrees is also at 210."""
    return float((a - b + 90.0) % 180.0 - 90.0)


def alias_frequency(frequency_hz: float, frame_rate_hz: float) -> float:
    """Where a tone appears after sampling at the frame rate."""
    folded = frequency_hz % frame_rate_hz
    return float(min(folded, frame_rate_hz - folded))


def exposure_attenuation(frequency_hz: float, scenario: Scenario, n_exposed: int) -> float:
    """
    Factor by which averaging over the exposure shrinks a sinusoid.

    A camera integrates continuously, which gives sin(pi f E) / (pi f E)
    for an exposure of length E: 1 for slow motion, 0 when the exposure
    holds a whole cycle. The simulator averages n samples spaced d apart,
    which gives sin(n pi f d) / (n sin(pi f d)). The two agree to 0.1% up to
    150 Hz and differ by 1% at 620 Hz; this function returns the second,
    because that is what the frames contain.
    """
    step = np.pi * frequency_hz / scenario.fine_rate_hz
    if abs(np.sin(step)) < 1e-12:
        return 1.0
    return float(abs(np.sin(n_exposed * step) / (n_exposed * np.sin(step))))


def analyse(t: np.ndarray, xy: np.ndarray, scenario: Scenario, n_exposed: int,
            floor_psd: float, thresholds: dict) -> dict:
    """
    Everything the script concludes from one measured series: is there a
    line, and if so its frequency, amplitude and direction, each compared
    with what was injected.
    """
    kept, filled, n_gated = clean_series(t, xy)
    line = detect_line(filled, scenario.frame_rate_hz, thresholds[len(t)])
    duration = len(t) / scenario.frame_rate_hz
    apparent = alias_frequency(scenario.disturbance_hz, scenario.frame_rate_hz)
    seen_amplitude = scenario.disturbance_amplitude_px * exposure_attenuation(
        scenario.disturbance_hz, scenario, n_exposed)
    sigma_a, sigma_f = tone_precision(seen_amplitude, floor_psd, duration)
    row = {
        "duration_s": duration,
        "gated": n_gated,
        "line_ratio": line["peak_ratio"],
        "threshold": line["threshold"],
        "detected": line["detected"],
        "peak_hz": line["peak_hz"],
        "apparent_hz": apparent,
        "seen_amplitude_px": seen_amplitude,
        "predicted_sigma_amplitude_px": sigma_a,
        "predicted_sigma_frequency_hz": sigma_f,
        "expected_line_ratio": expected_line_ratio(seen_amplitude, floor_psd, duration),
        "correct": False,
        "frequency_hz": float("nan"), "amplitude_px": float("nan"),
        "cross_amplitude_px": float("nan"), "angle_deg": float("nan"),
        "frequency_error_hz": float("nan"), "amplitude_error_px": float("nan"),
        "angle_error_deg": float("nan"),
    }
    if line["detected"]:
        tone = fit_tone(t, kept, line["peak_hz"])
        row |= {key: tone[key] for key in
                ("frequency_hz", "amplitude_px", "cross_amplitude_px", "angle_deg")}
        # A tone whose frequency ramps occupies a band, not a point; a fit
        # anywhere in that band is a correct detection.
        row["correct"] = bool(
            scenario.disturbance_amplitude_px > 0
            and abs(tone["frequency_hz"] - apparent)
            <= MATCH_BINS / duration + abs(scenario.disturbance_drift_hz) / 2
        )
        row["frequency_error_hz"] = tone["frequency_hz"] - apparent
        row["amplitude_error_px"] = tone["amplitude_px"] - seen_amplitude
        row["angle_error_deg"] = angle_difference_deg(tone["angle_deg"],
                                                      scenario.disturbance_angle_deg)
    return row


def fold(t: np.ndarray, xy: np.ndarray, tone: dict, n_bins: int = 20):
    """
    Average the motion along the fitted direction over one cycle of the
    fitted frequency. Slow motion is removed first by subtracting a running
    mean exactly five cycles long: averaging over whole cycles leaves no
    trace of the tone itself, so the subtraction takes out the drift and
    leaves the tone untouched. Returns (phase centres in cycles, mean,
    standard error).
    """
    along = interpolate_gaps(t, xy) @ tone["direction"]
    span = int(round(5 / (tone["frequency_hz"] * (t[1] - t[0]))))
    smooth = np.convolve(along, np.ones(span) / span, mode="same")
    inner = slice(span, len(along) - span)   # the running mean is wrong near the ends
    residual = (along - smooth)[inner]
    cycle = (tone["frequency_hz"] * t[inner]) % 1.0
    which = np.minimum((cycle * n_bins).astype(int), n_bins - 1)
    centres = (np.arange(n_bins) + 0.5) / n_bins
    mean = np.array([residual[which == b].mean() for b in range(n_bins)])
    # Samples inside one cycle move together (the jitter is slow compared
    # with a frame), so the number of independent values per bin is the
    # number of cycles, not the number of samples.
    cycles = tone["frequency_hz"] * (t[inner][-1] - t[inner][0])
    error = np.array([residual[which == b].std(ddof=1) for b in range(n_bins)]) / np.sqrt(cycles)
    return centres, mean, error
