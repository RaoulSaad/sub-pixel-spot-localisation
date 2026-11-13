"""
The moving-spot scenario and the true trajectory it produces.

`Scenario` holds every parameter of the motion. `make_trajectory` draws one
true path (slow drift, random jitter, one periodic disturbance) on a time
grid finer than the frame rate, so that motion inside an exposure exists.
`per_frame` averages a fine-grid quantity over each frame's exposure.

Positions are (x, y) in pixels, as in simulator.py. Nothing here knows about
cameras, estimators or spectra.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Scenario:
    """
    The moving-spot scenario.
    Lengths are in pixels because the camera's plate scale (angle per pixel)
    is not known.
    """

    frame_rate_hz: float = 1000.0
    duration_s: float = 10.0
    sub_steps: int = 8                 # trajectory samples per frame period
    target_snr: float = 20.0
    start_xy: tuple[float, float] = (14.2, 15.3)

    # Slow drift: a steady creep plus a random walk.
    drift_rate_px_per_s: tuple[float, float] = (0.25, -0.10)
    wander_px_per_sqrt_s: float = 0.05   # RMS distance the random walk covers in 1 s

    # Random jitter: noise through a 2nd-order low-pass filter, same on both axes.
    jitter_rms_px: float = 0.20          # per axis
    jitter_corner_hz: float = 30.0

    # One periodic disturbance: a straight-line vibration.
    disturbance_hz: float = 24.37
    disturbance_amplitude_px: float = 0.10
    disturbance_angle_deg: float = 30.0  # direction of the vibration, from the x axis
    # Total change of the disturbance frequency over the record: a steady
    # ramp from disturbance_hz - drift/2 to disturbance_hz + drift/2. Zero
    # everywhere except the unsteady-frequency sweep.
    disturbance_drift_hz: float = 0.0

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.frame_rate_hz))

    @property
    def fine_rate_hz(self) -> float:
        return self.frame_rate_hz * self.sub_steps

    @property
    def direction(self) -> np.ndarray:
        angle = np.radians(self.disturbance_angle_deg)
        return np.array([np.cos(angle), np.sin(angle)])


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------

def jitter_filter(scenario: Scenario):
    """
    The low-pass filter that shapes the jitter, and the factor that scales
    its output to the requested RMS. White noise of unit variance through a
    filter with impulse response h comes out with variance sum(h^2).
    """
    sos = signal.butter(2, scenario.jitter_corner_hz, fs=scenario.fine_rate_hz, output="sos")
    impulse = np.zeros(int(20 * scenario.fine_rate_hz / scenario.jitter_corner_hz))
    impulse[0] = 1.0
    energy = float((signal.sosfilt(sos, impulse) ** 2).sum())
    return sos, scenario.jitter_rms_px / np.sqrt(energy), energy


def make_trajectory(scenario: Scenario, rng: np.random.Generator) -> dict:
    """
    The true spot position on a time grid `sub_steps` times finer than the
    frame rate, so that motion inside an exposure is represented. Returns
    the time axis and each component, shape (n, 2), plus their sum.
    """
    n = scenario.n_frames * scenario.sub_steps
    dt = 1.0 / scenario.fine_rate_hz
    t = (np.arange(n) + 0.5) * dt   # midpoints of the fine steps

    steady = np.outer(t, scenario.drift_rate_px_per_s)
    wander = np.cumsum(
        rng.normal(0.0, scenario.wander_px_per_sqrt_s * np.sqrt(dt), size=(n, 2)), axis=0
    )

    # Run the filter for a while before the record starts, so the jitter is
    # already at its steady level in the first frame.
    sos, scale, _ = jitter_filter(scenario)
    settle = int(10 * scenario.fine_rate_hz / scenario.jitter_corner_hz)
    white = rng.standard_normal((settle + n, 2))
    jitter = scale * signal.sosfilt(sos, white, axis=0)[settle:]

    phase = rng.uniform(0.0, 2 * np.pi)
    # Cycles added by the frequency ramp: the integral of
    # drift * (t / T - 1/2). Exactly zero when the drift is zero.
    ramp = scenario.disturbance_drift_hz * (t**2 / (2 * scenario.duration_s) - t / 2)
    wave = scenario.disturbance_amplitude_px * np.sin(
        2 * np.pi * scenario.disturbance_hz * t + phase + 2 * np.pi * ramp
    )
    disturbance = np.outer(wave, scenario.direction)

    drift = np.asarray(scenario.start_xy) + steady + wander
    return {
        "t": t, "drift": drift, "jitter": jitter, "disturbance": disturbance,
        "total": drift + jitter + disturbance, "phase": phase,
    }


def exposure_samples(scenario: Scenario, exposure_s: float) -> int:
    """Number of fine trajectory samples that fall inside one exposure."""
    exact = exposure_s * scenario.fine_rate_hz
    n = int(round(exact))
    if n < 1 or n > scenario.sub_steps or abs(exact - n) > 1e-9:
        raise ValueError("exposure must be a whole number of trajectory steps within one frame")
    return n


def per_frame(values: np.ndarray, scenario: Scenario, n_exposed: int) -> np.ndarray:
    """
    Average a fine-grid quantity over each frame's exposure. The exposure is
    the first `n_exposed` fine steps of each frame period; the shutter is
    closed for the rest.
    """
    shaped = values.reshape(scenario.n_frames, scenario.sub_steps, *values.shape[1:])
    return shaped[:, :n_exposed].mean(axis=1)
