"""
Sub-pixel spot position estimators.

Every estimator has the same call signature:

    estimate = some_estimator(image_dn, camera, hot_mask=None, ...)

and returns an `Estimate`. On failure it returns `ok=False` with a reason and
NaN coordinates; it never returns a bad position silently.

Conventions (same as simulator.py):
  - pixel centres have integer coordinates
  - image[row, column] is (y, x); positions are returned as (x, y)
  - all positions are in full-frame pixel coordinates, never window coordinates
"""

from dataclasses import dataclass
import functools

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from scipy.optimize import Bounds, least_squares, minimize
from scipy.special import erf, xlogy

from simulator import CameraConfig, SensorMaps


FloatImage = NDArray[np.float64]
BoolImage = NDArray[np.bool_]

HALF_WIDTH = 5                                      # window is (2 * HALF_WIDTH + 1) pixels wide: 11 x 11
SIGMA_GUESS = 1.75                                  # nominal spot sigma in pixels (1/e^2 diameter of 7 px)
BACKGROUND_EXCLUDE_RADIUS = 4 * SIGMA_GUESS         # spot is ~0.03% of peak out here
DETECTION_THRESHOLD = 5.0                           # spot core must exceed this many background sigmas
SATURATION_MARGIN_SIGMAS = 4.0                      # read-noise sigmas below the top DN code
MIN_BACKGROUND_PIXELS = 50                          # fewer than this and the plane fit is not trusted
RECENTRE_ROUNDS = 2                                 # times the window may be re-centred on the centroid
SIGMA_BOUNDS = (0.5, 4.0)                           # allowed fitted spot sigma in pixels
AMPLITUDE_BOUND_FACTOR = 10.0                       # fitted amplitude may reach this times the guess
EDGE_TOLERANCE_PX = 0.01                            # a fitted position this close to a bound is a failure
MLE_GRADIENT_TOLERANCE = 1e-2                       # likelihood fit stops when its gradient is below this


# --------------------------------------------------------------------------
# Helpers & Data classes
# --------------------------------------------------------------------------

@dataclass
class Estimate:
    x: float
    y: float
    ok: bool
    reason: str = ""                   # Stores error in case of failure
    n_iter: int = 0
    amplitude_e: float = float("nan")  # fitted total signal, fit methods only
    sigma_px: float = float("nan")     # fitted spot sigma, fit methods only
    n_saturated: int = 0               # saturated pixels inside the window


class EstimationFailure(Exception):
    """Raise anywhere inside an estimator to report a failure with a reason."""


def reports_failure(func):
    """Turn an EstimationFailure raised inside `func` into Estimate(ok=False)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EstimationFailure as exc:
            return Estimate(x=float("nan"), y=float("nan"), ok=False, reason=str(exc))

    return wrapper


@dataclass
class Window:
    """A square cut-out around the spot, plus what the estimators need to know."""

    data: FloatImage        # electrons, background not subtracted
    background: FloatImage  # fitted background plane evaluated on the window
    valid: BoolImage        # False for hot or saturated pixels; the fits skip these
    saturated: BoolImage    # True for pixels at full well
    xs: FloatImage          # 1-D column coordinates of the window, full-frame
    ys: FloatImage          # 1-D row coordinates of the window, full-frame
    read_noise_e: float
    background_std_e: float  # measured scatter of background pixels

    @property
    def signal(self) -> FloatImage:
        return self.data - self.background

    @property
    def centroid_signal(self) -> FloatImage:
        """
        The signal as the centroid methods need it. The two kinds of bad
        pixel are treated differently:

          - hot pixels are filled from their neighbours (see fill_invalid);
          - saturated pixels keep their clipped value. They come as a block
            at the spot core, where there are no good neighbours to fill
            from, and a flat-topped spot is still symmetric about its true
            centre, so the clipped values barely move the centre of mass.
        """
        return fill_invalid(self.signal, self.valid | self.saturated)


def hot_pixel_mask(sensor_maps: SensorMaps, camera: CameraConfig) -> BoolImage:
    """
    Known hot pixels, True where hot.

    This reads the simulator's ground-truth dark map. On a real camera the
    same mask comes from a dark-frame calibration, so using it here is fair.
    """
    return sensor_maps.dark_current_e_per_s > 10 * camera.dark_current_e_per_s


def fill_invalid(image: FloatImage, valid: BoolImage) -> FloatImage:
    """
    Replace each invalid pixel by the median of its valid 3 x 3 neighbours.

    The centroid needs this: simply dropping a pixel leaves a hole that shifts
    the centre of mass away from it. Limitation: on the spot core the median
    of the neighbours underestimates the true value, so a hot pixel right on
    the peak still costs some accuracy. The fit methods do not use this; they
    leave invalid pixels out of the sum instead.
    """
    filled = image.copy()
    for row, col in zip(*np.nonzero(~valid)):
        rows = slice(max(row - 1, 0), row + 2)
        cols = slice(max(col - 1, 0), col + 2)
        neighbours = image[rows, cols][valid[rows, cols]]
        filled[row, col] = np.median(neighbours) if neighbours.size else 0.0
    return filled


def preprocess(
    image_dn: NDArray[np.uint16],
    camera: CameraConfig,
    hot_mask: BoolImage | None = None,
    half_width: int = HALF_WIDTH,
) -> Window:
    """Shared front end: DN to electrons, mask, locate, background, window."""
    image_e = dn_to_electrons(image_dn, camera)

    hot = np.zeros(image_e.shape, dtype=bool) if hot_mask is None else hot_mask

    # Saturated pixels. Full well maps to the top DN code, but the sensor adds
    # read noise after the full-well clip, so a saturated pixel can read a few
    # DN below the top. Testing for the top code alone misses a third to a
    # half of them (measured: a third on the silicon preset); flag everything
    # within a few read-noise sigmas of it instead.
    max_dn = 2**camera.bit_depth - 1
    margin_dn = SATURATION_MARGIN_SIGMAS * camera.read_noise_e / camera.electrons_per_dn
    saturated = image_dn >= max_dn - margin_dn

    # Two kinds of bad pixel, treated differently:
    #   - `valid` (neither hot nor saturated) is what the background fit and
    #     the Gaussian fits use: a saturated pixel's value is wrong, so it
    #     cannot be compared with a model.
    #   - `~hot` is what the peak search, the detection test and the centroid
    #     use: a saturated pixel still says "the spot is here".
    valid = ~hot & ~saturated

    # Two passes, because each step needs the other. Finding the spot on the
    # raw image fails at low signal: the bright corner of the background
    # gradient plus a noise spike can beat a dim spot. So fit a rough plane to
    # everything, find the spot on what is left, then refit the plane with
    # the spot excluded.
    rough_plane, _ = fit_background_plane(image_e, valid, None, 0.0)
    peak_xy = find_peak(image_e - rough_plane, ~hot)
    plane, background_std = fit_background_plane(
        image_e, valid, peak_xy, BACKGROUND_EXCLUDE_RADIUS
    )
    window = cut_window(
        image_e, plane, valid, saturated, peak_xy, half_width,
        camera.read_noise_e, background_std,
    )

    # Is there a spot at all? Compare the mean of the central 3 x 3 pixels
    # with the background scatter (a mean of n pixels has scatter
    # std / sqrt(n)). Hot pixels do not count: a hot pixel next to the peak
    # must not pass the test on the spot's behalf.
    centre = slice(half_width - 1, half_width + 2)
    not_hot = window.valid | window.saturated
    core = window.signal[centre, centre][not_hot[centre, centre]]
    if core.size == 0:
        raise EstimationFailure("no usable pixels at the spot core")
    if core.mean() < DETECTION_THRESHOLD * background_std / np.sqrt(core.size):
        raise EstimationFailure("no spot above background")

    # Re-centre the window on the pixel nearest the centroid. The brightest
    # pixel is a poor centre when the spot is saturated: its top is flat,
    # every pixel on it ties, and the search returns the first one, up to
    # two pixels from the middle. An off-centre window clips one tail more
    # than the other and biases the centroid. Two rounds are enough because
    # the peak search is never more than two pixels out.
    for _ in range(RECENTRE_ROUNDS):
        x, y = centroid_of(window)
        nearest = (int(round(x)), int(round(y)))
        if nearest == peak_xy:
            break
        peak_xy = nearest
        plane, background_std = fit_background_plane(
            image_e, valid, peak_xy, BACKGROUND_EXCLUDE_RADIUS
        )
        window = cut_window(
            image_e, plane, valid, saturated, peak_xy, half_width,
            camera.read_noise_e, background_std,
        )

    return window


# --------------------------------------------------------------------------
# DN to electrons
# --------------------------------------------------------------------------

def dn_to_electrons(image_dn: NDArray[np.uint16], camera: CameraConfig) -> FloatImage:
    """
    Undo the simulator's `digitize`: electrons = (DN - bias) * electrons_per_dn.

    image_dn is uint16, so it is converted to float64 first: subtracting the
    bias from unsigned integers would wrap negative values around to about
    65,000.
    """

    img = image_dn.astype(np.float64)
    img = (img - camera.bias_dn) * camera.electrons_per_dn

    return img


# --------------------------------------------------------------------------
# Coarse spot location
# --------------------------------------------------------------------------

def find_peak(signal_e: FloatImage, valid: BoolImage) -> tuple[int, int]:
    """
    Return the integer pixel (x, y) nearest the spot. `signal_e` is the frame
    with a rough background plane already subtracted (see `preprocess`).

    Invalid pixels are first overwritten with the image median, so a hot
    pixel cannot be chosen as the peak. The image is then smoothed with a
    3 x 3 box: a single noisy pixel can beat the spot at low SNR; a 3 x 3
    average cannot do so as easily.
    """

    img = signal_e.copy()  
    img[~valid] = np.median(signal_e)

    smooth = uniform_filter(img, size=3, mode="nearest")
    row, col = np.unravel_index(np.argmax(smooth), smooth.shape)

    return int(col), int(row)



# --------------------------------------------------------------------------
# Background plane
# --------------------------------------------------------------------------

def fit_background_plane(
    image_e: FloatImage,
    valid: BoolImage,
    peak_xy: tuple[int, int] | None,
    exclude_radius: float,
) -> tuple[FloatImage, float]:
    """
    Fit B(x, y) = c0 + c1 * x + c2 * y to the pixels far from the spot.
    If peak_xy is None, use every valid pixel (the rough first pass).

    Returns (plane over the full frame, standard deviation of the residuals).

    Why a plane and not one number: the simulator's background has a gradient.
    A leftover tilt drags the centroid sideways by an amount that grows as the
    signal falls.

    The plane is fitted by ordinary least squares, with columns [1, x, y],
    to the pixels that are valid and (when peak_xy is given) further than
    exclude_radius from it. Too few such pixels is a failure. The standard
    deviation of the residuals over those pixels is returned with the plane.

    Hot pixels that are not in the mask are not handled: there is no sigma
    clipping of the residuals.
    """
    ys, xs = np.indices(image_e.shape)

    use = valid.copy()
    if peak_xy is not None:
        distance = np.hypot(xs - peak_xy[0], ys - peak_xy[1])
        use &= distance > exclude_radius
    if use.sum() < MIN_BACKGROUND_PIXELS:
        raise EstimationFailure("too few background pixels to fit the plane")

    values = image_e[use]
    design = np.column_stack([np.ones(values.size), xs[use], ys[use]])
    (c0, c1, c2), *_ = np.linalg.lstsq(design, values, rcond=None)

    plane = c0 + c1 * xs + c2 * ys
    residual_std = float(np.std(values - plane[use]))
    return plane, residual_std


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

def cut_window(
    image_e: FloatImage,
    plane: FloatImage,
    valid: BoolImage,
    saturated: BoolImage,
    peak_xy: tuple[int, int],
    half_width: int,
    read_noise_e: float,
    background_std_e: float,
) -> Window:
    """
    Cut a (2 * half_width + 1) square centred on peak_xy from all four images.

    The window is never shrunk at the frame edge: an asymmetric window
    biases the centroid, so a window that leaves the frame is a failure.
    xs and ys are full-frame coordinates, so no estimator adds an offset
    back by hand.
    """
    x, y = peak_xy
    height, width = image_e.shape
    if (
        x - half_width < 0
        or y - half_width < 0
        or x + half_width >= width
        or y + half_width >= height
    ):
        raise EstimationFailure("window off frame")

    rows = slice(y - half_width, y + half_width + 1)
    cols = slice(x - half_width, x + half_width + 1)
    return Window(
        data=image_e[rows, cols],
        background=plane[rows, cols],
        valid=valid[rows, cols],
        saturated=saturated[rows, cols],
        xs=np.arange(cols.start, cols.stop, dtype=np.float64),
        ys=np.arange(rows.start, rows.stop, dtype=np.float64),
        read_noise_e=read_noise_e,
        background_std_e=background_std_e,
    )


# --------------------------------------------------------------------------
# Windowed centroid
# --------------------------------------------------------------------------

def centroid_of(window: Window) -> tuple[float, float]:
    """
    Centre of mass of the background-subtracted window. Used by the centroid
    estimator and as the starting point for every other estimator.

        x = sum(x_i * s_i) / sum(s_i),   s = signal with hot pixels filled

    Hot pixels are filled and saturated ones kept (window.centroid_signal).
    No net signal, or a centroid outside the window, is a failure.

    Negative pixels are not clipped to zero. Noise in the wings is symmetric
    around zero and cancels on average; clipping keeps only the positive
    half, adds fake light everywhere, and pulls the result to the window
    centre.
    """
    s = window.centroid_signal
    total = s.sum()
    if total <= 0:
        raise EstimationFailure("no net signal in window")

    x = float(s.sum(axis=0) @ window.xs / total)
    y = float(s.sum(axis=1) @ window.ys / total)

    # The window spans half a pixel beyond its first and last pixel centres.
    inside = (
        window.xs[0] - 0.5 <= x <= window.xs[-1] + 0.5
        and window.ys[0] - 0.5 <= y <= window.ys[-1] + 0.5
    )
    if not inside:
        raise EstimationFailure("centroid outside window")
    return x, y


@reports_failure
def windowed_centroid(
    image_dn: NDArray[np.uint16],
    camera: CameraConfig,
    hot_mask: BoolImage | None = None,
    half_width: int = HALF_WIDTH,
) -> Estimate:
    """
    Centre of mass of the background-subtracted window (see centroid_of).

    A saturated spot is not refused. The clipped pixels keep their value,
    their number is reported in Estimate.n_saturated, and the caller decides
    what to do with it. Measured on the silicon preset, RMS error against the
    number of saturated pixels:

      plain centroid     0.007 px with none; at most 0.012 px up to 29;
                         0.017 to 0.020 px from 30 to 39.
      weighted centroid  0.006 px with none; 0.011 to 0.015 px from 10 to
                         29; fails to converge on some frames above 40.

    The error grows with the count and there is no count at which it jumps,
    so no count is made a hard limit: nothing shows that 29 is acceptable
    and 30 is not. What is refused is a calculation that fails (no net
    signal, centroid outside the window, no convergence).

    The Gaussian fits skip saturated pixels and locate the spot from its
    unsaturated wings; their error was unchanged up to 59 saturated pixels,
    the most that was tested.
    """
    window = preprocess(image_dn, camera, hot_mask, half_width)
    x, y = centroid_of(window)
    return Estimate(x=x, y=y, ok=True, n_saturated=int(window.saturated.sum()))


# --------------------------------------------------------------------------
# The spot model shared by both fits
# --------------------------------------------------------------------------

def gaussian_model(
    xs: FloatImage, ys: FloatImage, x0: float, y0: float, sigma: float
) -> FloatImage:
    """
    Fraction of the spot's light landing in each window pixel, for a spot at
    (x0, y0). Shape (len(ys), len(xs)); sums to slightly under 1 because the
    window cuts the tails.

    This is the simulator's render_pixel_integrated_gaussian restricted to the window:
      1. Pixel edges along x: xs - 0.5, plus one final edge at xs[-1] + 0.5.
      2. cdf = 0.5 * (1 + erf((edges - x0) / (sqrt(2) * sigma))).
      3. Per-pixel fraction along x = np.diff(cdf). Same along y.
      4. Return np.outer(fraction_y, fraction_x).

    Why integrate over the pixel instead of sampling the Gaussian at the
    pixel centre: the model must describe what the sensor measures. A sampled
    model fits a slightly wrong width and is no longer the model the data
    came from.
    """
    return np.outer(
        _pixel_fractions(ys, y0, sigma),
        _pixel_fractions(xs, x0, sigma),
    )


def _pixel_fractions(centres: FloatImage, centre0: float, sigma: float) -> FloatImage:
    """Fraction of a 1-D Gaussian at `centre0` landing in each pixel."""
    edges = np.append(centres - 0.5, centres[-1] + 0.5)
    cdf = 0.5 * (1 + erf((edges - centre0) / (np.sqrt(2) * sigma)))
    return np.diff(cdf)


def _pixel_fraction_derivatives(
    centres: FloatImage, centre0: float, sigma: float
) -> tuple[FloatImage, FloatImage]:
    """
    Derivatives of `_pixel_fractions` with respect to centre0 and to sigma.

    A pixel's fraction is cdf(z_hi) - cdf(z_lo), with z = (edge - centre0) /
    sigma at its two edges. The slope of the cdf is the bell curve itself,
    pdf(z) = exp(-z^2 / 2) / sqrt(2 * pi), so by the chain rule:

        d / d centre0 = -(pdf(z_hi) - pdf(z_lo)) / sigma
        d / d sigma   = -(z_hi * pdf(z_hi) - z_lo * pdf(z_lo)) / sigma
    """
    edges = np.append(centres - 0.5, centres[-1] + 0.5)
    z = (edges - centre0) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return -np.diff(pdf) / sigma, -np.diff(z * pdf) / sigma


def initial_guess(window: Window, sigma_guess: float) -> tuple[float, float, float, float]:
    """
    Starting parameters (x0, y0, amplitude, sigma) for the fits:
    centroid_of(window) for the position, the sum of window.centroid_signal
    for the amplitude, sigma_guess for the width. Raise EstimationFailure if
    the amplitude is not positive.

    The amplitude uses centroid_signal, not signal[valid]: leaving out a hot
    or saturated pixel at the core would start the fit far too low.
    """
    x0, y0 = centroid_of(window)  # also fails if there is no net signal
    amplitude = float(window.centroid_signal.sum())
    return x0, y0, amplitude, sigma_guess


def fit_bounds(window: Window, amplitude_max: float) -> tuple[FloatImage, FloatImage]:
    """
    Lower and upper bounds on (x0, y0, amplitude, sigma), shared by both
    fits: position inside the window, amplitude from 0 to amplitude_max,
    sigma inside SIGMA_BOUNDS.
    """
    lower = np.array([window.xs[0], window.ys[0], 0.0, SIGMA_BOUNDS[0]])
    upper = np.array([window.xs[-1], window.ys[-1], amplitude_max, SIGMA_BOUNDS[1]])
    return lower, upper


def check_fit(
    converged: bool, x: float, y: float, sigma: float,
    lower: FloatImage, upper: FloatImage,
) -> None:
    """
    Failure checks shared by both fits. A parameter stopped by a bound was
    not found by the data, so the result is not a measurement.
    """
    if not converged:
        raise EstimationFailure("fit did not converge")
    if (
        min(x - lower[0], upper[0] - x) < EDGE_TOLERANCE_PX
        or min(y - lower[1], upper[1] - y) < EDGE_TOLERANCE_PX
    ):
        raise EstimationFailure("fitted position at window edge")
    if np.isclose(sigma, lower[3]) or np.isclose(sigma, upper[3]):
        raise EstimationFailure("fitted sigma at bound")


# --------------------------------------------------------------------------
# Gaussian fit, least squares
# --------------------------------------------------------------------------

@reports_failure
def gaussian_fit_lsq(
    image_dn: NDArray[np.uint16],
    camera: CameraConfig,
    hot_mask: BoolImage | None = None,
    half_width: int = HALF_WIDTH,
    sigma_guess: float = SIGMA_GUESS,
) -> Estimate:
    """
    Find (x0, y0, A, sigma) minimising  sum((data - model)^2)  over valid
    pixels, where  model = A * gaussian_model(...) + window.background.

    The background plane is held fixed, not fitted: it was measured from
    hundreds of pixels outside the window, far more than the window has.

    Solved with scipy.optimize.least_squares from initial_guess, with
    bounds: x0 and y0 inside the window, A from 0 to 10 times its starting
    value, sigma from 0.5 to 4. The amplitude is scaled (x_scale): A is
    thousands to millions while x0 is about 15, and without scaling the
    optimiser takes badly sized steps.

    A fit that did not converge, or that ended with x0 or y0 within 0.01 px
    of a bound or with sigma at a bound, is a failure.
    """
    window = preprocess(image_dn, camera, hot_mask, half_width)
    x0, y0, a0, s0 = initial_guess(window, sigma_guess)

    def residuals(theta):
        x, y, amplitude, sigma = theta
        spot = amplitude * gaussian_model(window.xs, window.ys, x, y, sigma)
        return (spot + window.background - window.data)[window.valid]

    lower, upper = fit_bounds(window, AMPLITUDE_BOUND_FACTOR * a0)
    # least_squares refuses a start outside the bounds; the centroid can sit
    # up to half a pixel beyond the outermost pixel centre.
    start = np.clip([x0, y0, a0, s0], lower, upper)

    result = least_squares(
        residuals, start, bounds=(lower, upper), x_scale=[1.0, 1.0, a0, 1.0]
    )
    x, y, amplitude, sigma = result.x
    check_fit(result.success, x, y, sigma, lower, upper)

    return Estimate(
        x=float(x), y=float(y), ok=True, n_iter=int(result.nfev),
        amplitude_e=float(amplitude), sigma_px=float(sigma),
        n_saturated=int(window.saturated.sum()),
    )


# --------------------------------------------------------------------------
# Gaussian fit, maximum likelihood
# --------------------------------------------------------------------------

@reports_failure
def gaussian_fit_mle(
    image_dn: NDArray[np.uint16],
    camera: CameraConfig,
    hot_mask: BoolImage | None = None,
    half_width: int = HALF_WIDTH,
    sigma_guess: float = SIGMA_GUESS,
) -> Estimate:
    """
    Same model as the least-squares fit, different cost: the Poisson negative
    log-likelihood

        nll(theta) = sum(mu - d * log(mu))   over valid pixels

    Read noise is Gaussian, not Poisson. Standard workaround: a Gaussian of
    variance r^2 looks like a Poisson of mean r^2 shifted down by r^2. So add
    r^2 = read_noise_e^2 to both sides:

        d  = window.data + r^2                       (clip at 0)
        mu = A * gaussian_model(...) + background + r^2   (clip at a tiny
                                                           positive number)

    Now every pixel has variance about mu, which is what Poisson assumes.

    Solved with scipy.optimize.minimize (L-BFGS-B), from the same start and
    with the same bounds and failure checks as the least-squares fit. The
    amplitude is optimised as a ratio a = A / A0, so that all four
    parameters are of order 1 to 10: L-BFGS-B has no x_scale option.

    The cost that is actually minimised is the nll minus its smallest
    possible value (the value it takes when mu equals d in every pixel):

        cost(theta) = sum(mu - d + d * log(d / mu))

    The subtracted part does not depend on theta, so the minimum is in the
    same place. The reason is numerical: the raw nll is tens of millions on
    the InGaAs preset, and the optimiser stops when the cost changes by less
    than about 2e-9 of its own size, which there is still 0.0015 px away
    from the minimum. This form is of order 100, so the same rule stops it
    within a millionth of a pixel.

    The gradient is supplied analytically, not left to the optimiser's
    finite differences, which need five cost evaluations per gradient where
    this needs one. It halved the time per fit. The chain rule gives, for
    each parameter p,

        d cost / d p = sum((1 - d / mu) * d mu / d p)

    Stopping rule. L-BFGS-B stops when the gradient is below a tolerance,
    by default 1e-5. That is stricter than the cost can support: close to
    the minimum a further step changes the cost by less than its rounding
    error (about 1e-12), the optimiser cannot confirm any improvement, and
    it reports failure ("ABNORMAL") although it is already at the minimum.
    This happened on about 0.2% of frames, with the gradient stuck between
    1e-4 and 4e-4. MLE_GRADIENT_TOLERANCE = 1e-2 stops it before that
    point. The position is then within gradient / curvature of the minimum,
    and the curvature is 1 / (position noise)^2, so the leftover error is
    about 1e-2 * (position noise)^2. Measured against the default
    tolerance: at most 0.0014 px at the dimmest signal tested, where the
    noise itself is 0.27 px, and under 0.00001 px at the silicon preset.

    Why this differs from least squares: least squares trusts every pixel
    equally. This cost trusts each pixel in proportion to 1 / mu, so bright
    pixels, which carry more shot noise, count for less.
    """
    window = preprocess(image_dn, camera, hot_mask, half_width)
    x0, y0, a0, s0 = initial_guess(window, sigma_guess)

    valid = window.valid
    r2 = window.read_noise_e**2
    data = np.clip(window.data + r2, 0.0, None)[valid]
    no_spot = (window.background + r2)[valid]  # mu when the amplitude is zero

    def cost_and_gradient(theta):
        x, y, ratio, sigma = theta
        amplitude = ratio * a0
        # The model is separable: spot = amplitude * outer(fy, fx).
        fx = _pixel_fractions(window.xs, x, sigma)
        fy = _pixel_fractions(window.ys, y, sigma)
        dfx_dx, dfx_dsigma = _pixel_fraction_derivatives(window.xs, x, sigma)
        dfy_dy, dfy_dsigma = _pixel_fraction_derivatives(window.ys, y, sigma)

        mu = np.maximum(amplitude * np.outer(fy, fx)[valid] + no_spot, 1e-9)
        # xlogy(d, d / mu) is d * log(d / mu), and 0 where d is 0.
        cost = float(np.sum(mu - data + xlogy(data, data / mu)))

        # d cost / d mu per pixel, laid back on the window with zeros at the
        # skipped pixels. Then `fy @ slope @ fx` is the sum over all pixels
        # of fy[row] * slope[row, col] * fx[col].
        slope = np.zeros(window.data.shape)
        slope[valid] = 1 - data / mu
        gradient = np.array([
            amplitude * (fy @ slope @ dfx_dx),
            amplitude * (dfy_dy @ slope @ fx),
            a0 * (fy @ slope @ fx),
            amplitude * (dfy_dsigma @ slope @ fx + fy @ slope @ dfx_dsigma),
        ])
        return cost, gradient

    lower, upper = fit_bounds(window, AMPLITUDE_BOUND_FACTOR)
    start = np.clip([x0, y0, 1.0, s0], lower, upper)

    result = minimize(
        cost_and_gradient, start, jac=True, method="L-BFGS-B",
        bounds=Bounds(lower, upper), options={"gtol": MLE_GRADIENT_TOLERANCE},
    )
    x, y, ratio, sigma = result.x
    check_fit(result.success, x, y, sigma, lower, upper)

    return Estimate(
        x=float(x), y=float(y), ok=True, n_iter=int(result.nit),
        amplitude_e=float(ratio * a0), sigma_px=float(sigma),
        n_saturated=int(window.saturated.sum()),
    )


# --------------------------------------------------------------------------
# Iteratively weighted centroid
# --------------------------------------------------------------------------

@reports_failure
def iteratively_weighted_centroid(
    image_dn: NDArray[np.uint16],
    camera: CameraConfig,
    hot_mask: BoolImage | None = None,
    half_width: int = HALF_WIDTH,
    sigma_weight: float = SIGMA_GUESS,
    max_iter: int = 30,
    tolerance_px: float = 1e-4,
) -> Estimate:
    """
    A centroid where each pixel is multiplied by a Gaussian weight centred on
    the current estimate, repeated until the estimate stops moving:

        w = gaussian_model(xs, ys, x_k, y_k, sigma_weight)
        x_next = sum(x * w * s) / sum(w * s)

    The weight suppresses the noisy wings, which is where the plain centroid
    loses at low signal. One pass is biased toward where the weight sits: with
    sigma_weight equal to the spot sigma it moves only halfway to the true
    position. Iterating removes that, halving the error each pass.

    It starts from the plain centroid and stops when a pass moves the
    estimate by less than tolerance_px. No net weighted signal, an estimate
    that leaves the window, or max_iter passes without converging is a
    failure.

    A saturated spot is not refused (see windowed_centroid); the number of
    saturated pixels is reported in Estimate.n_saturated.
    """
    window = preprocess(image_dn, camera, hot_mask, half_width)
    s = window.centroid_signal
    x, y = centroid_of(window)

    for iteration in range(1, max_iter + 1):
        weighted = gaussian_model(window.xs, window.ys, x, y, sigma_weight) * s
        total = weighted.sum()
        if total <= 0:
            raise EstimationFailure("no net weighted signal")

        x_next = float(weighted.sum(axis=0) @ window.xs / total)
        y_next = float(weighted.sum(axis=1) @ window.ys / total)
        move = np.hypot(x_next - x, y_next - y)
        x, y = x_next, y_next

        if not (
            window.xs[0] <= x <= window.xs[-1]
            and window.ys[0] <= y <= window.ys[-1]
        ):
            raise EstimationFailure("estimate left the window")
        # Each pass halves the error, so the distance still to go is about
        # the size of the last move.
        if move < tolerance_px:
            return Estimate(
                x=x, y=y, ok=True, n_iter=iteration,
                n_saturated=int(window.saturated.sum()),
            )

    raise EstimationFailure("weighted centroid did not converge")
