"""
The theoretical floor on position error: the Cramer-Rao bound for a
pixel-integrated Gaussian spot on this camera model.

"""

import numpy as np

import estimators as est
from simulator import render_background


def cramer_rao_floor_px(camera, spot, background, total_signal_e: float) -> float:
    """
    Smallest standard deviation any unbiased estimator of x can reach.

    Each pixel i has an expected value mu_i (spot + background + dark) and a
    noise variance var_i = mu_i + read_noise^2 + gain^2 / 12 (shot noise,
    read noise, quantization). The information the frame carries about the
    parameters theta = (x0, y0, amplitude, sigma) is the 4 x 4 matrix

        I[a, b] = sum_i (d mu_i / d theta_a) * (d mu_i / d theta_b) / var_i

    and the floor on x0 is sqrt(inverse(I)[0, 0]). Inverting the full matrix,
    instead of using only the x0 term, charges for not knowing the amplitude
    and the width in advance.

    Idealisations, so the floor is a lower limit and not a prediction: the
    whole frame is used, the background is known exactly, there is no
    pixel-gain non-uniformity, no hot pixel and no saturation. The floor
    varies slightly with the sub-pixel position, so it is averaged over a
    grid of positions.
    """
    height, width = camera.shape
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    floor_e = (
        render_background(camera.shape, background)
        + camera.dark_current_e_per_s * camera.exposure_s
    )
    quantization_var = camera.electrons_per_dn**2 / 12
    centre_x, centre_y = (round(c) for c in spot.center_xy)
    sigma = spot.sigma_x

    variances = []
    for dx in np.linspace(-0.4, 0.4, 5):
        for dy in np.linspace(-0.4, 0.4, 5):
            x0, y0 = centre_x + dx, centre_y + dy
            fx = est._pixel_fractions(xs, x0, sigma)
            fy = est._pixel_fractions(ys, y0, sigma)
            dfx_dx, dfx_ds = est._pixel_fraction_derivatives(xs, x0, sigma)
            dfy_dy, dfy_ds = est._pixel_fraction_derivatives(ys, y0, sigma)

            mu = total_signal_e * np.outer(fy, fx) + floor_e
            var = mu + camera.read_noise_e**2 + quantization_var
            derivatives = np.stack([
                total_signal_e * np.outer(fy, dfx_dx),
                total_signal_e * np.outer(dfy_dy, fx),
                np.outer(fy, fx),
                total_signal_e * (np.outer(dfy_ds, fx) + np.outer(fy, dfx_ds)),
            ]).reshape(4, -1)
            information = (derivatives / var.ravel()) @ derivatives.T
            variances.append(np.linalg.inv(information)[0, 0])
    return float(np.sqrt(np.mean(variances)))
