import numpy as np
from . import analysis as an

def rho(x, y, R, alpha):

    if alpha <= 1.5:
        beta = 0.0
        c2 = alpha / 2.0
        c0 = 1.0 - alpha / 2.0

    else:
        beta = alpha * (2.0 - alpha) / (
            3.0 * R * (R**2 - 1.0)
        )

        c2 = (
            alpha
            - beta * (R - 1.0)**2 * (R + 2.0)
        ) / 2.0

        c0 = beta * (R - 1.0)**3 + 1.0 - c2

    r = np.sqrt(
        (x[0] - y[0])**2 +
        (x[1] - y[1])**2
    )

    out = np.zeros_like(r)

    # case 1
    rSubOne = r <= 1.0
    out[rSubOne] = (c0 - r**alpha + c2 * r**2)[rSubOne]

    # case2
    rSubR = (r <= R) * (out == 0)
    out[rSubR] = beta * (R - r[rSubR])**3 / r[rSubR]

    return out, c0, c2


def fractal(N, M, R, H, Len, eps=None, nu_0=None, baseline=None):

    if R == 1:
        N *= 2
        M *= 2
        Len *= 2

    alpha = 2.0 * H

    # form grid
    tx = (np.arange(1, N+1) / N) * R
    ty = (np.arange(1, M+1) / M) * R
    ttx, tty = np.meshgrid(tx, ty)

    # construct block circulant covariance matrix
    Rows, _, _ = rho(
        [ttx, tty], [tx[0], ty[0]],
        R, alpha
    )

    # form circulant embedding
    BlkCirc_row = np.block([
        [
            Rows,
            Rows[:, -2:0:-1]
        ],
        [
            Rows[-2:0:-1, :],
            Rows[-2:0:-1, -2:0:-1]
        ]
    ])

    # extract eigenvalues of circulant covariance matrix
    lam_raw = (
        np.real(np.fft.fft2(BlkCirc_row))
        / (4.0 * (M - 1) * (N - 1))
    )

    # make all positive and sqrt
    lam = np.maximum(lam_raw, 0.0)
    lam = np.sqrt(lam)

    # generate gaussian field
    Z = (
        np.random.randn(2 * (M - 1), 2 * (N - 1))
        + 1j * np.random.randn(
            2 * (M - 1),
            2 * (N - 1)
        )
    )

    F = np.fft.fft2(lam * Z)

    # extract desired covariance block
    F = F[:M, :N]

    field1 = np.real(F)
    field2 = np.imag(F)

    # extract valid region
    # in reality this is circular, but we just extract it to be square
    keep_idx = tx <= R / 2.0
    field1 = field1[np.ix_(keep_idx, keep_idx)]
    field2 = field2[np.ix_(keep_idx, keep_idx)]
    x_sim = tx[keep_idx]
    y_sim = ty[keep_idx]

    dx_phys = Len / N

    # SCALE TO DESIRED eps_0
    if eps is not None and baseline is not None:
        # get 1 pixel baseline (not always 1 meter baseline)
        eps_px = eps * ((dx_phys / baseline) ** H)
        # Measure structure function at unit lag (1 pixel)
        lag_px = 1  # unit lag in pixels
        dx1 = field1[:, lag_px:] - field1[:, :-lag_px]
        dy1 = field1[lag_px:, :] - field1[:-lag_px, :]
        rms_current_f1 = np.sqrt(0.5 * (np.nanmean(dx1**2) + np.nanmean(dy1**2)))
        
        dx2 = field2[:, lag_px:] - field2[:, :-lag_px]
        dy2 = field2[lag_px:, :] - field2[:-lag_px, :]
        rms_current_f2 = np.sqrt(0.5 * (np.nanmean(dx2**2) + np.nanmean(dy2**2)))
        
        # Scale both fields
        scale_f1 = eps_px / rms_current_f1 if rms_current_f1 > 0 else 1.0
        scale_f2 = eps_px / rms_current_f2 if rms_current_f2 > 0 else 1.0
        
        field1 *= scale_f1
        field2 *= scale_f2

    # SCALE TO DESIRED nu_0
    if nu_0 is not None:

        # get RMS deviations at unit scale for both fields
        alx1, aly1 = an.get_allan(field1, 1)
        alx2, aly2 = an.get_allan(field2, 1)

        # scale both fields
        scale_f1 = nu_0 / np.mean([alx1, aly1])
        scale_f2 = nu_0 / np.mean([alx2, aly2])

        field1 *= scale_f1
        field2 *= scale_f2

    # map simulation coordinates to physical coordinates
    x_phys = x_sim * Len
    y_phys = y_sim * Len

    return field1, field2, x_phys, y_phys