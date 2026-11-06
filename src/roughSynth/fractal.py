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


def ideal_psd_from_generator(N, M, R, H, Len):
    
    """
    Compute the ideal power spectral density for a fractal surface
    """
    
    alpha = 2.0 * H

    tx = (np.arange(1, N+1) / N) * R
    ty = (np.arange(1, M+1) / M) * R
    ttx, tty = np.meshgrid(tx, ty)

    Rows, _, _ = rho([ttx, tty], [tx[0], ty[0]], R, alpha)

    BlkCirc_row = np.block([
        [Rows, Rows[:, -2:0:-1]],
        [Rows[-2:0:-1, :], Rows[-2:0:-1, -2:0:-1]]
    ])

    lam_raw = np.real(np.fft.fft2(BlkCirc_row)) / (4.0 * (M - 1) * (N - 1))
    psd2d   = np.maximum(lam_raw, 0.0)

    return psd2d


def empirical_psd2d(field, dx, dy, window=True, detrend_mean=True):

    """
    Compute the psd of a generated field
    """

    h = field.copy()

    if detrend_mean:
        h = h - np.mean(h)

    if window:
        wx = np.hanning(h.shape[1])
        wy = np.hanning(h.shape[0])
        W = np.outer(wy, wx)

        # power correction so total power is preserved on average
        U = np.mean(W**2)
        h = h * W
    else:
        U = 1.0

    H = np.fft.fft2(h)
    psd2d = (dx * dy) * (np.abs(H)**2) / (h.size * U)
    psd2d = np.fft.fftshift(psd2d)
    return psd2d


def covariance_from_psd2d(psd2d):

    """
    Convert from 2D power spectral density to surface covariance. 
    """

    return np.fft.fftshift(
        np.real(
            np.fft.ifft2(psd2d)
        )
    )

def covariance_from_empirical_psd2d(psd2d):

    """
    Convert from 2D power spectral density to surface covariance.
    Use when the power spectral density is computed empirically. 
    """

    return np.fft.fftshift(np.real(
        np.fft.ifft2(
            np.fft.ifftshift(psd2d)
        )
    ))


def autocovariance2d(field):
    r"""
    Compute properly normalized 2D autocovariance.

    Returns covariance in the same units as var(field).
    Therefore:

        autocovariance2d(field)[center]
            \approx np.var(field)
    """

    h = field - np.mean(field)

    cov = np.real(
        np.fft.fftshift(
            np.fft.ifft2(
                np.abs(np.fft.fft2(h))**2
            )
        )
    ) / h.size

    return cov


def covariance_from_generator_psd(psd2d):
    """
    Convert generator eigenvalue field into covariance.

    The generator PSD is NOT a physical PSD.
    It is the circulant embedding eigenvalue field.
    """

    return np.real(
        np.fft.fftshift(
            np.fft.ifft2(psd2d)
        )
    )


def freq_axes_2d(ny, nx, dy, dx):
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    return fx, fy


def spatial_axes_2d(ny, nx, dy, dx):
    y = (np.arange(ny) - ny // 2) * dy
    x = (np.arange(nx) - nx // 2) * dx
    return x, y


def center_crop(arr, target_shape):

    ny, nx = arr.shape
    ty, tx = target_shape

    y0 = (ny - ty) // 2
    x0 = (nx - tx) // 2

    return arr[
        y0:y0 + ty,
        x0:x0 + tx
    ]


def fractal(N, M, R, H, Len, eps=None, nu_0=None, baseline=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

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

    # this is the discrete spectral density / eigenvalue field
    # of the embedded circulant covariance matrix
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