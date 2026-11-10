import numpy as np


def gaussian_surface(N, M, dx, dy, sigma=1.0, corr_length=10.0,
                     seed=None):
    """
    Generate a stationary Gaussian random surface using an
    analytical Gaussian PSD.

    Parameters
    ----------
    N, M : int
        Number of samples in x and y.
    dx, dy : float
        Physical sample spacing.
    sigma : float
        Desired RMS surface height.
    corr_length : float
        Gaussian correlation length.
    seed : int, optional
        Random seed.

    Returns
    -------
    field : ndarray
        Real-valued Gaussian random surface.
    psd : ndarray
        The discrete spectral density used to generate the field.
        This is in unshifted FFT ordering.
    cov : ndarray
        The covariance implied by the discrete PSD.
        This is in centered ordering.
    """

    rng = np.random.default_rng(seed)

    # Frequency axes
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dy)

    fxx, fyy = np.meshgrid(fx, fy)

    # Analytical Gaussian PSD
    psd = (
        2.0 * np.pi * sigma**2 * corr_length**2
        * np.exp(
            -2.0 * np.pi**2 * corr_length**2
            * (fxx**2 + fyy**2)
        )
    )

    # Convert physical PSD to discrete spectral eigenvalues
    lam = psd / (dx * dy)

    # Generate Gaussian white noise
    noise = rng.standard_normal((M, N))

    # Filter in Fourier domain
    field = np.real(
        np.fft.ifft2(
            np.sqrt(lam) * np.fft.fft2(noise)
        )
    )

    # Discrete theoretical covariance
    cov = np.fft.fftshift(
        np.real(np.fft.ifft2(lam))
    )

    return field, psd, cov


def gaussian_covariance(r, sigma, corr_length):

    return sigma**2 * np.exp(
        -r**2 / (2.0 * corr_length**2)
    )


def gaussian_psd(fx, fy, sigma, corr_length):

    return (
        2.0 * np.pi * sigma**2 * corr_length**2
        * np.exp(
            -2.0 * np.pi**2 * corr_length**2
            * (fx**2 + fy**2)
        )
    )