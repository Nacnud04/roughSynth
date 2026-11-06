# import
import roughSynth as rs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === INPUT PARAMETERS ===

# surface parameters
H      = 0.3985   # Hurst exponent
N      = 410      # samples per side
L_phys = 4100     # physical surface size [m]
L_rms  = 10e3     # RMS height profile length
h_rms  = 13       # RMS height

# find R
if 2 * H <= 1.5:
    R = 1.0
else:
    R = 2.0

# get sample spacing
dx_phys = L_phys / N

# compute baseline RMS height
eps_0 = h_rms * ((1 / L_rms) ** H)


# === GENERATE SURFACES ===

psd_ideals_db = []
cov_ideals = []
psd_empirs_db = []
cov_empirs = []

for i in tqdm(range(50)):

    # generate surfaces
    f1, f2, xs, ys = rs.fractal(N, N, R, H, L_phys, eps=h_rms, baseline=L_rms)

    # empirical PSD and covariance
    psd2d_empir = rs.empirical_psd2d(f1, dx_phys, dx_phys, window=True)
    cov2d_empir = rs.autocovariance2d(f1)

    # ideal PSD and covariance
    psd2d_ideal = rs.ideal_psd_from_generator(N, N, R, H, L_phys)
    cov2d_ideal = rs.covariance_from_generator_psd(psd2d_ideal)

    # Match ideal variance level to realized variance so that
    # covariance residuals compare shape rather than amplitude.

    cy_i = cov2d_ideal.shape[0] // 2
    cx_i = cov2d_ideal.shape[1] // 2

    target_variance = np.var(f1)

    scale_factor = (
        target_variance
        / cov2d_ideal[cy_i, cx_i]
    )

    cov2d_ideal *= scale_factor
    psd2d_ideal *= scale_factor


    psd2d_ideal = rs.center_crop(
        np.fft.fftshift(psd2d_ideal),
        psd2d_empir.shape
    )

    cov2d_ideal = rs.center_crop(
        cov2d_ideal,
        cov2d_empir.shape
    )


    # Match total power between ideal and realization

    power_scale = (
        np.sum(psd2d_empir)
        / np.sum(psd2d_ideal)
    )

    psd2d_ideal *= power_scale


    # frequency and spatial axes
    fx, fy = rs.freq_axes_2d(psd2d_empir.shape[0], psd2d_empir.shape[1], dx_phys, dx_phys)
    xlag, ylag = rs.spatial_axes_2d(cov2d_empir.shape[0], cov2d_empir.shape[1], dx_phys, dx_phys)

    # dB conversion
    psd_ideal_db = 10 * np.log10(
        np.maximum(psd2d_ideal, 1e-30)
    )

    psd_empir_db = 10 * np.log10(np.maximum(psd2d_empir, 1e-30))

    psd_ideals_db.append(psd_ideal_db)
    cov_ideals.append(cov2d_ideal)
    psd_empirs_db.append(psd_empir_db)
    cov_empirs.append(cov2d_empir)

psd_ideals_db  = np.stack(psd_ideals_db, axis=2)
cov_ideals     = np.stack(cov_ideals, axis=2)
psd_empirs_db  = np.stack(psd_empirs_db, axis=2)
cov_empirs     = np.stack(cov_empirs, axis=2)

psd_ideal_db = np.mean(psd_ideals_db, axis=2)
cov2d_ideal  = np.mean(cov_ideals, axis=2)
psd_empir_db = np.mean(psd_empirs_db, axis=2)
cov2d_empir  = np.mean(cov_empirs, axis=2)


# === PLOT ===

fig, ax = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)

im0 = ax[0,0].imshow(
    psd_ideal_db,
    origin='lower',
    extent=[fx[0], fx[-1], fy[0], fy[-1]],
    aspect='auto'
)
ax[0,0].set_title("Ideal PSD")
ax[0,0].set_xlabel("fx [cycles/m]")
ax[0,0].set_ylabel("fy [cycles/m]")
fig.colorbar(im0, ax=ax[0,0])

ext_lag = extent=[
    xlag[0], xlag[-1],
    ylag[0], ylag[-1]
]
im1 = ax[0,1].imshow(
    cov2d_ideal,
    origin='lower',
    extent=ext_lag,
    aspect='auto'
)
ax[0,1].set_title("Ideal covariance")
ax[0,1].set_xlabel("x lag [m]")
ax[0,1].set_ylabel("y lag [m]")
fig.colorbar(im1, ax=ax[0,1])

im2 = ax[1,0].imshow(
    psd_empir_db,
    origin='lower',
    extent=[fx[0], fx[-1], fy[0], fy[-1]],
    aspect='auto'
)
ax[1,0].set_title("Empirical PSD")
ax[1,0].set_xlabel("fx [cycles/m]")
ax[1,0].set_ylabel("fy [cycles/m]")
fig.colorbar(im2, ax=ax[1,0])

im3 = ax[1,1].imshow(
    cov2d_empir,
    origin='lower',
    extent=ext_lag,
    aspect='auto'
)
ax[1,1].set_title("Empirical covariance")
ax[1,1].set_xlabel("x lag [m]")
ax[1,1].set_ylabel("y lag [m]")
fig.colorbar(im3, ax=ax[1,1])

psd_residual = psd_empir_db - psd_ideal_db
max_residual = np.max(np.abs(psd_residual))
im4 = ax[2,0].imshow(
    psd_residual,
    origin='lower',
    extent=[fx[0], fx[-1], fy[0], fy[-1]],
    aspect='auto', cmap="coolwarm", 
    vmin=-1 * max_residual, vmax=max_residual
)
ax[2,0].set_title("Residual PSD (Empirical - Ideal)")
ax[2,0].set_xlabel("fx [cycles/m]")
ax[2,0].set_ylabel("fy [cycles/m]")
fig.colorbar(im4, ax=ax[2,0])

cov_residual = cov2d_empir - cov2d_ideal
max_residual = np.max(np.abs(cov_residual))
im4 = ax[2,1].imshow(
    cov_residual,
    origin='lower',
    extent=ext_lag,
    aspect='auto', cmap="coolwarm",
    vmin=-1 * max_residual, vmax=max_residual
)
ax[2,1].set_title("Residual Covariance (Empirical - Ideal)")
ax[2,1].set_xlabel("fx [cycles/m]")
ax[2,1].set_ylabel("fy [cycles/m]")
fig.colorbar(im4, ax=ax[2,1])

plt.show()
