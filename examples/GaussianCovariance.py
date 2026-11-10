import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import roughSynth as rs


# --- PARAMETERS --- 

N = 410
M = 410

L_phys = 4100.0

dx = L_phys / N
dy = L_phys / M

sigma = 13.0
corr_length = 50.0

n_surfaces = 250


# --- GENERATE SURFACES ---

fields = []

psd_empirs = []
cov_empirs = []
dhs_empirs = []

for i in tqdm(range(n_surfaces)):

    field, psd_theory, cov_theory = rs.gaussian_surface(
        N, M,
        dx, dy,
        sigma=sigma,
        corr_length=corr_length
    )

    fields.append(field)

    # empirical PSD (no window)
    psd_empir = rs.empirical_psd2d(
        field, dx, dy,
        window=False,
        detrend_mean=True
    )

    # empirical covariance
    cov_empir = rs.autocovariance2d(
        field,
        unbiased=False
    )

    # structure height function
    dhs_empirs.append(
        rs.structure_height_function(field)
    )

    psd_empirs.append(psd_empir)
    cov_empirs.append(cov_empir)


# --- ENSEMBLE AVERAGING --- 

psd_empirs = np.stack(psd_empirs, axis=2)
cov_empirs = np.stack(cov_empirs, axis=2)
dhs_empirs = np.stack(dhs_empirs, axis=2)

psd_empir = np.mean(psd_empirs, axis=2)
cov_empir = np.mean(cov_empirs, axis=2)
dhs_empir = np.mean(dhs_empirs, axis=2)


# --- THEORETICAL RESULTS ---

# recalculate theoretical PSD in centered ordering
psd_theory = np.fft.fftshift(psd_theory)

# covariance is already centered
cov_theory = cov_theory


# --- ANALYTICAL COVARIANCE --- 

ny, nx = cov_empir.shape

x = (np.arange(nx) - nx // 2) * dx
y = (np.arange(ny) - ny // 2) * dy

xx, yy = np.meshgrid(x, y)

r = np.sqrt(xx**2 + yy**2)

cov_analytic = rs.gaussian_covariance(
    r,
    sigma,
    corr_length
)


# --- MAKE RADIAL ---

rho_theory, cov1d_theory, _ = rs.radial_covariance(cov_theory)

rho_empir, cov1d_empir, _ = rs.radial_covariance(cov_empir)
_, dh1d_empir, _ = rs.radial_covariance(dhs_empir)

rho_analytic, cov1d_analytic, _ = rs.radial_covariance(cov_analytic)



# --- PLOT ---

plt.rcParams["text.usetex"] = True


fig, ax = plt.subplots(2, 1, figsize=(6, 10))

ax[0].plot(
    rho_theory * dx,
    cov1d_theory,
    color="black",
    label="Ideal"
)

ax[0].plot(
    rho_empir * dx,
    cov1d_empir,
    color="red",
    label="Empirical"
)

ax[0].set_ylabel(r"$C_h(r)$")

# compute ideal structure height function
dh1d_ideal = 2 * (cov1d_theory[0] - cov1d_theory)
ax[1].plot(rho_empir * dx, dh1d_ideal, color="black", label=r"Ideal")
ax[1].plot(rho_empir * dx, dh1d_empir, color="red", label=r"Empirical")
ax[1].set_ylabel(r"$D_h(\rho)$")

for a in ax:
    a.grid()
    a.legend()
    a.set_xlabel(r"$\rho$ [m]")

plt.tight_layout()
plt.show()