# import
import roughSynth as rs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# === INPUT PARAMETERS ===

# surface parameters
H      = 0.2   # Hurst exponent
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

psd_ideals = []
cov_ideals = []
psd_empirs = []
cov_empirs = []
Dhs    = []
Chs    = []
RMS_sq = []
fss    = []

for i in tqdm(range(100)):

    # generate surfaces
    f1, f2, xs, ys = rs.fractal(N, N, R, H, L_phys, eps=h_rms, baseline=L_rms)
    fss.append(f1)
    fss.append(f2)

    for f in (f1, f2):

        # empirical PSD and covariance
        psd2d_empir = rs.empirical_psd2d(f, dx_phys, dx_phys)
        cov2d_empir = rs.autocovariance2d(f, unbiased=True)

        # ideal PSD and covariance
        psd2d_ideal = rs.ideal_psd_from_generator(N, N, R, H, L_phys)
        cov2d_ideal = rs.covariance_from_generator_psd(psd2d_ideal)

        # Match ideal variance level to realized variance so that
        # covariance residuals compare shape rather than amplitude.
        psd2d_ideal, cov2d_ideal = rs.match_ideal_to_empirical(
            psd2d_ideal, cov2d_ideal, psd2d_empir, cov2d_empir, f
        )

        # get the structure height function
        Dhs.append(
            rs.structure_height_function(f)
        )

        # get the surface covariance functino
        Chs.append(
            rs.covariance(f)
        )

        # get the RMS height squared
        # since the mean is removed this is equivalent to the standard deviation
        RMS_sq.append(np.std(f)**2)

        psd_ideals.append(psd2d_ideal)
        cov_ideals.append(cov2d_ideal)
        psd_empirs.append(psd2d_empir)
        cov_empirs.append(cov2d_empir)

psd_ideals  = np.stack(psd_ideals, axis=2)
cov_ideals  = np.stack(cov_ideals, axis=2)
psd_empirs  = np.stack(psd_empirs, axis=2)
cov_empirs  = np.stack(cov_empirs, axis=2)
Dhs         = np.stack(Dhs, axis=2)
Chs         = np.stack(Chs, axis=2)

psd_ideal_db = 10 * np.log10(np.mean(psd_ideals, axis=2))
cov2d_ideal  = np.mean(cov_ideals, axis=2)
psd_empir_db = 10 * np.log10(np.mean(psd_empirs, axis=2))
cov2d_empir  = np.mean(cov_empirs, axis=2)
Dhs          = np.mean(Dhs, axis=2)
Chs          = np.mean(Chs, axis=2)

mean_RMS_sq = np.mean(RMS_sq)



# === PLOT ===

plt.rcParams["text.usetex"] = True

# compare ideal vs empirical PSD and covariance
rs.ideal_vs_empirical(psd_ideal_db, cov2d_ideal, psd_empir_db, cov2d_empir, dx_phys, dx_phys, savefig="figures/PSD_Covariance.pdf")

# compare radially collapsed covariance
rho_ideal, cov1d_ideal, count_ideal = rs.radial_covariance(cov2d_ideal)
rho_empir, cov1d_empir, count_empir = rs.radial_covariance(cov2d_empir)
rho_empir_meas, cov1d_empir_meas, _ = rs.radial_covariance(Chs)   

offset = cov1d_empir[rho_empir * dx_phys < L_phys][-1]
cov1d_empir -= offset

scl = cov1d_empir[0] / cov1d_ideal[0]
cov1d_empir /= scl

# add the covariance to the plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(rho_ideal * dx_phys, cov1d_ideal, color="black", linewidth=1, label="Ideal Covariance")
ax[0].plot(rho_empir * dx_phys, cov1d_empir, color="magenta", linewidth=1, label=r"$C_h(\rho)$ from PSD")
ax[0].plot(rho_empir_meas * dx_phys, cov1d_empir_meas, color="red", linewidth=1, label=r"$C_h(\rho)=\langle h(r + \rho) h(r) \rangle$")
ax[0].set_xlabel(r"$\rho$")
ax[0].set_ylabel(r"$C_h(\rho)$")

# add the structure height function to the plot
rho_pix, Dhs_1d, _ = rs.radial_covariance(Dhs)
ax[1].plot(rho_pix * dx_phys, Dhs_1d, color="red", linewidth=1, label=r"$\langle [h(r + \rho) h(r)]^2 \rangle$")
ax[1].set_xlabel(r"$\rho$")
ax[1].set_ylabel(r"$D_h(\rho)$")

# estimate covariance from structure height function
cov_est = mean_RMS_sq - Dhs_1d / 2
ax[0].plot(rho_pix * dx_phys, cov_est, color="blue", linewidth=1, label=r"$C_h(\rho)$ from $D_h(\rho)$")

# estimate the structure height function from PSD derived covariance
Dhs_ideal = 2 * (mean_RMS_sq - cov1d_empir)
ax[1].plot(rho_ideal * dx_phys, Dhs_ideal, color="black", linewidth=1, label=r"$D_h(\rho)$ from $C_h(\rho)$")

for a in ax:
    a.set_xlim(0, np.sqrt(2) * (L_phys/2))
    a.axvline(L_phys/2, color="grey", linewidth=1, linestyle=":")
    a.grid()
    a.legend()

plt.savefig("figures/C_h.pdf")
plt.close()

# compare surface hurst exponents and rms heights
output_stats = rs.gen_surface_stats(fss, 50, 500, 20, dx_phys)
rs.plot_surface_stats(output_stats, fit_min=50, fit_max=500, H_targ=H, baseline_targ=[eps_0, None, None], savefig="figures/SurfaceFits.pdf")