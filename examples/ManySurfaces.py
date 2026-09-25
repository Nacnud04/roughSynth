# import
from roughSynth import *

# === INPUT PARAMETERS ===

# surface parameters
H      = 0.3985   # Hurst exponent
N      = 410      # samples per side
L_phys = 4100     # physical surface size [m]
L_rms  = 10e3     # RMS height profile length
h_rms  = 13       # RMS height
Niters = 50       # number of iterations

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

fs = []
for i in range(Niters):
    print(f"Generating surface {i+1}/{Niters}", end="      \r")
    f1, f2, xs, ys = fractal(N, N, R, H, L_phys, eps=h_rms, baseline=L_rms)
    fs.append(f1)
    fs.append(f2)


# === PRODUCE SURFACE STATISTICS ===

# first evaluate limits on which to evaluate the surface
fitmin = dx_phys * 10
fitmax = L_phys  * 0.25
fitpts = 50

# generate statistics
stats = gen_surface_stats(fs, fitmin, fitmax, fitpts, dx_phys)
report_stats(stats)

# export as a png
figfile = "SurfaceStats.png"
print(f"Exporting to {figfile}")
plot_surface_stats(stats, savefig=figfile, H_targ=H, baseline_targ=[eps_0, None, None], fit_min=fitmin, fit_max=fitmax)
