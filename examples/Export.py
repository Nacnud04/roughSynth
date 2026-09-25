# import
import numpy as np
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


# === GENERATE SURFACE ===

f1, _, xs, ys = fractal(N, N, R, H, L_phys, eps=h_rms, baseline=L_rms)

xxs, yys = np.meshgrid(xs-L_phys/2, ys-L_phys/2)
export_grid_to_facet(xxs, yys, f1, "Export.fct", obj_filename="Export.obj")