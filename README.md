# roughSynth

This repository is designed to generate rough, self-affine surfaces. While targeted at making surfaces which resemble planetary scale roughness (m to km scale), the functions present can be applied down to arbitrarily small scales. This package also comes with tools to evaluate the parameters of either synthetic or imported surfaces such as RMS height, RMS deviation (also known as Allan deviation), RMS slope and Hurst exponent.

## Installation

Run the command:

`pip install git+https://github.com/Nacnud04/roughSynth.git`

## Usage

Here is how to generate and export a surface. Note that the __dimensionless embedding parameter__ must be carefully chosen.

```python
from roughSynth import fractal

# Surface parameters
H = 0.3985               # Hurst exponent
R = 1.0                  # dimensionless embedding parameter
N = 410                  # samples/pixels per side
M = N

L_phys = 4100.0          # physical size of surface             [m]
dx_phys = L_phys / N     # physical sample spacing              [m]

L_rms = 10e3             # RMS height profile length            [m]
h_rms = 5.0              # desired RMS height at profile length [m]

# find the unit scale (1 meter) RMS height
eps_0 = h_rms * ((1 / L_rms) ** H)

# generate surface
# NOTE: because of how the surface is generated the function always produces
#       2 surfaces. we output both, just in case as one can be voided later.
# xs/ys describe the x/y grid as 1d arrays
f1, f2, xs, ys = fractal(N, M, R, H, L_phys, eps=h_rms, baseline=L_rms)

# plot
extent=[xs[0], xs[-1], ys[0], ys[-1]]

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

im1 = ax[0].imshow(f1, origin="lower", extent=extent, aspect="equal", cmap="bone")
fig.colorbar(im1, ax=ax[0], label="Height [m]")

im2 = ax[1].imshow(f2, origin="lower", extent=extent, aspect="equal", cmap="bone")
fig.colorbar(im2, ax=ax[1], label="Height [m]")

for a in ax: a.set_xlabel("x [m]")
for a in ax: a.set_ylabel("y [m]")

plt.suptitle(
    f"Synthetic self-affine surfaces: "
    f"H={H}, RMS={h_rms} m"
)

plt.tight_layout()

plt.show()
```


## Theory

### Dimensionless Embedding Parameter $R$
$R$ defines the valid domain of the generated surface. `fractal` automatically adjusts to the valid domain, but if the wrong $R$ is chosen artifacts may appear. Choose $R$ according to:
|   | $0 < 2H \leq 1.5$ | $1.5 < 2H < 2$ |
| - | :---------------: | :------------: |
| $R$ | 1 | 2 | 

### Surface statistics
#### **RMS Height**
RMS height is a function of the **profile length** it is evaluated over, as a longer profile results in a larger RMS height. Therefore it is **not** a function of lag or separation. It is computed as:


$$\varepsilon=\sqrt{\frac{1}{n-1}\sum_{i=1}^n\left(z(x_i)-\bar{z}\right)^2}$$


Then, for a fractal surface the RMS height scales with profile length following the Hurst exponent. 


$$\frac{\varepsilon(L)}{L^H}=\varepsilon_0$$


$\varepsilon_0$ describes the RMS height at unit length scale. In the above example this is computed, where the unit length scale is $1\;\mathrm{m}$.

The `fractal` function takes in both $\varepsilon$ and the profile length $L$ (through parameter `baseline`) to compute the RMS height at *pixel scale* which is not necessarily *unit scale*. This is necessary for the output of `fractal` to have the desired RMS height at some desired profile length. 

#### **RMS Deviation / Allan Deviation**
RMS deviation (also known as Allan deviation) is a function of **lag** (also refered to as separation or point-to-point distance). RMS deviation describes the RMS difference in height for points separated by some lag. This is computed as:


$$\nu(\Delta x)=\sqrt{\frac{1}{n}\sum_{i=1}^n\left(z(x_i)-z(x_i+\Delta x)\right)^2}$$


The RMS deviation's relationship with the Hurst exponent is similar to that for RMS Height:


$$\frac{\nu(\Delta x)}{L^H}=\nu_0$$


When controlling the output surface via $\nu$ at some $\Delta x$, the `fractal` function uses the above relationship to find the RMS deviation at *pixel scale* to scale the surface accordingly. 

#### RMS Slope
RMS slope is simply:


$$s_{rms}=\frac{\nu(\Delta x)}{\Delta x}$$


As the lag increases, $s_{rms}$ tends to decrease. This leads it to have the following relationship with the Hurst exponent:


$$\frac{s_{rms}(\Delta x)}{(\Delta x)^{H-1}}=s_{rms,0}$$


Where s_{rms,0} is the value of the RMS slope at 1 unit of lag.


## References

1. Schmidt, V. (2014). *Stochastic Geometry, Spatial Statistics and Random Fields*. Springer Lecture Notes in Mathematics - Models and Algorithms https://doi.org/10.1007/978-3-319-10064-7
2. Shepard, M. (2001). *The roughness of natural terrain: A planetary and remote sensing perspective.* Journal of Geophysical Research: Planets https://doi.org/10.1029%2F2000JE001429
