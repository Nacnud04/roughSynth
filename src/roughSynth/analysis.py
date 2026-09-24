import numpy as np
import matplotlib.pyplot as plt

def get_rms(field, lags, scl, verbose=False):

    rmss = []

    lag_ids = np.rint(lags / scl).astype(int)
    
    for lag in lag_ids:

        # check to make sure within reasonable domain
        if lag > np.min(field.shape) / 2 or lag == 0:
            if verbose:
                print(f"Lag length of {lag} is invalid, filling with nan")
            rmss.append(np.nan)
            continue
    
        dx = field[:, lag:] - field[:, :-lag]
        dy = field[lag:, :] - field[:-lag, :]
    
        rms = np.sqrt(
            0.5 *
            (
                np.nanmean(dx**2)
                +
                np.nanmean(dy**2)
            )
        )
    
        rmss.append(rms)

    return rmss


def get_allan(field, lag):

    diffx = field[:, :-lag] - field[:, lag:]
    diffy = field[:-lag, :] - field[lag:, :]

    allanx = np.sqrt(np.mean(diffx**2))
    allany = np.sqrt(np.mean(diffy**2))

    return allanx, allany


def get_allans(field, lags, scl, verbose=False):

    alxs, alys = [], []

    lag_ids = np.rint(lags / scl).astype(int)
    
    for lag in lag_ids:

        # check to make sure within reasonable domain
        if lag > np.min(field.shape) / 2 or lag == 0:
            if verbose:
                print(f"Lag length of {lag} is invalid, filling with nan")
            alxs.append(np.nan)
            alys.append(np.nan)

        else:
            alx, aly = get_allan(field, lag) 
            alxs.append(alx)
            alys.append(aly)

    return np.array(alxs), np.array(alys)


def get_rms_slope(field, lag, scl):

    alx, aly = get_allan(field, lag)

    s_x = alx / (lag * scl)
    s_y = aly / (lag * scl)

    return s_x, s_y


def get_rms_slopes(field, lags, scl, verbose=False):

    sxss, syss = [], []

    lag_ids = np.rint(lags / scl).astype(int)
    
    for lag in lag_ids:

        # check to make sure within reasonable domain
        if lag > np.min(field.shape) / 2 or lag == 0:
            if verbose:
                print(f"Lag length of {lag} is invalid, filling with nan")
            sxss.append(np.nan)
            syss.append(np.nan)

        else:
            sxs, sys = get_rms_slope(field, lag, scl) 
            sxss.append(sxs)
            syss.append(sys)

    return np.array(sxss), np.array(syss)


def est_H(lags, rmss):

    fit_min = 10      # metres
    fit_max = 200    # metres
    
    mask = (
        (lags >= fit_min)
        &
        (lags <= fit_max)
    )

    # mask away where nans are in rmss
    mask *= (np.isnan(np.array(rmss)) == 0)
    
    p = np.polyfit(
        np.log10(np.array(lags)[mask]),
        np.log10(np.array(rmss)[mask]),
        1
    )
    
    return p[0], p


def gen_surface_stats(fs, lag_start, lag_end, nlag, dx_phys):

    output = {}

    lags = np.unique(
        np.logspace(
            np.log10(lag_start),
            np.log10(lag_end),
            nlag
        ).astype(int)
    )
    output['lags'] = lags

    # first evaluate RMS heights over various profile lengths
    print(f"Evaluating RMS heights...")
    rmsss = []
    for fld in fs:
        rmsss.append(get_rms(fld, lags, dx_phys))

    ps = []
    Hs = []
    for rmss in rmsss:
        H, p = est_H(lags, rmss)
        ps.append(p)
        Hs.append(H)

    output['RMS_H'] = rmsss
    output['RMS_H-fits'] = ps

    # evaluate RMS deviation
    print(f"Evaluating RMS deviations...")
    allans = []
    for fld in fs:
        alx, aly = get_allans(fld, lags, dx_phys)
        allans.append((alx + aly) / 2)

    alPs = []
    for als in allans:
        H, p = est_H(lags, als)
        alPs.append(p)

    output['RMS_D'] = allans
    output['RMS_D-fits'] = alPs

    # evaluate RMS slopes
    print(f"Evaluating RMS slopes...")
    Srms_s = []
    for fld in fs:
        srmsx, srmsy = get_rms_slopes(fld, lags, dx_phys)
        Srms_s.append((srmsx+srmsy)/2)

    Srms_Ps = []
    for Srms in Srms_s:
        H, p = est_H(lags, Srms)
        Srms_Ps.append(p)

    output['RMS_S'] = Srms_s
    output['RMS_S-fits'] = Srms_Ps

    return output


def report_stats(output):

    minLag = np.min(output['lags'])
    maxLag = np.max(output['lags'])
    nLags = len(output['lags'])
    
    nSrfs = len(output['RMS_H'])

    # first print parameter summary
    print(f"\n==== FRACTAL SURFACE STATISTICS ====\n")
    print(f"# surfaces: {nSrfs}")
    print(f"# lags: {nLags}, from {minLag:.1f} to {maxLag:.1f} meters")
    print(f"\n")
    
    # now print RMS height statistics
    RMS_H_Pavg = np.mean(output['RMS_H-fits'], axis=0)
    RMS_H_Hstd = np.std(np.array(output['RMS_H-fits'])[:,0])
    RMS_H_Bstd = np.std(10**(np.array(output['RMS_H-fits'])[:,1])) 
    print(f"RMS Height for Varying Profile Length Yields:")
    print(f"H = {RMS_H_Pavg[0]:.5f} w/ std of {RMS_H_Hstd:.5f}")
    print(f"baseline RMS height = {10**RMS_H_Pavg[1]:.5f} w/ std of {RMS_H_Bstd:.5f}\n")

    # now print RMS deviation statistics
    RMS_D_Pavg = np.mean(output['RMS_D-fits'], axis=0)
    RMS_D_Hstd = np.std(np.array(output['RMS_D-fits'])[:,0])
    RMS_D_Bstd = np.std(10**(np.array(output['RMS_D-fits'])[:,1])) 
    print(f"RMS Deviation for Varying Profile Length Yields:")
    print(f"H = {RMS_D_Pavg[0]:.5f} w/ std of {RMS_D_Hstd:.5f}")
    print(f"baseline RMS deviation = {10**RMS_D_Pavg[1]:.5f} w/ std of {RMS_D_Bstd:.5f}\n")

    # now print RMS slope statistics
    RMS_S_Pavg = np.mean(output['RMS_S-fits'], axis=0)
    RMS_S_Hstd = np.std(np.array(output['RMS_S-fits'])[:,0])
    RMS_S_Bstd = np.std(10**(np.array(output['RMS_S-fits'])[:,1])) 
    print(f"RMS Slope for Varying Profile Length Yields:")
    print(f"H = {RMS_S_Pavg[0]+1:.5f} w/ std of {RMS_S_Hstd:.5f}")
    print(f"baseline RMS slope = {10**RMS_S_Pavg[1]:.5f} w/ std of {RMS_S_Bstd:.5f}\n")


def plot_surface_stats(output, fit_min=10, fit_max=200, n_hist_bins=10, savefig=None, H_targ=None, baseline_targ=(None, None, None)):
    lags = output["lags"]
    mask = (lags >= fit_min) & (lags <= fit_max)

    metrics = [
        ("RMS_H", "RMS Height Difference [m]", "H", r"$\varepsilon_0$ [m]"),
        ("RMS_D", "RMS Deviation [m]", "H", r"$\nu_0$ [m]"),
        ("RMS_S", "RMS Slope [.]", "H", r"$S_{rms_0}$ [.]"),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 14),
        gridspec_kw={"width_ratios": [2.4, 1, 1]},
    )

    for row, (key, ylabel, hlabel, blabel) in enumerate(metrics):
        series = output[key]
        fits = output[f"{key}-fits"]

        ax_main = axes[row, 0]
        ax_h = axes[row, 1]
        ax_b = axes[row, 2]

        # Plot all profiles
        for s in series:
            ax_main.scatter(lags, s, color="black", s=4, alpha=0.08)

        # Plot all fitted lines
        for p in fits:
            ax_main.plot(
                lags[mask],
                10 ** np.polyval(p, np.log10(lags[mask])),
                "r-",
                alpha=0.05
            )

        # Mean fit
        p_mean = np.mean(fits, axis=0)
        ax_main.plot(
            lags[mask],
            10 ** np.polyval(p_mean, np.log10(lags[mask])),
            "b-",
            lw=2,
            label="Mean fit"
        )

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlabel("Lag distance [m]")
        ax_h.set_xlabel(hlabel)
        ax_b.set_xlabel(blabel)
        ax_main.set_ylabel(ylabel)
        ax_main.legend()

        # H histogram
        Hs = np.array(fits)[:, 0]
        if key == "RMS_S":
            Hs_plot = Hs + 1
            ax_h.set_title("H")
        else:
            Hs_plot = Hs
            ax_h.set_title("H")

        ax_h.hist(Hs_plot, bins=n_hist_bins, color="black", alpha=0.8)
        if H_targ:
            ax_h.axvline(H_targ, color="red", lw=2, label="Target H")
        ax_h.set_ylabel("Count")

        # Baseline histogram
        baselines = 10 ** (np.array(fits)[:, 1])
        ax_b.hist(baselines, bins=n_hist_bins, color="gray", alpha=0.8)
        if baseline_targ[row]:
            ax_b.axvline(baseline_targ[row], color="red", lw=2, label="Target")
        ax_b.set_ylabel("Count")

    if H_targ:
        sptitle = f"Nsurfs: {len(fits)}  | H: {H_targ}  | "
        if baseline_targ[0]:
            sptitle += r"$\varepsilon_0$: " + f"{baseline_targ[0]:.5f}"
        plt.suptitle(sptitle, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
