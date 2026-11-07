"""Publication figures (EPS, Type-42 fonts, no transparency) for the paper.

Each panel is drawn at its printed size (3.4 in wide, 8 pt text) so that no
scaling happens in LaTeX. Colors are the Okabe-Ito colorblind-safe set; every
policy additionally has its own marker and line style, so identity never
relies on color alone (also in grayscale print).
"""
import os
import pickle
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter

from common import RES, ROOT, Config, one
import analysis as A

FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)

C = dict(zoh="#D55E00", cv="#0072B2", loi="#009E73", loir="#CC79A7",
         oracle="#000000", rand="#E69F00", grey="#6C757D")
MK = dict(zoh="s", cv="o", loi="D", loir="v", oracle="", rand="^")
LAB = dict(zoh="Periodic ZOH", cv="Periodic CV (imagination)", loi="LOI", loir="LOI-R",
           oracle="Oracle")

W1, H1 = 3.4, 2.35


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.3, "lines.markersize": 4, "legend.frameon": True,
        "legend.framealpha": 1.0, "legend.edgecolor": "0.8", "legend.fancybox": False,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5, "grid.linestyle": "-",
        "ps.fonttype": 42, "pdf.fonttype": 42, "savefig.dpi": 600,
        "axes.axisbelow": True,
    })


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".eps"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def logx_ticks(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
def fig_thm1(th):
    fig, ax = plt.subplots(figsize=(W1, H1))
    x = th["gx"]
    ls = {4: (0, (1, 1.2)), 16: (0, (4, 1.5)), 64: (0, (6, 1.5, 1, 1.5)), 256: "-"}
    cols = {4: "#E69F00", 16: "#009E73", 64: "#CC79A7", 256: "#0072B2"}
    for n, (g, b) in th["gcurves"].items():
        ax.plot(x, g, color=cols[n], lw=1.3, label=f"$N={n}$")
    # bound in N*eps coordinates is ~1-(pi^2/12)(N eps)^2 for all N >= 16
    n = 256
    ax.plot(x, th["gcurves"][n][1], color="k", lw=1.0, ls="--", label=r"bound $1-\kappa_N\epsilon^2$")
    ax.fill_between(x, -0.5, th["gcurves"][n][1], color="#F1F3F5", lw=0, zorder=0)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"normalized pointing error $N\epsilon$")
    ax.set_ylabel(r"normalized gain $g(\epsilon)$")
    ax.legend(loc="upper left", ncol=1, handlelength=2.2, borderpad=0.4, fontsize=6.8)
    ax.annotate("tight at\nalignment", xy=(0.15, 0.97), xytext=(1.25, 0.72),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3"), fontsize=7, color="0.2")
    save(fig, "fig_thm1")


def fig_thm2(th):
    fig, ax = plt.subplots(figsize=(W1, H1))
    a = th["ages"]
    cv_e, cv_p, _ = th["growth"]["cv"]
    z_e, z_p, _ = th["growth"]["zoh"]
    ax.loglog(a, cv_e, color=C["cv"], marker="o", ms=3, markevery=5, label="CV world model: true error")
    ax.loglog(a, cv_p, color=C["cv"], ls="--", lw=1.1, label="CV world model: belief, Theorem 2")
    ax.loglog(a, z_e, color=C["zoh"], marker="s", ms=3, markevery=5, label="ZOH: true error")
    ax.loglog(a, z_p, color=C["zoh"], ls="--", lw=1.1, label="ZOH: reported uncertainty")
    bw = 2.0 / Config().n_ant
    ax.axhline(bw / 2, color="0.35", lw=0.7, ls=":")
    ax.text(1500, bw / 2 * 1.2, r"$1/N\approx$ half 3-dB beamwidth", fontsize=6.8, color="0.3", ha="right")
    ax.set_xlabel("belief age $a$ [slots]")
    ax.set_ylabel("RMS error of $u$")
    ax.set_xlim(1, 1500); ax.set_ylim(2e-4, 1.5)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=2.2, borderpad=0.35, labelspacing=0.25)
    save(fig, "fig_thm2")


def fig_aoi_T(res):
    """AoI relative to the best-tuned ZOH of the same realization (common
    random numbers), mean and paired 95% confidence intervals."""
    from scipy import stats
    Tz, ref = A.best(res, "zoh")
    seeds = sorted(ref)
    rel = lambda d: np.array([100 * (d[s]["aoi"] / ref[s]["aoi"] - 1) for s in seeds])
    tq = stats.t.ppf(0.975, len(seeds) - 1)
    fig, ax = plt.subplots(figsize=(W1, H1 + 0.2))
    ax.axhline(0, color="0.5", lw=0.6)
    for pol in ("zoh", "cv"):
        Ts, m, d = A.curve(res, pol)
        mu, hw = [], []
        for T in Ts:
            x = rel(d[T]); mu.append(x.mean()); hw.append(tq * x.std(ddof=1) / np.sqrt(len(x)))
        mu = np.array(mu)
        ax.errorbar(Ts, mu, yerr=hw, color=C[pol], marker=MK[pol], ms=3.6, capsize=1.5,
                    elinewidth=0.6, label=LAB[pol], zorder=3)
        i = int(np.argmin(m))
        ax.plot(Ts[i], mu[i], marker="*", ms=10, color=C[pol], mec="k", mew=0.5, zorder=4)
    for pol, lst in (("loi", "-"), ("loir", (0, (5, 2))), ("oracle", (0, (1, 1.3)))):
        d = A.table_by(res, pol)[-1]
        x = rel(d); y = x.mean()
        ax.axhline(y, color=C[pol], ls=lst, lw=1.1, label=LAB[pol], zorder=2)
        if pol != "oracle":
            tm = np.median([r["Tmed"] for r in d.values()])
            ax.plot(tm, y, marker=MK[pol], color=C[pol], ms=5.5, mec="k", mew=0.4, zorder=5)
    logx_ticks(ax, [16, 32, 64, 128, 256, 512, 1024])
    ax.set_xlim(14, 1150)
    ax.set_ylim(-30, 52)
    ax.set_xlabel("look period $T$ [slots]")
    ax.set_ylabel(r"AoI relative to ZOH$^\star$ [%]")
    ax.legend(loc="upper right", ncol=2, fontsize=6.6, handlelength=2.0, columnspacing=0.9,
              borderpad=0.35, labelspacing=0.25)
    save(fig, "fig_aoi_T")


def fig_rand(res):
    x, y, p = A.rand_points(res)
    fig, ax = plt.subplots(figsize=(W1, H1 + 0.15))
    order = np.argsort(-p)
    sc = ax.scatter(x[order], y[order], c=p[order], cmap="viridis", vmin=0.5, vmax=1.0, s=9,
                    edgecolors="k", linewidths=0.2, zorder=3)
    lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25]
    ax.plot(lim, lim, color="k", lw=0.8, ls=":", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    tk = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000) if lim[0] <= t <= lim[1]]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(tk)); axis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"closed form $\Phi(T,\tau,\bar p_k/K)$ [slots]")
    ax.set_ylabel("simulated AoI of vehicle $k$ [slots]")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"measured delivery probability $\bar p_k$", fontsize=7)
    cb.ax.tick_params(labelsize=7, width=0.5)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig_rand_T")


def fig_sens(sens, key, xs, xlabel, name):
    """AoI relative to the best-tuned ZOH of the same realization (paired),
    for the fleet (solid) and for the vehicles in clear line of sight (dashed)."""
    import collections
    from scipy import stats
    for r in sens:
        m = r["blk"] <= A.CLEAR
        r["aoi_clear"] = r["per_aoi"][m].mean() if m.any() else np.nan
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in sens:
        d[(r[key], r["policy"])][r["T"]][r["seed"]] = r

    def best(dd, metric):
        Ts = sorted(dd)
        mm = [np.nanmean([v[metric] for v in dd[T].values()]) for T in Ts]
        return dd[Ts[int(np.nanargmin(mm))]]

    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.axhline(0, color=C["zoh"], lw=1.0, label="best-tuned ZOH (reference)")
    for metric, ls, suffix in (("aoi", "-", "all vehicles"), ("aoi_clear", "--", "line of sight")):
        for pol in ("cv", "loi", "oracle"):
            if pol == "oracle" and metric == "aoi_clear":
                continue
            ys, es = [], []
            for v in xs:
                ref = best(d[(v, "zoh")], metric)
                dd = d[(v, pol)] or d[(12, pol)]
                rec = best(dd, metric) if pol == "cv" else dd[-1]
                seeds = [s_ for s_ in sorted(ref) if s_ in rec and np.isfinite(rec[s_][metric])
                         and np.isfinite(ref[s_][metric])]
                x = np.array([100 * (rec[s_][metric] / ref[s_][metric] - 1) for s_ in seeds])
                ys.append(x.mean()); es.append(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
            lab = {"cv": "best-tuned CV", "loi": "LOI", "oracle": "Oracle"}[pol] + f", {suffix}"
            lso = (0, (1, 1.3)) if pol == "oracle" else ls
            ax.errorbar(np.array(xs) * (1.0 + (0.012 if ls == "--" else 0.0)), ys, yerr=es, color=C[pol],
                        marker=MK[pol] or None, ls=lso, ms=3.8, capsize=1.5, elinewidth=0.6,
                        mfc="white" if ls == "--" else C[pol], label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel(r"AoI relative to ZOH$^\star$ [%]")
    ax.set_xticks(xs)
    ax.set_ylim(-60, 25)
    ax.legend(loc="lower left", fontsize=6.3, ncol=2, borderpad=0.35, labelspacing=0.25,
              columnspacing=0.8, handlelength=2.0)
    save(fig, name)


def fig_coverage(res):
    alphas = np.linspace(0.02, 0.5, 25)
    cov, n = A.coverage(res, alphas)
    fig, ax = plt.subplots(figsize=(W1, H1))
    nom = 1 - alphas
    ax.fill_between(nom, nom, 1.0, color="#E7F5EF", lw=0, zorder=0)
    ax.plot(nom, nom, color="k", ls=":", lw=0.8, label="nominal $1-\\alpha$")
    ax.plot(nom, cov, color=C["loi"], marker="o", ms=3.2, label="empirical (true pointing error)")
    ax.set_xlabel(r"nominal coverage $1-\alpha$"); ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.48, 1.01)
    ax.text(0.62, 0.9, "certificate valid", fontsize=7, color="#1B6B4D")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig_coverage")
    return alphas, cov, n


def fig_calib():
    d = pickle.load(open(os.path.join(RES, "calib.pkl"), "rb"))
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot([0, 1], [0, 1], color="k", ls=":", lw=0.8, label="perfect calibration")
    for pol, lab in (("zoh", f"ZOH$^\\star$ ($T={d['Tz']}$)"), ("cv", f"CV$^\\star$ world model ($T={d['Tc']}$)")):
        t = d["tab"][pol]
        ax.plot(t[:, 0], t[:, 1], color=C[pol], lw=1.0, zorder=2)
        ax.scatter(t[:, 0], t[:, 1], s=6 + 60 * np.sqrt(t[:, 2] / t[:, 2].max()), color=C[pol],
                   marker=MK[pol], edgecolors="k", linewidths=0.3, zorder=3, label=lab)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"predicted success probability $\hat p_k$")
    ax.set_ylabel("realized success rate")
    ax.legend(loc="upper left", fontsize=6.8, markerscale=0.8)
    save(fig, "fig_calib")


def fig_timeline(res):
    """LOI in action on realization 1, versus the best-tuned periodic policies."""
    loi = [r for r in res if r["policy"] == "loi" and r["seed"] == 1][0]
    orc = [r for r in res if r["policy"] == "oracle" and r["seed"] == 1][0]
    Tz, _ = A.best(res, "zoh"); Tc, _ = A.best(res, "cv")
    rz = one(Config(), 1, "zoh", Tz, record=True)
    rc = one(Config(), 1, "cv", Tc, record=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.08))
    t_s = np.arange(len(loi["aoi_t"])) * 1e-3
    lt = loi["look_t"] * 1e-3
    a1.step(lt, loi["periods"], where="post", color=C["loi"], lw=1.1)
    a1.plot(lt, loi["periods"], "D", color=C["loi"], ms=2.6)
    a1.axhline(Tc, color=C["cv"], ls="--", lw=0.9)
    a1.axhline(Tz, color=C["zoh"], ls="--", lw=0.9)
    a1.text(0.08, Tc * 0.9, f"CV$^\\star$: $T={Tc}$", color=C["cv"], fontsize=6.8, ha="left", va="top")
    a1.text(0.08, Tz * 0.9, f"ZOH$^\\star$: $T={Tz}$", color=C["zoh"], fontsize=6.8, ha="left", va="top")
    a1.set_yscale("log"); a1.set_ylabel("LOI period $T$")
    a1.yaxis.set_major_locator(FixedLocator([16, 64, 256, 1024]))
    a1.yaxis.set_major_formatter(FixedFormatter(["16", "64", "256", "1024"]))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_ylim(12, 1500)

    def smooth(v, w=100):
        k = np.ones(w) / w
        return np.convolve(v, k, mode="same")
    for r, pol, lab in ((rz, "zoh", f"ZOH$^\\star$"), (rc, "cv", "CV$^\\star$"), (loi, "loi", "LOI"),
                        (orc, "oracle", "Oracle")):
        ls = {"oracle": (0, (1, 1.2))}.get(pol, "-")
        a2.plot(t_s, smooth(r["aoi_t"]), color=C[pol], lw=1.0 if pol != "loi" else 1.3, ls=ls,
                label=f"{lab} ({np.mean(r['aoi_t']):.1f})")
    a2.set_xlabel("time [s]"); a2.set_ylabel("fleet-average AoI [slots]")
    a2.set_xlim(0, 8)
    a2.legend(loc="upper left", ncol=2, fontsize=6.6, columnspacing=0.8, borderpad=0.35,
              labelspacing=0.25, handlelength=1.8)
    yl = a2.get_ylim(); a2.set_ylim(0, yl[1] * 1.3)
    fig.align_ylabels([a1, a2])
    save(fig, "fig_timeline")


def fig_system():
    src = os.path.join(ROOT, "figs_src")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig_system.tex"], cwd=src,
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["pdftops", "-eps", os.path.join(src, "fig_system.pdf"),
                    os.path.join(FIG, "fig_system.eps")], check=True)
    import shutil
    shutil.copy(os.path.join(src, "fig_system.pdf"), os.path.join(FIG, "fig_system.pdf"))


if __name__ == "__main__":
    style()
    th = pickle.load(open(os.path.join(RES, "theory.pkl"), "rb"))
    fig_thm1(th); fig_thm2(th)
    res = A.load_main()
    fig_aoi_T(res); fig_rand(res); fig_coverage(res); fig_timeline(res)
    sens_fn = os.path.join(RES, "sens.pkl")
    if os.path.exists(sens_fn):
        sens = pickle.load(open(sens_fn, "rb"))
        fig_sens([r for r in sens if r["K"] == 16], "tau", [4, 8, 12, 16, 24],
                 r"look duration $\tau$ [slots]", "fig_tau")
        fig_sens([r for r in sens if r["tau"] == 12], "K", [8, 12, 16, 20, 24],
                 "number of vehicles $K$", "fig_K")
    if os.path.exists(os.path.join(RES, "calib.pkl")):
        fig_calib()
    fig_system()
    print("figures written to", FIG)
