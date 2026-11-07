"""Write ../paper/numbers.tex: every number quoted in the paper is a macro
computed here from the raw results, so text, tables and figures agree."""
import collections
import json
import os
import pickle
import sys

import numpy as np

from common import RES, ROOT, Config, get_q, MAIN_SEEDS, SENS_SEEDS
import analysis as A

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "numbers.tex")
M = {}


def sci(v, d=1):
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.{d}f}\\times10^{{{e}}}"


def f(v, d=1):
    return f"{v:.{d}f}"


cfg = Config()
qx, qy = get_q()
M["QX"] = f(qx, 3); M["QY"] = f(qy, 3)
M["NSLOTS"] = str(cfg.n_slots); M["NSEEDS"] = str(len(MAIN_SEEDS)); M["NSENS"] = str(len(SENS_SEEDS))
M["TGRIDN"] = str(len(cfg.T_grid()))

# ---------------- theory ------------------------------------------------------
th = pickle.load(open(os.path.join(RES, "theory.pkl"), "rb"))
rows = []
for T, tau, r, phi, dp, mc, se in th["exact"]:
    rows.append(f"{T} & {tau} & {r:g} & {phi:.4f} & {dp:.4f} & {mc:.3f} $\\pm$ {se:.3f}\\\\")
M["EXACTROWS"] = "\n".join(rows)
M["EXACTMAXZ"] = f(max(abs(mc - phi) / se for T, tau, r, phi, dp, mc, se in th["exact"]), 1)
M["EXACTMAXDP"] = sci(max(abs(dp - phi) for T, tau, r, phi, dp, mc, se in th["exact"]) + 1e-16, 0)

a = th["ages"]
cv_e, cv_p, _ = th["growth"]["cv"]; z_e, z_p, _ = th["growth"]["zoh"]
M["GROWCVa"] = sci(cv_e[0]); M["GROWCVb"] = sci(cv_e[-1])
M["GROWZa"] = sci(z_e[0]); M["GROWZb"] = sci(z_e[-1], 2)
M["GROWAGE"] = str(int(a[-1]))
res = A.load_main()
loi_periods = np.concatenate([r["periods"] for r in res if r["policy"] == "loi"])
op = int(np.percentile(loi_periods, 95))
M["GROWOPAGE"] = str(op)
rel = cv_p / cv_e - 1
m = a <= op
M["GROWCVUNDER"] = f(max(0.0, -100 * rel[m].min()), 0)
M["GROWCVOVER"] = f(100 * rel[m].max(), 0)
M["GROWCVMAX"] = f(max(max(0.0, -100 * rel[m].min()), 100 * rel[m].max()), 0)
M["GROWLATE"] = f(max(0.0, -100 * rel[~m].min()), 0)
M["GROWLATEOVER"] = f(max(0.0, 100 * rel[~m].max()), 0)


def dbl(e):
    return float(np.interp(2 * e[0], e, a))


dz, dc = dbl(z_e), dbl(cv_e)
M["DOUBLEZ"] = f(dz, 0) if dz >= 2 else f(dz, 1)
M["DOUBLECV"] = f(dc, 0); M["DOUBLERATIO"] = f(dc / dz, 0)
bw = 1.0 / cfg.n_ant
M["CROSSZ"] = f(float(np.interp(bw, z_e, a)), 0)
M["CROSSCV"] = f(float(np.interp(bw, cv_e, a)), 0)
M["ZOHOVERCONF"] = f(z_e[-1] / z_p[-1], 0)

# ---------------- main study --------------------------------------------------
Tz, dz_ = A.best(res, "zoh")
Tc, dc_ = A.best(res, "cv")
Tsz, mz, dzz = A.curve(res, "zoh")
Tsc, mc, dcc = A.curve(res, "cv")
loi = A.table_by(res, "loi")[-1]; loir = A.table_by(res, "loir")[-1]
orc = A.table_by(res, "oracle")[-1]
M["BESTZT"] = str(Tz); M["BESTZ"] = f(A.agg(dz_, "aoi"))
M["BESTCVT"] = str(Tc); M["BESTCV"] = f(A.agg(dc_, "aoi"))
M["ZSUCCa"] = f(A.agg(dz_, "succ"), 3)
Tz4 = min([T for T in Tsz if T >= 4 * Tz], default=Tsz[-1])
M["ZSUCCTb"] = str(Tz4); M["ZSUCCb"] = f(A.agg(dzz[Tz4], "succ"), 3)
M["CVSUCCb"] = f(A.agg(dcc[Tc], "succ"), 3)
M["ZOVH"] = f(100 * A.agg(dz_, "look")); M["CVOVH"] = f(100 * A.agg(dc_, "look"))
mean, hw, _ = A.paired(dc_, dz_)
M["IMPROVEBEST"] = f(-mean); M["IMPROVEBESTLO"] = f(-mean - hw); M["IMPROVEBESTHI"] = f(-mean + hw)
M["LOIG"] = f(A.agg(loi, "aoi")); M["LOIR"] = f(A.agg(loir, "aoi")); M["ORACLE"] = f(A.agg(orc, "aoi"))
mean, hw, _ = A.paired(loi, dc_)
M["LOIVSCV"] = f(mean); M["LOIVSCVHW"] = f(hw)
M["LOIVSCVPHRASE"] = (f"{abs(mean):.1f}\\% below" if mean < 0 else f"{mean:.1f}\\% above")
mean, hw, xs = A.paired(loi, dz_)
M["LOIVSZ"] = f(-mean); M["LOIVSZLO"] = f(-mean - hw); M["LOIVSZHI"] = f(-mean + hw)
M["LOIWINSZ"] = str(int((xs < 0).sum()))
M["LOITMED"] = f(np.median([r["Tmed"] for r in loi.values()]), 0)
mm, hh, _ = A.paired(loir, loi)
M["LOIRVSLOI"] = f(mm); M["LOIRVSLOIHW"] = f(hh)
l1 = [r for r in res if r["policy"] == "loi" and r["seed"] == 1][0]
M["TLMIN"] = str(int(l1["periods"].min())); M["TLMAX"] = str(int(l1["periods"].max()))
M["TLRATIO"] = f(l1["periods"].max() / l1["periods"].min(), 0)
M["LOIWINSCV"] = str(int((A.paired(loi, dc_)[2] < 0).sum()))
from scipy.stats import spearmanr
mb = {s_: loi[s_]["blk"].mean() for s_ in MAIN_SEEDS}
low = [s_ for s_ in MAIN_SEEDS if mb[s_] < 0.02]
high = [s_ for s_ in MAIN_SEEDS if s_ not in low]
M["NLOWBLK"] = str(len(low))
M["LOITLOWBLKLO"] = f(min(loi[s_]["Tmed"] for s_ in low), 0)
M["LOITLOWBLKHI"] = f(max(loi[s_]["Tmed"] for s_ in low), 0)
M["LOITHIGHBLKMED"] = f(np.median([loi[s_]["Tmed"] for s_ in high]), 0)
rho, pv = spearmanr([loi[s_]["Tmed"] for s_ in MAIN_SEEDS], [mb[s_] for s_ in MAIN_SEEDS])
M["LOICORR"] = f(rho, 2); M["LOICORRP"] = f(pv, 3)
M["LOIRTMED"] = f(np.median([r["Tmed"] for r in loir.values()]), 0)
M["LOITLO"] = str(int(np.percentile(loi_periods, 5))); M["LOITHI"] = str(op)
M["LOIOVH"] = f(100 * A.agg(loi, "look"))
M["ZOVHRATIO"] = f(A.agg(dz_, "look") / A.agg(dc_, "look"), 1)
M["LOIOVHRATIO"] = f(A.agg(dz_, "look") / A.agg(loi, "look"), 1)
M["LOOKMS"] = f(1e3 * np.median([r["cpu_look"] for r in loi.values()]), 0)
# per-realization oracle of the period (best fixed CV period chosen per realization)
per_best = {s: min(dcc[T][s]["aoi"] for T in Tsc) for s in MAIN_SEEDS}
M["CVPERSEED"] = f(np.mean(list(per_best.values())))
M["LOIVSPERSEED"] = f(100 * np.mean([(loi[s]["aoi"] - per_best[s]) / per_best[s] for s in MAIN_SEEDS]))

rows = []
Tshort = max(T for T in Tsz if T < Tz)


def dcol(recs, ref, key):
    if recs is ref:
        return "0 (ref.)"
    mm, hh, _ = A.paired(recs, ref, key)
    return f"{mm:+.1f} $\\pm$ {hh:.1f}"


def row(name, recs, ref):
    return (f"{name} & {A.agg(recs, 'aoi'):.2f} & {dcol(recs, ref, 'aoi')} & "
            f"{A.agg(recs, 'aoi_clear'):.2f} & {dcol(recs, ref, 'aoi_clear')} & "
            f"{A.agg(recs, 'succ'):.3f} & {100 * A.agg(recs, 'look'):.1f}\\\\")


rows.append(row(f"ZOH, $T={Tshort}$", dzz[Tshort], dz_))
rows.append(row(f"ZOH$^\\star$, $T={Tz}$", dz_, dz_))
Tc_short = Tsc[0]
rows.append(row(f"CV, $T={Tz}$" if Tz in dcc else f"CV, $T={Tc_short}$", dcc.get(Tz, dcc[Tc_short]), dz_))
rows.append(row(f"CV$^\\star$, $T={Tc}$", dc_, dz_))
rows.append(row("LOI (adaptive)", loi, dz_))
rows.append(row("LOI-R (adaptive)", loir, dz_))
rows.append(row("Oracle", orc, dz_))
M["MAINROWS"] = "\n".join(rows)
# vehicles in (mostly) clear line of sight
allr = [r for r in res if r["policy"] == "oracle"]
M["NCLEAR"] = f(100 * np.mean([r["n_clear"] / len(r["blk"]) for r in allr]), 0)
M["CLEARDEF"] = f(100 * A.CLEAR, 0)
mm, hh, _ = A.paired(dc_, dz_, "aoi_clear")
M["CLEARIMP"] = f(-mm); M["CLEARIMPHW"] = f(hh)
mm, hh, _ = A.paired(loi, dz_, "aoi_clear")
M["LOICLEAR"] = f(-mm); M["LOICLEARHW"] = f(hh)
mm, hh, _ = A.paired(loi, dc_, "aoi_clear")
M["LOIVSCVCLEAR"] = f(-mm); M["LOIVSCVCLEARHW"] = f(hh)
zc, cc, lc, oc = (A.agg(x, "aoi_clear") for x in (dz_, dc_, loi, orc))
M["ZCLEAR"] = f(zc); M["CVCLEAR"] = f(cc); M["LOICLEARAOI"] = f(lc); M["ORCLEAR"] = f(oc)
M["GAPCLOSE"] = f(100 * (zc - lc) / (zc - oc), 0)
M["GAPCLOSECV"] = f(100 * (zc - cc) / (zc - oc), 0)
zs, cs, osd = (A.agg(x, "aoi_shad") for x in (dz_, dc_, orc))
M["ZSHAD"] = f(zs); M["CVSHAD"] = f(cs); M["ORSHAD"] = f(osd)
# CV at the ZOH-optimal period: imagination alone, same overhead
if Tz in dcc:
    mm, hh, _ = A.paired(dcc[Tz], dz_)
    M["SAMETGAIN"] = f(-mm)

# ---------------- randomized validation ---------------------------------------
x, y, p = A.rand_points(res)
hi = p >= 0.95
M["RANDNPTS"] = str(len(x)); M["RANDNHI"] = f(100 * hi.mean(), 0)
M["RANDMEDHI"] = f(100 * np.median(np.abs(y[hi] - x[hi]) / x[hi]), 1)
M["RANDOPT"] = f(100 * (y >= x).mean(), 0)

# ---------------- coverage ----------------------------------------------------
alphas = np.linspace(0.02, 0.5, 25)
cov, n = A.coverage(res, alphas)
M["COVSHORT"] = f(max(0.0, 100 * np.max((1 - alphas) - cov)), 1)
M["COVN"] = f"{n:,}".replace(",", "{,}")
M["COVTEN"] = f(100 * cov[np.argmin(np.abs(alphas - 0.1))], 1)

# ---------------- sensitivity -------------------------------------------------
sens_fn = os.path.join(RES, "sens.pkl")
if os.path.exists(sens_fn):
    sens = pickle.load(open(sens_fn, "rb"))
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in sens:
        d[(r["tau"], r["K"], r["policy"])][r["T"]][r["seed"]] = r

    def bestrec(tau, K, pol):
        dd = d[(tau, K, pol)]
        Ts = sorted(dd)
        mm = [np.mean([v["aoi"] for v in dd[T].values()]) for T in Ts]
        i = int(np.argmin(mm))
        return Ts[i], dd[Ts[i]]

    gains, tmeds, tcv, tz = [], [], [], []
    taus = [4, 8, 12, 16, 24]
    for tau in taus:
        Tzb, rz = bestrec(tau, 16, "zoh"); Tcb, rc = bestrec(tau, 16, "cv")
        gains.append(-A.paired(rc, rz)[0])
        tmeds.append(np.median([v["Tmed"] for v in d[(tau, 16, "loi")][-1].values()]))
        tcv.append(Tcb); tz.append(Tzb)
    M["TAUGAINa"] = f(gains[0]); M["TAUGAINb"] = f(gains[-1])
    M["TAUTABLE"] = ", ".join(f(g) for g in gains)
    M["TAUTa"] = f(tmeds[0], 0); M["TAUTb"] = f(tmeds[-1], 0)
    slope = np.polyfit(np.log(taus), np.log(tmeds), 1)[0]
    M["TAUEXP"] = f(slope, 2)
    M["PEFF"] = f(1.0 / slope - 1.0, 1)
    M["TAUCVEXP"] = f(np.polyfit(np.log(taus), np.log(tcv), 1)[0], 2)
    M["TAUZTa"] = str(tz[0]); M["TAUZTb"] = str(tz[-1])
    M["TAUCVTa"] = str(tcv[0]); M["TAUCVTb"] = str(tcv[-1])
    kg, kl = [], []
    for K in [8, 12, 16, 20, 24]:
        _, rz = bestrec(12, K, "zoh"); _, rc = bestrec(12, K, "cv")
        rl = d[(12, K, "loi")][-1]
        kg.append(-A.paired(rc, rz)[0]); kl.append(-A.paired(rl, rz)[0])
    M["KGAINmin"] = f(min(kg)); M["KGAINmax"] = f(max(kg))
    for r in sens:
        mm_ = r["blk"] <= A.CLEAR
        r["aoi_clear"] = r["per_aoi"][mm_].mean() if mm_.any() else np.nan

    def bestc(tau, K, pol):
        dd = d[(tau, K, pol)]
        Ts = sorted(dd)
        mm = [np.nanmean([v["aoi_clear"] for v in dd[T].values()]) for T in Ts]
        return dd[Ts[int(np.nanargmin(mm))]]
    for tag, (tau, K) in (("Ta", (4, 16)), ("Tb", (24, 16)), ("Ka", (8, 12)), ("Kb", (24, 12))):
        if tag.startswith("K"):
            tau, K = 12, K if False else (8 if tag == "Ka" else 24)
        rz = bestc(tau, K, "zoh"); rc = bestc(tau, K, "cv"); rl = d[(tau, K, "loi")][-1]
        M["CLCV" + tag] = f(-A.paired(rc, rz, "aoi_clear")[0])
        M["CLLOI" + tag] = f(-A.paired(rl, rz, "aoi_clear")[0])
    M["KLOImin"] = f(min(kl)); M["KLOImax"] = f(max(kl))

# ---------------- calibration --------------------------------------------------
cal_fn = os.path.join(RES, "calib.pkl")
if os.path.exists(cal_fn):
    cal = pickle.load(open(cal_fn, "rb"))
    tc, tz = cal["tab"]["cv"], cal["tab"]["zoh"]
    big = tc[:, 2] >= 50
    M["CALCV"] = f(100 * np.max(np.abs(tc[big, 1] - tc[big, 0])), 1)
    M["CALZTOP"] = f(100 * (tz[-2:, 1] * tz[-2:, 2]).sum() / tz[-2:, 2].sum(), 1)
    M["CALCVTOP"] = f(100 * (tc[-2:, 1] * tc[-2:, 2]).sum() / tc[-2:, 2].sum(), 1)
    zf = 100 - 100 * (tz[-2:, 1] * tz[-2:, 2]).sum() / tz[-2:, 2].sum()
    cf = 100 - 100 * (tc[-2:, 1] * tc[-2:, 2]).sum() / tc[-2:, 2].sum()
    M["CALZFAIL"] = f(zf, 1); M["CALCVFAIL"] = f(cf, 2); M["CALFAILRATIO"] = f(zf / cf, 0)
    M["CALCVCONS"] = "yes" if np.all(tc[:, 1] >= tc[:, 0] - 0.005) else "no"

# ---------------- robustness of LOI to its fixed parameters --------------------
rob_fn = os.path.join(RES, "robust.pkl")
if os.path.exists(rob_fn):
    rob = pickle.load(open(rob_fn, "rb"))
    by = collections.defaultdict(dict)
    for r in rob["res"]:
        by[r["var"]][r["seed"]] = r
    names = {0: "$\\pi_{\\min}=0.2$, $\\epsilon_B=0.02$ (default)", 1: "$\\epsilon_B=0$",
             2: "$\\epsilon_B=0.05$", 3: "$\\pi_{\\min}=0.1$", 4: "$\\pi_{\\min}=0.4$"}
    rows = []
    worst = 0.0
    for i in sorted(by):
        mm, hh, _ = A.paired(by[i], dz_)
        d0 = "0 (ref.)" if i == 0 else "{:+.1f} $\\pm$ {:.1f}".format(*A.paired(by[i], by[0])[:2])
        if i:
            worst = max(worst, abs(A.paired(by[i], by[0])[0]))
        rows.append(f"{names[i]} & {A.agg(by[i], 'aoi'):.2f} & {d0} & {-mm:.1f} & "
                    f"{100 * A.agg(by[i], 'look'):.1f} & {np.median([v['Tmed'] for v in by[i].values()]):.0f}\\\\")
    M["ROBUSTROWS"] = "\n".join(rows)
    M["ROBUSTWORST"] = f(worst)

with open(OUT, "w") as fh:
    fh.write("% generated by experiments/make_numbers.py -- do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(M, open(os.path.join(RES, "numbers.json"), "w"), indent=1)
print(json.dumps(M, indent=1))
