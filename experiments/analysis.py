"""Aggregation of the raw results shared by make_figs.py and make_numbers.py."""
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy import stats

from common import RES, Config
from loi.aoi import Phi

POL_ORDER = ["zoh", "cv", "loi", "loir", "oracle"]


CLEAR = 0.05          # a vehicle is "clear" if obstructed (>3 dB) at most 5% of the time


def add_clear(res):
    for r in res:
        if "blk" in r and r.get("per_aoi") is not None:
            m = r["blk"] <= CLEAR
            r["aoi_clear"] = float(r["per_aoi"][m].mean()) if m.any() else np.nan
            r["aoi_shad"] = float(r["per_aoi"][~m].mean()) if (~m).any() else np.nan
            r["n_clear"] = int(m.sum())
    return res


def load_main():
    return add_clear(pickle.load(open(os.path.join(RES, "main.pkl"), "rb")))


def table_by(res, policy):
    """{T: {seed: record}} for one policy."""
    d = defaultdict(dict)
    for r in res:
        if r["policy"] == policy:
            d[r["T"]][r["seed"]] = r
    return d


def curve(res, policy):
    d = table_by(res, policy)
    Ts = sorted(d)
    m = np.array([np.mean([d[T][s]["aoi"] for s in d[T]]) for T in Ts])
    return np.array(Ts), m, d


def best(res, policy):
    Ts, m, d = curve(res, policy)
    i = int(np.argmin(m))
    return Ts[i], d[Ts[i]]


def paired(a, b, key="aoi"):
    """Mean and 95% t-interval half-width of the relative difference
    100 (a - b) / b over common seeds (dicts seed -> record)."""
    seeds = [s for s in sorted(set(a) & set(b)) if np.isfinite(a[s][key]) and np.isfinite(b[s][key])]
    x = np.array([100 * (a[s][key] - b[s][key]) / b[s][key] for s in seeds])
    hw = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), hw, x


def agg(recs, key):
    return float(np.nanmean([r[key] for r in recs.values()]))


def rand_points(res, cfg=None):
    cfg = cfg or Config()
    xs, ys, ps = [], [], []
    for r in res:
        if r["policy"] != "rand":
            continue
        pbar = r["p_served"]
        ok = (r["served"] > 0) & (pbar > 0)
        x = Phi(r["T"], cfg.tau, pbar[ok] / cfg.K)
        xs.append(x); ys.append(r["per_aoi"][ok]); ps.append(pbar[ok])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def coverage(res, alphas):
    """Empirical coverage of the conformal certificate on the true pointing
    error of every data transmission (LOI runs, all realizations)."""
    hit = np.zeros(len(alphas)); n = 0
    for r in res:
        if r["policy"] != "loi" or "cov_z" not in r:
            continue
        z, nc, sc = r["cov_z"], r["cov_n"], r["scores"]
        for ncal in np.unique(nc):
            m = nc == ncal
            if ncal < 20:
                continue
            ss = np.sort(sc[:ncal])
            for i, a in enumerate(alphas):
                j = int(np.ceil((ncal + 1) * (1 - a)))
                qhat = ss[j - 1] if j <= ncal else np.inf
                hit[i] += (z[m] <= qhat).sum()
            n += m.sum()
    return hit / n, n
