"""Theory validation: Theorem 1 (gain bound), Theorem 2 (imagination growth
law, open loop in the full traffic simulator) and Theorem 3(i) (exactness of
the closed form against dynamic programming and Monte Carlo)."""
from common import *  # noqa: F401,F403
import pickle
from loi import channel as ch
from loi.aoi import Phi, Phi_dp, Phi_mc
from loi.worldmodel import WorldModel

EXACT_CASES = [(10, 0, 0.3), (10, 3, 0.3), (40, 12, 0.05), (200, 12, 0.055),
               (17, 16, 0.4), (96, 12, 0.06), (400, 24, 0.02)]
GROWTH_SEEDS = list(range(1, 9))
MAX_AGE = 1500
LOOK_GAP = 100          # sweep period before each open-loop start
START_EVERY = 3         # every third sweep starts an open-loop imagination


def exact_table():
    rows = []
    for i, (T, tau, r) in enumerate(EXACT_CASES):
        m, se = Phi_mc(T, tau, r, n_slots=2_000_000, seed=i)
        rows.append((T, tau, r, float(Phi(T, tau, r)), Phi_dp(T, tau, r), m, se))
    return rows


def growth(seed):
    cfg = Config()
    tr = get_truth(cfg, seed)
    q = get_q()
    ages = np.unique(np.round(np.geomspace(1, MAX_AGE, 60)).astype(int))
    out = {}
    for name, motion in (("cv", True), ("zoh", False)):
        wm = WorldModel(cfg, tr, q, table(cfg), motion=motion)
        err2 = np.zeros(len(ages)); pred2 = np.zeros(len(ages)); cnt = np.zeros(len(ages))
        n_tot = tr["x"].shape[0]
        i = 0
        for s0 in range(0, n_tot - MAX_AGE - 1, LOOK_GAP):
            wm.look(s0)
            i += 1
            if i < 8 or i % START_EVERY:
                continue
            uh, su2 = wm.beam(ages)
            sl = s0 + ages
            ut, _, _ = ch.dcos(tr["x"][sl], tr["y"][sl], tr["hant"][None], cfg)
            crit = np.abs(tr["x"][sl]) < 60.0
            e2 = (ut - uh) ** 2
            err2 += np.where(crit, e2, 0).sum(1)
            pred2 += np.where(crit, su2, 0).sum(1)
            cnt += crit.sum(1)
        out[name] = (err2, pred2, cnt)
    return ages, out


def gain_curves():
    x = np.linspace(-3, 3, 1201)          # N * eps
    curves = {}
    for n in (4, 16, 64, 256):
        eps = x / n
        curves[n] = (ch.ula_gain(eps, n), 1 - ch.kappa(n) * eps ** 2)
    return x, curves


if __name__ == "__main__":
    get_q()
    print("exact table"); rows = exact_table()
    print("growth law")
    prebuild(Config(), GROWTH_SEEDS)
    g = pmap(growth, [(s,) for s in GROWTH_SEEDS])
    ages = g[0][0]
    agg = {}
    for name in ("cv", "zoh"):
        e = sum(r[1][name][0] for r in g); p = sum(r[1][name][1] for r in g)
        c = sum(r[1][name][2] for r in g)
        agg[name] = (np.sqrt(e / c), np.sqrt(p / c), c)
    x, curves = gain_curves()
    pickle.dump(dict(exact=rows, ages=ages, growth=agg, gx=x, gcurves=curves),
                open(os.path.join(RES, "theory.pkl"), "wb"))
    print("done")
