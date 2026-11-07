"""Re-runs only the adaptive policies (LOI, LOI-R) of the main and the
sensitivity studies and replaces their entries in main.pkl / sens.pkl.
Useful after a change that affects only the period selection; the periodic
and oracle runs do not depend on it."""
from common import *  # noqa: F401,F403
import pickle
import exp_main
import exp_sens

if __name__ == "__main__":
    fn = os.path.join(RES, "main.pkl")
    res = pickle.load(open(fn, "rb"))
    jobs = [(s, p, None, p == "loi") for s in MAIN_SEEDS for p in ("loi", "loir")]
    new = pmap(exp_main.job, jobs)
    res = [r for r in res if r["policy"] not in ("loi", "loir")] + new
    pickle.dump(res, open(fn, "wb"))
    fn = os.path.join(RES, "sens.pkl")
    if os.path.exists(fn):
        sens = pickle.load(open(fn, "rb"))
        keys = sorted({(r["tau"], r["K"], r["seed"]) for r in sens if r["policy"] == "loi"})
        new = pmap(exp_sens.job, [(tau, K, s, "loi", None) for tau, K, s in keys])
        sens = [r for r in sens if r["policy"] != "loi"] + [r for r in new if r is not None]
        pickle.dump(sens, open(fn, "wb"))
    print("done")
