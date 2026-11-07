"""Calibration of the predicted success probability (Fig. 6a): every data
transmission of the best-tuned periodic CV and ZOH, all realizations."""
from common import *  # noqa: F401,F403
import pickle
import analysis as A

BINS = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0001])


def job(seed, policy, T):
    r = one(Config(), seed, policy, T, record=True)
    return dict(seed=seed, policy=policy, phat=r["rec"]["phat"], ok=r["rec"]["ok"])


if __name__ == "__main__":
    res = A.load_main()
    Tz, _ = A.best(res, "zoh"); Tc, _ = A.best(res, "cv")
    out = pmap(job, [(s, p, T) for s in MAIN_SEEDS for p, T in (("zoh", Tz), ("cv", Tc))])
    tab = {}
    for pol in ("zoh", "cv"):
        ph = np.concatenate([o["phat"] for o in out if o["policy"] == pol])
        ok = np.concatenate([o["ok"] for o in out if o["policy"] == pol]).astype(float)
        b = np.clip(np.digitize(ph, BINS) - 1, 0, len(BINS) - 2)
        rows = []
        for i in range(len(BINS) - 1):
            m = b == i
            if m.sum():
                rows.append((ph[m].mean(), ok[m].mean(), int(m.sum())))
        tab[pol] = np.array(rows)
    pickle.dump(dict(tab=tab, Tz=Tz, Tc=Tc, bins=BINS), open(os.path.join(RES, "calib.pkl"), "wb"))
    print("done")
