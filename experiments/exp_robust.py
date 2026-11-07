"""Robustness of LOI to its two fixed hyper-parameters (pi_min, eps_B), which
were set a priori and never tuned (Table IV)."""
from common import *  # noqa: F401,F403
import pickle

VARIANTS = [dict(), dict(eps_B=0.0), dict(eps_B=0.05), dict(pi_min=0.1), dict(pi_min=0.4)]


def job(i, seed):
    cfg = with_(Config(), **VARIANTS[i])
    r = one(cfg, seed, "loi")
    return dict(var=i, seed=seed, aoi=r["aoi"], look=r["look"], Tmed=r["Tmed"])


if __name__ == "__main__":
    get_q()
    jobs = [(i, s) for i in range(len(VARIANTS)) for s in MAIN_SEEDS]
    res = pmap(job, jobs)
    pickle.dump(dict(variants=VARIANTS, res=res), open(os.path.join(RES, "robust.pkl"), "wb"))
    print("done")
