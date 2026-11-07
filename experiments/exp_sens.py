"""Sensitivity to the look duration tau and the fleet size K (Fig. 5)."""
from common import *  # noqa: F401,F403
import pickle

TAUS = [4, 8, 12, 16, 24]
KS = [8, 12, 16, 20, 24]
TZ = [12, 16, 24, 32, 48, 64, 96, 128]
TC = [32, 64, 96, 128, 192, 256, 384, 512, 768]


def job(tau, K, seed, policy, T):
    cfg = with_(Config(), tau=tau, K=K)
    if T is not None and T <= tau:
        return None
    r = one(cfg, seed, policy, T)
    return dict(tau=tau, K=K, seed=seed, policy=policy, T=-1 if T is None else T,
                aoi=r["aoi"], succ=r["succ"], look=r["look"], Tmed=r["Tmed"],
                periods=r.get("periods"), per_aoi=r["per_aoi"], blk=r["blk"])


if __name__ == "__main__":
    get_q()
    for K in KS:
        prebuild(with_(Config(), K=K), SENS_SEEDS)
    jobs = []
    for s in SENS_SEEDS:
        for tau in TAUS:
            jobs += [(tau, 16, s, "zoh", T) for T in TZ]
            jobs += [(tau, 16, s, "cv", T) for T in TC]
            jobs += [(tau, 16, s, "loi", None)]
        for K in KS:
            if K == 16:
                continue          # identical to the tau = 12 runs above
            jobs += [(12, K, s, "zoh", T) for T in TZ]
            jobs += [(12, K, s, "cv", T) for T in TC]
            jobs += [(12, K, s, "loi", None)]
            jobs += [(12, K, s, "oracle", None)]
        jobs += [(12, 16, s, "oracle", None)]
    jobs.sort(key=lambda j: (j[3] != "loi", -j[1]))
    res = [r for r in pmap(job, jobs) if r is not None]
    pickle.dump(res, open(os.path.join(RES, "sens.pkl"), "wb"))
    print("done")
