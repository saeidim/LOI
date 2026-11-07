"""Main study (Figs. 4, 6, 7, Table III): AoI versus look period for periodic
ZOH and CV, the adaptive LOI and LOI-R, the oracle, the uniform randomized
scheduler (Theorem 3(ii)) and the conformal coverage (Theorem 4)."""
from common import *  # noqa: F401,F403

RAND_T = [32, 64, 96, 128, 256, 512]
RAND_SEEDS = [1, 2, 3, 4]


def job(seed, policy, T, record):
    r = one(Config(), seed, policy, T, record)
    keep = dict(seed=seed, policy=policy, T=-1 if T is None else T, aoi=r["aoi"], succ=r["succ"],
                look=r["look"], tail=r["tail"], Tmed=r["Tmed"], per_aoi=r["per_aoi"],
                p_served=r.get("p_served"), served=r.get("served"), blk=r["blk"])
    if policy in ("loi", "loir"):
        keep["periods"] = r["periods"]
        keep["cpu_look"] = r["cpu_look"]
    if record:
        keep["aoi_t"] = r["aoi_t"]
    if record and "rec" in r:
        rec = r["rec"]
        keep["cov_z"] = np.abs(rec["err"]) / rec["sig"]
        keep["cov_n"] = rec["ncal"]
        keep["scores"] = r["scores"]
        keep["aoi_t"] = r["aoi_t"]
        keep["look_t"] = r["look_t"]
    return keep


if __name__ == "__main__":
    cfg = Config()
    get_q(cfg)
    prebuild(cfg, MAIN_SEEDS)
    jobs = []
    for s in MAIN_SEEDS:
        jobs += [(s, "zoh", T, False) for T in T_ZOH]
        jobs += [(s, "cv", T, False) for T in T_CV]
        jobs += [(s, "loi", None, True), (s, "loir", None, False), (s, "oracle", None, s == 1)]
    for s in RAND_SEEDS:
        jobs += [(s, "rand", T, False) for T in RAND_T]
    # longest jobs first for better load balance
    jobs.sort(key=lambda j: {"loir": 0, "loi": 1}.get(j[1], 2))
    res = pmap(job, jobs)
    import pickle
    pickle.dump(res, open(os.path.join(RES, "main.pkl"), "wb"))
    print("done")
