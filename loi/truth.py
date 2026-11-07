"""Ground truth of one traffic realization (common random numbers).

Everything that does not depend on the policy is drawn here once and cached:
trajectories, shadowing, blockage, per-slot wideband success thresholds and
the noise of every possible sweep measurement. All policies evaluated on the
same seed therefore face identical traffic and channels.
"""
import os
import numpy as np
from .traffic import simulate_traffic
from . import channel as ch

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
N_INIT = 300          # slots reserved for the initialisation looks
N_EXTRA = 2000        # slots appended for open-loop experiments


def build_truth(cfg, seed, n_slots=None, extra=N_EXTRA):
    n_slots = cfg.n_slots if n_slots is None else n_slots
    n_total = N_INIT + n_slots + extra
    dur = cfg.warmup_s + n_total * cfg.slot_s
    tr = simulate_traffic(cfg, seed, dur + 0.01, cfg.warmup_s)
    for k in ("x", "y", "vx", "vy", "odo"):
        tr[k] = tr[k][:n_total]
    rng = np.random.default_rng([seed, 7])
    K = cfg.K
    # shadowing: Gudmundson OU in travelled distance
    sh = np.empty((n_total, K))
    s = cfg.sigma_sf_db * rng.standard_normal(K)
    dodo = np.diff(tr["odo"], axis=0, prepend=tr["odo"][:1])
    eps = rng.standard_normal((n_total, K))
    for t in range(n_total):
        r = np.exp(-dodo[t] / cfg.d_corr_m)
        s = r * s + cfg.sigma_sf_db * np.sqrt(1 - r * r) * eps[t]
        sh[t] = s
    # blockage (chunked)
    knife = np.empty((n_total, K))
    for i in range(0, n_total, 500):
        knife[i:i + 500] = ch.blockage_db(tr["x"][i:i + 500], tr["y"][i:i + 500],
                                          tr["hant"], tr["L"], tr["W"], tr["H"], cfg)
    snr0 = ch.snr0_db(tr["x"], tr["y"], tr["hant"][None], cfg)
    snr = snr0 - knife + sh
    # per-slot wideband success thresholds (Rician unless obstructed > 3 dB)
    rng_f = np.random.default_rng([seed, 11])
    thr = np.empty((n_total, K))
    for i in range(0, n_total, 1000):
        sl = slice(i, min(i + 1000, n_total))
        n = sl.stop - sl.start
        p_r = ch.fading_power(rng_f, (n, K, cfg.n_sub), cfg.k_rice_db)
        p_y = ch.fading_power(rng_f, (n, K, cfg.n_sub), None)
        t_r = ch.snr_threshold_db(p_r, cfg.r_req, 30)
        t_y = ch.snr_threshold_db(p_y, cfg.r_req, 30)
        thr[sl] = np.where(knife[sl] > 3.0, t_y, t_r)
    meas = np.random.default_rng([seed, 13]).standard_normal((n_total, K, 3))
    return dict(tr, sh=sh, knife=knife, snr=snr, thr=thr, meas=meas,
                n_init=N_INIT, n_slots=n_slots)


def get_truth(cfg, seed, tag=""):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"truth_K{cfg.K}_s{seed}{tag}.npz")
    if os.path.exists(fn):
        d = dict(np.load(fn))
        d["n_init"] = int(d["n_init"]); d["n_slots"] = int(d["n_slots"])
        return d
    d = build_truth(cfg, seed)
    tmp = fn + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, fn)              # atomic: safe under parallel workers
    return d
