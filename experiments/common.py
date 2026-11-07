"""Shared helpers for the experiment scripts."""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from loi.config import Config, with_          # noqa: E402
from loi.channel import SuccessTable          # noqa: E402
from loi.truth import get_truth, build_truth  # noqa: E402
from loi.worldmodel import learn_q            # noqa: E402
from loi.sim import run                        # noqa: E402

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
N_JOBS = int(os.environ.get("LOI_JOBS", os.cpu_count() or 1))

MAIN_SEEDS = list(range(1, 17))      # 16 traffic realizations, main study
SENS_SEEDS = list(range(1, 7))       # 6 realizations, sensitivity studies
T_ZOH = [16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 192]
T_CV = [32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024]

_TAB = {}


def table(cfg):
    key = (cfg.n_sub, cfg.k_rice_db, cfg.r_req)
    if key not in _TAB:
        _TAB[key] = SuccessTable(cfg)
    return _TAB[key]


def get_q(cfg=None):
    """World-model intensities learned from a separate 30 s training
    trajectory observed only through sweeps (cached)."""
    fn = os.path.join(RES, "q.json")
    if os.path.exists(fn):
        d = json.load(open(fn))
        return d["qx"], d["qy"]
    cfg = cfg or Config()
    trn = build_truth(cfg, 1000, n_slots=30000, extra=0)
    qx, qy = learn_q(cfg, trn, every=50)
    json.dump(dict(qx=qx, qy=qy, train_seed=1000, train_s=30, sweep_every=50), open(fn, "w"))
    return qx, qy


def one(cfg, seed, policy, T=None, record=False):
    tr = get_truth(cfg, seed)
    r = run(tr, cfg, policy, table(cfg), get_q(), T=T, record=record, seed=seed)
    r.update(seed=seed, policy=policy, T=T)
    return r


def pmap(fn, jobs):
    from joblib import Parallel, delayed
    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(fn)(*j) for j in jobs)
    print(f"  {len(jobs)} jobs in {time.time() - t0:.0f} s", flush=True)
    return out


def prebuild(cfg, seeds):
    for s in seeds:
        get_truth(cfg, s)


def save(name, **kw):
    np.savez_compressed(os.path.join(RES, name + ".npz"), **kw)


def load(name):
    return dict(np.load(os.path.join(RES, name + ".npz"), allow_pickle=True))
