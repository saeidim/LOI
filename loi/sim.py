"""Slot-level simulator of the look/imagine RSU and all policies.

Policies
  'cv'     periodic looks every T slots, world-model imagination (CWNA)
  'zoh'    periodic looks every T slots, beams held between sweeps
  'loi'    Look-Or-Imagine: adaptive period from the exact certificate
  'loir'   LOI with ACK reliability (Clopper-Pearson) and conformal beam error
  'rand'   periodic CV with the uniform stationary randomized scheduler
  'oracle' genie beams and channels, no looks
"""
import time
import numpy as np
from scipy.stats import beta as beta_dist
from . import channel as ch
from .worldmodel import WorldModel
from .aoi import Phi

AGE_EDGES = np.array([0, 16, 32, 64, 128, 256, 512, 1024, 4096])


class Reliability:
    """Clopper-Pearson calibration of predicted success per belief-age bin."""

    def __init__(self, delta=0.05, n_min=20):
        nb = len(AGE_EDGES) - 1
        self.n = np.zeros(nb); self.s = np.zeros(nb); self.p = np.zeros(nb)
        self.delta, self.n_min = delta, n_min

    def add(self, age, phat, ok):
        b = min(np.searchsorted(AGE_EDGES, age, side="right") - 1, len(self.n) - 1)
        self.n[b] += 1; self.s[b] += ok; self.p[b] += phat

    def factors(self):
        nb = len(self.n)
        f = np.ones(nb)
        last = 1.0
        for b in range(nb):
            if self.n[b] >= self.n_min:
                a = self.delta / nb
                lo = 0.0 if self.s[b] == 0 else beta_dist.ppf(a, self.s[b], self.n[b] - self.s[b] + 1)
                last = min(1.0, lo / max(self.p[b] / self.n[b], 1e-6))
            f[b] = last
        return f

    def refresh(self):
        self.f = self.factors()

    def factor_at(self, ages):
        f = getattr(self, "f", None)
        if f is None:
            self.refresh(); f = self.f
        b = np.clip(np.searchsorted(AGE_EDGES, ages, side="right") - 1, 0, len(f) - 1)
        return f[b]


def conformal_nodes(scores, n=8):
    """Equal-weight quantile nodes of the folded score distribution."""
    if len(scores) < 50:
        return None, None
    s = np.asarray(scores[-3000:])
    lv = (np.arange(n) + 0.5) / n
    return np.quantile(s, lv), np.full(n, 1.0 / n)


def choose_period(cfg, P_coarse, ages_coarse, Tgrid, h=None, lam=0.0):
    """LOI period selection (Algorithm 1): success floors pi_k(T) = min over
    the data ages of a period, servable set, certificate B(T) of Corollary 2
    extended by the outage-exit term of Proposition 2, and the shortest
    period within (1+eps_B) of its minimum.

    A vehicle outside the servable set can only be rediscovered by a look.
    If it leaves its outage with hazard lam per slot, the expected excess
    AoI per slot caused by waiting for the next look is
    lam * (h_k * T / 2 + T^2 / 6)."""
    tau = cfg.tau
    cm = np.minimum.accumulate(P_coarse, axis=0)                  # (A, K)
    # pi_k(T) = min over data ages 1..T-tau
    idx = np.searchsorted(ages_coarse, Tgrid - tau, side="right") - 1
    pi = cm[np.clip(idx, 0, len(ages_coarse) - 1)]               # (nT, K)
    serv = pi[0] >= cfg.pi_min
    if not serv.any():
        return int(Tgrid[0]), np.full(len(Tgrid), np.nan), serv
    pis = np.maximum(pi[:, serv], 1e-4)
    w = np.sqrt(1.0 / pis)
    mu = w / w.sum(1, keepdims=True)
    K = P_coarse.shape[1]
    B = Phi(Tgrid[:, None], tau, mu * pis).sum(1)
    if h is not None and lam > 0 and (~serv).any():
        ho = h[~serv][None, :]
        Tg = Tgrid[:, None].astype(float)
        B = B + (lam * (ho * Tg / 2.0 + Tg ** 2 / 6.0)).sum(1)
    B = B / K
    Bs = B.min()
    T = int(Tgrid[np.flatnonzero(B <= (1 + cfg.eps_B) * Bs)[0]])
    return T, B, serv


def run(truth, cfg, policy, table, q, T=None, record=False, seed=0):
    """Simulate one policy on one traffic realization. Returns metrics."""
    K = cfg.K
    n0, N = truth["n_init"], truth["n_slots"]
    tau = cfg.tau
    X, Y, hant = truth["x"], truth["y"], truth["hant"]
    snr_true, thr = truth["snr"], truth["thr"]
    rng = np.random.default_rng([seed, 99])
    h = np.arange(1, K + 1, dtype=float)[rng.permutation(K)]
    aoi_sum = 0.0
    hist = np.zeros(K)                    # per-vehicle sum of AoI
    tail = 0
    n_tx = n_ok = 0
    n_look_slots = 0
    periods = []
    rec = dict(age=[], ok=[], k=[], phat=[], zcov=[], ncal=[], x=[], sig=[], err=[])
    served = np.zeros(K); deliv = np.zeros(K)

    aoi_t = []

    def tick(nslots, delivered_k=None):
        nonlocal aoi_sum, tail, h, hist
        if nslots == 1:
            h += 1
            if delivered_k is not None:
                h[delivered_k] = 1
            aoi_sum += h.sum(); hist += h; tail += (h > 4 * K).sum()
            if record:
                aoi_t.append(h.mean())
        else:
            ar = np.arange(1, nslots + 1)
            hh = h[None] + ar[:, None]
            aoi_sum += hh.sum(); hist += hh.sum(0); tail += (hh > 4 * K).sum()
            h = h + nslots
            if record:
                aoi_t.extend(hh.mean(1).tolist())

    blk = (truth["knife"][n0:n0 + N] > 3.0).mean(0)

    if policy == "oracle":
        u_t, _, _ = ch.dcos(X, Y, hant[None], cfg)
        for t in range(N):
            s = n0 + t
            okv = snr_true[s] >= thr[s]
            idx = np.flatnonzero(okv)
            if len(idx):
                k = idx[np.argmax(h[idx] * (h[idx] + 2))]
                tick(1, k); n_ok += 1
            else:
                k = np.argmax(h); tick(1)
            n_tx += 1
        return dict(aoi=aoi_sum / (N * K), succ=n_ok / n_tx, look=0.0,
                    tail=tail / (N * K), per_aoi=hist / N, Tmed=np.nan, blk=blk,
                    aoi_t=np.array(aoi_t))

    motion = policy != "zoh"
    wm = WorldModel(cfg, truth, q, table, motion=motion)
    for i in range(cfg.n_init_looks):
        wm.look(i * cfg.init_look_gap)
    rel = Reliability() if policy == "loir" else None
    Tgrid = cfg.T_grid()
    t = 0
    look_times = []
    prev_out = np.zeros(K, bool)          # outage state at the previous look
    out_time = 0.0; exits = 0             # online estimate of the exit hazard
    t_look_cpu = []
    while t < N:
        # ---------------- LOOK ----------------
        nl = min(tau, N - t)
        tick(nl)
        n_look_slots += nl
        t_meas = n0 + t + tau - 1
        if t + tau > N:
            break
        c0 = time.perf_counter()
        wm.look(t_meas)
        enodes = ew = None
        if policy == "loir":
            enodes, ew = conformal_nodes(wm.scores)
            rel.refresh()
        if policy in ("loi", "loir"):
            ages_c = np.arange(1, cfg.T_max - tau + 1, cfg.loi_age_step)
            im = wm.imagine(ages_c)
            Pc = wm.success(im, enodes, ew)
            if rel is not None:
                Pc = Pc * rel.factor_at(ages_c)[:, None]
            out_now = Pc[0] < cfg.pi_min
            if look_times:
                out_time += prev_out.sum() * (t - look_times[-1])
                exits += int((prev_out & ~out_now).sum())
            prev_out = out_now
            lam = (exits + 1.0) / (out_time + cfg.exit_prior_slots)
            Tcur, _, _ = choose_period(cfg, Pc, ages_c, Tgrid, h=h, lam=lam)  # h: ages at the end of the look
        else:
            Tcur = T
        periods.append(Tcur)
        look_times.append(t)
        D = Tcur - tau
        ages = np.arange(1, D + 1)
        im = wm.imagine(ages)
        P = wm.success(im, enodes, ew)
        if rel is not None:
            P = P * rel.factor_at(ages)[:, None]
        t_look_cpu.append(time.perf_counter() - c0)
        t += tau
        # ---------------- IMAGINE ----------------
        for a in range(1, D + 1):
            if t >= N:
                break
            s = n0 + t
            if record:
                rec.setdefault("Prow", []).append(P[a - 1].copy())
                rec.setdefault("slot", []).append(s)
            if policy == "rand":
                k = int(rng.integers(K))
            else:
                pa = P[a - 1]
                # a transmission whose predicted success is below p_tx is almost
                # surely wasted: such vehicles are not offered to the index
                pa = np.where(pa >= cfg.p_tx, pa, 0.0) if (pa >= cfg.p_tx).any() else pa
                k = int(np.argmax(pa * h * (h + 2)))
            uh = im["uhat"][a - 1, k]
            ut, _, _ = ch.dcos(X[s, k], Y[s, k], hant[k], cfg)
            g = ch.ula_gain(ut - uh, cfg.n_ant)
            ok = snr_true[s] [k] + 10 * np.log10(max(g, 1e-12)) >= thr[s, k]
            n_tx += 1; n_ok += ok; served[k] += 1; deliv[k] += ok
            if rel is not None:
                rel.add(a, P[a - 1, k] / max(rel.factor_at(np.array([a]))[0], 1e-9), ok)
            if record:
                sig = np.sqrt(im["su2"][a - 1, k] + im["sm2"][a - 1, k])
                rec["age"].append(a); rec["ok"].append(ok); rec["k"].append(k)
                rec["phat"].append(P[a - 1, k]); rec["x"].append(X[s, k])
                rec["err"].append(ut - uh); rec["sig"].append(sig)
                rec["ncal"].append(len(wm.scores))
            tick(1, k if ok else None)
            if not ok and policy != "rand" and a < D:
                wm.nack(k, a, {kk: vv[..., a - 1:a, k:k + 1] for kk, vv in im.items()})
                im_k = wm.imagine(ages[a:], idx=np.array([k]))
                Pk = wm.success(im_k, enodes, ew)[:, 0]
                if rel is not None:
                    Pk = Pk * rel.factor_at(ages[a:])
                P[a:, k] = Pk
                for kk in ("snr", "svar"):
                    im[kk][a:, k] = im_k[kk][:, 0]
                im["snr_sp"][:, a:, k] = im_k["snr_sp"][:, :, 0]
            t += 1
    out = dict(aoi=aoi_sum / (N * K), succ=n_ok / max(n_tx, 1), look=n_look_slots / N,
               tail=tail / (N * K), per_aoi=hist / N, Tmed=float(np.median(periods)), blk=blk,
               periods=np.array(periods), cpu_look=float(np.median(t_look_cpu)),
               p_served=deliv / np.maximum(served, 1), served=served)
    if record:
        out["rec"] = {k: np.array(v) for k, v in rec.items()}
        out["aoi_t"] = np.array(aoi_t)
        out["look_t"] = np.array(look_times)
        out["scores"] = np.array(wm.scores)
    return out
