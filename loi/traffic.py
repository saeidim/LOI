"""Microscopic traffic: intelligent driver model (IDM) with heterogeneous
desired speeds, Ornstein-Uhlenbeck driver acceleration noise and randomly
triggered lane changes (raised-cosine lateral profile, gap acceptance) on a
four-lane two-direction ring road. Dynamics are integrated at 10 ms and
expanded to the 1 ms slot grid with the exact constant-acceleration
kinematics of each step, so slot-level positions are physically consistent.
"""
import numpy as np


def wrap(z, L):
    return (z + 0.5 * L) % L - 0.5 * L


class Fleet:
    def __init__(self, cfg, rng):
        K = cfg.K
        self.cfg = cfg
        n_tr = int(round(cfg.truck_frac * K))
        truck = np.zeros(K, bool)
        truck[rng.choice(K, n_tr, replace=False)] = True
        self.truck = truck
        self.L = np.where(truck, cfg.truck_len, cfg.car_len)
        self.W = np.where(truck, cfg.truck_w, cfg.car_w)
        self.H = np.where(truck, cfg.truck_h, cfg.car_h)
        self.hant = self.H.copy()                     # roof-mounted antenna
        lo = np.where(truck, cfg.truck_v0[0], cfg.car_v0[0])
        hi = np.where(truck, cfg.truck_v0[1], cfg.car_v0[1])
        self.v0 = rng.uniform(lo, hi)
        self.amax = np.where(truck, cfg.truck_a, cfg.car_a)
        # direction: half of the fleet each way; lanes two per direction
        dirn = np.where(np.arange(K) % 2 == 0, 1, -1)
        rng.shuffle(dirn)
        self.dir = dirn
        ycen, ldir = cfg.lanes()
        self.ycen = ycen
        lane = np.empty(K, int)
        for d in (1, -1):
            idx = np.flatnonzero(dirn == d)
            opts = np.flatnonzero(ldir == d)
            lane[idx] = rng.choice(opts, len(idx))
            # trucks keep to the right-hand (slow) lane of their direction
            right = opts[0] if d == 1 else opts[-1]
            lane[idx[truck[idx]]] = right
        self.lane = lane
        # longitudinal coordinate s along the direction of travel (ring)
        s = np.empty(K)
        for d in (1, -1):
            idx = np.flatnonzero(dirn == d)
            base = np.sort(rng.uniform(0, cfg.road_len, len(idx)))
            # enforce spacing by redistributing evenly with jitter
            sp = cfg.road_len / max(len(idx), 1)
            base = np.arange(len(idx)) * sp + rng.uniform(0, 0.3 * sp, len(idx))
            s[idx] = base
        self.s = s
        self.v = self.v0 * rng.uniform(0.85, 1.0, K)
        self.ou = np.zeros(K)
        self.y = ycen[lane].astype(float)
        self.lc_t = np.full(K, np.inf)      # time since lane change start
        self.lc_from = self.y.copy()
        self.lc_to = self.y.copy()
        self.odo = np.zeros(K)

    # ---------------------------------------------------------------
    def _gaps(self):
        """Gap to and speed of the leader in the same lane (ring)."""
        K = len(self.s)
        Lr = self.cfg.road_len
        same = (self.lane[:, None] == self.lane[None, :]) & ~np.eye(K, dtype=bool)
        ds = (self.s[None, :] - self.s[:, None]) % Lr        # j ahead of i
        ds = np.where(same, ds, np.inf)
        j = np.argmin(ds, axis=1)
        gap = ds[np.arange(K), j] - 0.5 * (self.L + self.L[j])
        vl = np.where(np.isfinite(ds[np.arange(K), j]), self.v[j], self.v)
        gap = np.where(np.isfinite(gap), gap, 1e3)
        return np.maximum(gap, 0.1), vl

    def step(self, dt, rng):
        cfg = self.cfg
        gap, vl = self._gaps()
        dv = self.v - vl
        sstar = cfg.idm_s0 + np.maximum(0, self.v * cfg.idm_T
                                        + self.v * dv / (2 * np.sqrt(self.amax * cfg.idm_b)))
        acc = self.amax * (1 - (self.v / self.v0) ** 4 - (sstar / gap) ** 2)
        # OU driver noise
        a_ou = np.exp(-dt / cfg.ou_tau)
        self.ou = a_ou * self.ou + cfg.ou_sigma * np.sqrt(1 - a_ou ** 2) * rng.standard_normal(len(acc))
        acc = np.clip(acc + self.ou, -9.0, 3.0)
        acc = np.where(self.v + acc * dt < 0, -self.v / dt, acc)
        # lane changes (lc_t = time since the start of the manoeuvre): random
        # (rate lc_rate) and incentive-driven overtaking of a slower leader
        # (rate lc_overtake_rate, simplified MOBIL incentive), both subject to
        # gap acceptance in the target lane
        free = ~np.isfinite(self.lc_t) | (self.lc_t > cfg.lc_dur)
        stuck = (self.v0 - vl > cfg.lc_dv) & (gap < cfg.lc_look)
        rate = cfg.lc_rate + np.where(stuck, cfg.lc_overtake_rate, 0.0)
        start = free & ~self.truck & (rng.random(len(acc)) < rate * dt)
        for i in np.flatnonzero(start):
            _, ldir = cfg.lanes()
            opts = [l for l in np.flatnonzero(ldir == self.dir[i]) if l != self.lane[i]]
            tgt = opts[0]
            others = np.flatnonzero((self.lane == tgt) & (np.arange(len(acc)) != i))
            ds = wrap(self.s[others] - self.s[i], cfg.road_len)
            if np.all(np.abs(ds) > cfg.lc_gap + 0.5 * (self.L[others] + self.L[i])):
                self.lc_from[i] = self.y[i]
                self.lc_to[i] = self.ycen[tgt]
                self.lane[i] = tgt
                self.lc_t[i] = 0.0
        return acc

    def lateral(self, tl):
        """Lateral position and velocity given time since lane-change start."""
        cfg = self.cfg
        act = tl < cfg.lc_dur
        ph = np.clip(tl / cfg.lc_dur, 0, 1)
        y = np.where(act, self.lc_from + (self.lc_to - self.lc_from) * 0.5 * (1 - np.cos(np.pi * ph)),
                     self.lc_to)
        vy = np.where(act, (self.lc_to - self.lc_from) * 0.5 * np.pi / cfg.lc_dur * np.sin(np.pi * ph), 0.0)
        return y, vy


def simulate_traffic(cfg, seed, duration_s, record_from_s):
    """Return dict with slot-level arrays x, y, vx, vy, odo (n, K) and the
    static fleet description. Time zero of the record is record_from_s."""
    rng = np.random.default_rng(seed)
    fl = Fleet(cfg, rng)
    dt = 0.01
    sub = int(round(dt / cfg.slot_s))
    n_steps = int(round(duration_s / dt))
    rec0 = int(round(record_from_s / dt))
    n_rec = (n_steps - rec0) * sub
    K = cfg.K
    X = np.empty((n_rec, K)); Y = np.empty((n_rec, K))
    VX = np.empty((n_rec, K)); VY = np.empty((n_rec, K)); OD = np.empty((n_rec, K))
    frac = (np.arange(sub) * cfg.slot_s)[:, None]
    for st in range(n_steps):
        acc = fl.step(dt, rng)
        if st >= rec0:
            i0 = (st - rec0) * sub
            s_sub = fl.s[None] + fl.v[None] * frac + 0.5 * acc[None] * frac ** 2
            v_sub = fl.v[None] + acc[None] * frac
            tl = fl.lc_t[None] + frac
            y_sub, vy_sub = fl.lateral(tl)
            X[i0:i0 + sub] = wrap(fl.dir[None] * s_sub, cfg.road_len)
            VX[i0:i0 + sub] = fl.dir[None] * v_sub
            Y[i0:i0 + sub] = y_sub
            VY[i0:i0 + sub] = vy_sub
            OD[i0:i0 + sub] = fl.odo[None] + fl.v[None] * frac + 0.5 * acc[None] * frac ** 2
        fl.s = (fl.s + fl.v * dt + 0.5 * acc * dt ** 2) % cfg.road_len
        fl.odo = fl.odo + fl.v * dt + 0.5 * acc * dt ** 2
        fl.v = np.maximum(fl.v + acc * dt, 0.0)
        fl.lc_t = fl.lc_t + dt
        fl.y, _ = fl.lateral(fl.lc_t)
    static = dict(truck=fl.truck, L=fl.L, W=fl.W, H=fl.H, hant=fl.hant, dir=fl.dir)
    return dict(x=X, y=Y, vx=VX, vy=VY, odo=OD, **static)
