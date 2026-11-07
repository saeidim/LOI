"""Propagation: array gain, 3GPP TR 38.901 path loss, ITU-R P.526 knife-edge
blockage by vehicle bodies, Gudmundson shadowing and wideband Rician fading.

The same geometric functions are used for the ground truth (true positions)
and by the world model's renderer (imagined positions), so the world model is
"physics-grounded" but never sees the true state.
"""
import numpy as np

# --------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------

def ula_gain(eps, n):
    """Normalised power gain |1/N sum exp(j pi n eps)|^2 of an N-element
    half-wavelength ULA for direction-cosine error eps (any shape)."""
    eps = np.asarray(eps, dtype=float)
    th = 0.5 * np.pi * eps
    s = np.sin(th)
    num = np.sin(n * th)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (num / (n * s)) ** 2
    return np.where(np.abs(s) < 1e-12, 1.0, g)


def kappa(n):
    return np.pi ** 2 * (n ** 2 - 1) / 12.0


def dcos(x, y, h, cfg):
    """Direction cosine (eq. (1)) and its gradient w.r.t. (x, y)."""
    dy = y - cfg.rsu_y
    dh = h - cfg.rsu_h
    d2 = x * x + dy * dy + dh * dh
    d = np.sqrt(d2)
    u = x / d
    jx = (d2 - x * x) / (d2 * d)
    jy = -x * dy / (d2 * d)
    return u, jx, jy


# --------------------------------------------------------------------------
# Large-scale
# --------------------------------------------------------------------------

def pathloss_db(d3, cfg):
    return 32.4 + 21.0 * np.log10(np.maximum(d3, 1.0)) + 20.0 * np.log10(cfg.fc_ghz)


def snr0_db(x, y, h, cfg):
    """Mean SNR with perfect alignment, no shadowing, no blockage."""
    d3 = np.sqrt(x ** 2 + (y - cfg.rsu_y) ** 2 + (h - cfg.rsu_h) ** 2)
    return (cfg.ptx_dbm + 10 * np.log10(cfg.n_ant) + cfg.rx_gain_dbi
            - pathloss_db(d3, cfg) - cfg.noise_dbm)


def knife_J(nu):
    """ITU-R P.526 single knife-edge loss [dB] (0 for nu <= -0.78)."""
    nu = np.asarray(nu, dtype=float)
    v = nu - 0.1
    J = 6.9 + 20.0 * np.log10(np.sqrt(v * v + 1.0) + v)
    return np.where(nu > -0.78, J, 0.0)


def blockage_db(x, y, hant, L, W, H, cfg, xr=None, yr=None):
    """Knife-edge loss of each receiver caused by the other vehicles.

    x, y : (..., K) positions (the leading axes are e.g. time).
    hant : (K,) antenna heights; L, W, H : (K,) body dimensions.
    xr, yr : optional receiver positions (default x, y); the blockers stay at
    (x, y). Used for sigma-point rendering of the relative-position uncertainty.
    Returns loss (..., K) in dB: for every blocker the easier of the roof and
    the near-side diffraction path is charged and the dominant blocker kept.
    """
    lam = cfg.lam
    xr = x if xr is None else xr
    yr = y if yr is None else yr
    dx = xr[..., :, None]                      # receiver k  (..., K, 1)
    dy = yr[..., :, None] - cfg.rsu_y
    bx = x[..., None, :]                       # blocker j   (..., 1, K)
    by = y[..., None, :] - cfg.rsu_y
    dh = np.sqrt(dx * dx + dy * dy)            # horizontal RSU-Rx distance
    cphi = dx / dh
    sphi = dy / dh
    t = bx * cphi + by * sphi                  # along-ray coordinate of blocker
    perp = np.abs(-bx * sphi + by * cphi)
    Lj, Wj, Hj = L[None, :], W[None, :], H[None, :]
    half_perp = 0.5 * (Lj * np.abs(sphi) + Wj * np.abs(cphi))
    half_along = 0.5 * (Lj * np.abs(cphi) + Wj * np.abs(sphi))
    d1 = np.clip(t, 0.5, None)
    d2 = np.clip(dh - t, 0.5, None)
    fres = np.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / d2))
    hk = hant[:, None]
    hray = cfg.rsu_h + (hk - cfg.rsu_h) * np.clip(t / dh, 0, 1)
    nu_top = (Hj - hray) * fres
    nu_side = (half_perp - perp) * fres
    J = np.minimum(knife_J(nu_top), knife_J(nu_side))
    K = x.shape[-1]
    valid = (t > half_along) & (t < dh - half_along) & ~np.eye(K, dtype=bool)
    J = np.where(valid, J, 0.0)
    return J.max(axis=-1)


# --------------------------------------------------------------------------
# Small-scale: wideband success
# --------------------------------------------------------------------------

def fading_power(rng, shape, k_db):
    """|h|^2 of Rician fading with factor k_db (k_db=None -> Rayleigh)."""
    g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    if k_db is None:
        return np.abs(g) ** 2
    k = 10 ** (k_db / 10)
    th = rng.uniform(0, 2 * np.pi, shape[:-1] + (1,))
    h = np.sqrt(k / (k + 1)) * np.exp(1j * th) + np.sqrt(1 / (k + 1)) * g
    return np.abs(h) ** 2


def snr_threshold_db(p, r_req, iters=40):
    """For sub-band powers p (..., F) return gamma* [dB] such that
    mean_f log2(1 + gamma |h_f|^2) >= r_req  iff  gamma >= gamma*."""
    lo = np.full(p.shape[:-1], -30.0)
    hi = np.full(p.shape[:-1], 70.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        se = np.log2(1 + 10 ** (mid[..., None] / 10) * p).mean(-1)
        ok = se >= r_req
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


class SuccessTable:
    """Tabulated wideband success probability P_s(SNR_dB) for Rician and
    Rayleigh fading (empirical CDF of the per-slot SNR threshold)."""

    def __init__(self, cfg, n=200_000, seed=12345):
        rng = np.random.default_rng(seed)
        self.grid = np.arange(-20.0, 60.0, 0.05)
        self.tab = {}
        for name, kdb in (("rice", cfg.k_rice_db), ("rayl", None)):
            thr = np.sort(snr_threshold_db(fading_power(rng, (n, cfg.n_sub), kdb), cfg.r_req))
            self.tab[name] = np.searchsorted(thr, self.grid, side="right") / n
        self.thr50 = {k: float(np.interp(0.5, v, self.grid)) for k, v in self.tab.items()}
        self.stack = np.stack([self.tab["rice"], self.tab["rayl"]])
        self.step = self.grid[1] - self.grid[0]

    def __call__(self, snr_db, rayleigh):
        """snr_db: array; rayleigh: boolean array broadcastable to snr_db.
        Nearest-node lookup on the 0.05 dB grid."""
        i = np.rint((np.asarray(snr_db) - self.grid[0]) / self.step).astype(np.intp)
        np.clip(i, 0, len(self.grid) - 1, out=i)
        r = np.broadcast_to(np.asarray(rayleigh, dtype=np.intp), i.shape)
        return self.stack[r, i]
