"""The RSU world model: CWNA latent kinematics (Kalman filters fed by sweeps),
a deterministic physics renderer of the imagined scene, and a link belief
(shadowing estimate, censoring of missed detections, soft censoring of NACKs).
With motion=False it degenerates to the zero-order hold (ZOH) of
conventional beam management.
"""
import numpy as np
from . import channel as ch
from .traffic import wrap

GH_X, GH_W = np.polynomial.hermite.hermgauss(15)
GH_W = GH_W / np.sqrt(np.pi)                       # E[f(Z)] = sum w f(sqrt2 x)
# folded nodes for the symmetric beam error
_pos = GH_X >= -1e-12
FX = np.abs(GH_X[_pos]); FW = np.where(np.abs(GH_X[_pos]) < 1e-12, 1.0, 2.0) * GH_W[_pos]


SP_W = (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0)        # unscented weights (1-D, kappa=2)


def cwna_FQ(t, q):
    F = np.array([[1.0, t], [0.0, 1.0]])
    Q = q * np.array([[t ** 3 / 3, t ** 2 / 2], [t ** 2 / 2, t]])
    return F, Q


def meas_var(snr_db, cfg):
    return cfg.sigma0 ** 2 * 10 ** ((cfg.gamma_ref_db - snr_db) / 10)


class WorldModel:
    def __init__(self, cfg, truth, q, table, motion=True):
        self.cfg, self.tr, self.tab, self.motion = cfg, truth, table, motion
        self.qx, self.qy = q
        K = cfg.K
        self.K = K
        self.hant, self.L, self.W, self.H = truth["hant"], truth["L"], truth["W"], truth["H"]
        self.init = np.zeros(K, bool)
        self.mx = np.zeros((K, 2)); self.Px = np.zeros((K, 2, 2))
        self.my = np.zeros((K, 2)); self.Py = np.zeros((K, 2, 2))
        self.zx = np.zeros(K); self.zy = np.zeros(K); self.R = np.ones(K)
        self.t_post = 0                  # slot of the posterior (last look)
        self.mu_s = np.zeros(K); self.var_s = np.full(K, cfg.sigma_sf_db ** 2)
        self.a0 = np.zeros(K)            # age at which the shadow belief is anchored
        self.cens = np.zeros(K, bool)
        self.wsp = np.tile(SP_W, (K, 1))  # weights of the blockage sigma points
        self.scores = []                 # conformal calibration scores

    # ------------------------------------------------------------------
    def _predict_post(self, dt_s):
        """Propagate posteriors by dt_s seconds (all vehicles)."""
        if dt_s <= 0:
            return
        Fx, Qx = cwna_FQ(dt_s, self.qx)
        Fy, Qy = cwna_FQ(dt_s, self.qy)
        self.mx = self.mx @ Fx.T; self.mx[:, 0] = wrap(self.mx[:, 0], self.cfg.road_len)
        self.my = self.my @ Fy.T
        self.Px = Fx @ self.Px @ Fx.T + Qx
        self.Py = Fy @ self.Py @ Fy.T + Qy

    def look(self, slot):
        """A sweep whose measurements refer to `slot` (end of the look).
        Returns the indices of detected vehicles."""
        cfg, tr = self.cfg, self.tr
        nz = tr["meas"][slot]
        snr_meas = tr["snr"][slot] + cfg.snr_read_db * nz[:, 2]
        det = snr_meas >= cfg.gamma_det_db
        R = meas_var(snr_meas, cfg)
        sd = np.sqrt(R)
        zx = wrap(tr["x"][slot] + sd * nz[:, 0], cfg.road_len)
        zy = tr["y"][slot] + sd * nz[:, 1]
        dt_s = (slot - self.t_post) * cfg.slot_s
        # conformal scores of the imagined beams (before the update), CV only
        if self.motion and dt_s > 0:
            Fx, Qx = cwna_FQ(dt_s, self.qx); Fy, Qy = cwna_FQ(dt_s, self.qy)
            mxp = wrap(self.mx[:, 0] + dt_s * self.mx[:, 1], cfg.road_len)
            myp = self.my[:, 0] + dt_s * self.my[:, 1]
            Pxx = (Fx @ self.Px @ Fx.T + Qx)[:, 0, 0]
            Pyy = (Fy @ self.Py @ Fy.T + Qy)[:, 0, 0]
            uh, jx, jy = ch.dcos(mxp, myp, self.hant, cfg)
            uz, jzx, jzy = ch.dcos(zx, zy, self.hant, cfg)
            su2 = jx ** 2 * Pxx + jy ** 2 * Pyy
            sm2 = (jzx ** 2 + jzy ** 2) * R
            ok = det & self.init
            self.scores.extend((np.abs(uz - uh) / np.sqrt(su2 + sm2))[ok].tolist())
        self._predict_post(dt_s)
        self.t_post = slot
        for k in np.flatnonzero(det):
            if not self.init[k]:
                self.mx[k] = [zx[k], 0.0]; self.Px[k] = np.diag([R[k], 30.0 ** 2])
                self.my[k] = [zy[k], 0.0]; self.Py[k] = np.diag([R[k], 1.0])
                self.init[k] = True
                continue
            for m, P, z in ((self.mx, self.Px, zx[k]), (self.my, self.Py, zy[k])):
                nu = z - m[k, 0]
                if m is self.mx:
                    nu = wrap(nu, cfg.road_len)
                S = P[k, 0, 0] + R[k]
                G = P[k, :, 0] / S
                m[k] += G * nu
                P[k] -= np.outer(G, G) * S
            self.mx[k, 0] = wrap(self.mx[k, 0], cfg.road_len)
        self.zx = np.where(det, zx, self.zx); self.zy = np.where(det, zy, self.zy)
        self.R = np.where(det, R, self.R)
        # link belief: residual between measured and rendered SNR
        X, Y = (self.mx[:, 0], self.my[:, 0]) if self.motion else (self.zx, self.zy)
        rend = ch.snr0_db(X, Y, self.hant, cfg) - ch.blockage_db(X[None], Y[None], self.hant,
                                                                  self.L, self.W, self.H, cfg)[0]
        self.mu_s = np.where(det, snr_meas - rend, self.mu_s)
        self.var_s = np.where(det, cfg.snr_read_db ** 2, self.var_s)
        self.a0[:] = 0.0
        self.wsp[:] = SP_W
        self.cens = ~det
        # censored: excess loss at least the value that puts the SNR at gamma_det
        self.mu_s = np.where(self.cens, np.minimum(self.mu_s, cfg.gamma_det_db - rend), self.mu_s)
        self.var_s = np.where(self.cens, 1.0, self.var_s)
        return det

    # ------------------------------------------------------------------
    def kinematics(self, ages):
        """Imagined positions and position variances at belief ages (A,):
        eq. (7) for the CWNA world model, the held measurement for ZOH."""
        cfg = self.cfg
        t = np.asarray(ages, float)[:, None] * cfg.slot_s
        if self.motion:
            X = wrap(self.mx[:, 0][None] + self.mx[:, 1][None] * t, cfg.road_len)
            Y = np.clip(self.my[:, 0][None] + self.my[:, 1][None] * t, 0.0, 4 * cfg.lane_w)
            sx2 = (self.Px[:, 0, 0] + 2 * t * self.Px[:, 0, 1] + t ** 2 * self.Px[:, 1, 1]
                   + self.qx * t ** 3 / 3)
            sy2 = (self.Py[:, 0, 0] + 2 * t * self.Py[:, 0, 1] + t ** 2 * self.Py[:, 1, 1]
                   + self.qy * t ** 3 / 3)
        else:
            X = np.broadcast_to(self.zx, (len(t), self.K))
            Y = np.broadcast_to(self.zy, (len(t), self.K))
            sx2 = sy2 = np.broadcast_to(self.R, (len(t), self.K))
        return X, Y, sx2, sy2

    def beam(self, ages):
        """Imagined direction cosine and its reported variance (Thm. 2(iii))."""
        X, Y, sx2, sy2 = self.kinematics(ages)
        uh, jx, jy = ch.dcos(X, Y, self.hant, self.cfg)
        return uh, jx ** 2 * sx2 + jy ** 2 * sy2

    def imagine(self, ages, idx=None):
        """Render the imagined scene at belief ages (A,) for vehicles idx.
        Returns dict of (A, k) arrays: uhat, su2, sm2, snr (mean, incl.
        shadow belief), svar (shadow variance), rayl (bool) and the
        sigma-point versions snr_sp, rayl_sp."""
        cfg = self.cfg
        t = np.asarray(ages, float)[:, None] * cfg.slot_s
        X, Y, sx2, sy2 = self.kinematics(ages)
        speed = np.abs(self.mx[:, 1]) if self.motion else np.zeros(self.K)
        # unscented rendering of blockage: the receiver is displaced along the
        # road by the sigma points of the relative position uncertainty
        srel = np.sqrt(2.0 * sx2) if self.motion else np.zeros_like(sx2)
        knives = [ch.blockage_db(X, Y, self.hant, self.L, self.W, self.H, cfg, xr=X + d * srel)
                  for d in (0.0, -np.sqrt(3.0), np.sqrt(3.0))]
        knife = knives[0]
        if idx is not None:
            X, Y, sx2, sy2 = X[:, idx], Y[:, idx], sx2[:, idx], sy2[:, idx]
            knives = [kn[:, idx] for kn in knives]
            knife = knives[0]
            sl = idx
        else:
            sl = slice(None)
        h = self.hant[sl]
        uh, jx, jy = ch.dcos(X, Y, h, cfg)
        su2 = jx ** 2 * sx2 + jy ** 2 * sy2
        R = self.R[sl]
        sm2 = (jx ** 2 + jy ** 2) * R
        snr0 = ch.snr0_db(X, Y, h, cfg)
        snr = snr0 - knife
        dist = np.maximum(t - self.a0[sl] * cfg.slot_s, 0.0) * speed[sl]
        rho = np.exp(-dist / cfg.d_corr_m)
        mu = rho * self.mu_s[sl]
        var = rho ** 2 * self.var_s[sl] + cfg.sigma_sf_db ** 2 * (1 - rho ** 2)
        sp = np.stack([snr0 - kn + mu for kn in knives])            # (3, A, k)
        return dict(uhat=uh, su2=su2, sm2=sm2, snr=snr + mu, svar=var,
                    rayl=knife > 3.0, snr_nosh=snr, snr_sp=sp,
                    rayl_sp=np.stack([kn > 3.0 for kn in knives]),
                    wsp=self.wsp[sl].T.copy())

    def success(self, im, eps_nodes=None, eps_w=None):
        """Predicted success probability by Gauss-Hermite quadrature over the
        Gaussian beam error and the residual shadowing. Optionally uses
        conformal score quantiles (eps_nodes, multiples of the score scale)."""
        n = self.cfg.n_ant
        if eps_nodes is None:
            e = np.sqrt(2 * im["su2"])[..., None] * FX            # (A,k,ne)
            we = FW
        else:
            e = np.sqrt(im["su2"] + im["sm2"])[..., None] * eps_nodes
            we = eps_w
        gdb = 10 * np.log10(np.maximum(ch.ula_gain(e, n), 1e-6))   # (A,k,ne)
        s = np.sqrt(2 * im["svar"])[..., None] * GH_X                # (A,k,ns)
        out = 0.0
        for wsp, snr_m, ray in zip(im["wsp"], im["snr_sp"], im["rayl_sp"]):
            snr = snr_m[..., None, None] + gdb[..., :, None] + s[..., None, :]
            ps = self.tab(snr, ray[..., None, None])
            out = out + wsp[None, :] * np.einsum("akij,i,j->ak", ps, we, GH_W)
        return out

    def nack(self, k, age, im_k):
        """Soft-censored Bayesian update of the link belief of vehicle k after
        a NACK at belief age `age` (im_k: rendering of k at that age, (1,1)).
        The joint posterior over the blockage sigma point (relative-position
        hypothesis) and the shadowing is formed with the failure likelihood
        1 - P_s; the sigma-point weights are kept and the shadowing posterior
        is moment matched. Repeated NACKs thus rule out the hypotheses under
        which the link would have worked."""
        cfg = self.cfg
        mu0, var0 = im_k["snr"][0, 0] - im_k["snr_nosh"][0, 0], im_k["svar"][0, 0]
        s = mu0 + np.sqrt(2 * var0) * GH_X                                  # (ns,)
        e = np.sqrt(2 * im_k["su2"][0, 0]) * FX
        gdb = 10 * np.log10(np.maximum(ch.ula_gain(e, cfg.n_ant), 1e-6))
        W = np.empty((3, len(s)))
        for i in range(3):
            snr = im_k["snr_sp"][i, 0, 0] - mu0 + s[:, None] + gdb[None, :]
            ps = self.tab(snr, im_k["rayl_sp"][i, 0, 0]) @ FW
            W[i] = self.wsp[k, i] * GH_W * (1.0 - ps)
        tot = W.sum()
        if tot < 1e-12:
            return
        W /= tot
        self.wsp[k] = np.maximum(W.sum(1), 1e-9)
        self.wsp[k] /= self.wsp[k].sum()
        ws = W.sum(0)
        m = (ws * s).sum()
        # variance floor: a NACK signals that the belief may be wrong, so the
        # posterior is never more confident than a fresh SNR reading
        v = max((ws * (s - m) ** 2).sum(), cfg.snr_read_db ** 2)
        self.mu_s[k], self.var_s[k], self.a0[k] = m, v, age
        self.cens[k] = False


def learn_q(cfg, truth_train, every=50, n_skip=3):
    """Learn the CWNA intensities (q_x, q_y) by maximising the innovation
    likelihood of sweeps (every `every` slots) on a training trajectory.
    Ground truth positions are never used directly, only noisy sweeps."""
    from scipy.optimize import minimize_scalar
    tr = truth_train
    slots = np.arange(0, tr["x"].shape[0], every)
    nz = tr["meas"][slots]
    snr_m = tr["snr"][slots] + cfg.snr_read_db * nz[..., 2]
    det = snr_m >= cfg.gamma_det_db
    R = meas_var(snr_m, cfg)
    zx = wrap(tr["x"][slots] + np.sqrt(R) * nz[..., 0], cfg.road_len)
    zy = tr["y"][slots] + np.sqrt(R) * nz[..., 1]
    dt = every * cfg.slot_s

    def nll(logq, Z, wrapx, v0var):
        q = np.exp(logq)
        F, Q = cwna_FQ(dt, q)
        tot = 0.0
        for k in range(Z.shape[1]):
            m = None; P = None; cnt = 0; tgap = 0
            for i in range(Z.shape[0]):
                tgap += 1
                if not det[i, k]:
                    continue
                if m is None:
                    m = np.array([Z[i, k], 0.0]); P = np.diag([R[i, k], v0var]); tgap = 0
                    continue
                Fg, Qg = cwna_FQ(dt * tgap, q); tgap = 0
                m = Fg @ m; P = Fg @ P @ Fg.T + Qg
                nu = Z[i, k] - m[0]
                if wrapx:
                    nu = wrap(nu, cfg.road_len)
                S = P[0, 0] + R[i, k]
                cnt += 1
                if cnt > n_skip:
                    tot += 0.5 * (np.log(2 * np.pi * S) + nu * nu / S)
                G = P[:, 0] / S
                m = m + G * nu; P = P - np.outer(G, G) * S
        return tot

    qx = np.exp(minimize_scalar(nll, bounds=(-6, 3), args=(zx, True, 900.0), method="bounded").x)
    qy = np.exp(minimize_scalar(nll, bounds=(-7, 2), args=(zy, False, 1.0), method="bounded").x)
    return float(qx), float(qy)
