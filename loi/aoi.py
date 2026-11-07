"""Exact freshness under periodic looks (Theorem 3) and its validation."""
import numpy as np


def Phi(T, tau, r):
    """Closed form (eq. (8)) of the time-average AoI of a link with Bernoulli(r)
    deliveries in data slots and a look of tau slots every T slots."""
    T = np.asarray(T, float); r = np.asarray(r, float)
    D = T - tau
    x = 1.0 - r
    with np.errstate(divide="ignore", invalid="ignore"):
        xD = np.where(r >= 1, 0.0, x ** D)
        tail = np.where(xD > 0, tau ** 2 * xD / (1 - xD), 0.0)
        val = 1 / r + (tau * x / r + tau * (tau + 1) / 2 + tail) / T
    return np.where((r > 0) & (D > 0), val, np.inf)


def Phi_dp(T, tau, r, hmax=None):
    """Exact evaluation by propagating the stationary age distribution over
    the periodic phases (power iteration on the phase-sampled chain)."""
    if hmax is None:
        hmax = max(3000, int(80 / r + 40 * T))
    D = T - tau
    p = np.zeros(hmax + 1); p[1] = 1.0
    for _ in range(4000):
        p_old = p.copy()
        for j in range(T):
            q = np.zeros_like(p)
            if j < tau:
                q[1:] = p[:-1]; q[-1] += p[-1]
            else:
                q[1] = r
                q[1:] += (1 - r) * np.r_[0.0, p[1:-1]]
                q[-1] += (1 - r) * p[-1]
            p = q
        if np.abs(p - p_old).sum() < 1e-13:
            break
    tot = 0.0
    h = np.arange(hmax + 1)
    for j in range(T):
        q = np.zeros_like(p)
        if j < tau:
            q[1:] = p[:-1]; q[-1] += p[-1]
        else:
            q[1] = r
            q[1:] += (1 - r) * np.r_[0.0, p[1:-1]]
            q[-1] += (1 - r) * p[-1]
        p = q
        tot += (h * p).sum()
    return tot / T


def Phi_mc(T, tau, r, n_slots=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_slots)
    data = (t % T) >= tau
    deliv = data & (rng.random(n_slots) < r)
    idx = np.flatnonzero(deliv)
    # age at end of slot t = t - last delivery + 1 (start: age counted from -1)
    last = np.full(n_slots, -1)
    last[idx] = idx
    last = np.maximum.accumulate(last)
    h = np.where(last >= 0, t - last + 1, t + 2)
    burn = n_slots // 100
    h = h[burn:]
    nb = 50                                    # batch means for the standard error
    m = h[: len(h) // nb * nb].reshape(nb, -1).mean(1)
    return h.mean(), m.std(ddof=1) / np.sqrt(nb)


def certificate(T, tau, pi, beta=None):
    """B(T) of Corollary 2 with the optimal randomized distribution
    mu_k* ~ sqrt(beta_k / pi_k)."""
    pi = np.asarray(pi, float)
    beta = np.ones_like(pi) if beta is None else beta
    w = np.sqrt(beta / pi)
    mu = w / w.sum()
    return float(np.mean(beta * Phi(T, tau, mu * pi)))
