"""Scenario and algorithm parameters (Table I of the paper).

All quantities are SI unless stated otherwise. A single frozen dataclass is
used so that every experiment states exactly which parameters it changes.
"""
from dataclasses import dataclass, replace, field
import numpy as np

C0 = 299_792_458.0


@dataclass(frozen=True)
class Config:
    # --- radio -----------------------------------------------------------
    fc_ghz: float = 28.0
    bw_hz: float = 400e6
    slot_s: float = 1e-3
    ptx_dbm: float = 20.0
    nf_db: float = 9.0
    rx_gain_dbi: float = 5.0
    n_ant: int = 256
    r_req: float = 4.0                 # b/s/Hz per update (S = r_req*B*slot)
    n_sub: int = 16                    # wideband sub-bands F
    k_rice_db: float = 9.0
    sigma_sf_db: float = 4.0
    d_corr_m: float = 10.0
    snr_read_db: float = 1.0           # std of the SNR reading of a sweep
    # --- road / RSU ------------------------------------------------------
    road_len: float = 400.0
    lane_w: float = 3.5
    rsu_offset: float = 4.0            # lateral distance from road edge
    rsu_h: float = 6.0
    # --- fleet -----------------------------------------------------------
    K: int = 16
    truck_frac: float = 0.25
    car_len: float = 4.5
    car_w: float = 1.8
    car_h: float = 1.5
    truck_len: float = 12.0
    truck_w: float = 2.5
    truck_h: float = 3.8
    car_v0: tuple = (25.0, 33.0)
    truck_v0: tuple = (22.0, 25.0)
    car_a: float = 1.5
    truck_a: float = 0.8
    idm_b: float = 2.0
    idm_T: float = 1.2
    idm_s0: float = 2.0
    ou_sigma: float = 0.4              # driver acceleration noise [m/s^2]
    ou_tau: float = 1.0
    lc_rate: float = 0.05              # lane-change attempts per second
    lc_dur: float = 3.0
    lc_gap: float = 12.0               # gap acceptance (front and back) [m]
    lc_overtake_rate: float = 0.25      # lane-change rate when held up by a slower leader
    lc_dv: float = 1.5                 # speed deficit that triggers overtaking [m/s]
    lc_look: float = 60.0              # ... if the leader is closer than this [m]
    # --- looks -----------------------------------------------------------
    tau: int = 12
    sigma0: float = 0.15               # position noise at gamma_ref
    gamma_ref_db: float = 20.0
    gamma_det_db: float = -3.0
    # --- simulation ------------------------------------------------------
    n_slots: int = 8000
    warmup_s: float = 10.0
    n_init_looks: int = 6
    init_look_gap: int = 50
    traffic_dt: float = 1e-3
    # --- LOI -------------------------------------------------------------
    T_min: int = 13
    T_max: int = 2000
    n_T: int = 70
    pi_min: float = 0.2
    eps_B: float = 0.02
    n_gh: int = 15
    exit_prior_slots: float = 1000.0   # prior mean outage duration (exit-hazard estimate)
    p_tx: float = 0.05                 # minimum predicted success for a transmission
    loi_age_step: int = 8              # coarse age grid for the look-time rendering

    # derived ---------------------------------------------------------------
    @property
    def lam(self):
        return C0 / (self.fc_ghz * 1e9)

    @property
    def noise_dbm(self):
        return -174.0 + 10 * np.log10(self.bw_hz) + self.nf_db

    @property
    def rsu_y(self):
        return -self.rsu_offset

    @property
    def kappa(self):
        return np.pi ** 2 * (self.n_ant ** 2 - 1) / 12.0

    def lanes(self):
        """Lane centres (y) and travel direction (+1 / -1)."""
        w = self.lane_w
        return np.array([0.5 * w, 1.5 * w, 2.5 * w, 3.5 * w]), np.array([1, 1, -1, -1])

    def T_grid(self):
        g = np.unique(np.round(np.geomspace(self.T_min, self.T_max, self.n_T)).astype(int))
        return g[g > self.tau]


def with_(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)
