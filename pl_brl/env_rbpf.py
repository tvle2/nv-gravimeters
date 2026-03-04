# pl_brl/env_rbpf.py
"""
Paper-faithful RBPF environment for the levitated-nanoparticle gravimeter.

Latent parameters (true + belief):
  - g : gravitational acceleration (constant over an episode)
  - dt: control-time error δt from trap-frequency noise (constant over an episode)
  - phi_off: unknown phase offset aggregating ζ(δt)+φ0(α0,δt) (slowly drifting)

Visibility model (Supplement Eq. S65):
  A = exp(-(16 η cos(ωδt/4) sin^2(3ωδt/4))^2 / 2),
  η(B') = γ_e * (B' [T/m]) * y0 / ω,
  y0 = sqrt(ħ / (2 M ω)).

Phase model (Main text Eq. (3)):
  ΔΦ = (2η/y0) g T^2 + 16π η_g η.

With physical units restored, η_g = M g y0 / (ħ ω),
and the 16π η_g η term is also linear in g and simplifies to:
  16π η_g η = (8π γ_e B' / ω^3) * g   (ħ cancels, M cancels).

So we implement:
  ΔΦ(g;T,B') = k_g(T,B') * g
  k_g(T,B')  = (2γ/ω) B' T^2 + (8π γ / ω^3) B'.

Measurement model:
  P_plus = 0.5 * (1 + A(η,dt) * cos(ΔΦ + phi_off + varphi))

Optional MFG amplitude noise (Supplement discussion):
  B' -> B'(1+ε), ε ~ N(0, sigma_B_rel).
We marginalize this in the filter likelihood via:
  E[cos((1+ε)*φ)] = exp(-0.5*(sigma_B_rel*φ)^2) * cos(φ),
which acts as an additional g-dependent dephasing factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


# ----------------------------- helpers -----------------------------


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling indices for particle filtering."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    idx = np.searchsorted(cumsum, positions, side="right")
    return np.clip(idx, 0, n - 1)


def logsumexp_1d(a: np.ndarray) -> float:
    a = np.nan_to_num(a, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    m = float(np.max(a))
    if not np.isfinite(m):
        return float("-inf")
    return m + float(np.log(np.sum(np.exp(a - m)) + 1e-300))


def logsumexp_rowwise(a2: np.ndarray) -> np.ndarray:
    a2 = np.nan_to_num(a2, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    m = np.max(a2, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a2 - m), axis=1, keepdims=True)
    out = m + np.log(s + 1e-300)
    return out.squeeze(1)


# ----------------------------- configs -----------------------------


@dataclass
class Prior:
    """Prior ranges for latent parameters."""
    g_range: Tuple[float, float] = (9.7639, 9.8337)  # m/s^2
    dt_range: Tuple[float, float] = (-1e-6, 1e-6)    # seconds
    phi_range: Tuple[float, float] = (-np.pi, np.pi) # rad


@dataclass
class ActionGrid:
    """Discrete action grid for controls (B' in kT/m, T in seconds)."""
    Bp_vals_kTm: np.ndarray  # [nB]
    T_vals_s: np.ndarray     # [nT]


@dataclass
class ModelConfig:
    """
    Physics model parameters.

    k_g(T,B') [rad / (m/s^2)]:
      k_g = (2γ/ω) B' T^2 + (8π γ / ω^3) B'

    Visibility (Supplement Eq. S65):
      A(η,dt) = exp(-(16 η cos(ωdt/4) sin^2(3ωdt/4))^2 / 2)

    η(B') = γ * B' * y0 / ω,  y0 = sqrt(ħ / (2 M ω))
    """
    omega_rad_s: float = 2.0 * np.pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * np.pi * 28e9
    kT_to_T: float = 1e3

    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34

    eps_prob: float = 1e-6
    sigma_phi_drift: float = 0.03
    sigma_dt_drift: float = 0.0

    # MFG amplitude noise: B' -> B'(1+eps), eps ~ N(0, sigma_B_rel)
    sigma_B_rel: float = 0.0001
    B_rel_clip: float = 0.5

    def y0_m(self) -> float:
        return float(np.sqrt(self.hbar_J_s / (2.0 * self.mass_kg * self.omega_rad_s)))

    def Bp_T_per_m(self, Bp_kTm: float) -> float:
        return float(Bp_kTm) * float(self.kT_to_T)

    def eta(self, Bp_kTm: float) -> float:
        Bp = self.Bp_T_per_m(Bp_kTm)
        return float(self.gamma_e_rad_s_T * Bp * self.y0_m() / self.omega_rad_s)

    def k_g(self, T_s: float, Bp_kTm: float) -> float:
        Bp = self.Bp_T_per_m(Bp_kTm)
        w = self.omega_rad_s
        gma = self.gamma_e_rad_s_T
        term_T2 = (2.0 * gma / w) * Bp * (T_s ** 2)
        term_16pi = (8.0 * np.pi * gma / (w ** 3)) * Bp
        return float(term_T2 + term_16pi)

    def delta_phi_scalar(self, g: float, T_s: float, Bp_kTm: float) -> float:
        return float(self.k_g(T_s, Bp_kTm) * float(g))

    def dg_period_2pi(self, T_s: float, Bp_kTm: float) -> float:
        k = self.k_g(T_s, Bp_kTm)
        return float(2.0 * np.pi / max(k, 1e-30))

    def sample_Bp_effective_kTm(self, rng: np.random.Generator, Bp_kTm: float) -> float:
        """Shot-to-shot multiplicative MFG noise (kept positive)."""
        if self.sigma_B_rel <= 0.0:
            return float(Bp_kTm)
        eps = float(rng.normal(0.0, self.sigma_B_rel))
        eps = float(np.clip(eps, -self.B_rel_clip, self.B_rel_clip))
        return float(max(1e-9, (1.0 + eps) * float(Bp_kTm)))

    def visibility_A(self, eta: float, dt_s: np.ndarray) -> np.ndarray:
        x = self.omega_rad_s * dt_s
        f = 16.0 * eta * np.cos(x / 4.0) * (np.sin(3.0 * x / 4.0) ** 2)
        return np.exp(-(f ** 2) / 2.0)

    def dA_ddt(self, eta: float, dt_s: np.ndarray) -> np.ndarray:
        w = self.omega_rad_s
        x = w * dt_s

        c1 = np.cos(x / 4.0)
        s1 = np.sin(x / 4.0)
        s3 = np.sin(3.0 * x / 4.0)
        c3 = np.cos(3.0 * x / 4.0)

        f = 16.0 * eta * c1 * (s3 ** 2)
        A = np.exp(-(f ** 2) / 2.0)

        df_dx = 16.0 * eta * (-(s1 / 4.0) * (s3 ** 2) + c1 * (3.0 / 2.0) * s3 * c3)
        df_dt = df_dx * w
        return A * (-f) * df_dt


@dataclass
class EnvConfig:
    episode_len: int = 60

    # RBPF sizes
    n_g_grid: int = 2048
    n_nuisance: int = 256
    resample_ess_frac: float = 0.25

    # features
    g_hist_bins: int = 30

    # costs
    lambda_T: float = 0.0
    lambda_B: float = 0.0

    lambda_cycle: float = 0.0

    # probe schedule + reward shaping
    probe_every: int = 4
    lambda_dt: float = 0.10

    # hard safety constraints used by the policy mask
    alias_sigma_mult: float = 3.0
    alias_frac: float = 0.75
    theta_probe_max: float = np.pi / 4.0

    wrap_max: float = 5.0


# ----------------------------- belief (PROPER RBPF) -----------------------------


class RBPFGravBelief:
    """
    Proper RBPF:
      - particles i carry (dt_i, phi_i, weight w_i)
      - each particle has its own conditional g-grid posterior logw_g[i, :]
      - particle weights update by the particle-specific marginal likelihood
    """

    def __init__(self, prior: Prior, model: ModelConfig, cfg: EnvConfig, seed: int = 0):
        self.prior = prior
        self.model = model
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.g_grid = np.linspace(prior.g_range[0], prior.g_range[1], cfg.n_g_grid, dtype=np.float64)

        # nuisance particles
        self.dt = np.zeros(cfg.n_nuisance, dtype=np.float64)
        self.phi = np.zeros(cfg.n_nuisance, dtype=np.float64)

        # particle log-weights
        self.logw_n = np.full(cfg.n_nuisance, -np.log(cfg.n_nuisance), dtype=np.float64)

        # conditional g log-weights per particle: shape [Nn, Ng]
        self.logw_g = np.full((cfg.n_nuisance, cfg.n_g_grid), -np.log(cfg.n_g_grid), dtype=np.float64)

        self.reset()

    @property
    def w_n(self) -> np.ndarray:
        logw_n = np.nan_to_num(self.logw_n, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        lw = logw_n - np.max(logw_n)
        w = np.exp(lw)
        s = float(np.sum(w))
        if s <= 0.0 or (not np.isfinite(s)):
            return np.full_like(w, 1.0 / len(w))
        return w / s

    def ess_n(self) -> float:
        w = self.w_n
        return 1.0 / float(np.sum(w ** 2))

    def reset(self) -> None:
        self.dt = self.rng.uniform(*self.prior.dt_range, size=self.cfg.n_nuisance)
        self.phi = self.rng.uniform(*self.prior.phi_range, size=self.cfg.n_nuisance)

        self.logw_n.fill(-np.log(self.cfg.n_nuisance))
        self.logw_g.fill(-np.log(self.cfg.n_g_grid))

    def predict_drift(self) -> None:
        m = self.model
        if m.sigma_dt_drift > 0.0:
            self.dt = self.dt + self.rng.normal(0.0, m.sigma_dt_drift, size=self.cfg.n_nuisance)
        self.phi = wrap_to_pi(self.phi + self.rng.normal(0.0, m.sigma_phi_drift, size=self.cfg.n_nuisance))

    def w_g_marginal(self) -> np.ndarray:
        """
        Stable marginal mixture:
            p(g) = sum_i w_n(i) * p_i(g)

        where logw_n and each row logw_g[i,:] represent the particle's conditional posterior over g.
        Returns normalized float64 array of shape [Ng].
        """
        # sanitize
        logw_n = np.nan_to_num(self.logw_n, nan=-np.inf, posinf=0.0, neginf=-np.inf)      # [Nn]
        logw_g = np.nan_to_num(self.logw_g, nan=-np.inf, posinf=0.0, neginf=-np.inf)      # [Nn,Ng]

        # stabilize nuisance weights
        # (note: logw_n should be normalized already, but we re-normalize safely)
        Z_n = logsumexp_1d(logw_n)
        if not np.isfinite(Z_n):
            # fallback to uniform over g
            return np.full(self.cfg.n_g_grid, 1.0 / self.cfg.n_g_grid, dtype=np.float64)
        logw_n = logw_n - Z_n  # now log w_n

        # log p(g_j) = logsumexp_i [ log w_n(i) + log p_i(g_j) ]
        # We want logsumexp over nuisance dimension (axis=0 of [Nn,Ng])
        a = logw_n[:, None] + logw_g  # [Nn, Ng]

        # stable logsumexp over axis=0 without extra helpers
        m = np.max(a, axis=0)  # [Ng]
        m = np.where(np.isfinite(m), m, 0.0)
        s = np.sum(np.exp(a - m[None, :]), axis=0)  # [Ng]
        logp = m + np.log(s + 1e-300)               # [Ng]

        # normalize over g
        Zg = logsumexp_1d(logp)
        if not np.isfinite(Zg):
            return np.full(self.cfg.n_g_grid, 1.0 / self.cfg.n_g_grid, dtype=np.float64)

        logp = logp - Zg
        p = np.exp(logp)
        p = p / (np.sum(p) + 1e-300)
        return p.astype(np.float64)

    # ---- phase-lock using mixture moment E[e^{i2(k g + phi)}] ----
    def _C2(self, T_s: float, Bp_kTm: float) -> complex:
        """
        Mixture moment used for phase-lock:
            C2 = sum_i w_n(i) * exp(i*2*phi_i) * E_{p_i(g)}[exp(i*2*k*g)]
        """
        # ----- nuisance weights wn (safe) -----
        logw_n = np.nan_to_num(self.logw_n, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        Z_n = logsumexp_1d(logw_n)
        if not np.isfinite(Z_n):
            wn = np.full(self.cfg.n_nuisance, 1.0 / self.cfg.n_nuisance, dtype=np.float64)
        else:
            wn = np.exp(logw_n - Z_n)
            wn = wn / (np.sum(wn) + 1e-300)

        # ----- g-phase vector (bounded) -----
        k = float(self.model.k_g(T_s, Bp_kTm))
        if not np.isfinite(k):
            return 0.0 + 0.0j

        phase2 = np.exp(1j * np.remainder(2.0 * k * self.g_grid, 2.0 * np.pi))  # [Ng] complex128

        # ----- conditional g weights per nuisance particle (robust) -----
        logw_g = np.nan_to_num(self.logw_g, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)  # [Nn,Ng]

        # clamp to avoid exp overflow if logw_g ever goes positive due to corruption
        logw_g = np.minimum(logw_g, 0.0)

        # detect rows that are completely invalid (all -inf)
        row_max = np.max(logw_g, axis=1)                # [Nn]
        bad = ~np.isfinite(row_max)                     # True if row is all -inf (or all nan pre-sanitize)

        # row-wise log-normalization: logw_g_norm = logw_g - logsumexp(logw_g)
        row_Z = logsumexp_rowwise(logw_g)               # [Nn]
        row_Z = np.nan_to_num(row_Z, nan=0.0, posinf=0.0, neginf=0.0)

        logw_g_norm = logw_g - row_Z[:, None]

        # for bad rows, force uniform
        if np.any(bad):
            logw_g_norm[bad, :] = -np.log(self.cfg.n_g_grid)

        # exponentiate -> weights
        wg_i = np.exp(logw_g_norm)                      # [Nn,Ng] float64

        # final sanitize + renormalize rows (important!)
        wg_i = np.nan_to_num(wg_i, nan=0.0, posinf=0.0, neginf=0.0)

        row_s = np.sum(wg_i, axis=1, keepdims=True)     # [Nn,1]
        zero_rows = (row_s[:, 0] <= 0.0) | (~np.isfinite(row_s[:, 0]))
        if np.any(zero_rows):
            wg_i[zero_rows, :] = 1.0 / self.cfg.n_g_grid
            row_s = np.sum(wg_i, axis=1, keepdims=True)

        wg_i = wg_i / (row_s + 1e-300)

        # ----- per-particle expectation E_{p_i(g)}[phase2] -----
        # using elementwise sum instead of BLAS matmul to avoid warnings
        C2_g_i = np.sum(wg_i * phase2[None, :], axis=1)  # [Nn] complex128

        # ----- include nuisance phase phi and mix over wn -----
        phi = np.nan_to_num(self.phi, nan=0.0, posinf=0.0, neginf=0.0)
        C2 = np.sum(wn * np.exp(1j * 2.0 * phi) * C2_g_i)
        return complex(C2)

    def phase_lock_quadrature(self, T_s: float, Bp_kTm: float) -> float:
        C2 = self._C2(T_s, Bp_kTm)
        if abs(C2) < 1e-14:
            return 0.0
        varphi = 0.5 * (np.pi - np.angle(C2))
        return float(wrap_to_pi(np.array([varphi]))[0])

    def phase_lock_extrema(self, T_s: float, Bp_kTm: float) -> float:
        C2 = self._C2(T_s, Bp_kTm)
        if abs(C2) < 1e-14:
            return 0.0
        varphi = -0.5 * np.angle(C2)
        return float(wrap_to_pi(np.array([varphi]))[0])

    # ---- RBPF update ----
    def update(self, outcome_plus: int, T_s: float, Bp_kTm: float, varphi: float) -> None:
        """
        For each particle i:
          p_i(g) ∝ p_i(g) * like_i(g)
          evidence Z_i = ∫ p_i(g) like_i(g) dg  (discrete sum)
        Particle weights:
          w_i ∝ w_i * Z_i
        """
        m = self.model
        eps = m.eps_prob

        k = m.k_g(T_s, Bp_kTm)
        phi_g = k * self.g_grid                                     # [Ng] rad
        theta_g = np.remainder(phi_g, 2.0 * np.pi)                  # [Ng]

        eta = m.eta(Bp_kTm)
        A_i = m.visibility_A(eta, self.dt)                          # [Nn]

        # Marginalize shot-to-shot MFG amplitude noise as extra dephasing
        if m.sigma_B_rel > 0.0:
            deph = np.exp(-0.5 * (m.sigma_B_rel * phi_g) ** 2)      # [Ng]
            A_ig = A_i[:, None] * deph[None, :]                     # [Nn,Ng]
        else:
            A_ig = A_i[:, None]

        theta = theta_g[None, :] + self.phi[:, None] + float(varphi)
        p_plus = 0.5 * (1.0 + A_ig * np.cos(theta))
        p_plus = np.clip(p_plus, eps, 1.0 - eps)

        like = p_plus if outcome_plus == 1 else (1.0 - p_plus)
        log_like = np.log(like + 1e-300)

        # per-particle evidence in log-space
        logZ = logsumexp_rowwise(self.logw_g + log_like)  # [Nn]

        # update conditional g posteriors and renormalize per particle
        self.logw_g = self.logw_g + log_like - logZ[:, None]

        # update particle weights
        self.logw_n = self.logw_n + logZ
        self.logw_n = self.logw_n - logsumexp_1d(self.logw_n)

        # resample if needed
        if self.ess_n() < self.cfg.resample_ess_frac * self.cfg.n_nuisance:
            wn = self.w_n
            idx = systematic_resample(wn, self.rng)
            self.dt = self.dt[idx]
            self.phi = self.phi[idx]
            self.logw_g = self.logw_g[idx, :]
            self.logw_n.fill(-np.log(self.cfg.n_nuisance))
        # ---- numerical safety guard (put this at the very end) ----
        if (not np.isfinite(self.logw_g).all()) or (not np.isfinite(self.logw_n).all()):
            self.reset()

    def stats(self) -> Dict[str, float]:
        wg = self.w_g_marginal()
        mu_g = float(np.sum(wg * self.g_grid))
        var_g = float(np.sum(wg * (self.g_grid - mu_g) ** 2))

        wn = self.w_n
        mu_dt = float(np.sum(wn * self.dt))
        var_dt = float(np.sum(wn * (self.dt - mu_dt) ** 2))

        c = float(np.sum(wn * np.cos(self.phi)))
        s = float(np.sum(wn * np.sin(self.phi)))
        mu_phi = float(np.arctan2(s, c))
        R_phi = float(np.sqrt(c * c + s * s))

        return {
            "mu_g": mu_g, "std_g": float(np.sqrt(max(var_g, 0.0))),
            "mu_dt": mu_dt, "std_dt": float(np.sqrt(max(var_dt, 0.0))),
            "mu_phi": mu_phi, "R_phi": R_phi,
        }


# ----------------------------- environment -----------------------------


class GravimeterEnvPaperRBPF:
    """
    Observation vector (float32):
      [ hist_g (bins),
        mu_g, std_g,
        mu_dt, std_dt,
        cos(mu_phi), sin(mu_phi), R_phi,
        progress,
        probe_now ]
    """

    def __init__(self, prior: Prior, grid: ActionGrid, model: ModelConfig, cfg: EnvConfig, seed: int = 0):
        self.prior = prior
        self.grid = grid
        self.model = model
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.belief = RBPFGravBelief(prior=prior, model=model, cfg=cfg, seed=seed + 123)

        self.k = 0
        self._prev_var_g = 0.0
        self._prev_var_dt = 0.0

        self._sample_true()
        self.fallback_joint_idx = int(np.argmax([
            self.model.dg_period_2pi(float(T), float(Bp))
            for T in self.grid.T_vals_s
            for Bp in self.grid.Bp_vals_kTm
        ]))
        self.fallback_aT = self.fallback_joint_idx // self.nB
        self.fallback_aB = self.fallback_joint_idx % self.nB

    @property
    def nT(self) -> int:
        return int(len(self.grid.T_vals_s))

    @property
    def nB(self) -> int:
        return int(len(self.grid.Bp_vals_kTm))

    @property
    def obs_dim(self) -> int:
        return int(self.cfg.g_hist_bins) + 9

    def _is_probe_now(self) -> bool:
        return (self.cfg.probe_every > 0) and ((self.k % self.cfg.probe_every) == 0) and (self.k > 0)

    def _sample_true(self) -> None:
        self.true_g = float(self.rng.uniform(*self.prior.g_range))
        self.true_dt = float(self.rng.uniform(*self.prior.dt_range))
        self.true_phi = float(self.rng.uniform(*self.prior.phi_range))

    def _true_drift(self) -> None:
        m = self.model
        if m.sigma_dt_drift > 0.0:
            self.true_dt = float(self.true_dt + self.rng.normal(0.0, m.sigma_dt_drift))
        self.true_phi = float(wrap_to_pi(np.array([self.true_phi + self.rng.normal(0.0, m.sigma_phi_drift)]))[0])

    def _features(self) -> np.ndarray:
        bins = int(self.cfg.g_hist_bins)
        gmin, gmax = self.prior.g_range

        wg = self.belief.w_g_marginal().astype(np.float32)
        g_grid = self.belief.g_grid

        edges = np.linspace(gmin, gmax, bins + 1)
        idx = np.searchsorted(edges, g_grid, side="right") - 1
        idx = np.clip(idx, 0, bins - 1)

        hist = np.zeros(bins, dtype=np.float32)
        np.add.at(hist, idx, wg)

        st = self.belief.stats()
        mu_phi = st["mu_phi"]
        probe = 1.0 if self._is_probe_now() else 0.0

        feat = np.concatenate(
            [
                hist,
                np.array(
                    [
                        st["mu_g"], st["std_g"],
                        st["mu_dt"], st["std_dt"],
                        np.cos(mu_phi), np.sin(mu_phi), st["R_phi"],
                        self.k / max(1, (self.cfg.episode_len - 1)),
                        probe,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        assert feat.shape[0] == self.obs_dim, f"obs_dim mismatch: feat={feat.shape[0]} vs obs_dim={self.obs_dim}"
        return feat

    def _entropy_g(self) -> float:
        p = self.belief.w_g_marginal().astype(np.float64)
        p = p / (np.sum(p) + 1e-300)
        return float(-np.sum(p * np.log(p + 1e-300)))
    
    def reset(self) -> np.ndarray:
        self._sample_true()
        self.belief.reset()
        self.k = 0

        st0 = self.belief.stats()
        self._prev_ent_g = self._entropy_g()
        self._prev_var_g = float(st0["std_g"] ** 2)
        self._prev_var_dt = float(st0["std_dt"] ** 2)

        return self._features()

    def step(self, aT: int, aB: int):
        T_s = float(self.grid.T_vals_s[int(aT)])
        Bp_kTm = float(self.grid.Bp_vals_kTm[int(aB)])

        k_g_nom = float(self.model.k_g(T_s, Bp_kTm))
        dg2pi_nom = float(self.model.dg_period_2pi(T_s, Bp_kTm))

        # probe_now = self._is_probe_now()

        # # phase-lock choice (fast classical feedback)
        # if probe_now:
        #     varphi = self.belief.phase_lock_extrema(T_s, Bp_kTm)
        # else:
        #     varphi = self.belief.phase_lock_quadrature(T_s, Bp_kTm)
        probe_now = False  # keep for logging if you want
        varphi = self.belief.phase_lock_quadrature(T_s, Bp_kTm)

        # true shot uses noisy applied gradient
        Bp_true_kTm = self.model.sample_Bp_effective_kTm(self.rng, Bp_kTm)

        # true visibility per S65
        eta = self.model.eta(Bp_true_kTm)
        A_true = float(self.model.visibility_A(eta, np.array([self.true_dt]))[0])

        theta_true = self.model.delta_phi_scalar(self.true_g, T_s, Bp_true_kTm) + self.true_phi + varphi
        p_plus = 0.5 * (1.0 + A_true * np.cos(theta_true))
        p_plus = float(np.clip(p_plus, self.model.eps_prob, 1.0 - self.model.eps_prob))
        outcome_plus = 1 if (self.rng.random() < p_plus) else 0

        # drift
        self._true_drift()
        self.belief.predict_drift()

        # update belief (filter assumes nominal B' but marginalizes noise)
        self.belief.update(outcome_plus, T_s, Bp_kTm, varphi)

        # reward: per-shot log-variance reduction in g (+ optional dt on probes)
        st = self.belief.stats()

        ent_g = self._entropy_g()
        r_ent = self._prev_ent_g - ent_g
        self._prev_ent_g = ent_g

        var_g = float(st["std_g"] ** 2)
        var_dt = float(st["std_dt"] ** 2)
        eps = 1e-14

        r_g = 0.5 * (np.log(self._prev_var_g + eps) - np.log(var_g + eps))
        self._prev_var_g = var_g

        # reward = float(r_g)
        # use entropy reduction as the main signal (alias-aware)
        reward = float(r_ent)

        if probe_now and self.cfg.lambda_dt > 0.0:
            r_dt = 0.5 * (np.log(self._prev_var_dt + eps) - np.log(var_dt + eps))
            reward += float(self.cfg.lambda_dt * r_dt)
        self._prev_var_dt = var_dt

        reward -= float(self.cfg.lambda_T * T_s + self.cfg.lambda_B * abs(Bp_kTm))

        # tau = 2π/ω, cycle_time = 7τ/2 + 2T
        if self.cfg.lambda_cycle > 0.0:
            tau = 2.0 * np.pi / float(self.model.omega_rad_s)
            cycle_time = 3.5 * tau + 2.0 * T_s
            reward -= float(self.cfg.lambda_cycle * cycle_time)

        self.k += 1
        done = self.k >= self.cfg.episode_len

        obs = self._features()
        info = {
            "k": self.k,
            "T_s": T_s,
            "Bp_kTm": Bp_kTm,
            "Bp_true_kTm": Bp_true_kTm,
            "varphi": varphi,
            "p_plus": p_plus,
            "outcome_plus": outcome_plus,
            "true_g": self.true_g,
            "true_dt": self.true_dt,
            "true_phi": self.true_phi,
            "A_true": A_true,
            "post_mu_g": st["mu_g"],
            "post_std_g": st["std_g"],
            "post_mu_dt": st["mu_dt"],
            "post_std_dt": st["std_dt"],
            "post_mu_phi": st["mu_phi"],
            "post_R_phi": st["R_phi"],
            "probe_now": probe_now,
            "r_g": float(r_g),
            "k_g_nom": k_g_nom,
            "dg2pi_nom": dg2pi_nom
        }
        return obs, float(reward), bool(done), info
