from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


ArrayF = np.ndarray


# ============================================================
# Configs and action grid
# ============================================================


@dataclass(frozen=True)
class PriorConfig:
    g_range: tuple[float, float] = (9.7639, 9.8337)


@dataclass(frozen=True)
class ActionGrid:
    T_vals_s: ArrayF
    Bp_vals_kTm: ArrayF

    @property
    def nT(self) -> int:
        return int(self.T_vals_s.size)

    @property
    def nB(self) -> int:
        return int(self.Bp_vals_kTm.size)

    @property
    def n_actions(self) -> int:
        return int(self.nT * self.nB)

    def encode(self, aT: int, aB: int) -> int:
        return int(aT) * self.nB + int(aB)

    def decode(self, idx: int) -> tuple[int, int]:
        idx = int(idx)
        return idx // self.nB, idx % self.nB


def make_paper_scale_grid(
    n_T: int = 21,
    n_B: int = 21,
    T_min_s: float = 50e-6,
    T_max_s: float = 1.0e-3,
    B_min_kTm: float = 0.5,
    B_max_kTm: float = 50.0,
    log_spacing: bool = True,
) -> ActionGrid:
    if log_spacing:
        T_vals = np.geomspace(T_min_s, T_max_s, n_T, dtype=np.float64)
        B_vals = np.geomspace(B_min_kTm, B_max_kTm, n_B, dtype=np.float64)
    else:
        T_vals = np.linspace(T_min_s, T_max_s, n_T, dtype=np.float64)
        B_vals = np.linspace(B_min_kTm, B_max_kTm, n_B, dtype=np.float64)
    return ActionGrid(T_vals_s=T_vals, Bp_vals_kTm=B_vals)


@dataclass(frozen=True)
class NoiseConfig:
    """
    In this hierarchical version:

    - sigma_omega_rel controls trap-frequency-induced visibility reduction
    - mfg_rel_noise_bound defines the support of the stable unknown hardware bias:
          true B' = B'_nom * (1 + eps_true_global)
      where eps_true_global is fixed over the whole evaluation run

    So mfg_rel_noise_bound is NOT shot-to-shot noise here.
    It is the prior support for the global unknown epsilon.
    """

    sigma_omega_rel: float = 0.0
    trap_visibility_mode: Literal["none", "small_noise_avg", "exact_single_delta"] = "small_noise_avg"
    mfg_rel_noise_bound: float = 0.0
    T2_spin_s: float | None = None


@dataclass(frozen=True)
class PlannerConfig:
    """
    objective:
        "variance_reduction" or "information_gain"

    phase_mode:
        "analytic_quadrature" -> uses circular mean of phase under joint posterior
        "grid_expected_fi"    -> searches phase grid by expected FI wrt g

    Alias gate:
        reject if k_max * h_q > alias_halfwidth_max_rad

        where h_q is the credible half-width of the g-marginal posterior.
    """

    objective: Literal["variance_reduction", "information_gain"] = "variance_reduction"
    phase_mode: Literal["analytic_quadrature", "grid_expected_fi"] = "analytic_quadrature"
    phase_grid_size: int = 181

    # kept for compatibility; no longer needed when epsilon is explicitly modeled
    robust_mfg_mode: Literal["nominal", "average3", "worst3"] = "nominal"

    alias_halfwidth_max_rad: float | None = None
    alias_credible_mass: float = 0.90

    cycle_penalty: float = 0.0
    mfg_penalty: float = 0.0
    min_visibility: float = 0.0


@dataclass(frozen=True)
class EnvConfig:
    episode_len: int = 64
    n_g_grid: int = 512
    n_eps_grid: int = 31


# ============================================================
# Physical model
# ============================================================


@dataclass(frozen=True)
class GravimeterModel:
    """
    Paper-faithful effective model:

        P(+|g) = 1/2 [1 + A cos(DeltaPhi(g) + phi_MW)]

    with
        DeltaPhi(g; T, B') = k_g(T, B') * g

    and
        k_g(T, B') = (2 gamma_e / omega) B' T^2 + (8 pi gamma_e / omega^3) B'
    """

    omega_rad_s: float = 2.0 * np.pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * np.pi * 28e9
    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34
    kT_to_T: float = 1e3
    eps_prob: float = 1e-9

    def validate(self) -> None:
        if self.omega_rad_s <= 0.0:
            raise ValueError("omega_rad_s must be positive")
        if self.gamma_e_rad_s_T <= 0.0:
            raise ValueError("gamma_e_rad_s_T must be positive")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.hbar_J_s <= 0.0:
            raise ValueError("hbar_J_s must be positive")

    @property
    def tau_s(self) -> float:
        return float(2.0 * np.pi / self.omega_rad_s)

    def cycle_time_s(self, T_s: float) -> float:
        return float(3.5 * self.tau_s + 2.0 * T_s)

    def y0_m(self) -> float:
        return float(np.sqrt(self.hbar_J_s / (2.0 * self.mass_kg * self.omega_rad_s)))

    def Bp_T_per_m(self, Bp_kTm: float | ArrayF) -> float | ArrayF:
        return np.asarray(Bp_kTm, dtype=np.float64) * self.kT_to_T

    def eta(self, Bp_kTm: float | ArrayF) -> float | ArrayF:
        Bp = self.Bp_T_per_m(Bp_kTm)
        return self.gamma_e_rad_s_T * Bp * self.y0_m() / self.omega_rad_s

    def k_g(self, T_s: float, Bp_kTm: float) -> float:
        Bp = float(self.Bp_T_per_m(Bp_kTm))
        w = self.omega_rad_s
        gma = self.gamma_e_rad_s_T
        return float((2.0 * gma / w) * Bp * (T_s ** 2) + (8.0 * np.pi * gma / (w ** 3)) * Bp)

    def phase_total(self, g_m_s2: float | ArrayF, T_s: float, Bp_kTm: float) -> float | ArrayF:
        return self.k_g(T_s, Bp_kTm) * np.asarray(g_m_s2, dtype=np.float64)

    def delta_g_period_2pi(self, T_s: float, Bp_kTm: float) -> float:
        k = self.k_g(T_s, Bp_kTm)
        return float(2.0 * np.pi / max(k, 1e-30))

    # ---------------- visibility ----------------

    def visibility_exact_from_delta_omega(self, Bp_kTm: float, delta_omega_rad_s: float) -> float:
        eta = float(self.eta(Bp_kTm))
        delta_t = -self.tau_s * float(delta_omega_rad_s) / self.omega_rad_s
        x = self.omega_rad_s * delta_t
        A = np.exp(-0.5 * (16.0 * eta * np.cos(x / 4.0) * (np.sin(3.0 * x / 4.0) ** 2)) ** 2)
        return float(np.clip(A, 0.0, 1.0))

    def visibility_avg_small_noise(self, Bp_kTm: float, sigma_omega_rel: float) -> float:
        eta = float(self.eta(Bp_kTm))
        correction = 1944.0 * (np.pi ** 4) * (eta ** 2) * (sigma_omega_rel ** 4)
        return float(np.clip(1.0 - correction, 0.0, 1.0))

    def planning_visibility(self, T_s: float, Bp_kTm: float, noise: NoiseConfig) -> float:
        if noise.trap_visibility_mode == "none":
            A = 1.0
        else:
            A = self.visibility_avg_small_noise(Bp_kTm, noise.sigma_omega_rel)

        if noise.T2_spin_s is not None and noise.T2_spin_s > 0.0:
            A *= float(np.exp(-self.cycle_time_s(T_s) / noise.T2_spin_s))
        return float(np.clip(A, 0.0, 1.0))

    def shot_visibility(self, T_s: float, Bp_kTm: float, noise: NoiseConfig, rng: np.random.Generator) -> float:
        if noise.trap_visibility_mode == "none":
            A = 1.0
        elif noise.trap_visibility_mode == "small_noise_avg":
            A = self.visibility_avg_small_noise(Bp_kTm, noise.sigma_omega_rel)
        elif noise.trap_visibility_mode == "exact_single_delta":
            delta_omega = rng.normal(0.0, noise.sigma_omega_rel * self.omega_rad_s)
            A = self.visibility_exact_from_delta_omega(Bp_kTm, delta_omega)
        else:
            raise ValueError(f"Unknown trap_visibility_mode={noise.trap_visibility_mode}")

        if noise.T2_spin_s is not None and noise.T2_spin_s > 0.0:
            A *= float(np.exp(-self.cycle_time_s(T_s) / noise.T2_spin_s))
        return float(np.clip(A, 0.0, 1.0))

    # ---------------- readout ----------------

    def prob_plus(
        self,
        g_m_s2: float | ArrayF,
        T_s: float,
        Bp_kTm: float,
        mw_phase_rad: float,
        visibility: float,
    ) -> float | ArrayF:
        theta = self.phase_total(g_m_s2, T_s, Bp_kTm) + float(mw_phase_rad)
        p = 0.5 * (1.0 + float(visibility) * np.cos(theta))
        return np.clip(p, self.eps_prob, 1.0 - self.eps_prob)

    def fisher_information_g(
        self,
        g_m_s2: float | ArrayF,
        T_s: float,
        Bp_kTm: float,
        mw_phase_rad: float,
        visibility: float,
    ) -> float | ArrayF:
        k = self.k_g(T_s, Bp_kTm)
        theta = self.phase_total(g_m_s2, T_s, Bp_kTm) + float(mw_phase_rad)
        num = (visibility ** 2) * (np.sin(theta) ** 2)
        den = 1.0 - (visibility ** 2) * (np.cos(theta) ** 2)
        return (num / np.maximum(den, 1e-18)) * (k ** 2)


# ============================================================
# Joint belief p(g, eps)
# ============================================================


class JointGravityEpsBelief:
    """
    Joint posterior over:
        g   : episode-specific gravity
        eps : global MFG scale bias

    eps is assumed fixed over the whole run, but unknown.
    """

    def __init__(
        self,
        prior: PriorConfig,
        noise: NoiseConfig,
        n_g_grid: int = 512,
        n_eps_grid: int = 31,
    ):
        if n_g_grid < 64:
            raise ValueError("n_g_grid must be >= 64")

        self.prior = prior
        self.noise = noise
        self.n_g_grid = int(n_g_grid)
        self.n_eps_grid = int(max(1, n_eps_grid))

        self.g_grid = np.linspace(prior.g_range[0], prior.g_range[1], self.n_g_grid, dtype=np.float64)
        self.g2_grid = self.g_grid ** 2

        b = float(max(noise.mfg_rel_noise_bound, 0.0))
        if b <= 0.0 or self.n_eps_grid == 1:
            self.eps_grid = np.array([0.0], dtype=np.float64)
        else:
            self.eps_grid = np.linspace(-b, +b, self.n_eps_grid, dtype=np.float64)
        self.eps2_grid = self.eps_grid ** 2

        self.n_eps_grid = int(self.eps_grid.size)
        self.logw = np.full((self.n_g_grid, self.n_eps_grid), -np.log(self.n_g_grid * self.n_eps_grid), dtype=np.float64)

    def copy(self) -> "JointGravityEpsBelief":
        out = JointGravityEpsBelief(
            prior=self.prior,
            noise=self.noise,
            n_g_grid=self.n_g_grid,
            n_eps_grid=self.n_eps_grid,
        )
        out.logw = self.logw.copy()
        return out

    # ---------------- normalization / marginals ----------------

    def normalize_(self) -> None:
        m = float(np.max(self.logw))
        z = m + np.log(np.sum(np.exp(self.logw - m)) + 1e-300)
        self.logw -= z

    @property
    def w_joint(self) -> np.ndarray:
        m = float(np.max(self.logw))
        x = np.exp(self.logw - m)
        return x / (np.sum(x) + 1e-300)

    def uniform_logw_eps(self) -> np.ndarray:
        return np.full(self.n_eps_grid, -np.log(self.n_eps_grid), dtype=np.float64)

    def reset_factorized(self, logw_eps_global: np.ndarray | None = None) -> None:
        if logw_eps_global is None:
            logw_eps_global = self.uniform_logw_eps()

        if logw_eps_global.shape != (self.n_eps_grid,):
            raise ValueError("logw_eps_global shape mismatch")

        logw_g_uniform = np.full(self.n_g_grid, -np.log(self.n_g_grid), dtype=np.float64)
        self.logw = logw_g_uniform[:, None] + logw_eps_global[None, :]
        self.normalize_()

    def w_g_marginal(self) -> np.ndarray:
        return np.sum(self.w_joint, axis=1)

    def w_eps_marginal(self) -> np.ndarray:
        return np.sum(self.w_joint, axis=0)

    def logw_eps_marginal(self) -> np.ndarray:
        w = self.w_eps_marginal()
        return np.log(w + 1e-300)

    # ---------------- weighted summaries ----------------

    @staticmethod
    def _weighted_quantile(grid: np.ndarray, w: np.ndarray, q: float) -> float:
        q = float(np.clip(q, 0.0, 1.0))
        cdf = np.cumsum(w / (np.sum(w) + 1e-300))
        return float(np.interp(q, cdf, grid))

    def mean_g(self) -> float:
        wg = self.w_g_marginal()
        return float(np.sum(wg * self.g_grid))

    def std_g(self) -> float:
        wg = self.w_g_marginal()
        mu = float(np.sum(wg * self.g_grid))
        ex2 = float(np.sum(wg * self.g2_grid))
        return float(np.sqrt(max(ex2 - mu * mu, 0.0)))

    def median_g(self) -> float:
        return self._weighted_quantile(self.g_grid, self.w_g_marginal(), 0.5)

    def map_g(self) -> float:
        wg = self.w_g_marginal()
        return float(self.g_grid[int(np.argmax(wg))])

    def entropy_g_nats(self) -> float:
        wg = self.w_g_marginal()
        return float(-np.sum(wg * np.log(wg + 1e-300)))

    def credible_interval_g(self, mass: float = 0.90) -> tuple[float, float]:
        mass = float(np.clip(mass, 1e-6, 0.999999))
        alpha = 0.5 * (1.0 - mass)
        wg = self.w_g_marginal()
        return (
            self._weighted_quantile(self.g_grid, wg, alpha),
            self._weighted_quantile(self.g_grid, wg, 1.0 - alpha),
        )

    def credible_halfwidth_g(self, mass: float = 0.90) -> float:
        q_lo, q_hi = self.credible_interval_g(mass=mass)
        return 0.5 * float(q_hi - q_lo)

    def peak_stats_g(self) -> tuple[float, float, float]:
        wg = self.w_g_marginal()
        idx1 = int(np.argmax(wg))
        p1 = float(wg[idx1])
        p2 = float(np.max(np.delete(wg, idx1))) if wg.size > 1 else 0.0
        return p1, p2, p1 - p2

    def top_peaks_g(self, k: int = 5) -> list[tuple[float, float]]:
        wg = self.w_g_marginal()
        K = int(min(max(1, k), wg.size))
        idx = np.argpartition(wg, -K)[-K:]
        idx = idx[np.argsort(wg[idx])[::-1]]
        return [(float(self.g_grid[i]), float(wg[i])) for i in idx]

    def mean_eps(self) -> float:
        we = self.w_eps_marginal()
        return float(np.sum(we * self.eps_grid))

    def std_eps(self) -> float:
        we = self.w_eps_marginal()
        mu = float(np.sum(we * self.eps_grid))
        ex2 = float(np.sum(we * self.eps2_grid))
        return float(np.sqrt(max(ex2 - mu * mu, 0.0)))

    def median_eps(self) -> float:
        return self._weighted_quantile(self.eps_grid, self.w_eps_marginal(), 0.5)

    def map_eps(self) -> float:
        we = self.w_eps_marginal()
        return float(self.eps_grid[int(np.argmax(we))])

    def entropy_joint_nats(self) -> float:
        wj = self.w_joint
        return float(-np.sum(wj * np.log(wj + 1e-300)))

    def credible_interval_eps(self, mass: float = 0.90) -> tuple[float, float]:
        mass = float(np.clip(mass, 1e-6, 0.999999))
        alpha = 0.5 * (1.0 - mass)
        we = self.w_eps_marginal()
        return (
            self._weighted_quantile(self.eps_grid, we, alpha),
            self._weighted_quantile(self.eps_grid, we, 1.0 - alpha),
        )

    # ---------------- posterior under hypothetical likelihood ----------------

    def posterior_stats_g_for_likelihood(self, like_ge: np.ndarray) -> tuple[float, float, float]:
        w = self.w_joint
        s = w * like_ge
        Z = float(np.sum(s) + 1e-300)
        wg_post = np.sum(s, axis=1) / Z
        mu = float(np.sum(wg_post * self.g_grid))
        ex2 = float(np.sum(wg_post * self.g2_grid))
        var = float(max(ex2 - mu * mu, 0.0))
        return Z, mu, var

    def posterior_entropy_g_for_likelihood(self, like_ge: np.ndarray) -> tuple[float, float]:
        w = self.w_joint
        s = w * like_ge
        Z = float(np.sum(s) + 1e-300)
        wg_post = np.sum(s, axis=1) / Z
        H = float(-np.sum(wg_post * np.log(wg_post + 1e-300)))
        return Z, H

    # ---------------- exact Bayesian update ----------------

    def update_from_outcome(
        self,
        model: GravimeterModel,
        T_s: float,
        Bp_nom_kTm: float,
        mw_phase_rad: float,
        noise: NoiseConfig,
        outcome_plus: int,
    ) -> None:
        """
        Exact joint update under stable latent epsilon:

            Bp_true = Bp_nom * (1 + eps)

        likelihood is computed on the full (g, eps) grid.
        """
        B_eff_eps = np.maximum(1e-12, float(Bp_nom_kTm) * (1.0 + self.eps_grid))
        k_eps = np.asarray([model.k_g(T_s, float(B)) for B in B_eff_eps], dtype=np.float64)
        A_eps = np.asarray([model.planning_visibility(T_s, float(B), noise) for B in B_eff_eps], dtype=np.float64)

        theta = self.g_grid[:, None] * k_eps[None, :] + float(mw_phase_rad)
        p_plus = 0.5 * (1.0 + A_eps[None, :] * np.cos(theta))
        p_plus = np.clip(p_plus, model.eps_prob, 1.0 - model.eps_prob)

        like = p_plus if int(outcome_plus) == 1 else (1.0 - p_plus)
        self.logw += np.log(like + 1e-300)
        self.normalize_()


# ============================================================
# Adaptive Bayesian controller
# ============================================================


class AdaptiveBayesController:
    def __init__(
        self,
        model: GravimeterModel,
        grid: ActionGrid,
        noise: NoiseConfig,
        cfg: PlannerConfig,
    ):
        self.model = model
        self.grid = grid
        self.noise = noise
        self.cfg = cfg

        T_list = []
        B_list = []
        k_list = []
        cyc_list = []
        A_list = []

        for T_s in self.grid.T_vals_s:
            for Bp_kTm in self.grid.Bp_vals_kTm:
                T = float(T_s)
                B = float(Bp_kTm)
                T_list.append(T)
                B_list.append(B)
                k_list.append(float(self.model.k_g(T, B)))
                cyc_list.append(float(self.model.cycle_time_s(T)))
                A_list.append(float(self.model.planning_visibility(T, B, self.noise)))

        self.T_flat = np.asarray(T_list, dtype=np.float64)
        self.B_flat = np.asarray(B_list, dtype=np.float64)
        self.k_flat = np.asarray(k_list, dtype=np.float64)
        self.cycle_flat = np.asarray(cyc_list, dtype=np.float64)
        self.A_flat = np.asarray(A_list, dtype=np.float64)

    @staticmethod
    def wrap_to_pi(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    def _phase_components(
        self,
        belief: JointGravityEpsBelief,
        T_s: float,
        Bp_nom_kTm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B_eff_eps = np.maximum(1e-12, float(Bp_nom_kTm) * (1.0 + belief.eps_grid))
        k_eps = np.asarray([self.model.k_g(T_s, float(B)) for B in B_eff_eps], dtype=np.float64)
        A_eps = np.asarray([self.model.planning_visibility(T_s, float(B), self.noise) for B in B_eff_eps], dtype=np.float64)
        theta0 = belief.g_grid[:, None] * k_eps[None, :]
        return B_eff_eps, k_eps, A_eps, theta0

    def choose_phase(self, belief: JointGravityEpsBelief, T_s: float, Bp_nom_kTm: float) -> float:
        B_eff_eps, k_eps, A_eps, theta0 = self._phase_components(belief, T_s, Bp_nom_kTm)

        if self.cfg.phase_mode == "analytic_quadrature":
            wj = belief.w_joint
            C = np.sum(wj * np.exp(1j * theta0))
            if abs(C) < 1e-18:
                return 0.0
            phi = 0.5 * np.pi - np.angle(C)
            return self.wrap_to_pi(float(phi))

        if self.cfg.phase_mode == "grid_expected_fi":
            phases = np.linspace(-np.pi, np.pi, self.cfg.phase_grid_size, dtype=np.float64)
            wj = belief.w_joint
            fi_vals = np.empty_like(phases)

            for i, phi in enumerate(phases):
                theta = theta0 + float(phi)
                num = (A_eps[None, :] ** 2) * (np.sin(theta) ** 2)
                den = 1.0 - (A_eps[None, :] ** 2) * (np.cos(theta) ** 2)
                fi_ge = (num / np.maximum(den, 1e-18)) * (k_eps[None, :] ** 2)
                fi_vals[i] = float(np.sum(wj * fi_ge))

            return float(phases[int(np.argmax(fi_vals))])

        raise ValueError(f"Unknown phase_mode={self.cfg.phase_mode}")

    def _like_matrix(
        self,
        belief: JointGravityEpsBelief,
        T_s: float,
        Bp_nom_kTm: float,
        phi_rad: float,
    ) -> tuple[np.ndarray, float]:
        B_eff_eps, k_eps, A_eps, theta0 = self._phase_components(belief, T_s, Bp_nom_kTm)
        A_nom = float(np.sum(belief.w_eps_marginal() * A_eps))

        theta = theta0 + float(phi_rad)
        p_plus = 0.5 * (1.0 + A_eps[None, :] * np.cos(theta))
        p_plus = np.clip(p_plus, self.model.eps_prob, 1.0 - self.model.eps_prob)
        return p_plus, A_nom

    def _single_utility(
        self,
        belief: JointGravityEpsBelief,
        T_s: float,
        Bp_nom_kTm: float,
        phi_rad: float,
    ) -> float:
        p_plus_ge, A_nom = self._like_matrix(belief, T_s, Bp_nom_kTm, phi_rad)

        if A_nom < self.cfg.min_visibility:
            return -np.inf

        if self.cfg.objective == "variance_reduction":
            _, _, var1 = belief.posterior_stats_g_for_likelihood(p_plus_ge)
            _, _, var0 = belief.posterior_stats_g_for_likelihood(1.0 - p_plus_ge)
            p1 = float(np.sum(belief.w_joint * p_plus_ge))
            p0 = 1.0 - p1
            utility = belief.std_g() ** 2 - (p1 * var1 + p0 * var0)

        elif self.cfg.objective == "information_gain":
            H_prior = belief.entropy_g_nats()
            Z1, H1 = belief.posterior_entropy_g_for_likelihood(p_plus_ge)
            Z0, H0 = belief.posterior_entropy_g_for_likelihood(1.0 - p_plus_ge)
            utility = H_prior - (Z1 * H1 + Z0 * H0)

        else:
            raise ValueError(f"Unknown objective={self.cfg.objective}")

        return float(utility)

    def _passes_alias_gate(self, belief: JointGravityEpsBelief, T_s: float, Bp_nom_kTm: float) -> bool:
        if self.cfg.alias_halfwidth_max_rad is None:
            return True

        h_q = belief.credible_halfwidth_g(mass=self.cfg.alias_credible_mass)
        max_k = 0.0
        for eps in belief.eps_grid:
            B_eff = max(1e-12, float(Bp_nom_kTm) * (1.0 + float(eps)))
            max_k = max(max_k, self.model.k_g(T_s, B_eff))

        return bool(max_k * h_q <= float(self.cfg.alias_halfwidth_max_rad))

    def score_action(self, belief: JointGravityEpsBelief, aT: int, aB: int) -> tuple[float, float, float]:
        idx = self.grid.encode(aT, aB)
        T_s = float(self.T_flat[idx])
        B_nom = float(self.B_flat[idx])
        A_nom = float(self.A_flat[idx])

        phi = self.choose_phase(belief, T_s, B_nom)

        if not self._passes_alias_gate(belief, T_s, B_nom):
            return -np.inf, phi, A_nom

        utility = self._single_utility(belief, T_s, B_nom, phi)

        utility -= float(self.cfg.cycle_penalty) * float(self.cycle_flat[idx])
        utility -= float(self.cfg.mfg_penalty) * abs(B_nom)

        return float(utility), phi, A_nom

    def plan_action(self, belief: JointGravityEpsBelief) -> tuple[int, int, float, float, float]:
        best_score = -np.inf
        best = None

        for aT in range(self.grid.nT):
            for aB in range(self.grid.nB):
                score, phi, A = self.score_action(belief, aT, aB)
                if np.isfinite(score) and (best is None or score > best_score):
                    best_score = score
                    best = (aT, aB, phi, A)

        if best is None:
            idx = int(np.argmin(self.k_flat))
            aT, aB = self.grid.decode(idx)
            T_s = float(self.grid.T_vals_s[aT])
            Bp_nom_kTm = float(self.grid.Bp_vals_kTm[aB])
            phi = self.choose_phase(belief, T_s, Bp_nom_kTm)
            A = self.model.planning_visibility(T_s, Bp_nom_kTm, self.noise)
            return int(aT), int(aB), float(phi), float(A), float("-inf")

        aT, aB, phi, A = best
        return int(aT), int(aB), float(phi), float(A), float(best_score)


# ============================================================
# Environment
# ============================================================


class GravimeterEnv:
    """
    Hierarchical environment:

    - true_g changes every episode
    - global_true_eps is fixed across the whole run
    - belief is episode-local joint p(g, eps), initialized from current global eps posterior
    """

    def __init__(
        self,
        prior: PriorConfig,
        grid: ActionGrid,
        model: GravimeterModel,
        noise: NoiseConfig,
        env_cfg: EnvConfig,
        seed: int = 0,
    ):
        self.prior = prior
        self.grid = grid
        self.model = model
        self.noise = noise
        self.cfg = env_cfg
        self.rng = np.random.default_rng(seed)

        self.belief = JointGravityEpsBelief(
            prior=prior,
            noise=noise,
            n_g_grid=env_cfg.n_g_grid,
            n_eps_grid=env_cfg.n_eps_grid,
        )

        self.true_g = 0.0
        self.global_true_eps: float | None = None
        self.t = 0

    def sample_true_g(self) -> float:
        return float(self.rng.uniform(self.prior.g_range[0], self.prior.g_range[1]))

    def sample_global_true_eps(self) -> float:
        b = float(max(self.noise.mfg_rel_noise_bound, 0.0))
        if b <= 0.0:
            return 0.0
        return float(self.rng.uniform(-b, +b))

    def reset(self, logw_eps_global: np.ndarray | None = None) -> None:
        self.true_g = self.sample_true_g()
        if self.global_true_eps is None:
            self.global_true_eps = self.sample_global_true_eps()
        self.belief.reset_factorized(logw_eps_global)
        self.t = 0

    def step(
        self,
        aT: int,
        aB: int,
        mw_phase_rad: float,
    ) -> tuple[bool, Dict[str, float]]:
        T_s = float(self.grid.T_vals_s[int(aT)])
        Bp_nom_kTm = float(self.grid.Bp_vals_kTm[int(aB)])

        eps_true = float(0.0 if self.global_true_eps is None else self.global_true_eps)
        Bp_true_kTm = float(max(1e-12, Bp_nom_kTm * (1.0 + eps_true)))

        A_model = self.model.planning_visibility(T_s, Bp_nom_kTm, self.noise)
        A_true = self.model.shot_visibility(T_s, Bp_true_kTm, self.noise, self.rng)

        p_plus_true = float(self.model.prob_plus(self.true_g, T_s, Bp_true_kTm, mw_phase_rad, A_true))
        outcome_plus = 1 if (self.rng.random() < p_plus_true) else 0

        self.belief.update_from_outcome(
            model=self.model,
            T_s=T_s,
            Bp_nom_kTm=Bp_nom_kTm,
            mw_phase_rad=mw_phase_rad,
            noise=self.noise,
            outcome_plus=outcome_plus,
        )

        self.t += 1
        done = bool(self.t >= self.cfg.episode_len)

        post_mu_g = self.belief.mean_g()
        post_std_g = self.belief.std_g()
        post_med_g = self.belief.median_g()
        post_map_g = self.belief.map_g()

        post_mu_eps = self.belief.mean_eps()
        post_std_eps = self.belief.std_eps()
        post_med_eps = self.belief.median_eps()
        post_map_eps = self.belief.map_eps()

        info = {
            "t": float(self.t),
            "T_s": T_s,
            "Bp_nom_kTm": Bp_nom_kTm,
            "Bp_true_kTm": Bp_true_kTm,
            "mw_phase_rad": float(mw_phase_rad),
            "A_model": float(A_model),
            "A_true": float(A_true),
            "p_plus_true": float(p_plus_true),
            "outcome_plus": float(outcome_plus),

            "true_g": float(self.true_g),
            "true_eps": float(eps_true),

            "post_mu_g": float(post_mu_g),
            "post_std_g": float(post_std_g),
            "post_median_g": float(post_med_g),
            "post_map_g": float(post_map_g),

            "post_mu_eps": float(post_mu_eps),
            "post_std_eps": float(post_std_eps),
            "post_median_eps": float(post_med_eps),
            "post_map_eps": float(post_map_eps),

            "post_entropy_g": float(self.belief.entropy_g_nats()),
            "post_entropy_joint": float(self.belief.entropy_joint_nats()),

            "g_err_mean": float(post_mu_g - self.true_g),
            "g_err_median": float(post_med_g - self.true_g),
            "g_err_map": float(post_map_g - self.true_g),

            "eps_err_mean": float(post_mu_eps - eps_true),
            "eps_err_median": float(post_med_eps - eps_true),
            "eps_err_map": float(post_map_eps - eps_true),

            "k_g_nom": float(self.model.k_g(T_s, Bp_nom_kTm)),
            "dg2pi_nom": float(self.model.delta_g_period_2pi(T_s, Bp_nom_kTm)),
            "cycle_time_s": float(self.model.cycle_time_s(T_s)),
        }
        return done, info