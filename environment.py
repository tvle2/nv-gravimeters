# environment.py
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
    Purely exogenous control/noise model.

    - trap_visibility_mode:
        "none"              -> A = 1
        "small_noise_avg"   -> uses supplement Eq. (S67)-style average visibility
        "exact_single_delta"-> samples one trap-frequency deviation per shot and uses Eq. (S65)

    - mfg_rel_noise_bound:
        bounded multiplicative control uncertainty on B'
        i.e. true B' = B'_nom * (1 + eps), eps ~ Uniform[-b, +b]

    - mfg_marginal_points:
        number of equally spaced quadrature points used to marginalize bounded
        MFG noise in the Bayesian likelihood.
        5 is a good default for mild-noise studies.
    """

    sigma_omega_rel: float = 0.0
    trap_visibility_mode: Literal["none", "small_noise_avg", "exact_single_delta"] = "small_noise_avg"

    mfg_rel_noise_bound: float = 0.0
    mfg_marginal_points: int = 5

    T2_spin_s: float | None = None


@dataclass(frozen=True)
class PlannerConfig:
    """
    objective:
        "variance_reduction" or "information_gain"

    phase_mode:
        "analytic_quadrature" -> phi = pi/2 - E[phase]
        "grid_expected_fi"    -> search phase grid by expected Fisher information

    robust_mfg_mode:
        "nominal"  -> plan using nominal B'
        "average3" -> average utility over B'*(1-b), B', B'*(1+b)
        "worst3"   -> minimum utility over B'*(1-b), B', B'*(1+b)

    Alias gate:
        We now gate actions using a posterior credible half-width h_q:
            h_q = (q_high - q_low) / 2

        and reject actions if
            k_max * h_q > alias_halfwidth_max_rad

        This replaces the old std-based gate.
    """

    objective: Literal["variance_reduction", "information_gain"] = "variance_reduction"
    phase_mode: Literal["analytic_quadrature", "grid_expected_fi"] = "analytic_quadrature"
    phase_grid_size: int = 181

    robust_mfg_mode: Literal["nominal", "average3", "worst3"] = "nominal"

    # ---- credible-width alias gate ----
    alias_halfwidth_max_rad: float | None = None
    alias_credible_mass: float = 0.90  # e.g. 0.90 => use [q05, q95]

    cycle_penalty: float = 0.0
    mfg_penalty: float = 0.0
    min_visibility: float = 0.0


@dataclass(frozen=True)
class EnvConfig:
    episode_len: int = 64
    n_g_grid: int = 1024


# ============================================================
# Physical model
# ============================================================


@dataclass(frozen=True)
class GravimeterModel:
    """
    Paper-faithful effective model:

        P(+|g) = 1/2 [1 + A cos(DeltaPhi(g) + phi_MW)]

    with affine-in-g accumulated phase:
        DeltaPhi(g) = k_g(T, B') * g

    where
        k_g(T, B') = (2 gamma_e / omega) B' T^2 + (8 pi gamma_e / omega^3) B'

    This is equivalent to using
        DeltaPhi = (2 eta / y0) g T^2 + 16 pi eta_g eta
    with eta = gamma_e B' y0 / omega and eta_g = M g y0 / (hbar omega).
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
        # five-step protocol total time: 7τ/2 + 2T
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

    # ---------------- visibility model ----------------

    def visibility_exact_from_delta_omega(self, Bp_kTm: float, delta_omega_rad_s: float) -> float:
        """
        Supplement Eq. (S65), with delta_t = -tau * delta_omega / omega.
        """
        eta = float(self.eta(Bp_kTm))
        delta_t = -self.tau_s * float(delta_omega_rad_s) / self.omega_rad_s
        x = self.omega_rad_s * delta_t
        A = np.exp(-0.5 * (16.0 * eta * np.cos(x / 4.0) * (np.sin(3.0 * x / 4.0) ** 2)) ** 2)
        return float(np.clip(A, 0.0, 1.0))

    def visibility_avg_small_noise(self, Bp_kTm: float, sigma_omega_rel: float) -> float:
        """
        Supplement Eq. (S67) in relative-noise form:
            Abar ≈ 1 - 1944*pi^4 * eta^2 * (sigma_omega/omega)^4
        """
        eta = float(self.eta(Bp_kTm))
        correction = 1944.0 * (np.pi ** 4) * (eta ** 2) * (sigma_omega_rel ** 4)
        return float(np.clip(1.0 - correction, 0.0, 1.0))

    def planning_visibility(self, T_s: float, Bp_kTm: float, noise: NoiseConfig) -> float:
        if noise.trap_visibility_mode == "none":
            A = 1.0
        else:
            # planner uses deterministic average visibility
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

    # ---------------- control perturbation ----------------

    def sample_effective_Bp_kTm(self, Bp_nom_kTm: float, noise: NoiseConfig, rng: np.random.Generator) -> float:
        bound = float(max(noise.mfg_rel_noise_bound, 0.0))
        if bound <= 0.0:
            return float(Bp_nom_kTm)
        xi = rng.uniform(-bound, +bound)
        return float(max(1e-12, Bp_nom_kTm * (1.0 + xi)))
    
    # ---------------- bounded-MFG likelihood marginalization ----------------

    def mfg_eps_grid(self, noise: NoiseConfig) -> np.ndarray:
        """
        Deterministic quadrature grid for bounded multiplicative MFG noise:
            eps ~ Uniform[-b, +b]

        Returns [0.] if no MFG noise is enabled.
        """
        b = float(max(noise.mfg_rel_noise_bound, 0.0))
        n = int(max(getattr(noise, "mfg_marginal_points", 5), 1))

        if b <= 0.0 or n <= 1:
            return np.array([0.0], dtype=np.float64)

        return np.linspace(-b, +b, n, dtype=np.float64)

    def prob_plus_marginalized_mfg(
        self,
        g_m_s2: float | ArrayF,
        T_s: float,
        Bp_nom_kTm: float,
        mw_phase_rad: float,
        noise: NoiseConfig,
    ) -> float | ArrayF:
        """
        Readout probability marginalized over bounded multiplicative MFG noise:

            eps ~ Uniform[-b, +b]
            Bp_true = Bp_nom * (1 + eps)

        We average the full probability over an equally spaced quadrature grid.
        This is the correct Bayesian likelihood to use when the simulator samples
        shot-to-shot bounded MFG perturbations but the filter does not estimate eps
        as a latent state.
        """
        eps_grid = self.mfg_eps_grid(noise)

        acc = None
        for eps in eps_grid:
            B_eff = max(1e-12, float(Bp_nom_kTm) * (1.0 + float(eps)))
            A_eff = self.planning_visibility(T_s, B_eff, noise)
            p = self.prob_plus(g_m_s2, T_s, B_eff, mw_phase_rad, A_eff)

            if acc is None:
                acc = np.asarray(p, dtype=np.float64)
            else:
                acc = acc + np.asarray(p, dtype=np.float64)

        out = acc / float(len(eps_grid))
        return np.clip(out, self.eps_prob, 1.0 - self.eps_prob)

    # ---------------- readout probability / Fisher ----------------

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
# 1D Bayesian belief over gravity
# ============================================================


class GravityGridBelief:
    def __init__(self, prior: PriorConfig, n_grid: int = 1024):
        if n_grid < 64:
            raise ValueError("n_grid must be >= 64")
        self.prior = prior
        self.n_grid = int(n_grid)
        self.g_grid = np.linspace(prior.g_range[0], prior.g_range[1], self.n_grid, dtype=np.float64)
        self.g2_grid = self.g_grid ** 2
        self.logw = np.full(self.n_grid, -np.log(self.n_grid), dtype=np.float64)

    def copy(self) -> "GravityGridBelief":
        out = GravityGridBelief(self.prior, self.n_grid)
        out.logw = self.logw.copy()
        return out

    @property
    def w(self) -> np.ndarray:
        m = float(np.max(self.logw))
        x = np.exp(self.logw - m)
        return x / (np.sum(x) + 1e-300)

    def reset_uniform(self) -> None:
        self.logw.fill(-np.log(self.n_grid))

    def normalize_(self) -> None:
        m = float(np.max(self.logw))
        z = m + np.log(np.sum(np.exp(self.logw - m)) + 1e-300)
        self.logw -= z

    def mean(self) -> float:
        w = self.w
        return float(np.sum(w * self.g_grid))

    def second_moment(self) -> float:
        w = self.w
        return float(np.sum(w * self.g2_grid))

    def variance(self) -> float:
        mu = self.mean()
        ex2 = self.second_moment()
        return float(max(ex2 - mu * mu, 0.0))

    def std(self) -> float:
        return float(np.sqrt(self.variance()))

    def quantile(self, q: float) -> float:
        """
        Weighted posterior quantile on the fixed gravity grid.
        """
        q = float(np.clip(q, 0.0, 1.0))
        w = self.w
        cdf = np.cumsum(w)
        return float(np.interp(q, cdf, self.g_grid))

    def credible_interval(self, mass: float = 0.90) -> tuple[float, float]:
        """
        Central posterior credible interval [q_low, q_high].
        Example:
            mass=0.90 -> [q05, q95]
        """
        mass = float(np.clip(mass, 1e-6, 0.999999))
        alpha = 0.5 * (1.0 - mass)
        q_lo = self.quantile(alpha)
        q_hi = self.quantile(1.0 - alpha)
        return float(q_lo), float(q_hi)

    def credible_halfwidth(self, mass: float = 0.90) -> float:
        q_lo, q_hi = self.credible_interval(mass=mass)
        return 0.5 * float(q_hi - q_lo)

    def median(self) -> float:
        return self.quantile(0.5)

    def map(self) -> float:
        return float(self.g_grid[int(np.argmax(self.logw))])

    def entropy_nats(self) -> float:
        w = self.w
        return float(-np.sum(w * np.log(w + 1e-300)))

    def peak_stats(self) -> tuple[float, float, float]:
        p = self.w
        idx1 = int(np.argmax(p))
        p1 = float(p[idx1])
        p2 = float(np.max(np.delete(p, idx1))) if p.size > 1 else 0.0
        return p1, p2, p1 - p2

    def top_peaks(self, k: int = 5) -> list[tuple[float, float]]:
        """
        Return top-k grid peaks as (g_value, prob).
        """
        p = self.w
        K = int(min(max(1, k), p.size))
        idx = np.argpartition(p, -K)[-K:]
        idx = idx[np.argsort(p[idx])[::-1]]
        return [(float(self.g_grid[i]), float(p[i])) for i in idx]

    def posterior_stats_for_likelihood(self, like: np.ndarray) -> tuple[float, float, float]:
        """
        Return (Z, mean_post, var_post) for posterior proportional to w * like.
        """
        w = self.w
        s = w * like
        Z = float(np.sum(s) + 1e-300)
        mu = float(np.sum(s * self.g_grid) / Z)
        ex2 = float(np.sum(s * self.g2_grid) / Z)
        var = float(max(ex2 - mu * mu, 0.0))
        return Z, mu, var

    def posterior_entropy_for_likelihood(self, like: np.ndarray) -> tuple[float, float]:
        """
        Return (Z, H_post) for posterior proportional to w * like.
        """
        w = self.w
        s = w * like
        Z = float(np.sum(s) + 1e-300)
        wn = s / Z
        H = float(-np.sum(wn * np.log(wn + 1e-300)))
        return Z, H

    def update_from_outcome(
        self,
        model: GravimeterModel,
        T_s: float,
        Bp_kTm: float,
        mw_phase_rad: float,
        noise: NoiseConfig,
        outcome_plus: int,
    ) -> None:
        """
        Bayesian update using the MFG-noise-marginalized likelihood.
        """
        p_plus = model.prob_plus_marginalized_mfg(
            self.g_grid,
            T_s,
            Bp_kTm,
            mw_phase_rad,
            noise,
        )
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

        for i, T_s in enumerate(self.grid.T_vals_s):
            for j, Bp_kTm in enumerate(self.grid.Bp_vals_kTm):
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

    def robust_deltas(self) -> list[float]:
        b = float(max(self.noise.mfg_rel_noise_bound, 0.0))
        if b <= 0.0 or self.cfg.robust_mfg_mode == "nominal":
            return [0.0]
        return [-b, 0.0, +b]

    @staticmethod
    def wrap_to_pi(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    def choose_phase(self, belief: GravityGridBelief, T_s: float, Bp_kTm: float) -> float:
        if self.cfg.phase_mode == "analytic_quadrature":
            mu = belief.mean()
            phi = 0.5 * np.pi - self.model.phase_total(mu, T_s, Bp_kTm)
            return self.wrap_to_pi(float(phi))

        if self.cfg.phase_mode == "grid_expected_fi":
            A = self.model.planning_visibility(T_s, Bp_kTm, self.noise)
            phases = np.linspace(-np.pi, np.pi, self.cfg.phase_grid_size, dtype=np.float64)
            w = belief.w
            g = belief.g_grid
            fi_vals = np.empty_like(phases)
            for k, phi in enumerate(phases):
                fi_vals[k] = float(np.sum(w * self.model.fisher_information_g(g, T_s, Bp_kTm, float(phi), A)))
            return float(phases[int(np.argmax(fi_vals))])

        raise ValueError(f"Unknown phase_mode={self.cfg.phase_mode}")

    def _single_utility(
        self,
        belief: GravityGridBelief,
        T_s: float,
        Bp_eff_kTm: float,
        phi_rad: float,
    ) -> float:
        """
        One-step utility using the same MFG-marginalized likelihood as the Bayesian filter.

        IMPORTANT:
        This keeps planning and inference consistent under bounded multiplicative MFG noise.
        """
        # Use the marginalized readout model, not the nominal one
        p_plus_g = self.model.prob_plus_marginalized_mfg(
            belief.g_grid,
            T_s,
            Bp_eff_kTm,
            phi_rad,
            self.noise,
        )

        # Optional visibility guard (based on nominal planning visibility at the action center)
        A_nom = self.model.planning_visibility(T_s, Bp_eff_kTm, self.noise)
        if A_nom < self.cfg.min_visibility:
            return -np.inf

        if self.cfg.objective == "variance_reduction":
            _, _, var1 = belief.posterior_stats_for_likelihood(p_plus_g)
            _, _, var0 = belief.posterior_stats_for_likelihood(1.0 - p_plus_g)
            p1 = float(np.sum(belief.w * p_plus_g))
            p0 = 1.0 - p1
            expected_post_var = p1 * var1 + p0 * var0
            utility = belief.variance() - expected_post_var

        elif self.cfg.objective == "information_gain":
            H_prior = belief.entropy_nats()
            Z1, H1 = belief.posterior_entropy_for_likelihood(p_plus_g)
            Z0, H0 = belief.posterior_entropy_for_likelihood(1.0 - p_plus_g)
            utility = H_prior - (Z1 * H1 + Z0 * H0)

        else:
            raise ValueError(f"Unknown objective={self.cfg.objective}")

        return float(utility)

    def _passes_alias_gate(self, belief: GravityGridBelief, T_s: float, B_nom: float) -> bool:
        """
        Credible-width alias gate:

            reject if k_max * h_q > alias_halfwidth_max_rad

        where h_q = (q_high - q_low)/2 for the posterior credible interval.

        This version is MFG-noise-aware: it checks the maximum k over the bounded
        multiplicative MFG support as well.
        """
        if self.cfg.alias_halfwidth_max_rad is None:
            return True

        h_q = belief.credible_halfwidth(mass=self.cfg.alias_credible_mass)

        max_k = 0.0
        eps_grid = self.model.mfg_eps_grid(self.noise)

        for delta in self.robust_deltas():
            B_center = max(1e-12, B_nom * (1.0 + delta))
            for eps in eps_grid:
                B_eff = max(1e-12, B_center * (1.0 + float(eps)))
                max_k = max(max_k, self.model.k_g(T_s, B_eff))

        return bool(max_k * h_q <= float(self.cfg.alias_halfwidth_max_rad))

    def score_action(self, belief: GravityGridBelief, aT: int, aB: int) -> tuple[float, float, float]:
        idx = self.grid.encode(aT, aB)
        T_s = float(self.T_flat[idx])
        B_nom = float(self.B_flat[idx])
        A_nom = float(self.A_flat[idx])

        phi = self.choose_phase(belief, T_s, B_nom)

        if not self._passes_alias_gate(belief, T_s, B_nom):
            return -np.inf, phi, A_nom

        deltas = self.robust_deltas()
        utilities = []
        for delta in deltas:
            B_eff = max(1e-12, B_nom * (1.0 + delta))
            u = self._single_utility(belief, T_s, B_eff, phi)
            utilities.append(u)

        if not np.all(np.isfinite(utilities)):
            return -np.inf, phi, A_nom

        if self.cfg.robust_mfg_mode in ("nominal", "average3"):
            utility = float(np.mean(utilities))
        elif self.cfg.robust_mfg_mode == "worst3":
            utility = float(np.min(utilities))
        else:
            raise ValueError(f"Unknown robust_mfg_mode={self.cfg.robust_mfg_mode}")

        utility -= float(self.cfg.cycle_penalty) * float(self.cycle_flat[idx])
        utility -= float(self.cfg.mfg_penalty) * abs(B_nom)

        return float(utility), phi, A_nom

    def plan_action(self, belief: GravityGridBelief) -> tuple[int, int, float, float, float]:
        best_score = -np.inf
        best = None

        for aT in range(self.grid.nT):
            for aB in range(self.grid.nB):
                score, phi, A = self.score_action(belief, aT, aB)
                if np.isfinite(score) and (best is None or score > best_score):
                    best_score = score
                    best = (aT, aB, phi, A)

        if best is None:
            # fallback: safest lowest-k action
            idx = int(np.argmin(self.k_flat))
            aT, aB = self.grid.decode(idx)
            T_s = float(self.grid.T_vals_s[aT])
            Bp_kTm = float(self.grid.Bp_vals_kTm[aB])
            phi = self.choose_phase(belief, T_s, Bp_kTm)
            A = self.model.planning_visibility(T_s, Bp_kTm, self.noise)
            return int(aT), int(aB), float(phi), float(A), float("-inf")

        aT, aB, phi, A = best
        return int(aT), int(aB), float(phi), float(A), float(best_score)


# ============================================================
# Simulation environment
# ============================================================


class GravimeterEnv:
    """
    Simulation environment for the pure Bayesian adaptive controller.

    Hidden state:
        true_g

    Belief state:
        1D Bayesian posterior over g only

    Important:
        The filter does NOT estimate hidden nuisance states.
        Control/noise uncertainty is treated exogenously in simulation/planning.
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

        self.belief = GravityGridBelief(prior, n_grid=env_cfg.n_g_grid)
        self.true_g = 0.0
        self.t = 0

    def sample_true_g(self) -> float:
        return float(self.rng.uniform(self.prior.g_range[0], self.prior.g_range[1]))

    def reset(self) -> None:
        self.true_g = self.sample_true_g()
        self.belief.reset_uniform()
        self.t = 0

    def step(
        self,
        aT: int,
        aB: int,
        mw_phase_rad: float,
    ) -> tuple[bool, Dict[str, float]]:
        T_s = float(self.grid.T_vals_s[int(aT)])
        Bp_nom_kTm = float(self.grid.Bp_vals_kTm[int(aB)])

        # Filter/planner model
        A_model = self.model.planning_visibility(T_s, Bp_nom_kTm, self.noise)

        # True applied control / true shot visibility
        Bp_true_kTm = self.model.sample_effective_Bp_kTm(Bp_nom_kTm, self.noise, self.rng)
        A_true = self.model.shot_visibility(T_s, Bp_true_kTm, self.noise, self.rng)

        # Generate measurement
        p_plus_true = float(self.model.prob_plus(self.true_g, T_s, Bp_true_kTm, mw_phase_rad, A_true))
        outcome_plus = 1 if (self.rng.random() < p_plus_true) else 0

        # Bayesian update under nominal control model
        self.belief.update_from_outcome(
            model=self.model,
            T_s=T_s,
            Bp_kTm=Bp_nom_kTm,
            mw_phase_rad=mw_phase_rad,
            noise=self.noise,
            outcome_plus=outcome_plus,
        )

        self.t += 1
        done = bool(self.t >= self.cfg.episode_len)

        post_mu = self.belief.mean()
        post_std = self.belief.std()
        post_med = self.belief.median()
        post_map = self.belief.map()
        post_H = self.belief.entropy_nats()

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
            "post_mu_g": float(post_mu),
            "post_std_g": float(post_std),
            "post_median_g": float(post_med),
            "post_map_g": float(post_map),
            "post_entropy_g": float(post_H),
            "g_err_mean": float(post_mu - self.true_g),
            "g_err_median": float(post_med - self.true_g),
            "g_err_map": float(post_map - self.true_g),
            "k_g_nom": float(self.model.k_g(T_s, Bp_nom_kTm)),
            "dg2pi_nom": float(self.model.delta_g_period_2pi(T_s, Bp_nom_kTm)),
            "cycle_time_s": float(self.model.cycle_time_s(T_s)),
        }
        return done, info