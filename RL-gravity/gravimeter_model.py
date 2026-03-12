# gravimeter_model.py
from __future__ import annotations


from dataclasses import dataclass, asdict
from math import floor, pi
from pathlib import Path
from typing import Optional, Sequence

import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# -----------------------------------------------------------------------------
# Robust imports: use qsensoropt if installed; otherwise load local flat files
# as a synthetic package so that their relative imports continue to work.
# -----------------------------------------------------------------------------

def _load_local_qsensoropt_module(module_name: str):
    root = Path(__file__).resolve().parent
    pkg_name = "_qsensoropt_local"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root)]
        sys.modules[pkg_name] = pkg

    full_name = f"{pkg_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = root / f"{module_name}.py"
    if not path.exists():
        raise ImportError(f"Cannot find local qsensoropt module: {path}")

    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for local module: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from qsensoropt.parameter import Parameter
    from qsensoropt.physical_model import Control
    from qsensoropt.stateless_phys_model import StatelessPhysicalModel
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.stateless_metrology import StatelessMetrology
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
    from qsensoropt.utils import (
        train,
        performance_evaluation,
        store_input_control,
        sqrt_hmatrix,
        get_seed,
    )
except Exception:
    Parameter = _load_local_qsensoropt_module("parameter").Parameter
    Control = _load_local_qsensoropt_module("physical_model").Control
    StatelessPhysicalModel = _load_local_qsensoropt_module("stateless_phys_model").StatelessPhysicalModel
    ParticleFilter = _load_local_qsensoropt_module("particle_filter").ParticleFilter
    StatelessMetrology = _load_local_qsensoropt_module("stateless_metrology").StatelessMetrology
    SimulationParameters = _load_local_qsensoropt_module("simulation_parameters").SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay
    _utils = _load_local_qsensoropt_module("utils")
    train = _utils.train
    performance_evaluation = _utils.performance_evaluation
    store_input_control = _utils.store_input_control
    sqrt_hmatrix = _utils.sqrt_hmatrix
    get_seed = _utils.get_seed


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_to_minus1_plus1(x: Tensor, bounds: tuple[float, float]) -> Tensor:
    lo = tf.cast(bounds[0], x.dtype)
    hi = tf.cast(bounds[1], x.dtype)
    width = tf.maximum(hi - lo, tf.cast(1e-18, x.dtype))
    y = 2.0 * (x - lo) / width - 1.0
    return tf.clip_by_value(y, -1.2, 1.2)


def denormalize_from_minus1_plus1(x: Tensor, bounds: tuple[float, float]) -> Tensor:
    lo = tf.cast(bounds[0], x.dtype)
    hi = tf.cast(bounds[1], x.dtype)
    return lo + 0.5 * (x + 1.0) * (hi - lo)


def wrap_to_pi_tf(x: Tensor) -> Tensor:
    two_pi = tf.cast(2.0 * pi, x.dtype)
    return tf.math.floormod(x + tf.cast(pi, x.dtype), two_pi) - tf.cast(pi, x.dtype)


def safe_clip_prob(x: Tensor, eps: float = 1e-9) -> Tensor:
    eps_t = tf.cast(eps, x.dtype)
    return tf.clip_by_value(x, eps_t, 1.0 - eps_t)


def _latest_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _decode_affine_np(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return lo + 0.5 * (x + 1.0) * (hi - lo)


def _decode_logstd_np(x: np.ndarray) -> np.ndarray:
    log_std = -10.0 + 0.5 * (x + 1.0) * 10.0
    return np.exp(log_std)


def _prob01_to_pm1(x: Tensor) -> Tensor:
    return 2.0 * x - 1.0


def _pm1_to_prob01_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + 1.0)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GravimeterConfig:
    # physical constants
    omega_rad_s: float = 2.0 * pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * pi * 28e9
    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34
    kT_to_T: float = 1e3

    # parameter priors / admissible ranges
    g_range: tuple[float, float] = (9.7639, 9.8337)
    phi_off_range: tuple[float, float] = (-pi, pi)
    A_range: tuple[float, float] = (0.45, 1.0)

    # control ranges
    T_range_s: tuple[float, float] = (50e-6, 1.0e-3)
    Bp_range_kTm: tuple[float, float] = (0.5, 50.0)
    delta_max_rad: float = pi / 3.0

    # hidden-noise model for true measurement simulation
    mfg_rel_noise_bound: float = 0.05
    mfg_noise_quad_points: int = 9
    sigma_omega_rel: float = 0.003
    trap_visibility_mode: str = "exact_single_delta"   # none | small_noise_avg | exact_single_delta
    T2_spin_s: Optional[float] = 0.015
    readout_flip_prob: float = 0.0

    # resource model
    dead_time_s: float = 0.0

    # tensorflow precision
    prec: str = "float64"

    @property
    def tau_s(self) -> float:
        return 2.0 * pi / self.omega_rad_s


@dataclass(frozen=True)
class BranchBankConfig:
    num_branches: int = 4
    particles_per_branch: int = 512
    top_k_branches: int = 4
    init_mode: str = "stratified_g"   # stratified_g | grid_g_phi
    hidden_sizes: tuple[int, ...] = (128, 128, 128, 128)


# -----------------------------------------------------------------------------
# Gravity physical model (stateless probe)
# -----------------------------------------------------------------------------

class GravityStatelessPhysicalModel(StatelessPhysicalModel):
    """
    Unknown parameters:
        theta = (g, phi_off, A)

    Controls:
        x = (T_s, Bp_kTm, mw_phase_rad)

    Likelihood:
        p(y=1 | theta, x) = 1/2 [1 + V(T,B) * A * cos((1+eps) k_g(T,B) g + phi_off + mw_phase)]
    """

    def __init__(self, batchsize: int, cfg: GravimeterConfig) -> None:
        self.cfg = cfg

        controls = [
            Control(name="T_s", is_discrete=False),
            Control(name="Bp_kTm", is_discrete=False),
            Control(name="mw_phase_rad", is_discrete=False),
        ]

        params = [
            Parameter(bounds=cfg.g_range, name="g"),
            Parameter(bounds=cfg.phi_off_range, name="phi_off"),
            Parameter(bounds=cfg.A_range, name="A"),
        ]

        super().__init__(
            batchsize=batchsize,
            controls=controls,
            params=params,
            outcomes_size=1,
            prec=cfg.prec,
        )

    def y0_m(self) -> Tensor:
        cfg = self.cfg
        return tf.cast(
            tf.sqrt(cfg.hbar_J_s / (2.0 * cfg.mass_kg * cfg.omega_rad_s)),
            dtype=cfg.prec,
        )

    def eta(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, Bp_kTm.dtype)
        return (
            tf.cast(cfg.gamma_e_rad_s_T, Bp_kTm.dtype)
            * Bp_T_per_m
            * tf.cast(self.y0_m(), Bp_kTm.dtype)
            / tf.cast(cfg.omega_rad_s, Bp_kTm.dtype)
        )

    def k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * (T_s ** 2) + (
            8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)
        ) * Bp_T_per_m

    def cycle_time_s(self, T_s: Tensor) -> Tensor:
        cfg = self.cfg
        return tf.cast(cfg.dead_time_s + 3.5 * cfg.tau_s, T_s.dtype) + 2.0 * T_s

    def trap_visibility_avg_small_noise(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        correction = tf.cast(1944.0 * (pi ** 4) * (cfg.sigma_omega_rel ** 4), eta.dtype) * (eta ** 2)
        return tf.clip_by_value(1.0 - correction, 0.0, 1.0)

    def trap_visibility_exact_from_delta_omega(self, Bp_kTm: Tensor, delta_omega_rad_s: Tensor) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        tau = tf.cast(cfg.tau_s, eta.dtype)
        x = tf.cast(cfg.omega_rad_s, eta.dtype) * (-tau * delta_omega_rad_s / tf.cast(cfg.omega_rad_s, eta.dtype))
        amp = 16.0 * eta * tf.cos(x / 4.0) * (tf.sin(3.0 * x / 4.0) ** 2)
        return tf.exp(-0.5 * amp ** 2)

    def known_visibility_factor(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        else:
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)

        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))
        return tf.clip_by_value(vis, 0.0, 1.0)

    def sample_true_visibility_factor(self, T_s: Tensor, Bp_kTm: Tensor, rangen: tf.random.Generator) -> Tensor:
        cfg = self.cfg
        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            delta_omega = rangen.normal(tf.shape(T_s), dtype=T_s.dtype) * tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, T_s.dtype)
            vis = self.trap_visibility_exact_from_delta_omega(Bp_kTm, delta_omega)
        else:
            raise ValueError(f"Unknown trap_visibility_mode={cfg.trap_visibility_mode}")

        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))
        return tf.clip_by_value(vis, 0.0, 1.0)

    def mfg_quadrature(self) -> Tensor:
        cfg = self.cfg
        if cfg.mfg_rel_noise_bound <= 0.0:
            return tf.constant([0.0], dtype=cfg.prec)
        return tf.linspace(
            tf.cast(-cfg.mfg_rel_noise_bound, cfg.prec),
            tf.cast(+cfg.mfg_rel_noise_bound, cfg.prec),
            int(max(3, cfg.mfg_noise_quad_points)),
        )

    def model(self, outcomes: Tensor, controls: Tensor, parameters: Tensor, meas_step: Tensor, num_systems: int = 1) -> Tensor:
        del meas_step, num_systems

        T_s = controls[:, :, 0]
        Bp_kTm = controls[:, :, 1]
        mw_phase = controls[:, :, 2]

        g = parameters[:, :, 0]
        phi_off = parameters[:, :, 1]
        A = parameters[:, :, 2]

        vis_known = self.known_visibility_factor(T_s, Bp_kTm)
        total_vis = tf.clip_by_value(A * vis_known, 0.0, 1.0)
        kg = self.k_g(T_s, Bp_kTm)

        eps_grid = tf.cast(self.mfg_quadrature(), T_s.dtype)
        theta = (
            tf.expand_dims(kg * g, axis=2) * (1.0 + eps_grid[None, None, :])
            + tf.expand_dims(phi_off + mw_phase, axis=2)
        )

        p_plus = 0.5 * (1.0 + tf.expand_dims(total_vis, axis=2) * tf.cos(theta))
        p_plus = safe_clip_prob(p_plus)
        p_plus = tf.reduce_mean(p_plus, axis=2)

        y = outcomes[:, :, 0]
        prob = tf.where(y > 0.5, p_plus, 1.0 - p_plus)
        return safe_clip_prob(prob)

    def perform_measurement(self, controls: Tensor, parameters: Tensor, meas_step: Tensor, rangen: tf.random.Generator) -> tuple[Tensor, Tensor]:
        del meas_step

        T_s = controls[:, 0, 0]
        Bp_kTm = controls[:, 0, 1]
        mw_phase = controls[:, 0, 2]

        g = parameters[:, 0, 0]
        phi_off = parameters[:, 0, 1]
        A = parameters[:, 0, 2]

        if self.cfg.mfg_rel_noise_bound > 0.0:
            eps_true = rangen.uniform(
                shape=tf.shape(T_s),
                minval=tf.cast(-self.cfg.mfg_rel_noise_bound, T_s.dtype),
                maxval=tf.cast(+self.cfg.mfg_rel_noise_bound, T_s.dtype),
                dtype=T_s.dtype,
            )
        else:
            eps_true = tf.zeros_like(T_s)

        vis_true = self.sample_true_visibility_factor(T_s, Bp_kTm, rangen)
        total_vis = tf.clip_by_value(A * vis_true, 0.0, 1.0)

        theta = (1.0 + eps_true) * self.k_g(T_s, Bp_kTm) * g + phi_off + mw_phase
        p_plus = safe_clip_prob(0.5 * (1.0 + total_vis * tf.cos(theta)))

        if self.cfg.readout_flip_prob > 0.0:
            flip = tf.cast(self.cfg.readout_flip_prob, p_plus.dtype)
            p_plus = (1.0 - flip) * p_plus + flip * (1.0 - p_plus)
            p_plus = safe_clip_prob(p_plus)

        u = rangen.uniform(shape=tf.shape(p_plus), minval=0.0, maxval=1.0, dtype=p_plus.dtype)
        y = tf.cast(u < p_plus, p_plus.dtype)

        outcomes = tf.expand_dims(tf.expand_dims(y, axis=1), axis=2)
        log_prob = tf.expand_dims(tf.math.log(tf.where(y > 0.5, p_plus, 1.0 - p_plus)), axis=1)
        return outcomes, log_prob

    def count_resources(self, resources: Tensor, outcomes: Tensor, controls: Tensor, true_values: Tensor, meas_step: Tensor) -> Tensor:
        del outcomes, true_values, meas_step
        T_s = controls[:, 0:1]
        return resources + self.cycle_time_s(T_s)


# -----------------------------------------------------------------------------
# Fixed-bank branch particle filter
# -----------------------------------------------------------------------------

class GravityBranchBankParticleFilter(ParticleFilter):
    """
    PF-compatible fixed bank of branches represented internally as
        (bs, L, Nb, d)
    and exposed externally as the flattened
        (bs, L*Nb, d).

    Bayes updates remain standard on the flattened ensemble.
    The key multimodality fix is branch-local resampling.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank_cfg: BranchBankConfig,
        *,
        resampling_allowed: bool = True,
        resample_threshold: float = 0.5,
        resample_fraction: float = 0.98,
        alpha: float = 0.5,
        beta: float = 0.98,
        gamma: float = 1.0,
        scibior_trick: bool = True,
        trim: bool = True,
        prec: str = "float64",
    ) -> None:
        self.bank_cfg = bank_cfg
        self.L = int(bank_cfg.num_branches)
        self.npb = int(bank_cfg.particles_per_branch)
        self.K = int(min(bank_cfg.top_k_branches, bank_cfg.num_branches))
        total_particles = self.L * self.npb

        super().__init__(
            num_particles=total_particles,
            phys_model=phys_model,
            resampling_allowed=resampling_allowed,
            resample_threshold=resample_threshold,
            resample_fraction=resample_fraction,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scibior_trick=scibior_trick,
            trim=trim,
            prec=prec,
        )

        self.local_nrp = floor(self.gamma * self.npb)
        self.local_nnp = self.npb - self.local_nrp
        if self.local_nrp <= 0:
            raise ValueError("local_nrp must be positive; use gamma > 0.")

    # ---- reshape helpers ----

    def _bank_weights(self, weights: Tensor) -> Tensor:
        return tf.reshape(weights, (self.bs, self.L, self.npb))

    def _bank_particles(self, particles: Tensor) -> Tensor:
        return tf.reshape(particles, (self.bs, self.L, self.npb, self.d))

    def _flat_weights(self, bank_weights: Tensor) -> Tensor:
        return tf.reshape(bank_weights, (self.bs, self.L * self.npb))

    def _flat_particles(self, bank_particles: Tensor) -> Tensor:
        return tf.reshape(bank_particles, (self.bs, self.L * self.npb, self.d))

    # ---- initialization ----

    def reset(self, rangen: tf.random.Generator):
        cfg = self.phys_model.cfg
        dtype = cfg.prec

        weights = tf.ones((self.bs, self.L, self.npb), dtype=dtype)
        weights = weights / tf.cast(self.L * self.npb, dtype)

        if self.bank_cfg.init_mode == "stratified_g":
            g_edges = tf.linspace(tf.cast(cfg.g_range[0], dtype), tf.cast(cfg.g_range[1], dtype), self.L + 1)
            g_lo = g_edges[:-1][None, :, None, None]
            g_hi = g_edges[1:][None, :, None, None]

            ug = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)
            uphi = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)
            uA = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)

            g = g_lo + ug * (g_hi - g_lo)
            phi = tf.cast(cfg.phi_off_range[0], dtype) + uphi * tf.cast(cfg.phi_off_range[1] - cfg.phi_off_range[0], dtype)
            A = tf.cast(cfg.A_range[0], dtype) + uA * tf.cast(cfg.A_range[1] - cfg.A_range[0], dtype)

        elif self.bank_cfg.init_mode == "grid_g_phi":
            Lg = int(round(self.L ** 0.5))
            if Lg * Lg != self.L:
                raise ValueError("grid_g_phi requires num_branches to be a perfect square.")
            Lphi = Lg

            g_edges = tf.linspace(tf.cast(cfg.g_range[0], dtype), tf.cast(cfg.g_range[1], dtype), Lg + 1)
            phi_edges = tf.linspace(tf.cast(cfg.phi_off_range[0], dtype), tf.cast(cfg.phi_off_range[1], dtype), Lphi + 1)

            g_parts = []
            phi_parts = []
            for ig in range(Lg):
                for ip in range(Lphi):
                    ug = rangen.uniform((self.bs, 1, self.npb, 1), dtype=dtype)
                    up = rangen.uniform((self.bs, 1, self.npb, 1), dtype=dtype)
                    g_part = g_edges[ig] + ug * (g_edges[ig + 1] - g_edges[ig])
                    phi_part = phi_edges[ip] + up * (phi_edges[ip + 1] - phi_edges[ip])
                    g_parts.append(g_part)
                    phi_parts.append(phi_part)

            g = tf.concat(g_parts, axis=1)
            phi = tf.concat(phi_parts, axis=1)
            uA = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)
            A = tf.cast(cfg.A_range[0], dtype) + uA * tf.cast(cfg.A_range[1] - cfg.A_range[0], dtype)
        else:
            raise ValueError(f"Unknown init_mode={self.bank_cfg.init_mode}")

        particles = tf.concat([g, phi, A], axis=3)
        return self._flat_weights(weights), self._flat_particles(particles)

    # ---- branch statistics ----

    def branch_masses(self, weights: Tensor) -> Tensor:
        w = self._bank_weights(weights)
        return tf.reduce_sum(w, axis=2)

    def branch_local_weights(self, weights: Tensor) -> Tensor:
        w = self._bank_weights(weights)
        q = tf.reduce_sum(w, axis=2, keepdims=True)
        return tf.math.divide_no_nan(w, q)

    def branch_statistics(self, weights: Tensor, particles: Tensor):
        wloc = self.branch_local_weights(weights)
        q = self.branch_masses(weights)
        p = self._bank_particles(particles)

        g = p[:, :, :, 0]
        phi = p[:, :, :, 1]
        A = p[:, :, :, 2]

        mu_g = tf.reduce_sum(wloc * g, axis=2)
        cphi = tf.reduce_sum(wloc * tf.cos(phi), axis=2)
        sphi = tf.reduce_sum(wloc * tf.sin(phi), axis=2)
        mu_phi = tf.atan2(sphi, cphi)
        mu_A = tf.reduce_sum(wloc * A, axis=2)

        d_g = g - mu_g[:, :, None]
        d_phi = wrap_to_pi_tf(phi - mu_phi[:, :, None])
        d_A = A - mu_A[:, :, None]

        centered = tf.stack([d_g, d_phi, d_A], axis=-1)
        cov = tf.einsum("bln,blni,blnj->blij", wloc, centered, centered)
        var = tf.linalg.diag_part(cov)
        std = tf.sqrt(tf.abs(var) + tf.cast(1e-18, cov.dtype))
        denom = std[:, :, :, None] * std[:, :, None, :]
        corr = tf.math.divide_no_nan(cov, denom)

        means = tf.stack([mu_g, mu_phi, mu_A], axis=-1)
        return q, means, std, corr

    def topk_branch_summary(self, weights: Tensor, particles: Tensor, K: Optional[int] = None):
        K = self.K if K is None else int(K)
        q, means, std, corr = self.branch_statistics(weights, particles)
        order = tf.argsort(q, axis=1, direction="DESCENDING")[:, :K]
        qk = tf.gather(q, order, batch_dims=1)
        mk = tf.gather(means, order, batch_dims=1)
        sk = tf.gather(std, order, batch_dims=1)
        ck = tf.gather(corr, order, batch_dims=1)
        return qk, mk, sk, ck

    # ---- branch-local resampling ----

    def full_resampling(self, weights: Tensor, particles: Tensor, count_for_resampling: Tensor, rangen):
        wbank = self._bank_weights(weights)
        q = tf.reduce_sum(wbank, axis=2, keepdims=True)
        wloc = tf.math.divide_no_nan(wbank, q)

        eff_n = tf.math.divide_no_nan(
            tf.ones_like(tf.reduce_sum(wloc * wloc, axis=2)),
            tf.reduce_sum(wloc * wloc, axis=2),
        )
        need_branch = eff_n < tf.cast(self.res_thres * self.npb, eff_n.dtype)
        active = tf.squeeze(count_for_resampling, axis=1)
        need_run = tf.logical_and(active, tf.reduce_any(need_branch, axis=1))

        threshold_for_resampling = self.res_frac * tf.cast(tf.math.count_nonzero(count_for_resampling), tf.float64)
        resample_flag = tf.cond(
            tf.greater_equal(tf.cast(tf.math.count_nonzero(need_run), tf.float64), threshold_for_resampling),
            lambda: tf.constant(True),
            lambda: tf.constant(False),
        )

        def _do_resample():
            pbank = self._bank_particles(particles)
            wloc_new, pbank_new = self._resample_all_branches(wloc, pbank, rangen)
            wbank_new = wloc_new * q
            return self._flat_weights(wbank_new), self._flat_particles(pbank_new)

        weights, particles = tf.cond(resample_flag, _do_resample, lambda: (weights, particles))
        return weights, particles, resample_flag

    def _resample_all_branches(self, wloc: Tensor, pbank: Tensor, rangen):
        bsL = self.bs * self.L
        w = tf.reshape(wloc, (bsL, self.npb))
        p = tf.reshape(pbank, (bsL, self.npb, self.d))

        g = p[:, :, 0]
        phi = p[:, :, 1]
        A = p[:, :, 2]

        mu_g = tf.reduce_sum(w * g, axis=1, keepdims=True)
        cphi = tf.reduce_sum(w * tf.cos(phi), axis=1, keepdims=True)
        sphi = tf.reduce_sum(w * tf.sin(phi), axis=1, keepdims=True)
        mu_phi = tf.atan2(sphi, cphi)
        mu_A = tf.reduce_sum(w * A, axis=1, keepdims=True)

        cg = g - mu_g
        cphi_dev = wrap_to_pi_tf(phi - mu_phi)
        cA = A - mu_A
        centered = tf.stack([cg, cphi_dev, cA], axis=2)

        cov = tf.einsum("bn,bni,bnj->bij", w, centered, centered)
        dev = sqrt_hmatrix(cov)

        resampling_weights = self.alpha * w + (1.0 - self.alpha) / tf.cast(self.npb, w.dtype)
        seed = get_seed(rangen)
        idx = tf.random.stateless_categorical(tf.math.log(resampling_weights), self.local_nrp, seed, dtype=tf.int32)

        p_sel = tf.gather(p, idx, batch_dims=1)
        w_sel = tf.gather(w, idx, batch_dims=1)

        denom = self.alpha * w_sel + (1.0 - self.alpha) / tf.cast(self.npb, w.dtype)
        w_soft = tf.math.divide_no_nan(w_sel, denom)
        w_soft = tf.math.divide_no_nan(w_soft, tf.reduce_sum(w_soft, axis=1, keepdims=True)) * tf.cast(self.local_nrp / self.npb, w.dtype)

        if self.scibior_trick:
            selected_resampling_weights = tf.gather(resampling_weights, idx, batch_dims=1)
            w_soft = tf.math.divide_no_nan(
                w_soft * selected_resampling_weights,
                tf.stop_gradient(selected_resampling_weights),
            )

        p_sel_g = p_sel[:, :, 0] - mu_g
        p_sel_phi = wrap_to_pi_tf(p_sel[:, :, 1] - mu_phi)
        p_sel_A = p_sel[:, :, 2] - mu_A
        p_sel_centered = tf.stack([p_sel_g, p_sel_phi, p_sel_A], axis=2)

        noise_seed = get_seed(rangen)
        noise = tf.random.stateless_normal((bsL, self.local_nrp, self.d), noise_seed, dtype=w.dtype)
        delta = tf.einsum("bij,bnj->bni", dev, noise) * tf.cast((1.0 - self.beta ** 2), w.dtype)
        p_soft_centered = self.beta * p_sel_centered + delta

        p_soft = tf.stack(
            [
                mu_g + p_soft_centered[:, :, 0],
                wrap_to_pi_tf(mu_phi + p_soft_centered[:, :, 1]),
                mu_A + p_soft_centered[:, :, 2],
            ],
            axis=2,
        )

        if self.local_nnp > 0:
            extra_seed = get_seed(rangen)
            extra_noise = tf.random.stateless_normal((bsL, self.local_nnp, self.d), extra_seed, dtype=w.dtype)
            extra_centered = tf.einsum("bij,bnj->bni", dev, extra_noise)
            p_extra = tf.stack(
                [
                    mu_g + extra_centered[:, :, 0],
                    wrap_to_pi_tf(mu_phi + extra_centered[:, :, 1]),
                    mu_A + extra_centered[:, :, 2],
                ],
                axis=2,
            )
            w_extra = tf.ones((bsL, self.local_nnp), dtype=w.dtype) / tf.cast(self.npb, w.dtype)
            p_new = tf.concat([p_soft, p_extra], axis=1)
            w_new = tf.concat([w_soft, w_extra], axis=1)
        else:
            p_new = p_soft
            w_new = w_soft

        p_new = self._trim_gravity_particles_flat(p_new)
        w_new = tf.reshape(w_new, (self.bs, self.L, self.npb))
        p_new = tf.reshape(p_new, (self.bs, self.L, self.npb, self.d))
        return w_new, p_new

    def _trim_gravity_particles_flat(self, particles: Tensor) -> Tensor:
        cfg = self.phys_model.cfg
        g = tf.clip_by_value(
            particles[:, :, 0:1],
            tf.cast(cfg.g_range[0], particles.dtype),
            tf.cast(cfg.g_range[1], particles.dtype),
        )
        phi = wrap_to_pi_tf(particles[:, :, 1:2])
        A = tf.clip_by_value(
            particles[:, :, 2:3],
            tf.cast(cfg.A_range[0], particles.dtype),
            tf.cast(cfg.A_range[1], particles.dtype),
        )
        return tf.concat([g, phi, A], axis=2)


# -----------------------------------------------------------------------------
# Branch-aware metrology wrapper
# -----------------------------------------------------------------------------

class BranchAwareGravityMetrology(StatelessMetrology):
    """
    NN input = top-K branch summaries + global branch diagnostics.

    Per branch:
        Mass, Mean_g, CosMeanPhi, SinMeanPhi, Mean_A,
        LogStd_g, LogStd_phi, LogStd_A,
        Corr_gphi, Corr_gA, Corr_phiA

    Extras:
        BranchEntropy, QGap12, StepOverMaxStep, ResOverMaxRes
    """

    FEATS_PER_BRANCH = 11

    def __init__(
        self,
        particle_filter: GravityBranchBankParticleFilter,
        phys_model: GravityStatelessPhysicalModel,
        control_strategy,
        simpars: SimulationParameters,
        cov_weight_matrix: Optional[list[list[float]]] = None,
    ) -> None:
        super().__init__(particle_filter, phys_model, control_strategy, simpars, cov_weight_matrix)
        K = particle_filter.K
        self.input_size = K * self.FEATS_PER_BRANCH + 4

        names = []
        for k in range(K):
            prefix = f"Branch{k+1}"
            names += [
                f"{prefix}_Mass",
                f"{prefix}_Mean_g",
                f"{prefix}_CosMeanPhi",
                f"{prefix}_SinMeanPhi",
                f"{prefix}_Mean_A",
                f"{prefix}_LogStd_g",
                f"{prefix}_LogStd_phi",
                f"{prefix}_LogStd_A",
                f"{prefix}_Corr_gphi",
                f"{prefix}_Corr_gA",
                f"{prefix}_Corr_phiA",
            ]
        names += ["BranchEntropy", "QGap12", "StepOverMaxStep", "ResOverMaxRes"]
        self.input_name = names

    def generate_input(self, weights: Tensor, particles: Tensor, meas_step: Tensor, used_resources: Tensor, rangen) -> Tensor:
        del rangen
        pf = self.pf
        cfg = self.phys_model.cfg
        eps = tf.cast(1e-18, cfg.prec)

        qk, mk, sk, ck = pf.topk_branch_summary(weights, particles, K=pf.K)
        qk = tf.math.divide_no_nan(qk, tf.reduce_sum(qk, axis=1, keepdims=True))

        mass_feat = _prob01_to_pm1(tf.clip_by_value(qk, 0.0, 1.0))
        mean_g = normalize_to_minus1_plus1(mk[:, :, 0], cfg.g_range)
        mean_phi_cos = tf.cos(mk[:, :, 1])
        mean_phi_sin = tf.sin(mk[:, :, 1])
        mean_A = normalize_to_minus1_plus1(mk[:, :, 2], cfg.A_range)

        log_std_g = normalize_to_minus1_plus1(tf.math.log(sk[:, :, 0] + eps), (-10.0, 0.0))
        log_std_phi = normalize_to_minus1_plus1(tf.math.log(sk[:, :, 1] + eps), (-10.0, 0.0))
        log_std_A = normalize_to_minus1_plus1(tf.math.log(sk[:, :, 2] + eps), (-10.0, 0.0))

        corr_gphi = tf.clip_by_value(ck[:, :, 0, 1], -1.0, 1.0)
        corr_gA = tf.clip_by_value(ck[:, :, 0, 2], -1.0, 1.0)
        corr_phiA = tf.clip_by_value(ck[:, :, 1, 2], -1.0, 1.0)

        branch_feats = tf.stack(
            [
                mass_feat,
                mean_g,
                mean_phi_cos,
                mean_phi_sin,
                mean_A,
                log_std_g,
                log_std_phi,
                log_std_A,
                corr_gphi,
                corr_gA,
                corr_phiA,
            ],
            axis=2,
        )
        branch_feats = tf.reshape(branch_feats, (self.bs, pf.K * self.FEATS_PER_BRANCH))

        q_all = pf.branch_masses(weights)
        q_all = tf.math.divide_no_nan(q_all, tf.reduce_sum(q_all, axis=1, keepdims=True))
        entropy = -tf.reduce_sum(q_all * tf.math.log(q_all + eps), axis=1, keepdims=True)
        entropy = tf.math.divide_no_nan(entropy, tf.cast(tf.math.log(float(max(2, pf.L))), entropy.dtype))
        entropy = _prob01_to_pm1(tf.clip_by_value(entropy, 0.0, 1.0))

        q_sorted = tf.sort(q_all, axis=1, direction="DESCENDING")
        qgap = q_sorted[:, 0:1] - q_sorted[:, 1:2] if pf.L > 1 else tf.ones((self.bs, 1), dtype=cfg.prec)
        qgap = _prob01_to_pm1(tf.clip_by_value(qgap, 0.0, 1.0))

        step_scaled = normalize_to_minus1_plus1(meas_step, (0.0, float(self.simpars.num_steps)))
        res_scaled = normalize_to_minus1_plus1(used_resources, (0.0, float(self.simpars.max_resources)))

        return tf.concat([branch_feats, entropy, qgap, step_scaled, res_scaled], axis=1)


# -----------------------------------------------------------------------------
# Branch-aware control strategy
# -----------------------------------------------------------------------------

class BranchAwareGravityControlStrategy(Model):
    FEATS_PER_BRANCH = 11

    def __init__(
        self,
        cfg: GravimeterConfig,
        input_size: int,
        num_top_branches: int,
        hidden_sizes: Sequence[int] = (128, 128, 128, 128),
    ) -> None:
        super().__init__(dtype=cfg.prec)
        self.cfg = cfg
        self.K = int(num_top_branches)
        self.hidden_layers = [Dense(h, activation="tanh", dtype=cfg.prec) for h in hidden_sizes]
        self.out_layer = Dense(3, activation="tanh", dtype=cfg.prec)
        self(tf.zeros((1, input_size), dtype=cfg.prec))

    def _map_log_interval(self, z: Tensor, low: float, high: float) -> Tensor:
        lo = tf.cast(tf.math.log(low), z.dtype)
        hi = tf.cast(tf.math.log(high), z.dtype)
        return tf.exp(lo + 0.5 * (z + 1.0) * (hi - lo))

    def _k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        Bp_T_per_m = Bp_kTm * tf.cast(self.cfg.kT_to_T, T_s.dtype)
        w = tf.cast(self.cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(self.cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * (T_s ** 2) + (
            8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)
        ) * Bp_T_per_m

    def _decode_branch_block(self, x: Tensor):
        block = x[:, : self.K * self.FEATS_PER_BRANCH]
        block = tf.reshape(block, (tf.shape(x)[0], self.K, self.FEATS_PER_BRANCH))

        q = tf.clip_by_value(0.5 * (block[:, :, 0] + 1.0), 0.0, 1.0)
        q = tf.math.divide_no_nan(q, tf.reduce_sum(q, axis=1, keepdims=True))
        g = denormalize_from_minus1_plus1(block[:, :, 1], self.cfg.g_range)
        phi = tf.atan2(block[:, :, 3], block[:, :, 2])
        A = denormalize_from_minus1_plus1(block[:, :, 4], self.cfg.A_range)
        return q, g, phi, A

    def call(self, input_strategy: Tensor) -> Tensor:
        x = input_strategy
        for layer in self.hidden_layers:
            x = layer(x)
        z = self.out_layer(x)

        zT = z[:, 0:1]
        zB = z[:, 1:2]
        zD = z[:, 2:3]

        T_s = self._map_log_interval(zT, self.cfg.T_range_s[0], self.cfg.T_range_s[1])
        Bp_kTm = self._map_log_interval(zB, self.cfg.Bp_range_kTm[0], self.cfg.Bp_range_kTm[1])
        delta = tf.cast(self.cfg.delta_max_rad, zD.dtype) * zD

        q, g, phi, A = self._decode_branch_block(input_strategy)
        kg = self._k_g(T_s, Bp_kTm)
        phase = kg * g + phi

        c = tf.reduce_sum(q * A * tf.cos(phase), axis=1, keepdims=True)
        s = tf.reduce_sum(q * A * tf.sin(phase), axis=1, keepdims=True)
        amp = tf.sqrt(c * c + s * s)
        mixture_phase = tf.atan2(s, c)
        dom_phase = phase[:, 0:1]
        phase_ref = tf.where(amp > tf.cast(1e-8, amp.dtype), mixture_phase, dom_phase)

        phi_lock = tf.cast(pi / 2.0, phase_ref.dtype) - phase_ref
        mw_phase = wrap_to_pi_tf(phi_lock + delta)
        return tf.concat([T_s, Bp_kTm, mw_phase], axis=1)


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def build_branchbank_gravity_simulation(
    batchsize: int,
    sim_name: str,
    cfg: GravimeterConfig,
    bank_cfg: BranchBankConfig,
    *,
    max_steps: int = 256,
    max_resources: float = 0.08,
    resources_fraction: float = 1.0,
    pf_alpha: float = 0.5,
    pf_beta: float = 0.98,
    pf_gamma: float = 1.0,
    resample_threshold: float = 0.5,
    resample_fraction: float = 0.98,
    initial_lr: float = 3e-4,
    cov_weight_matrix: Optional[list[list[float]]] = None,
    cumulative_loss: bool = True,
    loss_logl_outcomes: bool = True,
    baseline_correction: bool = True,
    loss_logl_controls: bool = False,
):
    phys = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)

    pf = GravityBranchBankParticleFilter(
        phys_model=phys,
        bank_cfg=bank_cfg,
        resampling_allowed=True,
        resample_threshold=resample_threshold,
        resample_fraction=resample_fraction,
        alpha=pf_alpha,
        beta=pf_beta,
        gamma=pf_gamma,
        scibior_trick=True,
        trim=True,
        prec=cfg.prec,
    )

    simpars = SimulationParameters(
        sim_name=sim_name,
        num_steps=max_steps,
        max_resources=max_resources,
        resources_fraction=resources_fraction,
        prec=cfg.prec,
        stop_gradient_input=True,
        stop_gradient_pf=False,
        cumulative_loss=cumulative_loss,
        log_loss=False,
        loss_logl_outcomes=loss_logl_outcomes,
        loss_logl_controls=loss_logl_controls,
        baseline=baseline_correction,
    )

    dummy_net = lambda x: tf.zeros((tf.shape(x)[0], 3), dtype=cfg.prec)
    sim_tmp = BranchAwareGravityMetrology(
        particle_filter=pf,
        phys_model=phys,
        control_strategy=dummy_net,
        simpars=simpars,
        cov_weight_matrix=cov_weight_matrix,
    )

    net = BranchAwareGravityControlStrategy(
        cfg=cfg,
        input_size=sim_tmp.input_size,
        num_top_branches=pf.K,
        hidden_sizes=bank_cfg.hidden_sizes,
    )

    sim = BranchAwareGravityMetrology(
        particle_filter=pf,
        phys_model=phys,
        control_strategy=net,
        simpars=simpars,
        cov_weight_matrix=cov_weight_matrix,
    )

    lr_schedule = InverseSqrtDecay(initial_learning_rate=initial_lr, prec=cfg.prec)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return phys, pf, sim, net, optimizer


# -----------------------------------------------------------------------------
# Export canonicalization
# -----------------------------------------------------------------------------

def _canonicalize_history_export(out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_history.csv")
    if src is None:
        return None
    df = pd.read_csv(src)
    if "Checkpoint" not in df.columns:
        df.insert(0, "Checkpoint", np.arange(1, len(df) + 1, dtype=np.int64))
    dst = out_dir / "training_history.csv"
    df.to_csv(dst, index=False)
    return dst


def _canonicalize_eval_export(out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_eval.csv")
    if src is None:
        return None
    df = pd.read_csv(src)
    ordered = [c for c in ["Resources", "Weighted MSE"] if c in df.columns]
    if ordered:
        df = df[ordered]
    dst = out_dir / "branchbank_eval.csv"
    df.to_csv(dst, index=False)
    return dst


def _canonicalize_control_export(sim: BranchAwareGravityMetrology, out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_ext.csv")
    if src is None:
        return None

    df = pd.read_csv(src)
    cfg = sim.phys_model.cfg
    K = sim.pf.K

    out: dict[str, np.ndarray] = {}
    for col in ["Estimation", "g", "phi_off", "A"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    branch_masses = []
    branch_mean_g = []
    branch_mean_phi = []
    branch_mean_A = []
    branch_std_g = []
    branch_std_phi = []
    branch_std_A = []

    for k in range(1, K + 1):
        pref = f"Branch{k}"
        mass_col = f"{pref}_Mass"
        mean_g_col = f"{pref}_Mean_g"
        cos_phi_col = f"{pref}_CosMeanPhi"
        sin_phi_col = f"{pref}_SinMeanPhi"
        mean_A_col = f"{pref}_Mean_A"
        logstd_g_col = f"{pref}_LogStd_g"
        logstd_phi_col = f"{pref}_LogStd_phi"
        logstd_A_col = f"{pref}_LogStd_A"

        if mass_col in df.columns:
            mass = np.clip(_pm1_to_prob01_np(df[mass_col].to_numpy()), 0.0, 1.0)
        else:
            mass = np.zeros(len(df), dtype=float)
        if mean_g_col in df.columns:
            mg = _decode_affine_np(df[mean_g_col].to_numpy(), cfg.g_range)
        else:
            mg = np.full(len(df), np.nan)
        if cos_phi_col in df.columns and sin_phi_col in df.columns:
            mphi = np.arctan2(df[sin_phi_col].to_numpy(), df[cos_phi_col].to_numpy())
        else:
            mphi = np.full(len(df), np.nan)
        if mean_A_col in df.columns:
            mA = _decode_affine_np(df[mean_A_col].to_numpy(), cfg.A_range)
        else:
            mA = np.full(len(df), np.nan)
        if logstd_g_col in df.columns:
            sg = _decode_logstd_np(df[logstd_g_col].to_numpy())
        else:
            sg = np.full(len(df), np.nan)
        if logstd_phi_col in df.columns:
            sphi = _decode_logstd_np(df[logstd_phi_col].to_numpy())
        else:
            sphi = np.full(len(df), np.nan)
        if logstd_A_col in df.columns:
            sA = _decode_logstd_np(df[logstd_A_col].to_numpy())
        else:
            sA = np.full(len(df), np.nan)

        out[f"{pref}_Mass"] = mass
        out[f"{pref}_Mean_g"] = mg
        out[f"{pref}_Mean_phi_off"] = mphi
        out[f"{pref}_Mean_A"] = mA
        out[f"{pref}_Std_g"] = sg
        out[f"{pref}_Std_phi_off"] = sphi
        out[f"{pref}_Std_A"] = sA

        for corr_name in ["Corr_gphi", "Corr_gA", "Corr_phiA"]:
            col = f"{pref}_{corr_name}"
            if col in df.columns:
                out[col] = df[col].to_numpy()

        branch_masses.append(mass)
        branch_mean_g.append(mg)
        branch_mean_phi.append(mphi)
        branch_mean_A.append(mA)
        branch_std_g.append(sg)
        branch_std_phi.append(sphi)
        branch_std_A.append(sA)

    if branch_masses:
        q = np.stack(branch_masses, axis=1)
        q = np.clip(q, 1e-18, None)
        q = q / q.sum(axis=1, keepdims=True)

        mg = np.stack(branch_mean_g, axis=1)
        mphi = np.stack(branch_mean_phi, axis=1)
        mA = np.stack(branch_mean_A, axis=1)
        sg = np.stack(branch_std_g, axis=1)
        sphi = np.stack(branch_std_phi, axis=1)
        sA = np.stack(branch_std_A, axis=1)

        global_mean_g = np.sum(q * mg, axis=1)
        global_mean_A = np.sum(q * mA, axis=1)
        global_mean_phi = np.arctan2(np.sum(q * np.sin(mphi), axis=1), np.sum(q * np.cos(mphi), axis=1))

        global_var_g = np.sum(q * (sg ** 2 + (mg - global_mean_g[:, None]) ** 2), axis=1)
        global_var_A = np.sum(q * (sA ** 2 + (mA - global_mean_A[:, None]) ** 2), axis=1)
        phi_dev = np.angle(np.exp(1j * (mphi - global_mean_phi[:, None])))
        global_var_phi = np.sum(q * (sphi ** 2 + phi_dev ** 2), axis=1)

        out["Mean_g"] = global_mean_g
        out["Mean_phi_off"] = global_mean_phi
        out["Mean_A"] = global_mean_A
        out["Std_g"] = np.sqrt(np.maximum(global_var_g, 0.0))
        out["Std_phi_off"] = np.sqrt(np.maximum(global_var_phi, 0.0))
        out["Std_A"] = np.sqrt(np.maximum(global_var_A, 0.0))
        out["BranchEntropy"] = -(q * np.log(q + 1e-18)).sum(axis=1) / np.log(max(2, q.shape[1]))
        out["BranchDominance"] = q.max(axis=1)
        out["DominantBranchIndex"] = q.argmax(axis=1)
        out["QGap12"] = q[:, 0] - q[:, 1] if q.shape[1] > 1 else np.ones(len(df))

    for col in ["BranchEntropy", "QGap12", "StepOverMaxStep", "ResOverMaxRes", "T_s", "Bp_kTm", "mw_phase_rad"]:
        if col in df.columns and col not in out:
            out[col] = df[col].to_numpy()

    dst = out_dir / "branchbank_controls.csv"
    pd.DataFrame(out).to_csv(dst, index=False)
    return dst


# -----------------------------------------------------------------------------
# One-call train / export / eval wrappers
# -----------------------------------------------------------------------------

def train_branchbank_gravity_modelaware(
    out_dir: str | Path,
    *,
    batchsize: int = 128,
    iterations: int = 2000,
    interval_save: int = 128,
    cfg: GravimeterConfig = GravimeterConfig(),
    bank_cfg: BranchBankConfig = BranchBankConfig(),
    sim_name: str = "gravimeter_branchbank_modelaware",
    max_steps: int = 256,
    max_resources: float = 0.08,
    initial_lr: float = 3e-4,
    seed: int = 123,
    cumulative_loss: bool = True,
    loss_logl_outcomes: bool = True,
    baseline_correction: bool = True,
    gradient_accumulation: int = 4,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_weight_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
    ]

    phys, pf, sim, net, optimizer = build_branchbank_gravity_simulation(
        batchsize=batchsize,
        sim_name=sim_name,
        cfg=cfg,
        bank_cfg=bank_cfg,
        max_steps=max_steps,
        max_resources=max_resources,
        initial_lr=initial_lr,
        cov_weight_matrix=cov_weight_matrix,
        cumulative_loss=cumulative_loss,
        loss_logl_outcomes=loss_logl_outcomes,
        baseline_correction=baseline_correction,
        pf_gamma=1.0,
    )

    rangen = tf.random.Generator.from_seed(seed)
    train(
        simulation=sim,
        optimizer=optimizer,
        iterations=iterations,
        save_path=str(out_dir),
        interval_save=interval_save,
        network=net,
        gradient_accumulation=gradient_accumulation,
        xla_compile=False,
        rangen=rangen,
    )

    _canonicalize_history_export(out_dir)
    return phys, pf, sim, net, optimizer


def export_branchbank_control_history(sim, out_dir: str | Path, iterations: int = 32, seed: int = 999):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rangen = tf.random.Generator.from_seed(seed)
    store_input_control(
        simulation=sim,
        data_dir=str(out_dir),
        iterations=iterations,
        xla_compile=False,
        rangen=rangen,
    )
    return _canonicalize_control_export(sim=sim, out_dir=out_dir)


def evaluate_branchbank_precision(sim, out_dir: str | Path, iterations: int = 64, seed: int = 2025, delta_resources: float | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rangen = tf.random.Generator.from_seed(seed)

    if delta_resources is None:
        delta_resources = float(sim.simpars.max_resources) / 40.0

    performance_evaluation(
        simulation=sim,
        iterations=iterations,
        data_dir=str(out_dir),
        xla_compile=False,
        precision_fit=None,
        delta_resources=delta_resources,
        y_label="Weighted MSE",
        rangen=rangen,
    )
    return _canonicalize_eval_export(out_dir)


# -----------------------------------------------------------------------------
# Convenience defaults + config dump helper
# -----------------------------------------------------------------------------

def default_cfg() -> GravimeterConfig:
    return GravimeterConfig()


def default_bank_cfg() -> BranchBankConfig:
    return BranchBankConfig(num_branches=4, particles_per_branch=512, top_k_branches=4)


def dump_run_config(path: str | Path, *, cfg: GravimeterConfig, bank_cfg: BranchBankConfig, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gravimeter_config": asdict(cfg),
        "branch_bank_config": asdict(bank_cfg),
        **kwargs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path