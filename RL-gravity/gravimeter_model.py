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
print("TensorFlow version:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print(f"Could not set memory growth for {gpu}: {exc}")

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
BRANCH_FEATS_PER_BRANCH = 12
BRANCH_EXTRA_FEATS = 11

def _decode_unit_interval_np(x: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (x + 1.0), 0.0, 1.0)

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
    g_range: tuple[float, float] = (9.7806 , 9.825)
    A_range: tuple[float, float] = (0.80, 1.0)

    # control ranges
    T_range_s: tuple[float, float] = (1.0e-4, 6.0e-4)
    Bp_range_kTm: tuple[float, float] = (2.0, 20.0)
    delta_max_rad: float = pi / 2.0

    # hidden-noise model for true measurement simulation
    mfg_rel_noise_bound: float = 0.025
    mfg_noise_quad_points: int = 17
    sigma_omega_rel: float = 0.001
    trap_visibility_mode: str = "small_noise_avg"   # none | small_noise_avg | exact_single_delta
    T2_spin_s: Optional[float] = None
    readout_flip_prob: float = 0.0

    # resource model
    dead_time_s: float = 0.0

    # tensorflow precision
    prec: str = "float32"

    @property
    def tau_s(self) -> float:
        return 2.0 * pi / self.omega_rad_s

@dataclass(frozen=True)
class BranchBankConfig:
    num_branches: int = 4
    particles_per_branch: int = 512
    init_mode: str = "stratified_g"   # only supported mode in the (g, A) model
    hidden_sizes: tuple[int, ...] = (128, 128, 128, 128)

# -----------------------------------------------------------------------------
# Gravity physical model (stateless probe)
# -----------------------------------------------------------------------------

class GravityStatelessPhysicalModel(StatelessPhysicalModel):
    """
    Unknown parameters:
        theta = (g, A)

    Controls:
        x = (T_s, Bp_kTm, mw_phase_rad)

    Likelihood:
        p(y=1 | theta, x) = 1/2 [1 + V(T,B) * A * cos((1+eps) k_g(T,B) g + mw_phase)]
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
            delta_omega = rangen.normal(
                tf.shape(T_s),
                dtype=T_s.dtype,
            ) * tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, T_s.dtype)
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
        A = parameters[:, :, 1]

        vis_known = self.known_visibility_factor(T_s, Bp_kTm)
        total_vis = tf.clip_by_value(A * vis_known, 0.0, 1.0)
        kg = self.k_g(T_s, Bp_kTm)

        eps_grid = tf.cast(self.mfg_quadrature(), T_s.dtype)
        theta = (
            tf.expand_dims(kg * g, axis=2) * (1.0 + eps_grid[None, None, :])
            + tf.expand_dims(mw_phase, axis=2)
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
        A = parameters[:, 0, 1]

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

        theta = (1.0 + eps_true) * self.k_g(T_s, Bp_kTm) * g + mw_phase
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

        T_s = controls[..., 0]
        if T_s.shape.rank == 1:
            T_s = T_s[:, None]

        return resources + self.cycle_time_s(T_s)


class GravityBranchBankParticleFilter(ParticleFilter):
    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank_cfg: BranchBankConfig,
        *,
        resampling_allowed: bool = True,
        resample_threshold: float = 0.6,
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

        if self.bank_cfg.init_mode != "stratified_g":
            raise ValueError(f"Unknown init_mode={self.bank_cfg.init_mode}")

        g_edges = tf.linspace(tf.cast(cfg.g_range[0], dtype), tf.cast(cfg.g_range[1], dtype), self.L + 1)
        g_lo = g_edges[:-1][None, :, None, None]
        g_hi = g_edges[1:][None, :, None, None]

        ug = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)
        uA = rangen.uniform((self.bs, self.L, self.npb, 1), dtype=dtype)

        g = g_lo + ug * (g_hi - g_lo)
        A = tf.cast(cfg.A_range[0], dtype) + uA * tf.cast(cfg.A_range[1] - cfg.A_range[0], dtype)

        particles = tf.concat([g, A], axis=3)
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
        A = p[:, :, :, 1]

        mu_g = tf.reduce_sum(wloc * g, axis=2)
        mu_A = tf.reduce_sum(wloc * A, axis=2)

        d_g = g - mu_g[:, :, None]
        d_A = A - mu_A[:, :, None]

        centered = tf.stack([d_g, d_A], axis=-1)
        cov = tf.einsum("bln,blni,blnj->blij", wloc, centered, centered)
        var = tf.linalg.diag_part(cov)
        std = tf.sqrt(tf.abs(var) + tf.cast(1e-18, cov.dtype))
        denom = std[:, :, :, None] * std[:, :, None, :]
        corr = tf.math.divide_no_nan(cov, denom)

        means = tf.stack([mu_g, mu_A], axis=-1)
        return q, means, std, corr

    # ---- branch-local resampling ----
    def full_resampling(self, weights: Tensor, particles: Tensor, count_for_resampling: Tensor, rangen):
        wbank = self._bank_weights(weights)          # (bs, L, npb)
        pbank = self._bank_particles(particles)      # (bs, L, npb, d)

        q = tf.reduce_sum(wbank, axis=2, keepdims=True)         # (bs, L, 1)
        wloc = tf.math.divide_no_nan(wbank, q)                  # (bs, L, npb)

        eff_n = tf.math.divide_no_nan(
            tf.ones_like(tf.reduce_sum(wloc * wloc, axis=2)),
            tf.reduce_sum(wloc * wloc, axis=2),
        )  # (bs, L)

        active = tf.squeeze(count_for_resampling, axis=1)       # (bs,)
        q_scalar = tf.squeeze(q, axis=2)                        # (bs, L)
        mass_alive = q_scalar > tf.cast(1e-12, q_scalar.dtype)  # (bs, L)

        need_branch = eff_n < tf.cast(self.res_thres * self.npb, eff_n.dtype)  # (bs, L)
        branch_mask = tf.logical_and(
            tf.logical_and(need_branch, mass_alive),
            active[:, None],
        )  # (bs, L)

        need_run = tf.reduce_any(branch_mask, axis=1)  # (bs,)

        # num_active = tf.cast(tf.math.count_nonzero(active), tf.float64)
        # num_trigger = tf.cast(tf.math.count_nonzero(need_run), tf.float64)

        # resample_flag = tf.logical_and(
        #     num_active > 0.0,
        #     num_trigger >= self.res_frac * num_active,
        # )

        resample_flag = tf.reduce_any(branch_mask)
        
        def _do_resample():
            wloc_prop, pbank_prop = self._resample_all_branches(wloc, pbank, rangen)

            wloc_new = tf.where(branch_mask[:, :, None], wloc_prop, wloc)
            pbank_new = tf.where(branch_mask[:, :, None, None], pbank_prop, pbank)

            wbank_new = wloc_new * q
            return self._flat_weights(wbank_new), self._flat_particles(pbank_new)

        weights, particles = tf.cond(
            resample_flag,
            _do_resample,
            lambda: (weights, particles),
        )
        return weights, particles, resample_flag
    
    
    def _resample_all_branches(self, wloc: Tensor, pbank: Tensor, rangen):
        bsL = self.bs * self.L
        w = tf.reshape(wloc, (bsL, self.npb))
        p = tf.reshape(pbank, (bsL, self.npb, self.d))

        g = p[:, :, 0]
        A = p[:, :, 1]

        mu_g = tf.reduce_sum(w * g, axis=1, keepdims=True)
        mu_A = tf.reduce_sum(w * A, axis=1, keepdims=True)

        cg = g - mu_g
        cA = A - mu_A
        centered = tf.stack([cg, cA], axis=2)

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
        p_sel_A = p_sel[:, :, 1] - mu_A
        p_sel_centered = tf.stack([p_sel_g, p_sel_A], axis=2)

        noise_seed = get_seed(rangen)
        noise = tf.random.stateless_normal((bsL, self.local_nrp, self.d), noise_seed, dtype=w.dtype)

        delta = tf.einsum("bij,bnj->bni", dev, noise) * tf.cast(tf.sqrt(1.0 - self.beta ** 2), w.dtype)

        p_soft_centered = self.beta * p_sel_centered + delta

        p_soft = tf.stack(
            [
                mu_g + p_soft_centered[:, :, 0],
                mu_A + p_soft_centered[:, :, 1],
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
                    mu_A + extra_centered[:, :, 1],
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
        A = tf.clip_by_value(
            particles[:, :, 1:2],
            tf.cast(cfg.A_range[0], particles.dtype),
            tf.cast(cfg.A_range[1], particles.dtype),
        )
        return tf.concat([g, A], axis=2)


class BranchAwareGravityMetrology(StatelessMetrology):

    FEATS_PER_BRANCH = BRANCH_FEATS_PER_BRANCH
    EXTRA_FEATS = BRANCH_EXTRA_FEATS

    def __init__(
        self,
        particle_filter: GravityBranchBankParticleFilter,
        phys_model: GravityStatelessPhysicalModel,
        control_strategy,
        simpars: SimulationParameters,
        cov_weight_matrix: Optional[list[list[float]]] = None,
    ) -> None:
        super().__init__(particle_filter, phys_model, control_strategy, simpars, cov_weight_matrix)

        L = particle_filter.L
        self.input_size = L * self.FEATS_PER_BRANCH + self.EXTRA_FEATS

        names = []
        for k in range(L):
            prefix = f"Branch{k+1}"
            names += [
                f"{prefix}_Mass",
                f"{prefix}_Mean_g",
                f"{prefix}_Mean_A",
                f"{prefix}_LogStd_g",
                f"{prefix}_LogStd_A",
                f"{prefix}_Corr_gA",
                f"{prefix}_PhaseLowCos",
                f"{prefix}_PhaseLowSin",
                f"{prefix}_PhaseMidCos",
                f"{prefix}_PhaseMidSin",
                f"{prefix}_PhaseHighCos",
                f"{prefix}_PhaseHighSin",
            ]

        names += [
            "BranchEntropy",
            "QGap12",
            "StepOverMaxStep",
            "ResOverMaxRes",
            "Gap12_g",
            "Gap12_psi_low_cos",
            "Gap12_psi_low_sin",
            "Gap12_psi_mid_cos",
            "Gap12_psi_mid_sin",
            "Gap12_psi_high_cos",
            "Gap12_psi_high_sin",
        ]
        self.input_name = names

    def _phase_anchor_gains(self, dtype):
        cfg = self.phys_model.cfg

        T_min, T_max = cfg.T_range_s
        B_min, B_max = cfg.Bp_range_kTm
        T_mid = float(np.sqrt(T_min * T_max))
        B_mid = float(np.sqrt(B_min * B_max))

        def _scalar_kg(T_ref: float, B_ref: float):
            T_t = tf.constant([T_ref], dtype=dtype)
            B_t = tf.constant([B_ref], dtype=dtype)
            return self.phys_model.k_g(T_t, B_t)[0]

        k_low = _scalar_kg(T_min, B_min)
        k_mid = _scalar_kg(T_mid, B_mid)
        k_high = _scalar_kg(T_max, B_max)
        return k_low, k_mid, k_high

    def generate_input(
        self,
        weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen,
    ) -> Tensor:
        del rangen
        pf = self.pf
        cfg = self.phys_model.cfg
        eps = tf.cast(1e-18, cfg.prec)

        q_all, means_all, std_all, corr_all = pf.branch_statistics(weights, particles)
        q_all = tf.math.divide_no_nan(q_all, tf.reduce_sum(q_all, axis=1, keepdims=True))

        mu_g = means_all[:, :, 0]
        mu_A = means_all[:, :, 1]

        k_low, k_mid, k_high = self._phase_anchor_gains(cfg.prec)
        psi_low = wrap_to_pi_tf(k_low * mu_g)
        psi_mid = wrap_to_pi_tf(k_mid * mu_g)
        psi_high = wrap_to_pi_tf(k_high * mu_g)

        mass_feat = _prob01_to_pm1(tf.clip_by_value(q_all, 0.0, 1.0))
        mean_g = normalize_to_minus1_plus1(mu_g, cfg.g_range)
        mean_A = normalize_to_minus1_plus1(mu_A, cfg.A_range)

        log_std_g = normalize_to_minus1_plus1(tf.math.log(std_all[:, :, 0] + eps), (-10.0, 0.0))
        log_std_A = normalize_to_minus1_plus1(tf.math.log(std_all[:, :, 1] + eps), (-10.0, 0.0))

        corr_gA = tf.clip_by_value(corr_all[:, :, 0, 1], -1.0, 1.0)

        branch_feats = tf.stack(
            [
                mass_feat,
                mean_g,
                mean_A,
                log_std_g,
                log_std_A,
                corr_gA,
                tf.cos(psi_low),
                tf.sin(psi_low),
                tf.cos(psi_mid),
                tf.sin(psi_mid),
                tf.cos(psi_high),
                tf.sin(psi_high),
            ],
            axis=2,
        )
        branch_feats = tf.reshape(branch_feats, (self.bs, pf.L * self.FEATS_PER_BRANCH))

        entropy = -tf.reduce_sum(q_all * tf.math.log(q_all + eps), axis=1, keepdims=True)
        entropy = tf.math.divide_no_nan(
            entropy,
            tf.cast(tf.math.log(float(max(2, pf.L))), entropy.dtype),
        )
        entropy = _prob01_to_pm1(tf.clip_by_value(entropy, 0.0, 1.0))

        q_sorted = tf.sort(q_all, axis=1, direction="DESCENDING")
        qgap = q_sorted[:, 0:1] - q_sorted[:, 1:2] if pf.L > 1 else tf.ones((self.bs, 1), dtype=cfg.prec)
        qgap = _prob01_to_pm1(tf.clip_by_value(qgap, 0.0, 1.0))

        step_scaled = normalize_to_minus1_plus1(meas_step, (0.0, float(self.simpars.num_steps)))
        res_scaled = normalize_to_minus1_plus1(used_resources, (0.0, float(self.simpars.max_resources)))

        if pf.L > 1:
            top2_idx = tf.argsort(q_all, axis=1, direction="DESCENDING")[:, :2]
            means_top2 = tf.gather(means_all, top2_idx, batch_dims=1)

            mu_g_top2 = means_top2[:, :, 0]

            psi_low_top2 = wrap_to_pi_tf(k_low * mu_g_top2)
            psi_mid_top2 = wrap_to_pi_tf(k_mid * mu_g_top2)
            psi_high_top2 = wrap_to_pi_tf(k_high * mu_g_top2)

            dg12 = mu_g_top2[:, 0] - mu_g_top2[:, 1]
            g_span = tf.cast(cfg.g_range[1] - cfg.g_range[0], cfg.prec)
            dg12 = tf.clip_by_value(dg12 / tf.maximum(g_span, eps), -1.0, 1.0)[:, None]

            dpsi_low = wrap_to_pi_tf(psi_low_top2[:, 0] - psi_low_top2[:, 1])[:, None]
            dpsi_mid = wrap_to_pi_tf(psi_mid_top2[:, 0] - psi_mid_top2[:, 1])[:, None]
            dpsi_high = wrap_to_pi_tf(psi_high_top2[:, 0] - psi_high_top2[:, 1])[:, None]

            extra_feats = tf.concat(
                [
                    entropy,
                    qgap,
                    step_scaled,
                    res_scaled,
                    dg12,
                    tf.cos(dpsi_low),
                    tf.sin(dpsi_low),
                    tf.cos(dpsi_mid),
                    tf.sin(dpsi_mid),
                    tf.cos(dpsi_high),
                    tf.sin(dpsi_high),
                ],
                axis=1,
            )
        else:
            extra_feats = tf.concat(
                [
                    entropy,
                    qgap,
                    step_scaled,
                    res_scaled,
                    tf.zeros((self.bs, 7), dtype=cfg.prec),
                ],
                axis=1,
            )

        return tf.concat([branch_feats, extra_feats], axis=1)

class BranchAwareGravityControlStrategy(Model):
    FEATS_PER_BRANCH = BRANCH_FEATS_PER_BRANCH
    EXTRA_FEATS = BRANCH_EXTRA_FEATS

    def __init__(
        self,
        cfg: GravimeterConfig,
        input_size: int,
        num_visible_branches: int,
        hidden_sizes: Sequence[int] = (128, 128, 128, 128),
    ) -> None:
        super().__init__(dtype=cfg.prec)
        self.cfg = cfg
        self.L = int(num_visible_branches)

        self.branch_layers = [
            Dense(64, activation="tanh", dtype=cfg.prec),
            Dense(64, activation="tanh", dtype=cfg.prec),
        ]


        self.trunk_layers = [Dense(h, activation="tanh", dtype=cfg.prec) for h in hidden_sizes]

        self.T_head_layers = [
            Dense(64, activation="tanh", dtype=cfg.prec),
            Dense(1, activation="tanh", dtype=cfg.prec),
        ]

        self.B_head_layers = [
            Dense(64, activation="tanh", dtype=cfg.prec),
            Dense(1, activation="tanh", dtype=cfg.prec),
        ]

        self.phi_head_layers = [
            Dense(64, activation="tanh", dtype=cfg.prec),
            Dense(1, activation="tanh", dtype=cfg.prec),
        ]

        self(tf.zeros((1, input_size), dtype=cfg.prec))

    def _run_head(self, x: Tensor, layers: Sequence[Dense]) -> Tensor:
        h = x
        for layer in layers:
            h = layer(h)
        return h
    
    def _map_log_interval(self, z: Tensor, low: float, high: float) -> Tensor:
        lo = tf.cast(tf.math.log(low), z.dtype)
        hi = tf.cast(tf.math.log(high), z.dtype)
        return tf.exp(lo + 0.5 * (z + 1.0) * (hi - lo))

    def _split_input(self, input_strategy: Tensor):
        branch_flat = input_strategy[:, : self.L * self.FEATS_PER_BRANCH]
        extras = input_strategy[:, self.L * self.FEATS_PER_BRANCH :]

        branch_block = tf.reshape(
            branch_flat,
            (tf.shape(input_strategy)[0], self.L, self.FEATS_PER_BRANCH),
        )

        q = tf.clip_by_value(0.5 * (branch_block[:, :, 0] + 1.0), 0.0, 1.0)
        q = tf.math.divide_no_nan(q, tf.reduce_sum(q, axis=1, keepdims=True))
        return branch_block, q, extras

    def _k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * (T_s ** 2) + (
            8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)
        ) * Bp_T_per_m
    
    def _apply_disambiguation_ceiling(
        self,
        T_s: Tensor,
        Bp_kTm: Tensor,
        qgap01: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Ambiguity-dependent control ceiling.

        qgap01:
            0 -> top-2 branch masses are equal   -> highly ambiguous
            1 -> one branch clearly dominates    -> largely resolved

        When ambiguity is high, keep T and B small to avoid phase aliasing:
            k_g * Δg should stay O(1), not hundreds of radians.

        When ambiguity is low, release the ceiling and allow larger-sensitivity
        controls.
        """
        need_disc = tf.clip_by_value(1.0 - qgap01, 0.0, 1.0)   # 1=ambiguous, 0=resolved
        resolved = 1.0 - need_disc                             # 0=ambiguous, 1=resolved
        
        # Slower release than a linear ramp.
        release = tf.square(resolved)

        logT_lo = tf.cast(tf.math.log(self.cfg.T_range_s[0]), T_s.dtype)
        logT_hi = tf.cast(tf.math.log(self.cfg.T_range_s[1]), T_s.dtype)
        logB_lo = tf.cast(tf.math.log(self.cfg.Bp_range_kTm[0]), Bp_kTm.dtype)
        logB_hi = tf.cast(tf.math.log(self.cfg.Bp_range_kTm[1]), Bp_kTm.dtype)

        # Ceiling is smallest when ambiguous, largest when resolved.
        T_ceil = tf.exp(logT_lo + release * (logT_hi - logT_lo))
        B_ceil = tf.exp(logB_lo + release * (logB_hi - logB_lo))

        T_s = tf.minimum(T_s, T_ceil)
        Bp_kTm = tf.minimum(Bp_kTm, B_ceil)

        return T_s, Bp_kTm, need_disc

    def call(self, input_strategy: Tensor) -> Tensor:
        branch_block, q, extras = self._split_input(input_strategy)

        h = branch_block
        for layer in self.branch_layers:
            h = layer(h)

        h_flat = tf.reshape(h, (tf.shape(h)[0], self.L * tf.shape(h)[2]))
        weighted_pool = tf.reduce_sum(q[:, :, None] * h, axis=1)
        max_pool = tf.reduce_max(h, axis=1)

        order = tf.argsort(q, axis=1, direction="DESCENDING")

        idx1 = order[:, :1]
        top1 = tf.squeeze(tf.gather(h, idx1, batch_dims=1), axis=1)

        if self.L > 1:
            idx2 = order[:, 1:2]
            top2 = tf.squeeze(tf.gather(h, idx2, batch_dims=1), axis=1)
        else:
            top2 = tf.zeros_like(top1)

        pair = top1 - top2

        x = tf.concat(
            [
                h_flat,
                weighted_pool,
                max_pool,
                top1,
                top2,
                pair,
                extras,
            ],
            axis=1,
        )

        for layer in self.trunk_layers:
            x = layer(x)

        x_shared = x

        zT = self._run_head(x_shared, self.T_head_layers)
        zB = self._run_head(x_shared, self.B_head_layers)
        zD = self._run_head(x_shared, self.phi_head_layers)

        # Raw network controls in physical ranges.
        T_s = self._map_log_interval(zT, self.cfg.T_range_s[0], self.cfg.T_range_s[1])
        Bp_kTm = self._map_log_interval(zB, self.cfg.Bp_range_kTm[0], self.cfg.Bp_range_kTm[1])

        mean_g_pm1 = branch_block[:, :, 1]
        mean_A_pm1 = branch_block[:, :, 2]

        mu_g = denormalize_from_minus1_plus1(mean_g_pm1, self.cfg.g_range)
        mu_A = denormalize_from_minus1_plus1(mean_A_pm1, self.cfg.A_range)

        # extras[1] = QGap12 encoded to [-1, +1]
        qgap01 = tf.clip_by_value(0.5 * (extras[:, 1:2] + 1.0), 0.0, 1.0)

        T_s, Bp_kTm, need_disc = self._apply_disambiguation_ceiling(
            T_s=T_s,
            Bp_kTm=Bp_kTm,
            qgap01=qgap01,
        )

        kg = self._k_g(T_s, Bp_kTm)   # (bs, 1)
        psi = kg * mu_g               # (bs, L)

        # Global phase lock.
        c_re = tf.reduce_sum(q * mu_A * tf.cos(psi), axis=1, keepdims=True)
        c_im = tf.reduce_sum(q * mu_A * tf.sin(psi), axis=1, keepdims=True)
        phase_lock_global = wrap_to_pi_tf(
            tf.cast(pi / 2.0, c_re.dtype) - tf.atan2(c_im, c_re)
        )

        # Top-2 branch discriminative lock.
        if self.L > 1:
            top2_mu_g = tf.gather(mu_g, order[:, :2], batch_dims=1)   # (bs, 2)
            psi_top2 = kg * top2_mu_g                                 # (bs, 2)

            # Circular geodesic midpoint between the top-2 branch phases.
            dpsi = wrap_to_pi_tf(psi_top2[:, 0:1] - psi_top2[:, 1:2])
            psi_disc_mid = wrap_to_pi_tf(psi_top2[:, 1:2] + 0.5 * dpsi)

            phase_lock_disc = wrap_to_pi_tf(
                tf.cast(pi / 2.0, psi_disc_mid.dtype) - psi_disc_mid
            )
        else:
            phase_lock_disc = phase_lock_global

        gate = need_disc

        phase_target = wrap_to_pi_tf(
            tf.atan2(
                (1.0 - gate) * tf.sin(phase_lock_global) + gate * tf.sin(phase_lock_disc),
                (1.0 - gate) * tf.cos(phase_lock_global) + gate * tf.cos(phase_lock_disc),
            )
        )

        delta = tf.cast(self.cfg.delta_max_rad, zD.dtype) * zD
        mw_phase = wrap_to_pi_tf(phase_target + delta)

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
    resample_threshold: float = 0.6,
    resample_fraction: float = 0.98,
    initial_lr: float = 3e-4,
    cov_weight_matrix: Optional[list[list[float]]] = None,
    cumulative_loss: bool = False,
    log_loss: bool = False,
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
        log_loss=log_loss,
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
        num_visible_branches=pf.L,
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

    preferred_metric_order = ["MSE_g", "Weighted MSE"]
    metric_cols = [c for c in preferred_metric_order if c in df.columns]

    if not metric_cols:
        metric_cols = [c for c in df.columns if c != "Resources"]

    keep = ["Resources"] + metric_cols[:1] if "Resources" in df.columns and metric_cols else list(df.columns)
    df = df[keep]

    dst = out_dir / "branchbank_eval.csv"
    df.to_csv(dst, index=False)
    return dst


def _canonicalize_control_export(sim: BranchAwareGravityMetrology, out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_ext.csv")
    if src is None:
        return None

    df = pd.read_csv(src)
    cfg = sim.phys_model.cfg
    K = sim.pf.L

    out: dict[str, np.ndarray] = {}
    for col in ["Estimation", "g", "A"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    branch_masses = []
    branch_mean_g = []
    branch_mean_A = []
    branch_std_g = []
    branch_std_A = []

    for k in range(1, K + 1):
        pref = f"Branch{k}"
        mass_col = f"{pref}_Mass"
        mean_g_col = f"{pref}_Mean_g"
        mean_A_col = f"{pref}_Mean_A"
        logstd_g_col = f"{pref}_LogStd_g"
        logstd_A_col = f"{pref}_LogStd_A"

        if mass_col in df.columns:
            mass = np.clip(_pm1_to_prob01_np(df[mass_col].to_numpy()), 0.0, 1.0)
        else:
            mass = np.zeros(len(df), dtype=float)

        if mean_g_col in df.columns:
            mg = _decode_affine_np(df[mean_g_col].to_numpy(), cfg.g_range)
        else:
            mg = np.full(len(df), np.nan)

        if mean_A_col in df.columns:
            mA = _decode_affine_np(df[mean_A_col].to_numpy(), cfg.A_range)
        else:
            mA = np.full(len(df), np.nan)

        if logstd_g_col in df.columns:
            sg = _decode_logstd_np(df[logstd_g_col].to_numpy())
        else:
            sg = np.full(len(df), np.nan)

        if logstd_A_col in df.columns:
            sA = _decode_logstd_np(df[logstd_A_col].to_numpy())
        else:
            sA = np.full(len(df), np.nan)

        out[f"{pref}_Mass"] = mass
        out[f"{pref}_Mean_g"] = mg
        out[f"{pref}_Mean_A"] = mA
        out[f"{pref}_Std_g"] = sg
        out[f"{pref}_Std_A"] = sA

        corr_col = f"{pref}_Corr_gA"
        if corr_col in df.columns:
            out[corr_col] = df[corr_col].to_numpy()

        branch_masses.append(mass)
        branch_mean_g.append(mg)
        branch_mean_A.append(mA)
        branch_std_g.append(sg)
        branch_std_A.append(sA)

    if branch_masses:
        q = np.stack(branch_masses, axis=1)
        q = np.clip(q, 1e-18, None)
        q = q / q.sum(axis=1, keepdims=True)

        mg = np.stack(branch_mean_g, axis=1)
        mA = np.stack(branch_mean_A, axis=1)
        sg = np.stack(branch_std_g, axis=1)
        sA = np.stack(branch_std_A, axis=1)

        global_mean_g = np.sum(q * mg, axis=1)
        global_mean_A = np.sum(q * mA, axis=1)

        global_var_g = np.sum(q * (sg ** 2 + (mg - global_mean_g[:, None]) ** 2), axis=1)
        global_var_A = np.sum(q * (sA ** 2 + (mA - global_mean_A[:, None]) ** 2), axis=1)

        out["Mean_g"] = global_mean_g
        out["Mean_A"] = global_mean_A
        out["Std_g"] = np.sqrt(np.maximum(global_var_g, 0.0))
        out["Std_A"] = np.sqrt(np.maximum(global_var_A, 0.0))

        q_order = np.argsort(-q, axis=1)
        q_sorted = np.take_along_axis(q, q_order, axis=1)

        out["BranchEntropy"] = -(q * np.log(q + 1e-18)).sum(axis=1) / np.log(max(2, q.shape[1]))
        out["BranchDominance"] = q_sorted[:, 0]
        out["DominantBranchIndex"] = q_order[:, 0]

        if q.shape[1] > 1:
            out["QGap12"] = q_sorted[:, 0] - q_sorted[:, 1]
            out["SecondDominantBranchIndex"] = q_order[:, 1]
            out["Top1Mass"] = q_sorted[:, 0]
            out["Top2Mass"] = q_sorted[:, 1]
        else:
            out["QGap12"] = np.ones(len(df))
            out["SecondDominantBranchIndex"] = -np.ones(len(df), dtype=int)
            out["Top1Mass"] = np.ones(len(df))
            out["Top2Mass"] = np.zeros(len(df))

    for col in ["BranchEntropy", "QGap12", "T_s", "Bp_kTm", "mw_phase_rad"]:
        if col in df.columns and col not in out:
            out[col] = df[col].to_numpy()

    if "StepOverMaxStep" in df.columns and "StepOverMaxStep" not in out:
        out["StepOverMaxStep"] = _decode_unit_interval_np(df["StepOverMaxStep"].to_numpy())

    if "ResOverMaxRes" in df.columns and "ResOverMaxRes" not in out:
        out["ResOverMaxRes"] = _decode_unit_interval_np(df["ResOverMaxRes"].to_numpy())
    dst = out_dir / "branchbank_controls.csv"
    pd.DataFrame(out).to_csv(dst, index=False)
    return dst


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
    cumulative_loss: bool = False,
    log_loss: bool = False,
    loss_logl_outcomes: bool = True,
    baseline_correction: bool = True,
    gradient_accumulation: int = 4,
    cov_weight_matrix: Optional[list[list[float]]] = None,
    pf_beta: float = 0.95,
    pf_gamma: float = 0.85,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cov_weight_matrix is None:
        cov_weight_matrix = default_joint_cov_weight_matrix()

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
        log_loss=log_loss,
        loss_logl_outcomes=loss_logl_outcomes,
        baseline_correction=baseline_correction,
        pf_beta=pf_beta,
        pf_gamma=pf_gamma,
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
        xla_compile=True,
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
        xla_compile=True,
        rangen=rangen,
    )
    return _canonicalize_control_export(sim=sim, out_dir=out_dir)


def evaluate_branchbank_precision(
    sim,
    out_dir: str | Path,
    iterations: int = 64,
    seed: int = 2025,
    delta_resources: float | None = None,
    metric_label: str = "Weighted MSE",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rangen = tf.random.Generator.from_seed(seed)

    if delta_resources is None:
        delta_resources = float(sim.simpars.max_resources) / 80.0

    performance_evaluation(
        simulation=sim,
        iterations=iterations,
        data_dir=str(out_dir),
        xla_compile=True,
        precision_fit=None,
        delta_resources=delta_resources,
        y_label=metric_label,
        rangen=rangen,
    )
    return _canonicalize_eval_export(out_dir)


# -----------------------------------------------------------------------------
# Convenience defaults + config dump helper
# -----------------------------------------------------------------------------

def default_joint_cov_weight_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0],
        [0.0, 0.01],
    ]


def gravity_only_cov_weight_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0],
        [0.0, 0.0],
    ]


def default_cfg() -> GravimeterConfig:
    return GravimeterConfig()


def default_bank_cfg() -> BranchBankConfig:
    return BranchBankConfig(num_branches=4, particles_per_branch=512)


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