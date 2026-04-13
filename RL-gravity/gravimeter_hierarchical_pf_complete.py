# gravimeter_hierarchical_pf_complete.py
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

import os as _os
_os.environ.pop("TF_DETERMINISTIC_OPS", None)
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model_complete import GravimeterConfig, GravityStatelessPhysicalModel, wrap_to_pi_tf


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
    import qsensoropt.particle_filter as _pf_mod
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.stateless_simulation import StatelessSimulation
    from qsensoropt.simulation_parameters import SimulationParameters
except Exception:
    _pf_mod = _load_local_qsensoropt_module("particle_filter")
    ParticleFilter = _pf_mod.ParticleFilter
    StatelessSimulation = _load_local_qsensoropt_module("stateless_simulation").StatelessSimulation
    SimulationParameters = _load_local_qsensoropt_module("simulation_parameters").SimulationParameters


def _install_d1_safe_sqrt_hmatrix() -> None:
    old = _pf_mod.sqrt_hmatrix
    def _safe(cov: Tensor) -> Tensor:
        if cov.shape.rank is not None and cov.shape.rank >= 3 and cov.shape[-1] == 1:
            val = tf.sqrt(tf.maximum(cov[..., 0, 0], tf.cast(0.0, cov.dtype)))
            return tf.reshape(val, (tf.shape(cov)[0], 1, 1))
        return old(cov)
    _pf_mod.sqrt_hmatrix = _safe


_install_d1_safe_sqrt_hmatrix()


@dataclass(frozen=True)
class HierarchicalPFConfig:
    n_particles: int = 512
    n_levels: int = 4
    n_disambig_per_level: int = 7
    prec: str = "float32"
    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = True
    trim: bool = True
    hierarchy_bce_weight: float = 1.0
    hierarchy_local_mse_weight: float = 0.05

def _safe_float(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if np.ndim(x) == 0:
        return float(x)
    raise ValueError("Expected scalar")


def _tensor_to_list(x, n=None):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x = np.asarray(x)
    if n is not None:
        x = x[:n]
    return x.tolist()


class HierarchicalPFBank:
    def __init__(self, phys_model: GravityStatelessPhysicalModel, cfg: HierarchicalPFConfig) -> None:
        self.phys_model = phys_model
        self.cfg = cfg
        self.bs = phys_model.bs
        self.prec = cfg.prec
        self._dtype = tf.float32 if cfg.prec == "float32" else tf.float64
        self._int_dtype = tf.int32

        self._g_lo_init = float(phys_model.cfg.g_range[0])
        self._g_hi_init = float(phys_model.cfg.g_range[1])

        self._pf_template = ParticleFilter(
            num_particles=cfg.n_particles,
            phys_model=phys_model,
            resampling_allowed=True,
            resample_threshold=cfg.resample_threshold,
            alpha=cfg.resample_alpha,
            beta=cfg.resample_beta,
            scibior_trick=cfg.scibior_trick,
            trim=cfg.trim,
            prec=cfg.prec,
        )
        self.pf = self._pf_template
        self.d = self._pf_template.d

        self.g_lo_vec: Tensor = tf.zeros((self.bs,), dtype=self._dtype)
        self.g_hi_vec: Tensor = tf.zeros((self.bs,), dtype=self._dtype)
        self.current_level_vec: Tensor = tf.zeros((self.bs,), dtype=self._int_dtype)
        self.disambig_step_vec: Tensor = tf.zeros((self.bs,), dtype=self._int_dtype)
        self.refining_mask: Tensor = tf.zeros((self.bs,), dtype=tf.bool)

        self.weights0: Optional[Tensor] = None
        self.particles0: Optional[Tensor] = None
        self.weights1: Optional[Tensor] = None
        self.particles1: Optional[Tensor] = None
        self.mode_weights: Tensor = tf.ones((self.bs, 2), dtype=self._dtype) * 0.5

        self._last_true_g: Optional[Tensor] = None
        self._loss_dis_mask: Optional[Tensor] = None
        self._loss_refine_mask: Optional[Tensor] = None
        self._loss_true_left: Optional[Tensor] = None
        self._loss_q0_snapshot: Optional[Tensor] = None
        self._loss_mean_snapshot: Optional[Tensor] = None
        self._loss_width_snapshot: Optional[Tensor] = None
        self._loss_refine_mean_snapshot: Optional[Tensor] = None

    @property
    def current_level(self) -> float:
        return float(tf.reduce_mean(tf.cast(self.current_level_vec, self._dtype)).numpy())

    @property
    def q0_mean(self) -> float:
        return float(tf.reduce_mean(self.mode_weights[:, 0]).numpy())

    @property
    def interval_width(self) -> float:
        return float(tf.reduce_mean(self.g_hi_vec - self.g_lo_vec).numpy())

    @property
    def target_k_g(self) -> float:
        widths = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-10, self._dtype))
        kg = 2.0 * tf.cast(pi, self._dtype) / widths
        return float(tf.reduce_mean(kg).numpy())

    def _sample_particles_for_interval(self, lo: Tensor, hi: Tensor, rangen: tf.random.Generator) -> Tensor:
        u = rangen.uniform((self.bs, self.cfg.n_particles, 1), dtype=self._dtype)
        g = lo[:, None, None] + (hi - lo)[:, None, None] * u
        cols = [g]
        if self.phys_model.cfg.infer_mfg_bias:
            blo, bhi = self.phys_model.cfg.beta_B_range
            ub = rangen.uniform((self.bs, self.cfg.n_particles, 1), dtype=self._dtype)
            beta = tf.cast(blo, self._dtype) + tf.cast(bhi - blo, self._dtype) * ub
            cols.append(beta)
        if self.phys_model.cfg.infer_phi_off:
            plo, phi = self.phys_model.cfg.phi_off_range_rad
            up = rangen.uniform((self.bs, self.cfg.n_particles, 1), dtype=self._dtype)
            phi_off = tf.cast(plo, self._dtype) + tf.cast(phi - plo, self._dtype) * up
            cols.append(phi_off)
        return tf.concat(cols, axis=2)

    def _uniform_weights(self) -> Tensor:
        return tf.ones((self.bs, self.cfg.n_particles), dtype=self._dtype) / tf.cast(self.cfg.n_particles, self._dtype)

    def _clip_particles_to_bounds(self, particles: Tensor, g_lo: Tensor, g_hi: Tensor) -> Tensor:
        cols = []
        g = tf.clip_by_value(particles[:, :, 0], g_lo[:, None], g_hi[:, None])
        cols.append(g[:, :, None])
        idx = 1
        if self.phys_model.cfg.infer_mfg_bias:
            blo, bhi = self.phys_model.cfg.beta_B_range
            beta = tf.clip_by_value(particles[:, :, idx], tf.cast(blo, self._dtype), tf.cast(bhi, self._dtype))
            cols.append(beta[:, :, None])
            idx += 1
        if self.phys_model.cfg.infer_phi_off:
            phi = wrap_to_pi_tf(particles[:, :, idx])
            cols.append(phi[:, :, None])
        return tf.concat(cols, axis=2)

    def _init_pair_from_interval(self, lo: Tensor, hi: Tensor, rangen: tf.random.Generator) -> None:
        mid = 0.5 * (lo + hi)
        self.weights0 = self._uniform_weights()
        self.weights1 = self._uniform_weights()
        self.particles0 = self._sample_particles_for_interval(lo, mid, rangen)
        self.particles1 = self._sample_particles_for_interval(mid, hi, rangen)
        self.mode_weights = tf.ones((self.bs, 2), dtype=self._dtype) * 0.5

    def reset(self, rangen: tf.random.Generator) -> None:
        self.g_lo_vec = tf.fill((self.bs,), tf.cast(self._g_lo_init, self._dtype))
        self.g_hi_vec = tf.fill((self.bs,), tf.cast(self._g_hi_init, self._dtype))
        self.current_level_vec = tf.zeros((self.bs,), dtype=self._int_dtype)
        self.disambig_step_vec = tf.zeros((self.bs,), dtype=self._int_dtype)
        self.refining_mask = tf.zeros((self.bs,), dtype=tf.bool)
        self._init_pair_from_interval(self.g_lo_vec, self.g_hi_vec, rangen)
        self._clear_loss_snapshots()

    def _clear_loss_snapshots(self) -> None:
        self._loss_dis_mask = tf.zeros((self.bs,), dtype=tf.bool)
        self._loss_refine_mask = tf.zeros((self.bs,), dtype=tf.bool)
        self._loss_true_left = tf.zeros((self.bs,), dtype=self._dtype)
        self._loss_q0_snapshot = tf.zeros((self.bs,), dtype=self._dtype)
        self._loss_mean_snapshot = tf.zeros((self.bs,), dtype=self._dtype)
        self._loss_width_snapshot = tf.ones((self.bs,), dtype=self._dtype)
        self._loss_refine_mean_snapshot = tf.zeros((self.bs,), dtype=self._dtype)

    def _particle_likelihood(self, outcomes: Tensor, controls: Tensor, particles: Tensor, meas_step: Tensor) -> Tensor:
        N = self.cfg.n_particles
        outcomes_b = tf.broadcast_to(tf.expand_dims(outcomes, axis=1), (self.bs, N, self.phys_model.outcomes_size))
        controls_b = tf.broadcast_to(tf.expand_dims(controls, axis=1), (self.bs, N, self.phys_model.controls_size))
        step_b = tf.broadcast_to(tf.expand_dims(meas_step, axis=2), (self.bs, N, 1))
        state = tf.zeros((self.bs, N, 0), dtype=self._dtype)
        prob, _ = self.phys_model.wrapper_model(outcomes_b, controls_b, particles, state, step_b, num_systems=N)
        return prob

    def _masked_bayes_update(self, weights: Tensor, particles: Tensor, outcomes: Tensor, controls: Tensor, meas_step: Tensor, active_mask: Tensor):
        prob = self._particle_likelihood(outcomes, controls, particles, meas_step)
        unnorm = weights * prob
        Z = tf.reduce_sum(unnorm, axis=1, keepdims=True)
        safe_Z = tf.maximum(Z, tf.cast(1e-300, self._dtype))
        new_weights = unnorm / safe_Z
        weights_out = tf.where(active_mask[:, None], new_weights, weights)
        evidence = tf.where(active_mask, Z[:, 0], tf.ones((self.bs,), dtype=self._dtype))
        return weights_out, evidence

    def _differentiable_resample_masked(self, weights: Tensor, particles: Tensor, g_lo: Tensor, g_hi: Tensor, can_update: Tensor, rangen: tf.random.Generator):
        if not bool(tf.reduce_any(can_update).numpy()):
            return weights, particles
        count_mask = tf.expand_dims(can_update, axis=1)
        new_w, new_p = self._pf_template.partial_resampling(weights, particles, count_mask, rangen)
        new_p = self._clip_particles_to_bounds(new_p, g_lo, g_hi)
        weights_out = tf.where(can_update[:, None], new_w, weights)
        particles_out = tf.where(can_update[:, None, None], new_p, particles)
        return weights_out, particles_out

    def _resample_child_from_posterior(self, norm_w: Tensor, parent_particles: Tensor, child_lo: Tensor, child_hi: Tensor, active_mask: Tensor, fallback_mask: Tensor, rangen: tf.random.Generator):
        N = self.cfg.n_particles
        uniform_w = tf.ones_like(norm_w) / tf.cast(N, self._dtype)
        safe_w = tf.where(fallback_mask[:, None], uniform_w, norm_w)
        res_w, res_p = self._pf_template.resample(safe_w, parent_particles, rangen, batchsize=self.bs)
        res_p = self._clip_particles_to_bounds(res_p, child_lo, child_hi)
        uniform_p = self._sample_particles_for_interval(child_lo, child_hi, rangen)
        res_w = tf.where(fallback_mask[:, None], uniform_w, res_w)
        res_p = tf.where(fallback_mask[:, None, None], uniform_p, res_p)
        res_w = tf.where(active_mask[:, None], res_w, self._uniform_weights())
        res_p = tf.where(active_mask[:, None, None], res_p, parent_particles)
        return res_w, res_p

    def _spawn_child_from_parent(self, parent_weights: Tensor, parent_particles: Tensor, child_lo: Tensor, child_hi: Tensor, active_mask: Tensor, rangen: tf.random.Generator):
        in_interval = tf.logical_and(parent_particles[:, :, 0] >= child_lo[:, None], parent_particles[:, :, 0] <= child_hi[:, None])
        masked_weights = parent_weights * tf.cast(in_interval, self._dtype)
        child_mass = tf.reduce_sum(masked_weights, axis=1)
        safe_mass = tf.maximum(child_mass, tf.cast(1e-30, self._dtype))
        norm_w = masked_weights / safe_mass[:, None]
        fallback = tf.logical_and(active_mask, child_mass <= tf.cast(1e-12, self._dtype))
        child_weights, child_particles = self._resample_child_from_posterior(norm_w, parent_particles, child_lo, child_hi, active_mask, fallback, rangen)
        return child_weights, child_particles, child_mass

    def _weighted_mean_var(self, weights: Tensor, particles: Tensor):
        mean = self._pf_template.compute_mean(weights, particles)[:, 0]
        var = self._pf_template.compute_covariance(weights, particles)[:, 0, 0]
        return mean, tf.maximum(var, tf.cast(0.0, self._dtype))

    def marginal_mean_and_var(self):
        mean0, var0 = self._weighted_mean_var(self.weights0, self.particles0)
        mean1, var1 = self._weighted_mean_var(self.weights1, self.particles1)
        q0 = self.mode_weights[:, 0]
        q1 = self.mode_weights[:, 1]
        mean = tf.where(self.refining_mask, mean0, q0 * mean0 + q1 * mean1)
        second = tf.where(self.refining_mask, var0 + tf.square(mean0), q0 * (var0 + tf.square(mean0)) + q1 * (var1 + tf.square(mean1)))
        var = tf.maximum(second - tf.square(mean), tf.cast(0.0, self._dtype))
        return mean, var

    def map_mode_mean(self):
        mean0, _ = self._weighted_mean_var(self.weights0, self.particles0)
        mean1, _ = self._weighted_mean_var(self.weights1, self.particles1)
        choose0 = tf.logical_or(self.refining_mask, self.mode_weights[:, 0] >= self.mode_weights[:, 1])
        return tf.where(choose0, mean0, mean1)

    def _advance_ready_rows(self, ready: Tensor, rangen: tf.random.Generator) -> None:
        if not bool(tf.reduce_any(ready).numpy()):
            return
        old_lo = self.g_lo_vec
        old_hi = self.g_hi_vec
        old_mid = 0.5 * (old_lo + old_hi)
        choose_left = tf.logical_and(self.mode_weights[:, 0] >= self.mode_weights[:, 1], ready)
        choose_right = tf.logical_and(tf.logical_not(choose_left), ready)

        parent_weights = tf.where(choose_left[:, None], self.weights0, self.weights1)
        parent_particles = tf.where(choose_left[:, None, None], self.particles0, self.particles1)

        new_lo = tf.where(choose_left, old_lo, tf.where(choose_right, old_mid, old_lo))
        new_hi = tf.where(choose_left, old_mid, tf.where(choose_right, old_hi, old_hi))
        next_level = self.current_level_vec + tf.cast(ready, self._int_dtype)
        will_refine = tf.logical_and(ready, next_level >= self.cfg.n_levels)
        will_split_again = tf.logical_and(ready, tf.logical_not(will_refine))

        mid_new = 0.5 * (new_lo + new_hi)
        left_w, left_p, left_mass = self._spawn_child_from_parent(parent_weights, parent_particles, new_lo, mid_new, will_split_again, rangen)
        right_w, right_p, right_mass = self._spawn_child_from_parent(parent_weights, parent_particles, mid_new, new_hi, will_split_again, rangen)
        child_total = tf.maximum(left_mass + right_mass, tf.cast(1e-30, self._dtype))
        child_q0 = left_mass / child_total
        child_q1 = right_mass / child_total
        child_q0 = tf.where(will_split_again, child_q0, self.mode_weights[:, 0])
        child_q1 = tf.where(will_split_again, child_q1, self.mode_weights[:, 1])

        refine_q0 = tf.where(will_refine, tf.ones((self.bs,), dtype=self._dtype), child_q0)
        refine_q1 = tf.where(will_refine, tf.zeros((self.bs,), dtype=self._dtype), child_q1)
        new_weights0 = tf.where(will_refine[:, None], parent_weights, tf.where(will_split_again[:, None], left_w, self.weights0))
        new_particles0 = tf.where(will_refine[:, None, None], parent_particles, tf.where(will_split_again[:, None, None], left_p, self.particles0))
        placeholder_w1 = self._uniform_weights()
        new_weights1 = tf.where(will_refine[:, None], placeholder_w1, tf.where(will_split_again[:, None], right_w, self.weights1))
        # new_particles1 = tf.where(will_refine[:, None, None], self.particles1, tf.where(will_split_again[:, None, None], right_p, self.particles1))
        new_particles1 = tf.where(
            will_refine[:, None, None],
            self._sample_particles_for_interval(new_lo, new_hi, rangen),
            tf.where(will_split_again[:, None, None], right_p, self.particles1),
        )

        self.g_lo_vec = tf.where(ready, new_lo, self.g_lo_vec)
        self.g_hi_vec = tf.where(ready, new_hi, self.g_hi_vec)
        self.current_level_vec = tf.where(ready, next_level, self.current_level_vec)
        self.disambig_step_vec = tf.where(ready, tf.zeros_like(self.disambig_step_vec), self.disambig_step_vec)
        self.refining_mask = tf.where(ready, tf.logical_or(self.refining_mask, will_refine), self.refining_mask)
        self.weights0, self.particles0 = new_weights0, new_particles0
        self.weights1, self.particles1 = new_weights1, new_particles1
        self.mode_weights = tf.stack([refine_q0, refine_q1], axis=1)

    def set_true_values_for_loss(self, true_values: Tensor) -> None:
        self._last_true_g = tf.cast(true_values[:, 0:1, 0], self._dtype)

    def apply_measurement(self, outcomes: Tensor, controls: Tensor, meas_step: Tensor, continue_mask: Tensor, rangen: tf.random.Generator) -> None:
        continue_mask = tf.cast(continue_mask, tf.bool)
        self._clear_loss_snapshots()

        refine_rows = tf.logical_and(continue_mask, self.refining_mask)
        if bool(tf.reduce_any(refine_rows).numpy()):
            self.weights0, _ = self._masked_bayes_update(self.weights0, self.particles0, outcomes, controls, meas_step, refine_rows)
            self.weights0, self.particles0 = self._differentiable_resample_masked(self.weights0, self.particles0, self.g_lo_vec, self.g_hi_vec, refine_rows, rangen)
            mean0, _ = self._weighted_mean_var(self.weights0, self.particles0)
            self.mode_weights = tf.where(refine_rows[:, None], tf.concat([tf.ones((self.bs, 1), dtype=self._dtype), tf.zeros((self.bs, 1), dtype=self._dtype)], axis=1), self.mode_weights)
            self._loss_refine_mask = refine_rows
            self._loss_refine_mean_snapshot = mean0

        dis_rows = tf.logical_and(continue_mask, tf.logical_not(self.refining_mask))
        if bool(tf.reduce_any(dis_rows).numpy()):
            if self._last_true_g is None:
                raise RuntimeError("set_true_values_for_loss must be called before apply_measurement")
            current_mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)
            true_left = tf.cast(self._last_true_g[:, 0] <= current_mid, self._dtype)

            new_w0, Z0 = self._masked_bayes_update(self.weights0, self.particles0, outcomes, controls, meas_step, dis_rows)
            new_w1, Z1 = self._masked_bayes_update(self.weights1, self.particles1, outcomes, controls, meas_step, dis_rows)
            q0 = self.mode_weights[:, 0]
            q1 = self.mode_weights[:, 1]
            q0_u = q0 * Z0
            q1_u = q1 * Z1
            Zt = tf.maximum(q0_u + q1_u, tf.cast(1e-300, self._dtype))
            new_q0 = q0_u / Zt
            new_q1 = q1_u / Zt
            self.weights0, self.weights1 = new_w0, new_w1
            self.mode_weights = tf.where(dis_rows[:, None], tf.stack([new_q0, new_q1], axis=1), self.mode_weights)

            left_lo = self.g_lo_vec
            left_hi = 0.5 * (self.g_lo_vec + self.g_hi_vec)
            right_lo = left_hi
            right_hi = self.g_hi_vec
            self.weights0, self.particles0 = self._differentiable_resample_masked(self.weights0, self.particles0, left_lo, left_hi, dis_rows, rangen)
            self.weights1, self.particles1 = self._differentiable_resample_masked(self.weights1, self.particles1, right_lo, right_hi, dis_rows, rangen)

            mean0, _ = self._weighted_mean_var(self.weights0, self.particles0)
            mean1, _ = self._weighted_mean_var(self.weights1, self.particles1)
            mix_mean = new_q0 * mean0 + new_q1 * mean1
            width = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, self._dtype))

            self._loss_dis_mask = dis_rows
            self._loss_true_left = true_left
            self._loss_q0_snapshot = tf.where(dis_rows, new_q0, self._loss_q0_snapshot)
            self._loss_mean_snapshot = tf.where(dis_rows, mix_mean, self._loss_mean_snapshot)
            self._loss_width_snapshot = tf.where(dis_rows, width, self._loss_width_snapshot)

            self.disambig_step_vec = tf.where(dis_rows, self.disambig_step_vec + 1, self.disambig_step_vec)
            ready = tf.logical_and(dis_rows, self.disambig_step_vec >= self.cfg.n_disambig_per_level)
            self._advance_ready_rows(ready, rangen)

    # def hierarchical_loss(self, true_values: Tensor) -> Tensor:
    #     g_true = tf.cast(true_values[:, 0, 0], self._dtype)
    #     loss = tf.zeros((self.bs,), dtype=self._dtype)

    #     refine_loss = tf.square(self._loss_refine_mean_snapshot - g_true)
    #     loss = tf.where(self._loss_refine_mask, refine_loss, loss)

    #     q0 = tf.clip_by_value(self._loss_q0_snapshot, tf.cast(1e-6, self._dtype), tf.cast(1.0 - 1e-6, self._dtype))
    #     q1 = tf.cast(1.0, self._dtype) - q0
    #     true_left = self._loss_true_left
    #     bce = -(true_left * tf.math.log(q0) + (tf.cast(1.0, self._dtype) - true_left) * tf.math.log(tf.maximum(q1, tf.cast(1e-6, self._dtype))))
    #     local_mse = tf.square((self._loss_mean_snapshot - g_true) / tf.maximum(self._loss_width_snapshot, tf.cast(1e-8, self._dtype)))
    #     dis_loss = tf.cast(self.cfg.hierarchy_bce_weight, self._dtype) * bce + tf.cast(self.cfg.hierarchy_local_mse_weight, self._dtype) * local_mse
    #     loss = tf.where(self._loss_dis_mask, dis_loss, loss)
    #     return tf.expand_dims(loss, axis=1)

    def hierarchical_loss_components(self, true_values: Tensor):
        g_true = tf.cast(true_values[:, 0, 0], self._dtype)

        # --- refinement snapshot ---
        refine_mse = tf.square(self._loss_refine_mean_snapshot - g_true)

        # --- disambiguation snapshot ---
        q0 = tf.clip_by_value(
            self._loss_q0_snapshot,
            tf.cast(1e-6, self._dtype),
            tf.cast(1.0 - 1e-6, self._dtype),
        )
        q1 = tf.cast(1.0, self._dtype) - q0
        true_left = self._loss_true_left

        bce = -(
            true_left * tf.math.log(q0)
            + (tf.cast(1.0, self._dtype) - true_left)
            * tf.math.log(tf.maximum(q1, tf.cast(1e-6, self._dtype)))
        )

        # Restore the normalized mixture-mean MSE scale.
        width = tf.maximum(self._loss_width_snapshot, tf.cast(1e-8, self._dtype))
        mix_mse = tf.square((self._loss_mean_snapshot - g_true) / width)

        dis_total = (
            tf.cast(self.cfg.hierarchy_bce_weight, self._dtype) * bce
            + tf.cast(self.cfg.hierarchy_local_mse_weight, self._dtype) * mix_mse
        )

        total = tf.zeros((self.bs,), dtype=self._dtype)
        total = tf.where(self._loss_refine_mask, refine_mse, total)
        total = tf.where(self._loss_dis_mask, dis_total, total)

        return {
            "total": tf.expand_dims(total, axis=1),
            "bce": tf.expand_dims(
                tf.where(self._loss_dis_mask, bce, tf.zeros_like(bce)),
                axis=1,
            ),
            "mix_mse": tf.expand_dims(
                tf.where(self._loss_dis_mask, mix_mse, tf.zeros_like(mix_mse)),
                axis=1,
            ),
            "refine_mse": tf.expand_dims(
                tf.where(self._loss_refine_mask, refine_mse, tf.zeros_like(refine_mse)),
                axis=1,
            ),
            "true_left": tf.expand_dims(true_left, axis=1),
        }

    def hierarchical_loss(self, true_values: Tensor) -> Tensor:
        return self.hierarchical_loss_components(true_values)["total"]
    
    def debug_snapshot(
        self,
        phys_model,
        true_values: Tensor,
        controls: Tensor,
        used_resources: Tensor,
        max_examples: int = 3,
    ) -> list[dict]:
        """
        Build a per-batch debug snapshot for a few batch elements.
        """
        g_true = true_values[:, 0, 0]
        g_mean, g_var = self.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-20, self._dtype)))

        q0 = self.mode_weights[:, 0]
        q1 = self.mode_weights[:, 1]

        mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)
        true_left = tf.cast(g_true <= mid, tf.int32)
        pred_left = tf.cast(q0 >= q1, tf.int32)

        T_s = controls[:, 0]
        Bp = controls[:, 1]
        phi = controls[:, 2]

        target_kg = 2.0 * tf.cast(pi, self._dtype) / tf.maximum(
            self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, self._dtype)
        )
        actual_kg = phys_model.k_g(T_s, Bp)
        kg_ratio = actual_kg / tf.maximum(target_kg, tf.cast(1e-8, self._dtype))

        out = []
        n = min(max_examples, self.bs)
        for b in range(n):
            out.append({
                "batch_idx": int(b),
                "level": int(self.current_level_vec[b].numpy()),
                "step_in_level": int(self.disambig_step_vec[b].numpy()),
                "refining": bool(self.refining_mask[b].numpy()),
                "g_lo": float(self.g_lo_vec[b].numpy()),
                "g_hi": float(self.g_hi_vec[b].numpy()),
                "interval_width": float((self.g_hi_vec[b] - self.g_lo_vec[b]).numpy()),
                "g_true": float(g_true[b].numpy()),
                "g_mean": float(g_mean[b].numpy()),
                "g_std": float(g_std[b].numpy()),
                "q0": float(q0[b].numpy()),
                "q1": float(q1[b].numpy()),
                "true_left": int(true_left[b].numpy()),
                "pred_left": int(pred_left[b].numpy()),
                "branch_correct": bool((true_left[b] == pred_left[b]).numpy()),
                "T_s": float(T_s[b].numpy()),
                "Bp_kTm": float(Bp[b].numpy()),
                "mw_phase_rad": float(phi[b].numpy()),
                "target_kg": float(target_kg[b].numpy()),
                "actual_kg": float(actual_kg[b].numpy()),
                "kg_ratio": float(kg_ratio[b].numpy()),
                "used_resources": float(used_resources[b, 0].numpy()),
            })
        return out


class GravityHierarchicalSimulation(StatelessSimulation):
    def __init__(self, phys_model: GravityStatelessPhysicalModel, bank: HierarchicalPFBank, controller, simpars: SimulationParameters, bank_cfg: HierarchicalPFConfig) -> None:
        input_size = 7
        input_name = ["mu_g_norm", "log_sigma_norm", "q0", "level_norm", "step_norm", "res_norm", "target_kg_norm"]
        super().__init__(particle_filter=bank.pf, phys_model=phys_model, control_strategy=controller, input_size=input_size, input_name=input_name, simpars=simpars)
        self.bank = bank
        self.bank_cfg = bank_cfg
        self._dtype = tf.float32 if simpars.prec == "float32" else tf.float64

    def generate_input(self, weights: Tensor, particles: Tensor, meas_step: Tensor, used_resources: Tensor, rangen) -> Tensor:
        del weights, particles, meas_step, rangen
        bank = self.bank
        g_mean, g_var = bank.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-20, self._dtype)))
        width = tf.maximum(bank.g_hi_vec - bank.g_lo_vec, tf.cast(1e-8, self._dtype))
        mu_norm = 2.0 * (g_mean - bank.g_lo_vec) / width - 1.0
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)
        log_sigma = -2.0 / 10.0 * tf.math.log(tf.maximum(g_std, tf.cast(1e-8, self._dtype))) / tf.math.log(tf.cast(10.0, self._dtype)) - 1.0
        log_sigma = tf.clip_by_value(log_sigma, -1.0, 1.0)
        q0 = tf.where(bank.refining_mask, tf.ones((self.bs,), dtype=self._dtype), bank.mode_weights[:, 0])
        level_norm = tf.cast(bank.current_level_vec, self._dtype) / tf.cast(max(bank.cfg.n_levels, 1), self._dtype)
        step_norm = 2.0 * tf.cast(bank.disambig_step_vec, self._dtype) / tf.cast(max(bank.cfg.n_disambig_per_level, 1), self._dtype) - 1.0
        step_norm = tf.where(bank.refining_mask, tf.ones_like(step_norm), step_norm)
        res_norm = 2.0 * used_resources[:, 0] / tf.cast(self.simpars.max_resources, self._dtype) - 1.0

        kg_target = 2.0 * tf.cast(pi, self._dtype) / width
        kg_min = tf.cast(self.phys_model.min_gain(self._dtype), self._dtype)
        kg_max = tf.cast(self.phys_model.max_gain(self._dtype), self._dtype)
        log_kg = tf.math.log(tf.maximum(kg_target, tf.cast(1e-8, self._dtype)))
        log_kg_min = tf.math.log(tf.maximum(kg_min, tf.cast(1e-8, self._dtype)))
        log_kg_max = tf.math.log(tf.maximum(kg_max, tf.cast(1e-8, self._dtype)))
        target_kg_norm = 2.0 * (log_kg - log_kg_min) / tf.maximum(log_kg_max - log_kg_min, tf.cast(1e-8, self._dtype)) - 1.0
        target_kg_norm = tf.clip_by_value(target_kg_norm, -1.0, 1.0)

        return tf.stack([mu_norm, log_sigma, q0, level_norm, step_norm, res_norm, target_kg_norm], axis=1)

    def loss_function(self, weights: Tensor, particles: Tensor, true_values: Tensor, used_resources: Tensor, meas_step: Tensor) -> Tensor:
        del weights, particles, used_resources, meas_step
        return self.bank.hierarchical_loss(true_values)

    def execute(self, rangen: tf.random.Generator, deploy: bool = False, debug: bool = False):
        pars = self.simpars
        prec = self._dtype
        bank = self.bank

        debug_records = []

        bank.reset(rangen)
        true_values = self.phys_model.true_values(rangen)
        bank.set_true_values_for_loss(true_values)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype=tf.bool)
        outcomes = tf.zeros((self.bs, self.phys_model.outcomes_size), dtype=prec)
        meas_step = tf.zeros((self.bs, 1), dtype=tf.int32)
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        loss_diff_accum = tf.zeros((), dtype=prec)
        loss_accum = tf.zeros((), dtype=prec)
        step_count = 0

        if deploy:
            hist_inputs: List[Tensor] = []
            hist_controls: List[Tensor] = []
            hist_resources: List[Tensor] = []
            hist_precisions: List[Tensor] = []

        for _ in range(pars.num_steps):
            num_finished = int(tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy())
            if num_finished >= pars.resources_fraction * self.bs:
                break

            weights = bank.weights0
            particles = bank.particles0
            self.pf = bank.pf

            input_strategy = self.generate_input(weights, particles, tf.cast(meas_step, prec), used_resources, rangen)
            cond_input = tf.stop_gradient(input_strategy) if pars.stop_gradient_input else input_strategy
            controls = self.control_strategy(cond_input)

            new_used_resources = self.phys_model.wrapper_count_resources(used_resources, outcomes, controls, true_values, true_state, meas_step)
            continue_flag = tf.math.less_equal(new_used_resources, tf.cast(pars.max_resources, prec) * tf.ones((self.bs, 1), dtype=prec))
            used_resources = tf.where(continue_flag, new_used_resources, used_resources)

            outcomes_raw, log_prob, post_true_state = self.phys_model.wrapper_perform_measurement(
                tf.expand_dims(controls, axis=1), true_values, true_state, tf.expand_dims(meas_step, axis=1), rangen
            )
            outcomes = outcomes_raw[:, 0, :]

            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(continue_flag, sum_log_prob + log_prob, sum_log_prob)

            bank.apply_measurement(outcomes, controls, meas_step, continue_flag[:, 0], rangen)
            # loss_vals = self.loss_function(weights, particles, true_values, used_resources, meas_step)

            loss_parts = self.bank.hierarchical_loss_components(true_values)
            loss_vals = loss_parts["total"]

            if debug:
                snapshots = self.bank.debug_snapshot(
                    phys_model=self.phys_model,
                    true_values=true_values,
                    controls=controls,
                    used_resources=used_resources,
                    max_examples=min(3, self.bs),
                )
                for s in snapshots:
                    b = s["batch_idx"]
                    s["loss_total"] = float(loss_parts["total"][b, 0].numpy())
                    s["loss_bce"] = float(loss_parts["bce"][b, 0].numpy())
                    s["loss_mix_mse"] = float(loss_parts["mix_mse"][b, 0].numpy())
                    s["loss_refine_mse"] = float(loss_parts["refine_mse"][b, 0].numpy())
                    s["global_step"] = int(meas_step[b, 0].numpy())
                    debug_records.append(s)

            true_state = post_true_state
            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            if pars.cumulative_loss and not deploy:
                active = tf.cast(continue_flag, prec)
                n_active = tf.maximum(tf.reduce_sum(active), tf.cast(1.0, prec))
                if pars.loss_logl_outcomes:
                    baseline = tf.reduce_sum(tf.where(continue_flag, loss_vals, tf.zeros_like(loss_vals))) / n_active if pars.baseline else tf.cast(0.0, prec)
                    diff_vals = loss_vals + (tf.stop_gradient(loss_vals) - tf.stop_gradient(baseline)) * sum_log_prob
                else:
                    diff_vals = loss_vals
                loss_diff_accum = loss_diff_accum + tf.reduce_sum(tf.where(continue_flag, diff_vals, tf.zeros_like(diff_vals))) / n_active
                loss_accum = loss_accum + tf.reduce_sum(tf.where(continue_flag, loss_vals, tf.zeros_like(loss_vals))) / n_active

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls.append(controls)
                hist_resources.append(used_resources)
                hist_precisions.append(loss_vals)

        if not deploy:
            if pars.cumulative_loss:
                denom = tf.cast(max(step_count, 1), prec)
                return loss_diff_accum / denom, loss_accum / denom
            else:
                loss_vals = self.loss_function(weights, particles, true_values, used_resources, meas_step)
                loss_mean = tf.reduce_mean(loss_vals)
                if pars.loss_logl_outcomes:
                    baseline = loss_mean if pars.baseline else tf.cast(0.0, prec)
                    loss_diff = tf.reduce_mean(loss_vals + (tf.stop_gradient(loss_vals) - tf.stop_gradient(baseline)) * sum_log_prob)
                else:
                    loss_diff = loss_mean
                return loss_diff, loss_mean

        ns = len(hist_inputs)
        if ns == 0:
            empty_i = tf.zeros((1, self.bs, self.input_size), dtype=prec)
            empty_c = tf.zeros((1, self.bs, self.phys_model.controls_size), dtype=prec)
            empty_r = tf.zeros((1, self.bs, 1), dtype=prec)
            empty_p = tf.zeros((1, self.bs, 1), dtype=prec)
        else:
            empty_i = tf.stack(hist_inputs, axis=0)
            empty_c = tf.stack(hist_controls, axis=0)
            empty_r = tf.stack(hist_resources, axis=0)
            empty_p = tf.stack(hist_precisions, axis=0)

        return (
            true_values,
            tf.reshape(empty_i, (self.bs * ns, self.input_size)),
            tf.reshape(empty_c, (self.bs * ns, self.phys_model.controls_size)),
            tf.reshape(empty_r, (self.bs * ns, 1)),
            tf.reshape(empty_p, (self.bs * ns, 1)),
            debug_records,
        )


def build_controller(phys_model: GravityStatelessPhysicalModel, input_size: int = 7, hidden_sizes: Tuple[int, ...] = (128, 128, 64)) -> tf.keras.Model:
    cfg = phys_model.cfg
    dtype = tf.float32 if cfg.prec == "float32" else tf.float64
    T_min, T_max = map(float, cfg.T_range_s)
    Bp_min, Bp_max = map(float, cfg.Bp_range_kTm)
    phi_min, phi_max = 0.0, 2.0 * pi

    T_mid, T_half = 0.5 * (T_max + T_min), 0.5 * (T_max - T_min)
    Bp_mid, Bp_half = 0.5 * (Bp_max + Bp_min), 0.5 * (Bp_max - Bp_min)
    phi_mid, phi_half = 0.5 * (phi_max + phi_min), 0.5 * (phi_max - phi_min)

    class ControlScalingLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._T_mid = tf.constant(T_mid, dtype=dtype)
            self._T_half = tf.constant(T_half, dtype=dtype)
            self._Bp_mid = tf.constant(Bp_mid, dtype=dtype)
            self._Bp_half = tf.constant(Bp_half, dtype=dtype)
            self._phi_mid = tf.constant(phi_mid, dtype=dtype)
            self._phi_half = tf.constant(phi_half, dtype=dtype)

        def call(self, x):
            T_s = self._T_mid + self._T_half * x[:, 0:1]
            Bp = self._Bp_mid + self._Bp_half * x[:, 1:2]
            phi = self._phi_mid + self._phi_half * x[:, 2:3]
            return tf.concat([T_s, Bp, phi], axis=1)

    inputs = tf.keras.Input(shape=(input_size,), dtype=dtype)
    x = inputs
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="tanh", dtype=dtype)(x)
    x = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    outputs = ControlScalingLayer(dtype=dtype)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="hierarchical_gravity_controller_complete")


def build_hierarchical_simulation(batchsize: int, cfg: GravimeterConfig, bank_cfg: HierarchicalPFConfig, simpars: SimulationParameters, rangen: tf.random.Generator):
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = HierarchicalPFBank(phys_model=phys_model, cfg=bank_cfg)
    controller = build_controller(phys_model, input_size=7)
    dummy = tf.zeros((batchsize, 7), dtype=tf.float32 if cfg.prec == "float32" else tf.float64)
    _ = controller(dummy)
    bank.reset(rangen)
    simulation = GravityHierarchicalSimulation(phys_model=phys_model, bank=bank, controller=controller, simpars=simpars, bank_cfg=bank_cfg)
    return simulation, bank, controller
