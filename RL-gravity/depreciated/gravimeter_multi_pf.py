# gravimeter_multi_pf.py

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model_complete import GravimeterConfig, GravityStatelessPhysicalModel


# ---------------------------------------------------------------------------
# qsensoropt loader
# ---------------------------------------------------------------------------
USE_FIXED_CONTROLLER = False
FIXED_ORACLE_PHASE = False

FIXED_U_KG = -0.5
FIXED_U_B = 0.0
FIXED_U_PHI = 0.5
FIXED_PHASE_FLIP = False

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
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.stateless_simulation import StatelessSimulation
    from qsensoropt.simulation_parameters import SimulationParameters
except Exception:
    ParticleFilter = _load_local_qsensoropt_module("particle_filter").ParticleFilter
    StatelessSimulation = _load_local_qsensoropt_module(
        "stateless_simulation"
    ).StatelessSimulation
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters

def wrap_to_pi_local(x: Tensor) -> Tensor:
    two_pi = tf.cast(2.0 * pi, x.dtype)
    return tf.math.floormod(x + tf.cast(pi, x.dtype), two_pi) - tf.cast(pi, x.dtype)
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiPFBankConfig:
    n_per_mode: int = 64
    k_max: int = 128
    n_scales: Optional[int] = None
    top_k_modes: int = 8

    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = False
    trim: bool = True

    smoothness_lambda: float = 0.0

    schedule_floor: float = 0.25
    k_unlock_steps: int = 8
    mode_penalty_coef: float = 0.0

    # NEW: full posterior-state and safe-gain schedule parameters.
    use_full_q_hist: bool = True

    # Width multiplier for posterior-safe gain:
    # k_g <= 2*pi / (posterior_gain_width_multiplier * sigma_g).
    # 4.0 means approximately one fringe across a ±2σ credible region.
    posterior_gain_width_multiplier: float = 4.0

    # Lower bound of allowed gain as a fraction of one fringe across prior.
    # Smaller than 1 allows coarse, sub-fringe global-disambiguation shots.
    min_gain_fringe_fraction: float = 0.25

    # Normalization range for log10(sigma_g / prior_width).
    log_sigma_frac_bounds: Tuple[float, float] = (-6.0, 0.0)


# ---------------------------------------------------------------------------
# Multi-PF Bank
# ---------------------------------------------------------------------------

class MultiPFBank:
    """Fixed-K bank of K=k_max independent qsensoropt particle filters."""

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        cfg: MultiPFBankConfig,
    ) -> None:
        self.phys_model = phys_model
        self.cfg = cfg
        self.bs: int = phys_model.bs
        self.prec: str = phys_model.prec

        g_lo, g_hi = phys_model.cfg.g_range
        self.g_lo: float = float(g_lo)
        self.g_hi: float = float(g_hi)
        self.K: int = int(cfg.k_max)

        if cfg.top_k_modes > self.K:
            raise ValueError(
                f"top_k_modes={cfg.top_k_modes} > k_max={self.K}"
            )

        self.weights_list: List[Tensor] = []
        self.particles_list: List[Tensor] = []
        self.state_list: List[Tensor] = []
        self.mode_weights: Optional[Tensor] = None
        self.pf_list: List[ParticleFilter] = []
        

        for _ in range(self.K):
            self.pf_list.append(
                ParticleFilter(
                    num_particles=cfg.n_per_mode,
                    phys_model=self.phys_model,
                    resampling_allowed=True,
                    resample_threshold=cfg.resample_threshold,
                    alpha=cfg.resample_alpha,
                    beta=cfg.resample_beta,
                    scibior_trick=cfg.scibior_trick,
                    trim=cfg.trim,
                    prec=self.prec,
                )
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def detach_state(self) -> None:
        """Stop gradients through the current bank state."""
        self.mode_weights = tf.stop_gradient(self.mode_weights)
        self.weights_list = [tf.stop_gradient(w) for w in self.weights_list]
        self.particles_list = [tf.stop_gradient(p) for p in self.particles_list]


    def resample_all(
        self,
        continue_flag: Tensor,
        rangen: tf.random.Generator,
        *,
        detach: bool = True,
    ) -> None:
        """Resample all modes after loss computation.

        If detach=True, resampling is treated as representation maintenance,
        not as part of the differentiable loss path.
        """
        for k in range(self.K):
            w_in = tf.stop_gradient(self.weights_list[k]) if detach else self.weights_list[k]
            p_in = tf.stop_gradient(self.particles_list[k]) if detach else self.particles_list[k]

            new_w_k, new_p_k, _ = self.pf_list[k].full_resampling(
                w_in,
                p_in,
                count_for_resampling=continue_flag,
                rangen=rangen,
            )

            self.weights_list[k] = tf.stop_gradient(new_w_k) if detach else new_w_k
            self.particles_list[k] = tf.stop_gradient(new_p_k) if detach else new_p_k
    def reset(self, rangen: tf.random.Generator) -> None:
        cfg = self.cfg
        prec = self.prec
        K = self.K
        N = cfg.n_per_mode
        g_lo, g_hi = self.g_lo, self.g_hi
        mode_width = (g_hi - g_lo) / float(K)

        fresh_w_list: List[Tensor] = []
        fresh_p_list: List[Tensor] = []
        fresh_s_list: List[Tensor] = []

        scale_py = mode_width / max(g_hi - g_lo, 1e-30)
        for k in range(K):
            w_k_init, p_k_init = self.pf_list[k].reset(rangen)
            g_old = p_k_init[:, :, 0]
            scale = tf.cast(scale_py, prec)
            offset = tf.cast(g_lo + k * mode_width - g_lo * scale_py, prec)
            g_new = scale * g_old + offset
            d = p_k_init.shape[2]
            if d == 1:
                p_k = tf.expand_dims(g_new, axis=2)
            else:
                other = p_k_init[:, :, 1:]
                p_k = tf.concat([tf.expand_dims(g_new, axis=2), other], axis=2)
            fresh_w_list.append(w_k_init)
            fresh_p_list.append(p_k)
            fresh_s_list.append(tf.zeros((self.bs, N, 0), dtype=prec))

        self.weights_list = fresh_w_list
        self.particles_list = fresh_p_list
        self.state_list = fresh_s_list
        self.mode_weights = tf.fill((self.bs, K), tf.cast(1.0 / float(K), prec))

    # ------------------------------------------------------------------
    # Per-step Bayes update
    # ------------------------------------------------------------------

    def apply_measurement(
        self,
        outcomes: Tensor,
        controls: Tensor,
        meas_step: Tensor,
        continue_flag: Tensor,
        rangen: tf.random.Generator,
        *,
        do_resample: bool = False,
    ) -> None:
        prec = self.prec
        K = self.K
        keep = tf.cast(continue_flag[:, 0], prec)
        keep_w = tf.expand_dims(keep, axis=1)

        Z_k_list: List[Tensor] = []
        new_weights_list: List[Tensor] = []

        for k in range(K):
            w_k = self.weights_list[k]
            p_k = self.particles_list[k]
            N_k = self.pf_list[k].np
            outcomes_b = tf.broadcast_to(
                tf.expand_dims(outcomes, axis=1),
                (self.bs, N_k, self.phys_model.outcomes_size),
            )
            controls_b = tf.broadcast_to(
                tf.expand_dims(controls, axis=1),
                (self.bs, N_k, self.phys_model.controls_size),
            )
            step_b = tf.broadcast_to(
                tf.expand_dims(meas_step, axis=2),
                (self.bs, N_k, 1),
            )
            state_k = self.state_list[k]
            prob_k, _ = self.phys_model.wrapper_model(
                outcomes_b, controls_b, p_k, state_k, step_b,
                num_systems=N_k,
            )
            unnorm_w_k = w_k * prob_k
            Z_k = tf.reduce_sum(unnorm_w_k, axis=1)
            safe_Z_k = tf.maximum(Z_k, tf.cast(1e-30, prec))
            updated_w_k = unnorm_w_k / tf.expand_dims(safe_Z_k, axis=1)
            new_w_k = tf.where(keep_w > 0.5, updated_w_k, w_k)
            Z_k_list.append(Z_k)
            new_weights_list.append(new_w_k)

        Z_stack = tf.stack(Z_k_list, axis=1)
        q_old = self.mode_weights
        new_q_unnorm = q_old * Z_stack
        Z_total = tf.reduce_sum(new_q_unnorm, axis=1, keepdims=True)
        safe_Z_total = tf.maximum(Z_total, tf.cast(1e-30, prec))
        new_q = new_q_unnorm / safe_Z_total
        self.mode_weights = tf.where(keep_w > 0.5, new_q, q_old)
        self.weights_list = new_weights_list

        # cont_for_resamp = continue_flag
        # for k in range(K):
        #     new_w_k, new_p_k, _ = self.pf_list[k].full_resampling(
        #         self.weights_list[k],
        #         self.particles_list[k],
        #         count_for_resampling=cont_for_resamp,
        #         rangen=rangen,
        #     )
        #     self.weights_list[k] = new_w_k
        #     self.particles_list[k] = new_p_k
        if do_resample:
            self.resample_all(
                continue_flag=continue_flag,
                rangen=rangen,
                detach=False,
            )

    def diagnostic_mode_evidence(
        self,
        outcomes: Tensor,
        controls: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """Return Z_k = sum_j w_kj p(y | theta_kj, controls), shape (B, K).

        Diagnostic only. Does not update the bank.
        """
        Z_k_list: List[Tensor] = []

        for k in range(self.K):
            w_k = self.weights_list[k]          # (B, N)
            p_k = self.particles_list[k]        # (B, N, d)
            N_k = self.pf_list[k].np

            outcomes_b = tf.broadcast_to(
                tf.expand_dims(outcomes, axis=1),
                (self.bs, N_k, self.phys_model.outcomes_size),
            )
            controls_b = tf.broadcast_to(
                tf.expand_dims(controls, axis=1),
                (self.bs, N_k, self.phys_model.controls_size),
            )
            step_b = tf.broadcast_to(
                tf.expand_dims(meas_step, axis=2),
                (self.bs, N_k, 1),
            )
            state_k = self.state_list[k]

            prob_k, _ = self.phys_model.wrapper_model(
                outcomes_b,
                controls_b,
                p_k,
                state_k,
                step_b,
                num_systems=N_k,
            )

            Z_k = tf.reduce_sum(w_k * prob_k, axis=1)  # (B,)
            Z_k_list.append(Z_k)

        return tf.stack(Z_k_list, axis=1)  # (B, K)
    # ------------------------------------------------------------------
    # Diagnostics & estimators
    # ------------------------------------------------------------------

    def mode_means_and_stds(self) -> Tuple[Tensor, Tensor]:
        means: List[Tensor] = []
        stds: List[Tensor] = []
        prec = self.prec
        for k in range(self.K):
            mean_k = self.pf_list[k].compute_mean(
                self.weights_list[k], self.particles_list[k]
            )[:, 0]
            cov_k = self.pf_list[k].compute_covariance(
                self.weights_list[k], self.particles_list[k]
            )[:, 0, 0]
            std_k = tf.sqrt(tf.maximum(cov_k, tf.cast(0.0, prec)))
            means.append(mean_k)
            stds.append(std_k)
        return tf.stack(means, axis=1), tf.stack(stds, axis=1)

    def marginal_mean_and_var(self) -> Tuple[Tensor, Tensor]:
        prec = self.prec
        means, stds = self.mode_means_and_stds()
        q = self.mode_weights
        g_mean = tf.reduce_sum(q * means, axis=1)
        sec = tf.reduce_sum(q * (tf.square(stds) + tf.square(means)), axis=1)
        g_var = tf.maximum(sec - tf.square(g_mean), tf.cast(0.0, prec))
        return g_mean, g_var

    def map_mode_estimate(self) -> Tuple[Tensor, Tensor]:
        """(mean, std) of the mode with the largest q.  This is the point
        estimator the loss tries to make accurate."""
        means, stds = self.mode_means_and_stds()
        best_k = tf.argmax(self.mode_weights, axis=1, output_type=tf.int32)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)
        return tf.gather_nd(means, gather_idx), tf.gather_nd(stds, gather_idx)

    def closest_mode_estimate(self, g_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Oracle estimator: of the K modes, the one whose mean is closest to
        the true value `g_true`.  Used **only** at training time to give the
        controller a training signal that doesn't depend on which mode has the
        highest weight (mode-selection is a discrete decision the gradient
        can't smooth).

        Parameters
        ----------
        g_true: Tensor of shape (B,) — the per-batch true g.

        Returns
        -------
        g_best:    (B,)  the closest-mode mean
        std_best:  (B,)  its std
        q_best:    (B,)  its q (so we can monitor whether MAP == closest)
        """
        means, stds = self.mode_means_and_stds()           # (B, K)
        # Distance of each mode mean to the true value.
        d2 = tf.square(means - tf.expand_dims(g_true, axis=1))   # (B, K)
        best_k = tf.argmin(d2, axis=1, output_type=tf.int32)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)
        g_best = tf.gather_nd(means, gather_idx)
        std_best = tf.gather_nd(stds, gather_idx)
        q_best = tf.gather_nd(self.mode_weights, gather_idx)
        return g_best, std_best, q_best

    # ------------------------------------------------------------------
    # Holevo variance and log-Holevo loss
    # ------------------------------------------------------------------

    def _complex_moment(self, k_ref: Tensor) -> Tuple[Tensor, Tensor]:
        """Mixture circular moment mu(k_ref) = sum_k q_k * sum_j w_kj *
        exp(i k_ref g_kj).  Returns (Re, Im) of shape (B,)."""
        prec = self.prec
        if k_ref.shape.rank == 0:
            k_ref_b = tf.fill((self.bs,), tf.cast(k_ref, prec))
        else:
            k_ref_b = tf.cast(k_ref, prec)

        re_acc = tf.zeros((self.bs,), dtype=prec)
        im_acc = tf.zeros((self.bs,), dtype=prec)
        for k in range(self.K):
            p_k = self.particles_list[k]
            w_k = self.weights_list[k]
            g_k = p_k[:, :, 0]
            phase = tf.expand_dims(k_ref_b, axis=1) * g_k
            re_k = tf.reduce_sum(w_k * tf.cos(phase), axis=1)
            im_k = tf.reduce_sum(w_k * tf.sin(phase), axis=1)
            q_k = self.mode_weights[:, k]
            re_acc = re_acc + q_k * re_k
            im_acc = im_acc + q_k * im_k
        return re_acc, im_acc

    def holevo_variance(self, k_ref: Tensor) -> Tensor:
        """V_H = 1/|mu|^2 - 1 (diagnostic; can be huge)."""
        prec = self.prec
        re, im = self._complex_moment(k_ref)
        abs_mu_sq = tf.square(re) + tf.square(im)
        v_h = 1.0 / tf.maximum(abs_mu_sq, tf.cast(1e-30, prec)) - 1.0
        return tf.expand_dims(v_h, axis=1)

    def log_holevo_at_scale(self, k_ref: Tensor) -> Tensor:
        """log(1 + V_H(k_ref)) at the given scale.  Bounded above by
        log(1/eps) ~ 69 (float64) when |mu|->0, bounded below by 0.
        Returns (B,)."""
        prec = self.prec
        re, im = self._complex_moment(k_ref)
        abs_mu_sq = tf.square(re) + tf.square(im)
        v_h = 1.0 / tf.maximum(abs_mu_sq, tf.cast(1e-30, prec)) - 1.0
        return tf.math.log1p(v_h)

    def multi_scale_log_holevo_loss(self, scales: Tensor) -> Tensor:
        """Average log-Holevo across scales (1D tensor).  Returns (B,)."""
        prec = self.prec
        n = int(scales.shape[0])
        acc = tf.zeros((self.bs,), dtype=prec)
        for j in range(n):
            acc = acc + self.log_holevo_at_scale(scales[j])
        return acc / tf.cast(n, prec)

class FixedMultiScaleController(tf.keras.Model):
    """Diagnostic controller that cycles through multiple gain scales."""

    def __init__(
        self,
        phys_model,
        bank_cfg,
        input_size,
        *,
        u_B_value: float = 0.0,
        u_phi_value: float = 0.5,
        phase_flip: bool = True,
        gain_values=(-1.0, -0.5, 0.0, 0.5),
    ):
        dtype = tf.float32 if phys_model.prec == "float32" else tf.float64
        super().__init__(name="fixed_multiscale_controller", dtype=dtype)

        self._dtype = dtype
        self.layer = ControlScheduleLayer(
            phys_model=phys_model,
            schedule_floor=bank_cfg.schedule_floor,
            k_max=bank_cfg.k_max,
            prec=phys_model.prec,
            posterior_gain_width_multiplier=bank_cfg.posterior_gain_width_multiplier,
            min_gain_fringe_fraction=bank_cfg.min_gain_fringe_fraction,
            log_sigma_frac_bounds=bank_cfg.log_sigma_frac_bounds,
            dtype=dtype,
        )
        self.u_B_value = float(u_B_value)
        self.u_phi_value = float(u_phi_value)
        self.phase_flip = bool(phase_flip)
        self.gain_values = [float(v) for v in gain_values]

    def call(self, inputs, training=False):
        del training
        inputs = tf.cast(inputs, self._dtype)
        mix_log_std_norm = inputs[:, -3:-2]
        step_norm = inputs[:, -2:-1]

        # Convert step_norm in [-1,1] into a rough step index bucket.
        # Four buckets over the episode.
        s = (step_norm + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        idx = tf.cast(tf.floor(s * tf.cast(len(self.gain_values), self._dtype)), tf.int32)
        idx = tf.clip_by_value(idx, 0, len(self.gain_values) - 1)

        gains = tf.constant(self.gain_values, dtype=self._dtype)
        u_kg = tf.gather(gains, idx[:, 0])
        u_kg = tf.expand_dims(u_kg, axis=1)

        u_B = tf.fill(tf.shape(step_norm), tf.cast(self.u_B_value, self._dtype))

        if self.phase_flip:
            u_phi = tf.where(
                tf.math.floormod(idx, 2) == 0,
                tf.fill(tf.shape(step_norm), tf.cast(+abs(self.u_phi_value), self._dtype)),
                tf.fill(tf.shape(step_norm), tf.cast(-abs(self.u_phi_value), self._dtype)),
            )
        else:
            u_phi = tf.fill(tf.shape(step_norm), tf.cast(self.u_phi_value, self._dtype))

        x = tf.concat([u_kg, u_B, u_phi, mix_log_std_norm, step_norm], axis=1)
        return self.layer(x)
    

class FixedRampController(tf.keras.Model):
    """Non-trainable diagnostic controller with configurable constants."""

    def __init__(
        self,
        phys_model,
        bank_cfg,
        input_size,
        *,
        u_kg_value: float = 0.0,
        u_B_value: float = 0.0,
        u_phi_value: float = 0.5,
        phase_flip: bool = False,
    ):
        dtype = tf.float32 if phys_model.prec == "float32" else tf.float64

        # Important: set the Keras model compute dtype, otherwise subclassed
        # models may default/autocast inputs to float32.
        super().__init__(name="fixed_ramp_controller", dtype=dtype)

        self._dtype = dtype
        self.layer = ControlScheduleLayer(
            phys_model=phys_model,
            schedule_floor=bank_cfg.schedule_floor,
            k_max=bank_cfg.k_max,
            prec=phys_model.prec,
            posterior_gain_width_multiplier=bank_cfg.posterior_gain_width_multiplier,
            min_gain_fringe_fraction=bank_cfg.min_gain_fringe_fraction,
            log_sigma_frac_bounds=bank_cfg.log_sigma_frac_bounds,
            dtype=dtype,
        )

        self.u_kg_value = float(u_kg_value)
        self.u_B_value = float(u_B_value)
        self.u_phi_value = float(u_phi_value)
        self.phase_flip = bool(phase_flip)

    def call(self, inputs, training=False):
        del training

        # Force all internal arithmetic to the fixed controller dtype.
        inputs = tf.cast(inputs, self._dtype)
        mix_log_std_norm = inputs[:, -3:-2]
        step_norm = inputs[:, -2:-1]

        u_kg = tf.fill(
            tf.shape(step_norm),
            tf.cast(self.u_kg_value, self._dtype),
        )
        u_B = tf.fill(
            tf.shape(step_norm),
            tf.cast(self.u_B_value, self._dtype),
        )

        if self.phase_flip:
            u_phi_pos = tf.fill(
                tf.shape(step_norm),
                tf.cast(+abs(self.u_phi_value), self._dtype),
            )
            u_phi_neg = tf.fill(
                tf.shape(step_norm),
                tf.cast(-abs(self.u_phi_value), self._dtype),
            )
            u_phi = tf.where(
                step_norm < tf.cast(0.0, self._dtype),
                u_phi_pos,
                u_phi_neg,
            )
        else:
            u_phi = tf.fill(
                tf.shape(step_norm),
                tf.cast(self.u_phi_value, self._dtype),
            )

        x = tf.concat([u_kg, u_B, u_phi, mix_log_std_norm, step_norm], axis=1)
        return self.layer(x)

# ---------------------------------------------------------------------------
# Controller: schedule layer
# ---------------------------------------------------------------------------

class ControlScheduleLayer(tf.keras.layers.Layer):
    """Maps the NN's raw tanh outputs to physical (T_s, Bp_kTm, mw_phase_rad)."""

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        schedule_floor: float,
        k_max: int,
        prec: str,
        posterior_gain_width_multiplier: float,
        min_gain_fringe_fraction: float,
        log_sigma_frac_bounds: Tuple[float, float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._cfg = phys_model.cfg
        self._schedule_floor = float(max(0.0, min(1.0, schedule_floor)))
        self._dtype = tf.float32 if prec == "float32" else tf.float64
        cfg = self._cfg
        T_min, T_max = float(cfg.T_range_s[0]), float(cfg.T_range_s[1])
        Bp_min, Bp_max = float(cfg.Bp_range_kTm[0]), float(cfg.Bp_range_kTm[1])
        self._T_min = tf.constant(T_min, dtype=self._dtype)
        self._T_max = tf.constant(T_max, dtype=self._dtype)
        self._Bp_min = tf.constant(Bp_min, dtype=self._dtype)
        self._Bp_max = tf.constant(Bp_max, dtype=self._dtype)
        self._gamma = tf.constant(cfg.gamma_e_rad_s_T, dtype=self._dtype)
        self._omega = tf.constant(cfg.omega_rad_s, dtype=self._dtype)
        self._hbar = tf.constant(cfg.hbar_J_s, dtype=self._dtype)
        self._kT = tf.constant(cfg.kT_to_T, dtype=self._dtype)
        self._posterior_gain_width_multiplier = tf.constant(
            float(posterior_gain_width_multiplier),
            dtype=self._dtype,
        )
        self._min_gain_fringe_fraction = float(min_gain_fringe_fraction)
        self._log_sigma_frac_lo = tf.constant(float(log_sigma_frac_bounds[0]), dtype=self._dtype)
        self._log_sigma_frac_hi = tf.constant(float(log_sigma_frac_bounds[1]), dtype=self._dtype)

        g_lo, g_hi = float(cfg.g_range[0]), float(cfg.g_range[1])
        self._g_range = tf.constant(max(g_hi - g_lo, 1e-30), dtype=self._dtype)
        # k_g at (T_min, B_min) and (T_max, B_max) define the global k_g range.
        k_g_min = (
            2.0 * self._gamma / self._omega
            * (self._Bp_min * self._kT) * tf.square(self._T_min)
            + 8.0 * tf.cast(pi, self._dtype) * self._gamma / (self._omega ** 3)
            * (self._Bp_min * self._kT)
        )

        k_g_max = (
            2.0 * self._gamma / self._omega
            * (self._Bp_max * self._kT) * tf.square(self._T_max)
            + 8.0 * tf.cast(pi, self._dtype) * self._gamma / (self._omega ** 3)
            * (self._Bp_max * self._kT)
        )
        # Floor of the schedule = one fringe across the prior.
        g_lo, g_hi = float(cfg.g_range[0]), float(cfg.g_range[1])
        k_g_floor = tf.cast(
            self._min_gain_fringe_fraction * 2.0 * pi / max(g_hi - g_lo, 1e-30),
            self._dtype,
        )
        self._k_g_min = tf.maximum(k_g_min, k_g_floor)
       
        g_range_py = max(g_hi - g_lo, 1e-30)
        mode_width = g_range_py / float(max(k_max, 1))
        k_g_alias_cap = tf.cast(pi / mode_width, self._dtype)
        self._k_g_max = tf.minimum(k_g_max, k_g_alias_cap)
        # Pre-cache log10 versions.
        self._log_k_g_min = tf.math.log(self._k_g_min) / tf.cast(np.log(10.0), self._dtype)
        self._log_k_g_max = tf.math.log(self._k_g_max) / tf.cast(np.log(10.0), self._dtype)

    def call(self, x_step):
        """x_step: tensor of shape (B, 4): (u_kg, u_B, u_phi, step_norm)
        where step_norm in [-1, 1] encodes the current meas_step.

        Returns (B, 3): (T_s, Bp_kTm, mw_phase_rad).
        """
        u_kg = x_step[:, 0:1]
        u_B = x_step[:, 1:2]
        u_phi = x_step[:, 2:3]
        mix_log_std_norm = x_step[:, 3:4]
        step_norm = x_step[:, 4:5]
        
        frac = tf.clip_by_value(
            (step_norm + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype),
            tf.cast(self._schedule_floor, self._dtype),
            tf.cast(1.0, self._dtype),
        )
        # Decode normalized log10(sigma_g / prior_width).
        log_sigma_frac = self._log_sigma_frac_lo + (
            (mix_log_std_norm + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (self._log_sigma_frac_hi - self._log_sigma_frac_lo)

        sigma_frac = tf.pow(tf.cast(10.0, self._dtype), log_sigma_frac)
        sigma_g = sigma_frac * self._g_range

        # Posterior-safe gain cap:
        # approximately one fringe across posterior_gain_width_multiplier * sigma_g.
        k_post_cap = (
            tf.cast(2.0 * pi, self._dtype)
            / tf.maximum(
                self._posterior_gain_width_multiplier * sigma_g,
                tf.cast(1e-12, self._dtype),
            )
        )

        effective_k_g_max = tf.minimum(self._k_g_max, k_post_cap)
        effective_k_g_max = tf.maximum(effective_k_g_max, self._k_g_min)

        log_k_g_max_eff = (
            tf.math.log(effective_k_g_max)
            / tf.cast(np.log(10.0), self._dtype)
        )
        # Schedule the upper bound of k_g.  Log-linear interpolation between
        # k_g_min (always allowed) and k_g_max * frac.
        # log_max = self._log_k_g_min + frac * (self._log_k_g_max - self._log_k_g_min)
        log_max = self._log_k_g_min + frac * (log_k_g_max_eff - self._log_k_g_min)
        log_kg_target = self._log_k_g_min + (
            (u_kg + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (log_max - self._log_k_g_min)
        k_g_target = tf.math.pow(tf.cast(10.0, self._dtype), log_kg_target)
        
        
        # ------------------------------------------------------------------
        # Choose B' from the feasible interval for k_g_target.
        #
        # k_g(T, B') = C(T) * Bp_T_per_m
        #            = C(T) * (Bp_kTm * kT_to_T)
        #
        # For fixed target k_g_target and T in [T_min, T_max],
        # feasible Bp_kTm must satisfy:
        #
        #   k_g_target / (C(T_max) * kT_to_T) <= Bp_kTm
        #   Bp_kTm <= k_g_target / (C(T_min) * kT_to_T)
        #
        # This prevents the old failure mode:
        #   choose high Bp -> solve T below T_min -> clip T -> actual k_g too high.
        # ------------------------------------------------------------------

        C_T_min = (
            tf.cast(2.0, self._dtype) * self._gamma / self._omega * tf.square(self._T_min)
            + tf.cast(8.0 * pi, self._dtype) * self._gamma / (self._omega ** 3)
        )
        C_T_max = (
            tf.cast(2.0, self._dtype) * self._gamma / self._omega * tf.square(self._T_max)
            + tf.cast(8.0 * pi, self._dtype) * self._gamma / (self._omega ** 3)
        )

        Bp_feas_low = k_g_target / tf.maximum(
            C_T_max * self._kT,
            tf.cast(1e-30, self._dtype),
        )
        Bp_feas_high = k_g_target / tf.maximum(
            C_T_min * self._kT,
            tf.cast(1e-30, self._dtype),
        )

        Bp_low = tf.maximum(self._Bp_min, Bp_feas_low)
        Bp_high = tf.minimum(self._Bp_max, Bp_feas_high)

        # If numerical/pathological settings make the interval invalid, collapse
        # to the closest feasible hardware value. This should rarely happen if
        # k_g_target is bounded between min and max achievable gain.
        Bp_high = tf.maximum(Bp_high, Bp_low)

        Bp = Bp_low + (
            (u_B + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (Bp_high - Bp_low)

        # Solve for T after choosing a feasible Bp.
        Bp_T = Bp * self._kT

        C_without_T = tf.cast(8.0 * pi, self._dtype) * self._gamma / (self._omega ** 3)
        a = tf.cast(2.0, self._dtype) * self._gamma / self._omega * Bp_T
        b_const = C_without_T * Bp_T

        T_sq = (k_g_target - b_const) / tf.maximum(a, tf.cast(1e-30, self._dtype))
        T_sq = tf.maximum(T_sq, tf.cast(0.0, self._dtype))
        T_s = tf.sqrt(T_sq)

        # This clip should now be inactive except for numerical edge cases.
        T_s = tf.clip_by_value(T_s, self._T_min, self._T_max)
        # MW phase free over (-pi, pi).
        mw_phase = tf.cast(pi, self._dtype) * u_phi
        return tf.concat([T_s, Bp, mw_phase], axis=1)


def build_controller(
    input_size: int,
    phys_model: GravityStatelessPhysicalModel,
    bank_cfg: MultiPFBankConfig,
    hidden_sizes: Tuple[int, ...] = (64, 64),
) -> tf.keras.Model:
    """MLP -> ControlScheduleLayer.

    Default architecture: 2 hidden layers of 64 units, ~6k parameters.
    Smaller than the original (~30k) to reduce REINFORCE gradient variance
    at fixed batch size.
    """
    cfg = phys_model.cfg
    prec = cfg.prec
    dtype = tf.float32 if prec == "float32" else tf.float64
    inputs = tf.keras.Input(shape=(input_size,), dtype=dtype)
    x = inputs
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="tanh", dtype=dtype)(x)
    x_raw = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    # Step_norm is the 2nd-to-last feature of the input.  We splice it back in
    # for the schedule layer.  See generate_input() for the layout contract.
    
    # step_norm = inputs[:, -2:-1]
    # x_with_step = tf.keras.layers.Concatenate(axis=1, dtype=dtype)([x_raw, step_norm])
    # outputs = ControlScheduleLayer(
    #     phys_model=phys_model,
    #     schedule_floor=bank_cfg.schedule_floor,
    #     k_max=bank_cfg.k_max,
    #     prec=prec,
    #     dtype=dtype,
    # )(x_with_step)

    # Global layout ends with:
    #   mix_log_std_norm, step_norm, res_norm
    mix_log_std_norm = inputs[:, -3:-2]
    step_norm = inputs[:, -2:-1]

    x_with_sched = tf.keras.layers.Concatenate(axis=1, dtype=dtype)(
        [x_raw, mix_log_std_norm, step_norm]
    )

    outputs = ControlScheduleLayer(
        phys_model=phys_model,
        schedule_floor=bank_cfg.schedule_floor,
        k_max=bank_cfg.k_max,
        prec=prec,
        posterior_gain_width_multiplier=bank_cfg.posterior_gain_width_multiplier,
        min_gain_fringe_fraction=bank_cfg.min_gain_fringe_fraction,
        log_sigma_frac_bounds=bank_cfg.log_sigma_frac_bounds,
        dtype=dtype,
    )(x_with_sched)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gravity_controller")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class GravityMultiPFSimulation(StatelessSimulation):
    """qsensoropt StatelessSimulation on top of the Multi-PF bank."""

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: MultiPFBank,
        controller: tf.keras.Model,
        simpars: SimulationParameters,
        bank_cfg: MultiPFBankConfig,
    ) -> None:
        
        top_k = bank_cfg.top_k_modes

        # Per top-mode features:
        #   mu_norm, log_std_norm, q_k
        per_top_mode_size = 3 * top_k

        # Full ordered mode histogram q_1...q_K.
        q_hist_size = bank.K if bank_cfg.use_full_q_hist else 0

        # Globals:
        #   log1p_VH_norm,
        #   H_q_norm,
        #   prev_g_mean_norm,
        #   max_q,
        #   mix_log_std_norm,
        #   step_norm,
        #   res_norm
        global_size = 7

        input_size = per_top_mode_size + q_hist_size + global_size

        input_name = []
        for i in range(top_k):
            input_name += [f"mu_mode_{i}", f"log_std_mode_{i}", f"q_mode_{i}"]

        if bank_cfg.use_full_q_hist:
            input_name += [f"q_hist_{k}" for k in range(bank.K)]

        input_name += [
            "log1p_VH_norm",
            "H_q_norm",
            "prev_g_mean_norm",
            "max_q",
            "mix_log_std_norm",
            "step_norm",
            "res_norm",
        ]
        super().__init__(
            particle_filter=bank.pf_list[0],
            phys_model=phys_model,
            control_strategy=controller,
            input_size=input_size,
            input_name=input_name,
            simpars=simpars,
        )

        self.bank = bank
        self.bank_cfg = bank_cfg
        self.g_lo = float(phys_model.cfg.g_range[0])
        self.g_hi = float(phys_model.cfg.g_range[1])
        prec = phys_model.prec
        dtype = tf.float32 if prec == "float32" else tf.float64
        self._k_g_max: float = float(phys_model.max_gain(dtype).numpy())
        # Diagnostic-only Holevo scales (used in _bank_snapshot for logging).
        k_coarsest = 2.0 * pi / max(self.g_hi - self.g_lo, 1e-30)
        self._k_ref_coarsest = tf.constant(k_coarsest, dtype=dtype)
        # State carried between steps (for prev_g_map feature).
        self._prev_g_map_norm = None  # set in execute()

    # ------------------------------------------------------------------
    # Per-step loss
    # ------------------------------------------------------------------

    # def _per_step_loss(self, true_values: Optional[Tensor] = None) -> Tensor:
    #     """Closest-mode oracle MSE plus a mode-resolution penalty.

    #     At training time we know g_true.  Of the K modes, we pick the one
    #     whose mean is closest to g_true ("oracle mode") and compute its
    #     squared error.  This gives a clean training signal that the
    #     controller can act on:
    #       * If the bank has a mode near g_true with small variance -> low loss.
    #       * If no mode is near g_true -> large loss; gradient pushes the bank
    #         to keep particles near g_true alive.
    #       * The mode-SELECTION (argmax q vs. argmin d2) is decoupled from the
    #         ESTIMATION precision.  At eval, MAP mode is used; if training
    #         succeeds, MAP and closest agree.

    #     A mode-resolution penalty `mu * (1 - q_closest)` rewards making the
    #     oracle mode have a high q-weight.  This drives MAP and closest
    #     toward agreement.

    #     Returns shape (B,).
    #     """
    #     prec = self.simpars.prec
    #     if true_values is None:
    #         # Pre-measurement state (used only by callers that pass no g_true,
    #         # e.g. for episode-start diagnostic).  Return zeros.
    #         return tf.zeros((self.bs,), dtype=prec)
    #     g_true = true_values[:, 0, 0]                        # (B,)
    #     g_best, _, q_best = self.bank.closest_mode_estimate(g_true)
    #     norm = tf.cast((self.g_hi - self.g_lo) ** 2, prec)
    #     err2 = tf.square(g_best - g_true) / tf.maximum(norm, tf.cast(1e-30, prec))
    #     # q-penalty: pushes the controller to make the oracle mode dominate.
    #     # mu_q is on the same scale as err2 once the bank starts localizing
    #     # (err2 << 1 once a single mode is right).
    #     mu_q = tf.cast(self.bank_cfg.mode_penalty_coef, prec)
    #     q_pen = mu_q * (tf.cast(1.0, prec) - q_best)
    #     return err2 + q_pen
    # def _per_step_loss(self, true_values: Optional[Tensor] = None) -> Tensor:
    #     """Bayes-posterior MSE under the bank mixture, normalised by prior width^2.

    #         L_b = sum_k q_k * (mu_k - g_true)^2 / (g_hi - g_lo)^2

    #     Properties (verified against Belliardo SI C.2 and Wang Eq. 3):
    #     * Linear in q_k, quadratic in within-mode weights => polynomial in PF
    #         weights, so Scibior-Wood resampling correction is UNBIASED.
    #     * Differentiable in q_k and mu_k (no argmin/argmax).
    #     * Matches the eval estimator (marginal mean, see Fix 5): training
    #         and deployment optimise the same quantity.
    #     * Dynamic range ~4 OOM in an episode (init ~0.17 -> localised ~1.5e-5),
    #         which is why Fix 4 introduces per-step log-scaling per Belliardo Eq. 109.
    #     """
    #     prec = self.simpars.prec
    #     if true_values is None:
    #         return tf.zeros((self.bs,), dtype=prec)

    #     g_true = true_values[:, 0, 0]                              # (B,)
    #     means, _ = self.bank.mode_means_and_stds()                 # (B, K)
    #     q = self.bank.mode_weights                                 # (B, K)
    #     norm = tf.cast((self.g_hi - self.g_lo) ** 2, prec)
    #     sq_err_per_mode = tf.square(means - tf.expand_dims(g_true, 1))   # (B, K)
    #     loss_b = tf.reduce_sum(q * sq_err_per_mode, axis=1) / tf.maximum(
    #         norm, tf.cast(1e-30, prec)
    #     )
    #     return loss_b
    def _per_step_loss(self, true_values: Optional[Tensor] = None) -> Tensor:
        """Posterior risk under the bank mixture, normalized by prior width^2.

        L_b = sum_k q_k [ (mu_k - g_true)^2 + sigma_k^2 ] / Delta_g^2
        """
        prec = self.simpars.prec
        if true_values is None:
            return tf.zeros((self.bs,), dtype=prec)

        g_true = true_values[:, 0, 0]
        means, stds = self.bank.mode_means_and_stds()
        q = self.bank.mode_weights

        norm = tf.cast((self.g_hi - self.g_lo) ** 2, prec)
        sq_bias = tf.square(means - tf.expand_dims(g_true, axis=1))
        local_var = tf.square(stds)

        loss_b = tf.reduce_sum(q * (sq_bias + local_var), axis=1)
        return loss_b / tf.maximum(norm, tf.cast(1e-30, prec))
    # ------------------------------------------------------------------
    # Controller input
    # ------------------------------------------------------------------

    def generate_input(self, weights, particles, meas_step, used_resources, rangen):
        del weights, particles, rangen
        bank = self.bank
        prec = self.simpars.prec
        simpars = self.simpars
        TOP_K = self.bank_cfg.top_k_modes
        bs = self.bs

        means_all, stds_all = bank.mode_means_and_stds()
        q_all = bank.mode_weights

        topk = tf.math.top_k(q_all, k=TOP_K, sorted=True)
        top_q = topk.values
        top_idx = topk.indices
        batch_idx = tf.broadcast_to(
            tf.expand_dims(tf.range(bs, dtype=tf.int32), axis=1),
            (bs, TOP_K),
        )
        gather = tf.stack([batch_idx, top_idx], axis=2)
        top_mu = tf.gather_nd(means_all, gather)
        top_std = tf.gather_nd(stds_all, gather)

        g_range = tf.cast(max(self.g_hi - self.g_lo, 1e-30), prec)
        mu_norm = 2.0 * (top_mu - tf.cast(self.g_lo, prec)) / g_range - 1.0
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)

        if prec == "float64":
            log_std_lo, log_std_hi = -13.0, -1.0
        else:
            log_std_lo, log_std_hi = -7.0, -1.0
        log_std = tf.math.log(tf.maximum(top_std, tf.cast(1e-30, prec))) / tf.cast(np.log(10.0), prec)
        log_std_norm = 2.0 * (log_std - log_std_lo) / (log_std_hi - log_std_lo) - 1.0
        log_std_norm = tf.clip_by_value(log_std_norm, -1.0, 1.0)

        per_mode = tf.reshape(
            tf.stack([mu_norm, log_std_norm, top_q], axis=2),
            (bs, 3 * TOP_K),
        )

        # Globals.
        # v_h_coarsest = bank.log_holevo_at_scale(self._k_ref_coarsest)
        # K = bank.K
        # v_h_max_log = tf.cast(np.log(float(K) + 1.0) + 30.0 * np.log(10.0), prec)
        # log1p_vh_norm = tf.clip_by_value(
        #     v_h_coarsest / v_h_max_log, tf.cast(0.0, prec), tf.cast(1.0, prec),
        # )
        # q_safe = tf.maximum(q_all, tf.cast(1e-30, prec))
        # h_q = -tf.reduce_sum(q_all * tf.math.log(q_safe), axis=1)
        # h_q_norm = h_q / tf.cast(np.log(float(K)), prec)
        # h_q_norm = tf.clip_by_value(h_q_norm, 0.0, 1.0)

        # # prev_g_map_norm: normalized previous MAP-mode estimate.  Zero at start.
        # if self._prev_g_map_norm is None:
        #     prev_g_map_norm = tf.zeros((bs,), dtype=prec)
        # else:
        #     prev_g_map_norm = self._prev_g_map_norm

        # # max_q: the highest mode weight — strong indicator of mode resolution.
        # max_q = tf.reduce_max(q_all, axis=1)

        # step_norm = 2.0 * tf.cast(meas_step[:, 0], prec) / float(simpars.num_steps) - 1.0
        # res_norm = 2.0 * used_resources[:, 0] / tf.cast(simpars.max_resources, prec) - 1.0
        # step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)
        # res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        # globals_ = tf.stack(
        #     [log1p_vh_norm, h_q_norm, prev_g_map_norm, max_q, step_norm, res_norm],
        #     axis=1,
        # )
        # return tf.concat([per_mode, globals_], axis=1)
        
        v_h_coarsest = bank.log_holevo_at_scale(self._k_ref_coarsest)
        K = bank.K
        v_h_max_log = tf.cast(np.log(float(K) + 1.0) + 30.0 * np.log(10.0), prec)
        log1p_vh_norm = tf.clip_by_value(
            v_h_coarsest / v_h_max_log,
            tf.cast(0.0, prec),
            tf.cast(1.0, prec),
        )

        q_safe = tf.maximum(q_all, tf.cast(1e-30, prec))
        h_q = -tf.reduce_sum(q_all * tf.math.log(q_safe), axis=1)
        h_q_norm = h_q / tf.cast(np.log(float(K)), prec)
        h_q_norm = tf.clip_by_value(h_q_norm, 0.0, 1.0)

        # Previous marginal-mean estimate, not MAP.
        if self._prev_g_map_norm is None:
            prev_g_mean_norm = tf.zeros((bs,), dtype=prec)
        else:
            prev_g_mean_norm = self._prev_g_map_norm

        max_q = tf.reduce_max(q_all, axis=1)

        # Marginal posterior width. This is the key structural signal
        # for safe gain scheduling.
        g_mix, g_var = bank.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-30, prec)))

        prior_width = tf.cast(max(self.g_hi - self.g_lo, 1e-30), prec)
        sigma_frac = g_std / prior_width
        sigma_frac = tf.maximum(sigma_frac, tf.cast(1e-12, prec))

        log_sigma_frac = tf.math.log(sigma_frac) / tf.cast(np.log(10.0), prec)
        lo_py, hi_py = self.bank_cfg.log_sigma_frac_bounds
        lo = tf.cast(lo_py, prec)
        hi = tf.cast(hi_py, prec)

        log_sigma_frac_clip = tf.clip_by_value(log_sigma_frac, lo, hi)
        mix_log_std_norm = (
            2.0 * (log_sigma_frac_clip - lo) / tf.maximum(hi - lo, tf.cast(1e-12, prec))
            - 1.0
        )
        mix_log_std_norm = tf.clip_by_value(mix_log_std_norm, -1.0, 1.0)

        step_norm = 2.0 * tf.cast(meas_step[:, 0], prec) / float(simpars.num_steps) - 1.0
        res_norm = 2.0 * used_resources[:, 0] / tf.cast(simpars.max_resources, prec) - 1.0
        step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)
        res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        globals_ = tf.stack(
            [
                log1p_vh_norm,
                h_q_norm,
                prev_g_mean_norm,
                max_q,
                mix_log_std_norm,
                step_norm,
                res_norm,
            ],
            axis=1,
        )

        if self.bank_cfg.use_full_q_hist:
            return tf.concat([per_mode, q_all, globals_], axis=1)

        return tf.concat([per_mode, globals_], axis=1)

    # ------------------------------------------------------------------
    # Loss function (required by StatelessSimulation API; called by parent)
    # ------------------------------------------------------------------

    def loss_function(
        self,
        weights: Tensor, particles: Tensor,
        true_values: Tensor, used_resources: Tensor, meas_step: Tensor,
    ) -> Tensor:
        del weights, particles, used_resources, meas_step
        return tf.expand_dims(self._per_step_loss(true_values=true_values), axis=1)

    # ------------------------------------------------------------------
    # Debug snapshot
    # ------------------------------------------------------------------

    def _bank_snapshot(
        self,
        true_values, controls, used_resources, meas_step, max_examples=3,
    ):
        bank = self.bank
        K = bank.K
        prec = self.simpars.prec

        g_mix, g_var = bank.marginal_mean_and_var()
        g_map, g_std_map = bank.map_mode_estimate()
        g_close, std_close, q_close = bank.closest_mode_estimate(true_values[:, 0, 0])
        l_per_b = self._per_step_loss(true_values=true_values)
        v_h_coarsest = bank.holevo_variance(self._k_ref_coarsest)[:, 0]
        k_g = self.phys_model.k_g(controls[:, 0], controls[:, 1])
        v_h_meas = bank.holevo_variance(k_g)[:, 0]

        q_np = bank.mode_weights.numpy()
        means_t, stds_t = bank.mode_means_and_stds()
        means_np = means_t.numpy()
        stds_np = stds_t.numpy()

        n_show = min(int(max_examples), self.bs)
        records = []
        for b in range(n_show):
            order = np.argsort(-q_np[b])
            mode_width = (self.g_hi - self.g_lo) / float(K)
            true_g_b = float(true_values[b, 0, 0].numpy())

            true_mode = int(np.floor((true_g_b - self.g_lo) / max(mode_width, 1e-30)))
            true_mode = int(np.clip(true_mode, 0, K - 1))

            q_true_mode = float(q_np[b, true_mode])
            true_mode_rank = int(np.where(order == true_mode)[0][0] + 1)

            map_mode = int(order[0])
            map_mode_error = float(means_np[b, map_mode] - true_g_b)

            top_modes = []
            for k in order[: min(3, K)]:
                top_modes.append({
                    "mode": int(k),
                    "q": float(q_np[b, k]),
                    "mu": float(means_np[b, k]),
                    "std": float(stds_np[b, k]),
                })
            records.append({
                "batch_idx": int(b),
                "meas_step": int(meas_step[b, 0].numpy()),
                "used_resources": float(used_resources[b, 0].numpy()),
                "T_s": float(controls[b, 0].numpy()),
                "Bp_kTm": float(controls[b, 1].numpy()),
                "mw_phase_rad": float(controls[b, 2].numpy()),
                "true_g": float(true_values[b, 0, 0].numpy()),
                "g_mix": float(g_mix[b].numpy()),
                "g_var": float(g_var[b].numpy()),
                "g_map": float(g_map[b].numpy()),
                "g_std_map": float(g_std_map[b].numpy()),
                "g_close": float(g_close[b].numpy()),
                "q_close": float(q_close[b].numpy()),
                "loss": float(l_per_b[b].numpy()),
                "V_H_coarsest": float(v_h_coarsest[b].numpy()),
                "V_H_at_kg": float(v_h_meas[b].numpy()),
                "k_g": float(k_g[b].numpy()),
                "K": int(K),
                "top_modes": top_modes,
                "true_mode": true_mode,
                "q_true_mode": q_true_mode,
                "true_mode_rank": true_mode_rank,
                "map_mode": map_mode,
                "map_mode_error": map_mode_error,
            })
        return records

    # ------------------------------------------------------------------
    # Measurement loop
    # ------------------------------------------------------------------

    def execute(
        self,
        rangen: tf.random.Generator,
        deploy: bool = False,
        debug: bool = False,
        debug_max_examples: int = 3,
    ):
        """Run one episode.  Returns (loss_diff, loss) or deploy payload.

        The REINFORCE baseline is always mean_b(L_t) within each step
        (Belliardo Eq 93, pars.baseline=True).  No external EMA baseline
        is accepted: passing one caused near-zero advantages and interacted
        badly with any per-step loss scaling.
        """
        pars = self.simpars
        prec = pars.prec
        bank = self.bank
        debug_records: List[dict] = [] if debug else None

        bank.reset(rangen)
        # Reset per-episode controller state.
        self._prev_g_map_norm = tf.zeros((self.bs,), dtype=prec)
        weights = bank.weights_list[0]
        particles = bank.particles_list[0]

        true_values = self.phys_model.true_values(rangen)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype="bool")
        outcomes = tf.zeros(
            (self.bs, self.phys_model.outcomes_size),
            dtype=self.phys_model.prec,
        )
        meas_step = tf.zeros((self.bs, 1), dtype="int32")
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        loss_diff_acc = tf.zeros((), dtype=prec)
        loss_acc = tf.zeros((), dtype=prec)
        step_count = 0
        g_lo_t = tf.cast(self.g_lo, prec)
        g_range_t = tf.cast(self.g_hi - self.g_lo, prec)

        if deploy:
            hist_inputs: List[Tensor] = []
            hist_controls: List[Tensor] = []
            hist_resources: List[Tensor] = []
            hist_precisions: List[Tensor] = []

        for _i in range(pars.num_steps):
            num_finished = int(
                tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy()
            )
            if num_finished >= pars.resources_fraction * self.bs:
                break

            input_strategy = self.generate_input(
                weights, particles,
                tf.cast(meas_step, prec),
                used_resources,
                rangen,
            )
            cond_input = (
                tf.stop_gradient(input_strategy)
                if pars.stop_gradient_input else input_strategy
            )
            controls = self.control_strategy(cond_input)
            
            if USE_FIXED_CONTROLLER and FIXED_ORACLE_PHASE:
                # Diagnostic only: use true g to set readout at quadrature.
                # theta = k_g * g_true + phi_mw = pi/2
                kg_now = self.phys_model.k_g(controls[:, 0], controls[:, 1])
                g_true_now = true_values[:, 0, 0]
                phi_oracle = wrap_to_pi_local(
                    tf.cast(0.5 * pi, controls.dtype) - kg_now * g_true_now
                )
                controls = tf.concat(
                    [controls[:, 0:2], tf.expand_dims(phi_oracle, axis=1)],
                    axis=1,
                )

            new_used_resources = self.phys_model.wrapper_count_resources(
                used_resources, outcomes, controls, true_values, true_state, meas_step,
            )
            new_continue_flag = tf.math.less_equal(
                new_used_resources,
                pars.max_resources * tf.ones((self.bs, 1), dtype=prec),
            )
            new_continue_flag = tf.logical_and(new_continue_flag, continue_flag)
            used_resources = tf.where(new_continue_flag, new_used_resources, used_resources)
            continue_flag = new_continue_flag

            outcomes_raw, log_prob, post_true_state = (
                self.phys_model.wrapper_perform_measurement(
                    tf.expand_dims(controls, axis=1),
                    true_values,
                    true_state,
                    tf.expand_dims(meas_step, axis=1),
                    rangen,
                )
            )
            outcomes = outcomes_raw[:, 0, :]
            if self.state_size > 0:
                cf3 = tf.reshape(continue_flag, (self.bs, 1, 1))
                true_state = tf.where(
                    tf.broadcast_to(cf3, (self.bs, 1, self.state_size)),
                    post_true_state, true_state,
                )

            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(
                    continue_flag, sum_log_prob + log_prob, sum_log_prob,
                )

            # bank.apply_measurement(
            #     outcomes=outcomes,
            #     controls=controls,
            #     meas_step=meas_step,
            #     continue_flag=continue_flag,
            #     rangen=rangen,
            # )
            bank.apply_measurement(
                outcomes=outcomes,
                controls=controls,
                meas_step=meas_step,
                continue_flag=continue_flag,
                rangen=rangen,
                do_resample=False,
            )
            weights = bank.weights_list[0]
            particles = bank.particles_list[0]
            self.pf.np = bank.pf_list[0].np

            # Update prev_g_map (used as a feature next step).
            # g_map_now, _ = bank.map_mode_estimate()
            # self._prev_g_map_norm = tf.clip_by_value(
            #     2.0 * (g_map_now - g_lo_t) / tf.maximum(g_range_t, tf.cast(1e-30, prec)) - 1.0,
            #     tf.cast(-1.0, prec), tf.cast(1.0, prec),
            # )
            g_mean_now, _ = bank.marginal_mean_and_var()
            self._prev_g_map_norm = tf.clip_by_value(
                2.0 * (g_mean_now - g_lo_t) / tf.maximum(g_range_t, tf.cast(1e-30, prec)) - 1.0,
                tf.cast(-1.0, prec),
                tf.cast(1.0, prec),
            )

            L_step_b = self._per_step_loss(true_values=true_values)

            # if pars.stop_gradient_pf:
            #     bank.mode_weights = tf.stop_gradient(bank.mode_weights)
            #     bank.weights_list = [tf.stop_gradient(w) for w in bank.weights_list]
            #     bank.particles_list = [tf.stop_gradient(p) for p in bank.particles_list]
            #     self._prev_g_map_norm = tf.stop_gradient(self._prev_g_map_norm)

            if debug:
                step_records = self._bank_snapshot(
                    true_values=true_values, controls=controls,
                    used_resources=used_resources, meas_step=meas_step,
                    max_examples=debug_max_examples,
                )
                mean_loss = float(tf.reduce_mean(L_step_b).numpy())
                for rec in step_records:
                    rec["loop_iter"] = int(_i)
                    rec["mean_step_loss"] = mean_loss
                debug_records.extend(step_records)

            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            if not deploy:
                active = tf.cast(continue_flag[:, 0], prec)
                n_active = tf.maximum(tf.reduce_sum(active), tf.cast(1.0, prec))
                mean_L_step = tf.reduce_sum(L_step_b * active) / n_active

                # Reported loss: plain cumulative mean (no log_loss scaling).
                # log_loss=True is for scale-invariant losses (e.g. Holevo).
                # Our closest-mode MSE is naturally bounded in [0, ~0.1 + penalty],
                # so log_loss amplification is harmful. We always use the linear path.
                # loss_acc = loss_acc + mean_L_step

                
                # if pars.loss_logl_outcomes:
                #     # Belliardo Eq. 96: surrogate = l + sg(l - B) * log p,
                #     # with B a CONSTANT across the batch. The previous code did
                #     # `(sg(l) - added)` with `added` NOT stop-gradient'd, which
                #     # leaks d(added)/dλ * sum_log_prob into the gradient.
                #     if pars.baseline:
                #         baseline_const = tf.stop_gradient(mean_L_step)
                #     else:
                #         baseline_const = tf.zeros((), dtype=prec)
                #     advantage = tf.stop_gradient(L_step_b - baseline_const)
                #     per_batch_surrogate = L_step_b + advantage * sum_log_prob[:, 0]
                #     per_batch_surrogate = per_batch_surrogate * active
                #     mean_surrogate = tf.reduce_sum(per_batch_surrogate) / n_active
                #     loss_diff_acc = loss_diff_acc + mean_surrogate
                # else:
                #     loss_diff_acc = loss_diff_acc + mean_L_step
                
                
                # ---- Log-scaled cumulative loss (Belliardo Eq. 109, Eq. 121) ----
                # With the q-weighted MSE the per-step loss spans ~4 OOM across an
                # episode. Using a plain linear cumulative sum lets the (large)
                # initial-step losses dominate the gradient and starves the
                # later-step optimisation. Belliardo Eq. 107: divide each l_t by an
                # η(t) of the order of l_t, OR equivalently use the logarithmic loss.
                # We use the latter (cleaner, no calibration of η needed).

                EPS_LOG = tf.cast(1e-30, prec)
                log_mean_L = tf.math.log(tf.maximum(mean_L_step, EPS_LOG))
                loss_acc = loss_acc + log_mean_L

                if pars.loss_logl_outcomes:
                    if pars.baseline:
                        baseline_const = tf.stop_gradient(mean_L_step)
                    else:
                        baseline_const = tf.zeros((), dtype=prec)
                    advantage = tf.stop_gradient(L_step_b - baseline_const)
                    # Belliardo Eq. 121: log-loss surrogate.
                    #   d/dλ log(mean_L_step) is correctly captured by
                    #   (sum_k l_kt + sum_k sg(l_kt) log p_kt) / sg(sum_k l_kt).
                    per_batch_surrogate = L_step_b + advantage * sum_log_prob[:, 0]
                    per_batch_surrogate = per_batch_surrogate * active
                    sum_surrogate = tf.reduce_sum(per_batch_surrogate)
                    denom = tf.maximum(
                        tf.stop_gradient(tf.reduce_sum(L_step_b * active)),
                        EPS_LOG,
                    )
                    loss_diff_acc = loss_diff_acc + sum_surrogate / denom
                else:
                    loss_diff_acc = loss_diff_acc + log_mean_L

            # After computing/accumulating the loss, detach and resample for the next step.
            if pars.stop_gradient_pf:
                bank.detach_state()
                bank.resample_all(
                    continue_flag=continue_flag,
                    rangen=rangen,
                    detach=True,
                )
                self._prev_g_map_norm = tf.stop_gradient(self._prev_g_map_norm)
            else:
                bank.resample_all(
                    continue_flag=continue_flag,
                    rangen=rangen,
                    detach=False,
                )
            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls.append(controls)
                hist_resources.append(used_resources)
                vh_coarsest = bank.log_holevo_at_scale(self._k_ref_coarsest)
                hist_precisions.append(tf.expand_dims(vh_coarsest, axis=1))

        if not deploy:
            denom = tf.cast(max(step_count, 1), prec)
            loss_diff_final = loss_diff_acc / denom
            loss_final = loss_acc / denom
            if debug:
                return loss_diff_final, loss_final, debug_records
            return loss_diff_final, loss_final

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
        deploy_payload = (
            true_values,
            tf.reshape(empty_i, (self.bs * ns, self.input_size)),
            tf.reshape(empty_c, (self.bs * ns, self.phys_model.controls_size)),
            tf.reshape(empty_r, (self.bs * ns, 1)),
            tf.reshape(empty_p, (self.bs * ns, 1)),
        )
        if debug:
            return deploy_payload, debug_records
        return deploy_payload


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gravity_multi_pf_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    bank_cfg: MultiPFBankConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
) -> Tuple["GravityMultiPFSimulation", "MultiPFBank", tf.keras.Model]:
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = MultiPFBank(phys_model=phys_model, cfg=bank_cfg)

    top_k = bank_cfg.top_k_modes
    q_hist_size = bank.K if bank_cfg.use_full_q_hist else 0
    input_size = 3 * top_k + q_hist_size + 7
    
    # controller = build_controller(input_size, phys_model, bank_cfg)
    # controller = FixedRampController(phys_model, bank_cfg, input_size)
    if USE_FIXED_CONTROLLER:
        # controller = FixedRampController(
        #     phys_model,
        #     bank_cfg,
        #     input_size,
        #     u_kg_value=FIXED_U_KG,
        #     u_B_value=FIXED_U_B,
        #     u_phi_value=FIXED_U_PHI,
        #     phase_flip=FIXED_PHASE_FLIP,
        # )
        controller = FixedMultiScaleController(
            phys_model,
            bank_cfg,
            input_size,
            u_B_value=0.0,
            u_phi_value=0.5,
            phase_flip=True,
            gain_values=(-1.0, -0.5, 0.0, 0.5),
        )
    else:
        controller = build_controller(input_size, phys_model, bank_cfg)

    dtype = tf.float32 if cfg.prec == "float32" else tf.float64
    _ = controller(tf.zeros((batchsize, input_size), dtype=dtype))

    bank.reset(rangen)

    sim = GravityMultiPFSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
    )
    return sim, bank, controller