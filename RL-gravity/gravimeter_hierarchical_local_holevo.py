
from __future__ import annotations

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from math import pi
from typing import List, Tuple

from gravimeter_model_complete import GravimeterConfig, GravityStatelessPhysicalModel, wrap_to_pi_tf
from gravimeter_hierarchical_pf_complete import (
    HierarchicalPFBank,
    GravityHierarchicalSimulation,
    SimulationParameters,
    StatelessSimulation,
)


@dataclass(frozen=True)
class LocalHolevoHierarchicalPFConfig:
    n_particles: int = 512
    n_levels: int = 4
    n_disambig_per_level: int = 7
    prec: str = "float32"
    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = True
    trim: bool = True

    # loss weights
    hierarchy_bce_weight: float = 1.0
    hierarchy_local_mse_weight: float = 0.05
    local_holevo_weight: float = 0.25
    gain_penalty_weight: float = 0.10

    # circular/gain shaping
    local_holevo_clip: float = 100.0
    gain_ratio_limit: float = 1.5
    bp_dis_max_ratio: float = 4.0
    phase_residual_max_rad: float = 0.35 * pi


class LocalHolevoHierarchicalPFBank(HierarchicalPFBank):
    """
    Hierarchical bank with two added snapshot losses during disambiguation:

    1) local Holevo variance on the active interval
    2) gain-tracking penalty wrt k* = 2π / width
    """
    def __init__(self, phys_model: GravityStatelessPhysicalModel, cfg: LocalHolevoHierarchicalPFConfig) -> None:
        super().__init__(phys_model=phys_model, cfg=cfg)
        self.cfg: LocalHolevoHierarchicalPFConfig = cfg
        self._loss_local_holevo_snapshot: tf.Tensor | None = None
        self._loss_gain_penalty_snapshot: tf.Tensor | None = None

    def _clear_loss_snapshots(self) -> None:
        super()._clear_loss_snapshots()
        self._loss_local_holevo_snapshot = tf.zeros((self.bs,), dtype=self._dtype)
        self._loss_gain_penalty_snapshot = tf.zeros((self.bs,), dtype=self._dtype)

    def _local_holevo_from_pair(
        self,
        weights0: tf.Tensor,
        particles0: tf.Tensor,
        weights1: tf.Tensor,
        particles1: tf.Tensor,
        q0: tf.Tensor,
        q1: tf.Tensor,
        mid: tf.Tensor,
        width: tf.Tensor,
    ) -> tf.Tensor:
        """
        Returns clipped V_H for *diagnostics* (not for training loss).

        Key correctness property:
        - If qk == 0, that branch's moment is forced to 0 so inactive-branch NaNs
        cannot contaminate the mixture moment (0 * NaN must not appear).
        """
        dtype = self._dtype
        width = tf.maximum(width, tf.cast(1e-8, dtype))
        k_loc = 2.0 * tf.cast(pi, dtype) / width  # (bs,)

        g0 = particles0[:, :, 0]
        g1 = particles1[:, :, 0]

        phase0 = k_loc[:, None] * (g0 - mid[:, None])
        phase1 = k_loc[:, None] * (g1 - mid[:, None])

        re0 = tf.reduce_sum(weights0 * tf.cos(phase0), axis=1)
        im0 = tf.reduce_sum(weights0 * tf.sin(phase0), axis=1)
        re1 = tf.reduce_sum(weights1 * tf.cos(phase1), axis=1)
        im1 = tf.reduce_sum(weights1 * tf.sin(phase1), axis=1)

        # Exact mixture semantics: if qk == 0, that component contributes 0.
        q0_pos = q0 > tf.cast(0.0, dtype)
        q1_pos = q1 > tf.cast(0.0, dtype)
        re0 = tf.where(q0_pos, re0, tf.zeros_like(re0))
        im0 = tf.where(q0_pos, im0, tf.zeros_like(im0))
        re1 = tf.where(q1_pos, re1, tf.zeros_like(re1))
        im1 = tf.where(q1_pos, im1, tf.zeros_like(im1))

        mu_re = q0 * re0 + q1 * re1
        mu_im = q0 * im0 + q1 * im1

        # Abs moment squared in [0,1]. Use a numerical floor only.
        eps = tf.cast(1e-20, dtype)
        abs_mu_sq = tf.maximum(mu_re * mu_re + mu_im * mu_im, eps)

        vh = 1.0 / abs_mu_sq - 1.0
        return tf.clip_by_value(vh, 0.0, tf.cast(self.cfg.local_holevo_clip, dtype))

    def current_local_holevo(self) -> tf.Tensor:
        width = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, self._dtype))
        mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)
        q0 = tf.where(self.refining_mask, tf.ones((self.bs,), dtype=self._dtype), self.mode_weights[:, 0])
        q1 = tf.where(self.refining_mask, tf.zeros((self.bs,), dtype=self._dtype), self.mode_weights[:, 1])
        return self._local_holevo_from_pair(
            self.weights0, self.particles0,
            self.weights1, self.particles1,
            q0, q1, mid, width,
        )
    
    def current_local_holevo_loss(self) -> tf.Tensor:
        """
        Training-safe Holevo-style loss:  -log(|mu|^2 + eps)

        Exactly equals log(1 + V_H) without constructing/clipping V_H:
            log(1 + V_H) = -log(|mu|^2)
        """
        dtype = self._dtype
        width = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, dtype))
        mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)

        q0 = tf.where(self.refining_mask, tf.ones((self.bs,), dtype=dtype), self.mode_weights[:, 0])
        q1 = tf.where(self.refining_mask, tf.zeros((self.bs,), dtype=dtype), self.mode_weights[:, 1])

        # Compute |mu|^2 using the same moment logic as _local_holevo_from_pair,
        # but without forming V_H (and with inactive-branch gating).
        k_loc = 2.0 * tf.cast(pi, dtype) / width

        g0 = self.particles0[:, :, 0]
        g1 = self.particles1[:, :, 0]
        phase0 = k_loc[:, None] * (g0 - mid[:, None])
        phase1 = k_loc[:, None] * (g1 - mid[:, None])

        re0 = tf.reduce_sum(self.weights0 * tf.cos(phase0), axis=1)
        im0 = tf.reduce_sum(self.weights0 * tf.sin(phase0), axis=1)
        re1 = tf.reduce_sum(self.weights1 * tf.cos(phase1), axis=1)
        im1 = tf.reduce_sum(self.weights1 * tf.sin(phase1), axis=1)

        q0_pos = q0 > tf.cast(0.0, dtype)
        q1_pos = q1 > tf.cast(0.0, dtype)
        re0 = tf.where(q0_pos, re0, tf.zeros_like(re0))
        im0 = tf.where(q0_pos, im0, tf.zeros_like(im0))
        re1 = tf.where(q1_pos, re1, tf.zeros_like(re1))
        im1 = tf.where(q1_pos, im1, tf.zeros_like(im1))

        mu_re = q0 * re0 + q1 * re1
        mu_im = q0 * im0 + q1 * im1

        eps = tf.cast(1e-20, dtype)
        abs_mu_sq = tf.maximum(mu_re * mu_re + mu_im * mu_im, eps)

        return -tf.math.log(abs_mu_sq)
    
    
    def phi_off_circular_mean(self) -> tf.Tensor:
        """
        Circular mean of phi_off under the current (mixture) posterior.

        Returns 0 if phi_off is not inferred.
        """
        dtype = self._dtype
        if not self.phys_model.cfg.infer_phi_off:
            return tf.zeros((self.bs,), dtype=dtype)

        idx_phi = 2 if self.phys_model.cfg.infer_mfg_bias else 1
        phi0 = self.particles0[:, :, idx_phi]
        phi1 = self.particles1[:, :, idx_phi]

        re0 = tf.reduce_sum(self.weights0 * tf.cos(phi0), axis=1)
        im0 = tf.reduce_sum(self.weights0 * tf.sin(phi0), axis=1)
        re1 = tf.reduce_sum(self.weights1 * tf.cos(phi1), axis=1)
        im1 = tf.reduce_sum(self.weights1 * tf.sin(phi1), axis=1)

        q0 = tf.where(self.refining_mask, tf.ones((self.bs,), dtype=dtype), self.mode_weights[:, 0])
        q1 = tf.where(self.refining_mask, tf.zeros((self.bs,), dtype=dtype), self.mode_weights[:, 1])

        re1 = tf.where(q1 > tf.cast(0.0, dtype), re1, tf.zeros_like(re1))
        im1 = tf.where(q1 > tf.cast(0.0, dtype), im1, tf.zeros_like(im1))

        mu_re = q0 * re0 + q1 * re1
        mu_im = q0 * im0 + q1 * im1
        return tf.atan2(mu_im, mu_re)

    def apply_measurement(
        self,
        outcomes: tf.Tensor,
        controls: tf.Tensor,
        meas_step: tf.Tensor,
        continue_mask: tf.Tensor,
        rangen: tf.random.Generator,
    ) -> None:
        continue_mask = tf.cast(continue_mask, tf.bool)
        self._clear_loss_snapshots()

        refine_rows = tf.logical_and(continue_mask, self.refining_mask)
        if bool(tf.reduce_any(refine_rows).numpy()):
            self.weights0, _ = self._masked_bayes_update(
                self.weights0, self.particles0, outcomes, controls, meas_step, refine_rows
            )
            self.weights0, self.particles0 = self._differentiable_resample_masked(
                self.weights0, self.particles0, self.g_lo_vec, self.g_hi_vec, refine_rows, rangen
            )
            mean0, _ = self._weighted_mean_var(self.weights0, self.particles0)
            self.mode_weights = tf.where(
                refine_rows[:, None],
                tf.concat(
                    [
                        tf.ones((self.bs, 1), dtype=self._dtype),
                        tf.zeros((self.bs, 1), dtype=self._dtype),
                    ],
                    axis=1,
                ),
                self.mode_weights,
            )
            self._loss_refine_mask = refine_rows
            self._loss_refine_mean_snapshot = mean0

        dis_rows = tf.logical_and(continue_mask, tf.logical_not(self.refining_mask))
        if bool(tf.reduce_any(dis_rows).numpy()):
            if self._last_true_g is None:
                raise RuntimeError("set_true_values_for_loss must be called before apply_measurement")

            current_mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)
            width = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, self._dtype))
            true_left = tf.cast(self._last_true_g[:, 0] <= current_mid, self._dtype)

            new_w0, Z0 = self._masked_bayes_update(
                self.weights0, self.particles0, outcomes, controls, meas_step, dis_rows
            )
            new_w1, Z1 = self._masked_bayes_update(
                self.weights1, self.particles1, outcomes, controls, meas_step, dis_rows
            )
            q0 = self.mode_weights[:, 0]
            q1 = self.mode_weights[:, 1]
            q0_u = q0 * Z0
            q1_u = q1 * Z1
            Zt = tf.maximum(q0_u + q1_u, tf.cast(1e-300, self._dtype))
            new_q0 = q0_u / Zt
            new_q1 = q1_u / Zt

            self.weights0, self.weights1 = new_w0, new_w1
            self.mode_weights = tf.where(
                dis_rows[:, None],
                tf.stack([new_q0, new_q1], axis=1),
                self.mode_weights,
            )

            left_lo = self.g_lo_vec
            left_hi = 0.5 * (self.g_lo_vec + self.g_hi_vec)
            right_lo = left_hi
            right_hi = self.g_hi_vec

            self.weights0, self.particles0 = self._differentiable_resample_masked(
                self.weights0, self.particles0, left_lo, left_hi, dis_rows, rangen
            )
            self.weights1, self.particles1 = self._differentiable_resample_masked(
                self.weights1, self.particles1, right_lo, right_hi, dis_rows, rangen
            )

            mean0, _ = self._weighted_mean_var(self.weights0, self.particles0)
            mean1, _ = self._weighted_mean_var(self.weights1, self.particles1)
            mix_mean = new_q0 * mean0 + new_q1 * mean1

            # local_vh = self._local_holevo_from_pair(
            #     self.weights0, self.particles0,
            #     self.weights1, self.particles1,
            #     new_q0, new_q1,
            #     current_mid, width,
            # )
            local_holevo_loss = self.current_local_holevo_loss()

            target_kg = 2.0 * tf.cast(pi, self._dtype) / width
            actual_kg = self.phys_model.k_g(controls[:, 0], controls[:, 1])
            gain_penalty = tf.square(
                tf.math.log(tf.maximum(actual_kg, tf.cast(1e-8, self._dtype)))
                - tf.math.log(tf.maximum(target_kg, tf.cast(1e-8, self._dtype)))
            )

            self._loss_dis_mask = dis_rows
            self._loss_true_left = true_left
            self._loss_q0_snapshot = tf.where(dis_rows, new_q0, self._loss_q0_snapshot)
            self._loss_mean_snapshot = tf.where(dis_rows, mix_mean, self._loss_mean_snapshot)
            self._loss_width_snapshot = tf.where(dis_rows, width, self._loss_width_snapshot)
            # self._loss_local_holevo_snapshot = tf.where(dis_rows, local_vh, self._loss_local_holevo_snapshot)
            self._loss_local_holevo_snapshot = tf.where(dis_rows, local_holevo_loss, self._loss_local_holevo_snapshot)
            self._loss_gain_penalty_snapshot = tf.where(dis_rows, gain_penalty, self._loss_gain_penalty_snapshot)

            self.disambig_step_vec = tf.where(
                dis_rows, self.disambig_step_vec + 1, self.disambig_step_vec
            )
            ready = tf.logical_and(dis_rows, self.disambig_step_vec >= self.cfg.n_disambig_per_level)
            self._advance_ready_rows(ready, rangen)

    def hierarchical_loss_components(self, true_values: tf.Tensor):
        g_true = tf.cast(true_values[:, 0, 0], self._dtype)

        refine_mse = tf.square(self._loss_refine_mean_snapshot - g_true)

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

        width = tf.maximum(self._loss_width_snapshot, tf.cast(1e-8, self._dtype))
        mix_mse = tf.square((self._loss_mean_snapshot - g_true) / width)
        # local_holevo = tf.math.log1p(self._loss_local_holevo_snapshot)
        local_holevo = self._loss_local_holevo_snapshot
        gain_penalty = self._loss_gain_penalty_snapshot

        dis_total = (
            tf.cast(self.cfg.hierarchy_bce_weight, self._dtype) * bce
            + tf.cast(self.cfg.local_holevo_weight, self._dtype) * local_holevo
            + tf.cast(self.cfg.hierarchy_local_mse_weight, self._dtype) * mix_mse
            + tf.cast(self.cfg.gain_penalty_weight, self._dtype) * gain_penalty
        )

        total = tf.zeros((self.bs,), dtype=self._dtype)
        total = tf.where(self._loss_refine_mask, refine_mse, total)
        total = tf.where(self._loss_dis_mask, dis_total, total)

        return {
            "total": tf.expand_dims(total, axis=1),
            "bce": tf.expand_dims(
                tf.where(self._loss_dis_mask, bce, tf.zeros_like(bce)), axis=1
            ),
            "mix_mse": tf.expand_dims(
                tf.where(self._loss_dis_mask, mix_mse, tf.zeros_like(mix_mse)), axis=1
            ),
            "local_holevo": tf.expand_dims(
                tf.where(self._loss_dis_mask, local_holevo, tf.zeros_like(local_holevo)), axis=1
            ),
            "gain_penalty": tf.expand_dims(
                tf.where(self._loss_dis_mask, gain_penalty, tf.zeros_like(gain_penalty)), axis=1
            ),
            "refine_mse": tf.expand_dims(
                tf.where(self._loss_refine_mask, refine_mse, tf.zeros_like(refine_mse)), axis=1
            ),
            "true_left": tf.expand_dims(true_left, axis=1),
        }


class GravityLocalHolevoHierarchicalSimulation(StatelessSimulation):
    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: LocalHolevoHierarchicalPFBank,
        controller,
        simpars: SimulationParameters,
        bank_cfg: LocalHolevoHierarchicalPFConfig,
    ) -> None:
        input_size = 9
        input_name = [
            "mu_g_norm", "log_sigma_norm", "q0", "level_norm", "step_norm",
            "res_norm", "target_kg_norm", "vh_local_norm", "entropy_norm",
        ]
        super().__init__(
            particle_filter=bank.pf,
            phys_model=phys_model,
            control_strategy=controller,
            input_size=input_size,
            input_name=input_name,
            simpars=simpars,
        )
        self.bank = bank
        self.bank_cfg = bank_cfg
        self._dtype = tf.float32 if simpars.prec == "float32" else tf.float64

    def _branch_entropy(self, q0: tf.Tensor) -> tf.Tensor:
        eps = tf.cast(1e-8, self._dtype)
        q0 = tf.clip_by_value(q0, eps, 1.0 - eps)
        q1 = 1.0 - q0
        return -(q0 * tf.math.log(q0) + q1 * tf.math.log(q1))

    def _blend_angles(self, a: tf.Tensor, b: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        s = tf.clip_by_value(s, 0.0, 1.0)
        sin_mix = (1.0 - s) * tf.sin(a) + s * tf.sin(b)
        cos_mix = (1.0 - s) * tf.cos(a) + s * tf.cos(b)
        return tf.atan2(sin_mix, cos_mix)

    def generate_input(
        self,
        weights: tf.Tensor,
        particles: tf.Tensor,
        meas_step: tf.Tensor,
        used_resources: tf.Tensor,
        rangen,
    ) -> tf.Tensor:
        del weights, particles, meas_step, rangen
        bank = self.bank
        g_mean, g_var = bank.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-20, self._dtype)))

        width = tf.maximum(bank.g_hi_vec - bank.g_lo_vec, tf.cast(1e-8, self._dtype))
        mu_norm = 2.0 * (g_mean - bank.g_lo_vec) / width - 1.0
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)

        log_sigma = (
            -2.0 / 10.0
            * tf.math.log(tf.maximum(g_std, tf.cast(1e-8, self._dtype)))
            / tf.math.log(tf.cast(10.0, self._dtype))
            - 1.0
        )
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

        # vh_local = bank.current_local_holevo()
        # vh_local_norm = tf.math.log1p(vh_local) / tf.math.log(
        #     tf.cast(1.0 + bank.cfg.local_holevo_clip, self._dtype)
        # )
        # vh_local_norm = tf.clip_by_value(vh_local_norm, 0.0, 1.0)
        holevo_log = bank.current_local_holevo_loss()
        # Normalize by the maximum possible value given the epsilon floor.
        holevo_max = -tf.math.log(tf.cast(1e-20, self._dtype))
        vh_local_norm = tf.clip_by_value(holevo_log / tf.maximum(holevo_max, tf.cast(1e-8, self._dtype)), 0.0, 1.0)

        entropy_norm = self._branch_entropy(q0) / tf.math.log(tf.cast(2.0, self._dtype))
        entropy_norm = tf.clip_by_value(entropy_norm, 0.0, 1.0)

        return tf.stack(
            [
                mu_norm, log_sigma, q0, level_norm, step_norm, res_norm,
                target_kg_norm, vh_local_norm, entropy_norm,
            ],
            axis=1,
        )

    def _residual_outputs_to_controls(self, raw_actions: tf.Tensor) -> tf.Tensor:
        bank = self.bank
        cfg = self.phys_model.cfg
        dtype = self._dtype

        u_k = tf.clip_by_value(raw_actions[:, 0], -1.0, 1.0)
        u_b = tf.clip_by_value(raw_actions[:, 1], -1.0, 1.0)
        u_phi = tf.clip_by_value(raw_actions[:, 2], -1.0, 1.0)

        # full-range refinement controls
        T_min = tf.cast(cfg.T_range_s[0], dtype)
        T_max = tf.cast(cfg.T_range_s[1], dtype)
        Bp_min = tf.cast(cfg.Bp_range_kTm[0], dtype)
        Bp_max = tf.cast(cfg.Bp_range_kTm[1], dtype)

        T_mid = 0.5 * (T_min + T_max)
        T_half = 0.5 * (T_max - T_min)
        Bp_mid = 0.5 * (Bp_min + Bp_max)
        Bp_half = 0.5 * (Bp_max - Bp_min)

        T_ref = T_mid + T_half * u_k
        Bp_ref = Bp_mid + Bp_half * u_b
        phi_ref = wrap_to_pi_tf(tf.cast(pi, dtype) * (u_phi + 1.0))

        # disambiguation gain scaffold
        width = tf.maximum(bank.g_hi_vec - bank.g_lo_vec, tf.cast(1e-8, dtype))
        kg_target = 2.0 * tf.cast(pi, dtype) / width
        gain_ratio = tf.exp(tf.math.log(tf.cast(bank.cfg.gain_ratio_limit, dtype)) * u_k)
        kg_des = kg_target * gain_ratio
        kg_des = tf.clip_by_value(
            kg_des,
            self.phys_model.min_gain(dtype),
            self.phys_model.max_gain(dtype),
        )

        bp_ratio = tf.exp(
            0.5 * (u_b + 1.0) * tf.math.log(tf.cast(bank.cfg.bp_dis_max_ratio, dtype))
        )
        Bp_dis = tf.clip_by_value(Bp_min * bp_ratio, Bp_min, Bp_max)

        ge = tf.cast(cfg.gamma_e_rad_s_T, dtype)
        w = tf.cast(cfg.omega_rad_s, dtype)
        kT_to_T = tf.cast(cfg.kT_to_T, dtype)
        Bp_T = Bp_dis * kT_to_T

        c1 = (2.0 * ge / w) * Bp_T
        c0 = (8.0 * tf.cast(pi, dtype) * ge / (w ** 3)) * Bp_T
        T_sq = tf.maximum((kg_des - c0) / tf.maximum(c1, tf.cast(1e-12, dtype)), 0.0)
        T_dis = tf.sqrt(T_sq)
        T_dis = tf.clip_by_value(T_dis, T_min, T_max)

        q0 = tf.where(bank.refining_mask, tf.ones((self.bs,), dtype=dtype), bank.mode_weights[:, 0])
        entropy = self._branch_entropy(q0)
        s_quad = 1.0 - entropy / tf.math.log(tf.cast(2.0, dtype))
        s_quad = tf.clip_by_value(s_quad, 0.0, 1.0)

        g_mean, _ = bank.marginal_mean_and_var()
        mid = 0.5 * (bank.g_lo_vec + bank.g_hi_vec)
        left_center = mid - 0.25 * width

        # posterior circular mean estimate of phi_off
        phi_hat = bank.phi_off_circular_mean()

       # subtract phi_hat because likelihood uses phi_total = phi_off + mw_phase
        phi_cls = wrap_to_pi_tf(-kg_des * left_center - phi_hat)
        phi_quad = wrap_to_pi_tf(-kg_des * g_mean - phi_hat - 0.5 * tf.cast(pi, dtype))
        phi_base = self._blend_angles(phi_cls, phi_quad, s_quad)

        phi_dis = wrap_to_pi_tf(
            phi_base + tf.cast(bank.cfg.phase_residual_max_rad, dtype) * u_phi
        )

        T = tf.where(bank.refining_mask, T_ref, T_dis)
        Bp = tf.where(bank.refining_mask, Bp_ref, Bp_dis)
        phi = tf.where(bank.refining_mask, phi_ref, phi_dis)
        return tf.stack([T, Bp, phi], axis=1)

    def loss_function(
        self,
        weights: tf.Tensor,
        particles: tf.Tensor,
        true_values: tf.Tensor,
        used_resources: tf.Tensor,
        meas_step: tf.Tensor,
    ) -> tf.Tensor:
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
            hist_inputs: List[tf.Tensor] = []
            hist_controls: List[tf.Tensor] = []
            hist_resources: List[tf.Tensor] = []
            hist_precisions: List[tf.Tensor] = []

        for _ in range(pars.num_steps):
            num_finished = int(tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy())
            if num_finished >= pars.resources_fraction * self.bs:
                break

            weights = bank.weights0
            particles = bank.particles0
            self.pf = bank.pf

            input_strategy = self.generate_input(weights, particles, tf.cast(meas_step, prec), used_resources, rangen)
            cond_input = tf.stop_gradient(input_strategy) if pars.stop_gradient_input else input_strategy

            raw_actions = self.control_strategy(cond_input)
            controls = self._residual_outputs_to_controls(raw_actions)

            new_used_resources = self.phys_model.wrapper_count_resources(
                used_resources, outcomes, controls, true_values, true_state, meas_step
            )
            continue_flag = tf.math.less_equal(
                new_used_resources,
                tf.cast(pars.max_resources, prec) * tf.ones((self.bs, 1), dtype=prec),
            )
            used_resources = tf.where(continue_flag, new_used_resources, used_resources)

            outcomes_raw, log_prob, post_true_state = self.phys_model.wrapper_perform_measurement(
                tf.expand_dims(controls, axis=1),
                true_values,
                true_state,
                tf.expand_dims(meas_step, axis=1),
                rangen,
            )
            outcomes = outcomes_raw[:, 0, :]

            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(continue_flag, sum_log_prob + log_prob, sum_log_prob)

            bank.apply_measurement(outcomes, controls, meas_step, continue_flag[:, 0], rangen)
            loss_parts = bank.hierarchical_loss_components(true_values)
            loss_vals = loss_parts["total"]

            true_state = post_true_state
            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            if debug:
                snapshots = bank.debug_snapshot(
                    phys_model=self.phys_model,
                    true_values=true_values,
                    controls=controls,
                    used_resources=used_resources,
                    max_examples=min(3, self.bs),
                )
                current_local_vh = bank.current_local_holevo()
                for s in snapshots:
                    b = s["batch_idx"]
                    s["loss_total"] = float(loss_parts["total"][b, 0].numpy())
                    s["loss_bce"] = float(loss_parts["bce"][b, 0].numpy())
                    s["loss_mix_mse"] = float(loss_parts["mix_mse"][b, 0].numpy())
                    s["loss_local_holevo"] = float(loss_parts["local_holevo"][b, 0].numpy())
                    s["loss_gain_penalty"] = float(loss_parts["gain_penalty"][b, 0].numpy())
                    s["loss_refine_mse"] = float(loss_parts["refine_mse"][b, 0].numpy())
                    s["local_holevo_current"] = float(current_local_vh[b].numpy())
                    s["global_step"] = int(meas_step[b, 0].numpy())
                    debug_records.append(s)

            if pars.cumulative_loss and not deploy:
                active = tf.cast(continue_flag, prec)
                n_active = tf.maximum(tf.reduce_sum(active), tf.cast(1.0, prec))
                if pars.loss_logl_outcomes:
                    baseline = (
                        tf.reduce_sum(tf.where(continue_flag, loss_vals, tf.zeros_like(loss_vals))) / n_active
                        if pars.baseline else tf.cast(0.0, prec)
                    )
                    diff_vals = loss_vals + (
                        tf.stop_gradient(loss_vals) - tf.stop_gradient(baseline)
                    ) * sum_log_prob
                else:
                    diff_vals = loss_vals
                loss_diff_accum = loss_diff_accum + tf.reduce_sum(
                    tf.where(continue_flag, diff_vals, tf.zeros_like(diff_vals))
                ) / n_active
                loss_accum = loss_accum + tf.reduce_sum(
                    tf.where(continue_flag, loss_vals, tf.zeros_like(loss_vals))
                ) / n_active

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls.append(controls)
                hist_resources.append(used_resources)
                hist_precisions.append(loss_vals)

        if not deploy:
            if pars.cumulative_loss:
                denom = tf.cast(max(step_count, 1), prec)
                return loss_diff_accum / denom, loss_accum / denom
            loss_parts = bank.hierarchical_loss_components(true_values)
            loss_vals = loss_parts["total"]
            loss_mean = tf.reduce_mean(loss_vals)
            if pars.loss_logl_outcomes:
                baseline = loss_mean if pars.baseline else tf.cast(0.0, prec)
                loss_diff = tf.reduce_mean(
                    loss_vals + (tf.stop_gradient(loss_vals) - tf.stop_gradient(baseline)) * sum_log_prob
                )
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


def build_residual_controller(
    phys_model: GravityStatelessPhysicalModel,
    input_size: int = 9,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
) -> tf.keras.Model:
    dtype = tf.float32 if phys_model.cfg.prec == "float32" else tf.float64
    inputs = tf.keras.Input(shape=(input_size,), dtype=dtype)
    x = inputs
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="tanh", dtype=dtype)(x)
    outputs = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="local_holevo_hierarchical_controller")


def build_local_holevo_hierarchical_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    bank_cfg: LocalHolevoHierarchicalPFConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
):
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = LocalHolevoHierarchicalPFBank(phys_model=phys_model, cfg=bank_cfg)
    controller = build_residual_controller(phys_model, input_size=9)
    dummy = tf.zeros((batchsize, 9), dtype=tf.float32 if cfg.prec == "float32" else tf.float64)
    _ = controller(dummy)
    bank.reset(rangen)
    simulation = GravityLocalHolevoHierarchicalSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
    )
    return simulation, bank, controller
