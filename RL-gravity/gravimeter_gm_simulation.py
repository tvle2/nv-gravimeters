# gravimeter_gm_simulation.py
"""StatelessSimulation wrapper for the Gaussian-mixture gravimeter.

Replaces the previous `GravityMultiPFSimulation`. Key differences:

1. Uses `GaussianMixtureBank` instead of a fixed-K bank of PFs.
2. Loss = posterior squared error of the **mixture mean** estimator
   (matches deployment, optimal under squared-error loss):

       ℓ(ĝ, g_true) = (ĝ - g_true)² / Δg²,    ĝ = Σ_k q_k μ_k

   This is the only loss function used; we no longer try to combine
   posterior variance with bias on the assumption that the estimator
   equals the truth.

3. REINFORCE gradient construction follows Belliardo App. D.3 exactly:

       L̃(λ) = ℓ + sg[ℓ - B] · log p(y | x, θ)

   with B = batch-mean baseline. The REINFORCE surrogate is built
   directly in the custom execute() loop using cumulative MSE
   (Belliardo Eq. 106) with batch-mean baseline (Eq. 96).
   log_loss=False; the framework's `_compute_scalar_loss` is NOT used.

4. Cumulative loss is the simple time-average of ℓ_t (Belliardo Eq.
   106). The MSE is normalised by Δg² for numerical stability.

5. The controller input is leaner: per-mode (μ, σ, q) for the top-K
   modes, plus 5 globals (within-mode log-σ, marginal log-σ, step,
   resources, max-q).
   No more "previous estimate" feature (it leaked over-confident
   information).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model_complete import (
    GravimeterConfig, GravityStatelessPhysicalModel,
)
from gravimeter_gm_bank import GaussianMixtureBank, GaussianMixtureBankConfig


def _load_local(module_name: str):
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
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.stateless_simulation import StatelessSimulation
    from qsensoropt.simulation_parameters import SimulationParameters
except Exception:
    ParticleFilter = _load_local("particle_filter").ParticleFilter
    StatelessSimulation = _load_local("stateless_simulation").StatelessSimulation
    SimulationParameters = _load_local("simulation_parameters").SimulationParameters

# ===========================================================================
# GM controller input layout
# ===========================================================================


GM_GLOBAL_INPUT_NAMES = [
    "within_log_std_norm",
    "mix_log_std_norm",
    "res_norm",
    "max_q",
    "q_gap",
    "phase_cos",   
    "phase_sin", 
]


def gm_input_size(top_k_modes: int) -> int:
    return 3 * int(top_k_modes) + len(GM_GLOBAL_INPUT_NAMES)


def gm_input_names(top_k_modes: int) -> list[str]:
    names: list[str] = []
    for i in range(int(top_k_modes)):
        names += [f"top{i}_mu", f"top{i}_logsigma", f"top{i}_q"]
    names += GM_GLOBAL_INPUT_NAMES
    return names
# ===========================================================================
# Control schedule layer
# ===========================================================================
#
# Maps the controller's raw tanh outputs (u_kg, u_B, u_phi) plus
# posterior-width and step-normalised features to physical controls
# (T_s, B'_kTm, mw_phase_rad).
#
# Key change from the old version:
#   * `min_gain_fringe_fraction` is allowed to be < 0.25 so that early
#     measurements can be sub-fringe (mode-disambiguation regime).
#   * Posterior-safe gain cap k_g ≤ 2π / (mult · σ_post) is applied
#     SMOOTHLY (with min/max) and only as an upper bound — the lower
#     bound is a fixed fraction of one fringe across the prior, ALWAYS
#     achievable.
#   * No schedule_floor: the controller is free to pick coarse or fine
#     gains at any step. We let the network learn the schedule, not
#     hard-code it.
# ===========================================================================

class ControlScheduleLayer(tf.keras.layers.Layer):
    """Decodes (u_kg, u_B, u_phi, mix_log_std_norm) -> (T_s, B', φ_mw)."""

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        K: int,
        posterior_gain_width_multiplier: float,
        min_gain_fringe_fraction: float,
        log_sigma_frac_bounds: Tuple[float, float],
        prec: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._cfg = phys_model.cfg
        self._dtype = tf.float32 if prec == "float32" else tf.float64

        cfg = self._cfg
        self._T_min = tf.constant(float(cfg.T_range_s[0]), dtype=self._dtype)
        self._T_max = tf.constant(float(cfg.T_range_s[1]), dtype=self._dtype)
        self._Bp_min = tf.constant(float(cfg.Bp_range_kTm[0]), dtype=self._dtype)
        self._Bp_max = tf.constant(float(cfg.Bp_range_kTm[1]), dtype=self._dtype)
        self._gamma = tf.constant(cfg.gamma_e_rad_s_T, dtype=self._dtype)
        self._omega = tf.constant(cfg.omega_rad_s, dtype=self._dtype)
        self._kT = tf.constant(cfg.kT_to_T, dtype=self._dtype)

        g_lo, g_hi = float(cfg.g_range[0]), float(cfg.g_range[1])
        self._g_range = tf.constant(max(g_hi - g_lo, 1e-30), dtype=self._dtype)

        # The available k_g range:
        #   k_g(T, B') = (2γ/ω) B' T² + (8π γ / ω³) B'
        # so at fixed B', k_g varies with T². And at fixed T, k_g ∝ B'.
        # We expose a feasible log-uniform k_g_target ∈ [k_g_min, k_g_max].

        ##########################
        # # Bring in hbar
        # hbar = tf.cast(phys_model.cfg.hbar_J_s, self._dtype)
        # const_term = 8.0 * pi * self._gamma * hbar / tf.pow(self._omega, 3)

        # k_g_min_raw = (
        #     2.0 * self._gamma / self._omega
        #     * (self._Bp_min * self._kT) * tf.square(self._T_min)
        #     + const_term * (self._Bp_min * self._kT)
        # )
        # k_g_max_raw = (
        #     2.0 * self._gamma / self._omega
        #     * (self._Bp_max * self._kT) * tf.square(self._T_max)
        #     + const_term * (self._Bp_max * self._kT)
        # )
        ##########################
        k_g_min_raw = (
            2.0 * self._gamma / self._omega
            * (self._Bp_min * self._kT) * tf.square(self._T_min)
        )
        k_g_max_raw = (
            2.0 * self._gamma / self._omega
            * (self._Bp_max * self._kT) * tf.square(self._T_max)
        )
        # Floor: at least `min_gain_fringe_fraction` fringes across prior.
        k_g_floor = tf.cast(
            float(min_gain_fringe_fraction) * 2.0 * pi / max(g_hi - g_lo, 1e-30),
            self._dtype,
        )
        self._k_g_min = tf.maximum(k_g_min_raw, k_g_floor)
        self._K_cast = tf.constant(float(K), dtype=self._dtype)
        # Physics-defined inter-mode discrimination optimum:
        # k_g_inter_opt = π * K / Δg
        # At this k_g, two adjacent modes give maximally distinguishable outcomes.
        self._k_g_inter_opt = (
            tf.cast(pi, self._dtype) * self._K_cast / self._g_range
        )
        self._log_k_g_inter_opt = (
            tf.math.log(self._k_g_inter_opt) / tf.cast(np.log(10.0), self._dtype)
        )
        # Hard prior-aliasing cap: at k_g > π/Δg the cosine readout has more than
        # half a fringe across the prior, so modes on different fringes are aliased
        # (same Z up to damp differences). Independent of K — the prior width is
        # what bounds the unambiguous range.
        k_g_alias_cap = tf.cast(
            pi / max(g_hi - g_lo, 1e-30),
            self._dtype,
        )
        self._k_g_alias_cap = k_g_alias_cap        
        self._k_g_hw_max    = k_g_max_raw            
        self._k_g_max       = tf.minimum(k_g_max_raw, k_g_alias_cap)   

        self._log_k_g_min = tf.math.log(self._k_g_min) / tf.cast(np.log(10.0), self._dtype)
        self._log_k_g_max = tf.math.log(self._k_g_max) / tf.cast(np.log(10.0), self._dtype)

        self._posterior_gain_width_multiplier = tf.constant(
            float(posterior_gain_width_multiplier), dtype=self._dtype,
        )
        self._log_sigma_frac_lo = tf.constant(
            float(log_sigma_frac_bounds[0]), dtype=self._dtype,
        )
        self._log_sigma_frac_hi = tf.constant(
            float(log_sigma_frac_bounds[1]), dtype=self._dtype,
        )

    def call(self, x):
        u_kg = x[:, 0:1]
        u_B = x[:, 1:2]
        u_phi = x[:, 2:3]
        within_log_std_norm = x[:, 3:4]
        mix_log_std_norm = x[:, 4:5]
        max_q = x[:, 5:6]
        q_gap = x[:, 6:7]
        phase_cos = x[:, 7:8]   
        phase_sin = x[:, 8:9]   


        log_sigma_frac = self._log_sigma_frac_lo + (
            (mix_log_std_norm + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (self._log_sigma_frac_hi - self._log_sigma_frac_lo)
        sigma_frac = tf.pow(tf.cast(10.0, self._dtype), log_sigma_frac)
        sigma_g = sigma_frac * self._g_range
        sigma_g = tf.maximum(sigma_g, tf.cast(1e-30, self._dtype))

        # # V8 posterior-adaptive cap: k_g_cap = π / σ_marginal.
        # # sigma_g (computed above from mix_log_std_norm) already IS the marginal
        # # posterior std. At init: σ_marginal ≈ Δg/√12 ≈ 0.013 → cap ≈ 245.
        # # As bank localises: σ_marginal shrinks → cap auto-grows.
        # # No threshold, no chicken-and-egg, no unimodality logic.
        # #
        # # Factor of π (not 2π) ensures the cosine fringe spacing 2π/k_g covers
        # # at least 2·σ_marginal on each side of the bank's centre of mass — i.e.
        # # the readout is unambiguous across the bulk of the current posterior.
        k_post_cap = tf.cast(pi, self._dtype) / sigma_g

        # Bound by hardware max and the alias-safe floor.
        eff_k_g_max = tf.minimum(k_post_cap, self._k_g_hw_max)
        eff_k_g_max = tf.maximum(eff_k_g_max, self._k_g_min)
        log_eff_max = tf.math.log(eff_k_g_max) / tf.cast(np.log(10.0), self._dtype)
        log_k_g_min = self._log_k_g_min

        # Physics-centered mapping:
        # At u_kg = 0, k_g = min(πK/Δg, eff_k_g_max) — physics optimum subject to feasibility
        # At u_kg = -1, k_g = k_g_min (coarsest hardware-allowed)
        # At u_kg = +1, k_g = eff_k_g_max (finest σ-allowed)
        # Piecewise-linear in log space, asymmetric to respect both bounds.
        # log_center = tf.minimum(self._log_k_g_inter_opt, log_eff_max)
        # log_range_neg = log_center - log_k_g_min        # half-range below center
        # log_range_pos = log_eff_max - log_center        # half-range above center

        # log_kg_target = log_center + tf.where(
        #     u_kg < tf.cast(0.0, self._dtype),
        #     u_kg * log_range_neg,
        #     u_kg * log_range_pos,
        # )

        log_kg_target = log_k_g_min + (
            (u_kg + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (log_eff_max - log_k_g_min)
        k_g_target = tf.pow(tf.cast(10.0, self._dtype), log_kg_target)
        

        # Solve for (T, B') given k_g_target. We pick B' first in its
        # feasible interval determined by T ∈ [T_min, T_max], then back
        # out T = sqrt((k_g_target - b)/a).
        C_T_min = (
            tf.cast(2.0, self._dtype) * self._gamma / self._omega * tf.square(self._T_min)
        )
        C_T_max = (
            tf.cast(2.0, self._dtype) * self._gamma / self._omega * tf.square(self._T_max)
        )
        Bp_feas_low = k_g_target / tf.maximum(
            C_T_max * self._kT, tf.cast(1e-30, self._dtype),
        )
        Bp_feas_high = k_g_target / tf.maximum(
            C_T_min * self._kT, tf.cast(1e-30, self._dtype),
        )

        Bp_low = tf.maximum(self._Bp_min, Bp_feas_low)
        Bp_high = tf.maximum(tf.minimum(self._Bp_max, Bp_feas_high), Bp_low)
        Bp = Bp_low + (
            (u_B + tf.cast(1.0, self._dtype)) * tf.cast(0.5, self._dtype)
        ) * (Bp_high - Bp_low)
        ####################
        # Bp_T = Bp * self._kT
        # a = tf.cast(2.0, self._dtype) * self._gamma / self._omega * Bp_T
        # # Wang Eq. (3) second term with ℏ — ~10⁻³⁴ smaller than the leading T²-scaled term.
        # hbar = tf.cast(self._cfg.hbar_J_s, self._dtype)
        # b_const = (
        #     8.0 * tf.cast(pi, self._dtype) * self._gamma * hbar
        #     / tf.pow(self._omega, 3) * Bp_T
        # )
        # T_sq = (k_g_target - b_const) / tf.maximum(a, tf.cast(1e-30, self._dtype))
        ####################
        Bp_T = Bp * self._kT
        a = tf.cast(2.0, self._dtype) * self._gamma / self._omega * Bp_T
        b_const = tf.cast(0.0, self._dtype)  # second term dropped (negligible)
        T_sq = (k_g_target - b_const) / tf.maximum(a, tf.cast(1e-30, self._dtype))
        T_sq = tf.maximum(T_sq, tf.cast(0.0, self._dtype))
        T_s = tf.sqrt(T_sq)
        T_s = tf.clip_by_value(T_s, self._T_min, self._T_max)

        # MW phase: F3-V10 — combine analytic greedy-φ optimum with a learned
        # residual correction.
        #
        # The cosine readout's expected KL information gain at k_ref = π/σ_marg
        # is maximised when the MW phase aligns the readout's cosine zero
        # crossings with the bank's q-weighted "phase centroid":
        #     φ_opt = -atan2(phase_sin, phase_cos) + π
        # (the +π shift puts the zero crossing AT the centroid, so the two
        # outcomes split mode probability mass as evenly as possible).
        #
        # u_phi ∈ [-1, 1] is the network's residual correction, scaled by
        # `residual_amp` rad. residual_amp = π/4 gives the network ±45°
        # of authority to override the analytic optimum when it has learned
        # a better strategy (e.g. for trajectories near localisation where
        # finer phase tuning matters).
        residual_amp = tf.cast(pi / 4.0, self._dtype)
        phi_opt = -tf.atan2(phase_sin, phase_cos) + tf.cast(pi, self._dtype)
        mw_phase = phi_opt + residual_amp * u_phi
        # Wrap to [-π, π] for downstream consumers (the bank uses cos/sin
        # so this is cosmetic, but keeps the debug output readable).
        mw_phase = tf.math.floormod(
            mw_phase + tf.cast(pi, self._dtype),
            tf.cast(2.0 * pi, self._dtype),
        ) - tf.cast(pi, self._dtype)

        return tf.concat([T_s, Bp, mw_phase], axis=1)

    

# ===========================================================================
# Build the controller MLP
# ===========================================================================

def build_controller(
    input_size: int,
    phys_model: GravityStatelessPhysicalModel,
    bank_cfg: GaussianMixtureBankConfig,
    hidden_sizes: Tuple[int, ...] = (64, 64),
) -> tf.keras.Model:
    cfg = phys_model.cfg
    prec = cfg.prec
    dtype = tf.float32 if prec == "float32" else tf.float64

    inputs = tf.keras.Input(shape=(input_size,), dtype=dtype)
    x = inputs
    for h in hidden_sizes:
        x = tf.keras.layers.Dense(h, activation="tanh", dtype=dtype)(x)
    
    # raw = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    # separate Dense + scaled tanh
    raw_pre = tf.keras.layers.Dense(3, activation=None, dtype=dtype)(x)
    # Scale by 0.5 keeps tanh in the gradient-rich regime regardless of weight magnitudes.
    # This is structural: any Dense layer with Glorot init produces pre-activations ~N(0, 1-3),
    # which puts tanh in the saturation tail. Scaling by 0.5 keeps |tanh-input| ≤ ~1-2,
    # where gradient is 0.4-0.6.
    raw = tf.keras.layers.Lambda(
        lambda x: tf.tanh(x * tf.cast(0.5, x.dtype)),
        dtype=dtype,
        name="scaled_tanh",
    )(raw_pre)

    # New global layout: [within, mix, res, max_q, q_gap, phase_cos, phase_sin]
    within_log_std_norm = inputs[:, -7:-6]
    mix_log_std_norm    = inputs[:, -6:-5]
    max_q_feat          = inputs[:, -4:-3]
    q_gap_feat          = inputs[:, -3:-2]
    phase_cos_feat      = inputs[:, -2:-1]
    phase_sin_feat      = inputs[:, -1:]

    x_with_sigma = tf.keras.layers.Concatenate(axis=1, dtype=dtype)(
        [raw, within_log_std_norm, mix_log_std_norm, max_q_feat, q_gap_feat,
         phase_cos_feat, phase_sin_feat]
    )

    outputs = ControlScheduleLayer(
        phys_model=phys_model,
        K=bank_cfg.K,
        posterior_gain_width_multiplier=bank_cfg.posterior_gain_width_multiplier,
        min_gain_fringe_fraction=bank_cfg.min_gain_fringe_fraction,
        log_sigma_frac_bounds=bank_cfg.log_sigma_frac_bounds,
        prec=prec,
        dtype=dtype,
    )(x_with_sigma)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gm_controller")


# ===========================================================================
# Simulation
# ===========================================================================

class GravityGMSimulation(StatelessSimulation):
    """qsensoropt Simulation over the Gaussian-mixture posterior bank.

    The loss returned to the framework's `_compute_scalar_loss` is the
    squared error of the mixture-mean estimator, normalised by Δg²:

        ℓ = (ĝ - g_true)² / Δg²,    ĝ = Σ_k q_k μ_k

    The framework's `log_loss + cumulative_loss` and `loss_logl_outcomes
    + baseline` then construct the correct REINFORCE surrogate
    (Belliardo Eq. 121).
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: GaussianMixtureBank,
        controller: tf.keras.Model,
        simpars: SimulationParameters,
        bank_cfg: GaussianMixtureBankConfig,
        top_k_modes: int = 4,
    ) -> None:
        self.bank = bank
        self.bank_cfg = bank_cfg
        self.top_k = int(top_k_modes)
        self.g_lo = float(phys_model.cfg.g_range[0])
        self.g_hi = float(phys_model.cfg.g_range[1])
        self.prior_width = max(self.g_hi - self.g_lo, 1e-30)

        input_size = gm_input_size(self.top_k)
        input_name = gm_input_names(self.top_k)

        if len(input_name) != input_size:
            raise ValueError(
                f"GM input layout mismatch: len(input_name)={len(input_name)} "
                f"but input_size={input_size}. Names={input_name}"
            )

        # We still need a "dummy" particle filter for the parent class.
        # It's only used to expose `.np` etc.; we don't update it.
        dummy_pf = ParticleFilter(
            num_particles=8,
            phys_model=phys_model,
            resampling_allowed=False,
            prec=phys_model.prec,
        )

        super().__init__(
            particle_filter=dummy_pf,
            phys_model=phys_model,
            control_strategy=controller,
            input_size=input_size,
            input_name=input_name,
            simpars=simpars,
        )

        prec = phys_model.prec
        dtype = tf.float32 if prec == "float32" else tf.float64
        self._k_ref_coarsest = tf.constant(
            2.0 * pi / self.prior_width, dtype=dtype,
        )
        # F-V11 publication metric: track pure terminal-step MSE per training iter.
        # This is NOT used for gradients — it's logged separately for the training-curve plot.
        self._last_final_mse_norm = tf.Variable(0.0, trainable=False, dtype=dtype,
                                                name="last_final_mse_norm")
        self._last_final_max_q = tf.Variable(0.0, trainable=False, dtype=dtype,
                                            name="last_final_max_q")
        self._last_final_qclose = tf.Variable(0.0, trainable=False, dtype=dtype,
                                            name="last_final_qclose")
        
        # EMA-of-loss baseline state. Used to give REINFORCE an advantage
        # signal even when the batch is uniformly bad (e.g. policy collapsed).
        self._ema_decay = tf.constant(0.95, dtype=dtype)
        self._ema_loss  = tf.Variable(0.0, trainable=False, dtype=dtype)
        self._ema_init  = tf.Variable(False, trainable=False)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_function(
        self,
        weights: Tensor, particles: Tensor, true_values: Tensor,
        used_resources: Tensor, meas_step: Tensor,
    ) -> Tensor:
        """V10: NLL on the true mode's q-weight, plus a tiny MSE term for early
        training signal when q is still uniform.

        Rationale: the mixture-mean MSE rewards splitting q across modes
        that flank g_true. For K=8 with ~3 bits of measurement information,
        the right objective is to maximise q on the mode containing g_true.
        The NLL provides a direct gradient toward putting q on the truth.

        For the first few steps (when q is uniform), NLL is ~log(K) constant
        and gives no signal — so we add a small MSE term as a regulariser
        that gets the bank moving in roughly the right direction.
        """
        del weights, particles, used_resources, meas_step
        prec = self.simpars.prec
        g_true = true_values[:, 0, 0]                                   # (B,)
        Δμ = tf.cast(self.prior_width / float(self.bank.K), prec)
        true_mode_idx = tf.clip_by_value(
            tf.cast(tf.floor((g_true - tf.cast(self.g_lo, prec)) / Δμ), tf.int32),
            0, self.bank.K - 1,
        )                                                                # (B,)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, true_mode_idx], axis=1)

        q_true = tf.gather_nd(self.bank.q, gather_idx)                  # (B,)
        q_true = tf.clip_by_value(q_true, tf.cast(1e-12, prec), tf.cast(1.0, prec))
        # F-V11: focal NLL (γ=2) — down-weights already-won and already-lost cases,
        # focuses gradient on mid-confidence improvable cases. Reduces REINFORCE variance.
        gamma = tf.cast(2.0, prec)
        focal_weight = tf.pow(tf.cast(1.0, prec) - q_true, gamma)
        nll_term = -tf.math.log(q_true) * focal_weight       
        

        # Small MSE term for early-trajectory signal
        g_hat, _ = self.bank.marginal_mean_and_var()
        err = g_hat - g_true
        norm = tf.cast(self.prior_width ** 2, prec)
        norm_safe = tf.maximum(norm, tf.cast(1e-30, prec))
        mse_term = tf.square(err) / norm_safe

        # Equal weighting; the NLL term dominates as q diverges from uniform
        beta = tf.cast(0.5, prec)
        loss = beta * nll_term + (tf.cast(1.0, prec) - beta) * mse_term
        return tf.expand_dims(loss, axis=1)

    # ------------------------------------------------------------------
    # Controller input
    # ------------------------------------------------------------------

    def generate_input(
        self,
        weights: Tensor, particles: Tensor,
        meas_step: Tensor, used_resources: Tensor,
        rangen,
    ) -> Tensor:
        del weights, particles, rangen
        prec = self.simpars.prec
        bank = self.bank
        K = bank.K
        bs = self.bs
        simpars = self.simpars

        mu = bank.mu                                                  # (B, K)
        sigma = bank.sigma                                            # (B, K)
        q = bank.q                                                    # (B, K)

        # Top-k modes by q
        topk = tf.math.top_k(q, k=self.top_k, sorted=True)
        top_q = topk.values
        q_gap = top_q[:, 0] - top_q[:, 1]                      # (B,)
        top_idx = topk.indices
        batch_idx = tf.broadcast_to(
            tf.expand_dims(tf.range(bs, dtype=tf.int32), axis=1),
            (bs, self.top_k),
        )
        gather = tf.stack([batch_idx, top_idx], axis=2)
        top_mu = tf.gather_nd(mu, gather)
        top_sigma = tf.gather_nd(sigma, gather)

        # Normalise μ to [-1, 1] over the prior
        mu_norm = (
            2.0 * (top_mu - tf.cast(self.g_lo, prec)) / tf.cast(self.prior_width, prec)
            - 1.0
        )
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)

        # Normalise log10(sigma / prior_width) ∈ [-6, 0]
        log_sigma = tf.math.log(tf.maximum(top_sigma, tf.cast(1e-30, prec))) / tf.cast(
            np.log(10.0), prec
        )
        log_sigma_frac = log_sigma - tf.cast(np.log10(self.prior_width), prec)
        log_sigma_norm = 2.0 * (log_sigma_frac - (-6.0)) / (0.0 - (-6.0)) - 1.0
        log_sigma_norm = tf.clip_by_value(log_sigma_norm, -1.0, 1.0)

        per_mode = tf.reshape(
            tf.stack([mu_norm, log_sigma_norm, top_q], axis=2),
            (bs, 3 * self.top_k),
        )

        # Marginal posterior width (between-mode + within-mode spread).
        g_mean, g_var = bank.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-30, prec)))
        sigma_frac_mar = g_std / tf.cast(self.prior_width, prec)
        sigma_frac_mar = tf.maximum(sigma_frac_mar, tf.cast(1e-12, prec))
        log_sigma_frac_mar = tf.math.log(sigma_frac_mar) / tf.cast(np.log(10.0), prec)
        lo, hi = self.bank_cfg.log_sigma_frac_bounds
        log_sigma_frac_mar = tf.clip_by_value(
            log_sigma_frac_mar, tf.cast(lo, prec), tf.cast(hi, prec),
        )
        mix_log_std_norm = 2.0 * (log_sigma_frac_mar - lo) / max(hi - lo, 1e-12) - 1.0
        mix_log_std_norm = tf.clip_by_value(mix_log_std_norm, -1.0, 1.0)

        # Within-mode std (q-weighted sqrt(Σ q_k σ_k²)). This is what
        # drives the k_g schedule cap.
        within_std = bank.within_mode_std()                          # (B,)
        within_frac = within_std / tf.cast(self.prior_width, prec)
        within_frac = tf.maximum(within_frac, tf.cast(1e-12, prec))
        log_within_frac = tf.math.log(within_frac) / tf.cast(np.log(10.0), prec)
        log_within_frac = tf.clip_by_value(
            log_within_frac, tf.cast(lo, prec), tf.cast(hi, prec),
        )
        within_log_std_norm = 2.0 * (log_within_frac - lo) / max(hi - lo, 1e-12) - 1.0
        within_log_std_norm = tf.clip_by_value(within_log_std_norm, -1.0, 1.0)

        # Step / resources
        step_norm = 2.0 * tf.cast(meas_step[:, 0], prec) / float(simpars.num_steps) - 1.0
        res_norm = 2.0 * used_resources[:, 0] / tf.cast(simpars.max_resources, prec) - 1.0
        step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)
        res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        # Max q — strong indicator of mode resolution
        max_q = tf.reduce_max(q, axis=1)

        # ----- F3-V10: phase-distribution features at the reference k_ref ----
        # These let the MLP express adaptive φ as a (near-)linear function:
        #   φ_opt(state) ≈ -atan2(phase_sin, phase_cos) + π
        # which is the analytic info-maximising MW phase for a cosine readout.
        # Without these features, the MLP has to invent sin/cos in its hidden
        # layers from (μ_k, q_k), which REINFORCE cannot teach in 500 iters.
        #
        # k_ref is chosen as the *current* V8/V9 posterior-adaptive cap π/σ_marg,
        # so the trig features track the same scale the schedule decoder uses.
        # Detached from the gradient (these are state diagnostics, not part of
        # the controllable surface) — gradient still flows through controls
        # → bank update → q,μ in future steps as usual.
        g_std_safe = tf.maximum(g_std, tf.cast(1e-30, prec))                # (B,)
        k_ref = tf.cast(np.pi, prec) / g_std_safe                            # (B,)
        # Offset μ_k by g_lo so the argument is order ~1 regardless of g_range location.
        mu_offset = mu - tf.cast(self.g_lo, prec)                            # (B, K)
        k_mu_arg = tf.expand_dims(k_ref, axis=1) * mu_offset                 # (B, K)
        phase_cos = tf.reduce_sum(q * tf.cos(k_mu_arg), axis=1)              # (B,)
        phase_sin = tf.reduce_sum(q * tf.sin(k_mu_arg), axis=1)              # (B,)
        # ---------------------------------------------------------------------

        # Global features
        globals_ = tf.stack(
            [within_log_std_norm, mix_log_std_norm, res_norm, max_q, q_gap,
             phase_cos, phase_sin],
            axis=1,
        )

        return tf.concat([per_mode, globals_], axis=1)

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    #
    # Implementation follows the *same skeleton* as the previous one, but
    # the loss/REINFORCE surrogate is built using the framework's
    # `_compute_scalar_loss` (which honours `log_loss`, `baseline`,
    # `loss_logl_outcomes`, `cumulative_loss`).
    # ------------------------------------------------------------------

    def execute(
        self,
        rangen: tf.random.Generator,
        deploy: bool = False,
        debug: bool = False,
        debug_max_examples: int = 3,
    ):
        pars = self.simpars
        prec = pars.prec
        bank = self.bank

        debug_records: List[dict] = [] if debug else None

        bank.reset(rangen)
        true_values = self.phys_model.true_values(rangen)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype="bool")
        outcomes = tf.zeros(
            (self.bs, self.phys_model.outcomes_size), dtype=prec,
        )
        meas_step = tf.zeros((self.bs, 1), dtype="int32")
        
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        loss_diff_acc = tf.zeros((), dtype=prec)
        loss_acc = tf.zeros((), dtype=prec)
        step_count = 0

        # Dummy weights/particles to satisfy `loss_function`'s signature.
        dummy_w = tf.zeros((self.bs, 1), dtype=prec)
        dummy_p = tf.zeros((self.bs, 1, 1), dtype=prec)

        if deploy:
            hist_inputs: List[Tensor] = []
            hist_controls: List[Tensor] = []
            hist_resources: List[Tensor] = []
            hist_loss: List[Tensor] = []

        for _i in range(pars.num_steps):
            num_finished = int(
                tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy()
            )
            if num_finished >= pars.resources_fraction * self.bs:
                break

            input_strategy = self.generate_input(
                dummy_w, dummy_p,
                tf.cast(meas_step, prec), used_resources, rangen,
            )
            cond_input = (
                tf.stop_gradient(input_strategy)
                if pars.stop_gradient_input else input_strategy
            )
            controls = self.control_strategy(cond_input)

            # Update resources & stopping flag
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

            # Simulate the measurement
            outcomes_raw, log_prob, _post_true_state = (
                self.phys_model.wrapper_perform_measurement(
                    tf.expand_dims(controls, axis=1),
                    true_values,
                    true_state,
                    tf.expand_dims(meas_step, axis=1),
                    rangen,
                )
            )
            outcomes = outcomes_raw[:, 0, :]                              # (B, 1)

            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(
                    continue_flag, sum_log_prob + log_prob, sum_log_prob,
                )

            # Bayes update of the Gaussian-mixture posterior
            bank.apply_measurement(
                outcomes=outcomes,
                controls=controls,
                meas_step=meas_step,
                continue_flag=continue_flag,
                rangen=rangen,
            )

            # ============================================================
            # Per-step REINFORCE surrogate for cumulative MSE.
            #
            # The objective is
            #     L(λ) = (1/M_eff) Σ_t E[ℓ(ĝ_t, g)],     ℓ = (ĝ-g)²/Δg²
            #
            # Unbiased gradient (Belliardo Eq. 91 + baseline Eq. 96,
            # applied per-step then averaged over steps):
            #     ∂_λ L = (1/M_eff) Σ_t  E[ ∂_λ ℓ_t
            #              + (sg[ℓ_t] - B_t) · ∂_λ Σ_{s≤t} log p_s ]
            #
            # On the batch, we use B_t = (1/B) Σ_b ℓ_t^{(b)} (batch-mean
            # baseline). The surrogate at step t is therefore
            #     s_t = ℓ_t + sg[ℓ_t - B_t] · sum_log_prob_t
            # and the *cumulative* objective gradient is (1/M_eff) Σ_t s_t
            # — i.e. we average over steps at the very end.
            #
            # NOTE: We deliberately do NOT use the framework's
            # `log_loss=True` arithmetic together with cumulative loss.
            # Per Belliardo App. D.4 these are alternatives. This block
            # implements cumulative-MSE only.
            # ============================================================
            if not deploy:
                loss_values = self.loss_function(
                    dummy_w, dummy_p, true_values,
                    used_resources, meas_step,
                )                                                       # (B, 1)


                if pars.loss_logl_outcomes:
                    if pars.baseline:
                        baseline_t = tf.stop_gradient(tf.reduce_mean(loss_values))
                    else:
                        baseline_t = tf.zeros((), dtype=prec)
                    advantage = tf.stop_gradient(loss_values - baseline_t)
                    if pars.cumulative_loss:
                        surrogate = loss_values + advantage * log_prob
                    else:
                        surrogate = loss_values + advantage * sum_log_prob
                else:
                    surrogate = loss_values

                # Per-step reduction across the batch (Eq. 76: only
                # over still-active estimations; for an "end-batch"
                # cumulative-loss interpretation, also include the
                # frozen final estimators of finished batches).
                if pars.end_batch:
                    loss_diff_partial = tf.reduce_mean(surrogate)
                    loss_partial = tf.reduce_mean(loss_values)
                else:
                    cf = tf.cast(continue_flag, prec)
                    denom_active = tf.maximum(
                        tf.reduce_sum(cf), tf.cast(1.0, prec),
                    )
                    loss_diff_partial = tf.reduce_sum(surrogate * cf) / denom_active
                    loss_partial = tf.reduce_sum(loss_values * cf) / denom_active

                # Accumulate over steps. We divide by step_count at the
                # end of the loop to produce the cumulative-MSE average.
                if pars.cumulative_loss:
                    loss_diff_acc = loss_diff_acc + loss_diff_partial
                    loss_acc = loss_acc + loss_partial
                else:
                    # Final-step loss only.
                    loss_diff_acc = loss_diff_partial
                    loss_acc = loss_partial

            if debug:
                debug_records.extend(self._bank_snapshot(
                    true_values=true_values, controls=controls,
                    used_resources=used_resources, meas_step=meas_step,
                    loop_iter=_i, max_examples=debug_max_examples,
                ))

            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            # End-of-step bank-state detach for stop_gradient_pf
            if pars.stop_gradient_pf:
                bank.detach_state()

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls.append(controls)
                hist_resources.append(used_resources)
                # Diagnostic: the per-step loss itself
                lv = self.loss_function(
                    dummy_w, dummy_p, true_values,
                    used_resources, meas_step,
                )
                hist_loss.append(lv)
       
        if not deploy:
            # Detach bank state for diagnostic computation
            bank_q_sg  = tf.stop_gradient(bank.q)
            bank_mu_sg = tf.stop_gradient(bank.mu)

            # Pure terminal MSE / Δg² (no NLL, no alpha blending)
            final_g_true = true_values[:, 0, 0]                              # (B,)
            final_g_hat = tf.reduce_sum(bank_q_sg * bank_mu_sg, axis=1)      # (B,)
            final_err = final_g_hat - final_g_true
            norm = tf.cast(self.prior_width ** 2, prec)
            norm_safe = tf.maximum(norm, tf.cast(1e-30, prec))
            final_mse_norm = tf.stop_gradient(
                tf.reduce_mean(tf.square(final_err) / norm_safe)
            )

            # Bank diagnostics
            final_max_q = tf.stop_gradient(
                tf.reduce_mean(tf.reduce_max(bank_q_sg, axis=1))
            )

            # q at TRUE mode (closest mode by distance — robust to revival)
            d = tf.abs(bank_mu_sg - tf.expand_dims(final_g_true, axis=1))    # (B, K)
            true_mode_idx_v = tf.argmin(d, axis=1, output_type=tf.int32)     # (B,)
            batch_idx_v = tf.range(self.bs, dtype=tf.int32)
            gi = tf.stack([batch_idx_v, true_mode_idx_v], axis=1)
            final_qclose = tf.stop_gradient(
                tf.reduce_mean(tf.gather_nd(bank_q_sg, gi))
            )

            # Assign to variables (trainable=False, no gradient through assign)
            self._last_final_mse_norm.assign(
                tf.cast(final_mse_norm, self._last_final_mse_norm.dtype))
            self._last_final_max_q.assign(
                tf.cast(final_max_q, self._last_final_max_q.dtype))
            self._last_final_qclose.assign(
                tf.cast(final_qclose, self._last_final_qclose.dtype))
                    
            if pars.cumulative_loss:
                denom = tf.cast(max(step_count, 1), prec)
                loss_diff_final = loss_diff_acc / denom
                loss_final = loss_acc / denom
            else:
                # Final-step only: loss_diff_acc already holds the last
                # step's surrogate — do NOT divide by step_count.
                loss_diff_final = loss_diff_acc
                loss_final = loss_acc
            if debug:
                return loss_diff_final, loss_final, debug_records
            return loss_diff_final, loss_final

        # Deploy payload
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
            empty_p = tf.stack(hist_loss, axis=0)
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _bank_snapshot(
        self,
        true_values, controls, used_resources, meas_step,
        loop_iter: int,
        max_examples: int = 3,
    ):
        bank = self.bank
        K = bank.K
        prec = self.simpars.prec

        g_mix, g_var = bank.marginal_mean_and_var()
        g_map, std_map = bank.map_mode_estimate()
        g_close, std_close, q_close = bank.closest_mode_estimate(true_values[:, 0, 0])

        # Per-step loss (mixture mean MSE)
        dummy_w = tf.zeros((self.bs, 1), dtype=prec)
        dummy_p = tf.zeros((self.bs, 1, 1), dtype=prec)
        loss_per_b = self.loss_function(
            dummy_w, dummy_p, true_values, used_resources, meas_step,
        )[:, 0]

        k_g = self.phys_model.k_g(controls[:, 0], controls[:, 1])
        vis = self.phys_model.known_visibility_factor(controls[:, 0], controls[:, 1])

        q_np = bank.q.numpy()
        mu_np = bank.mu.numpy()
        sigma_np = bank.sigma.numpy()

        n_show = min(int(max_examples), self.bs)
        records = []
        for b in range(n_show):
            order = np.argsort(-q_np[b])
            top_modes = []
            for k in order[: min(3, K)]:
                top_modes.append({
                    "mode": int(k),
                    "q": float(q_np[b, k]),
                    "mu": float(mu_np[b, k]),
                    "sigma": float(sigma_np[b, k]),
                })
            true_g_b = float(true_values[b, 0, 0].numpy())
            mode_width = (self.g_hi - self.g_lo) / float(K)
            true_mode = int(np.clip(
                np.floor((true_g_b - self.g_lo) / max(mode_width, 1e-30)),
                0, K - 1,
            ))
            true_mode_rank = int(np.where(order == true_mode)[0][0] + 1)
            records.append({
                "batch_idx": int(b),
                "loop_iter": int(loop_iter),
                "meas_step": int(meas_step[b, 0].numpy()),
                "used_resources": float(used_resources[b, 0].numpy()),
                "T_s": float(controls[b, 0].numpy()),
                "Bp_kTm": float(controls[b, 1].numpy()),
                "mw_phase_rad": float(controls[b, 2].numpy()),
                "true_g": float(true_values[b, 0, 0].numpy()),
                "g_mix": float(g_mix[b].numpy()),
                "g_var": float(g_var[b].numpy()),
                "g_map": float(g_map[b].numpy()),
                "g_close": float(g_close[b].numpy()),
                "q_close": float(q_close[b].numpy()),
                "loss": float(loss_per_b[b].numpy()),
                "k_g": float(k_g[b].numpy()),
                "vis": float(vis[b].numpy()),
                "K": int(K),
                "top_modes": top_modes,
                "true_mode": true_mode,
                "true_mode_rank": true_mode_rank,
                "max_q": float(np.max(q_np[b])),
                "q_true_mode": float(q_np[b, true_mode]),
            })
        return records


# ===========================================================================
# Factory
# ===========================================================================

def build_gravity_gm_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    bank_cfg: GaussianMixtureBankConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
    top_k_modes: int = 4,
    hidden_sizes: Tuple[int, ...] = (64, 64),
) -> Tuple[GravityGMSimulation, GaussianMixtureBank, tf.keras.Model]:
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = GaussianMixtureBank(phys_model=phys_model, cfg=bank_cfg)

    input_size = gm_input_size(top_k_modes)

    controller = build_controller(
        input_size, phys_model, bank_cfg, hidden_sizes=hidden_sizes,
    )

    dtype = tf.float32 if cfg.prec == "float32" else tf.float64
    _ = controller(tf.zeros((batchsize, input_size), dtype=dtype))

    bank.reset(rangen)

    sim = GravityGMSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
        top_k_modes=top_k_modes,
    )
    return sim, bank, controller