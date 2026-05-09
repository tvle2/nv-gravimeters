# gravimeter_multi_pf.py
"""Multi-PF Bank for the levitated-NV gravimeter, with multi-scale
log-Holevo loss and log-cumulative training, integrated into the
qsensoropt StatelessSimulation API.

Design (locked):
  * K = cfg.k_max sub-PFs, fixed for the entire episode.  Each sub-PF
    covers a 1/K slice of the prior, and they tile the prior together.
    The bank is therefore K independent qsensoropt ParticleFilters
    sharing one batch dimension; every batch element runs the same K
    modes (per-batch independence is preserved because all per-batch
    bank state has the leading dim B).
  * Per-step Bayesian update on the mode weights:
        q_k <- q_k * Z_k / sum_l q_l Z_l
    where Z_k is the marginal likelihood of mode k.
  * Within-mode update: standard qsensoropt ParticleFilter Bayes update
    + ESS-triggered full_resampling.  Scibior-Wood is OFF by default
    because the loss is not polynomial in the within-mode weights,
    so the correction would bias the gradient.
  * Loss: multi-scale log-Holevo,
        L = (1/(J+1)) sum_{j=0..J} log(1 + V_H(k_ref_j))
    with k_ref_j = 2^j * 2pi/(g_hi - g_lo).  Bounded growth, well-defined
    gradient even when the posterior is fully ambiguous (V_H -> infinity
    becomes log(1+V_H) -> finite log).
  * Episode loss is the LOG-CUMULATIVE form (Belliardo 2024 Eq. 109):
        L_log = (1/T) sum_t log E_b[L_t_b]
    averaged over the T steps run.  No reference precision eta needed.
  * REINFORCE surrogate adds (sg(L - baseline) * sum_log_prob) to the
    cumulative loss to propagate gradient through the sampled outcomes.

References
----------
Belliardo et al. (2024). Phys. Rev. A 109, 062609.
Berry & Wiseman (2000). Phys. Rev. Lett. 85, 5098 (multi-scale Holevo).
Wang et al. (2025). Phys. Rev. Lett. 135, 120803 (gravimeter model).
"""
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiPFBankConfig:
    """Configuration for the (fixed-K) Multi-PF Bank.

    Parameters
    ----------
    n_per_mode : int
        Number of particles in each sub-PF.  The total number of
        particles maintained is ``n_per_mode * k_max``.
    k_max : int
        Number of sub-PFs (modes), held fixed for the whole episode.
        Must be large enough to cover the worst-case fringe count at
        max controllable gain:  k_max >= ceil(k_g_max * (g_hi-g_lo)/2pi).
    n_scales : int, optional
        DEPRECATED — no longer used.  Loss scales are now picked
        adaptively per step inside the simulation.
    top_k_modes : int
        Number of mode summaries fed to the controller.  Must be <= k_max.
    resample_threshold : float
    resample_alpha : float
    resample_beta : float
        Hyperparameters of the within-mode ParticleFilter resampling.
    scibior_trick : bool
        Whether to enable the Scibior-Wood differentiable-resampling
        correction.  Default: False.  See module docstring for why.
    trim : bool
        Whether to trim particles to parameter bounds after resampling.
    """
    n_per_mode: int = 64
    k_max: int = 128
    n_scales: Optional[int] = None
    top_k_modes: int = 4

    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = False
    trim: bool = True


# ---------------------------------------------------------------------------
# Multi-PF Bank
# ---------------------------------------------------------------------------

class MultiPFBank:
    """Fixed-K bank of K=k_max independent qsensoropt particle filters.

    Each sub-PF k covers the disjoint g-interval
        [g_lo + k*Delta, g_lo + (k+1)*Delta]    where Delta = (g_hi-g_lo)/K
    Within each mode, particles are uniform over that interval at reset.
    Mode weights q_k are uniform 1/K at reset.

    All operations are vectorized over the batch dimension (B); per-batch
    elements share the same K but have independent (q_k, particles, weights).
    """

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

        # Bank state, populated by reset()
        self.weights_list: List[Tensor] = []     # K tensors of shape (B, N)
        self.particles_list: List[Tensor] = []   # K tensors of shape (B, N, d)
        self.state_list: List[Tensor] = []       # K tensors of shape (B, N, 0)
        self.mode_weights: Optional[Tensor] = None  # (B, K)
        self.pf_list: List[ParticleFilter] = []

        # Each mode owns one independent ParticleFilter.
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
    # Reset (start of every episode)
    # ------------------------------------------------------------------

    def reset(self, rangen: tf.random.Generator) -> None:
        """Initialize particles to uniformly tile the prior across K modes
        and set mode weights to 1/K."""
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
            # Affine-map the g-coord (axis=2, dim 0) into mode k's slice
            # so particles uniformly fill [g_lo + k*Delta, g_lo + (k+1)*Delta].
            #   g_new = (g_old - g_lo) * scale + (g_lo + k*Delta)
            # which expands to scale*g_old + (g_lo + k*Delta - g_lo*scale).
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

        # Mode weights start uniform 1/K.
        self.mode_weights = tf.fill((self.bs, K), tf.cast(1.0 / float(K), prec))

    # ------------------------------------------------------------------
    # Per-step Bayes update across all K modes
    # ------------------------------------------------------------------

    def apply_measurement(
        self,
        outcomes: Tensor,         # (B, outcomes_size)
        controls: Tensor,         # (B, controls_size)
        meas_step: Tensor,        # (B, 1)  int32
        continue_flag: Tensor,    # (B, 1)  bool — only update where True
        rangen: tf.random.Generator,
    ) -> None:
        """Apply one measurement to all K modes.

        Updates within-mode weights and particles, then the across-mode
        weights q_k.  Frozen batch elements (continue_flag=False) keep
        their previous bank state unchanged — implemented via tf.where
        on every output tensor of this method.
        """
        prec = self.prec
        K = self.K
        keep = tf.cast(continue_flag[:, 0], prec)         # (B,)  1 if running
        keep_w = tf.expand_dims(keep, axis=1)             # (B, 1)
        keep_p = tf.expand_dims(keep_w, axis=2)           # (B, 1, 1)

        Z_k_list: List[Tensor] = []
        new_weights_list: List[Tensor] = []

        for k in range(K):
            w_k = self.weights_list[k]                    # (B, N)
            p_k = self.particles_list[k]                  # (B, N, d)
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
            )                                             # (B, N_k)
            unnorm_w_k = w_k * prob_k
            Z_k = tf.reduce_sum(unnorm_w_k, axis=1)       # (B,)
            safe_Z_k = tf.maximum(Z_k, tf.cast(1e-30, prec))
            updated_w_k = unnorm_w_k / tf.expand_dims(safe_Z_k, axis=1)
            # Freeze finished batch elements at their previous state.
            new_w_k = tf.where(keep_w > 0.5, updated_w_k, w_k)
            Z_k_list.append(Z_k)
            new_weights_list.append(new_w_k)

        # Mode-weight Bayes update.
        Z_stack = tf.stack(Z_k_list, axis=1)              # (B, K)
        q_old = self.mode_weights                         # (B, K)
        new_q_unnorm = q_old * Z_stack                    # (B, K)
        Z_total = tf.reduce_sum(new_q_unnorm, axis=1, keepdims=True)
        safe_Z_total = tf.maximum(Z_total, tf.cast(1e-30, prec))
        new_q = new_q_unnorm / safe_Z_total
        self.mode_weights = tf.where(keep_w > 0.5, new_q, q_old)
        self.weights_list = new_weights_list

        # Within-mode ESS-triggered resampling.  full_resampling itself
        # checks ESS internally; we let it handle finished batch elements
        # via count_for_resampling.
        cont_for_resamp = continue_flag                   # (B, 1)  bool
        for k in range(K):
            new_w_k, new_p_k, _ = self.pf_list[k].full_resampling(
                self.weights_list[k],
                self.particles_list[k],
                count_for_resampling=cont_for_resamp,
                rangen=rangen,
            )
            self.weights_list[k] = new_w_k
            self.particles_list[k] = new_p_k

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mode_means_and_stds(self) -> Tuple[Tensor, Tensor]:
        """Return (means, stds) of shape (B, K) for every mode."""
        means: List[Tensor] = []
        stds: List[Tensor] = []
        prec = self.prec
        for k in range(self.K):
            mean_k = self.pf_list[k].compute_mean(
                self.weights_list[k], self.particles_list[k]
            )[:, 0]                                       # (B,)
            cov_k = self.pf_list[k].compute_covariance(
                self.weights_list[k], self.particles_list[k]
            )[:, 0, 0]
            std_k = tf.sqrt(tf.maximum(cov_k, tf.cast(0.0, prec)))
            means.append(mean_k)
            stds.append(std_k)
        return tf.stack(means, axis=1), tf.stack(stds, axis=1)

    def marginal_mean_and_var(self) -> Tuple[Tensor, Tensor]:
        """Mixture mean and variance of the full posterior."""
        prec = self.prec
        means, stds = self.mode_means_and_stds()          # (B, K), (B, K)
        q = self.mode_weights                             # (B, K)
        g_mean = tf.reduce_sum(q * means, axis=1)         # (B,)
        sec = tf.reduce_sum(q * (tf.square(stds) + tf.square(means)), axis=1)
        g_var = tf.maximum(sec - tf.square(g_mean), tf.cast(0.0, prec))
        return g_mean, g_var

    def map_mode_estimate(self) -> Tuple[Tensor, Tensor]:
        """Return (mean, std) of the mode with the largest q.  Used for
        point estimation when reporting MSE."""
        means, stds = self.mode_means_and_stds()          # (B, K)
        best_k = tf.argmax(self.mode_weights, axis=1, output_type=tf.int32)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)
        return tf.gather_nd(means, gather_idx), tf.gather_nd(stds, gather_idx)

    # ------------------------------------------------------------------
    # Holevo variance (single scale, diagnostic) and multi-scale loss
    # ------------------------------------------------------------------

    def _complex_moment(self, k_ref: Tensor) -> Tuple[Tensor, Tensor]:
        """Mixture circular moment mu(k_ref) = sum_k q_k * sum_j w_kj *
        exp(i k_ref g_kj).  Returns (Re mu, Im mu) of shape (B,).

        k_ref: shape (B,) or scalar."""
        prec = self.prec
        K = self.K
        # Promote k_ref to (B,)
        if k_ref.shape.rank == 0:
            k_ref_b = tf.fill((self.bs,), tf.cast(k_ref, prec))
        else:
            k_ref_b = tf.cast(k_ref, prec)

        re_acc = tf.zeros((self.bs,), dtype=prec)
        im_acc = tf.zeros((self.bs,), dtype=prec)
        for k in range(K):
            p_k = self.particles_list[k]                  # (B, N, d)
            w_k = self.weights_list[k]                    # (B, N)
            g_k = p_k[:, :, 0]                            # (B, N)
            phase = tf.expand_dims(k_ref_b, axis=1) * g_k # (B, N)
            re_k = tf.reduce_sum(w_k * tf.cos(phase), axis=1)
            im_k = tf.reduce_sum(w_k * tf.sin(phase), axis=1)
            q_k = self.mode_weights[:, k]                 # (B,)
            re_acc = re_acc + q_k * re_k
            im_acc = im_acc + q_k * im_k
        return re_acc, im_acc

    def holevo_variance(self, k_ref: Tensor) -> Tensor:
        """Diagnostic V_H = 1/|mu|^2 - 1.  NOT used in the loss (it can
        explode and is hard to clip without killing the gradient).
        Returns (B, 1).
        """
        prec = self.prec
        re, im = self._complex_moment(k_ref)              # (B,)
        abs_mu_sq = tf.square(re) + tf.square(im)
        # No clipping: callers should be aware this can be huge.
        v_h = 1.0 / tf.maximum(abs_mu_sq, tf.cast(1e-30, prec)) - 1.0
        return tf.expand_dims(v_h, axis=1)

    def log_holevo_at_scale(self, k_ref: Tensor) -> Tensor:
        """log(1 + V_H(k_ref)) = -log |mu(k_ref)|^2.  Bounded growth,
        well-defined gradient.  Returns (B,).  This is the loss we
        actually optimize at scale k_ref.
        """
        prec = self.prec
        re, im = self._complex_moment(k_ref)
        abs_mu_sq = tf.square(re) + tf.square(im)
        # log(1/|mu|^2) = -log|mu|^2.  Clamp |mu|^2 from below for stability.
        return -tf.math.log(tf.maximum(abs_mu_sq, tf.cast(1e-30, prec)))

    def multi_scale_log_holevo_loss(self, scales: Tensor) -> Tensor:
        """Average log-Holevo across a list of scales (1D tensor of length J+1).
        Returns (B,)."""
        prec = self.prec
        n_scales = int(scales.shape[0])
        acc = tf.zeros((self.bs,), dtype=prec)
        for j in range(n_scales):
            acc = acc + self.log_holevo_at_scale(scales[j])
        return acc / tf.cast(n_scales, prec)

    # ------------------------------------------------------------------
    # Adaptive k_ref (for the NN input only, never for the loss)
    # ------------------------------------------------------------------

    def adaptive_k_ref(self, k_g_max: float) -> Tensor:
        """Pick k_ref so the marginal posterior 6sigma matches one fringe.
        Used to feed a normalized 'how concentrated is the posterior'
        scalar to the controller, NOT for the loss.
        """
        prec = self.prec
        _, g_var = self.marginal_mean_and_var()
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-30, prec)))
        k_ref_adaptive = 2.0 * pi / tf.maximum(6.0 * g_std, tf.cast(1e-30, prec))
        k_ref_min_py = 2.0 * pi / max(self.g_hi - self.g_lo, 1e-30)
        k_ref_min = tf.cast(k_ref_min_py, prec)
        k_ref_max = tf.cast(max(k_g_max / 2.0, k_ref_min_py), prec)
        return tf.clip_by_value(k_ref_adaptive, k_ref_min, k_ref_max)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class GravityMultiPFSimulation(StatelessSimulation):
    """qsensoropt StatelessSimulation built on top of the Multi-PF bank.

    Implements:
      - generate_input(): builds a fixed-size feature vector for the NN.
      - loss_function(): multi-scale log-Holevo (averaged over scales).
      - execute(): bank-aware measurement loop with the log-cumulative
        REINFORCE surrogate.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: MultiPFBank,
        controller: tf.keras.Model,
        simpars: SimulationParameters,
        bank_cfg: MultiPFBankConfig,
    ) -> None:
        # Input size: per-mode features for top_k_modes plus a few globals.
        # Per-mode features: (mu_norm, log_std_norm, q_k).  All modes are
        # always active in the fixed-K design so we drop the 'active_k'
        # feature that was used in the variable-K version.
        top_k = bank_cfg.top_k_modes
        input_size = 3 * top_k + 4   # 3 per mode + (V_H_norm, H_q_norm, step_norm, res_norm)

        input_name = []
        for i in range(top_k):
            input_name += [f"mu_mode_{i}", f"log_std_mode_{i}", f"q_mode_{i}"]
        input_name += ["log1p_VH_norm", "H_q_norm", "step_norm", "res_norm"]

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
        dtype = tf.float32 if phys_model.prec == "float32" else tf.float64
        self._k_g_max: float = float(phys_model.max_gain(dtype).numpy())

        # Loss design (Berry & Wiseman multi-scale Holevo, adapted):
        # at every step the loss is computed at TWO scales:
        #   (a) the COARSEST scale k_ref = 2pi / (g_hi - g_lo) — one fringe
        #       spans the prior.  Penalizes failure to localize at all.
        #   (b) the CURRENT measurement scale k_ref = k_g(controls) — one
        #       fringe spans the controller's chosen aliasing distance.
        #       Penalizes the controller for picking gains where the bank
        #       still has unresolved fringes.
        # We do NOT include the coarsest..finest dyadic ladder up to
        # k_g_max, because the gradient through scales much finer than
        # the current localization explodes (those scales contribute
        # essentially noise but their gradients can be of order k_g_max
        # times the coarse-scale gradient, which is ~10^4 here).
        prec = phys_model.prec
        self._k_ref_coarsest: Tensor = tf.constant(
            2.0 * pi / max(self.g_hi - self.g_lo, 1e-30),
            dtype=tf.float32 if prec == "float32" else tf.float64,
        )
        # Weight of the current-scale term in the per-step loss; the
        # coarsest-scale term has weight 1.  alpha_meas = 0.5 makes the
        # two terms comparable in magnitude when both posteriors are
        # roughly delocalized.
        self._alpha_meas: float = 0.5

    # ------------------------------------------------------------------
    # Per-step loss
    # ------------------------------------------------------------------

    def _per_step_loss(self, controls: Optional[Tensor] = None) -> Tensor:
        """Per-batch-element loss at the current bank state.

        L_b = log(1 + V_H(k_coarse)) + alpha * log(1 + V_H(k_meas))
        where k_meas = k_g(controls) (or just the coarse term if controls
        is None — used at episode start before any measurement).

        Returns shape (B,).
        """
        bank = self.bank
        prec = self.simpars.prec
        coarse = bank.log_holevo_at_scale(self._k_ref_coarsest)
        if controls is None:
            return coarse
        # k_g of THIS step's controls.  We stop_gradient to avoid double-counting
        # the controls' gradient through the loss scale; the loss already
        # depends on controls through the bank Bayes update.
        k_meas = self.phys_model.k_g(controls[:, 0], controls[:, 1])
        k_meas = tf.stop_gradient(k_meas)
        meas = bank.log_holevo_at_scale(k_meas)
        alpha = tf.cast(self._alpha_meas, prec)
        return coarse + alpha * meas

    # ------------------------------------------------------------------
    # Controller input
    # ------------------------------------------------------------------


    def generate_input(self, weights, particles, meas_step, used_resources, rangen):
        del weights, particles, rangen  # passthrough only — bank holds state
        bank = self.bank
        prec = self.simpars.prec
        simpars = self.simpars
        TOP_K = self.bank_cfg.top_k_modes
        bs = self.bs

        # Per-mode features: take the TOP_K modes by current weight, NOT
        # by mode index.  This makes the input invariant to the (arbitrary)
        # mode ordering and helps the controller generalize.
        means_all, stds_all = bank.mode_means_and_stds()       # (B, K)
        q_all = bank.mode_weights                              # (B, K)

        # Sort by descending q, take top TOP_K.
        # tf.math.top_k(q, k=TOP_K) returns values and indices.
        topk = tf.math.top_k(q_all, k=TOP_K, sorted=True)
        top_q = topk.values                                    # (B, TOP_K)
        top_idx = topk.indices                                 # (B, TOP_K) int32
        batch_idx = tf.broadcast_to(
            tf.expand_dims(tf.range(bs, dtype=tf.int32), axis=1),
            (bs, TOP_K),
        )
        gather = tf.stack([batch_idx, top_idx], axis=2)        # (B, TOP_K, 2)
        top_mu = tf.gather_nd(means_all, gather)               # (B, TOP_K)
        top_std = tf.gather_nd(stds_all, gather)               # (B, TOP_K)

        # Normalize mu to [-1, 1]
        g_range = tf.cast(max(self.g_hi - self.g_lo, 1e-30), prec)
        mu_norm = 2.0 * (top_mu - tf.cast(self.g_lo, prec)) / g_range - 1.0
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)

        # log10(std), then linearly map to [-1, 1] over a 12-decade window.
        # std=1e-1 -> -1, std=1e-13 -> +1 in float64; for float32 we
        # compress because float32 reaches ~1e-7 minimum useful std.
        if prec == "float64":
            log_std_lo, log_std_hi = -13.0, -1.0
        else:
            log_std_lo, log_std_hi = -7.0, -1.0
        log_std = tf.math.log(tf.maximum(top_std, tf.cast(1e-30, prec))) / tf.cast(np.log(10.0), prec)
        log_std_norm = 2.0 * (log_std - log_std_lo) / (log_std_hi - log_std_lo) - 1.0
        log_std_norm = tf.clip_by_value(log_std_norm, -1.0, 1.0)

        # Stack interleaved per-mode features: [mu, log_std, q] x TOP_K.
        per_mode = tf.reshape(
            tf.stack([mu_norm, log_std_norm, top_q], axis=2),
            (bs, 3 * TOP_K),
        )

        # Global features.
        # log1p(V_H) at the COARSEST scale, normalized by log1p(K) (worst case).
        v_h_coarsest = bank.log_holevo_at_scale(self._k_ref_coarsest)     # (B,)
        # log_holevo_at_scale already returns log(1+V_H) up to the |mu|^2 floor.
        K = bank.K
        v_h_max_log = tf.cast(np.log(float(K) + 1.0) + 30.0 * np.log(10.0), prec)
        log1p_vh_norm = tf.clip_by_value(
            v_h_coarsest / v_h_max_log, tf.cast(0.0, prec), tf.cast(1.0, prec),
        )

        # Mode-weight entropy normalized by log K.
        q_safe = tf.maximum(q_all, tf.cast(1e-30, prec))
        h_q = -tf.reduce_sum(q_all * tf.math.log(q_safe), axis=1)         # (B,)
        h_q_norm = h_q / tf.cast(np.log(float(K)), prec)
        h_q_norm = tf.clip_by_value(h_q_norm, 0.0, 1.0)

        step_norm = 2.0 * tf.cast(meas_step[:, 0], prec) / float(simpars.num_steps) - 1.0
        res_norm = 2.0 * used_resources[:, 0] / tf.cast(simpars.max_resources, prec) - 1.0
        step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)
        res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        globals_ = tf.stack([log1p_vh_norm, h_q_norm, step_norm, res_norm], axis=1)
        return tf.concat([per_mode, globals_], axis=1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_function(
        self,
        weights: Tensor, particles: Tensor,
        true_values: Tensor, used_resources: Tensor, meas_step: Tensor,
    ) -> Tensor:
        """Per-step loss without controls reference (coarsest scale only).
        The full per-step loss with the current-scale term is computed
        inside execute() where controls are available.  Returns (B, 1).
        """
        del weights, particles, true_values, used_resources, meas_step
        return tf.expand_dims(self._per_step_loss(controls=None), axis=1)

    # ------------------------------------------------------------------
    # Debug snapshot
    # ------------------------------------------------------------------

    def _bank_snapshot(
        self,
        true_values: Tensor, controls: Tensor,
        used_resources: Tensor, meas_step: Tensor,
        max_examples: int = 3,
    ):
        bank = self.bank
        K = bank.K
        prec = self.simpars.prec

        g_mix, g_var = bank.marginal_mean_and_var()
        g_map, g_std_map = bank.map_mode_estimate()
        l_per_b = self._per_step_loss(controls=controls)
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
                "log_holevo_loss": float(l_per_b[b].numpy()),
                "V_H_coarsest": float(v_h_coarsest[b].numpy()),
                "V_H_at_kg": float(v_h_meas[b].numpy()),
                "k_g": float(k_g[b].numpy()),
                "K": int(K),
                "top_modes": top_modes,
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
        baseline_value: Optional[Tensor] = None,
    ):
        """Run one episode.  Returns (loss_diff, loss) by default, or a
        deploy-mode payload when deploy=True.

        loss_diff: scalar tensor that we differentiate (cumulative log-Holevo
                   plus REINFORCE surrogate).
        loss:      scalar tensor of the same magnitude that we report
                   (cumulative log-Holevo without the surrogate).
        """
        pars = self.simpars
        prec = pars.prec
        bank = self.bank
        debug_records: List[dict] = [] if debug else None

        # --- Episode init ---
        bank.reset(rangen)
        weights = bank.weights_list[0]
        particles = bank.particles_list[0]

        true_values = self.phys_model.true_values(rangen)            # (B, 1, d)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype="bool")
        outcomes = tf.zeros(
            (self.bs, self.phys_model.outcomes_size),
            dtype=self.phys_model.prec,
        )
        meas_step = tf.zeros((self.bs, 1), dtype="int32")
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        # We accumulate log-cumulative loss (Belliardo Eq. 109).
        # Tracker for the gradient: sum over t of log E_b[L_t] + sg(advantage)*sum_logprob.
        # Tracker for reporting: sum over t of log E_b[L_t].
        loss_diff_acc = tf.zeros((), dtype=prec)
        loss_acc = tf.zeros((), dtype=prec)
        step_count = 0

        if deploy:
            hist_inputs: List[Tensor] = []
            hist_controls: List[Tensor] = []
            hist_resources: List[Tensor] = []
            hist_precisions: List[Tensor] = []

        # --- Loop ---
        for _i in range(pars.num_steps):
            # Stop if a sufficient fraction has finished.
            num_finished = int(
                tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy()
            )
            if num_finished >= pars.resources_fraction * self.bs:
                break

            # 1. Build NN input.
            input_strategy = self.generate_input(
                weights, particles,
                tf.cast(meas_step, prec),
                used_resources,
                rangen,
            )

            # 2. Compute controls.
            cond_input = (
                tf.stop_gradient(input_strategy)
                if pars.stop_gradient_input else input_strategy
            )
            controls = self.control_strategy(cond_input)             # (B, controls_size)

            # 3. Update resource counter and continue flag.
            new_used_resources = self.phys_model.wrapper_count_resources(
                used_resources, outcomes, controls, true_values, true_state, meas_step,
            )
            new_continue_flag = tf.math.less_equal(
                new_used_resources,
                pars.max_resources * tf.ones((self.bs, 1), dtype=prec),
            )
            # Only allow batch elements that are CURRENTLY running to keep going
            # (a finished episode never resumes).
            new_continue_flag = tf.logical_and(new_continue_flag, continue_flag)
            used_resources = tf.where(new_continue_flag, new_used_resources, used_resources)
            continue_flag = new_continue_flag

            # 4. Sample outcomes from the true system.
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

            # 5. Accumulate sum_log_prob (only for episodes that ran THIS step).
            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(
                    continue_flag, sum_log_prob + log_prob, sum_log_prob,
                )

            # 6. Bank Bayes update (mode + within-mode), respects continue_flag.
            bank.apply_measurement(
                outcomes=outcomes,
                controls=controls,
                meas_step=meas_step,
                continue_flag=continue_flag,
                rangen=rangen,
            )

            # If the user has requested stop_gradient_pf, sever the
            # gradient through the bank state at this step.  Combined
            # with stop_gradient_input, this gives a fully greedy
            # optimization analogous to optbayesexpt (Belliardo Sec D.4).
            if pars.stop_gradient_pf:
                bank.mode_weights = tf.stop_gradient(bank.mode_weights)
                bank.weights_list = [tf.stop_gradient(w) for w in bank.weights_list]
                bank.particles_list = [tf.stop_gradient(p) for p in bank.particles_list]

            weights = bank.weights_list[0]
            particles = bank.particles_list[0]
            self.pf.np = bank.pf_list[0].np  # keep parent in sync

            # 7. Per-step loss.  Two-scale log-Holevo: coarsest + current k_g.
            L_step_b = self._per_step_loss(controls=controls)

            # Debug
            if debug:
                step_records = self._bank_snapshot(
                    true_values=true_values, controls=controls,
                    used_resources=used_resources, meas_step=meas_step,
                    max_examples=debug_max_examples,
                )
                mean_loss = float(tf.reduce_mean(L_step_b).numpy())
                for rec in step_records:
                    rec["loop_iter"] = int(_i)
                    rec["mean_log_holevo"] = mean_loss
                debug_records.extend(step_records)

            # Update step counter (only for active episodes).
            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            # 8. Accumulate log-cumulative loss (Belliardo Eq. 109).
            if not deploy:
                active = tf.cast(continue_flag[:, 0], prec)             # (B,)
                n_active = tf.maximum(tf.reduce_sum(active), tf.cast(1.0, prec))
                # Mean over active batch elements of L_step_b.
                mean_L_step = tf.reduce_sum(L_step_b * active) / n_active
                # log E[L] = log of the batch mean (Belliardo 2024 Eq. 109).
                # Floor to avoid log(0) when posterior collapses to a delta.
                log_step = tf.math.log(tf.maximum(mean_L_step, tf.cast(1e-30, prec)))
                loss_acc = loss_acc + log_step

                # REINFORCE surrogate.  d log E[L] / dlambda
                #     = (1/E[L]) * E[(L - B) * d log p_relevant / dlambda  +  dL/dlambda]
                # We construct a surrogate whose gradient equals the RHS:
                #     surrogate_t = log E[L]
                #                 + (1/sg(E[L])) * mean_b{ sg(L_b - B) * logp_relevant }
                # The first term provides dL/dlambda (path-derivative through
                # the bank).  The second term provides the score-function
                # correction (REINFORCE).
                #
                # Which "logp_relevant" to use:
                #   * stop_gradient_pf=False (full BPTT):  the loss at step t
                #     depends on every prior step's controls through the bank
                #     state, so the relevant log-prob is the CUMULATIVE
                #     sum_log_prob across all steps so far.
                #   * stop_gradient_pf=True  (1-step BPTT, greedy / paper-mode):
                #     the loss at step t depends ONLY on step t's controls.
                #     The relevant log-prob is the per-step log_prob — using
                #     the cumulative sum here would attribute step t's outcome
                #     randomness to controls of steps t' < t that no longer
                #     affect L_t, biasing the gradient with noise.
                if pars.loss_logl_outcomes:
                    if baseline_value is not None:
                        baseline_step = tf.cast(baseline_value, prec)
                    elif pars.baseline:
                        baseline_step = mean_L_step  # per-batch mean
                    else:
                        baseline_step = tf.zeros((), dtype=prec)
                    advantage = L_step_b - baseline_step                  # (B,)
                    if pars.stop_gradient_pf:
                        # Per-step log_prob (gated by continue_flag).
                        logp_for_score = tf.where(
                            continue_flag[:, 0],
                            log_prob[:, 0],
                            tf.zeros((self.bs,), dtype=prec),
                        )
                    else:
                        # Cumulative log_prob across all prior steps.
                        logp_for_score = sum_log_prob[:, 0]
                    inv_meanL = 1.0 / tf.stop_gradient(
                        tf.maximum(mean_L_step, tf.cast(1e-30, prec))
                    )
                    score_term = inv_meanL * (
                        tf.reduce_sum(
                            tf.stop_gradient(advantage) * logp_for_score * active
                        ) / n_active
                    )
                    loss_diff_acc = loss_diff_acc + log_step + score_term
                else:
                    loss_diff_acc = loss_diff_acc + log_step

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls.append(controls)
                hist_resources.append(used_resources)
                # For deploy we report log(1+V_H) at the coarsest scale.
                vh_coarsest = bank.log_holevo_at_scale(self._k_ref_coarsest)
                hist_precisions.append(tf.expand_dims(vh_coarsest, axis=1))

        # --- Normalize and return ---
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
# Controller (MLP)
# ---------------------------------------------------------------------------

def build_controller(
    input_size: int,
    phys_model: GravityStatelessPhysicalModel,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
) -> tf.keras.Model:
    """MLP that maps the bank summary to (T_s, Bp_kTm, mw_phase_rad)."""
    cfg = phys_model.cfg
    prec = cfg.prec

    T_min, T_max = float(cfg.T_range_s[0]), float(cfg.T_range_s[1])
    Bp_min, Bp_max = float(cfg.Bp_range_kTm[0]), float(cfg.Bp_range_kTm[1])
    phi_min, phi_max = -pi, pi
    T_mid, T_half = 0.5 * (T_max + T_min), 0.5 * (T_max - T_min)
    Bp_mid, Bp_half = 0.5 * (Bp_max + Bp_min), 0.5 * (Bp_max - Bp_min)
    phi_mid, phi_half = 0.5 * (phi_max + phi_min), 0.5 * (phi_max - phi_min)
    dtype = tf.float32 if prec == "float32" else tf.float64

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
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gravity_controller")


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
    """Wires up the physical model, bank, controller, and simulation."""
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = MultiPFBank(phys_model=phys_model, cfg=bank_cfg)

    top_k = bank_cfg.top_k_modes
    input_size = 3 * top_k + 4
    controller = build_controller(input_size, phys_model)

    # Warm-up call to build weights.
    dtype = tf.float32 if cfg.prec == "float32" else tf.float64
    _ = controller(tf.zeros((batchsize, input_size), dtype=dtype))

    # Reset bank BEFORE constructing the simulation, because the parent
    # Simulation.__init__ reads pf_list[0].
    bank.reset(rangen)

    sim = GravityMultiPFSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
    )
    return sim, bank, controller