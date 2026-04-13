"""Multi-PF Bank implementation for quantum gravity sensor.

This module implements the Multiple Model Adaptive Estimation (MMAE) architecture
for the levitated NV-center nanodiamond gravimeter described in
``MULTI_PF_REDESIGN.md``.

Architecture overview
---------------------
* :class:`MultiPFBankConfig` — frozen configuration dataclass for the bank.
* :class:`MultiPFBank` — manages K parallel qsensoropt
  :class:`~.ParticleFilter` instances (one per fringe hypothesis), their
  mode weights, adaptive splitting/pruning, and particle reallocation.
* :class:`GravityMultiPFSimulation` — subclasses
  :class:`~.StatelessSimulation`, overriding ``generate_input()`` and
  ``loss_function()`` (Holevo variance) to plug the Multi-PF bank into the
  standard ``utils.train()`` training loop.
* :func:`build_controller` — factory function for the MLP controller network.

Physics
-------
The measurement likelihood is::

    p(y=1 | g, T, B', φ) = 0.5 * (1 + vis * cos(k_g(T, B') * g + φ))

The gain ``k_g`` creates a cosine likelihood periodic in g with fringe
spacing ``Δ = 2π / k_g``.  When the gain is high, the posterior over g
becomes multimodal (one peak per fringe).  The Multi-PF bank maintains one
particle filter per fringe hypothesis and uses Bayesian weight updates (MMAE)
to disambiguate which fringe contains the true g.

Loss function
-------------
The Holevo variance :math:`V_H = |μ_H|^{-2} - 1` where
:math:`μ_H = E[e^{i k_{\\text{ref}} g}]` is the correct loss for multimodal
circular posteriors (Berry & Sanders 2009).

References
----------
Belliardo et al. (2024). Physical Review A 109, 062609.
  https://doi.org/10.1103/PhysRevA.109.062609

Berry & Sanders (2009). Physical Review A 80, 052114.
  https://arxiv.org/abs/0907.0014

Joas et al. (2021). npj Quantum Information 7, 56.
  https://www.nature.com/articles/s41534-021-00389-z

van den Berg (2021). Quantum 5, 469.
  https://quantum-journal.org/papers/q-2021-06-07-469/
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from math import ceil, pi
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model_complete import GravimeterConfig, GravityStatelessPhysicalModel


# ---------------------------------------------------------------------------
# Local qsensoropt loader
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
    from qsensoropt.utils import get_seed
except Exception:
    ParticleFilter = _load_local_qsensoropt_module("particle_filter").ParticleFilter
    StatelessSimulation = _load_local_qsensoropt_module(
        "stateless_simulation"
    ).StatelessSimulation
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters
    get_seed = _load_local_qsensoropt_module("utils").get_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiPFBankConfig:
    """Configuration for the Multi-PF Bank.

    Parameters
    ----------
    n_total : int
        Total particle budget shared across all active modes.
        Default: ``2048``.
    n_min : int
        Minimum number of particles per active mode.
        Prevents particle starvation when a mode has low but non-zero weight.
        Default: ``32``.
    k_max : int
        Maximum number of simultaneous modes (fringe hypotheses).
        Modes beyond this cap are pruned by weight even if above
        ``prune_threshold``.  Default: ``64``.
    prune_threshold : float
        Mode weights below this value are pruned (mode deactivated).
        Default: ``1e-6``.
    split_fringes_threshold : float
        A single-mode posterior is split when the 6σ width of the posterior
        spans more than ``split_fringes_threshold`` fringes of the current
        measurement gain.  Default: ``1.5``.
    top_k_modes : int
        Number of top-weighted modes to include explicitly in the controller
        input vector.  Default: ``4``.
    v_h_max : float
        Maximum Holevo variance for gradient clipping. Caps ``V_H`` to prevent
        gradient explosion when the posterior is fully ambiguous.
        Default: ``100.0``.
    resample_threshold : float
        ESS threshold fraction for per-mode resampling in the bank.
        Passed to each ``ParticleFilter`` instance.  Default: ``0.5``.
    resample_alpha : float
        Soft-resampling mixing coefficient α (Ścibior trick).  Default: ``0.5``.
    resample_beta : float
        Liu–West jitter parameter β (Gaussian perturbation scale).
        Default: ``0.98``.
    scibior_trick : bool
        Whether to use the Ścibior–Wood differentiable resampling trick.
        Must be ``True`` for gradient-based training.  Default: ``True``.
    trim : bool
        Whether to trim particles to parameter bounds after resampling.
        Default: ``True``.
    """

    n_total: int = 2048
    n_min: int = 32
    k_max: int = 64
    prune_threshold: float = 1e-6
    split_fringes_threshold: float = 1.5
    top_k_modes: int = 4
    v_h_max: float = 100.0
    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = True
    trim: bool = True


# ---------------------------------------------------------------------------
# Multi-PF Bank
# ---------------------------------------------------------------------------

class MultiPFBank:
    """Bank of K particle filters for multi-fringe phase disambiguation.

    Implements the Multiple Model Adaptive Estimation (MMAE) architecture:
    each active mode k corresponds to one fringe hypothesis — the interval
    ``[g_lo + k * Δ, g_lo + (k+1) * Δ]`` where ``Δ = 2π / k_g``.

    Mode weights ``q_k = P(true g in fringe k | data so far)`` are updated by
    Bayes rule after each measurement, with no mixing step (unlike IMM).

    **Key invariant:** ``sum_k (active_mask[b, k] * mode_weights[b, k]) = 1``
    for every batch element ``b``.

    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Shared physical model.  The g-parameter bounds define the full prior.
    cfg : MultiPFBankConfig
        Bank hyperparameters.
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

        # Mutable bank state — initialized by reset()
        self.weights_list: List[Tensor] = []    # (bs, N_k) normalized within-mode
        self.particles_list: List[Tensor] = []  # (bs, N_k, d)
        self.state_list: List[Tensor] = []      # (bs, N_k, 0) for stateless

        # mode_weights[b, k] = q_k for batch element b
        self.mode_weights = tf.Variable(
            tf.zeros((self.bs, cfg.k_max), dtype=self.prec),
            trainable=False,
            name="bank_mode_weights",
        )
        # active_mask[b, k] = 1 if mode k is active for batch b
        self.active_mask = tf.Variable(
            tf.zeros((self.bs, cfg.k_max), dtype=tf.int32),
            trainable=False,
            name="bank_active_mask",
        )
        self.k_active: int = 0  # current number of active modes (Python int)

        # Create a placeholder PF so that self.pf_list[0] is available
        # before reset() is called.  This is needed because the parent
        # Simulation.__init__ reads attributes from the particle filter.
        # reset() replaces this with a properly initialized PF.
        self._placeholder_pf = ParticleFilter(
            num_particles=cfg.n_total,
            phys_model=self.phys_model,
            resampling_allowed=True,
            resample_threshold=cfg.resample_threshold,
            alpha=cfg.resample_alpha,
            beta=cfg.resample_beta,
            scibior_trick=cfg.scibior_trick,
            trim=cfg.trim,
            prec=self.prec,
        )
        self.pf_list: List[ParticleFilter] = [self._placeholder_pf]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def reset(self, rangen: tf.random.Generator) -> None:
        """Initialize the bank with a single mode covering the full g prior.

        Equivalent to a standard qsensoropt ParticleFilter initialization.
        The multi-mode structure is created lazily when the first high-gain
        measurement triggers a split.

        Parameters
        ----------
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        cfg = self.cfg

        # Single PF with all particles covering [g_lo, g_hi]
        pf0 = ParticleFilter(
            num_particles=cfg.n_total,
            phys_model=self.phys_model,
            resampling_allowed=True,
            resample_threshold=cfg.resample_threshold,
            alpha=cfg.resample_alpha,
            beta=cfg.resample_beta,
            scibior_trick=cfg.scibior_trick,
            trim=cfg.trim,
            prec=self.prec,
        )
        w0, p0 = pf0.reset(rangen)
        # w0: (bs, n_total), p0: (bs, n_total, d)
        state0 = tf.zeros(
            (self.bs, cfg.n_total, 0),
            dtype=self.prec,
        )

        self.pf_list = [pf0]
        self.weights_list = [w0]
        self.particles_list = [p0]
        self.state_list = [state0]
        self.k_active = 1

        # Set mode weights: mode 0 has all weight
        new_mw = tf.zeros((self.bs, cfg.k_max), dtype=self.prec)
        new_mw = tf.tensor_scatter_nd_update(
            new_mw,
            tf.stack(
                [tf.range(self.bs, dtype=tf.int32),
                 tf.zeros((self.bs,), dtype=tf.int32)],
                axis=1,
            ),
            tf.ones((self.bs,), dtype=self.prec),
        )
        self.mode_weights.assign(new_mw)

        new_mask = tf.zeros((self.bs, cfg.k_max), dtype=tf.int32)
        new_mask = tf.tensor_scatter_nd_update(
            new_mask,
            tf.stack(
                [tf.range(self.bs, dtype=tf.int32),
                 tf.zeros((self.bs,), dtype=tf.int32)],
                axis=1,
            ),
            tf.ones((self.bs,), dtype=tf.int32),
        )
        self.active_mask.assign(new_mask)

    # ------------------------------------------------------------------
    # Mode weight update (MMAE Bayes rule)
    # ------------------------------------------------------------------

    def apply_measurement(
        self,
        outcomes: Tensor,
        controls: Tensor,
        meas_step: Tensor,
        rangen: tf.random.Generator,
    ) -> None:
        """Apply one measurement to all active modes.

        Steps:
        1. For each mode k: compute per-particle likelihoods and marginal
           likelihood ``Z_k = Σ_j w_{k,j} * p(y | g_{k,j}, x)``.
        2. Update mode weights: ``q_k ← q_k * Z_k``, renormalize.
        3. Update within-mode posterior weights (Bayes update).
        4. ESS-triggered resampling for each mode.
        5. Check for mode splitting (first high-gain measurement).
        6. Prune dead modes (q_k < prune_threshold).
        7. Reallocate particles proportional to mode weights.

        Parameters
        ----------
        outcomes : Tensor
            Shape ``(bs, outcomes_size)`` — binary measurement result.
        controls : Tensor
            Shape ``(bs, controls_size)`` — ``[T_s, Bp_kTm, mw_phase_rad]``.
        meas_step : Tensor
            Shape ``(bs, 1)`` — current step index.
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        prec = self.prec
        K = self.k_active

        # Broadcast outcomes and controls to (bs, N_k, ...) shapes
        # meas_step is (bs, 1), expand to (bs, 1, 1) for apply_measurement
        meas_step_1 = tf.expand_dims(meas_step, axis=2)  # (bs, 1, 1)

        Z_k_list: List[Tensor] = []
        new_weights_list: List[Tensor] = []

        for k in range(K):
            w_k = self.weights_list[k]       # (bs, N_k)
            p_k = self.particles_list[k]     # (bs, N_k, d)
            N_k = self.pf_list[k].np

            # Broadcast outcomes/controls to particle dimension
            outcomes_broad = tf.broadcast_to(
                tf.expand_dims(outcomes, axis=1),
                (self.bs, N_k, self.phys_model.outcomes_size),
            )
            controls_broad = tf.broadcast_to(
                tf.expand_dims(controls, axis=1),
                (self.bs, N_k, self.phys_model.controls_size),
            )
            step_broad = tf.broadcast_to(
                tf.expand_dims(meas_step, axis=2),
                (self.bs, N_k, 1),
            )
            state_k = self.state_list[k]  # (bs, N_k, 0)

            # Likelihood of each particle
            prob_k, _ = self.phys_model.wrapper_model(
                outcomes_broad, controls_broad, p_k, state_k, step_broad,
                num_systems=N_k,
            )
            # prob_k: (bs, N_k)

            # Marginal likelihood Z_k = sum_j w_{k,j} * prob_{k,j}
            unnorm_w_k = w_k * prob_k  # (bs, N_k)
            Z_k = tf.reduce_sum(unnorm_w_k, axis=1)  # (bs,)

            # Guard against degenerate weight collapse
            safe_Z_k = tf.maximum(Z_k, tf.cast(1e-300, prec))
            norm_w_k = unnorm_w_k / tf.expand_dims(safe_Z_k, axis=1)  # (bs, N_k)

            Z_k_list.append(Z_k)
            new_weights_list.append(norm_w_k)

        # --- Update mode weights ---
        # new_q_k ∝ q_k * Z_k
        Z_stack = tf.stack(Z_k_list, axis=1)           # (bs, K)
        q_active = self.mode_weights[:, :K]             # (bs, K)
        new_q_unnorm = q_active * Z_stack               # (bs, K)
        Z_total = tf.reduce_sum(new_q_unnorm, axis=1, keepdims=True)  # (bs, 1)
        safe_Z_total = tf.maximum(Z_total, tf.cast(1e-300, prec))
        new_q = new_q_unnorm / safe_Z_total             # (bs, K) normalized

        # Write back — preserve zeros for inactive slots
        mw_new = self.mode_weights.numpy()
        mw_new[:, :K] = new_q.numpy()
        self.mode_weights.assign(tf.cast(mw_new, prec))

        # --- Store new within-mode weights and apply resampling ---
        self.weights_list = new_weights_list

        for k in range(K):
            new_w_k, new_p_k, _ = self.pf_list[k].full_resampling(
                self.weights_list[k],
                self.particles_list[k],
                count_for_resampling=tf.ones((self.bs, 1), dtype="bool"),
                rangen=rangen,
            )
            self.weights_list[k] = new_w_k
            self.particles_list[k] = new_p_k

        # --- Prune dead modes (before split check, so K may drop to 1) ---
        self._prune_modes()

        # --- Adaptive mode creation ---
        # Check after pruning so that a bank that was pruned back to K=1
        # can re-split at the current (possibly higher) gain.
        if self.k_active == 1:
            k_g_val = self._compute_k_g_scalar(controls)
            if self._should_split(0, k_g_val):
                self._split_mode_0(k_g_val, rangen)

        # --- Reallocate particles ---
        self._reallocate_particles(rangen)

    # ------------------------------------------------------------------
    # Gain computation
    # ------------------------------------------------------------------

    def _compute_k_g_scalar(self, controls: Tensor) -> float:
        """Compute the mean k_g across the batch as a Python float.

        Parameters
        ----------
        controls : Tensor
            Shape ``(bs, controls_size)`` — ``[T_s, Bp_kTm, ...]``.

        Returns
        -------
        float
            Mean measurement gain ``k_g`` (rad·s²/m).
        """
        T_s = controls[:, 0]
        Bp_kTm = controls[:, 1]
        k_g_batch = self.phys_model.k_g(T_s, Bp_kTm)  # (bs,)
        return float(tf.reduce_mean(k_g_batch).numpy())

    # ------------------------------------------------------------------
    # Mode creation
    # ------------------------------------------------------------------

    def _should_split(self, mode_k: int, k_g_val: float) -> bool:
        """Check if mode k's posterior spans more than ``split_threshold`` fringes.

        Parameters
        ----------
        mode_k : int
            Index of the mode to check.
        k_g_val : float
            Current measurement gain (rad·s²/m).

        Returns
        -------
        bool
            True if the posterior 6σ width exceeds the split threshold.
        """
        if k_g_val <= 0.0:
            return False
        delta = 2.0 * pi / k_g_val
        # Compute std of mode k across batch
        mean_k = self.pf_list[mode_k].compute_mean(
            self.weights_list[mode_k], self.particles_list[mode_k]
        )  # (bs, d)
        cov_k = self.pf_list[mode_k].compute_covariance(
            self.weights_list[mode_k], self.particles_list[mode_k]
        )  # (bs, d, d)
        std_k = tf.sqrt(tf.maximum(cov_k[:, 0, 0], 0.0))  # (bs,)
        fringes_spanned = (6.0 * std_k) / delta  # (bs,)
        threshold = self.cfg.split_fringes_threshold
        return bool(tf.reduce_any(fringes_spanned > threshold).numpy())

    def _split_mode_0(self, k_g_val: float, rangen: tf.random.Generator) -> None:
        """Split the single mode into K sub-modes that tile the g-range.

        Each mode covers a contiguous interval ``[g_lo + k*Δg, g_lo + (k+1)*Δg]``
        where ``Δg = (g_hi - g_lo) / K``.  Particles from the parent mode are
        assigned to the sub-mode containing their g-value and resampled to fill
        the sub-mode's particle budget.

        **Why uniform tiling, not fringe-aligned?**  When ``K_physical`` (the
        number of fringes at gain ``k_g``) greatly exceeds ``K_new`` (the
        budget-limited mode count), fringe-index clipping would put almost all
        particles into the last mode.  Uniform tiling ensures each mode gets
        ``~N_total / K`` particles covering its g-interval.  The MMAE weight
        update then distinguishes modes because their particles live at
        different g-values, producing different marginal likelihoods.

        Parameters
        ----------
        k_g_val : float
            Current measurement gain (for K_new computation only).
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        cfg = self.cfg
        g_lo = self.g_lo
        g_hi = self.g_hi
        g_range_width = g_hi - g_lo

        K_physical = int(ceil(k_g_val * g_range_width / (2.0 * pi)))
        K_budget = max(1, cfg.n_total // cfg.n_min)
        K_new = max(1, min(K_physical, cfg.k_max, K_budget))

        w0 = self.weights_list[0]    # (bs, N_total)
        p0 = self.particles_list[0]  # (bs, N_total, d)
        g_vals = p0[:, :, 0]         # (bs, N_total)

        prec = self.prec
        N_total = cfg.n_total

        # Uniform g-range tiling: mode k covers
        # [g_lo + k * mode_width, g_lo + (k+1) * mode_width]
        mode_width = g_range_width / K_new
        mode_idx = tf.cast(
            tf.math.floor(
                (g_vals - tf.cast(g_lo, g_vals.dtype))
                / tf.cast(mode_width, g_vals.dtype)
            ),
            tf.int32,
        )  # (bs, N_total)
        mode_idx = tf.clip_by_value(mode_idx, 0, K_new - 1)

        new_weights_list: List[Tensor] = []
        new_q_list: List[Tensor] = []

        for k in range(K_new):
            mask_k = tf.cast(mode_idx == k, prec)  # (bs, N_total)
            q_new_k = tf.reduce_sum(w0 * mask_k, axis=1)  # (bs,)
            sum_k = tf.maximum(
                tf.reduce_sum(w0 * mask_k, axis=1, keepdims=True),
                tf.cast(1e-300, prec),
            )
            w_new_k = w0 * mask_k / sum_k  # (bs, N_total)
            new_weights_list.append(w_new_k)
            new_q_list.append(q_new_k)

        # Normalize mode weights (should be ~uniform for a uniform prior)
        q_stack = tf.stack(new_q_list, axis=1)  # (bs, K_new)
        q_sum = tf.maximum(
            tf.reduce_sum(q_stack, axis=1, keepdims=True),
            tf.cast(1e-300, prec),
        )
        q_norm = q_stack / q_sum  # (bs, K_new)

        mw_new = np.zeros(
            (self.bs, cfg.k_max),
            dtype=np.float32 if prec == "float32" else np.float64,
        )
        mw_new[:, :K_new] = q_norm.numpy()
        self.mode_weights.assign(tf.cast(mw_new, prec))

        mask_new = np.zeros((self.bs, cfg.k_max), dtype=np.int32)
        mask_new[:, :K_new] = 1
        self.active_mask.assign(mask_new)

        # Create fresh PF + particles for each mode via resampling
        fresh_pf_list: List[ParticleFilter] = []
        fresh_w_list: List[Tensor] = []
        fresh_p_list: List[Tensor] = []
        fresh_s_list: List[Tensor] = []

        for k in range(K_new):
            pf_k = ParticleFilter(
                num_particles=N_total,
                phys_model=self.phys_model,
                resampling_allowed=True,
                resample_threshold=cfg.resample_threshold,
                alpha=cfg.resample_alpha,
                beta=cfg.resample_beta,
                scibior_trick=cfg.scibior_trick,
                trim=cfg.trim,
                prec=prec,
            )
            w_k_raw = new_weights_list[k]  # (bs, N_total)
            log_w = tf.math.log(tf.maximum(w_k_raw, tf.cast(1e-30, prec)))
            seed_k = rangen.make_seeds(2)[:, 0]
            ancestors = tf.random.stateless_categorical(
                log_w, N_total, seed=seed_k, dtype=tf.int32,
            )  # (bs, N_total)
            batch_idx = tf.broadcast_to(
                tf.range(self.bs)[:, None], (self.bs, N_total)
            )
            idx = tf.stack([batch_idx, ancestors], axis=-1)
            p_k = tf.gather_nd(p0, idx)  # (bs, N_total, d)

            # Liu-West jitter (manual, avoids SVD)
            p_mean = tf.reduce_mean(p_k, axis=1, keepdims=True)
            p_std = tf.math.reduce_std(p_k, axis=1, keepdims=True)
            beta_lw = tf.cast(cfg.resample_beta, prec)
            jitter_std = tf.sqrt(1.0 - beta_lw ** 2) * p_std
            noise = rangen.normal(tf.shape(p_k), dtype=prec) * jitter_std
            p_k = beta_lw * p_k + (1.0 - beta_lw) * p_mean + noise

            # Clip to mode's g-interval (not full g-range)
            mode_lo = g_lo + k * mode_width
            mode_hi = g_lo + (k + 1) * mode_width
            p_k = tf.clip_by_value(
                p_k,
                tf.cast(mode_lo, prec),
                tf.cast(mode_hi, prec),
            )

            w_k = tf.ones((self.bs, N_total), dtype=prec) / float(N_total)
            fresh_pf_list.append(pf_k)
            fresh_w_list.append(w_k)
            fresh_p_list.append(p_k)
            fresh_s_list.append(tf.zeros((self.bs, N_total, 0), dtype=prec))

        self.pf_list = fresh_pf_list
        self.weights_list = fresh_w_list
        self.particles_list = fresh_p_list
        self.state_list = fresh_s_list
        self.k_active = K_new

    # ------------------------------------------------------------------
    # Mode pruning
    # ------------------------------------------------------------------

    def _prune_modes(self) -> None:
        """Deactivate modes with mean weight below ``prune_threshold``.

        Always preserves at least one mode (the maximum-weight mode).
        Updates ``mode_weights`` and ``active_mask`` in-place.
        """
        cfg = self.cfg
        K = self.k_active
        prec = self.prec

        q = self.mode_weights[:, :K].numpy()  # (bs, K)

        # Determine alive modes: q_k > threshold for majority of batch elements
        mean_q = np.mean(q, axis=0)  # (K,) mean over batch
        is_alive = mean_q > cfg.prune_threshold

        # Always keep at least one (the max-weight mode)
        if not np.any(is_alive):
            best_k = int(np.argmax(mean_q))
            is_alive[best_k] = True

        # Select surviving mode indices
        alive_indices = [k for k in range(K) if is_alive[k]]

        if len(alive_indices) == K:
            # Nothing pruned
            return

        # Rebuild bank with only surviving modes
        new_pf_list = [self.pf_list[k] for k in alive_indices]
        new_w_list = [self.weights_list[k] for k in alive_indices]
        new_p_list = [self.particles_list[k] for k in alive_indices]
        new_s_list = [self.state_list[k] for k in alive_indices]
        K_new = len(alive_indices)

        new_q = q[:, alive_indices]  # (bs, K_new)
        # Renormalize
        q_sum = np.maximum(new_q.sum(axis=1, keepdims=True), 1e-300)
        new_q = new_q / q_sum

        # Write back mode weights
        mw_new = np.zeros((self.bs, cfg.k_max), dtype=q.dtype)
        mw_new[:, :K_new] = new_q
        self.mode_weights.assign(tf.cast(mw_new, prec))

        mask_new = np.zeros((self.bs, cfg.k_max), dtype=np.int32)
        mask_new[:, :K_new] = 1
        self.active_mask.assign(mask_new)

        self.pf_list = new_pf_list
        self.weights_list = new_w_list
        self.particles_list = new_p_list
        self.state_list = new_s_list
        self.k_active = K_new

    def _remove_mode(self, idx: int) -> None:
        """Remove mode ``idx`` from the bank and renormalize weights."""
        K = self.k_active
        prec = self.prec
        cfg = self.cfg

        del self.pf_list[idx]
        del self.weights_list[idx]
        del self.particles_list[idx]
        del self.state_list[idx]
        K_new = K - 1
        self.k_active = K_new

        # Rebuild mode weights
        q = self.mode_weights.numpy()
        q_active = np.delete(q[:, :K], idx, axis=1)  # (bs, K_new)
        q_sum = np.maximum(q_active.sum(axis=1, keepdims=True), 1e-300)
        q_active = q_active / q_sum
        mw = np.zeros((self.bs, cfg.k_max), dtype=q.dtype)
        mw[:, :K_new] = q_active
        self.mode_weights.assign(tf.cast(mw, prec))

        mask = np.zeros((self.bs, cfg.k_max), dtype=np.int32)
        mask[:, :K_new] = 1
        self.active_mask.assign(mask)

    # ------------------------------------------------------------------
    # Particle reallocation
    # ------------------------------------------------------------------

    def _reallocate_particles(self, rangen: tf.random.Generator) -> None:
        """Redistribute N_total particles across active modes proportional
        to their mode weights.

        ``N_k = max(N_min, round(q_k * N_total))``

        The budget constraint is enforced by reducing the largest allocations
        until the sum equals N_total.

        When a mode's particle count changes, its particles are resampled to
        the new size via systematic resampling.

        Parameters
        ----------
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        cfg = self.cfg
        K = self.k_active
        prec = self.prec

        # Use mean mode weight across batch for allocation
        mean_q = tf.reduce_mean(self.mode_weights[:, :K], axis=0).numpy()  # (K,)

        # Proportional allocation respecting n_min and n_total
        # Step 1: give each mode its proportional share, floored to n_min
        raw_alloc = mean_q * cfg.n_total
        N_k = np.maximum(cfg.n_min, np.round(raw_alloc).astype(int))

        # Step 2: if total exceeds budget, trim from the lowest-weight modes
        # (never below n_min)
        while N_k.sum() > cfg.n_total:
            # Find the mode with the most "excess" above n_min
            excess = N_k - cfg.n_min
            if excess.max() <= 0:
                # All modes are at n_min; budget cannot shrink further.
                # This means K * n_min > n_total — prune weakest mode.
                weakest = int(np.argmin(mean_q))
                # Transfer its budget to strongest mode
                strongest = int(np.argmax(mean_q))
                N_k[strongest] += N_k[weakest]
                # Remove weakest mode from the bank
                self._remove_mode(weakest)
                K = self.k_active
                mean_q = np.delete(mean_q, weakest)
                N_k = np.delete(N_k, weakest)
                if K == 0:
                    break
                continue
            # Otherwise, trim 1 from the mode with most excess particles
            trim_idx = int(np.argmax(excess))
            N_k[trim_idx] -= 1

        # Step 3: if total is under budget, add to the highest-weight mode
        while N_k.sum() < cfg.n_total and K > 0:
            best = int(np.argmax(mean_q))
            N_k[best] += 1

        for k in range(K):
            old_N = self.pf_list[k].np
            new_N = int(N_k[k])
            if new_N == old_N:
                continue

            # Resample to new particle count
            w_k_old = self.weights_list[k]   # (bs, old_N)
            p_k_old = self.particles_list[k] # (bs, old_N, d)

            # Create new PF with updated particle count
            pf_new = ParticleFilter(
                num_particles=new_N,
                phys_model=self.phys_model,
                resampling_allowed=True,
                resample_threshold=cfg.resample_threshold,
                alpha=cfg.resample_alpha,
                beta=cfg.resample_beta,
                scibior_trick=cfg.scibior_trick,
                trim=cfg.trim,
                prec=prec,
            )

            if new_N > old_N:
                # Upsample: draw new_N samples from current posterior
                # Use systematic resampling via manual categorical sampling
                w_k_new, p_k_new = self._resample_to_size(
                    w_k_old, p_k_old, new_N, prec, rangen
                )
            else:
                # Downsample: take first new_N weighted samples
                w_k_new, p_k_new = self._resample_to_size(
                    w_k_old, p_k_old, new_N, prec, rangen
                )

            self.pf_list[k] = pf_new
            self.weights_list[k] = w_k_new
            self.particles_list[k] = p_k_new
            self.state_list[k] = tf.zeros((self.bs, new_N, 0), dtype=prec)

    @staticmethod
    def _resample_to_size(
        weights: Tensor,
        particles: Tensor,
        new_N: int,
        prec: str,
        rangen: tf.random.Generator,
    ) -> Tuple[Tensor, Tensor]:
        """Resample a particle set to a new size using stratified sampling.

        Parameters
        ----------
        weights : Tensor
            Shape ``(bs, old_N)``.
        particles : Tensor
            Shape ``(bs, old_N, d)``.
        new_N : int
            Target number of particles.
        prec : str
            Floating-point precision.
        rangen : tf.random.Generator
            TensorFlow random number generator.

        Returns
        -------
        new_weights : Tensor, shape ``(bs, new_N)``
            Uniform weights ``1/new_N``.
        new_particles : Tensor, shape ``(bs, new_N, d)``
            Resampled particles.
        """
        bs = weights.shape[0]
        old_N = weights.shape[1]
        d = particles.shape[2]

        # Cumulative weight for each batch element
        cum_weights = tf.cumsum(weights, axis=1)  # (bs, old_N)

        # Stratified sample positions
        u = rangen.uniform((bs, new_N), dtype=prec)
        u = tf.sort(u, axis=1)  # (bs, new_N)

        # Find indices: for each u, find first cum_weight >= u
        # Broadcast: (bs, 1, old_N) vs (bs, new_N, 1)
        cum_broad = tf.expand_dims(cum_weights, axis=1)  # (bs, 1, old_N)
        u_broad = tf.expand_dims(u, axis=2)              # (bs, new_N, 1)
        mask = tf.cast(cum_broad >= u_broad, tf.int32)   # (bs, new_N, old_N)

        # First index where cum_weight >= u
        # Reverse along last dim, take argmax, reverse back
        indices = old_N - 1 - tf.argmax(
            tf.reverse(mask, axis=[2]),
            axis=2,
            output_type=tf.int32,
        )  # (bs, new_N)
        indices = tf.clip_by_value(indices, 0, old_N - 1)

        # Gather particles
        batch_idx = tf.broadcast_to(
            tf.expand_dims(tf.range(bs, dtype=tf.int32), axis=1),
            (bs, new_N),
        )  # (bs, new_N)
        gather_idx = tf.stack([batch_idx, indices], axis=2)  # (bs, new_N, 2)
        new_particles = tf.gather_nd(particles, gather_idx)  # (bs, new_N, d)

        # Uniform weights
        new_weights = tf.ones((bs, new_N), dtype=prec) / tf.cast(new_N, prec)

        return new_weights, new_particles

    # ------------------------------------------------------------------
    # Posterior statistics
    # ------------------------------------------------------------------

    def marginal_mean_and_var(self) -> Tuple[Tensor, Tensor]:
        """Compute the marginal mean and variance of g.

        Returns the mixture mean ``μ = Σ_k q_k μ_k`` and variance
        ``σ² = Σ_k q_k (σ_k² + μ_k²) - μ²``.

        Returns
        -------
        g_mean : Tensor, shape ``(bs,)``
        g_var : Tensor, shape ``(bs,)``
        """
        K = self.k_active
        prec = self.prec
        q = self.mode_weights[:, :K]  # (bs, K)

        mean_list = []
        var_list = []
        for k in range(K):
            mean_k = self.pf_list[k].compute_mean(
                self.weights_list[k], self.particles_list[k]
            )[:, 0]  # (bs,)
            cov_k = self.pf_list[k].compute_covariance(
                self.weights_list[k], self.particles_list[k]
            )[:, 0, 0]  # (bs,)
            mean_list.append(mean_k)
            var_list.append(cov_k)

        means = tf.stack(mean_list, axis=1)  # (bs, K)
        vars_ = tf.stack(var_list, axis=1)   # (bs, K)

        g_mean = tf.reduce_sum(q * means, axis=1)  # (bs,)
        # Var(g) = E[g²] - E[g]² = Σ_k q_k (σ_k² + μ_k²) - μ²
        g_mean_sq = tf.reduce_sum(q * (vars_ + tf.square(means)), axis=1)
        g_var = tf.maximum(g_mean_sq - tf.square(g_mean), tf.cast(0.0, prec))
        return g_mean, g_var

    def map_mode_mean(self) -> Tuple[Tensor, Tensor]:
        """Return the mean and variance from the highest-weight mode.

        For multimodal posteriors, the mixture mean ``Σ q_k μ_k`` is
        unreliable (it may fall between modes where the density is zero).
        The MAP-mode estimator picks the mode with the largest ``q_k``
        and returns that mode's mean — a much better point estimate.

        Returns
        -------
        g_map : Tensor, shape ``(bs,)``
            Mean of the MAP (highest-weight) mode for each batch element.
        g_var_map : Tensor, shape ``(bs,)``
            Variance of the MAP mode.
        """
        K = self.k_active
        prec = self.prec
        q = self.mode_weights[:, :K]  # (bs, K)

        # Per-batch argmax of mode weights
        best_k = tf.argmax(q, axis=1, output_type=tf.int32)  # (bs,)

        mean_list = []
        var_list = []
        for k in range(K):
            mean_k = self.pf_list[k].compute_mean(
                self.weights_list[k], self.particles_list[k]
            )[:, 0]  # (bs,)
            cov_k = self.pf_list[k].compute_covariance(
                self.weights_list[k], self.particles_list[k]
            )[:, 0, 0]
            mean_list.append(mean_k)
            var_list.append(cov_k)

        means = tf.stack(mean_list, axis=1)  # (bs, K)
        vars_ = tf.stack(var_list, axis=1)   # (bs, K)

        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)  # (bs, 2)
        g_map = tf.gather_nd(means, gather_idx)    # (bs,)
        g_var_map = tf.gather_nd(vars_, gather_idx)  # (bs,)
        return g_map, g_var_map

    def mode_mean(self, mode_k: int) -> Tensor:
        """Mean of mode k's posterior.

        Parameters
        ----------
        mode_k : int
            Mode index.

        Returns
        -------
        Tensor, shape ``(bs,)``
        """
        return self.pf_list[mode_k].compute_mean(
            self.weights_list[mode_k], self.particles_list[mode_k]
        )[:, 0]

    def mode_std(self, mode_k: int) -> Tensor:
        """Standard deviation of mode k's posterior.

        Parameters
        ----------
        mode_k : int
            Mode index.

        Returns
        -------
        Tensor, shape ``(bs,)``
        """
        cov = self.pf_list[mode_k].compute_covariance(
            self.weights_list[mode_k], self.particles_list[mode_k]
        )  # (bs, d, d)
        return tf.sqrt(tf.maximum(cov[:, 0, 0], tf.cast(0.0, self.prec)))

    # ------------------------------------------------------------------
    # Holevo variance
    # ------------------------------------------------------------------

    def holevo_variance(self, k_ref: Tensor) -> Tensor:
        """Holevo variance of the marginal posterior.

        .. math::

            μ_H = \\sum_k q_k \\sum_j w_{k,j} e^{i k_{\\text{ref}} g_{k,j}}

            V_H = |μ_H|^{-2} - 1

        Parameters
        ----------
        k_ref : Tensor
            Shape ``(bs,)`` — reference gain for circular statistics.

        Returns
        -------
        Tensor, shape ``(bs, 1)``
            Holevo variance, clipped to ``[0, V_H_MAX]``.
        """
        K = self.k_active
        prec = self.prec
        v_h_max = tf.cast(self.cfg.v_h_max, prec)
        q = self.mode_weights[:, :K]  # (bs, K)

        mu_k_list: List[Tensor] = []
        for k in range(K):
            p_k = self.particles_list[k]   # (bs, N_k, d)
            w_k = self.weights_list[k]     # (bs, N_k)
            g_k = p_k[:, :, 0]            # (bs, N_k)
            q_k = q[:, k]                 # (bs,)

            # Phase of each particle: k_ref * g_{k,j}
            # k_ref: (bs,) — broadcast to (bs, N_k)
            phase_k = tf.expand_dims(k_ref, axis=1) * g_k  # (bs, N_k)

            # Complex moment within mode k
            cos_k = tf.cos(phase_k)
            sin_k = tf.sin(phase_k)
            # E_k[e^{i k_ref g}] = Σ_j w_{k,j} e^{i k_ref g_{k,j}}
            re_k = tf.reduce_sum(w_k * cos_k, axis=1)  # (bs,)
            im_k = tf.reduce_sum(w_k * sin_k, axis=1)  # (bs,)

            # Weight by mode probability
            mu_k_re = q_k * re_k  # (bs,)
            mu_k_im = q_k * im_k  # (bs,)
            mu_k_list.append(tf.stack([mu_k_re, mu_k_im], axis=1))  # (bs, 2)

        # Marginal complex moment μ_H = Σ_k q_k μ_k
        mu_H_components = tf.add_n(mu_k_list)          # (bs, 2)
        mu_H_re = mu_H_components[:, 0]                # (bs,)
        mu_H_im = mu_H_components[:, 1]                # (bs,)
        abs_mu_sq = mu_H_re ** 2 + mu_H_im ** 2       # (bs,)

        # V_H = 1/|μ_H|² - 1, clipped
        V_H = 1.0 / tf.maximum(abs_mu_sq, tf.cast(1e-10, prec)) - 1.0
        V_H_clipped = tf.minimum(V_H, v_h_max)
        V_H_clipped = tf.maximum(V_H_clipped, tf.cast(0.0, prec))

        return tf.expand_dims(V_H_clipped, axis=1)  # (bs, 1)

    # ------------------------------------------------------------------
    # Adaptive k_ref
    # ------------------------------------------------------------------

    def adaptive_k_ref(self, k_g_max: float) -> Tensor:
        """Compute an adaptive reference gain for the Holevo variance.

        Sets ``k_ref`` so that the 6σ width of the marginal posterior
        corresponds approximately to one period ``2π/k_ref``.  Clamped to
        the range ``[2π/(g_hi - g_lo), k_g_max / 2]``.

        Parameters
        ----------
        k_g_max : float
            Maximum gain ``k_g`` at current controls (for upper clamp).

        Returns
        -------
        Tensor, shape ``(bs,)``
        """
        prec = self.prec
        _, g_var = self.marginal_mean_and_var()  # (bs,)
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-20, prec)))  # (bs,)

        k_ref_adaptive = 2.0 * pi / tf.maximum(
            6.0 * g_std, tf.cast(1e-10, prec)
        )
        k_ref_min = tf.cast(
            2.0 * pi / max(self.g_hi - self.g_lo, 1e-10), prec
        )
        k_ref_max = tf.cast(max(k_g_max / 2.0, k_ref_min), prec)

        k_ref = tf.clip_by_value(k_ref_adaptive, k_ref_min, k_ref_max)
        return k_ref  # (bs,)


# ---------------------------------------------------------------------------
# Gravity Multi-PF Simulation
# ---------------------------------------------------------------------------

class GravityMultiPFSimulation(StatelessSimulation):
    """Simulation class integrating the Multi-PF Bank with qsensoropt.

    Subclasses :class:`~.StatelessSimulation`, overriding:

    * :meth:`generate_input` — builds a compact fixed-size NN input vector
      from bank statistics (top-K mode summaries + global features).
    * :meth:`loss_function` — returns the Holevo variance of the marginal
      posterior.
    * :meth:`execute` — replaces the qsensoropt ``tf.while_loop`` with a
      Python-level measurement loop so that the bank's Python state
      (particle lists, mode weight variables) can be updated at each step
      without graph-mode constraints.  This is Phase 1–5 compatible.
      XLA compilation (Phase 6) would require flattening the bank state
      into the loop-variable tensors, as described in Section 6.3 of the
      redesign document.

    **Integration strategy:**  The qsensoropt ``Simulation`` base class
    operates on a single ``ParticleFilter`` instance (``self.pf``).  In
    single-mode operation, ``self.pf`` IS the bank's mode-0 PF and the
    entire flow is identical to standard qsensoropt.  When the bank splits
    into multiple modes, the Python measurement loop routes each outcome
    through ``bank.apply_measurement()`` which handles all K modes, then
    synchronizes ``self.pf`` with bank mode 0 for consistent API compliance.

    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Shared physical model.
    bank : MultiPFBank
        The multi-PF bank.
    controller : callable
        Neural network ``input_vec → controls``.
    simpars : SimulationParameters
        qsensoropt simulation parameters.
    bank_cfg : MultiPFBankConfig
        Bank configuration.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: MultiPFBank,
        controller,
        simpars: SimulationParameters,
        bank_cfg: MultiPFBankConfig,
    ) -> None:
        # Input vector size: 4 * TOP_K + 5
        top_k = bank_cfg.top_k_modes
        input_size = 4 * top_k + 5

        input_name = []
        for i in range(top_k):
            input_name += [
                f"mu_mode_{i}", f"log_std_mode_{i}",
                f"q_mode_{i}", f"active_mode_{i}",
            ]
        input_name += ["V_H_norm", "H_q_norm", "N_active_norm",
                       "step_norm", "res_norm"]

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
        self._k_g_max: float = float(
            phys_model.max_gain(
                tf.float32 if phys_model.prec == "float32" else tf.float64
            ).numpy()
        )

    def generate_input(
        self,
        weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen,
    ) -> Tensor:
        """Build the NN input vector from the current bank state.

        Input layout (size = ``4 * TOP_K + 5``):

        For each of the top-K modes sorted by weight descending:

        * ``μ_k`` normalized: ``2*(μ_k - g_lo)/(g_hi - g_lo) - 1`` ∈ [-1, 1]
        * ``log_std_k``: ``-2/10 * log10(σ_k) - 1`` ∈ [-1, 1]
        * ``q_k`` ∈ [0, 1]
        * ``active_k`` ∈ {0, 1}

        Global features:

        * ``tanh(log(V_H + 1)) / tanh(log(101))`` ∈ [-1, 1]
        * ``H_q / log(K_max)`` — normalized mode-weight entropy
        * ``N_active / K_max``
        * ``meas_step / num_steps * 2 - 1`` ∈ [-1, 1]
        * ``used_resources / max_resources * 2 - 1`` ∈ [-1, 1]

        Parameters
        ----------
        weights : Tensor, shape ``(bs, pf.np)``
            Primary PF weights (from qsensoropt loop — kept for API compat).
        particles : Tensor, shape ``(bs, pf.np, d)``
            Primary PF particles (from qsensoropt loop — kept for API compat).
        meas_step : Tensor, shape ``(bs, 1)``
        used_resources : Tensor, shape ``(bs, 1)``
        rangen : tf.random.Generator

        Returns
        -------
        Tensor, shape ``(bs, input_size)``
        """
        bank = self.bank
        cfg = self.bank_cfg
        prec = self.simpars.prec
        simpars = self.simpars
        TOP_K = cfg.top_k_modes
        K_active = bank.k_active
        K_max = cfg.k_max
        bs = self.bs

        # --- Synchronize bank's mode 0 with primary PF (if K==1) ---
        # (for multi-mode case, bank already has its own state)
        if K_active == 1:
            bank.weights_list[0] = weights
            bank.particles_list[0] = particles

        # --- Sort modes by weight ---
        q_all = bank.mode_weights  # (bs, K_max)
        # Sort indices by descending weight
        # Use top-K of active modes
        q_active_np = q_all[:, :K_active].numpy()  # (bs, K_active)
        top_k_indices_np = np.argsort(-q_active_np, axis=1)[:, :TOP_K]  # (bs, TOP_K)

        mode_features_list = []
        for i in range(TOP_K):
            if i < K_active:
                k_vec = top_k_indices_np[:, i]  # (bs,) Python int array
                mu_list = []
                log_std_list = []
                q_list = []
                for b in range(bs):
                    k = int(k_vec[b])
                    mu_k_b = float(bank.mode_mean(k)[b].numpy())
                    std_k_b = float(bank.mode_std(k)[b].numpy())
                    q_k_b = float(q_all[b, k].numpy())
                    mu_list.append(mu_k_b)
                    log_std_list.append(std_k_b)
                    q_list.append(q_k_b)

                mu_arr = np.array(mu_list, dtype=np.float64 if prec == "float64" else np.float32)
                std_arr = np.array(log_std_list, dtype=mu_arr.dtype)
                q_arr = np.array(q_list, dtype=mu_arr.dtype)

                # Normalize mean to [-1, 1]
                mu_norm = tf.cast(
                    2.0 * (mu_arr - self.g_lo) / max(self.g_hi - self.g_lo, 1e-10) - 1.0,
                    prec,
                )
                mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)

                # log std encoding: -2/10 * log10(σ) - 1
                std_safe = np.maximum(std_arr, 1e-10 if prec == "float64" else 1e-6)
                log_std = tf.cast(
                    -2.0 / 10.0 * np.log10(std_safe) - 1.0, prec
                )
                log_std = tf.clip_by_value(log_std, -1.0, 1.0)

                q_t = tf.cast(q_arr, prec)
                active_t = tf.ones((bs,), dtype=prec)

                mode_features_list.append(tf.reshape(mu_norm, (bs, 1)))
                mode_features_list.append(tf.reshape(log_std, (bs, 1)))
                mode_features_list.append(tf.reshape(q_t, (bs, 1)))
                mode_features_list.append(tf.reshape(active_t, (bs, 1)))
            else:
                # Padding for missing modes
                zeros = tf.zeros((bs, 4), dtype=prec)
                mode_features_list.append(zeros)

        # mode_features: (bs, 4*TOP_K)
        if mode_features_list:
            # Each element is either (bs, 1) or (bs, 4)
            # Flatten to (bs, 4*TOP_K)
            mode_features_flat = []
            for feat in mode_features_list:
                if feat.shape[-1] == 4:
                    # Padding block
                    mode_features_flat.append(feat)
                else:
                    mode_features_flat.append(feat)
            mode_features = tf.concat(mode_features_flat, axis=1)  # (bs, 4*TOP_K)
        else:
            mode_features = tf.zeros((bs, 4 * TOP_K), dtype=prec)

        # --- Holevo variance ---
        k_ref = bank.adaptive_k_ref(self._k_g_max)  # (bs,)
        V_H = bank.holevo_variance(k_ref)[:, 0]  # (bs,)
        # Normalize: tanh(log(V_H + 1)) / tanh(log(101))
        tanh_101 = tf.cast(float(np.tanh(np.log(101.0))), prec)
        V_H_norm = tf.math.tanh(tf.math.log(tf.cast(V_H, prec) + 1.0)) / tanh_101
        V_H_norm = tf.clip_by_value(V_H_norm, -1.0, 1.0)

        # --- Mode weight entropy ---
        q_safe = tf.maximum(
            q_all * tf.cast(bank.active_mask, prec),
            tf.cast(1e-30, prec),
        )
        H_q = -tf.reduce_sum(q_safe * tf.math.log(q_safe), axis=1)  # (bs,)
        log_Kmax = tf.cast(max(np.log(float(K_max)), 1e-10), prec)
        H_q_norm = H_q / log_Kmax  # (bs,)
        H_q_norm = tf.clip_by_value(H_q_norm, 0.0, 1.0)

        # --- N_active / K_max ---
        N_active = tf.cast(tf.reduce_sum(bank.active_mask, axis=1), prec) / float(K_max)
        N_active = tf.clip_by_value(N_active, 0.0, 1.0)  # (bs,)

        # --- Step and resource normalized ---
        step_norm = (
            2.0 * tf.cast(meas_step[:, 0], prec) / float(simpars.num_steps) - 1.0
        )
        res_norm = 2.0 * used_resources[:, 0] / tf.cast(simpars.max_resources, prec) - 1.0
        step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)
        res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        # --- Assemble global features ---
        global_features = tf.stack(
            [V_H_norm, H_q_norm, N_active, step_norm, res_norm], axis=1
        )  # (bs, 5)

        input_vec = tf.concat([mode_features, global_features], axis=1)  # (bs, input_size)
        return input_vec

    def loss_function(
        self,
        weights: Tensor,
        particles: Tensor,
        true_values: Tensor,
        used_resources: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """Holevo variance of the marginal posterior.

        Correctly penalizes unresolved fringe ambiguity: the Holevo variance
        diverges for a symmetric bimodal posterior, whereas MSE would report
        a spuriously small value.

        Parameters
        ----------
        weights : Tensor, shape ``(bs, pf.np)``
            Primary PF weights (kept for API compatibility).
        particles : Tensor, shape ``(bs, pf.np, d)``
            Primary PF particles (kept for API compatibility).
        true_values : Tensor, shape ``(bs, 1, d)``
            True parameter values (not used in loss computation, for debug).
        used_resources : Tensor, shape ``(bs, 1)``
        meas_step : Tensor, shape ``(bs, 1)``

        Returns
        -------
        Tensor, shape ``(bs, 1)``
            Holevo variance, clipped to ``[0, V_H_MAX]``.
        """
        bank = self.bank
        if bank.k_active == 1:
            # Sync mode 0 with primary PF
            bank.weights_list[0] = weights
            bank.particles_list[0] = particles

        # FIXED k_ref for loss: use the coarsest scale so that V_H detects
        # whether the posterior has collapsed to a single region or is still
        # spread across the entire g-range.  Adaptive k_ref would make V_H
        # insensitive to wrong-fringe collapse (a posterior at the wrong
        # fringe with the same width reports the same V_H).
        k_ref_fixed = tf.cast(
            2.0 * pi / max(self.g_hi - self.g_lo, 1e-10), self.simpars.prec
        ) * tf.ones((self.bs,), dtype=self.simpars.prec)
        return bank.holevo_variance(k_ref_fixed)  # (bs, 1)
    
    def _bank_snapshot(
        self,
        true_values: Tensor,
        controls: Tensor,
        used_resources: Tensor,
        meas_step: Tensor,
        max_examples: int = 3,
    ):
        """
        Create JSON-serializable debug snapshots for a few batch elements.
        """
        bank = self.bank
        K = bank.k_active
        prec = self.simpars.prec

        g_mix, g_var = bank.marginal_mean_and_var()
        g_map, g_var_map = bank.map_mode_mean()

        k_ref_fixed = tf.cast(
            2.0 * pi / max(self.g_hi - self.g_lo, 1e-10), prec
        ) * tf.ones((self.bs,), dtype=prec)
        V_H = bank.holevo_variance(k_ref_fixed)[:, 0]
        k_g = self.phys_model.k_g(controls[:, 0], controls[:, 1])

        q_np = bank.mode_weights[:, :K].numpy() if K > 0 else np.zeros((self.bs, 0))
        active_np = bank.active_mask[:, :K].numpy() if K > 0 else np.zeros((self.bs, 0))

        mode_means = []
        mode_stds = []
        for k in range(K):
            mode_means.append(bank.mode_mean(k).numpy())
            mode_stds.append(bank.mode_std(k).numpy())

        n_show = min(int(max_examples), self.bs)
        records = []

        for b in range(n_show):
            order = np.argsort(-q_np[b]) if K > 0 else np.array([], dtype=np.int32)
            top_modes = []
            for k in order[: min(3, K)]:
                top_modes.append(
                    {
                        "mode": int(k),
                        "q": float(q_np[b, k]),
                        "mu": float(mode_means[k][b]),
                        "std": float(mode_stds[k][b]),
                        "active": int(active_np[b, k]),
                    }
                )

            records.append(
                {
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
                    "g_var_map": float(g_var_map[b].numpy()),
                    "V_H": float(V_H[b].numpy()),
                    "k_g": float(k_g[b].numpy()),
                    "k_active_bank": int(K),
                    "n_active_row": int(np.sum(active_np[b])) if K > 0 else 0,
                    "q0": float(q_np[b, 0]) if K > 0 else 0.0,
                    "top_modes": top_modes,
                }
            )

        return records

    def execute(self, rangen: tf.random.Generator, deploy: bool = False, debug: bool = False, debug_max_examples: int = 3,):
        """Bank-aware measurement loop.

        Replaces the qsensoropt ``tf.while_loop`` with a Python-level loop so
        that the Multi-PF bank's Python state (mode lists, ``tf.Variable``
        weights) can be updated at every step without graph-mode constraints.

        At each measurement step:

        1. ``generate_input(weights, particles, ...)`` — builds NN input from
           the current bank state (bank mode 0 is synced with primary PF
           weights/particles).
        2. ``controller(input)`` — produces controls ``[T_s, Bp, φ]``.
        3. ``phys_model.wrapper_perform_measurement(controls, true_values)``
           — samples binary outcome.
        4. ``bank.apply_measurement(outcomes, controls, step, rangen)``
           — Bayes-updates all active modes, updates mode weights,
           optionally splits/prunes/reallocates.
        5. ``loss_function(bank)`` — Holevo variance.

        For training, call this method inside ``tf.GradientTape``.  The
        gradient flows through steps 2, 4, and 5 via the model-aware
        gradient (differentiable Bayes update + REINFORCE for step 3).

        Parameters
        ----------
        rangen : tf.random.Generator
            TensorFlow random number generator.
        deploy : bool, optional
            If ``True``, returns history tensors; if ``False``, returns
            ``(loss_diff, loss)`` for training.  Default ``False``.

        Returns
        -------
        loss_diff : Tensor, shape ``()``
            Differentiable loss (REINFORCE-augmented Holevo variance).
            Returned when ``deploy=False``.
        loss : Tensor, shape ``()``
            Pure (non-augmented) mean Holevo variance.  Always returned.
        history_* : Tensors
            History arrays.  Returned when ``deploy=True``.
        """
        pars = self.simpars
        prec = pars.prec
        bank = self.bank
        debug_records = [] if debug else None

        # --- Initialize episode ---
        bank.reset(rangen)
        # Sync primary PF with bank mode 0 so generate_input sees a fresh state
        weights = bank.weights_list[0]        # (bs, N)
        particles = bank.particles_list[0]   # (bs, N, d)

        # Draw true parameter values
        true_values = self.phys_model.true_values(rangen)  # (bs, 1, d)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)
        state_ensemble = self.pf.phys_model.wrapper_initialize_state(
            particles, bank.cfg.n_total,
        )

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype="bool")
        outcomes = tf.zeros(
            (self.bs, self.phys_model.outcomes_size), dtype=self.phys_model.prec
        )
        meas_step = tf.zeros((self.bs, 1), dtype="int32")
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        loss_diff_accum = tf.zeros((), dtype=prec)
        loss_accum = tf.zeros((), dtype=prec)
        step_count = 0

        # --- Deploy-mode history ---
        if deploy:
            hist_inputs = []
            hist_controls_list = []
            hist_resources = []
            hist_precisions = []

        # --- Measurement loop ---
        for _i in range(pars.num_steps):
            # Check if all estimations exhausted resources
            num_finished = int(tf.math.count_nonzero(
                tf.logical_not(continue_flag)
            ).numpy())
            if num_finished >= pars.resources_fraction * self.bs:
                break

            # Sync primary PF with bank mode 0 for generate_input
            if bank.k_active == 1:
                bank.weights_list[0] = weights
                bank.particles_list[0] = particles

            # 1. Build NN input
            input_strategy = self.generate_input(
                weights, particles,
                tf.cast(meas_step, prec),
                used_resources,
                rangen,
            )

            # 2. Compute controls
            cond_input = (
                tf.stop_gradient(input_strategy)
                if pars.stop_gradient_input else input_strategy
            )
            controls = self.control_strategy(cond_input)  # (bs, controls_size)

            # 3. Update resources
            new_used_resources = self.phys_model.wrapper_count_resources(
                used_resources, outcomes, controls, true_values, true_state, meas_step,
            )
            continue_flag = tf.math.less_equal(
                new_used_resources,
                pars.max_resources * tf.ones((self.bs, 1), dtype=prec),
            )
            used_resources = tf.where(continue_flag, new_used_resources, used_resources)

            # 4. Perform measurement on the true system
            outcomes_raw, log_prob, post_true_state = \
                self.phys_model.wrapper_perform_measurement(
                    tf.expand_dims(controls, axis=1),
                    true_values,
                    true_state,
                    tf.expand_dims(meas_step, axis=1),
                    rangen,
                )
            outcomes = outcomes_raw[:, 0, :]  # (bs, outcomes_size)
            # For stateless models state_size == 0, skip state update
            if self.state_size > 0:
                continue_flag_state = tf.reshape(continue_flag, (self.bs, 1, 1))
                true_state = tf.where(
                    tf.broadcast_to(continue_flag_state, (self.bs, 1, self.state_size)),
                    post_true_state, true_state,
                )

            # Accumulate log prob for REINFORCE
            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(
                    continue_flag, sum_log_prob + log_prob, sum_log_prob,
                )

            # 5. Apply measurement to the bank (Bayes update all modes)
            bank.apply_measurement(
                outcomes, controls,
                meas_step, rangen,
            )

            # Sync primary PF from bank mode 0
            weights = bank.weights_list[0]
            particles = bank.particles_list[0]
            # Update primary pf.np to match mode 0 allocation
            self.pf.np = bank.pf_list[0].np

            # 6. Compute loss using FIXED k_ref (coarsest scale)
            k_ref_fixed = tf.cast(
                2.0 * pi / max(self.g_hi - self.g_lo, 1e-10), prec
            ) * tf.ones((self.bs,), dtype=prec)
            V_H = bank.holevo_variance(k_ref_fixed)  # (bs, 1)

            if debug:
                step_records = self._bank_snapshot(
                    true_values=true_values,
                    controls=controls,
                    used_resources=used_resources,
                    meas_step=meas_step,
                    max_examples=debug_max_examples,
                )
                for rec in step_records:
                    rec["loop_iter"] = int(_i)
                    rec["mean_V_H_batch"] = float(tf.reduce_mean(V_H).numpy())
                debug_records.extend(step_records)

            # Update step counter
            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            # 7. Accumulate cumulative loss
            if pars.cumulative_loss and not deploy:
                loss_mean = tf.reduce_mean(V_H)
                if pars.loss_logl_outcomes:
                    baseline = loss_mean if pars.baseline else tf.zeros((), dtype=prec)
                    loss_diff_partial = tf.reduce_mean(
                        V_H + (
                            tf.stop_gradient(V_H) - tf.stop_gradient(baseline)
                        ) * sum_log_prob
                    )
                else:
                    loss_diff_partial = loss_mean

                # Mask out finished estimations
                active = tf.cast(continue_flag, prec)
                n_active = tf.maximum(
                    tf.reduce_sum(active), tf.cast(1.0, prec)
                )
                loss_diff_accum = loss_diff_accum + tf.reduce_sum(
                    tf.where(continue_flag, V_H, tf.zeros_like(V_H))
                ) / n_active
                loss_accum = loss_accum + tf.reduce_mean(
                    tf.where(continue_flag, V_H, tf.zeros_like(V_H))
                )

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls_list.append(controls)
                hist_resources.append(used_resources)
                hist_precisions.append(V_H)

        # --- Normalize accumulated loss ---
        if not deploy:
            if pars.cumulative_loss:
                denom = tf.cast(max(step_count, 1), prec)
                loss_diff_final = loss_diff_accum / denom
                loss_final = loss_accum / denom
            else:
                # Compute loss at the end of the loop (fixed k_ref)
                k_ref_fixed = tf.cast(
                    2.0 * pi / max(self.g_hi - self.g_lo, 1e-10), prec
                ) * tf.ones((self.bs,), dtype=prec)
                V_H_final = bank.holevo_variance(k_ref_fixed)  # (bs, 1)
                loss_mean = tf.reduce_mean(V_H_final)
                if pars.loss_logl_outcomes:
                    baseline = loss_mean if pars.baseline else tf.zeros((), dtype=prec)
                    loss_diff_final = tf.reduce_mean(
                        V_H_final + (
                            tf.stop_gradient(V_H_final) - tf.stop_gradient(baseline)
                        ) * sum_log_prob
                    )
                else:
                    loss_diff_final = loss_mean
                loss_final = loss_mean
            
            if debug:
                return loss_diff_final, loss_final, debug_records
            return loss_diff_final, loss_final

        # Deploy mode: return history
        ns = len(hist_inputs)
        if ns == 0:
            empty_i = tf.zeros((1, self.bs, self.input_size), dtype=prec)
            empty_c = tf.zeros((1, self.bs, self.phys_model.controls_size), dtype=prec)
            empty_r = tf.zeros((1, self.bs, 1), dtype=prec)
            empty_p = tf.zeros((1, self.bs, 1), dtype=prec)
        else:
            empty_i = tf.stack(hist_inputs, axis=0)
            empty_c = tf.stack(hist_controls_list, axis=0)
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
# Controller factory
# ---------------------------------------------------------------------------

def build_controller(
    input_size: int,
    phys_model: GravityStatelessPhysicalModel,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
) -> tf.keras.Model:
    """Build the MLP controller network.

    The network maps the compact bank summary vector to physical controls.
    The output is scaled from the tanh range ``[-1, 1]`` to the physical
    control ranges defined in ``phys_model.cfg``.

    Architecture::

        input (input_size,)
        → Dense(128, tanh)
        → Dense(128, tanh)
        → Dense(64, tanh)
        → Dense(3, tanh)      # (T_norm, Bp_norm, φ_norm) ∈ [-1, 1]³
        → scaling layer       # → (T_s, Bp_kTm, mw_phase_rad) in physical units

    Parameters
    ----------
    input_size : int
        Size of the NN input vector (``4 * top_k_modes + 5``).
    phys_model : GravityStatelessPhysicalModel
        Physical model providing control bounds.
    hidden_sizes : tuple of int, optional
        Sizes of hidden layers. Default: ``(128, 128, 64)``.

    Returns
    -------
    tf.keras.Model
        Compiled Keras sequential model with signature
        ``controls = controller(input_vec)``, where ``controls`` has shape
        ``(bs, 3)`` in physical units.
    """
    cfg = phys_model.cfg
    prec = cfg.prec

    # Control bounds from cfg
    T_min = float(cfg.T_range_s[0])
    T_max = float(cfg.T_range_s[1])
    Bp_min = float(cfg.Bp_range_kTm[0])
    Bp_max = float(cfg.Bp_range_kTm[1])
    phi_min = 0.0
    phi_max = 2.0 * pi

    # Midpoints and half-ranges for denormalization
    T_mid = 0.5 * (T_max + T_min)
    T_half = 0.5 * (T_max - T_min)
    Bp_mid = 0.5 * (Bp_max + Bp_min)
    Bp_half = 0.5 * (Bp_max - Bp_min)
    phi_mid = 0.5 * (phi_max + phi_min)
    phi_half = 0.5 * (phi_max - phi_min)

    dtype = tf.float32 if prec == "float32" else tf.float64

    class ControlScalingLayer(tf.keras.layers.Layer):
        """Scales NN tanh output to physical control ranges."""

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
    for h_size in hidden_sizes:
        x = tf.keras.layers.Dense(h_size, activation="tanh", dtype=dtype)(x)
    x = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    outputs = ControlScalingLayer(dtype=dtype)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gravity_controller")
    return model


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_gravity_multi_pf_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    bank_cfg: MultiPFBankConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
) -> Tuple[GravityMultiPFSimulation, MultiPFBank, tf.keras.Model]:
    """Convenience factory that builds and wires all components.

    Creates:
    * ``GravityStatelessPhysicalModel``
    * ``MultiPFBank``
    * MLP controller
    * ``GravityMultiPFSimulation``

    and resets the bank (initializes particles).

    Parameters
    ----------
    batchsize : int
        Batch size for training.
    cfg : GravimeterConfig
        Sensor configuration.
    bank_cfg : MultiPFBankConfig
        Bank hyperparameters.
    simpars : SimulationParameters
        qsensoropt training parameters.
    rangen : tf.random.Generator
        TensorFlow random number generator.

    Returns
    -------
    simulation : GravityMultiPFSimulation
    bank : MultiPFBank
    controller : tf.keras.Model
    """
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = MultiPFBank(phys_model=phys_model, cfg=bank_cfg)

    top_k = bank_cfg.top_k_modes
    input_size = 4 * top_k + 5
    controller = build_controller(input_size, phys_model)

    # Warm up controller (builds weights)
    dummy = tf.zeros(
        (batchsize, input_size),
        dtype=tf.float32 if cfg.prec == "float32" else tf.float64,
    )
    _ = controller(dummy)

    # Initialize bank BEFORE constructing the simulation,
    # because Simulation.__init__ reads from bank.pf_list[0].
    bank.reset(rangen)

    simulation = GravityMultiPFSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
    )

    return simulation, bank, controller
