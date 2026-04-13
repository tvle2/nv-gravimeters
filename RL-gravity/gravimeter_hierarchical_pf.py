"""Hierarchical Binary Multi-PF for the quantum gravity sensor.

Implements the **Hierarchical Binary Multi-PF** (HBMPF) architecture that
solves the phase-wrapping disambiguation problem in the levitated NV-center
nanodiamond gravimeter.

Architecture overview
---------------------
Standard MMAE with K=8 simultaneous modes fails because the marginal
likelihood spread ``ΔZ/Z ≈ 3–6%`` requires >47 measurements per level to
achieve 10:1 mode weight ratios, far exceeding a 16–32 step budget.

The hierarchical approach uses a binary tree:

1. Start with full g-range ``[g_lo, g_hi]``.
2. At each level L: split the current interval into 2 halves, run
   ``n_disambig`` measurements tuned to give ~1 fringe per half-interval.
3. Pick the winning half (higher mode weight).
4. Recurse into the winning half at level L+1 with doubled gain.

After ``n_levels`` levels: interval width = ``g_range / 2^n_levels``.
Remaining steps refine within the final interval using a single particle
filter with high-gain measurements.

Key physics
-----------
With K=2 the likelihood ratio ``Z_0/Z_1`` scales as::

    Z_ratio ≈ exp(vis * cos(k_g * Δg/2))

where ``Δg = W_L / 2`` is the half-width of the current interval.  For
``k_g ≈ 2π/W_L`` (one fringe per interval), ``Z_ratio ≈ 1.5``, so only
6–8 measurements per level are needed for unambiguous disambiguation.

Gain target at level L (current interval width ``W_L = g_range / 2^L``)::

    k_g_L ≈ 2π / W_L

The RL agent learns to select ``(T, B')`` that achieves this target.

Input vector (size = 6)
-----------------------
``[μ_g_norm, log_σ_norm, q_0, level_norm, step_norm, res_norm]``

* ``μ_g_norm`` — weighted mean of the active PF, normalized to [-1, 1].
* ``log_σ_norm`` — log-std encoding of active PF std.
* ``q_0`` — weight of mode-0 in the current 2-mode bank (0 if refining).
* ``level_norm`` — ``current_level / n_levels``.
* ``step_norm`` — step within current disambiguation level / n_disambig.
* ``res_norm`` — used_resources / max_resources normalized to [-1, 1].

References
----------
Belliardo et al. (2024). Physical Review A 109, 062609.
  https://doi.org/10.1103/PhysRevA.109.062609

Berry & Sanders (2009). Physical Review A 80, 052114.
  https://arxiv.org/abs/0907.0014
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
import os as _os
_os.environ.pop("TF_DETERMINISTIC_OPS", None)
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model import GravimeterConfig, GravityStatelessPhysicalModel


# ---------------------------------------------------------------------------
# Local qsensoropt loader
# ---------------------------------------------------------------------------

def _load_local_qsensoropt_module(module_name: str):
    """Load a qsensoropt module from the local workspace directory.

    Parameters
    ----------
    module_name : str
        Name of the module file (without ``.py`` extension).

    Returns
    -------
    module
        The loaded Python module.
    """
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
class HierarchicalPFConfig:
    """Configuration for the Hierarchical Binary Multi-PF bank.

    Parameters
    ----------
    n_particles : int
        Total number of particles used in any single particle filter
        at each level.  Default: ``512``.
    n_levels : int
        Number of binary splitting levels.  After ``n_levels`` levels the
        interval shrinks by a factor of ``2^n_levels``.  Default: ``4``.
    n_disambig_per_level : int
        Number of measurements used at each disambiguation level before
        picking the winning half-interval.  Default: ``7``.
    prec : str
        Floating-point precision: ``"float32"`` or ``"float64"``.
        Default: ``"float32"``.
    resample_threshold : float
        Effective sample size (ESS) fraction that triggers resampling.
        Default: ``0.5``.
    resample_alpha : float
        Ścibior soft-resampling mixing coefficient α.  Default: ``0.5``.
    resample_beta : float
        Liu–West jitter parameter β.  Default: ``0.98``.
    scibior_trick : bool
        Use the Ścibior–Wood differentiable resampling trick.  Must be
        ``True`` for gradient-based training.  Default: ``True``.
    trim : bool
        Trim particles to parameter bounds after resampling.
        Default: ``True``.
    """

    n_particles: int = 512
    n_levels: int = 4
    n_disambig_per_level: int = 7
    prec: str = "float32"
    resample_threshold: float = 0.5
    resample_alpha: float = 0.5
    resample_beta: float = 0.98
    scibior_trick: bool = True
    trim: bool = True


# ---------------------------------------------------------------------------
# Hierarchical PF Bank
# ---------------------------------------------------------------------------

class HierarchicalPFBank:
    """Hierarchical binary multi-particle-filter bank.

    Implements the two-mode binary-tree disambiguation strategy:

    * At each level L the bank holds **exactly 2 modes** (mode 0 = left half,
      mode 1 = right half of the current interval).
    * After ``n_disambig_per_level`` measurements, the winning mode (higher
      weight) determines the next interval, a new 2-mode split is created,
      and the level counter advances.
    * Once all ``n_levels`` disambiguation levels are exhausted, the bank
      switches to **refining mode**: a single PF on the final narrow interval
      receives all remaining measurements.

    The physical model's g-range is narrowed progressively:

    .. math::

        W_L = (g_{\\mathrm{hi}} - g_{\\mathrm{lo}}) / 2^L

    The RL agent must learn to set the gain ``k_g ≈ 2π / W_L`` at each
    level, which the ``level_norm`` input feature enables.

    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Shared physical model.  The g-parameter bounds define the full prior.
    cfg : HierarchicalPFConfig
        Bank hyperparameters.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        cfg: HierarchicalPFConfig,
    ) -> None:
        self.phys_model = phys_model
        self.cfg = cfg
        self.bs: int = phys_model.bs
        self.prec: str = cfg.prec

        g_lo_raw, g_hi_raw = phys_model.cfg.g_range
        self._g_lo_init: float = float(g_lo_raw)
        self._g_hi_init: float = float(g_hi_raw)

        # Current interval — narrowed as levels progress.  These are Python
        # floats updated at Python level (one value per batch element would
        # be more accurate but adds complexity; we use the global interval
        # consistent with the phys_model prior for the PF initialization).
        self.g_lo: float = self._g_lo_init
        self.g_hi: float = self._g_hi_init

        # Hierarchical state
        self.current_level: int = 0        # 0 … n_levels (inclusive)
        self._disambig_step: int = 0       # steps taken in current level
        self._refining: bool = False       # True after all levels exhausted

        # Two PFs for the current binary split (or one PF in refining mode)
        self._pf0: Optional[ParticleFilter] = None
        self._pf1: Optional[ParticleFilter] = None

        # Bank state: weights/particles for each mode
        self.weights0: Optional[Tensor] = None    # (bs, n_particles)
        self.particles0: Optional[Tensor] = None  # (bs, n_particles, d)
        self.weights1: Optional[Tensor] = None
        self.particles1: Optional[Tensor] = None

        # Mode weights: q[b, 0] + q[b, 1] = 1
        # q0[b] = P(true g ∈ left half | data so far) for batch element b
        self.mode_weights: Tensor = tf.ones(
            (self.bs, 2), dtype=self.prec
        ) * 0.5

        # Create a placeholder PF for the Simulation base class.
        # This is replaced on the first call to reset().
        self._placeholder_pf = self._make_pf()
        # Expose primary PF (mode 0) for Simulation API compatibility
        self.pf: ParticleFilter = self._placeholder_pf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_pf(
        self,
        g_lo: Optional[float] = None,
        g_hi: Optional[float] = None,
    ) -> ParticleFilter:
        """Create a new :class:`ParticleFilter` with custom g-bounds.

        The physical model's parameter list is shared (stateless model),
        but we create a fresh ``ParticleFilter`` instance so that its
        ``reset()`` will sample particles uniformly in ``[g_lo, g_hi]``
        rather than the full prior.

        Parameters
        ----------
        g_lo : float, optional
            Lower bound for g in this mode.  Defaults to the full prior.
        g_hi : float, optional
            Upper bound for g in this mode.  Defaults to the full prior.

        Returns
        -------
        ParticleFilter
        """
        cfg = self.cfg
        if g_lo is None:
            g_lo = self._g_lo_init
        if g_hi is None:
            g_hi = self._g_hi_init

        # We need a physical model whose Parameter has bounds [g_lo, g_hi].
        # The cleanest approach: temporarily patch the phys_model's parameter
        # list for the PF reset, then create a fresh PF.  Since qsensoropt's
        # ParticleFilter.reset() samples from phys_model.params[0].bounds,
        # we create a lightweight wrapper that overrides the bounds.
        narrowed_model = _NarrowedPhysicalModel(
            base_model=self.phys_model,
            g_lo=g_lo,
            g_hi=g_hi,
        )

        pf = ParticleFilter(
            num_particles=cfg.n_particles,
            phys_model=narrowed_model,
            resampling_allowed=True,
            resample_threshold=cfg.resample_threshold,
            alpha=cfg.resample_alpha,
            beta=cfg.resample_beta,
            scibior_trick=cfg.scibior_trick,
            trim=cfg.trim,
            prec=cfg.prec,
        )
        return pf

    def _init_mode_pair(
        self,
        rangen: tf.random.Generator,
    ) -> None:
        """Create two PFs that tile the current ``[g_lo, g_hi]`` interval.

        Mode 0 covers ``[g_lo, g_mid]``; mode 1 covers ``[g_mid, g_hi]``.

        Parameters
        ----------
        rangen : tf.random.Generator
        """
        g_mid = 0.5 * (self.g_lo + self.g_hi)

        self._pf0 = self._make_pf(g_lo=self.g_lo, g_hi=g_mid)
        self._pf1 = self._make_pf(g_lo=g_mid, g_hi=self.g_hi)

        self.weights0, self.particles0 = self._pf0.reset(rangen)
        self.weights1, self.particles1 = self._pf1.reset(rangen)

        # Equal mode weights at the start of each level
        self.mode_weights = tf.cast(
            tf.ones((self.bs, 2), dtype=self.prec) * 0.5, self.prec
        )

        # Sync primary PF to mode 0
        self.pf = self._pf0

    def _init_refine_mode(
        self,
        rangen: tf.random.Generator,
    ) -> None:
        """Create a single PF covering the final narrow interval.

        Called when all disambiguation levels are exhausted.

        Parameters
        ----------
        rangen : tf.random.Generator
        """
        pf = self._make_pf(g_lo=self.g_lo, g_hi=self.g_hi)
        w, p = pf.reset(rangen)

        self._pf0 = pf
        self._pf1 = None
        self.weights0 = w
        self.particles0 = p
        self.weights1 = None
        self.particles1 = None

        # Dummy mode weights — q_0 = 1 signals refining mode
        self.mode_weights = tf.cast(
            tf.concat(
                [
                    tf.ones((self.bs, 1), dtype=self.prec),
                    tf.zeros((self.bs, 1), dtype=self.prec),
                ],
                axis=1,
            ),
            self.prec,
        )

        self.pf = self._pf0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, rangen: tf.random.Generator) -> None:
        """Reset the bank to the start of a new estimation episode.

        Restores the full prior interval and begins at level 0 with a
        fresh 2-mode split.

        Parameters
        ----------
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        self.g_lo = self._g_lo_init
        self.g_hi = self._g_hi_init
        self.current_level = 0
        self._disambig_step = 0
        self._refining = False

        self._init_mode_pair(rangen)

    def apply_measurement(
        self,
        outcomes: Tensor,
        controls: Tensor,
        meas_step: Tensor,
        rangen: tf.random.Generator,
    ) -> None:
        """Apply one measurement to the bank's active particle filters.

        In disambiguation mode (levels 0 … n_levels-1):

        1. Compute per-particle likelihoods for both mode PFs.
        2. Update mode weights: ``q_k ← q_k * Z_k``, renormalize.
        3. Bayes-update within-mode weights.
        4. ESS-triggered resampling.
        5. Increment disambiguation step counter.
        6. If ``n_disambig_per_level`` steps reached: pick the winning half,
           narrow the interval, advance to next level (or switch to refining).

        In refining mode:

        1. Bayes-update the single active PF.
        2. ESS-triggered resampling.

        Parameters
        ----------
        outcomes : Tensor
            Shape ``(bs, outcomes_size)`` — binary measurement results.
        controls : Tensor
            Shape ``(bs, controls_size)`` — ``[T_s, Bp_kTm, mw_phase_rad]``.
        meas_step : Tensor
            Shape ``(bs, 1)`` — global step index.
        rangen : tf.random.Generator
        """
        prec = self.prec

        if self._refining:
            # --- Single-mode refinement ---
            self._bayes_update_single(
                pf=self._pf0,
                weights=self.weights0,
                particles=self.particles0,
                outcomes=outcomes,
                controls=controls,
                meas_step=meas_step,
                rangen=rangen,
            )
            return

        # --- Disambiguation mode: two modes ---
        cfg = self.cfg
        N = cfg.n_particles

        outcomes_broad0 = tf.broadcast_to(
            tf.expand_dims(outcomes, axis=1),
            (self.bs, N, self.phys_model.outcomes_size),
        )
        controls_broad0 = tf.broadcast_to(
            tf.expand_dims(controls, axis=1),
            (self.bs, N, self.phys_model.controls_size),
        )
        step_broad = tf.broadcast_to(
            tf.expand_dims(meas_step, axis=2),
            (self.bs, N, 1),
        )

        state0 = tf.zeros((self.bs, N, 0), dtype=prec)
        state1 = tf.zeros((self.bs, N, 0), dtype=prec)

        # Likelihoods for both modes
        prob0, _ = self.phys_model.wrapper_model(
            outcomes_broad0, controls_broad0,
            self.particles0, state0, step_broad,
            num_systems=N,
        )  # (bs, N)
        prob1, _ = self.phys_model.wrapper_model(
            tf.broadcast_to(
                tf.expand_dims(outcomes, axis=1),
                (self.bs, N, self.phys_model.outcomes_size),
            ),
            tf.broadcast_to(
                tf.expand_dims(controls, axis=1),
                (self.bs, N, self.phys_model.controls_size),
            ),
            self.particles1, state1,
            tf.broadcast_to(
                tf.expand_dims(meas_step, axis=2),
                (self.bs, N, 1),
            ),
            num_systems=N,
        )  # (bs, N)

        # Marginal likelihoods Z_k = Σ_j w_{k,j} p_{k,j}
        unnorm0 = self.weights0 * prob0  # (bs, N)
        unnorm1 = self.weights1 * prob1  # (bs, N)
        Z0 = tf.reduce_sum(unnorm0, axis=1)  # (bs,)
        Z1 = tf.reduce_sum(unnorm1, axis=1)  # (bs,)

        safe_Z0 = tf.maximum(Z0, tf.cast(1e-300, prec))
        safe_Z1 = tf.maximum(Z1, tf.cast(1e-300, prec))

        # Normalize within-mode weights
        norm_w0 = unnorm0 / tf.expand_dims(safe_Z0, axis=1)
        norm_w1 = unnorm1 / tf.expand_dims(safe_Z1, axis=1)

        # Update mode weights q_k ← q_k * Z_k, normalize
        q0 = self.mode_weights[:, 0]  # (bs,)
        q1 = self.mode_weights[:, 1]  # (bs,)
        new_q0_unnorm = q0 * Z0
        new_q1_unnorm = q1 * Z1
        Z_total = tf.maximum(
            new_q0_unnorm + new_q1_unnorm, tf.cast(1e-300, prec)
        )
        new_q0 = new_q0_unnorm / Z_total
        new_q1 = new_q1_unnorm / Z_total
        self.mode_weights = tf.stack([new_q0, new_q1], axis=1)  # (bs, 2)

        # Store updated within-mode weights
        self.weights0 = norm_w0
        self.weights1 = norm_w1

        # ESS-triggered resampling for each mode.
        # We use manual categorical + Liu-West jitter instead of
        # pf.full_resampling() to avoid qsensoropt's SVD codepath
        # which crashes for d=1 on deterministic-mode GPU kernels.
        self.weights0, self.particles0 = self._manual_resample_if_needed(
            self.weights0, self.particles0, self._pf0, rangen,
        )
        self.weights1, self.particles1 = self._manual_resample_if_needed(
            self.weights1, self.particles1, self._pf1, rangen,
        )

        # Count the step and check if we should advance to the next level
        self._disambig_step += 1
        if self._disambig_step >= cfg.n_disambig_per_level:
            self._advance_level(rangen)

    def _manual_resample_if_needed(
        self,
        weights: Tensor,
        particles: Tensor,
        pf: "ParticleFilter",
        rangen: tf.random.Generator,
    ) -> Tuple[Tensor, Tensor]:
        """ESS-gated resampling using categorical + Liu-West jitter.

        Avoids ``pf.full_resampling()`` which calls ``sqrt_hmatrix → SVD``,
        crashing for ``d=1`` on deterministic-GPU kernels.
        """
        N = tf.shape(weights)[1]
        prec = self.cfg.prec
        beta = tf.cast(self.cfg.resample_beta, prec)

        ess = 1.0 / tf.reduce_sum(tf.square(weights), axis=1)  # (bs,)
        threshold = tf.cast(pf.res_thres * tf.cast(N, prec), prec)
        needs = tf.reduce_any(ess < threshold)

        if not needs:
            return weights, particles

        # Multinomial resampling
        log_w = tf.math.log(tf.maximum(weights, tf.cast(1e-30, prec)))
        seed = rangen.make_seeds(2)[:, 0]
        ancestors = tf.random.stateless_categorical(
            log_w, N, seed=seed, dtype=tf.int32,
        )  # (bs, N)
        batch_idx = tf.broadcast_to(
            tf.range(self.bs)[:, None], tf.shape(ancestors),
        )
        idx = tf.stack([batch_idx, ancestors], axis=-1)
        p_new = tf.gather_nd(particles, idx)  # (bs, N, d)

        # Liu-West jitter (manual, avoids SVD)
        p_mean = tf.reduce_mean(p_new, axis=1, keepdims=True)  # (bs,1,d)
        p_std = tf.math.reduce_std(p_new, axis=1, keepdims=True)
        jitter = tf.sqrt(1.0 - beta ** 2) * p_std
        noise = rangen.normal(tf.shape(p_new), dtype=prec) * jitter
        p_new = beta * p_new + (1.0 - beta) * p_mean + noise

        # Clip to physical bounds
        g_param = pf.phys_model.params[0]  # g parameter
        p_new = tf.clip_by_value(
            p_new,
            tf.cast(g_param.bounds[0], prec),
            tf.cast(g_param.bounds[1], prec),
        )

        w_new = tf.ones_like(weights) / tf.cast(N, prec)
        return w_new, p_new

    def _bayes_update_single(
        self,
        pf: ParticleFilter,
        weights: Tensor,
        particles: Tensor,
        outcomes: Tensor,
        controls: Tensor,
        meas_step: Tensor,
        rangen: tf.random.Generator,
    ) -> None:
        """Bayes update for the single refining-mode PF.

        Parameters
        ----------
        pf : ParticleFilter
        weights : Tensor, shape ``(bs, N)``
        particles : Tensor, shape ``(bs, N, d)``
        outcomes : Tensor, shape ``(bs, outcomes_size)``
        controls : Tensor, shape ``(bs, controls_size)``
        meas_step : Tensor, shape ``(bs, 1)``
        rangen : tf.random.Generator
        """
        prec = self.prec
        N = pf.np

        outcomes_b = tf.broadcast_to(
            tf.expand_dims(outcomes, axis=1),
            (self.bs, N, self.phys_model.outcomes_size),
        )
        controls_b = tf.broadcast_to(
            tf.expand_dims(controls, axis=1),
            (self.bs, N, self.phys_model.controls_size),
        )
        step_b = tf.broadcast_to(
            tf.expand_dims(meas_step, axis=2),
            (self.bs, N, 1),
        )
        state = tf.zeros((self.bs, N, 0), dtype=prec)

        prob, _ = self.phys_model.wrapper_model(
            outcomes_b, controls_b, particles, state, step_b, num_systems=N
        )  # (bs, N)

        unnorm_w = weights * prob
        Z = tf.reduce_sum(unnorm_w, axis=1, keepdims=True)
        safe_Z = tf.maximum(Z, tf.cast(1e-300, prec))
        new_weights = unnorm_w / safe_Z

        new_w, new_p = self._manual_resample_if_needed(
            new_weights, particles, pf, rangen,
        )
        # Update in-place
        self.weights0 = new_w
        self.particles0 = new_p

    def _advance_level(self, rangen: tf.random.Generator) -> None:
        """Pick the winning mode and advance to the next level.

        Selects the half-interval with the highest **batch-averaged** mode
        weight.  Per-element selection would be more accurate but requires
        variable-sized particle allocations; the batch-average approach is
        a practical approximation that works well in training.

        After picking the winner:

        * Updates ``g_lo``, ``g_hi`` to the winning half.
        * Increments ``current_level``.
        * Either creates a new 2-mode split (next disambiguation level) or
          switches to refining mode.

        Parameters
        ----------
        rangen : tf.random.Generator
        """
        # Batch-averaged mode weights to decide globally which half to keep
        q0_mean = float(tf.reduce_mean(self.mode_weights[:, 0]).numpy())
        q1_mean = float(tf.reduce_mean(self.mode_weights[:, 1]).numpy())

        g_mid = 0.5 * (self.g_lo + self.g_hi)
        if q0_mean >= q1_mean:
            # Mode 0 wins: interval shrinks to [g_lo, g_mid]
            self.g_hi = g_mid
        else:
            # Mode 1 wins: interval shrinks to [g_mid, g_hi]
            self.g_lo = g_mid

        self.current_level += 1
        self._disambig_step = 0

        if self.current_level >= self.cfg.n_levels:
            # Done with all levels — switch to refinement
            self._refining = True
            self._init_refine_mode(rangen)
        else:
            # Create a new 2-mode split for the next level
            self._init_mode_pair(rangen)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def marginal_mean_and_var(self) -> Tuple[Tensor, Tensor]:
        """Marginal mean and variance of g from the active particle filter(s).

        In disambiguation mode: mixture mean and variance over the two modes.
        In refining mode: mean and variance from the single active PF.

        Returns
        -------
        g_mean : Tensor, shape ``(bs,)``
        g_var : Tensor, shape ``(bs,)``
        """
        prec = self.prec

        if self._refining:
            mean = self._pf0.compute_mean(
                self.weights0, self.particles0
            )[:, 0]  # (bs,)
            cov = self._pf0.compute_covariance(
                self.weights0, self.particles0
            )[:, 0, 0]  # (bs,)
            return mean, tf.maximum(cov, tf.cast(0.0, prec))

        # Two-mode mixture
        q0 = self.mode_weights[:, 0]  # (bs,)
        q1 = self.mode_weights[:, 1]  # (bs,)

        mean0 = self._pf0.compute_mean(
            self.weights0, self.particles0
        )[:, 0]  # (bs,)
        mean1 = self._pf1.compute_mean(
            self.weights1, self.particles1
        )[:, 0]  # (bs,)

        cov0 = self._pf0.compute_covariance(
            self.weights0, self.particles0
        )[:, 0, 0]  # (bs,)
        cov1 = self._pf1.compute_covariance(
            self.weights1, self.particles1
        )[:, 0, 0]  # (bs,)

        g_mean = q0 * mean0 + q1 * mean1  # (bs,)
        # Var(g) = E[g²] - E[g]² = Σ_k q_k(σ_k² + μ_k²) - μ²
        second_moment = q0 * (cov0 + tf.square(mean0)) + q1 * (
            cov1 + tf.square(mean1)
        )
        g_var = tf.maximum(
            second_moment - tf.square(g_mean), tf.cast(0.0, prec)
        )
        return g_mean, g_var

    def map_mode_mean(self) -> Tensor:
        """Mean of the higher-weight mode's posterior.

        In disambiguation mode: returns the mean of whichever mode has the
        higher batch-averaged weight.  In refining mode: the single PF mean.

        Returns
        -------
        Tensor, shape ``(bs,)``
        """
        if self._refining:
            return self._pf0.compute_mean(
                self.weights0, self.particles0
            )[:, 0]  # (bs,)

        q0 = self.mode_weights[:, 0]  # (bs,)
        q1 = self.mode_weights[:, 1]  # (bs,)
        mean0 = self._pf0.compute_mean(
            self.weights0, self.particles0
        )[:, 0]  # (bs,)
        mean1 = self._pf1.compute_mean(
            self.weights1, self.particles1
        )[:, 0]  # (bs,)

        # Per-batch selection: use mode 0 mean where q0 >= q1
        return tf.where(q0 >= q1, mean0, mean1)  # (bs,)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def q0_mean(self) -> float:
        """Batch-averaged weight of mode 0 in the current level.

        Returns ``1.0`` in refining mode (no ambiguity).
        """
        if self._refining:
            return 1.0
        return float(tf.reduce_mean(self.mode_weights[:, 0]).numpy())

    @property
    def interval_width(self) -> float:
        """Width of the current active interval (m/s²)."""
        return self.g_hi - self.g_lo

    @property
    def target_k_g(self) -> float:
        """Target gain ``k_g ≈ 2π / W_L`` for the current level (rad·s²/m)."""
        W = max(self.interval_width, 1e-10)
        return 2.0 * pi / W


# ---------------------------------------------------------------------------
# NarrowedPhysicalModel — lightweight wrapper that shifts parameter bounds
# ---------------------------------------------------------------------------

class _NarrowedPhysicalModel:
    """Proxy around :class:`GravityStatelessPhysicalModel` that overrides
    the parameter ``bounds`` so that a :class:`ParticleFilter` initialised
    with this proxy will sample particles uniformly from ``[g_lo, g_hi]``.

    All other attributes and methods are forwarded unchanged to the base
    model.  The proxy is only used during :py:meth:`ParticleFilter.reset`;
    the actual likelihood computations continue to use the *base* model.

    Parameters
    ----------
    base_model : GravityStatelessPhysicalModel
        The underlying physics model.
    g_lo : float
        New lower bound for g.
    g_hi : float
        New upper bound for g.
    """

    def __init__(
        self,
        base_model: GravityStatelessPhysicalModel,
        g_lo: float,
        g_hi: float,
    ) -> None:
        self._base = base_model
        self._g_lo = g_lo
        self._g_hi = g_hi

        # Build a modified parameter list with narrowed bounds.
        # We only need to present the 'params' list, which is what
        # ParticleFilter.reset() iterates over.
        try:
            from qsensoropt.parameter import Parameter as _Param  # type: ignore
        except Exception:
            _Param = _load_local_qsensoropt_module("parameter").Parameter

        # Shallow-clone the parameter list, replacing the g-bounds.
        # IMPORTANT: set prec and bs on the new Parameter to match the base
        # model so that Parameter.reset() generates tensors of the correct
        # dtype and batch size.
        import copy
        original_params = base_model.params
        new_params = []
        for i, p in enumerate(original_params):
            if i == 0:  # g parameter
                np_param = _Param(bounds=(g_lo, g_hi), name="g")
                np_param.prec = base_model.prec
                np_param.bs = base_model.bs
                new_params.append(np_param)
            else:
                new_params.append(copy.copy(p))
        self.params = new_params

    def __getattr__(self, name: str):
        # Delegate everything not explicitly defined to the base model
        return getattr(self._base, name)


# ---------------------------------------------------------------------------
# Gravity Hierarchical Simulation
# ---------------------------------------------------------------------------

class GravityHierarchicalSimulation(StatelessSimulation):
    """Simulation integrating the Hierarchical PF Bank with qsensoropt.

    Subclasses :class:`~.StatelessSimulation` and overrides:

    * :meth:`generate_input` — builds a 6-element NN input vector from
      the current bank state.
    * :meth:`loss_function` — returns the MSE between the MAP estimate
      and the true g.
    * :meth:`execute` — Python-level measurement loop that routes each
      outcome through :meth:`HierarchicalPFBank.apply_measurement`.

    Input vector (size = 6)
    -----------------------
    1. ``μ_g_norm`` — marginal mean of g, normalized to [-1, 1] over
       the current active interval.
    2. ``log_σ_norm`` — log-std encoding:
       ``-2/10 * log10(σ) - 1`` clipped to [-1, 1].
    3. ``q_0`` — weight of mode-0 in the current 2-mode bank, or 1.0 in
       refining mode.  Encodes the current disambiguation progress.
    4. ``level_norm`` — ``current_level / n_levels`` ∈ [0, 1].
    5. ``step_norm`` — ``disambig_step / n_disambig_per_level * 2 - 1``
       ∈ [-1, 1].
    6. ``res_norm`` — ``used_resources / max_resources * 2 - 1`` ∈ [-1, 1].

    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Shared physical model.
    bank : HierarchicalPFBank
        The hierarchical PF bank.
    controller : callable
        Neural network mapping input vector to controls.
    simpars : SimulationParameters
        qsensoropt simulation parameters.
    bank_cfg : HierarchicalPFConfig
        Bank configuration.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        bank: HierarchicalPFBank,
        controller,
        simpars: SimulationParameters,
        bank_cfg: HierarchicalPFConfig,
    ) -> None:
        input_size = 6
        input_name = [
            "mu_g_norm",
            "log_sigma_norm",
            "q_0",
            "level_norm",
            "step_norm",
            "res_norm",
        ]

        # Pass bank's primary PF (mode 0) as the canonical PF for the
        # Simulation base class (needed for correct __init__).
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
        self.g_lo = float(phys_model.cfg.g_range[0])
        self.g_hi = float(phys_model.cfg.g_range[1])

    # ------------------------------------------------------------------
    # generate_input
    # ------------------------------------------------------------------

    def generate_input(
        self,
        weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen,
    ) -> Tensor:
        """Build the 6-element NN input vector from the current bank state.

        The input encodes the current posterior estimate, the disambiguation
        progress (mode weight ``q_0``), and the hierarchical level information
        so that the agent can learn to select the appropriate gain at each
        level.

        Parameters
        ----------
        weights : Tensor, shape ``(bs, pf.np)``
            Primary PF weights (used only in single-mode step 0).
        particles : Tensor, shape ``(bs, pf.np, d)``
            Primary PF particles (used only in single-mode step 0).
        meas_step : Tensor, shape ``(bs, 1)``
            Global step index.
        used_resources : Tensor, shape ``(bs, 1)``
            Accumulated resource usage in seconds.
        rangen : tf.random.Generator

        Returns
        -------
        Tensor, shape ``(bs, 6)``
        """
        bank = self.bank
        cfg = self.bank_cfg
        prec = self.simpars.prec
        simpars = self.simpars
        bs = self.bs

        # 1. Marginal mean and variance of g
        g_mean, g_var = bank.marginal_mean_and_var()  # (bs,), (bs,)
        g_std = tf.sqrt(tf.maximum(g_var, tf.cast(1e-20, prec)))  # (bs,)

        # --- Feature 1: μ_g normalized over full prior [g_lo, g_hi] ---
        g_range = max(self.g_hi - self.g_lo, 1e-10)
        mu_norm = tf.cast(
            2.0 * (g_mean - self.g_lo) / g_range - 1.0, prec
        )
        mu_norm = tf.clip_by_value(mu_norm, -1.0, 1.0)  # (bs,)

        # --- Feature 2: log-std encoding ---
        g_std_np = g_std.numpy().astype(
            np.float32 if prec == "float32" else np.float64
        )
        g_std_safe = np.maximum(g_std_np, 1e-8 if prec == "float32" else 1e-10)
        log_sigma = tf.cast(
            -2.0 / 10.0 * np.log10(g_std_safe) - 1.0, prec
        )
        log_sigma = tf.clip_by_value(log_sigma, -1.0, 1.0)  # (bs,)

        # --- Feature 3: q_0 — mode-0 weight ---
        q0 = bank.mode_weights[:, 0]  # (bs,)
        q0 = tf.cast(q0, prec)
        q0 = tf.clip_by_value(q0, 0.0, 1.0)

        # --- Feature 4: level_norm ---
        level_norm = tf.cast(
            float(bank.current_level) / float(cfg.n_levels), prec
        ) * tf.ones((bs,), dtype=prec)
        level_norm = tf.clip_by_value(level_norm, 0.0, 1.0)

        # --- Feature 5: step within current level (normalized to [-1, 1]) ---
        n_disambig = float(cfg.n_disambig_per_level)
        step_within = float(bank._disambig_step)
        step_norm_val = 2.0 * step_within / max(n_disambig, 1.0) - 1.0
        step_norm = tf.cast(step_norm_val, prec) * tf.ones((bs,), dtype=prec)
        step_norm = tf.clip_by_value(step_norm, -1.0, 1.0)

        # --- Feature 6: resource utilization ---
        res_norm = (
            2.0 * used_resources[:, 0]
            / tf.cast(simpars.max_resources, prec)
            - 1.0
        )
        res_norm = tf.clip_by_value(res_norm, -1.0, 1.0)

        input_vec = tf.stack(
            [mu_norm, log_sigma, q0, level_norm, step_norm, res_norm],
            axis=1,
        )  # (bs, 6)
        return input_vec

    # ------------------------------------------------------------------
    # loss_function
    # ------------------------------------------------------------------

    def loss_function(
        self,
        weights: Tensor,
        particles: Tensor,
        true_values: Tensor,
        used_resources: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """Mean squared error between the MAP estimate and true g.

        MSE is the correct loss for a unimodal posterior (which is what
        we have after the hierarchical disambiguation resolves ambiguity).
        It provides a simple, stable gradient signal and is proven to work
        for training particle-filter-based controllers.

        Parameters
        ----------
        weights : Tensor, shape ``(bs, pf.np)``
            (Kept for API compatibility.)
        particles : Tensor, shape ``(bs, pf.np, d)``
            (Kept for API compatibility.)
        true_values : Tensor, shape ``(bs, 1, d)``
            True gravitational accelerations.
        used_resources : Tensor, shape ``(bs, 1)``
        meas_step : Tensor, shape ``(bs, 1)``

        Returns
        -------
        Tensor, shape ``(bs, 1)``
            Per-episode MSE in (m/s²)².
        """
        prec = self.simpars.prec
        g_true = true_values[:, 0, 0]  # (bs,)
        g_hat = self.bank.map_mode_mean()  # (bs,)
        mse = tf.square(g_hat - g_true)   # (bs,)
        return tf.expand_dims(tf.cast(mse, prec), axis=1)  # (bs, 1)

    # ------------------------------------------------------------------
    # execute  (Python-level measurement loop)
    # ------------------------------------------------------------------

    def execute(
        self,
        rangen: tf.random.Generator,
        deploy: bool = False,
    ):
        """Hierarchical bank-aware measurement loop.

        Replaces the qsensoropt ``tf.while_loop`` with a Python-level loop
        so that the bank's Python state (level transitions, mode creation /
        pruning) can be updated at every step without graph-mode constraints.

        At each step:

        1. :meth:`generate_input` — builds 6-element NN input from bank state.
        2. ``controller(input)`` — outputs controls ``[T_s, Bp, φ]``.
        3. ``phys_model.wrapper_perform_measurement`` — samples binary outcome.
        4. :meth:`HierarchicalPFBank.apply_measurement` — Bayes-updates all
           active mode PFs, updates mode weights, and triggers level
           transitions when ``n_disambig`` steps are reached.
        5. Loss computation (MSE).

        For training, call inside ``tf.GradientTape``.  Gradients flow
        through steps 2, 4, and 5 via REINFORCE + model-aware gradients.

        Parameters
        ----------
        rangen : tf.random.Generator
        deploy : bool, optional
            If ``True``, returns history tensors for diagnostics.  If
            ``False`` (default), returns ``(loss_diff, loss)`` for training.

        Returns
        -------
        If ``deploy=False``:

        loss_diff : Tensor, shape ``()``
            Differentiable (REINFORCE-augmented) MSE loss.
        loss : Tensor, shape ``()``
            Pure MSE loss (for logging).

        If ``deploy=True``:

        true_values : Tensor, shape ``(bs, 1, d)``
        history_inputs : Tensor, shape ``(bs*T, input_size)``
        history_controls : Tensor, shape ``(bs*T, controls_size)``
        history_resources : Tensor, shape ``(bs*T, 1)``
        history_precision : Tensor, shape ``(bs*T, 1)``
        """
        pars = self.simpars
        prec = pars.prec
        bank = self.bank

        # --- Initialize episode ---
        bank.reset(rangen)
        # Sync primary PF with bank mode 0
        weights = bank.weights0
        particles = bank.particles0

        # Draw true parameter values
        true_values = self.phys_model.true_values(rangen)  # (bs, 1, d)
        true_state = self.phys_model.wrapper_initialize_state(true_values, 1)

        used_resources = tf.zeros((self.bs, 1), dtype=prec)
        continue_flag = tf.ones((self.bs, 1), dtype="bool")
        outcomes = tf.zeros(
            (self.bs, self.phys_model.outcomes_size),
            dtype=self.phys_model.prec,
        )
        meas_step = tf.zeros((self.bs, 1), dtype="int32")
        sum_log_prob = tf.zeros((self.bs, 1), dtype=prec)

        loss_diff_accum = tf.zeros((), dtype=prec)
        loss_accum = tf.zeros((), dtype=prec)
        step_count = 0

        # Deploy-mode history buffers
        if deploy:
            hist_inputs: List[Tensor] = []
            hist_controls_list: List[Tensor] = []
            hist_resources: List[Tensor] = []
            hist_precisions: List[Tensor] = []

        # --- Python-level measurement loop ---
        for _i in range(pars.num_steps):
            # Stop if enough batch elements have exhausted resources
            num_finished = int(
                tf.math.count_nonzero(tf.logical_not(continue_flag)).numpy()
            )
            if num_finished >= pars.resources_fraction * self.bs:
                break

            # Sync primary PF weights/particles from bank's current active PF
            weights = bank.weights0
            particles = bank.particles0
            # Keep self.pf.np consistent with current PF
            self.pf = bank.pf

            # 1. Build NN input
            input_strategy = self.generate_input(
                weights,
                particles,
                tf.cast(meas_step, prec),
                used_resources,
                rangen,
            )

            # 2. Compute controls
            cond_input = (
                tf.stop_gradient(input_strategy)
                if pars.stop_gradient_input
                else input_strategy
            )
            controls = self.control_strategy(cond_input)  # (bs, controls_size)

            # 3. Update resource counter
            new_used_resources = self.phys_model.wrapper_count_resources(
                used_resources,
                outcomes,
                controls,
                true_values,
                true_state,
                meas_step,
            )
            continue_flag = tf.math.less_equal(
                new_used_resources,
                pars.max_resources * tf.ones((self.bs, 1), dtype=prec),
            )
            used_resources = tf.where(
                continue_flag, new_used_resources, used_resources
            )

            # 4. Perform measurement on the true system
            outcomes_raw, log_prob, post_true_state = (
                self.phys_model.wrapper_perform_measurement(
                    tf.expand_dims(controls, axis=1),
                    true_values,
                    true_state,
                    tf.expand_dims(meas_step, axis=1),
                    rangen,
                )
            )
            outcomes = outcomes_raw[:, 0, :]  # (bs, outcomes_size)

            # Accumulate log-probability for REINFORCE
            if pars.loss_logl_outcomes:
                sum_log_prob = tf.where(
                    continue_flag, sum_log_prob + log_prob, sum_log_prob
                )

            # 5. Apply measurement to the bank
            bank.apply_measurement(
                outcomes,
                controls,
                meas_step,
                rangen,
            )

            # Compute per-step MSE for loss accumulation
            g_true = true_values[:, 0, 0]   # (bs,)
            g_hat = bank.map_mode_mean()     # (bs,)
            mse = tf.expand_dims(
                tf.cast(tf.square(g_hat - g_true), prec), axis=1
            )  # (bs, 1)

            # Advance step counter (only for active estimations)
            meas_step = tf.where(continue_flag, meas_step + 1, meas_step)
            step_count += 1

            # 6. Accumulate loss (cumulative mode)
            if pars.cumulative_loss and not deploy:
                active = tf.cast(continue_flag, prec)
                n_active = tf.maximum(
                    tf.reduce_sum(active), tf.cast(1.0, prec)
                )
                loss_diff_accum = loss_diff_accum + tf.reduce_sum(
                    tf.where(continue_flag, mse, tf.zeros_like(mse))
                ) / n_active
                loss_accum = loss_accum + tf.reduce_mean(
                    tf.where(continue_flag, mse, tf.zeros_like(mse))
                )

            if deploy:
                hist_inputs.append(input_strategy)
                hist_controls_list.append(controls)
                hist_resources.append(used_resources)
                hist_precisions.append(mse)

        # --- Normalize and return ---
        if not deploy:
            if pars.cumulative_loss:
                denom = tf.cast(max(step_count, 1), prec)
                loss_diff_final = loss_diff_accum / denom
                loss_final = loss_accum / denom
            else:
                # Terminal loss: MSE at end of episode
                g_true = true_values[:, 0, 0]  # (bs,)
                g_hat = bank.map_mode_mean()    # (bs,)
                mse_final = tf.expand_dims(
                    tf.cast(tf.square(g_hat - g_true), prec), axis=1
                )  # (bs, 1)
                loss_mean = tf.reduce_mean(mse_final)
                if pars.loss_logl_outcomes:
                    baseline = (
                        loss_mean if pars.baseline else tf.zeros((), dtype=prec)
                    )
                    loss_diff_final = tf.reduce_mean(
                        mse_final
                        + (
                            tf.stop_gradient(mse_final)
                            - tf.stop_gradient(baseline)
                        )
                        * sum_log_prob
                    )
                else:
                    loss_diff_final = loss_mean
                loss_final = loss_mean
            return loss_diff_final, loss_final

        # Deploy mode: stack and return history
        ns = len(hist_inputs)
        if ns == 0:
            empty_i = tf.zeros((1, self.bs, self.input_size), dtype=prec)
            empty_c = tf.zeros(
                (1, self.bs, self.phys_model.controls_size), dtype=prec
            )
            empty_r = tf.zeros((1, self.bs, 1), dtype=prec)
            empty_p = tf.zeros((1, self.bs, 1), dtype=prec)
        else:
            empty_i = tf.stack(hist_inputs, axis=0)       # (T, bs, 6)
            empty_c = tf.stack(hist_controls_list, axis=0)  # (T, bs, 3)
            empty_r = tf.stack(hist_resources, axis=0)    # (T, bs, 1)
            empty_p = tf.stack(hist_precisions, axis=0)   # (T, bs, 1)

        return (
            true_values,
            tf.reshape(empty_i, (self.bs * ns, self.input_size)),
            tf.reshape(empty_c, (self.bs * ns, self.phys_model.controls_size)),
            tf.reshape(empty_r, (self.bs * ns, 1)),
            tf.reshape(empty_p, (self.bs * ns, 1)),
        )


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def build_controller(
    phys_model: GravityStatelessPhysicalModel,
    input_size: int = 6,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
) -> tf.keras.Model:
    """Build the MLP controller network for the hierarchical PF.

    Maps the 6-element bank summary vector to physical controls
    ``[T_s (s), Bp_kTm (kT/m), mw_phase_rad (rad)]``.

    Architecture::

        input (6,)
        → Dense(128, tanh)
        → Dense(128, tanh)
        → Dense(64, tanh)
        → Dense(3, tanh)        # raw output ∈ [-1, 1]³
        → ControlScalingLayer   # → physical units

    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Used to extract control bounds from ``phys_model.cfg``.
    input_size : int, optional
        NN input dimension.  Default: ``6``.
    hidden_sizes : tuple of int, optional
        Hidden layer sizes.  Default: ``(128, 128, 64)``.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with signature
        ``controls = controller(input_vec)`` where ``controls`` has shape
        ``(bs, 3)`` in physical units.
    """
    cfg = phys_model.cfg
    prec = cfg.prec

    T_min = float(cfg.T_range_s[0])
    T_max = float(cfg.T_range_s[1])
    Bp_min = float(cfg.Bp_range_kTm[0])
    Bp_max = float(cfg.Bp_range_kTm[1])
    phi_min = 0.0
    phi_max = 2.0 * pi

    T_mid = 0.5 * (T_max + T_min)
    T_half = 0.5 * (T_max - T_min)
    Bp_mid = 0.5 * (Bp_max + Bp_min)
    Bp_half = 0.5 * (Bp_max - Bp_min)
    phi_mid = 0.5 * (phi_max + phi_min)
    phi_half = 0.5 * (phi_max - phi_min)

    dtype = tf.float32 if prec == "float32" else tf.float64

    class ControlScalingLayer(tf.keras.layers.Layer):
        """Scale tanh NN output to physical control ranges."""

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

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="hierarchical_gravity_controller",
    )
    return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_hierarchical_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    bank_cfg: HierarchicalPFConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
) -> Tuple[GravityHierarchicalSimulation, HierarchicalPFBank, tf.keras.Model]:
    """Build and wire all components of the hierarchical PF simulation.

    Creates:

    * :class:`GravityStatelessPhysicalModel` — gravimeter physics.
    * :class:`HierarchicalPFBank` — the hierarchical binary bank.
    * MLP controller — maps 6-element input to 3 physical controls.
    * :class:`GravityHierarchicalSimulation` — the simulation wrapper.

    Parameters
    ----------
    batchsize : int
        Number of parallel estimation episodes.
    cfg : GravimeterConfig
        Sensor configuration.
    bank_cfg : HierarchicalPFConfig
        Bank hyperparameters.
    simpars : SimulationParameters
        qsensoropt training parameters.
    rangen : tf.random.Generator
        TensorFlow random number generator.

    Returns
    -------
    simulation : GravityHierarchicalSimulation
    bank : HierarchicalPFBank
    controller : tf.keras.Model
    """
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)
    bank = HierarchicalPFBank(phys_model=phys_model, cfg=bank_cfg)

    input_size = 6
    controller = build_controller(phys_model, input_size=input_size)

    # Warm up controller to build its weights before training
    dummy = tf.zeros(
        (batchsize, input_size),
        dtype=tf.float32 if cfg.prec == "float32" else tf.float64,
    )
    _ = controller(dummy)

    # Initialize bank BEFORE constructing the simulation so that
    # bank.pf is a real (not placeholder) ParticleFilter when
    # Simulation.__init__ reads it.
    bank.reset(rangen)

    simulation = GravityHierarchicalSimulation(
        phys_model=phys_model,
        bank=bank,
        controller=controller,
        simpars=simpars,
        bank_cfg=bank_cfg,
    )

    return simulation, bank, controller
