# gravimeter_gm_bank.py
"""Gaussian-mixture posterior bank for gravity sensing with an NV-center.

Implements the Belliardo SI §B.5 explicit recommendation: rather than
maintain K separate particle filters (one per mode of a multimodal
posterior), maintain a **Gaussian mixture**

    p(g | y_{1..t}) ≈ Σ_k q_k(t) · N(g; μ_k(t), σ_k(t)^2)

with closed-form, fully differentiable Bayesian updates under the
cosine readout model

    p(y=+1 | g, x) = 1/2 [1 + A(x) · cos(k_g(x) · g + φ(x))]

The within-mode Bayes update is performed by *moment matching* against
the unnormalised posterior N(g; μ, σ²)·p(y|g), which has a closed form
because all moments of N(μ,σ²) under cos/sin can be evaluated
analytically:

    E[cos(kX+φ)]   = exp(-k²σ²/2) · cos(kμ+φ)
    E[X·cos(kX+φ)] = exp(-k²σ²/2) · [μ·cos(kμ+φ) - kσ²·sin(kμ+φ)]
    E[X²·cos(kX+φ)] = exp(-k²σ²/2) ·
        [(μ²+σ²-k²σ⁴)·cos(kμ+φ) - 2kμσ²·sin(kμ+φ)]

The mode-marginal evidence Z_k = ∫ p(y|g)·N(g;μ_k,σ_k²) dg is also closed
form and drives the q_k update by Bayes:

    q_k(t+1) ∝ q_k(t) · Z_k(y_t).

Mixture mean estimator
----------------------
The deployed estimator is the **mixture mean**

    ĝ = Σ_k q_k μ_k

which is the optimal Bayes estimator under squared-error loss. The
Bayes risk decomposes cleanly:

    E[(ĝ - g_true)²]  →  trained by REINFORCE on the prior π(g)

and the posterior variance

    Var(g | y) = Σ_k q_k σ_k² + Σ_k q_k (μ_k - ĝ)²

is the irreducible part that the controller drives down by sharpening
modes and concentrating q on the right one.

Key properties
--------------
* Fully differentiable: no resampling step, no stochastic mode reset.
* Mode-collapse safe: a numerical floor σ_min keeps modes from
  collapsing to delta functions (which would cause k²σ²→0, then a
  single perfect outcome makes Z_k undefined).
* Mode-revival: optional periodic uniform reset of the *lowest-q* modes
  to break local minima where the bank has prematurely collapsed onto a
  wrong mode.
* O(B·K) memory and compute (vs O(B·K·N) for per-mode PFs).

"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from gravimeter_model_complete import GravityStatelessPhysicalModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianMixtureBankConfig:
    """Configuration for the Gaussian-mixture posterior bank."""

    K: int = 64
    """Number of mixture components."""

    sigma_min_fraction: float = 1e-4
    """Minimum allowed σ_k as a fraction of the prior width.
    Prevents mode collapse to a δ-function (which would make k²σ²→0
    and the update ill-conditioned at high k_g)."""

    sigma_max_fraction: float = 1.0
    """Maximum allowed σ_k as a fraction of the prior width.
    Soft cap to keep moment-matching well-behaved when an update would
    expand a mode beyond the prior (which would happen if a likelihood
    happens to push variance up; under the ADF approximation, this is
    rare but possible numerically). 1.0 = mode can be as wide as the
    prior; values >1 produce "ghost modes" that span outside the prior
    and act as a uniform background, pulling ĝ toward the prior centre."""

    init_sigma_fraction: float = 0.5
    """Initial σ_k = (init_sigma_fraction)·(mode_width).
    0.5 = std-dev half the inter-mode spacing; together with uniform
    means this gives a coarse cover of the prior."""

    # NEW: regularization terms used by the loss path
    posterior_gain_width_multiplier: float = 4.0
    """Width multiplier for the posterior-safe gain cap used by the
    controller schedule layer. k_g <= 2π/(mult · σ_post)."""

    min_gain_fringe_fraction: float = 0.25
    """Lower bound on k_g as a fraction of one fringe across the prior."""

    log_sigma_frac_bounds: Tuple[float, float] = (-6.0, 0.0)
    """Bounds on log10(σ_post / prior_width) for normalising the
    posterior-width feature fed to the controller."""

    # Mode-revival: periodically reset the lowest-q modes to break local
    # minima where the bank has prematurely collapsed onto the wrong mode.
    revive_min_q: float = 0.0
    """If > 0: any mode with q_k < revive_min_q has its (μ_k, σ_k)
    reset to a fresh uniform sub-prior at every step. 0 = disabled."""

    @property
    def k_max(self) -> int:
        """Alias for backwards compatibility with the PF-bank API."""
        return self.K


# ---------------------------------------------------------------------------
# Gaussian-mixture bank
# ---------------------------------------------------------------------------

class GaussianMixtureBank:
    """Gaussian-mixture posterior for a single scalar parameter g.

    State (all tensors of shape (B, K) unless noted):
        mu        — mixture component means
        sigma     — mixture component std-devs (strictly > 0)
        q         — mixture weights (sum_k q[b,k] = 1)

    For ergonomics with the existing controller code, this class exposes
    `mode_weights` as an alias for `q`.
    """

    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        cfg: GaussianMixtureBankConfig,
    ) -> None:
        self.phys_model = phys_model
        self.cfg = cfg
        self.bs: int = phys_model.bs
        self.prec: str = phys_model.prec
        self.K: int = int(cfg.K)

        g_lo, g_hi = phys_model.cfg.g_range
        self.g_lo: float = float(g_lo)
        self.g_hi: float = float(g_hi)
        self.prior_width: float = max(g_hi - g_lo, 1e-30)

        # Cast helpers
        dtype = tf.float32 if self.prec == "float32" else tf.float64
        self._dtype = dtype
        self._sigma_min = tf.constant(
            cfg.sigma_min_fraction * self.prior_width, dtype=dtype
        )
        self._sigma_max = tf.constant(
            cfg.sigma_max_fraction * self.prior_width, dtype=dtype
        )

        # State tensors — initialised in reset()
        self.mu: Tensor | None = None
        self.sigma: Tensor | None = None
        self.q: Tensor | None = None

    # ------------------------------------------------------------------
    # Backwards-compat properties
    # ------------------------------------------------------------------

    @property
    def mode_weights(self) -> Tensor:
        return self.q

    @mode_weights.setter
    def mode_weights(self, v: Tensor) -> None:
        # The execute() loop sometimes assigns through this setter
        # after a stop_gradient.
        self.q = v

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rangen: tf.random.Generator) -> None:
        """Initialise to K equal-weight Gaussians tiling the prior."""
        del rangen
        prec = self.prec
        K = self.K
        bs = self.bs
        g_lo, g_hi = self.g_lo, self.g_hi
        mode_width = (g_hi - g_lo) / float(K)

        # Means: centred in each sub-prior
        centres = np.array(
            [g_lo + (k + 0.5) * mode_width for k in range(K)],
            dtype=np.float64,
        )
        mu_init = tf.constant(
            np.broadcast_to(centres[None, :], (bs, K)).copy(),
            dtype=prec,
        )
        sigma_init = tf.fill(
            (bs, K), tf.cast(self.cfg.init_sigma_fraction * mode_width, prec),
        )
        q_init = tf.fill(
            (bs, K), tf.cast(1.0 / float(K), prec),
        )

        self.mu = mu_init
        self.sigma = sigma_init
        self.q = q_init

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def detach_state(self) -> None:
        """Stop gradients through the bank state. Used between steps
        when stop_gradient_pf=True (Belliardo SI App. D.3 footnote)."""
        self.mu = tf.stop_gradient(self.mu)
        self.sigma = tf.stop_gradient(self.sigma)
        self.q = tf.stop_gradient(self.q)

    # ------------------------------------------------------------------
    # Bayes update — closed form
    # ------------------------------------------------------------------

    def _per_mode_evidence(
        self,
        outcomes: Tensor,   # (B,) values in {0, 1}
        k_g: Tensor,        # (B,) or (B, Q)  — Q = MFG quadrature nodes
        phi_total: Tensor,  # (B,) or (B, Q)
        vis: Tensor,        # (B,) or (B, Q)   — visibility A (>=0)
        quad_weights: Optional[Tensor] = None,  # (Q,) summing to 1; None ⇒ no quad
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """For each mixture component (B, K):
            Z_k        := P(y | mode k)             — used for q update
            mu_new_k   := E[g | y, mode k]          — under ADF
            sigma2_new := Var[g | y, mode k]        — under ADF

        Implementation note: we use scalar y∈{-1,+1} encoding internally
        so that one formula handles both outcomes.

        If `quad_weights` is provided, k_g/phi_total/vis carry an extra
        last axis Q indexing quadrature nodes over the MFG noise ε∈
        U[-bound, bound]. We compute the (k_g, vis)-dependent moments
        at each node and then average with `quad_weights`. The resulting
        Z_k, M1_k, M2_k are then the *MFG-marginalized* per-mode
        evidence and moments, matching how `perform_measurement`
        actually generates the data.
        """
        prec = self.prec
        mu = self.mu                                 # (B, K)
        sigma = self.sigma                           # (B, K)
        sigma2 = tf.square(sigma)

        # Decide layout: with quadrature, k_g/phi/A become (B, Q) and
        # we need to broadcast against (B, K) → final shape (B, K, Q).
        if quad_weights is not None:
            # k_g: (B, Q) -> (B, 1, Q); mu: (B, K) -> (B, K, 1)
            k = tf.expand_dims(k_g, axis=1)          # (B, 1, Q)
            phi = tf.expand_dims(phi_total, axis=1)  # (B, 1, Q)
            A = tf.expand_dims(vis, axis=1)          # (B, 1, Q)
            mu_b = tf.expand_dims(mu, axis=2)        # (B, K, 1)
            sigma2_b = tf.expand_dims(sigma2, axis=2)
        else:
            k = tf.expand_dims(k_g, axis=1)          # (B, 1)
            phi = tf.expand_dims(phi_total, axis=1)
            A = tf.expand_dims(vis, axis=1)
            mu_b = mu                                # (B, K)
            sigma2_b = sigma2

        y_sign = 2.0 * tf.expand_dims(outcomes, axis=1) - 1.0
        if quad_weights is not None:
            y_sign = tf.expand_dims(y_sign, axis=2)  # (B, 1, 1)
        y_sign = tf.cast(y_sign, prec)

        ksigma2 = k * sigma2_b
        damp = tf.exp(-0.5 * tf.square(k) * sigma2_b)
        arg = k * mu_b + phi
        cos_a = tf.cos(arg)
        sin_a = tf.sin(arg)

        # Closed-form moments
        E_c = damp * cos_a
        E_xc = damp * (mu_b * cos_a - ksigma2 * sin_a)
        E_x2c = damp * (
            (tf.square(mu_b) + sigma2_b - tf.square(k) * tf.square(sigma2_b)) * cos_a
            - 2.0 * k * mu_b * sigma2_b * sin_a
        )

        half = tf.cast(0.5, prec)
        Z_per_node = half * (tf.cast(1.0, prec) + y_sign * A * E_c)
        M1_per_node = half * (mu_b + y_sign * A * E_xc)
        M2_per_node = half * (tf.square(mu_b) + sigma2_b + y_sign * A * E_x2c)

        if quad_weights is not None:
            # Average over quadrature nodes (last axis), weights sum to 1.
            w = tf.cast(quad_weights, prec)          # (Q,)
            # broadcast to (1,1,Q)
            w = tf.reshape(w, [1, 1, -1])
            Z_k = tf.reduce_sum(Z_per_node * w, axis=2)
            M1_k = tf.reduce_sum(M1_per_node * w, axis=2)
            M2_k = tf.reduce_sum(M2_per_node * w, axis=2)
        else:
            Z_k, M1_k, M2_k = Z_per_node, M1_per_node, M2_per_node

        Z_eps = tf.cast(1e-30, prec)
        Z_safe = tf.maximum(Z_k, Z_eps)
        mu_new = M1_k / Z_safe
        sigma2_new = tf.maximum(
            M2_k / Z_safe - tf.square(mu_new),
            tf.cast(0.0, prec),
        )
        return Z_k, mu_new, sigma2_new

    def apply_measurement(
        self,
        outcomes: Tensor,        # (B, outcomes_size) — 1 scalar in {0,1}
        controls: Tensor,        # (B, controls_size) = (B, 3): (T, B', φ_mw)
        meas_step: Tensor,       # (B, 1) int
        continue_flag: Tensor,   # (B, 1) bool
        rangen: tf.random.Generator,
        *,
        do_resample: bool = False,  # accepted for API compatibility
    ) -> None:
        """Closed-form ADF update of all K Gaussians + Bayes update of q.

        Differentiable in `controls` through (k_g, vis, phi).
        """
        del meas_step, rangen, do_resample
        prec = self.prec
        # Pull the same physical quantities the model would compute, in
        # the same way. controls is shape (B, 3): T_s, Bp_kTm, mw_phase.
        T_s = controls[:, 0]
        Bp_kTm = controls[:, 1]
        mw_phase = controls[:, 2]

        # MFG-marginalization quadrature. In `paper` mode the true
        # outcome is sampled with B' · (1 + ε), ε ∈ U[-bound, bound];
        # the bank's likelihood must marginalize over ε too, or its
        # update is systematically biased. `mfg_quadrature` returns
        # (Q,) Legendre nodes scaled to [-bound, bound] and weights
        # summing to 1.
        eps_nodes, eps_weights = self.phys_model.mfg_quadrature(dtype=prec)
        Q = int(eps_nodes.shape[0])

        if Q == 1:
            # Fast path: no MFG noise.
            k_g = self.phys_model.k_g(T_s, Bp_kTm)
            vis = self.phys_model.known_visibility_factor(T_s, Bp_kTm)
            phi_off = tf.cast(self.phys_model.cfg.fixed_phi_off_rad, prec)
            phi_total = mw_phase + phi_off
            y = outcomes[:, 0]
            Z_k, mu_new, sigma2_new = self._per_mode_evidence(
                outcomes=y, k_g=k_g, phi_total=phi_total, vis=vis,
            )
        else:
            # Broadcast (B,) controls against (Q,) quadrature nodes.
            T_b = tf.expand_dims(T_s, axis=1)                # (B, 1)
            Bp_b = tf.expand_dims(Bp_kTm, axis=1)            # (B, 1)
            mw_b = tf.expand_dims(mw_phase, axis=1)          # (B, 1)
            eps_b = tf.expand_dims(eps_nodes, axis=0)        # (1, Q)
            Bp_eff = Bp_b * (1.0 + eps_b)                    # (B, Q)
            T_full = tf.broadcast_to(T_b, tf.shape(Bp_eff))
            k_g = self.phys_model.k_g(T_full, Bp_eff)        # (B, Q)
            vis = self.phys_model.known_visibility_factor(T_full, Bp_eff)
            phi_off = tf.cast(self.phys_model.cfg.fixed_phi_off_rad, prec)
            phi_total = tf.broadcast_to(mw_b + phi_off, tf.shape(k_g))
            y = outcomes[:, 0]
            Z_k, mu_new, sigma2_new = self._per_mode_evidence(
                outcomes=y, k_g=k_g, phi_total=phi_total, vis=vis,
                quad_weights=eps_weights,
            )

        # Mode-weight Bayes update.
        # q_new[b, k] ∝ q_old[b, k] * Z_k[b, k]
        new_q_unnorm = self.q * Z_k                     # (B, K)
        Z_total = tf.reduce_sum(new_q_unnorm, axis=1, keepdims=True)
        Z_total_safe = tf.maximum(Z_total, tf.cast(1e-30, prec))
        new_q = new_q_unnorm / Z_total_safe

        # Apply σ clipping (mode-collapse safety).
        new_sigma = tf.sqrt(tf.maximum(sigma2_new, tf.cast(0.0, prec)))
        new_sigma = tf.clip_by_value(new_sigma, self._sigma_min, self._sigma_max)

        # Gate by continue_flag: estimations that have finished retain
        # their previous state (no update).
        keep = tf.cast(continue_flag[:, 0:1], prec)     # (B, 1)
        gate_keep = keep > tf.cast(0.5, prec)
        self.mu    = tf.where(gate_keep, mu_new,    self.mu)
        self.sigma = tf.where(gate_keep, new_sigma, self.sigma)
        self.q     = tf.where(gate_keep, new_q,     self.q)

        # Optional mode revival.
        if self.cfg.revive_min_q > 0.0:
            self._revive_dead_modes()

    def _revive_dead_modes(self) -> None:
        """Periodically reseed components whose q has dropped below
        `cfg.revive_min_q` to a fresh uniform sub-prior placement.

        This is a non-differentiable maintenance step — we wrap it in
        stop_gradient and apply *only* to the masked entries.
        """
        prec = self.prec
        K = self.K
        g_lo, g_hi = self.g_lo, self.g_hi
        mode_width = (g_hi - g_lo) / float(K)
        thresh = tf.cast(self.cfg.revive_min_q, prec)
        # Per-mode fresh centres
        centres = np.array(
            [g_lo + (k + 0.5) * mode_width for k in range(K)],
            dtype=np.float64,
        )
        fresh_mu = tf.constant(
            np.broadcast_to(centres[None, :], (self.bs, K)).copy(),
            dtype=prec,
        )
        fresh_sigma = tf.fill(
            (self.bs, K),
            tf.cast(self.cfg.init_sigma_fraction * mode_width, prec),
        )
        dead = self.q < thresh                                          # (B, K)
        # Reset μ, σ for dead components; leave q unchanged so it can recover.
        new_mu = tf.where(dead, tf.stop_gradient(fresh_mu), self.mu)
        new_sigma = tf.where(dead, tf.stop_gradient(fresh_sigma), self.sigma)
        self.mu = new_mu
        self.sigma = new_sigma

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------

    def mode_means_and_stds(self) -> Tuple[Tensor, Tensor]:
        """Return (μ, σ) for each mode, shapes (B, K)."""
        return self.mu, self.sigma

    def marginal_mean_and_var(self) -> Tuple[Tensor, Tensor]:
        """Bayes mixture mean ĝ and posterior variance Var(g|y).

        ĝ        = Σ_k q_k μ_k
        Var(g|y) = Σ_k q_k σ_k² + Σ_k q_k (μ_k - ĝ)²

        Returns
        -------
        g_hat : (B,)
        var_g : (B,)
        """
        prec = self.prec
        q = self.q
        mu = self.mu
        sigma = self.sigma

        g_hat = tf.reduce_sum(q * mu, axis=1)                        # (B,)
        sec_mom = tf.reduce_sum(q * (tf.square(sigma) + tf.square(mu)), axis=1)
        var_g = tf.maximum(
            sec_mom - tf.square(g_hat),
            tf.cast(0.0, prec),
        )
        return g_hat, var_g

    def within_mode_std(self) -> Tensor:
        r"""Return the q-weighted within-mode standard deviation,

            σ_within(b) = sqrt( Σ_k q_{b,k} σ_{b,k}² )

        This is the scale relevant for "how finely can we resolve g
        *given which mode we're in*". It is the correct quantity to
        drive the controller's k_g cap (k_post_cap = 2π/(mult·σ_within)).

        Contrast with the marginal posterior std, which is dominated by
        the between-mode spread (∝ prior width) until q has collapsed to
        a single mode — and therefore *underestimates* the achievable
        resolution and pins k_g to coarse values for too long.

        Returns
        -------
        sigma_within : (B,)
        """
        prec = self.prec
        within2 = tf.reduce_sum(self.q * tf.square(self.sigma), axis=1)
        return tf.sqrt(tf.maximum(within2, tf.cast(0.0, prec)))

    def map_mode_estimate(self) -> Tuple[Tensor, Tensor]:
        """For backwards compatibility with diagnostics: return the
        mean and σ of the highest-q mode."""
        best_k = tf.argmax(self.q, axis=1, output_type=tf.int32)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)
        return (
            tf.gather_nd(self.mu, gather_idx),
            tf.gather_nd(self.sigma, gather_idx),
        )

    def closest_mode_estimate(
        self, g_true: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Oracle: pick the mode whose μ is closest to g_true.

        Useful as a *diagnostic* (do we have a mode near the truth?), but
        NOT used in the loss (which uses the mixture mean instead).
        """
        d2 = tf.square(self.mu - tf.expand_dims(g_true, axis=1))     # (B, K)
        best_k = tf.argmin(d2, axis=1, output_type=tf.int32)
        batch_idx = tf.range(self.bs, dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, best_k], axis=1)
        return (
            tf.gather_nd(self.mu, gather_idx),
            tf.gather_nd(self.sigma, gather_idx),
            tf.gather_nd(self.q, gather_idx),
        )

    # ------------------------------------------------------------------
    # Diagnostic moments
    # ------------------------------------------------------------------

    def holevo_log_at_scale(self, k_ref: Tensor) -> Tensor:
        """log(1 + V_H(k_ref)) — Holevo-style figure of merit for circular
        variance. Used only for diagnostics.

        Under a Gaussian mixture:
          μ(k) := E[exp(i k g)] = Σ_k q_k · exp(i k μ_k - k²σ_k²/2)
        """
        prec = self.prec
        if k_ref.shape.rank == 0:
            k = tf.fill((self.bs,), tf.cast(k_ref, prec))
        else:
            k = tf.cast(k_ref, prec)
        k_b = tf.expand_dims(k, axis=1)                              # (B, 1)
        damp = tf.exp(-0.5 * tf.square(k_b) * tf.square(self.sigma)) # (B, K)
        arg = k_b * self.mu                                          # (B, K)
        re = tf.reduce_sum(self.q * damp * tf.cos(arg), axis=1)
        im = tf.reduce_sum(self.q * damp * tf.sin(arg), axis=1)
        abs_mu_sq = tf.square(re) + tf.square(im)
        vh = 1.0 / tf.maximum(abs_mu_sq, tf.cast(1e-30, prec)) - 1.0
        return tf.math.log1p(vh)