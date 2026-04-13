"""Gravity sensor physical model — clean extraction from gravimeter_model_imm_wrap.py.

This module contains:
  - :class:`GravimeterConfig`  — frozen dataclass of all sensor parameters
  - :class:`GravityStatelessPhysicalModel` — qsensoropt ``StatelessPhysicalModel`` subclass
    implementing the levitated NV-center nanodiamond gravimeter measurement model
  - Helper functions: :func:`safe_clip_prob`, :func:`wrap_to_pi_tf`,
    :func:`normalize_to_minus1_plus1`

The physical model is completely unchanged from ``gravimeter_model_imm_wrap.py``.
All IMM-specific code (``GravityWrapIMMFilter``, ``GravityWrapIMMConfig``,
``ProtocolLibrary``, ``IMMState``) has been removed.

References
----------
Belliardo et al. (2024). "Model-aware reinforcement learning for high-performance
Bayesian experimental design in quantum metrology."
Physical Review A 109, 062609. https://doi.org/10.1103/PhysRevA.109.062609
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import Tensor


# ---------------------------------------------------------------------------
# Local qsensoropt module loader (mirrors the pattern in the original file)
# ---------------------------------------------------------------------------

def _load_local_qsensoropt_module(module_name: str):
    """Load a qsensoropt module from the local workspace directory."""
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
except Exception:
    Parameter = _load_local_qsensoropt_module("parameter").Parameter
    Control = _load_local_qsensoropt_module("physical_model").Control
    StatelessPhysicalModel = _load_local_qsensoropt_module(
        "stateless_phys_model"
    ).StatelessPhysicalModel


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_clip_prob(x: Tensor, eps: float = 1e-9) -> Tensor:
    """Clip a probability tensor to ``[eps, 1 - eps]`` to avoid log(0).

    Parameters
    ----------
    x : Tensor
        Probability values (arbitrary shape).
    eps : float, optional
        Clipping margin, default ``1e-9``.

    Returns
    -------
    Tensor
        Clipped tensor with the same shape and dtype as ``x``.
    """
    eps_t = tf.cast(eps, x.dtype)
    return tf.clip_by_value(x, eps_t, 1.0 - eps_t)


def wrap_to_pi_tf(x: Tensor) -> Tensor:
    """Wrap angles to ``(-π, π]`` using TensorFlow operations.

    Parameters
    ----------
    x : Tensor
        Angle values in radians.

    Returns
    -------
    Tensor
        Wrapped angles in ``(-π, π]``, same shape and dtype as ``x``.
    """
    two_pi = tf.cast(2.0 * pi, x.dtype)
    return tf.math.floormod(x + tf.cast(pi, x.dtype), two_pi) - tf.cast(pi, x.dtype)


def normalize_to_minus1_plus1(x: Tensor, bounds: tuple[float, float]) -> Tensor:
    """Linearly map ``x`` from ``[bounds[0], bounds[1]]`` to ``[-1, 1]``.

    Parameters
    ----------
    x : Tensor
        Values to normalize.
    bounds : tuple[float, float]
        ``(lo, hi)`` interval in which ``x`` resides.

    Returns
    -------
    Tensor
        Normalized values clipped to ``[-1, 1]``, same shape and dtype as ``x``.
    """
    lo = tf.cast(bounds[0], x.dtype)
    hi = tf.cast(bounds[1], x.dtype)
    width = tf.maximum(hi - lo, tf.cast(1e-18, x.dtype))
    y = 2.0 * (x - lo) / width - 1.0
    return tf.clip_by_value(y, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GravimeterConfig:
    """Configuration dataclass for the levitated NV-center nanodiamond gravimeter.

    All fields are physical constants or experimental parameters. The defaults
    correspond to the sensor described in Belliardo et al. (2024).

    Parameters
    ----------
    omega_rad_s : float
        Trap oscillation frequency (rad/s). Default: ``2π × 10 kHz``.
    gamma_e_rad_s_T : float
        Electron gyromagnetic ratio (rad/(s·T)). Default: ``2π × 28 GHz/T``.
    mass_kg : float
        Nanodiamond mass (kg). Default: ``1.47e-17`` kg.
    hbar_J_s : float
        Reduced Planck constant (J·s).
    kT_to_T : float
        Unit conversion factor kT/m → T/m. Default: ``1e3``.
    g_range : tuple[float, float]
        Prior interval ``[g_lo, g_hi]`` for gravitational acceleration (m/s²).
    infer_mfg_bias : bool
        If ``True``, the relative MFG calibration bias ``β_B`` is treated as an
        additional unknown parameter to infer jointly with ``g``.
    beta_B_range : tuple[float, float]
        Prior interval for the relative MFG bias ``β_B`` (dimensionless).
        Used only when ``infer_mfg_bias=True``.
    T_range_s : tuple[float, float]
        Admissible range of interrogation times ``T`` (s).
    Bp_range_kTm : tuple[float, float]
        Admissible range of magnetic field gradient ``B'`` (kT/m).
    mfg_rel_noise_bound : float
        Bound on the uniform MFG multiplicative noise ``ε ~ U(-ε₀, ε₀)``.
        Set to ``0`` to disable MFG noise.
    mfg_noise_quad_points : int
        Number of Gauss–Legendre quadrature nodes for MFG noise integration
        in the likelihood.
    sigma_omega_rel : float
        Relative standard deviation of trap frequency fluctuations
        ``σ_ω/ω₀`` used to compute trap-induced visibility reduction.
    trap_visibility_mode : str
        One of ``"none"``, ``"small_noise_avg"``, ``"exact_single_delta"``.
    trap_noise_quad_points : int
        Number of Gauss–Hermite quadrature nodes for trap visibility averaging.
    fixed_mfg_rel_bias : float
        Fixed (non-inferred) relative MFG bias applied in the measurement
        simulation. Only active when ``infer_mfg_bias=False``.
    apply_fixed_mfg_bias_in_model : bool
        If ``True``, the fixed MFG bias is also applied in ``model()``
        (the likelihood function used for Bayes updates).
    T2_spin_s : float or None
        Spin coherence time (s). If set, the visibility is multiplied by
        ``exp(-t_cycle / T2)``.
    readout_flip_prob : float
        Probability of a readout bit-flip (symmetric). Default: ``0``.
    dead_time_s : float
        Dead time per measurement cycle (s). Added to the cycle time for
        resource counting.
    mfg_resource_cost_s_at_ref : float
        Extra resource cost (s) attributed to MFG switching at the reference
        field ``mfg_resource_ref_kTm``. Default: ``0`` (no extra cost).
    mfg_resource_ref_kTm : float
        Reference MFG value (kT/m) for the quadratic MFG resource cost model.
    prec : str
        Floating-point precision, either ``"float32"`` or ``"float64"``.
    """

    omega_rad_s: float = 2.0 * pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * pi * 28e9
    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34
    kT_to_T: float = 1e3

    g_range: tuple[float, float] = (9.7806, 9.825)
    infer_mfg_bias: bool = False
    beta_B_range: tuple[float, float] = (-0.10, 0.10)

    T_range_s: tuple[float, float] = (10e-6, 1.2e-3)       # was (3.0e-4, 1.2e-3)
    Bp_range_kTm: tuple[float, float] = (0.5, 80.0) # was (20.0, 80.0)

    mfg_rel_noise_bound: float = 0.0
    mfg_noise_quad_points: int = 1
    sigma_omega_rel: float = 0.0
    trap_visibility_mode: str = "none"
    trap_noise_quad_points: int = 9

    fixed_mfg_rel_bias: float = 0.0
    apply_fixed_mfg_bias_in_model: bool = True

    T2_spin_s: Optional[float] = None
    readout_flip_prob: float = 0.0

    dead_time_s: float = 0.0
    mfg_resource_cost_s_at_ref: float = 0.0
    mfg_resource_ref_kTm: float = 50.0

    prec: str = "float32"

    @property
    def tau_s(self) -> float:
        """Trap oscillation period ``τ = 2π / ω₀`` (s)."""
        return 2.0 * pi / self.omega_rad_s


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------

class GravityStatelessPhysicalModel(StatelessPhysicalModel):
    """Stateless physical model for the levitated NV-center nanodiamond gravimeter.

    Implements the three-control cosine likelihood::

        p(y=1 | g, T, B', φ) = 0.5 * (1 + vis * cos(k_g(T, B') * g + φ))

    where ``k_g(T, B')`` is the measurement gain (effective phase per unit g),
    ``vis`` is the fringe visibility, and ``φ`` is the microwave phase.

    Controls
    --------
    ``T_s`` : float
        Interrogation time (seconds).
    ``Bp_kTm`` : float
        Magnetic field gradient (kT/m).
    ``mw_phase_rad`` : float
        Microwave phase (radians).

    Parameters (unknown)
    --------------------
    ``g`` : float
        Gravitational acceleration (m/s²) — the primary estimation target.
    ``beta_B`` : float, optional
        Relative MFG calibration bias (dimensionless). Only present when
        ``cfg.infer_mfg_bias=True``.

    Notes
    -----
    The ``model()`` method (used for Bayesian weight updates in the particle
    filter) and the ``perform_measurement()`` method (used for simulating
    the experiment) are separately implemented to allow different noise
    treatments: ``model()`` uses deterministic quadrature for MFG noise,
    while ``perform_measurement()`` draws stochastic MFG and trap-frequency
    noise samples.
    """

    def __init__(self, batchsize: int, cfg: GravimeterConfig) -> None:
        """Construct the gravity physical model.

        Parameters
        ----------
        batchsize : int
            Number of parallel estimations (batch size).
        cfg : GravimeterConfig
            Sensor configuration dataclass.
        """
        self.cfg = cfg
        controls = [
            Control(name="T_s", is_discrete=False),
            Control(name="Bp_kTm", is_discrete=False),
            Control(name="mw_phase_rad", is_discrete=False),
        ]
        params = [Parameter(bounds=cfg.g_range, name="g")]
        if cfg.infer_mfg_bias:
            params.append(Parameter(bounds=cfg.beta_B_range, name="beta_B"))
        super().__init__(
            batchsize=batchsize,
            controls=controls,
            params=params,
            outcomes_size=1,
            prec=cfg.prec,
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        """Split the particle tensor into (g, beta_B) components.

        Parameters
        ----------
        parameters : Tensor
            Shape ``(bs, num_particles, d)``.

        Returns
        -------
        g : Tensor
            Shape ``(bs, num_particles)``.
        beta_B : Tensor
            Shape ``(bs, num_particles)``, zeros when ``infer_mfg_bias=False``.
        """
        g = parameters[:, :, 0]
        if self.cfg.infer_mfg_bias:
            beta_B = parameters[:, :, 1]
        else:
            beta_B = tf.zeros_like(g)
        return g, beta_B

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def y0_m(self, dtype) -> Tensor:
        """Zero-point motion amplitude of the trapped nanodiamond (m).

        ``y₀ = sqrt(ℏ / (2 m ω₀))``
        """
        cfg = self.cfg
        return tf.cast(
            np.sqrt(cfg.hbar_J_s / (2.0 * cfg.mass_kg * cfg.omega_rad_s)), dtype
        )

    def eta(self, Bp_kTm: Tensor) -> Tensor:
        """Dimensionless coupling strength ``η = γ_e B' y₀ / ω₀``.

        Parameters
        ----------
        Bp_kTm : Tensor
            Magnetic field gradient (kT/m).
        """
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, Bp_kTm.dtype)
        return (
            tf.cast(cfg.gamma_e_rad_s_T, Bp_kTm.dtype)
            * Bp_T_per_m
            * self.y0_m(Bp_kTm.dtype)
            / tf.cast(cfg.omega_rad_s, Bp_kTm.dtype)
        )

    def k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        """Measurement gain ``k_g(T, B')`` (rad·s²/m).

        The gain determines the phase accumulated per unit g::

            k_g = (2γ_e / ω₀) * B'_T * T² + (8π γ_e / ω₀³) * B'_T

        where ``B'_T = B' × 1000`` converts kT/m → T/m.

        Parameters
        ----------
        T_s : Tensor
            Interrogation time (s).
        Bp_kTm : Tensor
            Magnetic field gradient (kT/m).

        Returns
        -------
        Tensor
            Gain ``k_g`` with the same shape as ``T_s``.
        """
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (
            (2.0 * ge / w) * Bp_T_per_m * tf.square(T_s)
            + (8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)) * Bp_T_per_m
        )

    def min_gain(self, dtype) -> Tensor:
        """Minimum achievable gain at ``(T_min, B'_min)``."""
        return self.k_g(
            tf.cast(self.cfg.T_range_s[0], dtype),
            tf.cast(self.cfg.Bp_range_kTm[0], dtype),
        )

    def max_gain(self, dtype) -> Tensor:
        """Maximum achievable gain at ``(T_max, B'_max)``."""
        return self.k_g(
            tf.cast(self.cfg.T_range_s[1], dtype),
            tf.cast(self.cfg.Bp_range_kTm[1], dtype),
        )

    def cycle_time_s(self, T_s: Tensor) -> Tensor:
        """Total measurement cycle time (s), including dead time and spin-up.

        ``t_cycle = t_dead + 3.5 τ + 2T``
        """
        cfg = self.cfg
        return tf.cast(cfg.dead_time_s + 3.5 * cfg.tau_s, T_s.dtype) + 2.0 * T_s

    def total_resource_cost_s(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        """Total resource cost (s) including optional MFG switching overhead.

        Parameters
        ----------
        T_s : Tensor
            Interrogation time (s).
        Bp_kTm : Tensor
            Magnetic field gradient (kT/m).
        """
        cfg = self.cfg
        extra = tf.zeros_like(Bp_kTm)
        if cfg.mfg_resource_cost_s_at_ref > 0.0:
            bref = tf.cast(cfg.mfg_resource_ref_kTm, Bp_kTm.dtype)
            c_ref = tf.cast(cfg.mfg_resource_cost_s_at_ref, Bp_kTm.dtype)
            extra = c_ref * tf.square(
                Bp_kTm / tf.maximum(bref, tf.cast(1e-18, Bp_kTm.dtype))
            )
        return self.cycle_time_s(T_s) + extra

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def trap_visibility_avg_small_noise(self, Bp_kTm: Tensor) -> Tensor:
        """Trap visibility in the small-noise approximation.

        Analytically averages over Gaussian trap-frequency fluctuations
        using the second-order Taylor expansion.
        """
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        correction = (
            tf.cast(1944.0 * (pi ** 4) * (cfg.sigma_omega_rel ** 4), eta.dtype)
            * tf.square(eta)
        )
        return tf.clip_by_value(1.0 - correction, 0.0, 1.0)

    def trap_visibility_exact_from_delta_omega(
        self, Bp_kTm: Tensor, delta_omega_rad_s: Tensor
    ) -> Tensor:
        """Exact trap visibility for a given frequency deviation ``δω``.

        Parameters
        ----------
        Bp_kTm : Tensor
            Magnetic field gradient.
        delta_omega_rad_s : Tensor
            Trap frequency deviation from nominal (rad/s).
        """
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        tau = tf.cast(cfg.tau_s, eta.dtype)
        x = tf.cast(cfg.omega_rad_s, eta.dtype) * (
            -tau * delta_omega_rad_s / tf.cast(cfg.omega_rad_s, eta.dtype)
        )
        amp = 16.0 * eta * tf.cos(x / 4.0) * tf.square(tf.sin(3.0 * x / 4.0))
        return tf.exp(-0.5 * tf.square(amp))

    def _trap_visibility_marginalized(self, Bp_kTm: Tensor) -> Tensor:
        """Trap visibility averaged over Gaussian frequency fluctuations
        using Gauss–Hermite quadrature."""
        cfg = self.cfg
        n = int(max(1, cfg.trap_noise_quad_points))
        if cfg.sigma_omega_rel <= 0.0 or n <= 1:
            return self.trap_visibility_avg_small_noise(Bp_kTm)
        x_np, w_np = np.polynomial.hermite.hermgauss(n)
        x = tf.constant(x_np, dtype=Bp_kTm.dtype)
        w = tf.constant(w_np / np.sqrt(np.pi), dtype=Bp_kTm.dtype)
        sigma = tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, Bp_kTm.dtype)
        delta_omega = tf.sqrt(tf.cast(2.0, Bp_kTm.dtype)) * sigma * x
        vis = self.trap_visibility_exact_from_delta_omega(
            Bp_kTm[..., None], delta_omega
        )
        return tf.reduce_sum(vis * w, axis=-1)

    def known_visibility_factor(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        """Deterministic fringe visibility factor used in the likelihood model.

        Combines trap-frequency noise averaging and optional T₂ spin decoherence.

        Parameters
        ----------
        T_s : Tensor
            Interrogation time (s).
        Bp_kTm : Tensor
            Magnetic field gradient (kT/m).
        """
        cfg = self.cfg
        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            vis = self._trap_visibility_marginalized(Bp_kTm)
        else:
            raise ValueError(
                f"Unknown trap_visibility_mode={cfg.trap_visibility_mode!r}"
            )
        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(
                -self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype)
            )
        return tf.clip_by_value(vis, 0.0, 1.0)

    def sample_true_visibility_factor(
        self, T_s: Tensor, Bp_kTm: Tensor, rangen: tf.random.Generator
    ) -> Tensor:
        """Stochastic fringe visibility for measurement simulation.

        Draws a random trap-frequency deviation and computes the exact
        resulting visibility. Used only in ``perform_measurement()``.

        Parameters
        ----------
        T_s : Tensor
            Interrogation time (s).
        Bp_kTm : Tensor
            Magnetic field gradient (kT/m).
        rangen : tf.random.Generator
            TensorFlow random number generator.
        """
        cfg = self.cfg
        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            delta_omega = rangen.normal(
                tf.shape(T_s), dtype=T_s.dtype
            ) * tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, T_s.dtype)
            vis = self.trap_visibility_exact_from_delta_omega(Bp_kTm, delta_omega)
        else:
            raise ValueError(
                f"Unknown trap_visibility_mode={cfg.trap_visibility_mode!r}"
            )
        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(
                -self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype)
            )
        return tf.clip_by_value(vis, 0.0, 1.0)

    # ------------------------------------------------------------------
    # MFG noise helpers
    # ------------------------------------------------------------------

    def mfg_quadrature(self, dtype) -> tuple[Tensor, Tensor]:
        """Gauss–Legendre quadrature nodes and weights for MFG noise integration.

        Returns
        -------
        nodes : Tensor, shape (n,)
            Quadrature nodes for ``ε ∈ [-ε₀, ε₀]``.
        weights : Tensor, shape (n,)
            Corresponding quadrature weights (normalized to sum 1).
        """
        bound = float(self.cfg.mfg_rel_noise_bound)
        n = int(max(1, self.cfg.mfg_noise_quad_points))
        if bound <= 0.0 or n <= 1:
            return tf.constant([0.0], dtype=dtype), tf.constant([1.0], dtype=dtype)
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        nodes = tf.constant(bound * x_np, dtype=dtype)
        weights = tf.constant(0.5 * w_np, dtype=dtype)
        return nodes, weights

    def _sample_mfg_rel_noise(
        self, shape, rangen: tf.random.Generator, dtype
    ) -> Tensor:
        """Draw a uniform MFG relative noise sample ``ε ~ U(-ε₀, ε₀)``.

        Parameters
        ----------
        shape : TensorShape
            Shape of the noise tensor to generate.
        rangen : tf.random.Generator
            TensorFlow random number generator.
        dtype : tf.DType
            Output dtype.
        """
        bound = float(self.cfg.mfg_rel_noise_bound)
        if bound <= 0.0:
            return tf.zeros(shape, dtype=dtype)
        return rangen.uniform(
            shape=shape,
            minval=tf.cast(-bound, dtype),
            maxval=tf.cast(+bound, dtype),
            dtype=dtype,
        )

    def _model_bias_factor(self, beta_B: Tensor) -> Tensor:
        """MFG bias factor used in ``model()`` (deterministic likelihood).

        Returns ``1 + β_B`` when inferring the bias, or the fixed bias
        constant ``1 + β_B_fixed`` when not.
        """
        cfg = self.cfg
        if cfg.infer_mfg_bias:
            return 1.0 + beta_B
        if cfg.apply_fixed_mfg_bias_in_model:
            return tf.cast(1.0 + cfg.fixed_mfg_rel_bias, beta_B.dtype)
        return tf.cast(1.0, beta_B.dtype)

    def _measurement_bias_factor(self, beta_B: Tensor) -> Tensor:
        """MFG bias factor used in ``perform_measurement()`` (stochastic).

        Returns ``1 + β_B`` regardless of whether the bias is inferred,
        since in measurement simulation the true bias is always applied.
        """
        cfg = self.cfg
        if cfg.infer_mfg_bias:
            return 1.0 + beta_B
        return tf.cast(1.0 + cfg.fixed_mfg_rel_bias, beta_B.dtype)

    # ------------------------------------------------------------------
    # Core likelihood
    # ------------------------------------------------------------------

    def likelihood_given_global_params(
        self,
        outcomes: Tensor,
        controls: Tensor,
        g: Tensor,
        beta_B: Tensor,
    ) -> Tensor:
        """Compute ``p(y | g, β_B, T, B', φ)`` with MFG quadrature.

        Parameters
        ----------
        outcomes : Tensor
            Binary outcomes, shape ``(..., 1)``. A value ``> 0.5`` means ``y=1``.
        controls : Tensor
            Controls ``[T_s, Bp_kTm, mw_phase_rad]``, shape ``(..., 3)``.
        g : Tensor
            Gravitational acceleration values, shape ``(...)``.
        beta_B : Tensor
            MFG relative bias values, shape ``(...)``.

        Returns
        -------
        Tensor
            Probabilities, shape ``(...)`` (same leading dims as inputs).
        """
        T_s = controls[..., 0]
        Bp_commanded_kTm = controls[..., 1]
        mw_phase = controls[..., 2]
        Bp_base_kTm = Bp_commanded_kTm * self._model_bias_factor(beta_B)
        eps_nodes, eps_weights = self.mfg_quadrature(dtype=T_s.dtype)
        if int(eps_nodes.shape[0]) == 1:
            Bp_kTm = Bp_base_kTm
            vis = self.known_visibility_factor(T_s, Bp_kTm)
            theta = self.k_g(T_s, Bp_kTm) * g + mw_phase
            p_plus = safe_clip_prob(0.5 * (1.0 + vis * tf.cos(theta)))
        else:
            Bp_kTm = Bp_base_kTm[..., None] * (1.0 + eps_nodes)
            T_e = T_s[..., None]
            mw_e = mw_phase[..., None]
            g_e = g[..., None]
            vis = self.known_visibility_factor(T_e, Bp_kTm)
            theta = self.k_g(T_e, Bp_kTm) * g_e + mw_e
            p_plus_nodes = safe_clip_prob(0.5 * (1.0 + vis * tf.cos(theta)))
            p_plus = safe_clip_prob(
                tf.reduce_sum(p_plus_nodes * eps_weights, axis=-1)
            )
        y = outcomes[..., 0]
        prob = tf.where(y > 0.5, p_plus, 1.0 - p_plus)
        return safe_clip_prob(prob)

    # ------------------------------------------------------------------
    # qsensoropt interface: model
    # ------------------------------------------------------------------

    def model(
        self,
        outcomes: Tensor,
        controls: Tensor,
        parameters: Tensor,
        meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tensor:
        """Bayesian likelihood ``p(y | parameters, controls)``.

        This is the method called by :class:`~.ParticleFilter.apply_measurement`
        to compute per-particle likelihoods for the Bayes weight update.

        Parameters
        ----------
        outcomes : Tensor
            Shape ``(bs, num_systems, outcomes_size)`` — binary outcomes.
        controls : Tensor
            Shape ``(bs, num_systems, controls_size)`` — ``[T_s, Bp, φ]``.
        parameters : Tensor
            Shape ``(bs, num_systems, d)`` — ``[g]`` or ``[g, β_B]``.
        meas_step : Tensor
            Shape ``(bs, num_systems, 1)`` — measurement step index (unused).
        num_systems : int
            Number of particles (``pf.np``).

        Returns
        -------
        Tensor
            Probabilities, shape ``(bs, num_systems)``.
        """
        del meas_step, num_systems  # unused in stateless model
        g, beta_B = self._split_parameters(parameters)
        return self.likelihood_given_global_params(outcomes, controls, g, beta_B)

    # ------------------------------------------------------------------
    # qsensoropt interface: perform_measurement
    # ------------------------------------------------------------------

    def perform_measurement(
        self,
        controls: Tensor,
        parameters: Tensor,
        meas_step: Tensor,
        rangen: tf.random.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Simulate one measurement: sample ``y ~ Bernoulli(p(y=1|g,x))``.

        Parameters
        ----------
        controls : Tensor
            Shape ``(bs, 1, controls_size)`` — ``[T_s, Bp_kTm, mw_phase_rad]``.
        parameters : Tensor
            Shape ``(bs, 1, d)`` — true values ``[g]`` or ``[g, β_B]``.
        meas_step : Tensor
            Shape ``(bs, 1, 1)`` — measurement step index (unused).
        rangen : tf.random.Generator
            TensorFlow random number generator.

        Returns
        -------
        outcomes : Tensor
            Shape ``(bs, 1, 1)`` — binary outcome (0.0 or 1.0).
        log_prob : Tensor
            Shape ``(bs, 1)`` — log-probability of the observed outcome.
        """
        del meas_step
        T_s = controls[:, 0, 0]
        Bp_commanded_kTm = controls[:, 0, 1]
        mw_phase = controls[:, 0, 2]
        g, beta_B = self._split_parameters(parameters)
        Bp_base_kTm = Bp_commanded_kTm * self._measurement_bias_factor(beta_B[:, 0])
        eps = self._sample_mfg_rel_noise(
            tf.shape(Bp_base_kTm), rangen, Bp_base_kTm.dtype
        )
        Bp_kTm = Bp_base_kTm * (1.0 + eps)
        vis_true = self.sample_true_visibility_factor(T_s, Bp_kTm, rangen)
        theta = self.k_g(T_s, Bp_kTm) * g[:, 0] + mw_phase
        p_plus = safe_clip_prob(0.5 * (1.0 + vis_true * tf.cos(theta)))
        if self.cfg.readout_flip_prob > 0.0:
            flip = tf.cast(self.cfg.readout_flip_prob, p_plus.dtype)
            p_plus = safe_clip_prob(
                (1.0 - flip) * p_plus + flip * (1.0 - p_plus)
            )
        u = rangen.uniform(
            shape=tf.shape(p_plus), minval=0.0, maxval=1.0, dtype=p_plus.dtype
        )
        y = tf.cast(u < p_plus, p_plus.dtype)
        outcomes = tf.expand_dims(tf.expand_dims(y, axis=1), axis=2)
        log_prob = tf.expand_dims(
            tf.math.log(tf.where(y > 0.5, p_plus, 1.0 - p_plus)), axis=1
        )
        return outcomes, log_prob

    # ------------------------------------------------------------------
    # qsensoropt interface: count_resources
    # ------------------------------------------------------------------

    def count_resources(
        self,
        resources: Tensor,
        outcomes: Tensor,
        controls: Tensor,
        true_values: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        """Compute resources consumed by the current measurement step.

        The resource is the total measurement cycle time (s):
        ``t_cycle = t_dead + 3.5 τ + 2T + t_MFG(B')``.

        Parameters
        ----------
        resources : Tensor
            Shape ``(bs, 1)`` — accumulated resources so far (unused here).
        outcomes : Tensor
            Shape ``(bs, 1, outcomes_size)`` — outcomes (unused).
        controls : Tensor
            Shape ``(bs, controls_size)`` — ``[T_s, Bp_kTm, mw_phase_rad]``.
        true_values : Tensor
            Shape ``(bs, 1, d)`` — true parameter values (unused).
        meas_step : Tensor
            Shape ``(bs, 1)`` — step index (unused).

        Returns
        -------
        Tensor
            Shape ``(bs, 1)`` — resource cost of this measurement step.
        """
        del resources, outcomes, true_values, meas_step
        T_s = controls[:, 0]
        Bp_kTm = controls[:, 1]
        cost = self.total_resource_cost_s(T_s, Bp_kTm)
        return tf.expand_dims(cost, axis=1)
