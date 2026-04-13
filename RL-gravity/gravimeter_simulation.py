"""Single-ParticleFilter gravity simulation for the levitated NV-center gravimeter.

This module implements a clean, single-PF-based Bayesian sensor control simulation
that subclasses qsensoropt's :class:`~.StatelessSimulation`. Unlike the multi-PF
bank approach, a single PF naturally handles the multimodal posterior that arises
in the early (coarse) measurement phase: particle weights spread across multiple
cosine fringes without requiring explicit mode management.

Architecture
------------
* Physical model : :class:`~gravimeter_model.GravityStatelessPhysicalModel`
* Particle filter: qsensoropt :class:`~.ParticleFilter` (single instance)
* Simulation     : :class:`GravitySimulation` (subclass of
  :class:`~.StatelessSimulation`)
* Controller     : MLP built by :func:`build_controller`

Controller input (4 scalars)
-----------------------------
1. ``μ_g`` — posterior mean of *g*, normalised to [-1, 1] using the prior bounds.
2. ``log(σ_g)`` — log of the posterior standard deviation of *g*, encoded to
   approximately [-1, 1] for σ ∈ [1, 1e-5] via the formula
   ``-2/10 * log(σ) - 1``.
3. ``step / num_steps`` normalised to [-1, 1].
4. ``used_resources / max_resources`` normalised to [-1, 1].

Controller output (3 scalars, scaled from tanh → physical ranges)
------------------------------------------------------------------
* ``T_s``         — interrogation time [s]
* ``Bp_kTm``      — magnetic field gradient [kT/m]
* ``mw_phase_rad``— microwave phase [rad]

Loss
----
MSE = (μ_g_hat - g_true)², as in qsensoropt's
:class:`~.StatelessMetrology`.

References
----------
Belliardo et al. (2024). Physical Review A 109, 062609.
https://doi.org/10.1103/PhysRevA.109.062609
"""

from __future__ import annotations

import importlib.util
import sys
import types
from math import pi
from pathlib import Path
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor


# ---------------------------------------------------------------------------
# Local qsensoropt module loader
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
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.stateless_simulation import StatelessSimulation
    from qsensoropt.utils import normalize
except Exception:
    ParticleFilter = _load_local_qsensoropt_module("particle_filter").ParticleFilter
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters
    StatelessSimulation = _load_local_qsensoropt_module(
        "stateless_simulation"
    ).StatelessSimulation
    normalize = _load_local_qsensoropt_module("utils").normalize


from gravimeter_model import GravimeterConfig, GravityStatelessPhysicalModel


# ---------------------------------------------------------------------------
# INPUT_SIZE constant
# ---------------------------------------------------------------------------

#: Number of scalars fed to the controller at each step.
#:
#: * 1  — normalised posterior mean of g
#: * 1  — log-encoded posterior std of g
#: * 1  — step counter normalised to [-1, 1]
#: * 1  — resources used, normalised to [-1, 1]
INPUT_SIZE: int = 4

#: Names of the four controller inputs (for logging / diagnostics).
INPUT_NAMES: List[str] = [
    "Mean_g",          # normalised posterior mean
    "LogDev_g",        # log-encoded posterior std
    "StepOverMaxStep", # step fraction
    "ResOverMaxRes",   # resource fraction
]


# ---------------------------------------------------------------------------
# GravitySimulation
# ---------------------------------------------------------------------------

class GravitySimulation(StatelessSimulation):
    """Bayesian metrological simulation for the gravimeter with a single PF.

    Subclasses :class:`~.StatelessSimulation` and plugs into qsensoropt's
    built-in ``execute()`` loop.  The controller receives a compact 4-scalar
    summary of the particle filter state together with progress counters, and
    outputs the three physical controls (T_s, Bp_kTm, mw_phase_rad).

    The loss is the mean square error of the posterior-mean estimator for *g*,
    which is qsensoropt's default MSE loss as defined in
    :class:`~.StatelessMetrology`.

    Parameters
    ----------
    particle_filter : ParticleFilter
        The single PF that tracks the posterior over *g*.
    phys_model : GravityStatelessPhysicalModel
        Physics model; also carries batchsize and precision.
    control_strategy : callable
        Keras model (or any callable) mapping the 4-d input tensor to
        physical controls (shape ``(bs, 3)``).
    simpars : SimulationParameters
        qsensoropt simulation flags and stopping criteria.
    """

    def __init__(
        self,
        particle_filter: ParticleFilter,
        phys_model: GravityStatelessPhysicalModel,
        control_strategy,
        simpars: SimulationParameters,
    ) -> None:
        super().__init__(
            particle_filter=particle_filter,
            phys_model=phys_model,
            control_strategy=control_strategy,
            input_size=INPUT_SIZE,
            input_name=INPUT_NAMES,
            simpars=simpars,
        )
        # Pre-compute the identity weight matrix for the MSE loss (d=1 here)
        self._cov_weight = tf.ones(
            (self.bs, 1, 1), dtype=simpars.prec, name="cov_weight"
        )

    # ------------------------------------------------------------------
    # generate_input — called at every step of the execute() loop
    # ------------------------------------------------------------------

    def generate_input(
        self,
        weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen,
    ) -> Tensor:
        r"""Build the 4-scalar input tensor for the controller.

        Parameters
        ----------
        weights : Tensor, shape (bs, np)
            Particle weights (normalised to sum 1).
        particles : Tensor, shape (bs, np, d)
            Particle positions (d=1 for *g* only, d=2 if inferring bias).
        meas_step : Tensor, shape (bs, 1), dtype=int32
            0-based measurement step index.
        used_resources : Tensor, shape (bs, 1)
            Cumulative resource consumption (seconds) for each batch element.
        rangen : tf.random.Generator
            Random generator (unused here; kept for API compatibility).

        Returns
        -------
        Tensor, shape (bs, 4)
            ``[mean_g_norm, log_std_g_norm, step_norm, res_norm]``
        """
        pars = self.simpars
        prec = pars.prec

        # ---- 1.  Posterior mean of g, normalised to [-1, 1] ----
        # pf.compute_mean returns (bs, d); take dimension 0 for g.
        mean = self.pf.compute_mean(weights, particles)  # (bs, d)
        g_par = self.pf.phys_model.params[0]             # Parameter object for g
        mean_g_norm = tf.expand_dims(
            normalize(mean[:, 0], g_par.bounds), axis=1
        )  # (bs, 1)

        # ---- 2.  Posterior std of g, log-encoded to ≈ [-1, 1] ----
        # Formula from StatelessMetrology: -2/10 * log(σ) - 1
        # where the range σ ∈ [e^(-10), 1] maps to [-1, 1].
        cov = self.pf.compute_covariance(weights, particles)  # (bs, d, d)
        var_g = cov[:, 0, 0]                                  # (bs,)
        shift = tf.cast(1e-35 if prec == "float32" else 1e-300, prec)
        std_g = tf.sqrt(tf.abs(var_g) + shift)                # (bs,)
        log_std_g = tf.math.log(std_g + shift)                # (bs,)
        # normalize log(std) from (-10, 0) → (-1, 1)
        log_std_g_norm = tf.expand_dims(
            normalize(log_std_g, (-10.0, 0.0)), axis=1
        )  # (bs, 1)

        # ---- 3.  Step counter, normalised to [-1, 1] ----
        step_float = tf.cast(meas_step, prec)                 # (bs, 1)
        step_norm = normalize(step_float, (0, pars.num_steps))

        # ---- 4.  Resources used, normalised to [-1, 1] ----
        res_norm = normalize(used_resources, (0, pars.max_resources))

        # ---- Concatenate ----
        input_tensor = tf.concat(
            [mean_g_norm, log_std_g_norm, step_norm, res_norm],
            axis=1,
            name="gravity_input",
        )  # (bs, 4)

        return input_tensor

    # ------------------------------------------------------------------
    # loss_function — MSE of the posterior mean estimator
    # ------------------------------------------------------------------

    def loss_function(
        self,
        weights: Tensor,
        particles: Tensor,
        true_values: Tensor,
        used_resources: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        r"""MSE loss: :math:`({\hat g} - g_{\rm true})^2`.

        Parameters
        ----------
        weights : Tensor, shape (bs, np)
        particles : Tensor, shape (bs, np, d)
        true_values : Tensor, shape (bs, 1, d)
            True parameter values used in the simulation.
        used_resources : Tensor, shape (bs, 1)
        meas_step : Tensor, shape (bs, 1), dtype=int32

        Returns
        -------
        Tensor, shape (bs, 1)
            Per-batch MSE in (m/s²)².
        """
        mean = self.pf.compute_mean(weights, particles)  # (bs, d)
        g_hat = mean[:, 0]                                # (bs,)
        g_true = true_values[:, 0, 0]                    # (bs,)
        mse = tf.square(g_hat - g_true)                  # (bs,)
        return tf.expand_dims(mse, axis=1)               # (bs, 1)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.simpars.sim_name


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def build_controller(
    input_size: int,
    phys_model: GravityStatelessPhysicalModel,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
) -> tf.keras.Model:
    """Build an MLP controller for the single-PF gravity simulation.

    Architecture::

        Input(input_size)
        → Dense(128, tanh)
        → Dense(128, tanh)
        → Dense(64,  tanh)
        → Dense(3,   tanh)          # raw tanh ∈ [-1, 1]
        → ControlScalingLayer       # → physical units

    The output controls are:

    * ``T_s``          — interrogation time (s), range ``T_range_s``
    * ``Bp_kTm``       — MFG gradient (kT/m), range ``Bp_range_kTm``
    * ``mw_phase_rad`` — microwave phase (rad), range ``[-π, π]``

    Parameters
    ----------
    input_size : int
        Number of scalars in the controller input tensor.
    phys_model : GravityStatelessPhysicalModel
        Used to read physical range constants from ``phys_model.cfg``.
    hidden_sizes : tuple of int, optional
        Sizes of the hidden Dense layers. Default: (128, 128, 64).

    Returns
    -------
    tf.keras.Model
        Keras functional model with trainable weights.
    """
    cfg = phys_model.cfg
    prec = cfg.prec
    dtype = tf.float32 if prec == "float32" else tf.float64

    # Physical control ranges
    T_min, T_max = cfg.T_range_s
    Bp_min, Bp_max = cfg.Bp_range_kTm
    phi_min, phi_max = -pi, pi

    # Mid-points and half-widths for affine rescaling
    T_mid = 0.5 * (T_min + T_max)
    T_half = 0.5 * (T_max - T_min)
    Bp_mid = 0.5 * (Bp_min + Bp_max)
    Bp_half = 0.5 * (Bp_max - Bp_min)
    phi_mid = 0.5 * (phi_min + phi_max)   # = 0
    phi_half = 0.5 * (phi_max - phi_min)  # = π

    class ControlScalingLayer(tf.keras.layers.Layer):
        """Scales NN tanh output ``[-1,1]³`` to physical control ranges."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._T_mid = tf.constant(T_mid, dtype=dtype)
            self._T_half = tf.constant(T_half, dtype=dtype)
            self._Bp_mid = tf.constant(Bp_mid, dtype=dtype)
            self._Bp_half = tf.constant(Bp_half, dtype=dtype)
            self._phi_mid = tf.constant(phi_mid, dtype=dtype)
            self._phi_half = tf.constant(phi_half, dtype=dtype)

        def call(self, x: Tensor) -> Tensor:
            T_s = self._T_mid + self._T_half * x[:, 0:1]
            Bp = self._Bp_mid + self._Bp_half * x[:, 1:2]
            phi = self._phi_mid + self._phi_half * x[:, 2:3]
            return tf.concat([T_s, Bp, phi], axis=1)

    # Build functional model
    inputs = tf.keras.Input(shape=(input_size,), dtype=dtype)
    x = inputs
    for h_size in hidden_sizes:
        x = tf.keras.layers.Dense(h_size, activation="tanh", dtype=dtype)(x)
    x = tf.keras.layers.Dense(3, activation="tanh", dtype=dtype)(x)
    outputs = ControlScalingLayer(dtype=dtype)(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="gravity_single_pf_controller"
    )
    return model


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_gravity_simulation(
    batchsize: int,
    cfg: GravimeterConfig,
    simpars: SimulationParameters,
    rangen: tf.random.Generator,
    num_particles: int = 1024,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
    resample_threshold: float = 0.5,
    resample_fraction: float = 0.98,
    alpha: float = 0.5,
    beta: float = 0.98,
    gamma: float = 1.0,
) -> Tuple[GravitySimulation, ParticleFilter, tf.keras.Model]:
    """Build and wire all components of the single-PF gravity simulation.

    Creates:

    1. :class:`~gravimeter_model.GravityStatelessPhysicalModel`
    2. :class:`~.ParticleFilter` (single instance)
    3. MLP controller via :func:`build_controller`
    4. :class:`GravitySimulation`

    The controller is warmed up (weights built) and the particle filter is
    **not** reset here — ``simulation.execute()`` calls ``pf.reset()``
    internally at the start of each training episode.

    Parameters
    ----------
    batchsize : int
        Number of parallel estimation episodes.
    cfg : GravimeterConfig
        Sensor configuration.
    simpars : SimulationParameters
        qsensoropt training flags.
    rangen : tf.random.Generator
        Random number generator.
    num_particles : int, optional
        Number of PF particles. Default: 1024.
    hidden_sizes : tuple of int, optional
        MLP hidden layer widths. Default: (128, 128, 64).
    resample_threshold : float, optional
        Effective-N threshold for triggering resampling. Default: 0.5.
    resample_fraction : float, optional
        Fraction of batch that must need resampling to trigger resampling.
        Default: 0.98.
    alpha : float, optional
        Soft-resampling mixing coefficient. Default: 0.5.
    beta : float, optional
        Liu-West jitter intensity (0 = no jitter, 1 = full jitter). Default: 0.98.
    gamma : float, optional
        Fraction of particles resampled (vs. freshly drawn). Default: 1.0.

    Returns
    -------
    simulation : GravitySimulation
    particle_filter : ParticleFilter
    controller : tf.keras.Model
    """
    # Physical model
    phys_model = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)

    # Particle filter
    pf = ParticleFilter(
        num_particles=num_particles,
        phys_model=phys_model,
        resampling_allowed=True,
        resample_threshold=resample_threshold,
        resample_fraction=resample_fraction,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        scibior_trick=True,
        trim=True,
        prec=cfg.prec,
    )

    # MLP controller
    controller = build_controller(
        input_size=INPUT_SIZE,
        phys_model=phys_model,
        hidden_sizes=hidden_sizes,
    )

    # Warm up controller (builds Keras weights before the tf.function trace)
    dtype = tf.float32 if cfg.prec == "float32" else tf.float64
    dummy = tf.zeros((batchsize, INPUT_SIZE), dtype=dtype)
    _ = controller(dummy)

    # Wrap controller in a lambda so the simulation calls it as:
    #   controls = control_strategy(input_tensor)
    def control_strategy(input_tensor: Tensor) -> Tensor:
        return controller(input_tensor)

    # Assemble simulation
    simulation = GravitySimulation(
        particle_filter=pf,
        phys_model=phys_model,
        control_strategy=control_strategy,
        simpars=simpars,
    )

    return simulation, pf, controller
