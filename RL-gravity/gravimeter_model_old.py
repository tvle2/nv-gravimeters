# gravimeter_model.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi
from pathlib import Path
from typing import Optional, Sequence

import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

import traceback
from typing import Iterable

# -----------------------------------------------------------------------------
# Robust imports: use qsensoropt if installed; otherwise load local flat files
# as a synthetic package so that their relative imports continue to work.
# -----------------------------------------------------------------------------


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
    from qsensoropt.parameter import Parameter
    from qsensoropt.physical_model import Control
    from qsensoropt.stateless_phys_model import StatelessPhysicalModel
    from qsensoropt.particle_filter import ParticleFilter
    from qsensoropt.stateless_metrology import StatelessMetrology
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
    from qsensoropt.utils import (
        train,
        performance_evaluation,
        store_input_control,
        sqrt_hmatrix,
        get_seed,
    )
except Exception:
    Parameter = _load_local_qsensoropt_module("parameter").Parameter
    Control = _load_local_qsensoropt_module("physical_model").Control
    StatelessPhysicalModel = _load_local_qsensoropt_module("stateless_phys_model").StatelessPhysicalModel
    ParticleFilter = _load_local_qsensoropt_module("particle_filter").ParticleFilter
    StatelessMetrology = _load_local_qsensoropt_module("stateless_metrology").StatelessMetrology
    SimulationParameters = _load_local_qsensoropt_module("simulation_parameters").SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay
    _utils = _load_local_qsensoropt_module("utils")
    train = _utils.train
    performance_evaluation = _utils.performance_evaluation
    store_input_control = _utils.store_input_control
    sqrt_hmatrix = _utils.sqrt_hmatrix
    get_seed = _utils.get_seed


# -----------------------------------------------------------------------------
# qsensoropt deterministic 1D sqrt_hmatrix patch
#
# TensorFlow deterministic GPU ops do not support SVD for matrices with 1 column.
# In our single-parameter gravity model, PF resampling can produce 1x1 covariance
# matrices. For that case, the matrix square root is analytic:
#
#     sqrt([[v]]) = [[sqrt(max(v, 0))]]
#
# We patch qsensoropt's sqrt_hmatrix in-place so the rest of qsensoropt's
# particle-filter code remains unchanged.
# -----------------------------------------------------------------------------

_ORIGINAL_SQRT_HMATRIX = sqrt_hmatrix

def sqrt_hmatrix_safe(hmat: Tensor) -> Tensor:
    """
    Drop-in replacement for qsensoropt.utils.sqrt_hmatrix that avoids SVD for 1x1
    matrices while preserving the original behavior for higher dimensions.
    """
    # Symmetrize to suppress tiny numerical asymmetries.
    hmat = 0.5 * (hmat + tf.linalg.matrix_transpose(hmat))
    dtype = hmat.dtype

    def _scalar_case() -> Tensor:
        v = tf.maximum(hmat[..., 0, 0], tf.cast(0.0, dtype))
        s = tf.sqrt(v)
        return s[..., None, None]

    static_last_dim = hmat.shape[-1]

    if static_last_dim == 1:
        return _scalar_case()

    if static_last_dim is not None:
        return _ORIGINAL_SQRT_HMATRIX(hmat)

    return tf.cond(
        tf.equal(tf.shape(hmat)[-1], 1),
        _scalar_case,
        lambda: _ORIGINAL_SQRT_HMATRIX(hmat),
    )

# Rebind local name used in this file
sqrt_hmatrix = sqrt_hmatrix_safe

# Patch already-imported qsensoropt modules in-place so ParticleFilter.resample()
# uses the safe version too.
for _mod_name in (
    "qsensoropt.utils",
    "qsensoropt.particle_filter",
    "_qsensoropt_local.utils",
    "_qsensoropt_local.particle_filter",
):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None and hasattr(_mod, "sqrt_hmatrix"):
        setattr(_mod, "sqrt_hmatrix", sqrt_hmatrix_safe)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def safe_clip_prob(x: Tensor, eps: float = 1e-9) -> Tensor:
    eps_t = tf.cast(eps, x.dtype)
    return tf.clip_by_value(x, eps_t, 1.0 - eps_t)

def _assert_finite_and_print(name: str, t: Tensor) -> Tensor:
    tf.debugging.assert_all_finite(t, f"{name} has NaN/Inf")
    tf.print(f"[debug:{name}] sample0=", t[0], summarize=-1)
    return t

def wrap_to_pi_tf(x: Tensor) -> Tensor:
    two_pi = tf.cast(2.0 * pi, x.dtype)
    return tf.math.floormod(x + tf.cast(pi, x.dtype), two_pi) - tf.cast(pi, x.dtype)



def normalize_to_minus1_plus1(x: Tensor, bounds: tuple[float, float]) -> Tensor:
    lo = tf.cast(bounds[0], x.dtype)
    hi = tf.cast(bounds[1], x.dtype)
    width = tf.maximum(hi - lo, tf.cast(1e-18, x.dtype))
    y = 2.0 * (x - lo) / width - 1.0
    return tf.clip_by_value(y, -1.2, 1.2)



def _latest_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None



def _decode_unit_interval_np(x: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (x + 1.0), 0.0, 1.0)



def _decode_affine_np(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return lo + 0.5 * (x + 1.0) * (hi - lo)



def _decode_logstd_np(x: np.ndarray) -> np.ndarray:
    log_std = -12.0 + 0.5 * (x + 1.0) * 12.0
    return np.exp(log_std)

DEBUG_DEPLOY_NUMERICS = False
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GravimeterConfig:
    # physical constants
    omega_rad_s: float = 2.0 * pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * pi * 28e9
    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34
    kT_to_T: float = 1e3

    # latent state
    g_range: tuple[float, float] = (9.7806, 9.825)
    infer_mfg_bias: bool = False
    beta_B_range: tuple[float, float] = (-0.10, 0.10)

    # effective control box (paper-centered single-NV regime)
    T_range_s: tuple[float, float] = (3.0e-4, 1.2e-3)
    Bp_range_kTm: tuple[float, float] = (20.0, 80.0)
    delta_max_rad: float = pi / 2.0

    # stochastic control / visibility noise
    mfg_rel_noise_bound: float = 0.0
    mfg_noise_quad_points: int = 1
    sigma_omega_rel: float = 0.0
    trap_visibility_mode: str = "none"   # none | small_noise_avg | exact_single_delta
    trap_noise_quad_points: int = 9

    # hidden fixed bias (only for deliberate misspecification tests)
    fixed_mfg_rel_bias: float = 0.0
    apply_fixed_mfg_bias_in_model: bool = True

    # optional extra channels
    T2_spin_s: Optional[float] = None
    readout_flip_prob: float = 0.0

    # resource model
    dead_time_s: float = 0.0
    mfg_resource_cost_s_at_ref: float = 0.0
    mfg_resource_ref_kTm: float = 50.0

    # precision
    prec: str = "float32"

    @property
    def tau_s(self) -> float:
        return 2.0 * pi / self.omega_rad_s


@dataclass(frozen=True)
class GravityPFConfig:
    num_particles: int = 1024
    alpha: float = 0.5
    beta: float = 0.98
    gamma: float = 0.95
    resample_threshold: float = 0.5
    resample_fraction: float = 0.75
    hidden_sizes: tuple[int, ...] = (128, 128, 128)


# -----------------------------------------------------------------------------
# Physical model
# -----------------------------------------------------------------------------


class GravityStatelessPhysicalModel(StatelessPhysicalModel):
    """
    Single-NV reduced readout-probability model.

    Default latent state:
        theta = (g,)

    Calibration-robust latent state:
        theta = (g, beta_B)

    Effective controls:
        x = (T_s, Bp_kTm, mw_phase_rad)
    """

    def __init__(self, batchsize: int, cfg: GravimeterConfig) -> None:
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

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        g = parameters[:, :, 0]
        if self.cfg.infer_mfg_bias:
            beta_B = parameters[:, :, 1]
        else:
            beta_B = tf.zeros_like(g)
        return g, beta_B

    def y0_m(self) -> Tensor:
        cfg = self.cfg
        return tf.cast(
            tf.sqrt(cfg.hbar_J_s / (2.0 * cfg.mass_kg * cfg.omega_rad_s)),
            dtype=cfg.prec,
        )

    def eta(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, Bp_kTm.dtype)
        return (
            tf.cast(cfg.gamma_e_rad_s_T, Bp_kTm.dtype)
            * Bp_T_per_m
            * tf.cast(self.y0_m(), Bp_kTm.dtype)
            / tf.cast(cfg.omega_rad_s, Bp_kTm.dtype)
        )

    def k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * (T_s ** 2) + (
            8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)
        ) * Bp_T_per_m

    def cycle_time_s(self, T_s: Tensor) -> Tensor:
        cfg = self.cfg
        return tf.cast(cfg.dead_time_s + 3.5 * cfg.tau_s, T_s.dtype) + 2.0 * T_s

    def extra_mfg_resource_cost_s(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.mfg_resource_cost_s_at_ref <= 0.0:
            return tf.zeros_like(Bp_kTm)
        bref = tf.cast(cfg.mfg_resource_ref_kTm, Bp_kTm.dtype)
        c_ref = tf.cast(cfg.mfg_resource_cost_s_at_ref, Bp_kTm.dtype)
        return c_ref * (Bp_kTm / tf.maximum(bref, tf.cast(1e-18, Bp_kTm.dtype))) ** 2

    def total_resource_cost_s(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        return self.cycle_time_s(T_s) + self.extra_mfg_resource_cost_s(Bp_kTm)

    def trap_visibility_avg_small_noise(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        correction = tf.cast(
            1944.0 * (pi ** 4) * (cfg.sigma_omega_rel ** 4),
            eta.dtype,
        ) * (eta ** 2)
        return tf.clip_by_value(1.0 - correction, 0.0, 1.0)

    def trap_visibility_exact_from_delta_omega(
        self,
        Bp_kTm: Tensor,
        delta_omega_rad_s: Tensor,
    ) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        tau = tf.cast(cfg.tau_s, eta.dtype)
        x = tf.cast(cfg.omega_rad_s, eta.dtype) * (
            -tau * delta_omega_rad_s / tf.cast(cfg.omega_rad_s, eta.dtype)
        )
        amp = 16.0 * eta * tf.cos(x / 4.0) * (tf.sin(3.0 * x / 4.0) ** 2)
        return tf.exp(-0.5 * amp ** 2)

    def _trap_visibility_marginalized(self, Bp_kTm: Tensor) -> Tensor:
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
            Bp_kTm[..., None],
            delta_omega,
        )
        return tf.reduce_sum(vis * w, axis=-1)

    def known_visibility_factor(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg

        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            vis = self._trap_visibility_marginalized(Bp_kTm)
        else:
            raise ValueError(f"Unknown trap_visibility_mode={cfg.trap_visibility_mode}")

        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))

        return tf.clip_by_value(vis, 0.0, 1.0)

    def sample_true_visibility_factor(
        self,
        T_s: Tensor,
        Bp_kTm: Tensor,
        rangen: tf.random.Generator,
    ) -> Tensor:
        cfg = self.cfg

        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            delta_omega = rangen.normal(
                tf.shape(T_s),
                dtype=T_s.dtype,
            ) * tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, T_s.dtype)
            vis = self.trap_visibility_exact_from_delta_omega(Bp_kTm, delta_omega)
        else:
            raise ValueError(f"Unknown trap_visibility_mode={cfg.trap_visibility_mode}")

        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))

        return tf.clip_by_value(vis, 0.0, 1.0)

    def mfg_quadrature(self, dtype) -> tuple[Tensor, Tensor]:
        bound = float(self.cfg.mfg_rel_noise_bound)
        n = int(max(1, self.cfg.mfg_noise_quad_points))

        if bound <= 0.0 or n <= 1:
            return (
                tf.constant([0.0], dtype=dtype),
                tf.constant([1.0], dtype=dtype),
            )

        x_np, w_np = np.polynomial.legendre.leggauss(n)
        nodes = tf.constant(bound * x_np, dtype=dtype)
        weights = tf.constant(0.5 * w_np, dtype=dtype)
        return nodes, weights

    def _sample_mfg_rel_noise(
        self,
        shape,
        rangen: tf.random.Generator,
        dtype,
    ) -> Tensor:
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
        cfg = self.cfg
        if cfg.infer_mfg_bias:
            return 1.0 + beta_B
        if cfg.apply_fixed_mfg_bias_in_model:
            return tf.cast(1.0 + cfg.fixed_mfg_rel_bias, beta_B.dtype)
        return tf.cast(1.0, beta_B.dtype)

    def _measurement_bias_factor(self, beta_B: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.infer_mfg_bias:
            return 1.0 + beta_B
        return tf.cast(1.0 + cfg.fixed_mfg_rel_bias, beta_B.dtype)

    def model(
        self,
        outcomes: Tensor,
        controls: Tensor,
        parameters: Tensor,
        meas_step: Tensor,
        num_systems: int = 1,
    ) -> Tensor:
        del meas_step, num_systems

        T_s = controls[:, :, 0]
        Bp_commanded_kTm = controls[:, :, 1]
        mw_phase = controls[:, :, 2]

        g, beta_B = self._split_parameters(parameters)
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
            p_plus = tf.reduce_sum(p_plus_nodes * eps_weights, axis=-1)
            p_plus = safe_clip_prob(p_plus)

        y = outcomes[:, :, 0]
        prob = tf.where(y > 0.5, p_plus, 1.0 - p_plus)
        return safe_clip_prob(prob)

    def perform_measurement(
        self,
        controls: Tensor,
        parameters: Tensor,
        meas_step: Tensor,
        rangen: tf.random.Generator,
    ) -> tuple[Tensor, Tensor]:
        del meas_step

        T_s = controls[:, 0, 0]
        Bp_commanded_kTm = controls[:, 0, 1]
        mw_phase = controls[:, 0, 2]

        g, beta_B = self._split_parameters(parameters)
        Bp_base_kTm = Bp_commanded_kTm * self._measurement_bias_factor(beta_B[:, 0])
        eps = self._sample_mfg_rel_noise(tf.shape(Bp_base_kTm), rangen, Bp_base_kTm.dtype)
        Bp_kTm = Bp_base_kTm * (1.0 + eps)

        vis_true = self.sample_true_visibility_factor(T_s, Bp_kTm, rangen)
        theta = self.k_g(T_s, Bp_kTm) * g[:, 0] + mw_phase
        p_plus = safe_clip_prob(0.5 * (1.0 + vis_true * tf.cos(theta)))

        if self.cfg.readout_flip_prob > 0.0:
            flip = tf.cast(self.cfg.readout_flip_prob, p_plus.dtype)
            p_plus = (1.0 - flip) * p_plus + flip * (1.0 - p_plus)
            p_plus = safe_clip_prob(p_plus)

        u = rangen.uniform(
            shape=tf.shape(p_plus),
            minval=0.0,
            maxval=1.0,
            dtype=p_plus.dtype,
        )
        y = tf.cast(u < p_plus, p_plus.dtype)

        outcomes = tf.expand_dims(tf.expand_dims(y, axis=1), axis=2)
        log_prob = tf.expand_dims(
            tf.math.log(tf.where(y > 0.5, p_plus, 1.0 - p_plus)),
            axis=1,
        )
        return outcomes, log_prob

    def count_resources(
        self,
        resources: Tensor,
        outcomes: Tensor,
        controls: Tensor,
        true_values: Tensor,
        meas_step: Tensor,
    ) -> Tensor:
        del outcomes, true_values, meas_step

        T_s = controls[..., 0]
        Bp_kTm = controls[..., 1]
        if T_s.shape.rank == 1:
            T_s = T_s[:, None]
            Bp_kTm = Bp_kTm[:, None]

        new_resources = resources + self.total_resource_cost_s(T_s, Bp_kTm)

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(T_s, "count_resources got NaN/Inf T_s")
            tf.debugging.assert_all_finite(Bp_kTm, "count_resources got NaN/Inf Bp_kTm")
            tf.debugging.assert_all_finite(new_resources, "count_resources produced NaN/Inf new_resources")
            tf.print(
                "[debug:count_resources]",
                "T0=", T_s[0, 0],
                "B0=", Bp_kTm[0, 0],
                "oldR0=", resources[0, 0],
                "newR0=", new_resources[0, 0],
                summarize=-1,
            )

        return new_resources


# -----------------------------------------------------------------------------
# Continuous particle filter
# -----------------------------------------------------------------------------


class GravityParticleFilter(ParticleFilter):
    def __init__(
        self,
        phys_model: GravityStatelessPhysicalModel,
        cfg_pf: GravityPFConfig,
        *,
        prec: str = "float32",
    ) -> None:
        self.cfg_pf = cfg_pf
        super().__init__(
            num_particles=int(cfg_pf.num_particles),
            phys_model=phys_model,
            resampling_allowed=True,
            resample_threshold=float(cfg_pf.resample_threshold),
            resample_fraction=float(cfg_pf.resample_fraction),
            alpha=float(cfg_pf.alpha),
            beta=float(cfg_pf.beta),
            gamma=float(cfg_pf.gamma),
            scibior_trick=True,
            trim=True,
            prec=prec,
        )

    def reset(self, rangen: tf.random.Generator):
        cfg = self.phys_model.cfg
        dtype = cfg.prec

        ug = rangen.uniform((self.bs, self.np, 1), dtype=dtype)
        g = tf.cast(cfg.g_range[0], dtype) + ug * tf.cast(cfg.g_range[1] - cfg.g_range[0], dtype)

        if cfg.infer_mfg_bias:
            ub = rangen.uniform((self.bs, self.np, 1), dtype=dtype)
            beta_B = tf.cast(cfg.beta_B_range[0], dtype) + ub * tf.cast(cfg.beta_B_range[1] - cfg.beta_B_range[0], dtype)
            particles = tf.concat([g, beta_B], axis=2)
        else:
            particles = g

        weights = tf.ones((self.bs, self.np), dtype=dtype) / tf.cast(self.np, dtype)
        return weights, particles


# -----------------------------------------------------------------------------
# Posterior-summary metrology
# -----------------------------------------------------------------------------


class GravityPosteriorSummaryMetrology(StatelessMetrology):
    def __init__(
        self,
        particle_filter: GravityParticleFilter,
        phys_model: GravityStatelessPhysicalModel,
        control_strategy,
        simpars: SimulationParameters,
        cov_weight_matrix: Optional[list[list[float]]] = None,
    ) -> None:
        super().__init__(particle_filter, phys_model, control_strategy, simpars, cov_weight_matrix)

        self.has_bias = phys_model.cfg.infer_mfg_bias
        self.input_name = [
            "Mean_g",
            "LogStd_g",
        ]
        if self.has_bias:
            self.input_name += ["Mean_beta_B", "LogStd_beta_B", "Corr_gbeta"]
        self.input_name += [
            "EffN01",
            "Entropy01",
            "StepOverMaxStep",
            "ResOverMaxRes",
            "PhasorLowRe",
            "PhasorLowIm",
            "PhasorMidRe",
            "PhasorMidIm",
            "PhasorHighRe",
            "PhasorHighIm",
        ]
        self.input_size = len(self.input_name)

    def _posterior_variance(self, w: Tensor, x: Tensor) -> Tensor:
        mean_x = tf.reduce_sum(w * x, axis=1, keepdims=True)
        mean_x2 = tf.reduce_sum(w * x * x, axis=1, keepdims=True)
        return tf.maximum(mean_x2 - mean_x * mean_x, tf.cast(0.0, w.dtype))

    def _posterior_correlation(self, w: Tensor, x: Tensor, y: Tensor) -> Tensor:
        mx = tf.reduce_sum(w * x, axis=1, keepdims=True)
        my = tf.reduce_sum(w * y, axis=1, keepdims=True)
        dx = x - mx
        dy = y - my
        cov = tf.reduce_sum(w * dx * dy, axis=1, keepdims=True)
        vx = tf.maximum(tf.reduce_sum(w * dx * dx, axis=1, keepdims=True), tf.cast(1e-18, w.dtype))
        vy = tf.maximum(tf.reduce_sum(w * dy * dy, axis=1, keepdims=True), tf.cast(1e-18, w.dtype))
        return tf.math.divide_no_nan(cov, tf.sqrt(vx * vy))

    def _anchor_controls(self, dtype) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        cfg = self.phys_model.cfg
        T_min, T_max = cfg.T_range_s
        B_min, B_max = cfg.Bp_range_kTm
        T_mid = float(np.sqrt(T_min * T_max))
        B_mid = float(np.sqrt(B_min * B_max))
        return (
            (tf.cast(T_min, dtype), tf.cast(B_min, dtype)),
            (tf.cast(T_mid, dtype), tf.cast(B_mid, dtype)),
            (tf.cast(T_max, dtype), tf.cast(B_max, dtype)),
        )

    def _noisy_global_phasor(self, weights: Tensor, particles: Tensor, T_ref: Tensor, B_cmd_ref: Tensor) -> tuple[Tensor, Tensor]:
        phys = self.phys_model
        dtype = weights.dtype

        g = particles[:, :, 0]
        if phys.cfg.infer_mfg_bias:
            beta_B = particles[:, :, 1]
            B_base = B_cmd_ref[None, None] * (1.0 + beta_B)
        else:
            bias = tf.cast(1.0 + phys.cfg.fixed_mfg_rel_bias, dtype) if phys.cfg.apply_fixed_mfg_bias_in_model else tf.cast(1.0, dtype)
            B_base = B_cmd_ref[None, None] * bias

        eps_nodes, eps_weights = phys.mfg_quadrature(dtype=dtype)

        if int(eps_nodes.shape[0]) == 1:
            T = tf.ones_like(B_base) * T_ref
            B = B_base
            vis = phys.known_visibility_factor(T, B)
            phase = phys.k_g(T, B) * g
            c_re = tf.reduce_sum(weights * vis * tf.cos(phase), axis=1, keepdims=True)
            c_im = tf.reduce_sum(weights * vis * tf.sin(phase), axis=1, keepdims=True)
            return tf.clip_by_value(c_re, -1.0, 1.0), tf.clip_by_value(c_im, -1.0, 1.0)

        B = B_base[:, :, None] * (1.0 + eps_nodes[None, None, :])
        T = tf.ones_like(B) * T_ref
        vis = phys.known_visibility_factor(T, B)
        phase = phys.k_g(T, B) * g[:, :, None]
        c_re_nodes = vis * tf.cos(phase)
        c_im_nodes = vis * tf.sin(phase)
        c_re = tf.reduce_sum(weights[:, :, None] * c_re_nodes * eps_weights[None, None, :], axis=(1, 2), keepdims=True)
        c_im = tf.reduce_sum(weights[:, :, None] * c_im_nodes * eps_weights[None, None, :], axis=(1, 2), keepdims=True)
        return tf.clip_by_value(c_re[:, 0, :], -1.0, 1.0), tf.clip_by_value(c_im[:, 0, :], -1.0, 1.0)

    def generate_input(
        self,
        weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen,
    ) -> Tensor:
        del rangen
        dtype = self.phys_model.cfg.prec
        eps = tf.cast(1e-18, dtype)

        w = tf.cast(weights, dtype)
        p = tf.cast(particles, dtype)

        g = p[:, :, 0]
        mean_g = tf.reduce_sum(w * g, axis=1, keepdims=True)
        var_g = self._posterior_variance(w, g)
        logstd_g = tf.math.log(tf.sqrt(tf.maximum(var_g, eps)))

        feats = [
            normalize_to_minus1_plus1(mean_g, self.phys_model.cfg.g_range),
            normalize_to_minus1_plus1(logstd_g, (-12.0, 0.0)),
        ]

        if self.has_bias:
            beta_B = p[:, :, 1]
            mean_b = tf.reduce_sum(w * beta_B, axis=1, keepdims=True)
            var_b = self._posterior_variance(w, beta_B)
            logstd_b = tf.math.log(tf.sqrt(tf.maximum(var_b, eps)))
            corr = tf.clip_by_value(self._posterior_correlation(w, g, beta_B), -1.0, 1.0)
            feats += [
                normalize_to_minus1_plus1(mean_b, self.phys_model.cfg.beta_B_range),
                normalize_to_minus1_plus1(logstd_b, (-12.0, 0.0)),
                corr,
            ]

        eff_n = tf.math.divide_no_nan(
            tf.cast(1.0, dtype) / tf.reduce_sum(w * w, axis=1, keepdims=True),
            tf.cast(tf.shape(w)[1], dtype),
        )
        entropy = -tf.reduce_sum(w * tf.math.log(w + eps), axis=1, keepdims=True)
        entropy = tf.math.divide_no_nan(entropy, tf.math.log(tf.cast(tf.shape(w)[1], dtype)))

        feats += [
            tf.clip_by_value(eff_n, 0.0, 1.0),
            tf.clip_by_value(entropy, 0.0, 1.0),
            _decode_tensor_unit(meas_step, float(self.simpars.num_steps)),
            _decode_tensor_unit(used_resources, float(self.simpars.max_resources)),
        ]

        for T_ref, B_ref in self._anchor_controls(dtype):
            ph_re, ph_im = self._noisy_global_phasor(w, p, T_ref, B_ref)
            feats += [ph_re, ph_im]

        # return tf.concat(feats, axis=1)
        input_tensor = tf.concat(feats, axis=1)

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(input_tensor, "generate_input produced NaN/Inf")
            tf.print(
                "[debug:generate_input]",
                "mean_g0=", mean_g[0, 0],
                "var_g0=", var_g[0, 0],
                "effN0=", eff_n[0, 0],
                "entropy0=", entropy[0, 0],
                "input0=", input_tensor[0],
                summarize=-1,
            )

        return input_tensor



def _decode_tensor_unit(x: Tensor, high: float) -> Tensor:
    high_t = tf.cast(high, x.dtype)
    return tf.clip_by_value(tf.math.divide_no_nan(x, high_t), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Continuous controller
# -----------------------------------------------------------------------------


class GravityContinuousController(Model):
    def __init__(
        self,
        cfg: GravimeterConfig,
        input_size: int,
        hidden_sizes: Sequence[int] = (128, 128, 128),
    ) -> None:
        super().__init__(dtype=cfg.prec)
        self.cfg = cfg
        self.input_size = int(input_size)
        self.has_bias = cfg.infer_mfg_bias

        self.trunk = [Dense(h, activation="tanh", dtype=cfg.prec) for h in hidden_sizes]
        self.t_head = Dense(1, activation="tanh", dtype=cfg.prec)
        self.b_head = Dense(1, activation="tanh", dtype=cfg.prec)
        self.phi_fallback_head = Dense(1, activation="tanh", dtype=cfg.prec)
        self.delta_head = Dense(1, activation="tanh", dtype=cfg.prec)

        self(tf.zeros((1, input_size), dtype=cfg.prec))

    def _pm1_to_01(self, z: Tensor) -> Tensor:
        return tf.clip_by_value(0.5 * (z + 1.0), 0.0, 1.0)

    def _map_log_interval(self, z: Tensor, low: float, high: float) -> Tensor:
        u = self._pm1_to_01(z)
        lo = tf.cast(tf.math.log(low), z.dtype)
        hi = tf.cast(tf.math.log(high), z.dtype)
        return tf.exp(lo + u * (hi - lo))

    def _k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * (T_s ** 2) + (
            8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)
        ) * Bp_T_per_m

    def _gain_interp_weights(self, k_cur: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dtype = k_cur.dtype
        eps = tf.cast(1e-18, dtype)
        T_min, T_max = self.cfg.T_range_s
        B_min, B_max = self.cfg.Bp_range_kTm
        T_mid = float(np.sqrt(T_min * T_max))
        B_mid = float(np.sqrt(B_min * B_max))

        k_low = self._k_g(tf.constant([[T_min]], dtype=dtype), tf.constant([[B_min]], dtype=dtype))[:, 0]
        k_mid = self._k_g(tf.constant([[T_mid]], dtype=dtype), tf.constant([[B_mid]], dtype=dtype))[:, 0]
        k_high = self._k_g(tf.constant([[T_max]], dtype=dtype), tf.constant([[B_max]], dtype=dtype))[:, 0]

        logk_cur = tf.math.log(tf.maximum(k_cur, eps))
        logk_anchors = tf.stack([
            tf.math.log(tf.maximum(k_low[0], eps)),
            tf.math.log(tf.maximum(k_mid[0], eps)),
            tf.math.log(tf.maximum(k_high[0], eps)),
        ], axis=0)
        d2 = (logk_cur - logk_anchors[None, :]) ** 2
        w = tf.nn.softmax(-8.0 * d2, axis=1)
        return w[:, 0:1], w[:, 1:2], w[:, 2:3]

    def call(self, input_strategy: Tensor) -> Tensor:
        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(input_strategy, "controller input_strategy has NaN/Inf")
            tf.print(
                "[debug:controller_input]",
                "input0=", input_strategy[0],
                summarize=32,
            )

        x = input_strategy
        for i, layer in enumerate(self.trunk):
            x = layer(x)
            if DEBUG_DEPLOY_NUMERICS:
                tf.debugging.assert_all_finite(x, f"controller trunk layer {i} output has NaN/Inf")
                tf.print(
                    f"[debug:trunk_{i}]",
                    "x0=", x[0],
                    summarize=32,
                )

        zT = self.t_head(x)
        zB = self.b_head(x)
        zPhiFallback = self.phi_fallback_head(x)
        zDelta = self.delta_head(x)

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(zT, "controller zT has NaN/Inf")
            tf.debugging.assert_all_finite(zB, "controller zB has NaN/Inf")
            tf.debugging.assert_all_finite(zPhiFallback, "controller zPhiFallback has NaN/Inf")
            tf.debugging.assert_all_finite(zDelta, "controller zDelta has NaN/Inf")
            tf.print(
                "[debug:heads]",
                "zT0=", zT[0, 0],
                "zB0=", zB[0, 0],
                "zPhiFallback0=", zPhiFallback[0, 0],
                "zDelta0=", zDelta[0, 0],
            )

        T_s = self._map_log_interval(zT, self.cfg.T_range_s[0], self.cfg.T_range_s[1])
        Bp_kTm = self._map_log_interval(zB, self.cfg.Bp_range_kTm[0], self.cfg.Bp_range_kTm[1])

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(T_s, "controller T_s has NaN/Inf after decode")
            tf.debugging.assert_all_finite(Bp_kTm, "controller Bp_kTm has NaN/Inf after decode")

        idx = 2 if not self.has_bias else 5
        # inputs after mean/logstd/(beta stats) are:
        # EffN01, Entropy01, StepOverMaxStep, ResOverMaxRes, then 6 phasor features
        ph_low_re = input_strategy[:, idx + 4: idx + 5]
        ph_low_im = input_strategy[:, idx + 5: idx + 6]
        ph_mid_re = input_strategy[:, idx + 6: idx + 7]
        ph_mid_im = input_strategy[:, idx + 7: idx + 8]
        ph_high_re = input_strategy[:, idx + 8: idx + 9]
        ph_high_im = input_strategy[:, idx + 9: idx + 10]

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(ph_low_re, "controller ph_low_re has NaN/Inf")
            tf.debugging.assert_all_finite(ph_low_im, "controller ph_low_im has NaN/Inf")
            tf.debugging.assert_all_finite(ph_mid_re, "controller ph_mid_re has NaN/Inf")
            tf.debugging.assert_all_finite(ph_mid_im, "controller ph_mid_im has NaN/Inf")
            tf.debugging.assert_all_finite(ph_high_re, "controller ph_high_re has NaN/Inf")
            tf.debugging.assert_all_finite(ph_high_im, "controller ph_high_im has NaN/Inf")

        k_cur = self._k_g(T_s, Bp_kTm)
        w_low, w_mid, w_high = self._gain_interp_weights(k_cur)
        c_re = w_low * ph_low_re + w_mid * ph_mid_re + w_high * ph_high_re
        c_im = w_low * ph_low_im + w_mid * ph_mid_im + w_high * ph_high_im

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(k_cur, "controller k_cur has NaN/Inf")
            tf.debugging.assert_all_finite(c_re, "controller c_re has NaN/Inf")
            tf.debugging.assert_all_finite(c_im, "controller c_im has NaN/Inf")

        phase_lock = wrap_to_pi_tf(tf.cast(pi / 2.0, c_re.dtype) - tf.atan2(c_im, c_re))
        c_mag = tf.sqrt(tf.maximum(c_re * c_re + c_im * c_im, tf.cast(1e-18, c_re.dtype)))
        phase_mix = tf.clip_by_value((c_mag - 0.05) / 0.20, 0.0, 1.0)

        phi_fallback = wrap_to_pi_tf(tf.cast(pi, zPhiFallback.dtype) * zPhiFallback)
        phase_err = wrap_to_pi_tf(phase_lock - phi_fallback)
        base_phase = wrap_to_pi_tf(phi_fallback + phase_mix * phase_err)

        delta = 0.25 * tf.cast(self.cfg.delta_max_rad, zDelta.dtype) * zDelta
        mw_phase = wrap_to_pi_tf(base_phase + delta)

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(phase_lock, "controller phase_lock has NaN/Inf")
            tf.debugging.assert_all_finite(c_mag, "controller c_mag has NaN/Inf")
            tf.debugging.assert_all_finite(phi_fallback, "controller phi_fallback has NaN/Inf")
            tf.debugging.assert_all_finite(mw_phase, "controller mw_phase has NaN/Inf")

        controls = tf.concat([T_s, Bp_kTm, mw_phase], axis=1)

        if DEBUG_DEPLOY_NUMERICS:
            tf.debugging.assert_all_finite(controls, "controller controls have NaN/Inf")
            tf.print(
                "[debug:controller_final]",
                "T0=", T_s[0, 0],
                "B0=", Bp_kTm[0, 0],
                "phi0=", mw_phase[0, 0],
                summarize=-1,
            )

        return controls


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


def default_joint_cov_weight_matrix(cfg: GravimeterConfig) -> list[list[float]]:
    if cfg.infer_mfg_bias:
        return [[1.0, 0.0], [0.0, 0.01]]
    return [[1.0]]



def gravity_only_cov_weight_matrix(cfg: GravimeterConfig) -> list[list[float]]:
    if cfg.infer_mfg_bias:
        return [[1.0, 0.0], [0.0, 0.0]]
    return [[1.0]]



def training_gravity_only_cov_weight_matrix(cfg: GravimeterConfig, scale: float = 1.0e4) -> list[list[float]]:
    if cfg.infer_mfg_bias:
        return [[scale, 0.0], [0.0, 0.0]]
    return [[scale]]



def training_joint_cov_weight_matrix(
    cfg: GravimeterConfig,
    g_scale: float = 1.0e4,
    beta_scale: float = 1.0e2,
) -> list[list[float]]:
    if cfg.infer_mfg_bias:
        return [[g_scale, 0.0], [0.0, beta_scale]]
    return [[g_scale]]



def build_gravity_simulation(
    batchsize: int,
    sim_name: str,
    cfg: GravimeterConfig,
    pf_cfg: GravityPFConfig,
    *,
    max_steps: int = 96,
    max_resources: float = 0.12,
    resources_fraction: float = 1.0,
    initial_lr: float = 3e-4,
    cov_weight_matrix: Optional[list[list[float]]] = None,
    cumulative_loss: bool = False,
    log_loss: bool = False,
    loss_logl_outcomes: bool = True,
    baseline_correction: bool = True,
):
    phys = GravityStatelessPhysicalModel(batchsize=batchsize, cfg=cfg)

    pf = GravityParticleFilter(
        phys_model=phys,
        cfg_pf=pf_cfg,
        prec=cfg.prec,
    )

    simpars = SimulationParameters(
        sim_name=sim_name,
        num_steps=max_steps,
        max_resources=max_resources,
        resources_fraction=resources_fraction,
        prec=cfg.prec,
        stop_gradient_input=True,
        stop_gradient_pf=True,
        cumulative_loss=cumulative_loss,
        log_loss=log_loss,
        loss_logl_outcomes=loss_logl_outcomes,
        loss_logl_controls=False,
        baseline=baseline_correction,
    )

    dummy_net = lambda x: tf.zeros((tf.shape(x)[0], 3), dtype=cfg.prec)
    sim_tmp = GravityPosteriorSummaryMetrology(
        particle_filter=pf,
        phys_model=phys,
        control_strategy=dummy_net,
        simpars=simpars,
        cov_weight_matrix=cov_weight_matrix,
    )

    net = GravityContinuousController(
        cfg=cfg,
        input_size=sim_tmp.input_size,
        hidden_sizes=pf_cfg.hidden_sizes,
    )

    sim = GravityPosteriorSummaryMetrology(
        particle_filter=pf,
        phys_model=phys,
        control_strategy=net,
        simpars=simpars,
        cov_weight_matrix=cov_weight_matrix,
    )

    lr_schedule = InverseSqrtDecay(initial_learning_rate=initial_lr, prec=cfg.prec)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return phys, pf, sim, net, optimizer


# -----------------------------------------------------------------------------
# Export canonicalization
# -----------------------------------------------------------------------------



def _canonicalize_history_export(out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_history.csv")
    if src is None:
        return None
    df = pd.read_csv(src)
    if "Checkpoint" not in df.columns:
        df.insert(0, "Checkpoint", np.arange(1, len(df) + 1, dtype=np.int64))
    dst = out_dir / "training_history.csv"
    df.to_csv(dst, index=False)
    return dst



def _canonicalize_eval_export(out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_eval.csv")
    if src is None:
        return None

    df = pd.read_csv(src)
    preferred_metric_order = ["MSE_g", "Weighted MSE"]
    metric_cols = [c for c in preferred_metric_order if c in df.columns]
    if not metric_cols:
        metric_cols = [c for c in df.columns if c != "Resources"]
    keep = ["Resources"] + metric_cols[:1] if "Resources" in df.columns and metric_cols else list(df.columns)
    df = df[keep]

    dst = out_dir / "branchbank_eval.csv"
    df.to_csv(dst, index=False)
    return dst



def _canonicalize_control_export(sim, out_dir: Path) -> Optional[Path]:
    src = _latest_matching_file(out_dir, "*_ext.csv")
    if src is None:
        return None

    df = pd.read_csv(src)
    cfg = sim.phys_model.cfg

    out: dict[str, np.ndarray] = {}
    for col in ["Estimation", "g", "T_s", "Bp_kTm", "mw_phase_rad"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    if "Mean_g" in df.columns:
        out["Mean_g"] = _decode_affine_np(df["Mean_g"].to_numpy(), cfg.g_range)
    if "LogStd_g" in df.columns:
        out["Std_g"] = _decode_logstd_np(df["LogStd_g"].to_numpy())

    if cfg.infer_mfg_bias:
        if "Mean_beta_B" in df.columns:
            out["Mean_beta_B"] = _decode_affine_np(df["Mean_beta_B"].to_numpy(), cfg.beta_B_range)
        if "LogStd_beta_B" in df.columns:
            out["Std_beta_B"] = _decode_logstd_np(df["LogStd_beta_B"].to_numpy())
        if "Corr_gbeta" in df.columns:
            out["Corr_gbeta"] = df["Corr_gbeta"].to_numpy()

    for col in [
        "EffN01",
        "Entropy01",
        "StepOverMaxStep",
        "ResOverMaxRes",
        "PhasorLowRe",
        "PhasorLowIm",
        "PhasorMidRe",
        "PhasorMidIm",
        "PhasorHighRe",
        "PhasorHighIm",
    ]:
        if col in df.columns:
            out[col] = df[col].to_numpy()

    dst = out_dir / "branchbank_controls.csv"
    pd.DataFrame(out).to_csv(dst, index=False)
    return dst


# -----------------------------------------------------------------------------
# Train / eval wrappers
# -----------------------------------------------------------------------------

def _debug_deploy_snapshot(sim, seed: int, tag: str = "deploy") -> None:
    """
    Run one deploy-mode simulation and print what comes back before any CSV export.
    This tells us whether histories are already zeroed out by qsensoropt.
    """
    rangen = tf.random.Generator.from_seed(seed)
    true_values, history_input, history_controls, history_resources, history_precision = sim.execute(
        rangen, deploy=True
    )

    ns = int(sim.simpars.num_steps)
    bs = int(sim.bs)
    input_size = int(sim.input_size)
    controls_size = int(sim.phys_model.controls_size)

    h_in = history_input.numpy().reshape(ns, bs, input_size)
    h_ctl = history_controls.numpy().reshape(ns, bs, controls_size)
    h_res = history_resources.numpy().reshape(ns, bs, 1)
    h_pre = history_precision.numpy().reshape(ns, bs, 1)

    print(f"[debug:{tag}] history_input shape      = {h_in.shape}")
    print(f"[debug:{tag}] history_controls shape   = {h_ctl.shape}")
    print(f"[debug:{tag}] history_resources shape  = {h_res.shape}")
    print(f"[debug:{tag}] history_precision shape  = {h_pre.shape}")

    print(f"[debug:{tag}] controls finite?  {np.isfinite(h_ctl).all()}")
    print(f"[debug:{tag}] resources finite? {np.isfinite(h_res).all()}")
    print(f"[debug:{tag}] precision finite? {np.isfinite(h_pre).all()}")

    ctrl_nonzero_mask = np.any(np.abs(h_ctl) > 1e-18, axis=2)   # (ns, bs)
    res_positive_mask = h_res[:, :, 0] > 0.0                    # (ns, bs)

    print(f"[debug:{tag}] nonzero control rows = {int(ctrl_nonzero_mask.sum())}")
    print(f"[debug:{tag}] positive resource rows = {int(res_positive_mask.sum())}")

    if ctrl_nonzero_mask.any():
        active_ctl = h_ctl[ctrl_nonzero_mask]
        print(f"[debug:{tag}] active controls min = {active_ctl.min(axis=0)}")
        print(f"[debug:{tag}] active controls max = {active_ctl.max(axis=0)}")
    else:
        print(f"[debug:{tag}] all logged controls are zero")

    if res_positive_mask.any():
        active_res = h_res[:, :, 0][res_positive_mask]
        print(f"[debug:{tag}] positive resources min/max = {active_res.min()} / {active_res.max()}")
    else:
        print(f"[debug:{tag}] all logged resources are <= 0")

    # First step visibility is especially important for the empty-file issue
    print(f"[debug:{tag}] step0 controls sample =\n{h_ctl[0]}")
    print(f"[debug:{tag}] step0 resources sample =\n{h_res[0, :, 0]}")
    print(f"[debug:{tag}] step0 precision sample =\n{h_pre[0, :, 0]}")


def _debug_latest_csv(directory: Path, pattern: str, tag: str) -> None:
    src = _latest_matching_file(directory, pattern)
    if src is None:
        print(f"[debug:{tag}] no file matching {pattern!r} in {directory}")
        return

    print(f"[debug:{tag}] latest file = {src}")
    try:
        df = pd.read_csv(src)
    except Exception as exc:
        print(f"[debug:{tag}] failed to read CSV: {exc}")
        return

    print(f"[debug:{tag}] shape = {df.shape}")
    print(f"[debug:{tag}] columns = {list(df.columns)}")
    if not df.empty:
        print(f"[debug:{tag}] head =\n{df.head(5)}")
        if "Resources" in df.columns:
            res = pd.to_numeric(df["Resources"], errors="coerce")
            print(f"[debug:{tag}] Resources>0 rows = {int((res > 0).sum())}")
            print(f"[debug:{tag}] Resources==0 rows = {int((res == 0).sum())}")
            print(f"[debug:{tag}] Resources finite? = {np.isfinite(res).all()}")
    else:
        print(f"[debug:{tag}] CSV has headers but no rows")


# =============================================================================
# Training-numerics debug helpers
# =============================================================================

DEBUG_TRAIN_NUMERICS = False          # set False after debugging
DEBUG_TRAIN_NUMERIC_ITERS = 8        # small debug run
DEBUG_TRAIN_NUMERIC_ACCUM = 1        # force simple accumulation while debugging


def _np_summary(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "all values are non-finite"
    return (
        f"finite_min={finite.min():.6g}, "
        f"finite_max={finite.max():.6g}, "
        f"finite_mean={finite.mean():.6g}"
    )


def _first_bad_index(arr: np.ndarray):
    bad = ~np.isfinite(arr)
    if not np.any(bad):
        return None
    return tuple(np.argwhere(bad)[0])


def _report_bad_tensor(tag: str, tensor) -> None:
    arr = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)
    idx = _first_bad_index(arr)
    if idx is None:
        print(f"[debug:{tag}] tensor is finite. {_np_summary(arr)}")
        return

    bad_value = arr[idx]
    print(f"[debug:{tag}] NON-FINITE DETECTED")
    print(f"[debug:{tag}] shape={arr.shape}")
    print(f"[debug:{tag}] first_bad_index={idx}")
    print(f"[debug:{tag}] first_bad_value={bad_value}")
    print(f"[debug:{tag}] {_np_summary(arr)}")


def _assert_finite_tensor(tag: str, tensor) -> None:
    arr = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)
    if not np.isfinite(arr).all():
        _report_bad_tensor(tag, tensor)
        raise FloatingPointError(f"{tag} has NaN/Inf")


def _gradient_norm_safe(grad: Optional[Tensor]) -> float:
    if grad is None:
        return float("nan")
    arr = grad.numpy()
    if not np.isfinite(arr).all():
        return float("nan")
    return float(np.linalg.norm(arr.reshape(-1)))


def train_with_numeric_debug(
    simulation,
    optimizer,
    iterations: int,
    network: Optional[Model],
    save_path: str | Path,
    interval_save: int,
    rangen: tf.random.Generator,
    gradient_accumulation: int = 1,
):
    """
    Debug-only replacement for qsensoropt.utils.train.

    It checks, in order:
      1. loss_diff
      2. loss
      3. each raw gradient
      4. each accumulated gradient
      5. each variable after optimizer.apply_gradients()

    so we can identify the first place where NaN/Inf appears.
    """
    if network is None:
        raise ValueError("train_with_numeric_debug requires network to be provided.")

    variables = network.trainable_variables
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    loss_history = []

    def single_iteration(local_rangen):
        with tf.GradientTape() as tape:
            loss_diff, loss = simulation.execute(local_rangen)
        grads = tape.gradient(loss_diff, variables)
        return loss_diff, loss, grads

    for j in range(iterations):
        print("=" * 80)
        print(f"[debug:train] iteration={j}")

        acc_loss = tf.zeros((1,), dtype=simulation.simpars.prec)
        acc_grads = [tf.zeros_like(v) for v in variables]

        # Check variables BEFORE any new update
        for vi, var in enumerate(variables):
            try:
                _assert_finite_tensor(f"var_pre[{vi}] {var.name}", var)
            except FloatingPointError:
                print(f"[debug:train] variable already bad before iteration {j}")
                raise

        for k in range(gradient_accumulation):
            print(f"[debug:train] accumulation_step={k}")

            loss_diff, loss, grads = single_iteration(rangen)

            # 1) loss_diff
            _assert_finite_tensor("loss_diff", loss_diff)

            # 2) loss
            _assert_finite_tensor("loss", loss)

            print(
                f"[debug:train] loss_diff={loss_diff.numpy()} "
                f"loss={loss.numpy()}"
            )

            # 3) raw gradients
            for gi, (var, grad) in enumerate(zip(variables, grads)):
                if grad is None:
                    print(f"[debug:grad_raw[{gi}]] {var.name}: grad is None")
                    continue

                try:
                    _assert_finite_tensor(f"grad_raw[{gi}] {var.name}", grad)
                except FloatingPointError:
                    print(f"[debug:train] first bad tensor is RAW GRADIENT {gi}: {var.name}")
                    raise

                print(
                    f"[debug:grad_raw[{gi}]] {var.name}: "
                    f"norm={_gradient_norm_safe(grad):.6g}"
                )

            acc_loss += loss
            acc_grads = [
                acc_g + (g if g is not None else tf.zeros_like(v))
                for acc_g, g, v in zip(acc_grads, grads, variables)
            ]

        # 4) accumulated gradients
        acc_grads = [g / float(gradient_accumulation) for g in acc_grads]
        acc_loss = acc_loss / float(gradient_accumulation)

        _assert_finite_tensor("acc_loss", acc_loss)

        for gi, (var, grad) in enumerate(zip(variables, acc_grads)):
            try:
                _assert_finite_tensor(f"grad_acc[{gi}] {var.name}", grad)
            except FloatingPointError:
                print(f"[debug:train] first bad tensor is ACCUMULATED GRADIENT {gi}: {var.name}")
                raise

            print(
                f"[debug:grad_acc[{gi}]] {var.name}: "
                f"norm={_gradient_norm_safe(grad):.6g}"
            )

        # Save pre-update weights snapshot for the first few iterations if needed
        if j < 3:
            for vi, var in enumerate(variables[:4]):  # only first few vars to keep logs manageable
                arr = var.numpy()
                print(
                    f"[debug:var_pre_stats[{vi}]] {var.name}: "
                    f"{_np_summary(arr)}"
                )

        # Apply gradients
        optimizer.apply_gradients(zip(acc_grads, variables))
        print("[debug:train] apply_gradients done")

        # 5) variables AFTER update
        for vi, var in enumerate(variables):
            try:
                _assert_finite_tensor(f"var_post[{vi}] {var.name}", var)
            except FloatingPointError:
                print(
                    f"[debug:train] first bad value appears AFTER apply_gradients "
                    f"in variable {vi}: {var.name}"
                )
                raise

        loss_history.append(float(np.asarray(acc_loss.numpy()).reshape(-1)[0]))

        if (j + 1) % interval_save == 0:
            # Save a lightweight debug history CSV
            hist_path = save_path / "training_history_debug.csv"
            pd.DataFrame(
                {
                    "Checkpoint": np.arange(1, len(loss_history) + 1, dtype=np.int64),
                    "Loss": loss_history,
                }
            ).to_csv(hist_path, index=False)
            print(f"[debug:train] wrote {hist_path}")

    # Final save
    hist_path = save_path / "training_history_debug.csv"
    pd.DataFrame(
        {
            "Checkpoint": np.arange(1, len(loss_history) + 1, dtype=np.int64),
            "Loss": loss_history,
        }
    ).to_csv(hist_path, index=False)
    print(f"[debug:train] finished cleanly, wrote {hist_path}")

def train_gravity_modelaware(
    out_dir: str | Path,
    *,
    batchsize: int = 128,
    iterations: int = 2000,
    interval_save: int = 128,
    cfg: GravimeterConfig = GravimeterConfig(),
    pf_cfg: GravityPFConfig = GravityPFConfig(),
    sim_name: str = "gravimeter_modelaware",
    max_steps: int = 96,
    max_resources: float = 0.12,
    resources_fraction: float = 0.98,
    initial_lr: float = 3e-4,
    seed: int = 123,
    cumulative_loss: bool = False,
    log_loss: bool = False,
    loss_logl_outcomes: bool = False,
    baseline_correction: bool = True,
    gradient_accumulation: int = 4,
    cov_weight_matrix: Optional[list[list[float]]] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cov_weight_matrix is None:
        cov_weight_matrix = default_joint_cov_weight_matrix(cfg)

    phys, pf, sim, net, optimizer = build_gravity_simulation(
        batchsize=batchsize,
        sim_name=sim_name,
        cfg=cfg,
        pf_cfg=pf_cfg,
        max_steps=max_steps,
        max_resources=max_resources,
        resources_fraction=resources_fraction,
        initial_lr=initial_lr,
        cov_weight_matrix=cov_weight_matrix,
        cumulative_loss=cumulative_loss,
        log_loss=log_loss,
        loss_logl_outcomes=loss_logl_outcomes,
        baseline_correction=baseline_correction,
    )

    rangen = tf.random.Generator.from_seed(seed)
    # train(
    #     simulation=sim,
    #     optimizer=optimizer,
    #     iterations=iterations,
    #     save_path=str(out_dir),
    #     interval_save=interval_save,
    #     network=net,
    #     gradient_accumulation=gradient_accumulation,
    #     xla_compile=False,
    #     rangen=rangen,
    # )
    if DEBUG_TRAIN_NUMERICS:
        print("[debug:train] using train_with_numeric_debug")
        train_with_numeric_debug(
            simulation=sim,
            optimizer=optimizer,
            iterations=min(iterations, DEBUG_TRAIN_NUMERIC_ITERS),
            save_path=out_dir,
            interval_save=1,
            network=net,
            gradient_accumulation=DEBUG_TRAIN_NUMERIC_ACCUM,
            rangen=rangen,
        )
    else:
        train(
            simulation=sim,
            optimizer=optimizer,
            iterations=iterations,
            save_path=str(out_dir),
            interval_save=interval_save,
            network=net,
            gradient_accumulation=gradient_accumulation,
            xla_compile=False,
            rangen=rangen,
        )
    

    _canonicalize_history_export(out_dir)
    return phys, pf, sim, net, optimizer



def export_control_history(sim, out_dir: str | Path, iterations: int = 32, seed: int = 999):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rangen = tf.random.Generator.from_seed(seed)
    store_input_control(
        simulation=sim,
        data_dir=str(out_dir),
        iterations=iterations,
        xla_compile=False,
        rangen=rangen,
    )
    return _canonicalize_control_export(sim=sim, out_dir=out_dir)


def evaluate_precision(
    sim,
    out_dir: str | Path,
    iterations: int = 64,
    seed: int = 2025,
    delta_resources: float | None = None,
    metric_label: str = "Weighted MSE",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rangen = tf.random.Generator.from_seed(seed)

    if delta_resources is None:
        delta_resources = float(sim.simpars.max_resources) / 80.0

    performance_evaluation(
        simulation=sim,
        iterations=iterations,
        data_dir=str(out_dir),
        xla_compile=False,
        precision_fit=None,
        delta_resources=delta_resources,
        y_label=metric_label,
        rangen=rangen,
    )
    return _canonicalize_eval_export(out_dir)
# -----------------------------------------------------------------------------
# Convenience defaults + config dump helper
# -----------------------------------------------------------------------------


def default_cfg() -> GravimeterConfig:
    return GravimeterConfig()



def default_pf_cfg() -> GravityPFConfig:
    return GravityPFConfig()



def dump_run_config(path: str | Path, *, cfg: GravimeterConfig, pf_cfg: GravityPFConfig, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gravimeter_config": asdict(cfg),
        "particle_filter_config": asdict(pf_cfg),
        **kwargs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
