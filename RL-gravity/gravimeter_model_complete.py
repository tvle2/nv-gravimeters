# gravimeter_model_complete.py
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor


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
except Exception:
    Parameter = _load_local_qsensoropt_module("parameter").Parameter
    Control = _load_local_qsensoropt_module("physical_model").Control
    StatelessPhysicalModel = _load_local_qsensoropt_module("stateless_phys_model").StatelessPhysicalModel


def safe_clip_prob(x: Tensor, eps: float = 1e-9) -> Tensor:
    eps_t = tf.cast(eps, x.dtype)
    return tf.clip_by_value(x, eps_t, 1.0 - eps_t)


def wrap_to_pi_tf(x: Tensor) -> Tensor:
    two_pi = tf.cast(2.0 * pi, x.dtype)
    return tf.math.floormod(x + tf.cast(pi, x.dtype), two_pi) - tf.cast(pi, x.dtype)


@dataclass(frozen=True)
class GravimeterConfig:
    omega_rad_s: float = 2.0 * pi * 10e3
    gamma_e_rad_s_T: float = 2.0 * pi * 28e9
    mass_kg: float = 1.47e-17
    hbar_J_s: float = 1.054_571_817e-34
    kT_to_T: float = 1e3

    g_range: tuple[float, float] = (9.7806, 9.825)
    infer_mfg_bias: bool = False
    beta_B_range: tuple[float, float] = (-0.10, 0.10)

    infer_phi_off: bool = True
    phi_off_range_rad: tuple[float, float] = (-pi, pi)
    fixed_phi_off_rad: float = 0.0

    T_range_s: tuple[float, float] = (10e-6, 1.2e-3)
    Bp_range_kTm: tuple[float, float] = (0.5, 80.0)

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
        return 2.0 * pi / self.omega_rad_s


class GravityStatelessPhysicalModel(StatelessPhysicalModel):
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
        if cfg.infer_phi_off:
            params.append(Parameter(bounds=cfg.phi_off_range_rad, name="phi_off"))
        super().__init__(batchsize=batchsize, controls=controls, params=params, outcomes_size=1, prec=cfg.prec)

    def _split_parameters(self, parameters: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        g = parameters[:, :, 0]
        idx = 1
        if self.cfg.infer_mfg_bias:
            beta_B = parameters[:, :, idx]
            idx += 1
        else:
            beta_B = tf.zeros_like(g)
        if self.cfg.infer_phi_off:
            phi_off = wrap_to_pi_tf(parameters[:, :, idx])
        else:
            phi_off = tf.cast(self.cfg.fixed_phi_off_rad, g.dtype) * tf.ones_like(g)
        return g, beta_B, phi_off

    def y0_m(self, dtype) -> Tensor:
        cfg = self.cfg
        return tf.cast(np.sqrt(cfg.hbar_J_s / (2.0 * cfg.mass_kg * cfg.omega_rad_s)), dtype)

    def eta(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, Bp_kTm.dtype)
        return tf.cast(cfg.gamma_e_rad_s_T, Bp_kTm.dtype) * Bp_T_per_m * self.y0_m(Bp_kTm.dtype) / tf.cast(cfg.omega_rad_s, Bp_kTm.dtype)

    def k_g(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        Bp_T_per_m = Bp_kTm * tf.cast(cfg.kT_to_T, T_s.dtype)
        w = tf.cast(cfg.omega_rad_s, T_s.dtype)
        ge = tf.cast(cfg.gamma_e_rad_s_T, T_s.dtype)
        return (2.0 * ge / w) * Bp_T_per_m * tf.square(T_s) + (8.0 * tf.cast(pi, T_s.dtype) * ge / (w ** 3)) * Bp_T_per_m

    def min_gain(self, dtype) -> Tensor:
        return self.k_g(tf.cast(self.cfg.T_range_s[0], dtype), tf.cast(self.cfg.Bp_range_kTm[0], dtype))

    def max_gain(self, dtype) -> Tensor:
        return self.k_g(tf.cast(self.cfg.T_range_s[1], dtype), tf.cast(self.cfg.Bp_range_kTm[1], dtype))

    def cycle_time_s(self, T_s: Tensor) -> Tensor:
        cfg = self.cfg
        return tf.cast(cfg.dead_time_s + 3.5 * cfg.tau_s, T_s.dtype) + 2.0 * T_s

    def total_resource_cost_s(self, T_s: Tensor, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        extra = tf.zeros_like(Bp_kTm)
        if cfg.mfg_resource_cost_s_at_ref > 0.0:
            bref = tf.cast(cfg.mfg_resource_ref_kTm, Bp_kTm.dtype)
            c_ref = tf.cast(cfg.mfg_resource_cost_s_at_ref, Bp_kTm.dtype)
            extra = c_ref * tf.square(Bp_kTm / tf.maximum(bref, tf.cast(1e-18, Bp_kTm.dtype)))
        return self.cycle_time_s(T_s) + extra

    def trap_visibility_avg_small_noise(self, Bp_kTm: Tensor) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        correction = tf.cast(1944.0 * (pi ** 4) * (cfg.sigma_omega_rel ** 4), eta.dtype) * tf.square(eta)
        return tf.clip_by_value(1.0 - correction, 0.0, 1.0)

    def trap_visibility_exact_from_delta_omega(self, Bp_kTm: Tensor, delta_omega_rad_s: Tensor) -> Tensor:
        cfg = self.cfg
        eta = self.eta(Bp_kTm)
        tau = tf.cast(cfg.tau_s, eta.dtype)
        x = tf.cast(cfg.omega_rad_s, eta.dtype) * (-tau * delta_omega_rad_s / tf.cast(cfg.omega_rad_s, eta.dtype))
        amp = 16.0 * eta * tf.cos(x / 4.0) * tf.square(tf.sin(3.0 * x / 4.0))
        return tf.exp(-0.5 * tf.square(amp))

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
        vis = self.trap_visibility_exact_from_delta_omega(Bp_kTm[..., None], delta_omega)
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
            raise ValueError(f"Unknown trap_visibility_mode={cfg.trap_visibility_mode!r}")
        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))
        return tf.clip_by_value(vis, 0.0, 1.0)

    def sample_true_visibility_factor(self, T_s: Tensor, Bp_kTm: Tensor, rangen: tf.random.Generator) -> Tensor:
        cfg = self.cfg
        if cfg.trap_visibility_mode == "none":
            vis = tf.ones_like(T_s)
        elif cfg.trap_visibility_mode == "small_noise_avg":
            vis = self.trap_visibility_avg_small_noise(Bp_kTm)
        elif cfg.trap_visibility_mode == "exact_single_delta":
            delta_omega = rangen.normal(tf.shape(T_s), dtype=T_s.dtype) * tf.cast(cfg.sigma_omega_rel * cfg.omega_rad_s, T_s.dtype)
            vis = self.trap_visibility_exact_from_delta_omega(Bp_kTm, delta_omega)
        else:
            raise ValueError(f"Unknown trap_visibility_mode={cfg.trap_visibility_mode!r}")
        if cfg.T2_spin_s is not None and cfg.T2_spin_s > 0.0:
            vis = vis * tf.exp(-self.cycle_time_s(T_s) / tf.cast(cfg.T2_spin_s, vis.dtype))
        return tf.clip_by_value(vis, 0.0, 1.0)

    def mfg_quadrature(self, dtype) -> Tuple[Tensor, Tensor]:
        bound = float(self.cfg.mfg_rel_noise_bound)
        n = int(max(1, self.cfg.mfg_noise_quad_points))
        if bound <= 0.0 or n <= 1:
            return tf.constant([0.0], dtype=dtype), tf.constant([1.0], dtype=dtype)
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        nodes = tf.constant(bound * x_np, dtype=dtype)
        weights = tf.constant(0.5 * w_np, dtype=dtype)
        return nodes, weights

    def _sample_mfg_rel_noise(self, shape, rangen: tf.random.Generator, dtype) -> Tensor:
        bound = float(self.cfg.mfg_rel_noise_bound)
        if bound <= 0.0:
            return tf.zeros(shape, dtype=dtype)
        return rangen.uniform(shape=shape, minval=tf.cast(-bound, dtype), maxval=tf.cast(+bound, dtype), dtype=dtype)

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

    def likelihood_given_global_params(self, outcomes: Tensor, controls: Tensor, g: Tensor, beta_B: Tensor, phi_off: Tensor) -> Tensor:
        T_s = controls[..., 0]
        Bp_commanded_kTm = controls[..., 1]
        mw_phase = controls[..., 2]
        Bp_base_kTm = Bp_commanded_kTm * self._model_bias_factor(beta_B)
        phi_total = wrap_to_pi_tf(phi_off + mw_phase)
        eps_nodes, eps_weights = self.mfg_quadrature(dtype=T_s.dtype)
        if int(eps_nodes.shape[0]) == 1:
            Bp_kTm = Bp_base_kTm
            vis = self.known_visibility_factor(T_s, Bp_kTm)
            theta = self.k_g(T_s, Bp_kTm) * g + phi_total
            p_plus = safe_clip_prob(0.5 * (1.0 + vis * tf.cos(theta)))
        else:
            Bp_kTm = Bp_base_kTm[..., None] * (1.0 + eps_nodes)
            T_e = T_s[..., None]
            phi_e = phi_total[..., None]
            g_e = g[..., None]
            vis = self.known_visibility_factor(T_e, Bp_kTm)
            theta = self.k_g(T_e, Bp_kTm) * g_e + phi_e
            p_plus_nodes = safe_clip_prob(0.5 * (1.0 + vis * tf.cos(theta)))
            p_plus = safe_clip_prob(tf.reduce_sum(p_plus_nodes * eps_weights, axis=-1))
        y = outcomes[..., 0]
        prob = tf.where(y > 0.5, p_plus, 1.0 - p_plus)
        return safe_clip_prob(prob)

    def model(self, outcomes: Tensor, controls: Tensor, parameters: Tensor, meas_step: Tensor, num_systems: int = 1) -> Tensor:
        del meas_step, num_systems
        g, beta_B, phi_off = self._split_parameters(parameters)
        return self.likelihood_given_global_params(outcomes, controls, g, beta_B, phi_off)

    def perform_measurement(self, controls: Tensor, parameters: Tensor, meas_step: Tensor, rangen: tf.random.Generator) -> Tuple[Tensor, Tensor]:
        del meas_step
        T_s = controls[:, 0, 0]
        Bp_commanded_kTm = controls[:, 0, 1]
        mw_phase = controls[:, 0, 2]
        g, beta_B, phi_off = self._split_parameters(parameters)
        Bp_base_kTm = Bp_commanded_kTm * self._measurement_bias_factor(beta_B[:, 0])
        eps = self._sample_mfg_rel_noise(tf.shape(Bp_base_kTm), rangen, Bp_base_kTm.dtype)
        Bp_kTm = Bp_base_kTm * (1.0 + eps)
        vis_true = self.sample_true_visibility_factor(T_s, Bp_kTm, rangen)
        theta = self.k_g(T_s, Bp_kTm) * g[:, 0] + wrap_to_pi_tf(phi_off[:, 0] + mw_phase)
        p_plus = safe_clip_prob(0.5 * (1.0 + vis_true * tf.cos(theta)))
        if self.cfg.readout_flip_prob > 0.0:
            flip = tf.cast(self.cfg.readout_flip_prob, p_plus.dtype)
            p_plus = safe_clip_prob((1.0 - flip) * p_plus + flip * (1.0 - p_plus))
        u = rangen.uniform(shape=tf.shape(p_plus), minval=0.0, maxval=1.0, dtype=p_plus.dtype)
        y = tf.cast(u < p_plus, p_plus.dtype)
        outcomes = tf.expand_dims(tf.expand_dims(y, axis=1), axis=2)
        log_prob = tf.expand_dims(tf.math.log(tf.where(y > 0.5, p_plus, 1.0 - p_plus)), axis=1)
        return outcomes, log_prob

    def count_resources(self, resources: Tensor, outcomes: Tensor, controls: Tensor, true_values: Tensor, meas_step: Tensor) -> Tensor:
        del outcomes, true_values, meas_step
        T_s = controls[:, 0]
        Bp_kTm = controls[:, 1]
        step_cost = self.total_resource_cost_s(T_s, Bp_kTm)
        return resources + tf.expand_dims(step_cost, axis=1)
