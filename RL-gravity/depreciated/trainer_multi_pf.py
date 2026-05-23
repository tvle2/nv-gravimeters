# trainer_multi_pf.py

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tqdm.auto import trange

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TF_DETERMINISTIC_OPS", None)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


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
        raise ImportError(f"Cannot load spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
except Exception:
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay


from gravimeter_model_complete import GravimeterConfig
from gravimeter_multi_pf import (
    MultiPFBankConfig,
    build_gravity_multi_pf_simulation,
)


# ---------------------------------------------------------------------------
# Top-level mode flags
# ---------------------------------------------------------------------------

RUN_PROFILE: str = "diag"
# RUN_MODE: "all" | "train-only" | "eval-only" | "evidence-diagnostic"
RUN_MODE: str = "train-only"
NOISE_MODE: str = "none"


# ---------------------------------------------------------------------------
# RunProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunProfile:
    name: str
    out_dir: str
    batchsize: int
    iterations: int
    interval_save: int
    max_steps: int
    max_resources: float
    initial_lr: float
    grad_clip_norm: float
    seed: int
    gradient_accumulation: int

    n_per_mode: int
    k_max: int

    top_k_modes: int
    prec: str

    cumulative_loss: bool
    baseline: bool
    loss_logl_outcomes: bool
    stop_gradient_input: bool
    stop_gradient_pf: bool
    log_loss: bool

    eval_iters: int

DIAG_LONG_PROFILE = RunProfile(
    name="diag_long",
    out_dir="runs/gravity_multi_pf_diag_long",
    batchsize=16,
    iterations=1,
    interval_save=1,
    max_steps=32,
    max_resources=100e-3,
    initial_lr=2e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=1,

    n_per_mode=32,
    k_max=64,
    top_k_modes=8,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=False,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    log_loss=True,

    eval_iters=16,
)


DIAG_PROFILE = RunProfile(
    name="diag",
    out_dir="runs/gravity_multi_pf_diag",
    batchsize=16,
    iterations=30,
    interval_save=10,
    max_steps=16,
    max_resources=50e-3,
    initial_lr=2e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=1,

    n_per_mode=32,
    k_max=64,
    top_k_modes=8,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=False,  
    stop_gradient_input=True,
    stop_gradient_pf=True,     
    log_loss=True,

    eval_iters=16,
)
DIAG_BPTT_PROFILE = RunProfile(
    name="diag_bptt",
    out_dir="runs/gravity_multi_pf_diag_bptt",
    batchsize=8,
    iterations=30,
    interval_save=10,
    max_steps=8,
    max_resources=20e-3,
    initial_lr=1e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=1,

    n_per_mode=32,
    k_max=16,
    top_k_modes=4,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=False,
    stop_gradient_input=True,
    stop_gradient_pf=False,
    log_loss=True,

    eval_iters=8,
)
SMOKE_PROFILE = RunProfile(
    name="smoke",
    out_dir="runs/gravity_multi_pf_smoke",
    batchsize=8,
    iterations=80,
    interval_save=20,
    max_steps=8,
    max_resources=10.0,
    initial_lr=5e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=1,
    n_per_mode=32,
    k_max=16,
    top_k_modes=4,
    prec="float64",
    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    log_loss=True,
    eval_iters=16,
)

PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_multi_pf_pilot_v4",
    batchsize=32,
    iterations=300,
    interval_save=50,
    max_steps=32,
    max_resources=50.0,
    initial_lr=5e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=2,
    n_per_mode=64,
    k_max=128,
    top_k_modes=8,
    prec="float64",
    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=False,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    log_loss=True,
    eval_iters=64,
)

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravity_multi_pf_full",
    batchsize=64,
    iterations=20000,
    interval_save=200,
    max_steps=64,
    max_resources=100.0,
    initial_lr=5e-3,
    grad_clip_norm=100.0,
    seed=123,
    gradient_accumulation=2,
    n_per_mode=64,
    k_max=128,
    top_k_modes=8,
    prec="float64",
    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    log_loss=True,
    eval_iters=256,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, records: List[dict]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def set_global_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_profile() -> RunProfile:
    key = RUN_PROFILE.strip().lower()
    if key == "smoke":
        return SMOKE_PROFILE
    if key == "pilot":
        return PILOT_PROFILE
    if key == "diag":
        return DIAG_PROFILE
    if key == "diag_long":
        return DIAG_LONG_PROFILE
    if key == "diag_bptt":
        return DIAG_BPTT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(
        f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Choose 'smoke', 'pilot', 'diag', 'diag_bptt', or 'full'."
    )


def make_gravimeter_cfg(profile: RunProfile) -> GravimeterConfig:
    common = dict(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=1.47e-17,
        hbar_J_s=1.054_571_817e-34,
        kT_to_T=1e3,
        g_range=(9.7806, 9.825),
        infer_mfg_bias=False,
        beta_B_range=(-0.10, 0.10),

        infer_phi_off=False,
        fixed_phi_off_rad=0.0,

        T_range_s=(10e-6, 5e-4),
        Bp_range_kTm=(0.5, 25.0),
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec=profile.prec,
    )
    mode = NOISE_MODE.strip().lower()
    if mode == "none":
        # Even in 'none' mode we keep a finite (conservative) spin coherence so
        # that visibility falls off when cycle_time grows.  This stops the
        # controller from picking T at the upper bound unconditionally.
        return GravimeterConfig(
            **common,
            T2_spin_s=1.0e-3,         # 1 ms; pre-DD, room-temp realistic
            mfg_rel_noise_bound=0.0,
            mfg_noise_quad_points=1,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.0,
            trap_visibility_mode="none",
            trap_noise_quad_points=1,
        )
    if mode == "paper":
        return GravimeterConfig(
            **common,
            T2_spin_s=0.6e-3,         # 0.6 ms, conservative pre-DD
            mfg_rel_noise_bound=0.025,
            mfg_noise_quad_points=9,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.01,
            trap_visibility_mode="exact_single_delta",
            trap_noise_quad_points=9,
        )
    raise ValueError(f"Unknown NOISE_MODE={NOISE_MODE!r}")


def make_bank_cfg(profile: RunProfile) -> MultiPFBankConfig:
    return MultiPFBankConfig(
        n_per_mode=profile.n_per_mode,
        k_max=profile.k_max,
        n_scales=None,
        top_k_modes=profile.top_k_modes,
        resample_threshold=0.5,
        resample_alpha=0.5,
        resample_beta=0.98,
        scibior_trick=False,
        trim=True,
        smoothness_lambda=0.0,
        schedule_floor=0.25,
        mode_penalty_coef=0.0,

        # NEW structural NN state/schedule settings.
        use_full_q_hist=True,
        posterior_gain_width_multiplier=4.0,
        min_gain_fringe_fraction=0.25,
        log_sigma_frac_bounds=(-6.0, 0.0),
    )


def make_sim_pars(profile: RunProfile) -> SimulationParameters:
    return SimulationParameters(
        sim_name=f"gravity_multi_pf_{profile.name}",
        num_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=1.0,
        prec=profile.prec,
        stop_gradient_input=profile.stop_gradient_input,
        loss_logl_outcomes=profile.loss_logl_outcomes,
        loss_logl_controls=False,
        cumulative_loss=profile.cumulative_loss,
        baseline=profile.baseline,
        stop_gradient_pf=profile.stop_gradient_pf,
        log_loss=profile.log_loss,
        permutation_invariant=False,
    )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_checks(cfg: GravimeterConfig, profile: RunProfile) -> None:
    tau = 2.0 * math.pi / cfg.omega_rad_s
    cycle_min = cfg.dead_time_s + 3.5 * tau + 2.0 * cfg.T_range_s[0]
    cycle_max = cfg.dead_time_s + 3.5 * tau + 2.0 * cfg.T_range_s[1]
    max_steps_by_resource = profile.max_resources / cycle_min
    print(f"[sanity] tau = {tau*1e6:.2f} us")
    print(f"[sanity] cycle_min = {cycle_min*1e6:.2f} us")
    print(f"[sanity] cycle_max = {cycle_max*1e6:.2f} us")
    print(f"[sanity] max_steps allowed by max_resources = {max_steps_by_resource:.0f}")
    print(f"[sanity] num_steps = {profile.max_steps}")
    if max_steps_by_resource > 2 * profile.max_steps:
        print("[sanity] WARNING: max_steps is the binding stopping condition, "
              "max_resources is non-binding.  Make sure that's what you want.")
    if cfg.T2_spin_s is not None:
        cycle_at_kmax = cycle_max
        vis = math.exp(-cycle_at_kmax / cfg.T2_spin_s)
        print(f"[sanity] T2_spin = {cfg.T2_spin_s*1e3:.2f} ms")
        print(f"[sanity] Visibility at T=T_max (cycle={cycle_at_kmax*1e6:.0f}us): {vis:.3f}")
        if vis < 0.05:
            print("[sanity] WARNING: visibility at T_max is below 5%; "
                  "controller will likely settle on T well below T_max.")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_loss_history(out_dir: Path, sim_str: str, loss_history: list, interval: int) -> None:
    arr = np.array(loss_history)
    n_blocks = len(arr) // max(interval, 1)
    if n_blocks == 0:
        return
    blocks = arr[: n_blocks * interval].reshape(n_blocks, interval)
    means = blocks.mean(axis=1)
    csv_path = out_dir / f"{sim_str}_history.csv"
    with open(csv_path, "w") as f:
        f.write("Loss\n")
        for v in means:
            f.write(f"{v:.6e}\n")
    print(f"[ok] Saved loss history to {csv_path}")


def _save_run_config(out_dir: Path, *, profile: RunProfile,
                     cfg: GravimeterConfig, bank_cfg: MultiPFBankConfig) -> None:
    config = {
        "run_profile": profile.name,
        "run_mode": RUN_MODE,
        "noise_mode": NOISE_MODE,
        "profile": {k: getattr(profile, k) for k in profile.__dataclass_fields__},
        "gravimeter_cfg": {
            "g_range": list(cfg.g_range),
            "T_range_s": list(cfg.T_range_s),
            "Bp_range_kTm": list(cfg.Bp_range_kTm),
            "mfg_rel_noise_bound": cfg.mfg_rel_noise_bound,
            "sigma_omega_rel": cfg.sigma_omega_rel,
            "trap_visibility_mode": cfg.trap_visibility_mode,
            "T2_spin_s": cfg.T2_spin_s,
            "readout_flip_prob": cfg.readout_flip_prob,
            "prec": cfg.prec,
            "infer_mfg_bias": cfg.infer_mfg_bias,
        },
        "bank_cfg": {
            "n_per_mode": bank_cfg.n_per_mode,
            "k_max": bank_cfg.k_max,
            "top_k_modes": bank_cfg.top_k_modes,
            "scibior_trick": bank_cfg.scibior_trick,
            "schedule_floor": bank_cfg.schedule_floor,
            "mode_penalty_coef": bank_cfg.mode_penalty_coef,
            "smoothness_lambda": bank_cfg.smoothness_lambda,

            "use_full_q_hist": bank_cfg.use_full_q_hist,
            "posterior_gain_width_multiplier": bank_cfg.posterior_gain_width_multiplier,
            "min_gain_fringe_fraction": bank_cfg.min_gain_fringe_fraction,
            "log_sigma_frac_bounds": list(bank_cfg.log_sigma_frac_bounds),
        },
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def simulation_str(profile: RunProfile) -> str:
    return f"gravity_multi_pf_{profile.name}"


def run_training(profile: RunProfile) -> None:
    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run profile : {profile.name}")
    print(f"Noise mode  : {NOISE_MODE}")
    print(f"Run mode    : {RUN_MODE}")
    print(f"Output dir  : {out_dir.resolve()}")
    print(f"Precision   : {profile.prec}")
    print(f"Batch size  : {profile.batchsize}")
    print(f"K modes     : {profile.k_max}")
    print(f"N per mode  : {profile.n_per_mode}")
    print(f"Episode len : {profile.max_steps}")
    print(f"Iterations  : {profile.iterations}")
    print(f"stop_gradient_pf : {profile.stop_gradient_pf}")
    print(f"grad_clip_norm   : {profile.grad_clip_norm}")
    print("=" * 72)

    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)
    _sanity_checks(cfg, profile)

    rangen = tf.random.Generator.from_seed(profile.seed)
    sim, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg, bank_cfg=bank_cfg, simpars=simpars, rangen=rangen,
    )
    variables = controller.trainable_variables
    print(f"[info] Controller params: "
          f"{sum(int(np.prod(v.shape)) for v in variables):,}")
    print(f"[info] Coarsest k_ref = {float(sim._k_ref_coarsest):.3e}, "
          f"k_g_max = {sim._k_g_max:.3e}")

    lr_schedule = InverseSqrtDecay(
        initial_learning_rate=profile.initial_lr,
        prec=profile.prec,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    sim_str = simulation_str(profile)
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_debug_jsonl = out_dir / "train_debug.jsonl"
    rollout_debug_jsonl = out_dir / "rollout_debug.jsonl"
    for p in (train_debug_jsonl, rollout_debug_jsonl):
        if p.exists():
            p.unlink()

    def single_step_eager(debug: bool = False):
        """One gradient-accumulation step.

        The EMA baseline is NOT used.  The execute() method uses
        Belliardo's within-step mean_b(L_t) as baseline, which is
        the correct REINFORCE baseline for this problem (Belliardo Eq 93).
        An external EMA baseline was producing near-zero advantages
        (because EMA tracks the loss closely) and amplified instability
        when combined with log_loss normalization.
        """
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        debug_records: List[dict] = []

        for acc_i in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                if debug and acc_i == profile.gradient_accumulation - 1:
                    loss_diff, loss, dbg = sim.execute(
                        rangen, debug=True, debug_max_examples=3,
                    )
                    debug_records.extend(dbg)
                else:
                    loss_diff, loss = sim.execute(rangen)
            grads = tape.gradient(loss_diff, variables)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, variables)]
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                     for g in grads]
            acc_loss += float(loss.numpy())
            acc_grads = [a + g for a, g in zip(acc_grads, grads)]

        acc_grads = [g / float(profile.gradient_accumulation) for g in acc_grads]
        raw_norm = float(tf.linalg.global_norm(acc_grads).numpy())
        # clipped_grads, clipped_norm_t = tf.clip_by_global_norm(
        #     acc_grads, profile.grad_clip_norm,
        # )
        # clipped_norm = float(clipped_norm_t.numpy())
        clipped_grads, preclip_norm_t = tf.clip_by_global_norm(
            acc_grads, profile.grad_clip_norm,
        )
        clipped_norm = float(tf.linalg.global_norm(clipped_grads).numpy())
        optimizer.apply_gradients(zip(clipped_grads, variables))

        avg_loss = acc_loss / profile.gradient_accumulation
        return (avg_loss, raw_norm, clipped_norm, debug_records)

    loss_history: List[float] = []
    best_loss = float("inf")
    best_ckpt_idx = 1

    pbar = trange(profile.iterations, desc="Training", unit="step")
    for j in pbar:
        debug_now = ((j + 1) % profile.interval_save == 0)
        step_loss, raw_norm, clipped_norm, dbg_records = single_step_eager(debug=debug_now)
        loss_history.append(step_loss)
        recent10 = float(np.mean(loss_history[max(0, j - 9): j + 1]))

        try:
            lr_val = float(lr_schedule(optimizer.iterations).numpy())
        except Exception:
            lr_val = float(optimizer.learning_rate.numpy())
        pbar.set_postfix(
            loss=f"{step_loss:+.4f}",
            avg10=f"{recent10:+.4f}",
            best=f"{best_loss:+.4f}",
            lr=f"{lr_val:.1e}",
            gn=f"{raw_norm:.1e}",
        )

        train_record = {
            "iter": int(j + 1),
            "loss": float(step_loss),
            "avg10": float(recent10),
            "best_window": float(best_loss) if np.isfinite(best_loss) else None,
            "raw_grad_norm": float(raw_norm),
            "clipped_grad_norm": float(clipped_norm),
            "lr": float(lr_val),
            "K": int(bank.K),
            "max_q": float(tf.reduce_mean(tf.reduce_max(bank.mode_weights, axis=1)).numpy()),
        }
        _append_jsonl(train_debug_jsonl, [train_record])
        if dbg_records:
            for rec in dbg_records:
                rec["train_iter"] = int(j + 1)
            _append_jsonl(rollout_debug_jsonl, dbg_records)

        if (j + 1) % profile.interval_save == 0:
            ckpt_idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / f"{ckpt_idx}.weights.h5"
            controller.save_weights(str(ckpt_path))
            window_loss = float(np.mean(loss_history[max(0, j + 1 - profile.interval_save): j + 1]))
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = ckpt_idx

    pbar.close()
    print("[training] Done.")

    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(f"[ok] Loaded best checkpoint (idx={best_ckpt_idx}, "
              f"window_loss={best_loss:+.4f})")

    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass

    weights_path = out_dir / f"gravity_multi_pf_{profile.name}_final.weights.h5"
    controller.save_weights(str(weights_path))
    print(f"[ok] Saved final weights to {weights_path}")

    _save_loss_history(out_dir, sim_str, loss_history, profile.interval_save)
    _save_run_config(out_dir, profile=profile, cfg=cfg, bank_cfg=bank_cfg)
    print(f"[done] All outputs saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(profile: RunProfile, weights_path: Optional[Path] = None) -> None:
    out_dir = Path(profile.out_dir)
    if weights_path is None:
        weights_path = out_dir / f"gravity_multi_pf_{profile.name}_final.weights.h5"
    if not weights_path.exists():
        print(f"[warn] Weights not found at {weights_path}. Skipping evaluation.")
        return

    print(f"\n[eval] Loading weights from {weights_path}")
    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)

    sim, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg, bank_cfg=bank_cfg, simpars=simpars, rangen=rangen,
    )
    controller.load_weights(str(weights_path))

    all_loss = []
    all_mse = []
    all_rmse = []
    all_g_true = []
    all_g_hat = []
    all_max_q = []

    pbar = trange(profile.eval_iters, desc="Evaluating", unit="ep")
    for _ in pbar:
        # deploy=True runs an episode and returns the true_values.  After
        # the call, bank state is the final-step state of that episode, so
        # bank.map_mode_estimate() gives the MAP-mode estimator we report.
        deploy_payload = sim.execute(rangen, deploy=True)
        true_values = deploy_payload[0]
        g_true = true_values[:, 0, 0]
        # g_hat, _ = bank.map_mode_estimate()
        g_hat, _ = bank.marginal_mean_and_var()

        # Also compute the per-step log-Holevo loss at the end-of-episode
        # bank state, as a coarse training-curve sanity check.
        # loss_val = float(
        #     tf.reduce_mean(sim._per_step_loss()).numpy()
        # )
        loss_val = float(
            tf.reduce_mean(sim._per_step_loss(true_values=true_values)).numpy()
        )
        all_loss.append(loss_val)

        sq = tf.square(g_hat - g_true)
        mse_b = float(tf.reduce_mean(sq).numpy())
        all_mse.append(mse_b)
        all_rmse.append(float(np.sqrt(mse_b)))
        all_g_true.append(g_true.numpy())
        all_g_hat.append(g_hat.numpy())
        all_max_q.append(float(tf.reduce_mean(tf.reduce_max(bank.mode_weights, axis=1)).numpy()))
        pbar.set_postfix(
            loss=f"{all_loss[-1]:+.4f}",
            rmse=f"{all_rmse[-1]:.2e}",
            maxq=f"{all_max_q[-1]:.3f}",
            mean_loss=f"{np.mean(all_loss):+.4f}",
        )
    pbar.close()

    all_loss = np.array(all_loss)
    all_mse = np.array(all_mse)
    all_rmse = np.array(all_rmse)
    all_max_q = np.array(all_max_q)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)

    rmse_global = float(np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat) ** 2)))

    print(f"\n[eval] Results over {profile.eval_iters} episodes:")
    print(f"  Mean posterior-risk loss: {np.mean(all_loss):+.4f}")
    print(f"  Mean MSE (marginal):     {np.mean(all_mse):.2e}")
    print(f"  Mean RMSE (marginal):    {np.mean(all_rmse):.2e} m/s^2")
    print(f"  Global RMSE (marginal):  {rmse_global:.2e} m/s^2")
    print(f"  Mean max_q:              {np.mean(all_max_q):.3f}")

    np.savez(
        str(out_dir / f"eval_{profile.name}_extended.npz"),
        loss=all_loss, mse=all_mse, rmse=all_rmse, max_q=all_max_q,
        g_true=all_g_true_flat, g_hat=all_g_hat_flat,
    )
    print(f"[ok] Saved eval to {out_dir / f'eval_{profile.name}_extended.npz'}")


def run_mode_evidence_diagnostic(profile: RunProfile) -> None:
    """One-step diagnostic: check whether Z_k varies across gravity modes."""

    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)

    rangen = tf.random.Generator.from_seed(profile.seed + 777)

    sim, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    bank.reset(rangen)

    prec = profile.prec
    B = profile.batchsize

    meas_step = tf.zeros((B, 1), dtype="int32")
    used_resources = tf.zeros((B, 1), dtype=prec)

    input_strategy = sim.generate_input(
        bank.weights_list[0],
        bank.particles_list[0],
        tf.cast(meas_step, prec),
        used_resources,
        rangen,
    )

    controls = controller(tf.stop_gradient(input_strategy))

    # Force outcome y=1 for all batch rows.
    outcomes = tf.ones((B, sim.phys_model.outcomes_size), dtype=prec)

    Z = bank.diagnostic_mode_evidence(
        outcomes=outcomes,
        controls=controls,
        meas_step=meas_step,
    )

    Z_mean = tf.reduce_mean(Z, axis=1)
    Z_std = tf.math.reduce_std(Z, axis=1)
    Z_cv = Z_std / tf.maximum(Z_mean, tf.cast(1e-30, prec))

    print("\n[mode-evidence diagnostic]")
    print(f"  mean(Z_mean): {float(tf.reduce_mean(Z_mean).numpy()):.6e}")
    print(f"  mean(Z_std):  {float(tf.reduce_mean(Z_std).numpy()):.6e}")
    print(f"  mean(Z_cv):   {float(tf.reduce_mean(Z_cv).numpy()):.6e}")
    print(f"  min(Z):       {float(tf.reduce_min(Z).numpy()):.6e}")
    print(f"  max(Z):       {float(tf.reduce_max(Z).numpy()):.6e}")
    print(f"  mean max_q before update: "
          f"{float(tf.reduce_mean(tf.reduce_max(bank.mode_weights, axis=1)).numpy()):.6e}")

    # Apply the update once and inspect q movement.
    bank.apply_measurement(
        outcomes=outcomes,
        controls=controls,
        meas_step=meas_step,
        continue_flag=tf.ones((B, 1), dtype="bool"),
        rangen=rangen,
    )

    print(f"  mean max_q after one update:  "
          f"{float(tf.reduce_mean(tf.reduce_max(bank.mode_weights, axis=1)).numpy()):.6e}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_fixed_policy_diagnostic(profile: RunProfile) -> None:
    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)
    rangen = tf.random.Generator.from_seed(profile.seed + 2024)

    sim, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    all_loss = []
    all_rmse = []
    all_max_q = []

    for ep in range(profile.eval_iters):
        payload, dbg = sim.execute(
            rangen,
            deploy=True,
            debug=True,
            debug_max_examples=3,
        )

        true_values = payload[0]
        g_true = true_values[:, 0, 0]
        g_hat, _ = bank.marginal_mean_and_var()

        loss_val = float(tf.reduce_mean(sim._per_step_loss(true_values=true_values)).numpy())
        rmse_val = float(tf.sqrt(tf.reduce_mean(tf.square(g_hat - g_true))).numpy())
        max_q_val = float(tf.reduce_mean(tf.reduce_max(bank.mode_weights, axis=1)).numpy())

        all_loss.append(loss_val)
        all_rmse.append(rmse_val)
        all_max_q.append(max_q_val)

        print(
            f"[fixed ep {ep+1:03d}] "
            f"loss={loss_val:.4e}, rmse={rmse_val:.4e}, max_q={max_q_val:.4f}"
        )

    print("\n[fixed-policy diagnostic summary]")
    print(f"  mean posterior-risk loss: {np.mean(all_loss):.4e}")
    print(f"  mean RMSE:                {np.mean(all_rmse):.4e}")
    print(f"  mean max_q:               {np.mean(all_max_q):.4f}")


def main() -> None:
    profile = get_profile()
    set_global_reproducibility(profile.seed)
    tf.keras.backend.clear_session()

    if RUN_MODE == "evidence-diagnostic":
        run_mode_evidence_diagnostic(profile)
        return

    if RUN_MODE in {"all", "train-only"}:
        run_training(profile)

    if RUN_MODE in {"all", "eval-only"}:
        run_evaluation(profile)
    
    if RUN_MODE == "fixed-diagnostic":
        run_fixed_policy_diagnostic(profile)
        return
    if RUN_MODE not in {"all", "train-only", "eval-only", "evidence-diagnostic"}:
        raise ValueError(
            f"Unknown RUN_MODE={RUN_MODE!r}. "
            "Choose 'all', 'train-only', 'eval-only', or 'evidence-diagnostic'."
        )
    

if __name__ == "__main__":
    main()