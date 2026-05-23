# trainer_multi_pf.py
"""Trainer for the levitated-NV gravimeter Multi-PF agent.

Defaults follow the design decisions locked with the user:
  * K=k_max sub-PFs fixed for the entire episode (no pruning, no splitting).
  * Two-scale log-Holevo loss (coarsest scale + current-step k_g).
  * Log-cumulative training loss (Belliardo 2024 Eq. 109).
  * REINFORCE surrogate, with per-step log_prob when stop_gradient_pf=True
    (the recommended default) or cumulative sum_log_prob when False.
  * Within-mode resampling stays ESS-triggered.  Scibior-Wood is OFF
    because the loss is non-polynomial in the within-mode weights.
  * Default precision: float64.
  * Default optimizer: Adam(InverseSqrtDecay) with global-norm clip 1.0.

Reference
---------
Belliardo et al. (2024). Phys. Rev. A 109, 062609.
"""
from __future__ import annotations

import importlib.util
import json
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

# --- Quiet TF and avoid the SVD-on-1-column op-determinism crash. ---
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

#: Select training profile: "smoke", "pilot", or "full".
RUN_PROFILE: str = "pilot"

#: "all" | "train-only" | "eval-only"
RUN_MODE: str = "all"

#: "none" | "paper"  (paper = realistic sensor noise)
NOISE_MODE: str = "none"


# ---------------------------------------------------------------------------
# RunProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunProfile:
    """Complete specification of a training run."""
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

    # Bank
    n_per_mode: int
    k_max: int

    # Controller / loss
    top_k_modes: int
    prec: str  # "float32" | "float64"

    # Training flags
    cumulative_loss: bool
    baseline: bool
    loss_logl_outcomes: bool
    stop_gradient_input: bool
    stop_gradient_pf: bool

    # Evaluation
    eval_iters: int


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

# Smoke: minimal — for verifying nothing is broken.
SMOKE_PROFILE = RunProfile(
    name="smoke",
    out_dir="runs/gravity_multi_pf_smoke",
    batchsize=4,
    iterations=40,
    interval_save=10,
    max_steps=6,
    max_resources=1.0,
    initial_lr=1e-3,
    grad_clip_norm=1.0,
    seed=42,
    gradient_accumulation=1,

    n_per_mode=32,
    k_max=8,

    top_k_modes=4,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,        # 1-step BPTT — required for stability

    eval_iters=8,
)

PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_multi_pf_pilot_fast",
    batchsize=32,
    iterations=200,
    interval_save=100,
    max_steps=32,
    max_resources=10.0,
    initial_lr=3e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=2,

    n_per_mode=64,
    k_max=128,

    top_k_modes=16,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=64,
)

# Full: production run.
FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravity_multi_pf_full",
    batchsize=64,
    iterations=20000,
    interval_save=200,
    max_steps=64,
    max_resources=20.0,
    initial_lr=1e-3,
    grad_clip_norm=1.0,
    seed=123,
    gradient_accumulation=2,

    n_per_mode=64,
    k_max=128,

    top_k_modes=4,
    prec="float64",

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,

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
    # NOT enable_op_determinism(): TF deterministic SVD breaks on
    # 1-column matrices, which qsensoropt's Liu-West jittering hits
    # whenever d=1 (single-parameter g).


def get_profile() -> RunProfile:
    key = RUN_PROFILE.strip().lower()
    if key == "smoke":
        return SMOKE_PROFILE
    if key == "pilot":
        return PILOT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(
        f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Choose 'smoke', 'pilot', or 'full'."
    )


def make_gravimeter_cfg(profile: RunProfile) -> GravimeterConfig:
    """Build the GravimeterConfig from NOISE_MODE and the profile prec."""
    common = dict(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=1.47e-17,
        hbar_J_s=1.054_571_817e-34,
        kT_to_T=1e3,
        g_range=(9.7806, 9.825),
        infer_mfg_bias=False,
        beta_B_range=(-0.10, 0.10),
        T_range_s=(10e-6, 5e-4),
        Bp_range_kTm=(0.5, 25.0),
        T2_spin_s=None,
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec=profile.prec,
    )
    mode = NOISE_MODE.strip().lower()
    if mode == "none":
        return GravimeterConfig(
            **common,
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
        n_scales=None,                  # unused with the new two-scale loss
        top_k_modes=profile.top_k_modes,
        resample_threshold=0.5,
        resample_alpha=0.5,
        resample_beta=0.98,
        scibior_trick=True,
        trim=True,
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
        log_loss=False,
        permutation_invariant=False,
    )


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
    print("=" * 72)

    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)

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

    # --- EMA baseline for REINFORCE ---
    dtype_t = tf.float32 if profile.prec == "float32" else tf.float64
    ema_baseline = tf.Variable(0.0, dtype=dtype_t, trainable=False, name="ema_baseline")
    ema_initialized = tf.Variable(False, trainable=False, name="ema_initialized")
    ema_decay = tf.constant(0.95, dtype=dtype_t)

    def single_step_eager(debug: bool = False):
        """One gradient-accumulation step.  Returns (avg_loss, raw_grad_norm,
        clipped_grad_norm, debug_records)."""
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        debug_records: List[dict] = []
        baseline_arg = ema_baseline if bool(ema_initialized.numpy()) else None

        for acc_i in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                if debug and acc_i == profile.gradient_accumulation - 1:
                    out = sim.execute(
                        rangen, debug=True, debug_max_examples=3,
                        baseline_value=baseline_arg,
                    )
                    loss_diff, loss, dbg = out
                    debug_records.extend(dbg)
                else:
                    loss_diff, loss = sim.execute(
                        rangen, baseline_value=baseline_arg,
                    )
            grads = tape.gradient(loss_diff, variables)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, variables)]
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                     for g in grads]
            acc_loss += float(loss.numpy())
            acc_grads = [a + g for a, g in zip(acc_grads, grads)]

        acc_grads = [g / float(profile.gradient_accumulation) for g in acc_grads]
        raw_norm = float(tf.linalg.global_norm(acc_grads).numpy())
        clipped_grads, clipped_norm_t = tf.clip_by_global_norm(
            acc_grads, profile.grad_clip_norm,
        )
        clipped_norm = float(clipped_norm_t.numpy())
        optimizer.apply_gradients(zip(clipped_grads, variables))

        avg_loss = acc_loss / profile.gradient_accumulation
        avg_loss_t = tf.cast(avg_loss, dtype_t)
        if not bool(ema_initialized.numpy()):
            ema_baseline.assign(avg_loss_t)
            ema_initialized.assign(True)
        else:
            ema_baseline.assign(
                ema_decay * ema_baseline + (1.0 - ema_decay) * avg_loss_t
            )
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
            "ema_baseline": float(ema_baseline.numpy()),
            "advantage": float(step_loss - ema_baseline.numpy()),
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

    pbar = trange(profile.eval_iters, desc="Evaluating", unit="ep")
    for _ in pbar:
        result = sim.execute(rangen, deploy=True)
        true_values = result[0]
        g_true = true_values[:, 0, 0]

        loss_b = sim._per_step_loss(true_values=true_values, controls=None)
        all_loss.append(float(tf.reduce_mean(loss_b).numpy()))

        # Point estimate from the highest-weight mode.
        g_hat, _ = bank.map_mode_estimate()
        sq = tf.square(g_hat - g_true)
        mse_b = float(tf.reduce_mean(sq).numpy())
        all_mse.append(mse_b)
        all_rmse.append(float(np.sqrt(mse_b)))
        all_g_true.append(g_true.numpy())
        all_g_hat.append(g_hat.numpy())
        pbar.set_postfix(
            loss=f"{all_loss[-1]:+.4f}",
            rmse=f"{all_rmse[-1]:.2e}",
            mean_loss=f"{np.mean(all_loss):+.4f}",
        )
    pbar.close()

    all_loss = np.array(all_loss)
    all_mse = np.array(all_mse)
    all_rmse = np.array(all_rmse)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)

    print(f"\n[eval] Results over {profile.eval_iters} episodes:")
    print(f"  Mean MSE loss (normalized): {np.mean(all_loss):+.4f}")
    print(f"  Mean MSE:             {np.mean(all_mse):.2e}")
    print(f"  Mean RMSE:            {np.mean(all_rmse):.2e} m/s²")
    rmse_global = float(np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat) ** 2)))
    print(f"  Global RMSE:          {rmse_global:.2e} m/s²")

    np.savez(
        str(out_dir / f"eval_{profile.name}_extended.npz"),
        loss=all_loss, mse=all_mse, rmse=all_rmse,
        g_true=all_g_true_flat, g_hat=all_g_hat_flat,
    )
    print(f"[ok] Saved eval to {out_dir / f'eval_{profile.name}_extended.npz'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    profile = get_profile()
    set_global_reproducibility(profile.seed)
    tf.keras.backend.clear_session()

    if RUN_MODE in {"all", "train-only"}:
        run_training(profile)
    if RUN_MODE in {"all", "eval-only"}:
        run_evaluation(profile)
    if RUN_MODE not in {"all", "train-only", "eval-only"}:
        raise ValueError(
            f"Unknown RUN_MODE={RUN_MODE!r}. Choose 'all', 'train-only', or 'eval-only'."
        )


if __name__ == "__main__":
    main()