
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TF_DETERMINISTIC_OPS", None)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tensorflow as tf
from tqdm.auto import trange

import importlib.util
import sys
import types


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = True
    every_train_steps: int = 25
    eval_episode_every: int = 100
    max_batch_examples: int = 3
    jsonl_path: str = "runs/debug/local_holevo_hybrid_debug_rollout.jsonl"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


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
    SimulationParameters = _load_local_qsensoropt_module("simulation_parameters").SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay

from gravimeter_model_complete import GravimeterConfig
from gravimeter_hierarchical_local_holevo import (
    LocalHolevoHierarchicalPFConfig,
    build_local_holevo_hierarchical_simulation,
)

RUN_PROFILE: str = "pilot"
RUN_MODE: str = "all"
NOISE_MODE: str = "none"


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
    seed: int
    gradient_accumulation: int
    n_particles: int
    n_levels: int
    n_disambig_per_level: int
    cumulative_loss: bool
    baseline: bool
    loss_logl_outcomes: bool
    stop_gradient_input: bool
    stop_gradient_pf: bool
    eval_iters: int


SMOKE_PROFILE = RunProfile(
    name="smoke",
    out_dir="runs/gravity_local_holevo_hybrid_smoke",
    batchsize=16,
    iterations=300,
    interval_save=50,
    max_steps=32,
    max_resources=0.04,
    initial_lr=3e-4,
    seed=42,
    gradient_accumulation=1,
    n_particles=512,
    n_levels=4,
    n_disambig_per_level=7,
    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,
    eval_iters=16,
)

PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_local_holevo_hybrid_pilot",
    batchsize=32,
    iterations=2000,
    interval_save=8,
    max_steps=32,
    max_resources=0.08,
    initial_lr=2e-4,
    seed=42,
    gradient_accumulation=1,
    n_particles=1024,
    n_levels=4,
    n_disambig_per_level=7,
    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,
    eval_iters=32,
)

DEBUG = DebugConfig()


def set_global_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def get_profile() -> RunProfile:
    key = RUN_PROFILE.strip().lower()
    if key == "smoke":
        return SMOKE_PROFILE
    if key == "pilot":
        return PILOT_PROFILE
    raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}.")


def make_gravimeter_cfg() -> GravimeterConfig:
    common = dict(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=1.47e-17,
        hbar_J_s=1.054_571_817e-34,
        kT_to_T=1e3,
        g_range=(9.7806, 9.825),
        infer_mfg_bias=False,
        beta_B_range=(-0.10, 0.10),
        infer_phi_off=True,
        phi_off_range_rad=(-np.pi, np.pi),
        fixed_phi_off_rad=0.0,
        T_range_s=(10e-6, 1.2e-3),
        Bp_range_kTm=(0.5, 80.0),
        T2_spin_s=None,
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec="float32",
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
    raise ValueError(f"Unknown NOISE_MODE={NOISE_MODE!r}.")


def make_bank_cfg(profile: RunProfile) -> LocalHolevoHierarchicalPFConfig:
    return LocalHolevoHierarchicalPFConfig(
        n_particles=profile.n_particles,
        n_levels=profile.n_levels,
        n_disambig_per_level=profile.n_disambig_per_level,
        prec="float32",
        resample_threshold=0.5,
        resample_alpha=0.5,
        resample_beta=0.98,
        scibior_trick=True,
        trim=True,
        hierarchy_bce_weight=1.0,
        hierarchy_local_mse_weight=0.05,
        local_holevo_weight=0.25,
        gain_penalty_weight=0.10,
        local_holevo_clip=100.0,
        gain_ratio_limit=1.5,
        bp_dis_max_ratio=4.0,
        phase_residual_max_rad=0.35 * np.pi,
    )


def make_sim_pars(profile: RunProfile, cfg: GravimeterConfig) -> SimulationParameters:
    return SimulationParameters(
        sim_name=f"gravity_local_holevo_hybrid_{profile.name}",
        num_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=1.0,
        prec=cfg.prec,
        stop_gradient_input=profile.stop_gradient_input,
        loss_logl_outcomes=profile.loss_logl_outcomes,
        loss_logl_controls=False,
        cumulative_loss=profile.cumulative_loss,
        baseline=profile.baseline,
        stop_gradient_pf=profile.stop_gradient_pf,
        log_loss=False,
        permutation_invariant=False,
    )


def _save_loss_history(out_dir: Path, sim_str: str, loss_history: list[float], interval_save: int) -> None:
    import pandas as pd
    arr = np.array(loss_history)
    num_blocks = len(arr) // interval_save
    if num_blocks == 0:
        return
    arr_trimmed = arr[: num_blocks * interval_save]
    mean_loss = arr_trimmed.reshape(num_blocks, interval_save).mean(axis=1)
    pd.DataFrame({"Loss": mean_loss}).to_csv(
        str(out_dir / f"{sim_str}_history.csv"),
        index=False,
        float_format="%.4e",
    )


def _save_run_config(out_dir: Path, *, profile: RunProfile, cfg: GravimeterConfig, bank_cfg: LocalHolevoHierarchicalPFConfig) -> None:
    config = {
        "run_profile": profile.name,
        "run_mode": RUN_MODE,
        "noise_mode": NOISE_MODE,
        "profile": profile.__dict__,
        "gravimeter_cfg": {
            "g_range": list(cfg.g_range),
            "T_range_s": list(cfg.T_range_s),
            "Bp_range_kTm": list(cfg.Bp_range_kTm),
            "infer_phi_off": cfg.infer_phi_off,
            "phi_off_range_rad": list(cfg.phi_off_range_rad),
            "prec": cfg.prec,
        },
        "bank_cfg": bank_cfg.__dict__,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


def run_training(profile: RunProfile) -> None:
    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run profile  : {profile.name}")
    print(f"Noise mode   : {NOISE_MODE}")
    print(f"Run mode     : {RUN_MODE}")
    print(f"Output dir   : {out_dir.resolve()}")
    print("=" * 72)

    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)
    rangen = tf.random.Generator.from_seed(profile.seed)

    simulation, bank, controller = build_local_holevo_hierarchical_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    variables = controller.trainable_variables
    print(f"[info] Controller parameters: {sum(int(np.prod(v.shape)) for v in variables):,}")
    print(
        f"[info] Bank: N_particles={bank_cfg.n_particles}, "
        f"N_levels={bank_cfg.n_levels}, N_disambig/level={bank_cfg.n_disambig_per_level}"
    )
    print(
        f"[info] Loss weights: BCE={bank_cfg.hierarchy_bce_weight}, "
        f"LocalHolevo={bank_cfg.local_holevo_weight}, MixMSE={bank_cfg.hierarchy_local_mse_weight}, "
        f"GainPen={bank_cfg.gain_penalty_weight}"
    )

    lr_schedule = InverseSqrtDecay(initial_learning_rate=profile.initial_lr, prec=cfg.prec)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    sim_str = str(simulation)
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    debug_path = Path(DEBUG.jsonl_path)
    if DEBUG.enabled and debug_path.exists():
        debug_path.unlink()

    loss_history: list[float] = []
    best_loss = float("inf")
    best_ckpt_idx = 1

    def compute_grad_norm(grads) -> float:
        sq = 0.0
        for g in grads:
            if g is not None:
                sq += float(tf.reduce_sum(tf.square(g)).numpy())
        return sq ** 0.5

    def single_step_eager() -> tuple[float, float]:
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        for _ in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                loss_diff, loss = simulation.execute(rangen, deploy=False, debug=False)
            grads = tape.gradient(loss_diff, variables)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, variables)]
            acc_loss += float(loss.numpy())
            acc_grads = [ag + g for ag, g in zip(acc_grads, grads)]

        acc_grads = [g / profile.gradient_accumulation for g in acc_grads]
        finite_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in acc_grads]
        clipped_grads, _ = tf.clip_by_global_norm(finite_grads, 5.0)
        grad_norm = compute_grad_norm(clipped_grads)
        optimizer.apply_gradients(zip(clipped_grads, variables))
        return acc_loss / profile.gradient_accumulation, grad_norm

    pbar = trange(profile.iterations, desc="Training", unit="step")
    for j in pbar:
        step_loss, grad_norm = single_step_eager()
        loss_history.append(step_loss)
        recent = float(np.mean(loss_history[max(0, j - 9): j + 1]))

        try:
            lr_val = float(lr_schedule(optimizer.iterations).numpy())
        except Exception:
            lr_val = float(optimizer.learning_rate.numpy())

        pbar.set_postfix(
            loss=f"{step_loss:.4f}",
            avg10=f"{recent:.4f}",
            best=f"{best_loss:.4f}",
            grad=f"{grad_norm:.2e}",
            lr=f"{lr_val:.1e}",
            lvl=f"{simulation.bank.current_level:.2f}",
            q0=f"{simulation.bank.q0_mean:.3f}",
        )

        if DEBUG.enabled and ((j + 1) % DEBUG.every_train_steps == 0):
            print(
                "[debug/train] "
                f"iter={j+1} loss={step_loss:.4e} avg10={recent:.4e} "
                f"grad_norm={grad_norm:.4e} mean_level={simulation.bank.current_level:.2f} "
                f"mean_q0={simulation.bank.q0_mean:.3f} "
                f"mean_width={simulation.bank.interval_width:.4e} "
                f"mean_target_kg={simulation.bank.target_k_g:.4e} lr={lr_val:.2e}"
            )

        if DEBUG.enabled and ((j + 1) % DEBUG.eval_episode_every == 0):
            deploy_result = simulation.execute(rangen, deploy=True, debug=True)
            debug_records = deploy_result[-1]
            for rec in debug_records:
                rec["train_iter"] = j + 1
                append_jsonl(debug_path, rec)

            if debug_records:
                sample = debug_records[0]
                print(
                    "[debug/rollout] "
                    f"iter={j+1} level={sample.get('level', 'NA')} "
                    f"step_in_level={sample.get('step_in_level', 'NA')} "
                    f"refining={sample.get('refining', 'NA')} "
                    f"q0={sample.get('q0', float('nan')):.3f} "
                    f"branch_correct={sample.get('branch_correct', 'NA')} "
                    f"target_kg={sample.get('target_kg', float('nan')):.3e} "
                    f"actual_kg={sample.get('actual_kg', float('nan')):.3e} "
                    f"kg_ratio={sample.get('kg_ratio', float('nan')):.3f} "
                    f"local_vh={sample.get('local_holevo_current', float('nan')):.3e} "
                    f"loss_total={sample.get('loss_total', float('nan')):.3e} "
                    f"loss_bce={sample.get('loss_bce', float('nan')):.3e} "
                    f"loss_h={sample.get('loss_local_holevo', float('nan')):.3e} "
                    f"loss_mix={sample.get('loss_mix_mse', float('nan')):.3e} "
                    f"loss_gain={sample.get('loss_gain_penalty', float('nan')):.3e}"
                )

        if (j + 1) % profile.interval_save == 0:
            ckpt_idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / str(ckpt_idx)
            controller.save_weights(str(ckpt_path) + ".weights.h5")
            window_start = max(0, j + 1 - profile.interval_save)
            window_loss = float(np.mean(loss_history[window_start : j + 1]))
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = ckpt_idx

    pbar.close()
    print("[training] Done.")

    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(f"[ok] Loaded best checkpoint (idx={best_ckpt_idx}, window_loss={best_loss:.4f}) from {best_path}")

    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass

    weights_path = out_dir / f"gravity_local_holevo_hybrid_{profile.name}_final.weights.h5"
    controller.save_weights(str(weights_path))
    print(f"[ok] Saved controller weights to {weights_path}")

    _save_loss_history(out_dir, sim_str, loss_history, profile.interval_save)
    _save_run_config(out_dir, profile=profile, cfg=cfg, bank_cfg=bank_cfg)

    if DEBUG.enabled:
        print(f"[ok] Debug rollout log saved to {debug_path.resolve()}")
    print(f"[done] All outputs saved to {out_dir.resolve()}")


def run_evaluation(profile: RunProfile, weights_path: Optional[Path] = None) -> None:
    out_dir = Path(profile.out_dir)
    if weights_path is None:
        weights_path = out_dir / f"gravity_local_holevo_hybrid_{profile.name}_final.weights.h5"
    if not weights_path.exists():
        print(f"[warn] Weights not found at {weights_path}.")
        return

    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)
    simulation, bank, controller = build_local_holevo_hierarchical_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )
    controller.load_weights(str(weights_path))

    all_mse = []
    all_g_true = []
    all_g_hat = []
    all_vh_local = []

    for _ in trange(profile.eval_iters, desc="Evaluating", unit="ep"):
        result = simulation.execute(rangen, deploy=True, debug=False)
        g_true = result[0][:, 0, 0].numpy()
        g_hat = bank.map_mode_mean().numpy()
        mse_batch = float(np.mean((g_hat - g_true) ** 2))
        all_mse.append(mse_batch)
        all_g_true.append(g_true)
        all_g_hat.append(g_hat)
        all_vh_local.append(float(tf.reduce_mean(bank.current_local_holevo()).numpy()))

    all_mse_arr = np.array(all_mse)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)
    global_rmse = float(np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat) ** 2)))
    print(f"Mean MSE: {np.mean(all_mse_arr):.4e}")
    print(f"Global RMSE: {global_rmse:.4e}")
    print(f"Mean local Holevo: {np.mean(np.array(all_vh_local)):.4e}")


def main() -> None:
    profile = get_profile()
    set_global_reproducibility(profile.seed)
    tf.keras.backend.clear_session()
    if RUN_MODE in {"all", "train-only"}:
        run_training(profile)
    if RUN_MODE in {"all", "eval-only"}:
        run_evaluation(profile)


if __name__ == "__main__":
    main()
