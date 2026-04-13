# trainer_hierarchical_complete.py
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
    every_train_steps: int = 10      # how often to print training summaries
    eval_episode_every: int = 10    # run one debug deploy episode every N iterations
    max_batch_examples: int = 3      # how many batch elements to log in detail
    jsonl_path: str = "debug_rollout.jsonl"


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
from gravimeter_hierarchical_pf_complete import HierarchicalPFConfig, build_hierarchical_simulation

RUN_PROFILE: str = "smoke"
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
    out_dir="runs/gravity_hierarchical_complete_smoke",
    batchsize=16,
    iterations=300,
    interval_save=50,
    max_steps=32,
    max_resources=0.04,
    initial_lr=5e-4,
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
    out_dir="runs/gravity_hierarchical_complete_pilot",
    batchsize=32,
    iterations=2000,
    interval_save=100,
    max_steps=32,
    max_resources=0.08,
    initial_lr=3e-4,
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

DEBUG = DebugConfig(
    enabled=True,
    every_train_steps=25,
    eval_episode_every=100,
    max_batch_examples=3,
    jsonl_path="runs/debug/debug_rollout.jsonl",
)


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


def make_bank_cfg(profile: RunProfile) -> HierarchicalPFConfig:
    return HierarchicalPFConfig(
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
    )


def make_sim_pars(profile: RunProfile, cfg: GravimeterConfig) -> SimulationParameters:
    return SimulationParameters(
        sim_name=f"gravity_hierarchical_{profile.name}",
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


def _save_loss_history(out_dir: Path, sim_str: str, loss_history: list, interval_save: int) -> None:
    import pandas as pd
    arr = np.array(loss_history)
    num_blocks = len(arr) // interval_save
    if num_blocks == 0:
        return
    arr_trimmed = arr[: num_blocks * interval_save]
    mean_loss = arr_trimmed.reshape(num_blocks, interval_save).mean(axis=1)
    pd.DataFrame({"Loss": mean_loss}).to_csv(str(out_dir / f"{sim_str}_history.csv"), index=False, float_format="%.4e")


def _save_run_config(out_dir: Path, *, profile: RunProfile, cfg: GravimeterConfig, bank_cfg: HierarchicalPFConfig) -> None:
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
            "mfg_rel_noise_bound": cfg.mfg_rel_noise_bound,
            "sigma_omega_rel": cfg.sigma_omega_rel,
            "trap_visibility_mode": cfg.trap_visibility_mode,
            "prec": cfg.prec,
        },
        "bank_cfg": {
            "n_particles": bank_cfg.n_particles,
            "n_levels": bank_cfg.n_levels,
            "n_disambig_per_level": bank_cfg.n_disambig_per_level,
            "resample_threshold": bank_cfg.resample_threshold,
            "resample_alpha": bank_cfg.resample_alpha,
            "resample_beta": bank_cfg.resample_beta,
        },
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

def run_training(profile: RunProfile) -> None:
    """Run the full hierarchical PF training pipeline with debug logging."""

    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run profile  : {profile.name}")
    print(f"Noise mode   : {NOISE_MODE}")
    print(f"Run mode     : {RUN_MODE}")
    print(f"Output dir   : {out_dir.resolve()}")
    print("=" * 72)

    # --- Configuration ---
    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)

    rangen = tf.random.Generator.from_seed(profile.seed)

    # --- Build all components ---
    simulation, bank, controller = build_hierarchical_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    variables = controller.trainable_variables
    n_params = sum(int(np.prod(v.shape)) for v in variables)

    print(f"[info] Controller parameters: {n_params:,}")
    print(
        f"[info] Bank: N_particles={bank_cfg.n_particles}, "
        f"N_levels={bank_cfg.n_levels}, "
        f"N_disambig/level={bank_cfg.n_disambig_per_level}"
    )
    print(
        f"[info] Simulation: {simpars.num_steps} steps, "
        f"max_resources={simpars.max_resources:.3f} s, "
        f"batch={profile.batchsize}"
    )
    print(
        f"[info] g_range=[{cfg.g_range[0]:.4f}, {cfg.g_range[1]:.4f}], "
        f"width={cfg.g_range[1] - cfg.g_range[0]:.4e} m/s²"
    )
    print(
        f"[info] After {bank_cfg.n_levels} levels: "
        f"interval width ≈ "
        f"{(cfg.g_range[1] - cfg.g_range[0]) / 2**bank_cfg.n_levels:.4e} m/s²"
    )

    # --- Optimizer ---
    lr_schedule = InverseSqrtDecay(
        initial_learning_rate=profile.initial_lr,
        prec=cfg.prec,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # --- Checkpoint directory ---
    sim_str = str(simulation)
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Debug output path ---
    debug_path = Path(DEBUG.jsonl_path)
    if DEBUG.enabled and debug_path.exists():
        debug_path.unlink()

    # --- Eager training loop state ---
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
        """
        One gradient-accumulation step.

        Returns
        -------
        avg_loss : float
            Average scalar loss over accumulation steps.
        grad_norm : float
            L2 norm of the averaged gradient.
        """
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]

        for _ in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                loss_diff, loss = simulation.execute(
                    rangen,
                    deploy=False,
                    debug=False,
                )
            grads = tape.gradient(loss_diff, variables)
            grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, variables)
            ]
            acc_loss += float(loss.numpy())
            acc_grads = [ag + g for ag, g in zip(acc_grads, grads)]

        acc_grads = [g / profile.gradient_accumulation for g in acc_grads]
        grad_norm = compute_grad_norm(acc_grads)
        optimizer.apply_gradients(zip(acc_grads, variables))

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

        # Main tqdm status
        pbar.set_postfix(
            loss=f"{step_loss:.4f}",
            avg10=f"{recent:.4f}",
            best=f"{best_loss:.4f}",
            grad=f"{grad_norm:.2e}",
            lr=f"{lr_val:.1e}",
            lvl=f"{simulation.bank.current_level:.2f}",
            q0=f"{simulation.bank.q0_mean:.3f}",
        )

        # -----------------------------
        # Console training summary
        # -----------------------------
        if DEBUG.enabled and ((j + 1) % DEBUG.every_train_steps == 0):
            print(
                "[debug/train] "
                f"iter={j+1} "
                f"loss={step_loss:.4e} "
                f"avg10={recent:.4e} "
                f"grad_norm={grad_norm:.4e} "
                f"mean_level={simulation.bank.current_level:.2f} "
                f"mean_q0={simulation.bank.q0_mean:.3f} "
                f"mean_width={simulation.bank.interval_width:.4e} "
                f"mean_target_kg={simulation.bank.target_k_g:.4e} "
                f"lr={lr_val:.2e}"
            )

        # -----------------------------
        # Periodic debug rollout
        # -----------------------------
        if DEBUG.enabled and ((j + 1) % DEBUG.eval_episode_every == 0):
            deploy_result = simulation.execute(
                rangen,
                deploy=True,
                debug=True,
            )
            debug_records = deploy_result[-1]

            for rec in debug_records:
                rec["train_iter"] = j + 1
                append_jsonl(debug_path, rec)

            if len(debug_records) > 0:
                sample = debug_records[0]
                print(
                    "[debug/rollout] "
                    f"iter={j+1} "
                    f"level={sample.get('level', 'NA')} "
                    f"step_in_level={sample.get('step_in_level', 'NA')} "
                    f"refining={sample.get('refining', 'NA')} "
                    f"q0={sample.get('q0', float('nan')):.3f} "
                    f"q1={sample.get('q1', float('nan')):.3f} "
                    f"branch_correct={sample.get('branch_correct', 'NA')} "
                    f"target_kg={sample.get('target_kg', float('nan')):.3e} "
                    f"actual_kg={sample.get('actual_kg', float('nan')):.3e} "
                    f"kg_ratio={sample.get('kg_ratio', float('nan')):.3f} "
                    f"T={sample.get('T_s', float('nan')):.3e} "
                    f"Bp={sample.get('Bp_kTm', float('nan')):.3f} "
                    f"phi={sample.get('mw_phase_rad', float('nan')):.3f} "
                    f"loss_total={sample.get('loss_total', float('nan')):.3e} "
                    f"loss_bce={sample.get('loss_bce', float('nan')):.3e} "
                    f"loss_mix={sample.get('loss_mix_mse', float('nan')):.3e} "
                    f"loss_refine={sample.get('loss_refine_mse', float('nan')):.3e}"
                )

        # -----------------------------
        # Save checkpoint
        # -----------------------------
        if (j + 1) % profile.interval_save == 0:
            ckpt_idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / str(ckpt_idx)
            controller.save_weights(str(ckpt_path) + ".weights.h5")

            window_start = max(0, j + 1 - profile.interval_save)
            window_loss = float(np.mean(loss_history[window_start: j + 1]))
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = ckpt_idx

    pbar.close()
    print("[training] Done.")

    # --- Load best checkpoint ---
    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(
            f"[ok] Loaded best checkpoint (idx={best_ckpt_idx}, "
            f"window_loss={best_loss:.4f}) from {best_path}"
        )

    # --- Remove temporary checkpoints ---
    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass

    # --- Save final weights ---
    weights_path = out_dir / f"gravity_hierarchical_{profile.name}_final.weights.h5"
    controller.save_weights(str(weights_path))
    print(f"[ok] Saved controller weights to {weights_path}")

    # --- Save loss history CSV ---
    _save_loss_history(out_dir, sim_str, loss_history, profile.interval_save)

    # --- Save run config ---
    _save_run_config(out_dir, profile=profile, cfg=cfg, bank_cfg=bank_cfg)

    if DEBUG.enabled:
        print(f"[ok] Debug rollout log saved to {debug_path.resolve()}")

    print(f"[done] All outputs saved to {out_dir.resolve()}")


def run_evaluation(profile: RunProfile, weights_path: Optional[Path] = None) -> None:
    out_dir = Path(profile.out_dir)
    if weights_path is None:
        weights_path = out_dir / f"gravity_hierarchical_{profile.name}_final.weights.h5"
    if not weights_path.exists():
        print(f"[warn] Weights not found at {weights_path}.")
        return
    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)
    simulation, bank, controller = build_hierarchical_simulation(batchsize=profile.batchsize, cfg=cfg, bank_cfg=bank_cfg, simpars=simpars, rangen=rangen)
    controller.load_weights(str(weights_path))
    all_mse = []
    all_g_true = []
    all_g_hat = []
    for _ in trange(profile.eval_iters, desc="Evaluating", unit="ep"):
        result = simulation.execute(rangen, deploy=True)
        g_true = result[0][:, 0, 0].numpy()
        g_hat = bank.map_mode_mean().numpy()
        mse_batch = float(np.mean((g_hat - g_true) ** 2))
        all_mse.append(mse_batch)
        all_g_true.append(g_true)
        all_g_hat.append(g_hat)
    all_mse_arr = np.array(all_mse)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)
    global_rmse = float(np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat) ** 2)))
    print(f"Mean MSE: {np.mean(all_mse_arr):.4e}")
    print(f"Global RMSE: {global_rmse:.4e}")


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
