# trainer.py
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from gravimeter_model import (
    GravimeterConfig,
    GravityPFConfig,
    build_gravity_simulation,
    train_gravity_modelaware,
    export_control_history,
    evaluate_precision,
    dump_run_config,
    default_joint_cov_weight_matrix,
    gravity_only_cov_weight_matrix,
    training_gravity_only_cov_weight_matrix,
    training_joint_cov_weight_matrix,
)
from gravity_plotting import plot_branchbank_run

# =============================================================================
# USER SWITCHES
# =============================================================================

RUN_PROFILE = "pilot"   # "pilot" or "full"
RUN_MODE = "all"        # "all" | "train-only" | "eval-only" | "plots-only"
OBJECTIVE_MODE = "gravity_only"   # "gravity_only" or "joint"
EVAL_METRIC_MODE = "g_only"       # "g_only" or "same_as_training"

NOISE_MODE = "none"              # "none" | "paper" | "calibration_mismatch"
INFER_MFG_BIAS = False             # set True to estimate a static multiplicative MFG bias

TRAIN_CUMULATIVE_LOSS = False
TRAIN_LOG_LOSS = False

TRAIN_RESOURCES_FRACTION = 1.0
EVAL_RESOURCES_FRACTION = 1.0

TRAIN_G_LOSS_SCALE = 1.0e4
TRAIN_BETA_LOSS_SCALE = 1.0e2

# =============================================================================
# PROFILE DEFINITION
# =============================================================================


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

    num_particles: int
    hidden_sizes: tuple[int, ...]
    pf_alpha: float
    pf_beta: float
    pf_gamma: float
    pf_resample_threshold: float
    pf_resample_fraction: float

    control_history_iters: int
    eval_iters: int
    plots_bins: int


PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/single_nv_gravity_clean_pilot_no_noise",
    batchsize=8,
    iterations=1000,
    interval_save=8,
    max_steps=128,
    max_resources=0.1,
    initial_lr=1e-5,
    seed=42,
    gradient_accumulation=4,
    num_particles=1024,
    hidden_sizes=(128, 128, 128),
    pf_alpha=0.5,
    pf_beta=0.98,
    pf_gamma=0.95,
    pf_resample_threshold=0.5,
    pf_resample_fraction=0.75,
    control_history_iters=96,
    eval_iters=128,
    plots_bins=60,
)

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/single_nv_gravity_clean_full",
    batchsize=16,
    iterations=6000,
    interval_save=16,
    max_steps=192,
    max_resources=0.24,
    initial_lr=3e-5,
    seed=123,
    gradient_accumulation=8,
    num_particles=2048,
    hidden_sizes=(128, 128, 128),
    pf_alpha=0.5,
    pf_beta=0.98,
    pf_gamma=0.95,
    pf_resample_threshold=0.5,
    pf_resample_fraction=0.75,
    control_history_iters=128,
    eval_iters=512,
    plots_bins=80,
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
    if key == "pilot":
        return PILOT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Use 'pilot' or 'full'.")



def make_cfg() -> GravimeterConfig:
    common = dict(
        g_range=(9.7806, 9.825),
        infer_mfg_bias=INFER_MFG_BIAS,
        beta_B_range=(-0.10, 0.10),
        T_range_s=(3.0e-4, 1.2e-3),
        Bp_range_kTm=(20.0, 80.0),
        delta_max_rad=pi / 2.0,
        T2_spin_s=None,
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec="float32",
    )

    if NOISE_MODE == "none":
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

    if NOISE_MODE == "paper":
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

    if NOISE_MODE == "calibration_mismatch":
        return GravimeterConfig(
            **{**common, "infer_mfg_bias": True},
            mfg_rel_noise_bound=0.025,
            mfg_noise_quad_points=9,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.01,
            trap_visibility_mode="exact_single_delta",
            trap_noise_quad_points=9,
        )

    raise ValueError(f"Unknown NOISE_MODE={NOISE_MODE!r}")



def make_pf_cfg(profile: RunProfile) -> GravityPFConfig:
    return GravityPFConfig(
        num_particles=profile.num_particles,
        alpha=profile.pf_alpha,
        beta=profile.pf_beta,
        gamma=profile.pf_gamma,
        resample_threshold=profile.pf_resample_threshold,
        resample_fraction=profile.pf_resample_fraction,
        hidden_sizes=profile.hidden_sizes,
    )



def selected_train_cov_weight_matrix(cfg: GravimeterConfig):
    if OBJECTIVE_MODE == "gravity_only":
        return training_gravity_only_cov_weight_matrix(cfg, scale=TRAIN_G_LOSS_SCALE)
    if OBJECTIVE_MODE == "joint":
        return training_joint_cov_weight_matrix(
            cfg,
            g_scale=TRAIN_G_LOSS_SCALE,
            beta_scale=TRAIN_BETA_LOSS_SCALE,
        )
    raise ValueError(f"Unknown OBJECTIVE_MODE={OBJECTIVE_MODE!r}")



def selected_eval_cov_weight_matrix(cfg: GravimeterConfig):
    if EVAL_METRIC_MODE == "g_only":
        return gravity_only_cov_weight_matrix(cfg)
    if EVAL_METRIC_MODE == "same_as_training":
        return selected_train_cov_weight_matrix(cfg)
    raise ValueError(f"Unknown EVAL_METRIC_MODE={EVAL_METRIC_MODE!r}")



def save_manifest(out_dir: Path, *, cfg: GravimeterConfig, pf_cfg: GravityPFConfig, profile: RunProfile) -> None:
    dump_run_config(
        out_dir / "run_config.json",
        cfg=cfg,
        pf_cfg=pf_cfg,
        run_profile=profile.name,
        run_mode=RUN_MODE,
        batchsize=profile.batchsize,
        iterations=profile.iterations,
        interval_save=profile.interval_save,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        initial_lr=profile.initial_lr,
        seed=profile.seed,
        gradient_accumulation=profile.gradient_accumulation,
        control_history_iters=profile.control_history_iters,
        eval_iters=profile.eval_iters,
        plots_bins=profile.plots_bins,
        objective_mode=OBJECTIVE_MODE,
        eval_metric_mode=EVAL_METRIC_MODE,
        train_cumulative_loss=TRAIN_CUMULATIVE_LOSS,
        train_log_loss=TRAIN_LOG_LOSS,
        train_g_loss_scale=TRAIN_G_LOSS_SCALE,
        train_beta_loss_scale=TRAIN_BETA_LOSS_SCALE,
        noise_mode=NOISE_MODE,
        infer_mfg_bias=cfg.infer_mfg_bias,
    )



def build_untrained_stack(profile: RunProfile, cfg: GravimeterConfig, pf_cfg: GravityPFConfig):
    return build_gravity_simulation(
        batchsize=profile.batchsize,
        sim_name=f"gravimeter_{profile.name}",
        cfg=cfg,
        pf_cfg=pf_cfg,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=EVAL_RESOURCES_FRACTION,
        initial_lr=profile.initial_lr,
        cov_weight_matrix=selected_eval_cov_weight_matrix(cfg),
        cumulative_loss=TRAIN_CUMULATIVE_LOSS,
        log_loss=TRAIN_LOG_LOSS,
        loss_logl_outcomes=False,
        baseline_correction=True,
    )



def build_loaded_eval_sim(
    profile: RunProfile,
    cfg: GravimeterConfig,
    pf_cfg: GravityPFConfig,
    weights_path: Path,
):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    _, _, eval_sim, eval_net, _ = build_gravity_simulation(
        batchsize=profile.batchsize,
        sim_name=f"gravimeter_{profile.name}_eval",
        cfg=cfg,
        pf_cfg=pf_cfg,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=EVAL_RESOURCES_FRACTION,
        initial_lr=profile.initial_lr,
        cov_weight_matrix=selected_eval_cov_weight_matrix(cfg),
        cumulative_loss=TRAIN_CUMULATIVE_LOSS,
        log_loss=TRAIN_LOG_LOSS,
        loss_logl_outcomes=False,
        baseline_correction=True,
    )
    eval_net.load_weights(str(weights_path))
    return eval_sim



def run_profile() -> None:
    profile = get_profile()
    tf.keras.backend.clear_session()
    set_global_reproducibility(profile.seed)

    cfg = make_cfg()
    pf_cfg = make_pf_cfg(profile)

    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(out_dir, cfg=cfg, pf_cfg=pf_cfg, profile=profile)

    print("=" * 72)
    print(f"RUN_PROFILE : {profile.name}")
    print(f"RUN_MODE    : {RUN_MODE}")
    print(f"OUT_DIR     : {out_dir.resolve()}")
    print("=" * 72)

    weights_path = out_dir / "gravity_control_net.weights.h5"

    if RUN_MODE in {"all", "train-only"}:
        phys, pf, sim, net, optimizer = train_gravity_modelaware(
            out_dir=out_dir,
            batchsize=profile.batchsize,
            iterations=profile.iterations,
            interval_save=profile.interval_save,
            cfg=cfg,
            pf_cfg=pf_cfg,
            sim_name=f"gravimeter_{profile.name}",
            max_steps=profile.max_steps,
            max_resources=profile.max_resources,
            resources_fraction=TRAIN_RESOURCES_FRACTION,
            initial_lr=profile.initial_lr,
            seed=profile.seed,
            cumulative_loss=TRAIN_CUMULATIVE_LOSS,
            log_loss=TRAIN_LOG_LOSS,
            gradient_accumulation=profile.gradient_accumulation,
            cov_weight_matrix=selected_train_cov_weight_matrix(cfg),
        )
        net.save_weights(str(weights_path))
        print(f"[ok] Saved network weights to {weights_path}")
    else:
        phys, pf, sim, net, optimizer = build_untrained_stack(profile, cfg, pf_cfg)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        net.load_weights(str(weights_path))
        print(f"[ok] Loaded network weights from {weights_path}")

    if RUN_MODE in {"all", "eval-only"}:
        eval_sim = build_loaded_eval_sim(profile=profile, cfg=cfg, pf_cfg=pf_cfg, weights_path=weights_path)

        controls_dir = out_dir / "controls"
        eval_dir = out_dir / "eval"
        for d in (controls_dir, eval_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        export_control_history(
            sim=eval_sim,
            out_dir=controls_dir,
            iterations=profile.control_history_iters,
            seed=profile.seed + 1,
        )

        eval_metric_label = "MSE_g" if EVAL_METRIC_MODE == "g_only" else "Weighted MSE"
        evaluate_precision(
            sim=eval_sim,
            out_dir=eval_dir,
            iterations=profile.eval_iters,
            seed=profile.seed + 2,
            metric_label=eval_metric_label,
        )

    if RUN_MODE in {"all", "plots-only", "eval-only"}:
        controls_csv = out_dir / "controls" / "branchbank_controls.csv"
        eval_csv = out_dir / "eval" / "branchbank_eval.csv"
        if not controls_csv.exists() or not eval_csv.exists():
            print("[warn] Skipping plots: control/eval CSVs not found. Run eval first.")
        else:
            try:
                plot_dir = plot_branchbank_run(out_dir, bins=profile.plots_bins)
                print(f"[ok] Saved plots to {plot_dir}")
            except Exception as exc:
                print(f"[warn] Plotting skipped: {exc}")

    print(f"[done] Outputs saved in: {out_dir.resolve()}")



def main() -> None:
    run_profile()


if __name__ == "__main__":
    main()
