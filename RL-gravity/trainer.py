# trainer.py
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dataclasses import dataclass
from pathlib import Path

from gravimeter_model import (
    GravimeterConfig,
    BranchBankConfig,
    build_branchbank_gravity_simulation,
    train_branchbank_gravity_modelaware,
    export_branchbank_control_history,
    evaluate_branchbank_precision,
    default_cfg,
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

RUN_PROFILE = "full"   # "pilot" or "full"
RUN_MODE = "all"        # "all" | "train-only" | "eval-only" | "plots-only"
OBJECTIVE_MODE = "gravity_only"   # "gravity_only" or "joint"
EVAL_METRIC_MODE = "g_only"   # "g_only" or "same_as_training"
TRAIN_CUMULATIVE_LOSS = True
TRAIN_LOG_LOSS = False
PF_BETA = 0.98
PF_GAMMA = 0.98
TRAIN_G_LOSS_SCALE = 1.0e5
TRAIN_A_LOSS_SCALE = 1.0e3

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

    num_branches: int
    particles_per_branch: int
    init_mode: str
    hidden_sizes: tuple[int, ...]

    control_history_iters: int
    eval_iters: int
    plots_bins: int


PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravimeter_branchbank_pilot_gA_joint",

    batchsize=32,
    iterations=4000,
    interval_save=32,
    max_steps=170,
    max_resources=0.1,
    initial_lr=3e-4,
    seed=123,
    gradient_accumulation=8,
    num_branches=4,
    particles_per_branch=256, 
    init_mode="stratified_g", 
    hidden_sizes=(128, 128, 128, 128),
    control_history_iters=128,
    eval_iters=1024,
    plots_bins=40,
)


# FULL_PROFILE = RunProfile(
#     name="full",
#     out_dir="runs/gravimeter_branchbank_full_gA_direct3head_joint",

#     batchsize=16,
#     iterations=5000,
#     interval_save=16,
#     max_steps=768,
#     max_resources=0.38,
#     initial_lr=3e-4,
#     seed=123,
#     gradient_accumulation=8,

#     num_branches=4,
#     particles_per_branch=512,
#     init_mode="stratified_g",
#     hidden_sizes=(128, 192, 192, 128),

#     control_history_iters=128,
#     eval_iters=1024,
#     plots_bins=40,
# )

# ################ Working profile for joint ################
# FULL_PROFILE = RunProfile( 
#     name="full", 
#     out_dir="runs/gravimeter_branchbank_full_gA_direct3head_joint", 
#     batchsize=32, 
#     iterations=5000, 
#     interval_save=16, 
#     max_steps=512, 
#     max_resources=0.38, 
#     initial_lr=3e-4, 
#     seed=123, 
#     gradient_accumulation=8, 
#     num_branches=4, 
#     particles_per_branch=512, 
#     init_mode="stratified_g", 
#     hidden_sizes=(128, 192, 192, 128), 
#     control_history_iters=128, 
#     eval_iters=1024, 
#     plots_bins=40, 
# )
#################################################

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravimeter_branchbank_full_gA_direct3head_gonly_v12",

    batchsize=16,
    iterations=4000,
    interval_save=32,
    max_steps=910,
    max_resources=0.5,
    initial_lr=3e-4,
    seed=123,
    gradient_accumulation=8, #8,12, 16

    num_branches=4,
    particles_per_branch=512,
    init_mode="stratified_g",
    hidden_sizes=(128,128, 128),

    control_history_iters=128,
    eval_iters=2048,
    plots_bins=80,
)


def get_profile() -> RunProfile:
    key = RUN_PROFILE.strip().lower()
    if key == "pilot":
        return PILOT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Use 'pilot' or 'full'.")

def selected_eval_cov_weight_matrix():
    if EVAL_METRIC_MODE == "g_only":
        return gravity_only_cov_weight_matrix()
    if EVAL_METRIC_MODE == "same_as_training":
        return selected_cov_weight_matrix()
    raise ValueError(f"Unknown EVAL_METRIC_MODE={EVAL_METRIC_MODE!r}")

def selected_train_cov_weight_matrix():
    if OBJECTIVE_MODE == "gravity_only":
        return training_gravity_only_cov_weight_matrix(scale=TRAIN_G_LOSS_SCALE)
    if OBJECTIVE_MODE == "joint":
        return training_joint_cov_weight_matrix(
            g_scale=TRAIN_G_LOSS_SCALE,
            A_scale=TRAIN_A_LOSS_SCALE,
        )
    raise ValueError(f"Unknown OBJECTIVE_MODE={OBJECTIVE_MODE!r}")

def make_cfg() -> GravimeterConfig:
    # Edit here if you want to override physical defaults.
    return default_cfg()


def make_bank_cfg(profile: RunProfile) -> BranchBankConfig:
    return BranchBankConfig(
        num_branches=profile.num_branches,
        particles_per_branch=profile.particles_per_branch,
        init_mode=profile.init_mode,
        hidden_sizes=profile.hidden_sizes,
    )

def selected_cov_weight_matrix():
    if OBJECTIVE_MODE == "gravity_only":
        return gravity_only_cov_weight_matrix()
    if OBJECTIVE_MODE == "joint":
        return default_joint_cov_weight_matrix()
    raise ValueError(f"Unknown OBJECTIVE_MODE={OBJECTIVE_MODE!r}")

def save_manifest(out_dir: Path, *, cfg: GravimeterConfig, bank_cfg: BranchBankConfig, profile: RunProfile) -> None:
    dump_run_config(
        out_dir / "run_config.json",
        cfg=cfg,
        bank_cfg=bank_cfg,
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
        pf_beta=PF_BETA,
        pf_gamma=PF_GAMMA,
        train_g_loss_scale=TRAIN_G_LOSS_SCALE,
        train_A_loss_scale=TRAIN_A_LOSS_SCALE,
    )


def build_untrained_stack(profile: RunProfile, cfg: GravimeterConfig, bank_cfg: BranchBankConfig):
    return build_branchbank_gravity_simulation(
        batchsize=profile.batchsize,
        sim_name=f"gravimeter_branchbank_{profile.name}",
        cfg=cfg,
        bank_cfg=bank_cfg,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        initial_lr=profile.initial_lr,
        cov_weight_matrix=selected_cov_weight_matrix(),
        cumulative_loss=TRAIN_CUMULATIVE_LOSS,
        log_loss=TRAIN_LOG_LOSS,
        loss_logl_outcomes=True,
        baseline_correction=True,
        pf_beta=PF_BETA,
        pf_gamma=PF_GAMMA,
    )

def build_loaded_eval_sim(
    profile: RunProfile,
    cfg: GravimeterConfig,
    bank_cfg: BranchBankConfig,
    weights_path: Path,
):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    _, _, eval_sim, eval_net, _ = build_branchbank_gravity_simulation(
        batchsize=profile.batchsize,
        sim_name=f"gravimeter_branchbank_{profile.name}_eval",
        cfg=cfg,
        bank_cfg=bank_cfg,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        initial_lr=profile.initial_lr,
        cov_weight_matrix=selected_eval_cov_weight_matrix(),
        cumulative_loss=TRAIN_CUMULATIVE_LOSS,
        log_loss=TRAIN_LOG_LOSS,
        loss_logl_outcomes=True,
        baseline_correction=True,
        pf_beta=PF_BETA,
        pf_gamma=PF_GAMMA,
    )
    eval_net.load_weights(str(weights_path))
    return eval_sim

def run_profile() -> None:
    profile = get_profile()
    cfg = make_cfg()
    bank_cfg = make_bank_cfg(profile)

    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(out_dir, cfg=cfg, bank_cfg=bank_cfg, profile=profile)

    print("=" * 72)
    print(f"RUN_PROFILE : {profile.name}")
    print(f"RUN_MODE    : {RUN_MODE}")
    print(f"OUT_DIR     : {out_dir.resolve()}")
    print("=" * 72)

    weights_path = out_dir / "gravity_branchbank_control_net.weights.h5"

    if RUN_MODE in {"all", "train-only"}:
        phys, pf, sim, net, optimizer = train_branchbank_gravity_modelaware(
            out_dir=out_dir,
            batchsize=profile.batchsize,
            iterations=profile.iterations,
            interval_save=profile.interval_save,
            cfg=cfg,
            bank_cfg=bank_cfg,
            sim_name=f"gravimeter_branchbank_{profile.name}",
            max_steps=profile.max_steps,
            max_resources=profile.max_resources,
            initial_lr=profile.initial_lr,
            seed=profile.seed,
            cumulative_loss=TRAIN_CUMULATIVE_LOSS,
            log_loss=TRAIN_LOG_LOSS,
            gradient_accumulation=profile.gradient_accumulation,
            cov_weight_matrix=selected_train_cov_weight_matrix(),
            pf_beta=PF_BETA,
            pf_gamma=PF_GAMMA,
        )
        net.save_weights(str(weights_path))
        print(f"[ok] Saved network weights to {weights_path}")
    else:
        phys, pf, sim, net, optimizer = build_untrained_stack(profile, cfg, bank_cfg)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        net.load_weights(str(weights_path))
        print(f"[ok] Loaded network weights from {weights_path}")

    if RUN_MODE in {"all", "eval-only"}:
        eval_sim = build_loaded_eval_sim(
            profile=profile,
            cfg=cfg,
            bank_cfg=bank_cfg,
            weights_path=weights_path,
        )

        controls_dir = out_dir / "controls"
        eval_dir = out_dir / "eval"

        controls_csv = export_branchbank_control_history(
            sim=eval_sim,
            out_dir=controls_dir,
            iterations=profile.control_history_iters,
            seed=profile.seed + 1,
        )

        eval_metric_label = "MSE_g" if EVAL_METRIC_MODE == "g_only" else "Weighted MSE"

        eval_csv = evaluate_branchbank_precision(
            sim=eval_sim,
            out_dir=eval_dir,
            iterations=profile.eval_iters,
            seed=profile.seed + 2,
            metric_label=eval_metric_label,
        )

        print(f"[ok] Exported control history to {controls_csv}")
        print(f"[ok] Exported evaluation data to {eval_csv}")

    if RUN_MODE in {"all", "plots-only", "eval-only"}:
        controls_csv = out_dir / "controls" / "branchbank_controls.csv"
        eval_csv = out_dir / "eval" / "branchbank_eval.csv"
        if not controls_csv.exists() or not eval_csv.exists():
            print("[warn] Skipping plots: control/eval CSVs not found. Run eval first.")
        else:
            plot_dir = plot_branchbank_run(out_dir, bins=profile.plots_bins)
            print(f"[ok] Saved plots to {plot_dir}")

    print(f"[done] Outputs saved in: {out_dir.resolve()}")


def main() -> None:
    run_profile()


if __name__ == "__main__":
    main()