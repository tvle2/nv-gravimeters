from __future__ import annotations

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
)

from gravity_plotting import plot_branchbank_run



# =============================================================================
# USER SWITCHES
# =============================================================================

RUN_PROFILE = "pilot"   # "pilot" or "full"
RUN_MODE = "all"        # "all" | "train-only" | "eval-only" | "plots-only"


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
    top_k_branches: int
    init_mode: str
    hidden_sizes: tuple[int, ...]

    control_history_iters: int
    eval_iters: int
    plots_bins: int


PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravimeter_branchbank_pilot",

    batchsize=128,
    iterations=2000,
    interval_save=128,
    max_steps=128,
    max_resources=0.6,
    initial_lr=3e-4,
    seed=123,
    gradient_accumulation=2,

    num_branches=4,
    particles_per_branch=256,
    top_k_branches=4,   # keep equal to num_branches for exact branch-aware encoding
    init_mode="stratified_g",
    hidden_sizes=(128, 128, 128, 128),

    control_history_iters=32,
    eval_iters=64,
    plots_bins=40,
)

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravimeter_branchbank_full",

    batchsize=128,
    iterations=4000,
    interval_save=128,
    max_steps=256,
    max_resources=0.6,
    initial_lr=3e-4,
    seed=123,
    gradient_accumulation=4,

    num_branches=4,
    particles_per_branch=512,
    top_k_branches=4,   # keep equal to num_branches for exact branch-aware encoding
    init_mode="stratified_g",
    hidden_sizes=(128, 128, 128, 128),

    control_history_iters=32,
    eval_iters=64,
    plots_bins=40,
)


def get_profile() -> RunProfile:
    key = RUN_PROFILE.strip().lower()
    if key == "pilot":
        return PILOT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Use 'pilot' or 'full'.")


def make_cfg() -> GravimeterConfig:
    # Edit here if you want to override physical defaults.
    return default_cfg()


def make_bank_cfg(profile: RunProfile) -> BranchBankConfig:
    if profile.top_k_branches > profile.num_branches:
        raise ValueError("top_k_branches must be <= num_branches")

    return BranchBankConfig(
        num_branches=profile.num_branches,
        particles_per_branch=profile.particles_per_branch,
        top_k_branches=profile.top_k_branches,
        init_mode=profile.init_mode,
        hidden_sizes=profile.hidden_sizes,
    )


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
    )


def build_untrained_stack(profile: RunProfile, cfg: GravimeterConfig, bank_cfg: BranchBankConfig):
    cov_weight_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
    ]
    return build_branchbank_gravity_simulation(
        batchsize=profile.batchsize,
        sim_name=f"gravimeter_branchbank_{profile.name}",
        cfg=cfg,
        bank_cfg=bank_cfg,
        max_steps=profile.max_steps,
        max_resources=profile.max_resources,
        initial_lr=profile.initial_lr,
        cov_weight_matrix=cov_weight_matrix,
        cumulative_loss=True,
        loss_logl_outcomes=True,
        baseline_correction=True,
        pf_gamma=1.0,
    )


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
            gradient_accumulation=profile.gradient_accumulation,
        )
        weights_path = out_dir / "gravity_branchbank_control_net.weights.h5"
        net.save_weights(str(weights_path))
        print(f"[ok] Saved network weights to {weights_path}")
    else:
        phys, pf, sim, net, optimizer = build_untrained_stack(profile, cfg, bank_cfg)
        weights_path = out_dir / "gravity_branchbank_control_net.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        net.load_weights(str(weights_path))
        print(f"[ok] Loaded network weights from {weights_path}")

    if RUN_MODE in {"all", "eval-only"}:
        controls_dir = out_dir / "controls"
        eval_dir = out_dir / "eval"

        controls_csv = export_branchbank_control_history(
            sim=sim,
            out_dir=controls_dir,
            iterations=profile.control_history_iters,
            seed=profile.seed + 1,
        )

        eval_csv = evaluate_branchbank_precision(
            sim=sim,
            out_dir=eval_dir,
            iterations=profile.eval_iters,
            seed=profile.seed + 2,
        )

        print(f"[ok] Exported control history to {controls_csv}")
        print(f"[ok] Exported evaluation data to {eval_csv}")

    if RUN_MODE in {"all", "plots-only", "eval-only"}:
        plot_dir = plot_branchbank_run(out_dir, bins=profile.plots_bins)
        print(f"[ok] Saved plots to {plot_dir}")

    print(f"[done] Outputs saved in: {out_dir.resolve()}")


def main() -> None:
    run_profile()


if __name__ == "__main__":
    main()