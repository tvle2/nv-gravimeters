#run_student_pipeline.py
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict

import torch

from environment import (
    ActionGrid,
    AdaptiveBayesController,
    EnvConfig,
    GravimeterEnv,
    GravimeterModel,
    NoiseConfig,
    PlannerConfig,
    PriorConfig,
    make_paper_scale_grid,
)
from student_policy import (
    DatasetConfig,
    EvalConfig,
    FeatureConfig,
    TrainConfig,
    evaluate_student_policy,
    evaluate_teacher_policy,
    generate_teacher_dataset,
    save_json,
    save_student_checkpoint,
    train_student_policy,
)


# ============================================================
# Local helper: summarize first teacher action
# ============================================================


def summarize_first_action_local(
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
) -> dict:
    from environment import JointGravityEpsBelief

    belief = JointGravityEpsBelief(
        prior=prior,
        noise=noise,
        n_g_grid=n_g_grid,
        n_eps_grid=n_eps_grid,
    )
    belief.reset_factorized()

    aT, aB, phi, A, score = controller.plan_action(belief)
    T_s = float(controller.grid.T_vals_s[aT])
    B_kTm = float(controller.grid.Bp_vals_kTm[aB])

    return {
        "aT": int(aT),
        "aB": int(aB),
        "T_s": T_s,
        "Bp_kTm": B_kTm,
        "mw_phase_rad": float(phi),
        "visibility_A": float(A),
        "score": float(score),
        "delta_g_2pi": float(controller.model.delta_g_period_2pi(T_s, B_kTm)),
    }


# ============================================================
# USER CONFIG: edit this block only
# ============================================================

SEED = 42
OUTPUT_DIR = "outputs_student_policy"

# ---- Prior over gravity ----
PRIOR = PriorConfig(
    g_range=(9.7639, 9.8337),
)

# ---- Action grid ----
GRID: ActionGrid = make_paper_scale_grid(
    n_T=9,
    n_B=9,
    T_min_s=50e-6,
    T_max_s=1.0e-3,
    B_min_kTm=0.5,
    B_max_kTm=50.0,
    log_spacing=True,
)

# ---- Physical model ----
def mass_from_radius(r_m: float, rho_kg_m3: float = 3510.0) -> float:
    return float((4.0 / 3.0) * math.pi * (r_m ** 3) * rho_kg_m3)


MODEL = GravimeterModel(
    omega_rad_s=2.0 * math.pi * 10e3,
    gamma_e_rad_s_T=2.0 * math.pi * 28e9,
    mass_kg=mass_from_radius(100e-9),
)

# ---- Noise / global epsilon support ----
NOISE = NoiseConfig(
    sigma_omega_rel=0.0,
    trap_visibility_mode="small_noise_avg",
    mfg_rel_noise_bound=0.02,
    T2_spin_s=None,
)

# ---- Environment ----
ENV_CFG = EnvConfig(
    episode_len=32,
    n_g_grid=512,
    n_eps_grid=31,
)

# ---- Teacher planner ----
PLANNER_CFG = PlannerConfig(
    objective="coarse_to_fine",
    phase_mode="utility_search",

    phase_grid_size=31,
    fine_phase_local_grid_size=9,
    fine_phase_local_halfwidth_rad=0.60,

    coarse_g_bins=16,
    coarse_entropy_threshold_nats=1.10,
    coarse_peak1_mass_threshold=0.55,
    coarse_peak_gap_threshold=0.15,

    use_hard_alias_gate=False,
    alias_halfwidth_max_rad=1.25,
    alias_credible_mass=0.90,
    wrap_penalty_weight=0.50,
    target_wraps=0.75,

    cycle_penalty=0.0,
    mfg_penalty=0.0,
    min_visibility=0.0,

    robust_mfg_mode="nominal",
)

# ---- Student features ----
FEATURE_CFG = FeatureConfig(
    g_hist_bins=64,
    eps_hist_bins=31,
    include_peak_features=True,
    include_step_features=True,
)

# ---- Dataset generation ----
DATASET_CFG = DatasetConfig(
    n_runs=512,
    episodes_per_run=8,
    base_seed=SEED + 1000,
    log_every_runs=32,
)

# ---- Student training ----
TRAIN_CFG = TrainConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1024,
    epochs=40,
    lr=3e-4,
    weight_decay=1e-4,
    hidden_dim=256,
    depth=3,
    dropout=0.05,
    val_fraction=0.10,
    label_smoothing=0.0,
    early_stop_patience=8,
    seed=SEED,
)

# ---- Evaluation ----
EVAL_CFG = EvalConfig(
    n_runs=64,
    episodes_per_run=8,
    base_seed=SEED + 2000,
    log_every_runs=8,
    wrong_branch_peak_abs_err_threshold=5e-3,
    unresolved_halfwidth90_threshold=5e-3,
    truth_mass_band_abs=1e-3,
)

SAVE_FIRST_TRACE = True


# ============================================================
# Builders
# ============================================================


def build_env(seed: int) -> GravimeterEnv:
    return GravimeterEnv(
        prior=PRIOR,
        grid=GRID,
        model=MODEL,
        noise=NOISE,
        env_cfg=ENV_CFG,
        seed=seed,
    )


def build_teacher_controller() -> AdaptiveBayesController:
    return AdaptiveBayesController(
        model=MODEL,
        grid=GRID,
        noise=NOISE,
        cfg=PLANNER_CFG,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL.validate()

    teacher = build_teacher_controller()

    print("=== Configuration summary ===")
    print(f"seed={SEED}")
    print(f"device={TRAIN_CFG.device}")
    print(f"g range: {PRIOR.g_range}")
    print(f"grid: n_T={GRID.nT}, n_B={GRID.nB}, n_actions={GRID.n_actions}")
    print(f"T range [s]: {GRID.T_vals_s[0]:.3e} .. {GRID.T_vals_s[-1]:.3e}")
    print(f"B' range [kT/m]: {GRID.Bp_vals_kTm[0]:.3f} .. {GRID.Bp_vals_kTm[-1]:.3f}")
    print(
        f"episode_len={ENV_CFG.episode_len}, "
        f"n_g_grid={ENV_CFG.n_g_grid}, "
        f"n_eps_grid={ENV_CFG.n_eps_grid}"
    )
    print(
        "noise: "
        f"sigma_omega_rel={NOISE.sigma_omega_rel}, "
        f"trap_mode={NOISE.trap_visibility_mode}, "
        f"mfg_rel_bound={NOISE.mfg_rel_noise_bound}, "
        f"T2={NOISE.T2_spin_s}"
    )
    print(
        "planner: "
        f"objective={PLANNER_CFG.objective}, "
        f"phase_mode={PLANNER_CFG.phase_mode}, "
        f"robust_mfg_mode={PLANNER_CFG.robust_mfg_mode}, "
        f"alias_halfwidth_max_rad={PLANNER_CFG.alias_halfwidth_max_rad}, "
        f"alias_credible_mass={PLANNER_CFG.alias_credible_mass}"
    )
    print(
        "dataset: "
        f"n_runs={DATASET_CFG.n_runs}, "
        f"episodes_per_run={DATASET_CFG.episodes_per_run}"
    )
    print(
        "eval: "
        f"n_runs={EVAL_CFG.n_runs}, "
        f"episodes_per_run={EVAL_CFG.episodes_per_run}"
    )

    print("\n=== Teacher first action on prior ===")
    first_action = summarize_first_action_local(
        prior=PRIOR,
        noise=NOISE,
        controller=teacher,
        n_g_grid=ENV_CFG.n_g_grid,
        n_eps_grid=ENV_CFG.n_eps_grid,
    )
    print(json.dumps(first_action, indent=2))

    print("\n=== Generating teacher dataset ===")
    dataset = generate_teacher_dataset(
        env_ctor=build_env,
        prior=PRIOR,
        noise=NOISE,
        controller=teacher,
        n_g_grid=ENV_CFG.n_g_grid,
        n_eps_grid=ENV_CFG.n_eps_grid,
        feature_cfg=FEATURE_CFG,
        dataset_cfg=DATASET_CFG,
    )
    print(
        json.dumps(
            {
                "n_samples": int(dataset["X"].shape[0]),
                "input_dim": int(dataset["X"].shape[1]),
                "n_actions": int(dataset["n_actions"]),
            },
            indent=2,
        )
    )

    print("\n=== Training student policy ===")
    trained, history = train_student_policy(
        X=dataset["X"],
        y_action=dataset["y_action"],
        run_ids=dataset["run_ids"],
        n_actions=dataset["n_actions"],
        feature_cfg=FEATURE_CFG,
        train_cfg=TRAIN_CFG,
    )
    ckpt_path = os.path.join(OUTPUT_DIR, "student_policy_checkpoint.pt")
    save_student_checkpoint(
        path=ckpt_path,
        trained=trained,
        n_actions=dataset["n_actions"],
        train_cfg=TRAIN_CFG,
    )

    print("\n=== Evaluating teacher policy ===")
    teacher_metrics, teacher_trace, teacher_cat = evaluate_teacher_policy(
        env_ctor=build_env,
        prior=PRIOR,
        noise=NOISE,
        controller=teacher,
        eval_cfg=EVAL_CFG,
        return_first_trace=SAVE_FIRST_TRACE,
    )
    print(json.dumps(teacher_metrics, indent=2))

    print("\n=== Evaluating student policy ===")
    student_metrics, student_trace, student_cat = evaluate_student_policy(
        env_ctor=build_env,
        prior=PRIOR,
        noise=NOISE,
        controller=teacher,   # phase still exact; only action search is learned
        trained=trained,
        eval_cfg=EVAL_CFG,
        device=TRAIN_CFG.device,
        return_first_trace=SAVE_FIRST_TRACE,
    )
    print(json.dumps(student_metrics, indent=2))

    dataset_meta = {
        "feature_cfg": asdict(FEATURE_CFG),
        "dataset_cfg": asdict(DATASET_CFG),
        "n_samples": int(dataset["X"].shape[0]),
        "input_dim": int(dataset["X"].shape[1]),
        "n_actions": int(dataset["n_actions"]),
    }

    run_config = {
        "seed": SEED,
        "prior": asdict(PRIOR),
        "grid": {
            "n_T": GRID.nT,
            "n_B": GRID.nB,
            "T_vals_s": GRID.T_vals_s.tolist(),
            "Bp_vals_kTm": GRID.Bp_vals_kTm.tolist(),
        },
        "model": {
            "omega_rad_s": MODEL.omega_rad_s,
            "gamma_e_rad_s_T": MODEL.gamma_e_rad_s_T,
            "mass_kg": MODEL.mass_kg,
            "hbar_J_s": MODEL.hbar_J_s,
            "kT_to_T": MODEL.kT_to_T,
            "eps_prob": MODEL.eps_prob,
        },
        "noise": asdict(NOISE),
        "env_cfg": asdict(ENV_CFG),
        "planner_cfg": asdict(PLANNER_CFG),
        "feature_cfg": asdict(FEATURE_CFG),
        "dataset_cfg": asdict(DATASET_CFG),
        "train_cfg": asdict(TRAIN_CFG),
        "eval_cfg": asdict(EVAL_CFG),
    }

    summary = {
        "teacher_first_action": first_action,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "teacher_catastrophic_count": int(len(teacher_cat)),
        "student_catastrophic_count": int(len(student_cat)),
    }

    save_json(os.path.join(OUTPUT_DIR, "run_config.json"), run_config)
    save_json(os.path.join(OUTPUT_DIR, "dataset_meta.json"), dataset_meta)
    save_json(os.path.join(OUTPUT_DIR, "training_history.json"), history)
    save_json(os.path.join(OUTPUT_DIR, "summary.json"), summary)
    save_json(os.path.join(OUTPUT_DIR, "teacher_metrics.json"), teacher_metrics)
    save_json(os.path.join(OUTPUT_DIR, "student_metrics.json"), student_metrics)
    save_json(os.path.join(OUTPUT_DIR, "teacher_catastrophic_records.json"), {"records": teacher_cat})
    save_json(os.path.join(OUTPUT_DIR, "student_catastrophic_records.json"), {"records": student_cat})

    if teacher_trace is not None:
        save_json(os.path.join(OUTPUT_DIR, "teacher_first_trace.json"), {"trace": teacher_trace})
    if student_trace is not None:
        save_json(os.path.join(OUTPUT_DIR, "student_first_trace.json"), {"trace": student_trace})

    print("\nSaved files:")
    print(f"  {os.path.join(OUTPUT_DIR, 'student_policy_checkpoint.pt')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'run_config.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'dataset_meta.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'training_history.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'summary.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'teacher_metrics.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'student_metrics.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'teacher_catastrophic_records.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'student_catastrophic_records.json')}")
    if teacher_trace is not None:
        print(f"  {os.path.join(OUTPUT_DIR, 'teacher_first_trace.json')}")
    if student_trace is not None:
        print(f"  {os.path.join(OUTPUT_DIR, 'student_first_trace.json')}")


if __name__ == "__main__":
    main()