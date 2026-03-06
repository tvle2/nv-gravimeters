# run_pipeline.py
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict

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
from adaptive_pipeline import (
    choose_reference_fixed_action,
    evaluate_adaptive_controller,
    evaluate_fixed_action_baseline,
    save_json,
    summarize_first_action,
)


# ============================================================
# USER CONFIG: edit this block only
# ============================================================

SEED = 0
OUTPUT_DIR = "outputs_bayesian_adaptive"

# ---- Prior over gravity ----
PRIOR = PriorConfig(
    g_range=(9.7639, 9.8337),
)

# ---- Action grid ----
# Debug example:
#   n_T=9, n_B=9, n_g_grid=256, episode_len=32
#
# Full example:
#   n_T=21, n_B=21, n_g_grid=1024, episode_len=64
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

# ---- Noise model ----
# paper-faithful clean baseline:
#   sigma_omega_rel=0.0
#   trap_visibility_mode="small_noise_avg"
#   mfg_rel_noise_bound=0.0
#
# mild robustness example:
#   sigma_omega_rel=0.002
#   trap_visibility_mode="small_noise_avg"
#   mfg_rel_noise_bound=0.02

NOISE = NoiseConfig(
    sigma_omega_rel=0.0,
    trap_visibility_mode="small_noise_avg",
    mfg_rel_noise_bound=0.02,
    T2_spin_s=None,
)

# ---- Environment ----
ENV_CFG = EnvConfig(
    episode_len=32,
    n_g_grid=1024,
)

# ---- Planner / adaptive controller ----
# ---- Planner / adaptive controller ----
PLANNER_CFG = PlannerConfig(
    objective="variance_reduction",
    phase_mode="analytic_quadrature",
    phase_grid_size=181,
    robust_mfg_mode="nominal",

    # Credible-width alias gate:
    # reject if k_max * h90 > alias_halfwidth_max_rad
    alias_halfwidth_max_rad=1.75,
    alias_credible_mass=0.90,

    cycle_penalty=0.0,
    mfg_penalty=0.0,
    min_visibility=0.0,
)

# ---- Evaluation ----
ADAPTIVE_EVAL_EPISODES = 256
ADAPTIVE_EVAL_LOG_EVERY = 32

# ---- Catastrophic diagnostics ----
CATASTROPHIC_ABS_ERR_THRESHOLD = 5e-4
CATASTROPHIC_LOG_TOP_K = 5
SAVE_CATASTROPHIC_RECORDS = True

# Fixed-action baseline for comparison
RUN_FIXED_BASELINE = True
FIXED_BASELINE_EPISODES = 64
FIXED_BASELINE_LOG_EVERY = 8
FIXED_BASELINE_ADAPTIVE_PHASE = True

# Save a full step-by-step trace for the first adaptive episode
SAVE_FIRST_TRACE = True


# ============================================================
# Helpers
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


def build_controller() -> AdaptiveBayesController:
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

    controller = build_controller()

    print("=== Configuration summary ===")
    print(f"seed={SEED}")
    print(f"g range: {PRIOR.g_range}")
    print(f"grid: n_T={GRID.nT}, n_B={GRID.nB}, n_actions={GRID.n_actions}")
    print(f"T range [s]: {GRID.T_vals_s[0]:.3e} .. {GRID.T_vals_s[-1]:.3e}")
    print(f"B' range [kT/m]: {GRID.Bp_vals_kTm[0]:.3f} .. {GRID.Bp_vals_kTm[-1]:.3f}")
    print(f"episode_len={ENV_CFG.episode_len}, n_g_grid={ENV_CFG.n_g_grid}")
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

    print("\n=== Adaptive controller first action ===")
    first_action = summarize_first_action(PRIOR, controller, ENV_CFG.n_g_grid)
    print(json.dumps(first_action, indent=2))

    print("\n=== Adaptive controller evaluation ===")
    adaptive_metrics, adaptive_trace, adaptive_catastrophics = evaluate_adaptive_controller(
        env_ctor=build_env,
        controller=controller,
        episodes=ADAPTIVE_EVAL_EPISODES,
        log_every=ADAPTIVE_EVAL_LOG_EVERY,
        return_first_trace=SAVE_FIRST_TRACE,
        catastrophic_abs_err_threshold=CATASTROPHIC_ABS_ERR_THRESHOLD,
        catastrophic_log_top_k=CATASTROPHIC_LOG_TOP_K,
        save_catastrophic_records=SAVE_CATASTROPHIC_RECORDS,
    )
    print(json.dumps(adaptive_metrics, indent=2))

    fixed_baseline_metrics = None
    fixed_baseline_trace = None
    fixed_action_summary = None

    if RUN_FIXED_BASELINE:
        fixed_aT, fixed_aB = choose_reference_fixed_action(PRIOR, controller, ENV_CFG.n_g_grid)
        fixed_T_s = float(GRID.T_vals_s[fixed_aT])
        fixed_B_kTm = float(GRID.Bp_vals_kTm[fixed_aB])

        fixed_action_summary = {
            "aT": int(fixed_aT),
            "aB": int(fixed_aB),
            "T_s": fixed_T_s,
            "Bp_kTm": fixed_B_kTm,
            "delta_g_2pi": float(MODEL.delta_g_period_2pi(fixed_T_s, fixed_B_kTm)),
            "adaptive_phase": bool(FIXED_BASELINE_ADAPTIVE_PHASE),
        }

        print("\n=== Fixed-(T,B') baseline action ===")
        print(json.dumps(fixed_action_summary, indent=2))

        print("\n=== Fixed-(T,B') baseline evaluation ===")
        fixed_baseline_metrics, fixed_baseline_trace, fixed_catastrophics = evaluate_fixed_action_baseline(
            env_ctor=build_env,
            controller=controller,
            fixed_aT=fixed_aT,
            fixed_aB=fixed_aB,
            episodes=FIXED_BASELINE_EPISODES,
            adaptive_phase=FIXED_BASELINE_ADAPTIVE_PHASE,
            fixed_phase_rad=None,
            log_every=FIXED_BASELINE_LOG_EVERY,
            return_first_trace=SAVE_FIRST_TRACE,
            catastrophic_abs_err_threshold=CATASTROPHIC_ABS_ERR_THRESHOLD,
            catastrophic_log_top_k=CATASTROPHIC_LOG_TOP_K,
            save_catastrophic_records=SAVE_CATASTROPHIC_RECORDS,
        )
        print(json.dumps(fixed_baseline_metrics, indent=2))

    # ---------------- save outputs ----------------

    metrics_summary = {
        "adaptive_first_action": first_action,
        "adaptive_metrics": adaptive_metrics,
        "adaptive_catastrophic_count": int(len(adaptive_catastrophics)),
        "fixed_action_summary": fixed_action_summary,
        "fixed_baseline_metrics": fixed_baseline_metrics,
        "fixed_catastrophic_count": int(len(fixed_catastrophics)) if fixed_baseline_metrics is not None else None,
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
        "adaptive_eval_episodes": ADAPTIVE_EVAL_EPISODES,
        "fixed_baseline_episodes": FIXED_BASELINE_EPISODES,
        "run_fixed_baseline": RUN_FIXED_BASELINE,
        "fixed_baseline_adaptive_phase": FIXED_BASELINE_ADAPTIVE_PHASE,
    }

    if SAVE_CATASTROPHIC_RECORDS:
        save_json(
            os.path.join(OUTPUT_DIR, "adaptive_catastrophic_records.json"),
            {"records": adaptive_catastrophics},
        )
        if RUN_FIXED_BASELINE:
            save_json(
                os.path.join(OUTPUT_DIR, "fixed_catastrophic_records.json"),
                {"records": fixed_catastrophics},
            )
    save_json(os.path.join(OUTPUT_DIR, "metrics_summary.json"), metrics_summary)
    save_json(os.path.join(OUTPUT_DIR, "run_config.json"), run_config)

    if adaptive_trace is not None:
        save_json(os.path.join(OUTPUT_DIR, "adaptive_first_episode_trace.json"), {"trace": adaptive_trace})

    if fixed_baseline_trace is not None:
        save_json(os.path.join(OUTPUT_DIR, "fixed_first_episode_trace.json"), {"trace": fixed_baseline_trace})

    print("\nSaved files:")
    print(f"  {os.path.join(OUTPUT_DIR, 'metrics_summary.json')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'run_config.json')}")
    if SAVE_CATASTROPHIC_RECORDS:
        print(f"  {os.path.join(OUTPUT_DIR, 'adaptive_catastrophic_records.json')}")
        if RUN_FIXED_BASELINE:
            print(f"  {os.path.join(OUTPUT_DIR, 'fixed_catastrophic_records.json')}")
    if adaptive_trace is not None:
        print(f"  {os.path.join(OUTPUT_DIR, 'adaptive_first_episode_trace.json')}")
    if fixed_baseline_trace is not None:
        print(f"  {os.path.join(OUTPUT_DIR, 'fixed_first_episode_trace.json')}")


if __name__ == "__main__":
    main()