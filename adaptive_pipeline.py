#adaptive_pipeline.py
from __future__ import annotations

from typing import Callable, Dict, List

import json
import numpy as np

from environment import (
    AdaptiveBayesController,
    GravimeterEnv,
    GravityGridBelief,
    PriorConfig,
)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _posterior_diagnostics(env: GravimeterEnv, top_k: int = 5) -> dict:
    belief = env.belief
    q05, q95 = belief.credible_interval(mass=0.90)
    q25, q75 = belief.credible_interval(mass=0.50)
    p1, p2, gap = belief.peak_stats()
    peaks = belief.top_peaks(k=top_k)

    return {
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(belief.median()),
        "q75": float(q75),
        "q95": float(q95),
        "halfwidth90": float(0.5 * (q95 - q05)),
        "halfwidth50": float(0.5 * (q75 - q25)),
        "peak1_prob": float(p1),
        "peak2_prob": float(p2),
        "peak_gap": float(gap),
        "top_peaks": [{"g": float(g), "p": float(p)} for g, p in peaks],
    }

def summarize_first_action(
    prior: PriorConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
) -> dict:
    belief = GravityGridBelief(prior, n_grid=n_g_grid)
    belief.reset_uniform()
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


def run_adaptive_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    store_trace: bool = False,
) -> tuple[dict, list[dict]]:
    env.reset()
    trace: list[dict] = []
    last_info: dict | None = None

    done = False
    while not done:
        aT, aB, phi, A_plan, score = controller.plan_action(env.belief)
        done, info = env.step(aT, aB, phi)
        last_info = info

        if store_trace:
            trace.append(
                {
                    "step": int(info["t"]),
                    "aT": int(aT),
                    "aB": int(aB),
                    "T_s": float(info["T_s"]),
                    "Bp_nom_kTm": float(info["Bp_nom_kTm"]),
                    "Bp_true_kTm": float(info["Bp_true_kTm"]),
                    "mw_phase_rad": float(phi),
                    "A_plan": float(A_plan),
                    "A_true": float(info["A_true"]),
                    "score": float(score),
                    "outcome_plus": int(info["outcome_plus"]),
                    "true_g": float(info["true_g"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_entropy_g": float(info["post_entropy_g"]),
                    "k_g_nom": float(info["k_g_nom"]),
                    "dg2pi_nom": float(info["dg2pi_nom"]),
                }
            )

    assert last_info is not None
    return last_info, trace


def choose_reference_fixed_action(
    prior: PriorConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
) -> tuple[int, int]:
    """
    Pick one fixed (T, B') action by applying the planner once on the uniform prior.
    This is a useful reference baseline, not the globally optimal static policy.
    """
    belief = GravityGridBelief(prior, n_grid=n_g_grid)
    belief.reset_uniform()
    aT, aB, _phi, _A, _score = controller.plan_action(belief)
    return int(aT), int(aB)


def run_fixed_action_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    fixed_aT: int,
    fixed_aB: int,
    adaptive_phase: bool = True,
    fixed_phase_rad: float | None = None,
    store_trace: bool = False,
) -> tuple[dict, list[dict]]:
    env.reset()
    trace: list[dict] = []
    last_info: dict | None = None

    if (not adaptive_phase) and (fixed_phase_rad is None):
        raise ValueError("fixed_phase_rad must be provided when adaptive_phase=False")

    done = False
    while not done:
        T_s = float(env.grid.T_vals_s[int(fixed_aT)])
        B_kTm = float(env.grid.Bp_vals_kTm[int(fixed_aB)])

        if adaptive_phase:
            phi = controller.choose_phase(env.belief, T_s, B_kTm)
        else:
            phi = float(fixed_phase_rad)

        done, info = env.step(fixed_aT, fixed_aB, phi)
        last_info = info

        if store_trace:
            trace.append(
                {
                    "step": int(info["t"]),
                    "aT": int(fixed_aT),
                    "aB": int(fixed_aB),
                    "T_s": float(info["T_s"]),
                    "Bp_nom_kTm": float(info["Bp_nom_kTm"]),
                    "Bp_true_kTm": float(info["Bp_true_kTm"]),
                    "mw_phase_rad": float(phi),
                    "A_true": float(info["A_true"]),
                    "outcome_plus": int(info["outcome_plus"]),
                    "true_g": float(info["true_g"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_entropy_g": float(info["post_entropy_g"]),
                    "k_g_nom": float(info["k_g_nom"]),
                    "dg2pi_nom": float(info["dg2pi_nom"]),
                }
            )

    assert last_info is not None
    return last_info, trace


def _metrics_from_infos(infos: list[dict]) -> Dict[str, float]:
    err_mean = np.asarray([x["g_err_mean"] for x in infos], dtype=np.float64)
    err_median = np.asarray([x["g_err_median"] for x in infos], dtype=np.float64)
    err_map = np.asarray([x["g_err_map"] for x in infos], dtype=np.float64)
    stds = np.asarray([x["post_std_g"] for x in infos], dtype=np.float64)
    ent = np.asarray([x["post_entropy_g"] for x in infos], dtype=np.float64)
    cyc = np.asarray([x["cycle_time_s"] for x in infos], dtype=np.float64)

    return {
        "rmse_g_mean": float(np.sqrt(np.mean(err_mean ** 2))),
        "rmse_g_median": float(np.sqrt(np.mean(err_median ** 2))),
        "rmse_g_map": float(np.sqrt(np.mean(err_map ** 2))),
        "mean_post_std_g": float(np.mean(stds)),
        "mean_entropy_g": float(np.mean(ent)),
        "mean_cycle_time_s_last_shot": float(np.mean(cyc)),
    }


def evaluate_adaptive_controller(
    env_ctor: Callable[[int], GravimeterEnv],
    controller: AdaptiveBayesController,
    episodes: int = 128,
    log_every: int = 0,
    return_first_trace: bool = False,
    catastrophic_abs_err_threshold: float = 5e-4,
    catastrophic_log_top_k: int = 5,
    save_catastrophic_records: bool = True,
) -> tuple[Dict[str, float], list[dict] | None, list[dict]]:
    infos: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []

    for ep in range(int(episodes)):
        env = env_ctor(10_000 + ep)
        store_trace = bool(return_first_trace and ep == 0)
        final_info, trace = run_adaptive_episode(env, controller, store_trace=store_trace)
        infos.append(final_info)

        if store_trace:
            first_trace = trace

        # ---- catastrophic diagnostics (use median as primary robust estimator) ----
        err_med = abs(float(final_info["g_err_median"]))
        err_mean = abs(float(final_info["g_err_mean"]))
        err_map = abs(float(final_info["g_err_map"]))

        if err_med > float(catastrophic_abs_err_threshold):
            diag = _posterior_diagnostics(env, top_k=catastrophic_log_top_k)
            record = {
                "episode": int(ep),
                "true_g": float(final_info["true_g"]),
                "post_mu_g": float(final_info["post_mu_g"]),
                "post_median_g": float(final_info["post_median_g"]),
                "post_map_g": float(final_info["post_map_g"]),
                "post_std_g": float(final_info["post_std_g"]),
                "post_entropy_g": float(final_info["post_entropy_g"]),
                "abs_err_mean": float(err_mean),
                "abs_err_median": float(err_med),
                "abs_err_map": float(err_map),
                "final_step": int(final_info["t"]),
                "last_T_s": float(final_info["T_s"]),
                "last_Bp_nom_kTm": float(final_info["Bp_nom_kTm"]),
                "last_k_g_nom": float(final_info["k_g_nom"]),
                "last_dg2pi_nom": float(final_info["dg2pi_nom"]),
                **diag,
            }
            if save_catastrophic_records:
                catastrophic_records.append(record)

        if log_every > 0 and ((ep + 1) % log_every == 0):
            cur = _metrics_from_infos(infos)
            n_cat = sum(abs(float(x["g_err_median"])) > float(catastrophic_abs_err_threshold) for x in infos)
            print(
                f"[Adaptive Eval] episode {ep + 1}/{episodes} | "
                f"rmse_mean={cur['rmse_g_mean']:.6e} | "
                f"rmse_map={cur['rmse_g_map']:.6e} | "
                f"mean_std={cur['mean_post_std_g']:.6e} | "
                f"cat_med={n_cat}"
            )

    return _metrics_from_infos(infos), first_trace, catastrophic_records


def evaluate_fixed_action_baseline(
    env_ctor: Callable[[int], GravimeterEnv],
    controller: AdaptiveBayesController,
    fixed_aT: int,
    fixed_aB: int,
    episodes: int = 128,
    adaptive_phase: bool = True,
    fixed_phase_rad: float | None = None,
    log_every: int = 0,
    return_first_trace: bool = False,
    catastrophic_abs_err_threshold: float = 5e-4,
    catastrophic_log_top_k: int = 5,
    save_catastrophic_records: bool = True,
) -> tuple[Dict[str, float], list[dict] | None, list[dict]]:
    infos: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []

    for ep in range(int(episodes)):
        env = env_ctor(20_000 + ep)
        store_trace = bool(return_first_trace and ep == 0)
        final_info, trace = run_fixed_action_episode(
            env=env,
            controller=controller,
            fixed_aT=fixed_aT,
            fixed_aB=fixed_aB,
            adaptive_phase=adaptive_phase,
            fixed_phase_rad=fixed_phase_rad,
            store_trace=store_trace,
        )
        infos.append(final_info)

        if store_trace:
            first_trace = trace

        err_med = abs(float(final_info["g_err_median"]))
        err_mean = abs(float(final_info["g_err_mean"]))
        err_map = abs(float(final_info["g_err_map"]))

        if err_med > float(catastrophic_abs_err_threshold):
            diag = _posterior_diagnostics(env, top_k=catastrophic_log_top_k)
            record = {
                "episode": int(ep),
                "true_g": float(final_info["true_g"]),
                "post_mu_g": float(final_info["post_mu_g"]),
                "post_median_g": float(final_info["post_median_g"]),
                "post_map_g": float(final_info["post_map_g"]),
                "post_std_g": float(final_info["post_std_g"]),
                "post_entropy_g": float(final_info["post_entropy_g"]),
                "abs_err_mean": float(err_mean),
                "abs_err_median": float(err_med),
                "abs_err_map": float(err_map),
                "final_step": int(final_info["t"]),
                "last_T_s": float(final_info["T_s"]),
                "last_Bp_nom_kTm": float(final_info["Bp_nom_kTm"]),
                "last_k_g_nom": float(final_info["k_g_nom"]),
                "last_dg2pi_nom": float(final_info["dg2pi_nom"]),
                **diag,
            }
            if save_catastrophic_records:
                catastrophic_records.append(record)

        if log_every > 0 and ((ep + 1) % log_every == 0):
            cur = _metrics_from_infos(infos)
            n_cat = sum(abs(float(x["g_err_median"])) > float(catastrophic_abs_err_threshold) for x in infos)
            print(
                f"[Fixed Eval] episode {ep + 1}/{episodes} | "
                f"rmse_mean={cur['rmse_g_mean']:.6e} | "
                f"rmse_map={cur['rmse_g_map']:.6e} | "
                f"mean_std={cur['mean_post_std_g']:.6e} | "
                f"cat_med={n_cat}"
            )

    return _metrics_from_infos(infos), first_trace, catastrophic_records