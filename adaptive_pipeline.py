from __future__ import annotations

from typing import Callable, Dict

import json
import numpy as np

from environment import (
    AdaptiveBayesController,
    GravimeterEnv,
    JointGravityEpsBelief,
    NoiseConfig,
    PriorConfig,
)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sample_global_true_eps(noise: NoiseConfig, seed: int) -> float:
    b = float(max(noise.mfg_rel_noise_bound, 0.0))
    if b <= 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-b, +b))


def _eps_stats_from_logw(eps_grid: np.ndarray, logw_eps: np.ndarray) -> dict:
    m = float(np.max(logw_eps))
    w = np.exp(logw_eps - m)
    w = w / (np.sum(w) + 1e-300)

    mu = float(np.sum(w * eps_grid))
    ex2 = float(np.sum(w * (eps_grid ** 2)))
    std = float(np.sqrt(max(ex2 - mu * mu, 0.0)))
    eps_map = float(eps_grid[int(np.argmax(w))])

    cdf = np.cumsum(w)
    q05 = float(np.interp(0.05, cdf, eps_grid))
    q50 = float(np.interp(0.50, cdf, eps_grid))
    q95 = float(np.interp(0.95, cdf, eps_grid))

    return {
        "global_eps_mean": mu,
        "global_eps_std": std,
        "global_eps_map": eps_map,
        "global_eps_q05": q05,
        "global_eps_q50": q50,
        "global_eps_q95": q95,
    }


def _posterior_diagnostics(env: GravimeterEnv, top_k: int = 5) -> dict:
    belief = env.belief

    g_q05, g_q95 = belief.credible_interval_g(mass=0.90)
    g_q25, g_q75 = belief.credible_interval_g(mass=0.50)
    p1, p2, gap = belief.peak_stats_g()
    peaks = belief.top_peaks_g(k=top_k)

    eps_q05, eps_q95 = belief.credible_interval_eps(mass=0.90)

    return {
        "g_q05": float(g_q05),
        "g_q25": float(g_q25),
        "g_q50": float(belief.median_g()),
        "g_q75": float(g_q75),
        "g_q95": float(g_q95),
        "g_halfwidth90": float(0.5 * (g_q95 - g_q05)),
        "g_halfwidth50": float(0.5 * (g_q75 - g_q25)),
        "peak1_prob": float(p1),
        "peak2_prob": float(p2),
        "peak_gap": float(gap),
        "top_peaks_g": [{"g": float(g), "p": float(p)} for g, p in peaks],
        "eps_q05": float(eps_q05),
        "eps_q50": float(belief.median_eps()),
        "eps_q95": float(eps_q95),
    }


def summarize_first_action(
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
) -> dict:
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


def choose_reference_fixed_action(
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
) -> tuple[int, int]:
    belief = JointGravityEpsBelief(
        prior=prior,
        noise=noise,
        n_g_grid=n_g_grid,
        n_eps_grid=n_eps_grid,
    )
    belief.reset_factorized()
    aT, aB, _phi, _A, _score = controller.plan_action(belief)
    return int(aT), int(aB)


def run_adaptive_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    logw_eps_global: np.ndarray,
    store_trace: bool = False,
) -> tuple[dict, list[dict], np.ndarray]:
    env.reset(logw_eps_global=logw_eps_global)
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
                    "true_eps": float(info["true_eps"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_mu_eps": float(info["post_mu_eps"]),
                    "post_std_eps": float(info["post_std_eps"]),
                    "post_map_eps": float(info["post_map_eps"]),
                    "post_entropy_g": float(info["post_entropy_g"]),
                    "post_entropy_joint": float(info["post_entropy_joint"]),
                    "k_g_nom": float(info["k_g_nom"]),
                    "dg2pi_nom": float(info["dg2pi_nom"]),
                }
            )

    assert last_info is not None
    new_logw_eps_global = env.belief.logw_eps_marginal()
    return last_info, trace, new_logw_eps_global


def run_fixed_action_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    fixed_aT: int,
    fixed_aB: int,
    logw_eps_global: np.ndarray,
    adaptive_phase: bool = True,
    fixed_phase_rad: float | None = None,
    store_trace: bool = False,
) -> tuple[dict, list[dict], np.ndarray]:
    env.reset(logw_eps_global=logw_eps_global)
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
                    "true_eps": float(info["true_eps"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_mu_eps": float(info["post_mu_eps"]),
                    "post_std_eps": float(info["post_std_eps"]),
                    "post_map_eps": float(info["post_map_eps"]),
                    "post_entropy_g": float(info["post_entropy_g"]),
                    "post_entropy_joint": float(info["post_entropy_joint"]),
                    "k_g_nom": float(info["k_g_nom"]),
                    "dg2pi_nom": float(info["dg2pi_nom"]),
                }
            )

    assert last_info is not None
    new_logw_eps_global = env.belief.logw_eps_marginal()
    return last_info, trace, new_logw_eps_global


def _metrics_from_infos(infos: list[dict]) -> Dict[str, float]:
    err_mean = np.asarray([x["g_err_mean"] for x in infos], dtype=np.float64)
    err_median = np.asarray([x["g_err_median"] for x in infos], dtype=np.float64)
    err_map = np.asarray([x["g_err_map"] for x in infos], dtype=np.float64)

    eps_err_mean = np.asarray([x["eps_err_mean"] for x in infos], dtype=np.float64)
    eps_err_map = np.asarray([x["eps_err_map"] for x in infos], dtype=np.float64)

    std_g = np.asarray([x["post_std_g"] for x in infos], dtype=np.float64)
    std_eps = np.asarray([x["post_std_eps"] for x in infos], dtype=np.float64)

    ent_g = np.asarray([x["post_entropy_g"] for x in infos], dtype=np.float64)
    ent_joint = np.asarray([x["post_entropy_joint"] for x in infos], dtype=np.float64)
    cyc = np.asarray([x["cycle_time_s"] for x in infos], dtype=np.float64)

    return {
        "rmse_g_mean": float(np.sqrt(np.mean(err_mean ** 2))),
        "rmse_g_median": float(np.sqrt(np.mean(err_median ** 2))),
        "rmse_g_map": float(np.sqrt(np.mean(err_map ** 2))),
        "mean_post_std_g": float(np.mean(std_g)),
        "mean_post_std_eps": float(np.mean(std_eps)),
        "mean_entropy_g": float(np.mean(ent_g)),
        "mean_entropy_joint": float(np.mean(ent_joint)),
        "rmse_eps_mean": float(np.sqrt(np.mean(eps_err_mean ** 2))),
        "rmse_eps_map": float(np.sqrt(np.mean(eps_err_map ** 2))),
        "mean_cycle_time_s_last_shot": float(np.mean(cyc)),
    }


def evaluate_adaptive_controller(
    env_ctor: Callable[[int], GravimeterEnv],
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
    episodes: int = 128,
    log_every: int = 0,
    return_first_trace: bool = False,
    catastrophic_abs_err_threshold: float = 5e-4,
    catastrophic_log_top_k: int = 5,
    save_catastrophic_records: bool = True,
    global_true_eps: float | None = None,
    global_eps_seed: int = 0,
) -> tuple[Dict[str, float], list[dict] | None, list[dict]]:
    infos: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []

    tmp_belief = JointGravityEpsBelief(
        prior=prior,
        noise=noise,
        n_g_grid=n_g_grid,
        n_eps_grid=n_eps_grid,
    )
    logw_eps_global = tmp_belief.uniform_logw_eps()

    if global_true_eps is None:
        global_true_eps = _sample_global_true_eps(noise, seed=global_eps_seed)

    for ep in range(int(episodes)):
        env = env_ctor(10_000 + ep)
        env.global_true_eps = float(global_true_eps)

        store_trace = bool(return_first_trace and ep == 0)
        final_info, trace, logw_eps_global = run_adaptive_episode(
            env=env,
            controller=controller,
            logw_eps_global=logw_eps_global,
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
                "true_eps": float(final_info["true_eps"]),
                "post_mu_g": float(final_info["post_mu_g"]),
                "post_median_g": float(final_info["post_median_g"]),
                "post_map_g": float(final_info["post_map_g"]),
                "post_std_g": float(final_info["post_std_g"]),
                "post_mu_eps": float(final_info["post_mu_eps"]),
                "post_median_eps": float(final_info["post_median_eps"]),
                "post_map_eps": float(final_info["post_map_eps"]),
                "post_std_eps": float(final_info["post_std_eps"]),
                "post_entropy_g": float(final_info["post_entropy_g"]),
                "post_entropy_joint": float(final_info["post_entropy_joint"]),
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

    metrics = _metrics_from_infos(infos)
    metrics["global_true_eps"] = float(global_true_eps)
    metrics.update(_eps_stats_from_logw(tmp_belief.eps_grid, logw_eps_global))
    return metrics, first_trace, catastrophic_records


def evaluate_fixed_action_baseline(
    env_ctor: Callable[[int], GravimeterEnv],
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
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
    global_true_eps: float | None = None,
    global_eps_seed: int = 0,
) -> tuple[Dict[str, float], list[dict] | None, list[dict]]:
    infos: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []

    tmp_belief = JointGravityEpsBelief(
        prior=prior,
        noise=noise,
        n_g_grid=n_g_grid,
        n_eps_grid=n_eps_grid,
    )
    logw_eps_global = tmp_belief.uniform_logw_eps()

    if global_true_eps is None:
        global_true_eps = _sample_global_true_eps(noise, seed=global_eps_seed)

    for ep in range(int(episodes)):
        env = env_ctor(20_000 + ep)
        env.global_true_eps = float(global_true_eps)

        store_trace = bool(return_first_trace and ep == 0)
        final_info, trace, logw_eps_global = run_fixed_action_episode(
            env=env,
            controller=controller,
            fixed_aT=fixed_aT,
            fixed_aB=fixed_aB,
            logw_eps_global=logw_eps_global,
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
                "true_eps": float(final_info["true_eps"]),
                "post_mu_g": float(final_info["post_mu_g"]),
                "post_median_g": float(final_info["post_median_g"]),
                "post_map_g": float(final_info["post_map_g"]),
                "post_std_g": float(final_info["post_std_g"]),
                "post_mu_eps": float(final_info["post_mu_eps"]),
                "post_median_eps": float(final_info["post_median_eps"]),
                "post_map_eps": float(final_info["post_map_eps"]),
                "post_std_eps": float(final_info["post_std_eps"]),
                "post_entropy_g": float(final_info["post_entropy_g"]),
                "post_entropy_joint": float(final_info["post_entropy_joint"]),
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

    metrics = _metrics_from_infos(infos)
    metrics["global_true_eps"] = float(global_true_eps)
    metrics.update(_eps_stats_from_logw(tmp_belief.eps_grid, logw_eps_global))
    return metrics, first_trace, catastrophic_records