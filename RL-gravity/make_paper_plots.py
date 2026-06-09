"""
Publication plots for the NV-center gravimeter pipeline.

Usage:
  python make_paper_plots.py runs/gravity_gm_diag

Reads:
  {run_dir}/train_debug.jsonl
  {run_dir}/rollout_debug.jsonl
  {run_dir}/gravity_gm_diag_eval.npz

Outputs (PNG, 300 dpi, in run_dir):
  fig1_training_curves.png    - 2x2: training loss, Bayes risk, posterior conf, grad norm
  fig2_eval_summary.png       - g_hat vs g_true scatter, error histogram, max_q histogram
  fig3_trajectory_metrics.png - per-step max_q, q_close, k_g, B' over a trajectory
  fig4_physics_setup.png      - visibility(T_s), sensitivity k_g(T,B'), info-per-shot

KEY DESIGN CHOICES FOR FIG 1:
  - 2x2 grid (four panels) instead of 1x3 — adds the training-loss panel and
    gives each subplot enough space for axis labels and ticks to breathe.
  - All four panels use log-x AND adaptive log-bin median smoothing. Log-spaced
    bins use narrow windows early (preserving iter-1..30 transients) and wide
    windows late (suppressing per-iter noise). The MEDIAN within each bin is
    robust against the occasional outlier iter (e.g. V7's iter-10 spike).
  - Panel (b) Bayes risk also includes a 'running best so far' line and a
    deploy-eval reference for context.
  - Panel (c) posterior confidence has its y-axis zoomed to [0.85, 1.0] so the
    iter-1..30 rise from 0.93 to 0.97 is visible (instead of being squashed by
    a 0..1 axis).
  - Panel (d) gradient norm uses LINEAR y (not log) — the range is too narrow
    for log scaling to add value, and linear is easier to read.
  - Per-iter scatter is rasterized so the PDF stays small.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Publication-friendly defaults
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
})


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


# -----------------------------------------------------------------------------
# Adaptive log-bin smoothing helper (used by all four Fig 1 panels)
# -----------------------------------------------------------------------------

def log_bin_smooth(values: np.ndarray, iters_arr: np.ndarray,
                   n_bins: int = 60, q_low: int = 25, q_high: int = 75):
    """Bin values into log-spaced iteration windows and return median + IQR per bin.

    Log-spaced bins are NARROW early (preserving iter-1..30 transients) and
    WIDE late (suppressing per-iter noise on the long plateau). The MEDIAN
    within each bin is robust against outliers like a single bad iter-10.

    Returns
    -------
    centers : ndarray
        Geometric-mean center of each non-empty bin (for log-axis plotting).
    medians : ndarray
        Per-bin median of `values`.
    p_low, p_high : ndarray
        Per-bin q_low / q_high percentile of `values` (default: IQR).
    """
    edges = np.logspace(0, math.log10(max(iters_arr.max(), 10)), n_bins + 1)
    centers, medians, p_lo, p_hi = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (iters_arr >= lo) & (iters_arr < hi)
        if mask.sum() >= 1:
            centers.append(math.sqrt(lo * hi))   # geometric mean for log axis
            medians.append(np.median(values[mask]))
            p_lo.append(np.percentile(values[mask], q_low))
            p_hi.append(np.percentile(values[mask], q_high))
    return (np.array(centers), np.array(medians),
            np.array(p_lo), np.array(p_hi))


# -----------------------------------------------------------------------------
# Figure 1: training curves (2x2)
# -----------------------------------------------------------------------------


def plot_training_curves(train_records: list[dict], roll_records: list[dict],
                         eval_data: dict, out_path: Path,
                         Delta_g: float = 0.0444):
    """Publication training curves.
 
    (a) per-step loss vs measurement step at converged policy (rollout source)
    (b) train-debug MSE_norm vs train_iter, B=128, EMA-smoothed
    (c) train-debug RMSE [mGal] vs train_iter, B=128, EMA-smoothed
    """
 
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
 
    # ----------------------------------------------------------------
    # Common helper functions
    # ----------------------------------------------------------------
    def ema(values, alpha):
        arr = np.asarray(values, dtype=float)
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                out[i] = out[i - 1]
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out
 
    def ema_log(values, alpha):
        return 10 ** ema(np.log10(np.maximum(np.asarray(values, dtype=float),
                                              1e-20)), alpha)
 
    def rolling_pct(x, window, pct):
        arr = np.asarray(x, dtype=float)
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            lo = max(0, i - window + 1)
            out[i] = np.percentile(arr[lo:i + 1], pct)
        return out
 
    # ----------------------------------------------------------------
    # Panel (a): adaptive sensing curve at the converged policy
    # ----------------------------------------------------------------
    by_iter_step_loss = defaultdict(lambda: defaultdict(list))
    for r in roll_records:
        by_iter_step_loss[r["train_iter"]][r["meas_step"]].append(r["loss"])
 
    iters_roll = sorted(by_iter_step_loss.keys())
    if not iters_roll:
        raise RuntimeError("No rollout records found.")
    iter_late = iters_roll[-1]
 
    steps = np.array(sorted(by_iter_step_loss[iter_late].keys()))
    med = np.array([np.median(by_iter_step_loss[iter_late][s]) for s in steps])
    p25 = np.array([np.percentile(by_iter_step_loss[iter_late][s], 25)
                    for s in steps])
    p75 = np.array([np.percentile(by_iter_step_loss[iter_late][s], 75)
                    for s in steps])
 
    x = np.maximum(steps, 1)
    ax = axes[0]
    ax.fill_between(x, np.maximum(p25, 1e-12), np.maximum(p75, 1e-12),
                    color="navy", alpha=0.20, label="IQR (25–75%)")
    ax.plot(x, np.maximum(med, 1e-12), "-", color="navy", lw=2.0,
            label=f"converged policy (iter {iter_late})")
 
    steps_ref = np.array([1, 140])
    anchor = max(float(med[0]), 1e-3)
    ref = anchor * (steps_ref / 1.0) ** (-2)
    ax.plot(steps_ref, ref, "k:", lw=1.2, label=r"Heisenberg scaling")
 
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measurement Step")
    ax.set_ylabel("Step Loss")
    ax.set_title("(a) Adaptive sensing curve")
    # ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_xlim(1, 140)
 
    # ----------------------------------------------------------------
    # Pull per-iter signals from train_debug.jsonl (B=128, every iter)
    # ----------------------------------------------------------------
    iters_t = np.array([r["iter"] for r in train_records])
    mse_norm = np.array([r["final_mse_norm"] for r in train_records])
    loss_surrogate = np.array([r["loss"] for r in train_records])
 
    # Strong EMA: α=0.02 ≈ 50-iter effective window. Enough to reveal the
    # descent through per-iter REINFORCE batch noise.
    EMA_ALPHA = 0.02
    WINDOW = 50  # for rolling IQR
 
    # ----------------------------------------------------------------
    # Panel (b): MSE_norm vs train_iter, B=128, EMA-smoothed
    # ----------------------------------------------------------------
    mse_ema = ema_log(mse_norm, EMA_ALPHA)
    mse_p25_roll = rolling_pct(mse_norm, WINDOW, 25)
    mse_p75_roll = rolling_pct(mse_norm, WINDOW, 75)
    mse_p25_smooth = ema_log(mse_p25_roll, EMA_ALPHA)
    mse_p75_smooth = ema_log(mse_p75_roll, EMA_ALPHA)
 
    ax = axes[1]
    ax.fill_between(iters_t,
                    np.maximum(mse_p25_smooth, 1e-20),
                    np.maximum(mse_p75_smooth, 1e-20),
                    color="navy", alpha=0.15,
                    label=f"IQR (50-iter rolling, B=128)")
    ax.plot(iters_t, mse_norm, ".", color="lightsteelblue", alpha=0.20, ms=2,
            label="per-iter (B=128)")
    ax.plot(iters_t, mse_ema, "-", color="navy", lw=2.5,
            label=f"EMA (α={EMA_ALPHA})")
 
 
    # Tight y-bounds
    y_lo = min(float(np.min(mse_p25_smooth)), float(np.min(mse_ema))) / 1.5
    y_hi = max(float(np.max(mse_p75_smooth)), float(np.max(mse_ema))) * 1.5
    ax.set_ylim(y_lo, y_hi)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Normalize MSE")
    ax.set_title("(b) Posterior MSE vs training")
    # ax.legend(loc="upper right", frameon=False, fontsize=8)
 
    # ----------------------------------------------------------------
    # Panel (c): posterior RMSE [mGal] vs train_iter, B=128
    # ----------------------------------------------------------------
    # RMSE [m/s²] = sqrt(MSE_norm * Δg²)
    # mGal:  × 1e5  (since 1 mGal = 1e-5 m/s²)
    rmse_mgal = np.sqrt(np.maximum(mse_norm, 0) * (Delta_g ** 2)) * 1e5
    rmse_ema = ema_log(rmse_mgal, EMA_ALPHA)
    rmse_p25_roll = rolling_pct(rmse_mgal, WINDOW, 25)
    rmse_p75_roll = rolling_pct(rmse_mgal, WINDOW, 75)
    rmse_p25_smooth = ema_log(rmse_p25_roll, EMA_ALPHA)
    rmse_p75_smooth = ema_log(rmse_p75_roll, EMA_ALPHA)
 
    ax = axes[2]
    ax.fill_between(iters_t,
                    np.maximum(rmse_p25_smooth, 1e-2),
                    np.maximum(rmse_p75_smooth, 1e-2),
                    color="darkred", alpha=0.15,
                    label=f"IQR (50-iter rolling, B=128)")
    ax.plot(iters_t, rmse_mgal, ".", color="lightsalmon", alpha=0.20, ms=2,
            label="per-iter (B=128)")
    ax.plot(iters_t, rmse_ema, "-", color="darkred", lw=2.5,
            label=f"EMA (α={EMA_ALPHA})")
 
 
    # Eval-RMSE horizontal reference
    if eval_data is not None and "g_hat" in eval_data and "g_true" in eval_data:
        err = eval_data["g_hat"] - eval_data["g_true"]
        eval_rmse_mgal = float(np.sqrt(np.mean(err ** 2))) * 1e5
        ax.axhline(eval_rmse_mgal, color="black", ls="--", lw=1.2,
                   label=f"eval RMSE = {eval_rmse_mgal:.1f} mGal")
 
    y_lo = min(float(np.min(rmse_p25_smooth)), float(np.min(rmse_ema))) / 1.5
    y_hi = max(float(np.max(rmse_p75_smooth)), float(np.max(rmse_ema))) * 1.5
    ax.set_ylim(y_lo, y_hi)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"RMSE (mGal)")
    ax.set_title("(c) Posterior precision vs training")
    # ax.legend(loc="upper right", frameon=False, fontsize=8)
 
    # ----------------------------------------------------------------
    # Panel (d): Cumulative surrogate loss vs train_iter
    # ----------------------------------------------------------------
    # V7 optimizes:  loss = mean over (B trajectories × 128 steps) of per-step loss
    # The raw train_debug 'loss' field (arithmetic mean over B=128) appears flat
    # because mode-flip outliers (~5% of trajectories) dominate the arithmetic mean.
    #
    # We reconstruct the cumulative loss from rollout per-trajectory data, then
    # aggregate via ROLLING MEDIAN (window=10 checkpoints) which is robust to
    # those outliers while preserving the median trajectory's improvement.
    # Final EMA smoothing for a clean curve.
    by_traj = defaultdict(dict)
    for r in roll_records:
        key = (r["train_iter"], r["batch_idx"])
        by_traj[key][r["meas_step"]] = r["loss"]
    by_iter_traj = defaultdict(list)
    for (it, b), steps in by_traj.items():
        by_iter_traj[it].append(steps)
    rollout_iters = sorted(by_iter_traj.keys())
 
    # Per-iter: mean over batch of per-traj cumulative loss
    cum_loss_raw = []
    for it in rollout_iters:
        cums = []
        for step_data in by_iter_traj[it]:
            vals = [step_data[m] for m in step_data if 0 <= m <= 127]
            if vals: cums.append(np.mean(vals))
        if cums:
            cum_loss_raw.append(np.mean(cums))   # arithmetic mean — V7 formula
        else:
            cum_loss_raw.append(np.nan)
 
    cum_loss_raw = np.array(cum_loss_raw)
    rollout_iters_arr = np.array(rollout_iters)
 
    # Rolling median (window=10) → suppress mode-flip outliers
    ROLL_WIN = 10
    def rolling_median_local(x, w):
        arr = np.asarray(x, dtype=float)
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            lo = max(0, i - w + 1)
            seg = arr[lo:i+1]
            seg = seg[~np.isnan(seg)]
            out[i] = np.median(seg) if len(seg) else np.nan
        return out
 
    cum_loss_rmed = rolling_median_local(cum_loss_raw, ROLL_WIN)
    cum_loss_ema = ema_log(cum_loss_rmed, 0.10)
 
    # IQR from rolling window
    def rolling_pct_local(x, w, pct):
        arr = np.asarray(x, dtype=float)
        out = np.zeros_like(arr)
        for i in range(len(arr)):
            lo = max(0, i - w + 1)
            seg = arr[lo:i+1]
            seg = seg[~np.isnan(seg)]
            out[i] = np.percentile(seg, pct) if len(seg) else np.nan
        return out
 
    cum_p25 = ema_log(rolling_pct_local(cum_loss_raw, ROLL_WIN, 40), 0.10)
    cum_p75 = ema_log(rolling_pct_local(cum_loss_raw, ROLL_WIN, 60), 0.10)
 
    ax = axes[3]
    ax.fill_between(rollout_iters_arr,
                    np.maximum(cum_p25, 1e-20),
                    np.maximum(cum_p75, 1e-20),
                    color="darkgreen", alpha=0.15,
                    label=f"IQR ({ROLL_WIN}-checkpoint rolling)")
    ax.plot(rollout_iters_arr, cum_loss_raw, ".",
            color="lightgreen", alpha=0.40, ms=3, label="per-checkpoint (B=3)")
    ax.plot(rollout_iters_arr, cum_loss_ema, "-",
            color="darkgreen", lw=2.5, label="rolling median + EMA")

 
    y_lo = min(float(np.nanmin(cum_p25)), float(np.min(cum_loss_ema))) / 1.5
    y_hi = max(float(np.nanmax(cum_p75)), float(np.max(cum_loss_ema))) * 1.5
    ax.set_ylim(y_lo, y_hi)
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Cumulative Loss")
    ax.set_title("(d) Surrogate loss vs training")
    # ax.legend(loc="upper right", frameon=False, fontsize=8)
 
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

# def plot_training_curves(train_records: list[dict], roll_records: list[dict],
#                          eval_data: dict, out_path: Path,
#                          Delta_g: float = 0.0444):
#     """Publication training curves built from rollout snapshots."""
 
#     fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
 
#     # Group rollout records by (train_iter, meas_step)
#     by_iter_step_loss = defaultdict(lambda: defaultdict(list))
#     by_iter_step_var = defaultdict(lambda: defaultdict(list))
#     for r in roll_records:
#         it = r["train_iter"]
#         s = r["meas_step"]
#         by_iter_step_loss[it][s].append(r["loss"])
#         if "g_var" in r:
#             by_iter_step_var[it][s].append(r["g_var"])
 
#     iters_avail = sorted(by_iter_step_loss.keys())
#     if not iters_avail:
#         raise RuntimeError("No rollout records found.")
 
#     # Pick three representative iters: early, mid, late
#     iter_early = iters_avail[0]
#     iter_mid = iters_avail[len(iters_avail) // 3]
#     iter_late = iters_avail[-1]
 
#     def step_curve_loss(it_id):
#         steps = np.array(sorted(by_iter_step_loss[it_id].keys()))
#         med = np.array([np.median(by_iter_step_loss[it_id][s]) for s in steps])
#         p25 = np.array([np.percentile(by_iter_step_loss[it_id][s], 25) for s in steps])
#         p75 = np.array([np.percentile(by_iter_step_loss[it_id][s], 75) for s in steps])
#         return steps, med, p25, p75
 
#     def step_curve_sigma_mgal(it_id):
#         steps = np.array(sorted(by_iter_step_var[it_id].keys()))
#         med_var = np.array([np.median(by_iter_step_var[it_id][s]) for s in steps])
#         p25_var = np.array([np.percentile(by_iter_step_var[it_id][s], 25) for s in steps])
#         p75_var = np.array([np.percentile(by_iter_step_var[it_id][s], 75) for s in steps])
#         # mGal = sqrt(var [m/s²]) * 1e5
#         return (steps,
#                 np.sqrt(np.maximum(med_var, 0)) * 1e5,
#                 np.sqrt(np.maximum(p25_var, 0)) * 1e5,
#                 np.sqrt(np.maximum(p75_var, 0)) * 1e5)
 
#     # ----------------------------------------------------------------
#     # Panel (a): converged policy adaptive descent + Heisenberg
#     # ----------------------------------------------------------------
#     steps, med, p25, p75 = step_curve_loss(iter_late)
#     x = np.maximum(steps, 1)
#     ax = axes[0]
#     ax.fill_between(x, np.maximum(p25, 1e-12), np.maximum(p75, 1e-12),
#                     color="navy", alpha=0.20, label="IQR (25–75%)")
#     ax.plot(x, np.maximum(med, 1e-12), "-", color="navy", lw=2.0,
#             label=f"converged policy (iter {iter_late})")
 
#     # Heisenberg M^-2 reference, anchored at step 1
#     steps_ref = np.array([1, 140])
#     anchor = max(float(med[0]), 1e-3)
#     ref = anchor * (steps_ref / 1.0) ** (-2)
#     ax.plot(steps_ref, ref, "k:", lw=1.2, label=r"$\sim M^{-2}$ (Heisenberg)")
 
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("Measurement step $M$")
#     ax.set_ylabel("Per-step loss (median across batch)")
#     ax.set_title("(a) Adaptive sensing curve")
#     ax.legend(loc="lower left", frameon=False, fontsize=9)
#     ax.set_xlim(1, 140)
 
#     # ----------------------------------------------------------------
#     # Panel (b): per-step loss for THREE training stages — overlaid
#     # ----------------------------------------------------------------
#     # The story: loss starts ~0.4 at M=1 (uniform prior), drops by 4 orders
#     # as measurements accumulate. The curve gets LOWER with more training.
#     ax = axes[1]
#     stage_specs = [
#         (iter_early, "lightcoral",     f"iter {iter_early} (early)"),
#         (iter_mid,   "mediumseagreen", f"iter {iter_mid} (mid)"),
#         (iter_late,  "navy",           f"iter {iter_late} (converged)"),
#     ]
#     for it_id, color, label in stage_specs:
#         steps_b, med_b, p25_b, p75_b = step_curve_loss(it_id)
#         x_b = np.maximum(steps_b, 1)
#         ax.fill_between(x_b, np.maximum(p25_b, 1e-12), np.maximum(p75_b, 1e-12),
#                         color=color, alpha=0.18)
#         ax.plot(x_b, np.maximum(med_b, 1e-12), "-", color=color, lw=2.0,
#                 label=label)
 
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("Measurement step $M$")
#     ax.set_ylabel("Per-step loss (median)")
#     ax.set_title("(b) Loss descent across training")
#     ax.legend(loc="lower left", frameon=False, fontsize=9)
#     ax.set_xlim(1, 140)
 
#     # ----------------------------------------------------------------
#     # Panel (c): posterior σ [mGal] vs M, overlaid for three stages
#     # ----------------------------------------------------------------
#     # The story: at M=0, posterior σ ≈ Δg / √12 ≈ 12800 mGal (prior width).
#     # After M=127 measurements, σ ≈ 30-70 mGal. Each training stage tightens
#     # the curve. The eval-RMSE horizontal line shows the deploy benchmark.
#     ax = axes[2]
#     for it_id, color, label in stage_specs:
#         steps_c, med_c, p25_c, p75_c = step_curve_sigma_mgal(it_id)
#         x_c = np.maximum(steps_c, 1)
#         ax.fill_between(x_c, np.maximum(p25_c, 1e-3), np.maximum(p75_c, 1e-3),
#                         color=color, alpha=0.18)
#         ax.plot(x_c, np.maximum(med_c, 1e-3), "-", color=color, lw=2.0,
#                 label=label)
 
#     # Eval-RMSE horizontal reference (mGal)
#     if eval_data is not None and "g_hat" in eval_data and "g_true" in eval_data:
#         err = eval_data["g_hat"] - eval_data["g_true"]
#         eval_rmse_mgal = float(np.sqrt(np.mean(err ** 2))) * 1e5
#         ax.axhline(eval_rmse_mgal, color="black", ls="--", lw=1.2,
#                    label=f"eval RMSE = {eval_rmse_mgal:.1f} mGal")
 
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlabel("Measurement step $M$")
#     ax.set_ylabel(r"Posterior $\sigma_g$ [mGal]")
#     ax.set_title("(c) Precision across training")
#     ax.legend(loc="lower left", frameon=False, fontsize=9)
#     ax.set_xlim(1, 140)
 
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()
#     print(f"  Saved: {out_path}")
 


# -----------------------------------------------------------------------------
# Figure 2: eval summary
# -----------------------------------------------------------------------------

def plot_eval_summary(eval_data: dict, out_path: Path,
                      Delta_g: float = 0.0444, g_lo: float = 9.7806):
    g_true = eval_data["g_true"]
    g_hat = eval_data["g_hat"]
    max_q = eval_data["max_q"]
    err = g_hat - g_true

    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.scatter(g_true, g_hat, s=4, alpha=0.4, c=max_q, cmap="viridis")
    lo, hi = g_true.min(), g_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=r"$\hat g = g$")
    ax.set_xlabel(r"True $g$ (m/s$^2$)")
    ax.set_ylabel(r"Estimated $\hat g$ (m/s$^2$)")
    ax.set_title(f"(a) Estimator vs truth (N={len(g_true)})")
    ax.set_aspect("equal")

    ax = axes[1]
    err_norm = err / Delta_g
    ax.hist(err_norm, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(r"$(\hat g - g)/\Delta g$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"(b) Error distribution\n"
        f"RMSE = {rmse * 1e5:.1f} mGal, MSE/$\\Delta g^2$ = {mse/Delta_g**2:.4f}"
    )

    ax = axes[2]
    ax.hist(max_q, bins=40, color="darkorange", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(max_q), color="red", lw=2, ls="--",
               label=fr"mean $= {np.mean(max_q):.3f}$")
    ax.set_xlabel(r"$\max_k q_k$ (posterior peak)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Posterior confidence dist.")
    ax.set_xlim(0, 1)
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------------
# Figure 3: per-step trajectory metrics (final ckpt)
# -----------------------------------------------------------------------------

def plot_trajectory_metrics(roll_records: list[dict], out_path: Path,
                            Delta_g: float = 0.0444):
    """Use the LAST training iter's rollout as the 'final trained' trajectory."""
    last_iter = max(r["train_iter"] for r in roll_records)
    final_records = [r for r in roll_records if r["train_iter"] == last_iter]
    steps = sorted(set(r["meas_step"] for r in final_records))

    by_step = defaultdict(list)
    for r in final_records:
        by_step[r["meas_step"]].append(r)

    max_q_mean = [np.mean([r["max_q"] for r in by_step[s]]) for s in steps]
    q_close_mean = [np.mean([r["q_close"] for r in by_step[s]]) for s in steps]
    k_g_mean = [np.mean([r["k_g"] for r in by_step[s]]) for s in steps]
    T_s_mean = [np.mean([r["T_s"] * 1e6 for r in by_step[s]]) for s in steps]
    Bp_mean = [np.mean([r["Bp_kTm"] for r in by_step[s]]) for s in steps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(steps, max_q_mean, "o-", color="darkred", label=r"$\max q_k$")
    ax.plot(steps, q_close_mean, "s-", color="darkgreen", label=r"$q_{\rm true}$")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel("Posterior weight")
    ax.set_ylim(0, 1)
    ax.set_title(f"(a) Bank localization (iter {last_iter})")
    ax.legend(loc="best", frameon=False)

    ax = axes[0, 1]
    ax.plot(steps, k_g_mean, "o-", color="steelblue")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(r"$k_g$ (rad / (m/s$^2$))")
    ax.set_yscale("log")
    ax.set_title(r"(b) Learned $k_g$ schedule")

    ax = axes[1, 0]
    ax.plot(steps, T_s_mean, "o-", color="purple")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(r"$T_s$ ($\mu$s)")
    ax.set_title(r"(c) Free-evolution time $T_s$ schedule")

    ax = axes[1, 1]
    ax.plot(steps, Bp_mean, "o-", color="darkorange")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(r"$B'$ (kT/m)")
    ax.set_title(r"(d) Magnetic-field-gradient $B'$ schedule")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------------
# Figure 4: physical setup
# -----------------------------------------------------------------------------

def plot_physics_setup(out_path: Path,
                       T_min_s: float = 10e-6, T_max_s: float = 500e-6,
                       Bp_min_kTm: float = 0.5, Bp_max_kTm: float = 25.0,
                       T2_spin_s: float = 1.0e-3,
                       omega_rad_s: float = 2 * math.pi * 1e4,
                       gamma_e_rad_s_T: float = 2 * math.pi * 28e9,
                       kT_to_T: float = 1e3,
                       Delta_g: float = 0.0444,
                       K_modes: int = 4):
    """Visibility, sensitivity, info-per-shot diagnostics."""
    tau = 2 * math.pi / omega_rad_s

    T_arr = np.geomspace(T_min_s, T_max_s, 200)
    Bp_arr = np.linspace(Bp_min_kTm, Bp_max_kTm, 50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    cycle = 3.5 * tau + 2 * T_arr
    vis = np.exp(-cycle / T2_spin_s)
    ax = axes[0, 0]
    ax.plot(T_arr * 1e6, vis, "-", color="darkorange", lw=2)
    ax.axvline(T_min_s * 1e6, color="gray", ls="--", alpha=0.6,
               label=f"$T_{{\\min}}={T_min_s*1e6:.0f}\\mu$s")
    ax.axvline(T_max_s * 1e6, color="gray", ls=":", alpha=0.6,
               label=f"$T_{{\\max}}={T_max_s*1e6:.0f}\\mu$s")
    ax.set_xlabel(r"$T_s$ ($\mu$s)")
    ax.set_ylabel(r"Visibility $\mathcal{V}(T_s) = \exp(-(3.5\tau + 2 T_s)/T_2)$")
    ax.set_xscale("log")
    ax.set_title(f"(a) Spin decoherence (T_2={T2_spin_s*1e3:.1f} ms)")
    ax.legend(loc="best", frameon=False)
    ax.set_ylim(0, 1)

    T_grid, Bp_grid = np.meshgrid(T_arr, Bp_arr)
    k_g = (2 * gamma_e_rad_s_T / omega_rad_s) * (Bp_grid * kT_to_T) * T_grid ** 2

    ax = axes[0, 1]
    im = ax.pcolormesh(T_grid * 1e6, Bp_grid, np.log10(k_g + 1e-12),
                       shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}(k_g)$")
    ax.set_xlabel(r"$T_s$ ($\mu$s)")
    ax.set_ylabel(r"$B'$ (kT/m)")
    ax.set_xscale("log")
    ax.set_title(r"(b) Sensitivity $k_g(T_s, B')$")

    alias_cap = math.pi / Delta_g
    ax.contour(T_grid * 1e6, Bp_grid, k_g, levels=[alias_cap],
               colors="red", linewidths=1.5)
    ax.text(0.05, 0.95, f"red: $k_g = \\pi/\\Delta g \\approx {alias_cap:.0f}$",
            transform=ax.transAxes, color="red", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    def info_per_shot(k, A, K, Delta_g):
        modes = np.linspace(0, Delta_g, K, endpoint=False) + Delta_g / (2 * K)
        p_per = 0.5 * (1 + A * np.cos(k * modes))
        p_avg = np.mean(p_per)
        H_avg = -(p_avg * np.log2(np.clip(p_avg, 1e-10, 1)) +
                  (1 - p_avg) * np.log2(np.clip(1 - p_avg, 1e-10, 1)))
        H_per = -np.mean(p_per * np.log2(np.clip(p_per, 1e-10, 1)) +
                         (1 - p_per) * np.log2(np.clip(1 - p_per, 1e-10, 1)))
        return H_avg - H_per

    k_arr = np.geomspace(20, 1000, 50)
    info_arr = [info_per_shot(k, 0.65, K_modes, Delta_g) for k in k_arr]

    ax = axes[1, 0]
    ax.plot(k_arr, info_arr, "-", color="steelblue", lw=2)
    ax.axvline(alias_cap, color="red", ls="--", lw=1,
               label=f"$\\pi/\\Delta g$ ($\\approx${alias_cap:.0f})")
    ax.axvline(math.pi / (Delta_g / math.sqrt(12)), color="red", ls=":", lw=1,
               label=f"$\\pi/\\sigma_{{\\rm prior}}$ ($\\approx${math.pi/(Delta_g/math.sqrt(12)):.0f})")
    ax.set_xlabel(r"$k_g$")
    ax.set_ylabel("Mutual information per shot (bits)")
    ax.set_xscale("log")
    ax.set_title(f"(c) Info per binary measurement (K={K_modes}, $\\mathcal{{V}}$=0.65)")
    ax.legend(loc="best", frameon=False)

    ax = axes[1, 1]
    k_levels = [50, 100, 200, 300, 500, 1000, 2000]
    CS = ax.contour(T_grid * 1e6, Bp_grid, k_g, levels=k_levels,
                    colors="black", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
    ax.fill_between([T_min_s * 1e6, T_max_s * 1e6],
                    [Bp_min_kTm, Bp_min_kTm],
                    [Bp_max_kTm, Bp_max_kTm],
                    alpha=0.1, color="green", label="Feasible (T,B')")
    ax.set_xlabel(r"$T_s$ ($\mu$s)")
    ax.set_ylabel(r"$B'$ (kT/m)")
    ax.set_xscale("log")
    ax.set_title(r"(d) Operating envelope with $k_g$ contours")
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_paper_plots.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    Delta_g = 0.0444

    print(f"Loading from {run_dir} ...")
    train = load_jsonl(run_dir / "train_debug.jsonl")
    roll = load_jsonl(run_dir / "rollout_debug.jsonl")
    eval_data = dict(np.load(run_dir / "gravity_gm_diag_eval.npz"))

    print(f"  {len(train)} training iterations")
    print(f"  {len(roll)} rollout records")
    print(f"  {len(eval_data['g_true'])} eval episodes")
    print()
    print(f"Eval summary:")
    err = eval_data["g_hat"] - eval_data["g_true"]
    mse = np.mean(err ** 2)
    rmse = math.sqrt(mse)
    print(f"  MSE          = {mse:.4e} m^2 s^-4")
    print(f"  RMSE         = {rmse * 1e5:.2f} mGal")
    print(f"  MSE/Δg²      = {mse / Delta_g**2:.4e}")
    print(f"  mean max_q   = {np.mean(eval_data['max_q']):.3f}")
    print()

    plot_training_curves(train, roll, eval_data,
                         run_dir / "fig1_training_curves.png", Delta_g=Delta_g)
    plot_eval_summary(eval_data, run_dir / "fig2_eval_summary.png", Delta_g=Delta_g)
    plot_trajectory_metrics(roll, run_dir / "fig3_trajectory_metrics.png", Delta_g=Delta_g)
    plot_physics_setup(run_dir / "fig4_physics_setup.png", Delta_g=Delta_g)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()