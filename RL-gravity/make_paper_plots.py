"""
Publication plots for the NV-center gravimeter pipeline.

Usage:
  python make_paper_plots.py runs/gravity_gm_diag

Reads:
  {run_dir}/train_debug.jsonl        # per-iter training losses + grad norms
  {run_dir}/rollout_debug.jsonl      # per-step rollout snapshots
  {run_dir}/gravity_gm_diag_eval.npz # eval results
  {run_dir}/gravity_gm_diag_history.csv # window-averaged loss

Outputs (PNG, 300 dpi, in run_dir):
  fig1_training_curves.png   - MSE, NLL, max_q over training iterations
  fig2_eval_summary.png      - g_hat vs g_true scatter, error histogram
  fig3_trajectory_metrics.png - per-step max_q, q_close, k_g, T_s
  fig4_physics_setup.png     - visibility(T_s), sensitivity k_g(T,B'), info-per-shot
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
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------------- 
# Figure 1: training curves
# ----------------------------------------------------------------------------- 

def plot_training_curves(train_records: list[dict], roll_records: list[dict],
                         eval_data: dict, out_path: Path,
                         Delta_g: float = 0.0444):
    """Publication training curves.

    Panel (a) — final-step MSE/Δg² from train_debug.jsonl (per-iter, batch-mean).
    Panel (b) — final-step max_q and q_close from train_debug.jsonl (per-iter).
    Panel (c) — gradient norm from train_debug.jsonl.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # Smoothing helpers
    def moving_avg(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode="valid")

    # --- Common iteration index ---
    iters_t = np.array([r["iter"] for r in train_records])

    # =====================================================================
    # Panel (a): final-step MSE/Δg² — the publication headline metric
    # =====================================================================
    # Prefer the per-iter logged value (full batch=128, every iter).
    # Fall back to rollout reconstruction if final_mse_norm is missing
    # (e.g. for older runs from before the patch).
    if all("final_mse_norm" in r for r in train_records):
        final_mse = np.array([r["final_mse_norm"] for r in train_records])
        source_label = "from train_debug (B=128, per-iter)"
    else:
        # Legacy fallback (older runs without final_mse_norm logged)
        by_iter_finalstep = defaultdict(list)
        final_step = max(r["meas_step"] for r in roll_records)
        for r in roll_records:
            if r["meas_step"] == final_step:
                by_iter_finalstep[r["train_iter"]].append(
                    (r["g_mix"] - r["true_g"]) ** 2 / Delta_g**2
                )
        iters_t = np.array(sorted(by_iter_finalstep))
        final_mse = np.array([np.mean(by_iter_finalstep[i]) for i in iters_t])
        source_label = "from rollout debug (B=3, sparse)"

    # Eval-MSE horizontal line for reference (uses best ckpt eval)
    eval_err = eval_data["g_hat"] - eval_data["g_true"]
    eval_mse = float(np.mean(eval_err**2) / Delta_g**2)

    ax = axes[0]
    ax.plot(iters_t, final_mse, ".", color="lightsteelblue", alpha=0.35, ms=2,
            label="per-iter")
    w = 50
    if len(final_mse) >= w:
        smooth = moving_avg(final_mse, w=w)
        # Offset iters so the moving-avg curve aligns at the right of the window
        ax.plot(iters_t[w-1:], smooth, "-", color="navy", lw=2,
                label=f"{w}-iter moving avg")
    ax.axhline(eval_mse, color="darkred", ls="--", lw=1.5,
               label=f"eval MSE/Δg² = {eval_mse:.3e}")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Final-step MSE / $\Delta g^2$")
    ax.set_yscale("log")
    ax.set_title(f"(a) Final-step MSE   ({source_label})")
    ax.legend(loc="best", frameon=False, fontsize=8)

    # =====================================================================
    # Panel (b): final-step max_q and q_close (per-iter, dense)
    # =====================================================================
    if all("final_max_q" in r for r in train_records):
        max_q_arr = np.array([r["final_max_q"] for r in train_records])
        qclose_arr = np.array([r["final_qclose"] for r in train_records])
        x_iters = iters_t
    else:
        # Legacy fallback
        by_iter_maxq = defaultdict(list)
        by_iter_qclose = defaultdict(list)
        final_step = max(r["meas_step"] for r in roll_records)
        for r in roll_records:
            if r["meas_step"] == final_step:
                by_iter_maxq[r["train_iter"]].append(r["max_q"])
                by_iter_qclose[r["train_iter"]].append(r["q_close"])
        x_iters = np.array(sorted(by_iter_maxq))
        max_q_arr = np.array([np.mean(by_iter_maxq[i]) for i in x_iters])
        qclose_arr = np.array([np.mean(by_iter_qclose[i]) for i in x_iters])

    ax = axes[1]
    # Lightweight per-iter scatter
    ax.plot(x_iters, max_q_arr, ".", color="lightcoral", alpha=0.25, ms=2)
    ax.plot(x_iters, qclose_arr, ".", color="palegreen", alpha=0.25, ms=2)
    # Bold smoothed lines
    if len(max_q_arr) >= w:
        ax.plot(x_iters[w-1:], moving_avg(max_q_arr, w=w),
                "-", color="darkred", lw=2, label=r"$\max_k q_k$ (smoothed)")
        ax.plot(x_iters[w-1:], moving_avg(qclose_arr, w=w),
                "-", color="darkgreen", lw=2, label=r"$q_{\rm true\ mode}$ (smoothed)")
    else:
        ax.plot(x_iters, max_q_arr, "-", color="darkred", lw=2, label=r"$\max_k q_k$")
        ax.plot(x_iters, qclose_arr, "-", color="darkgreen", lw=2,
                label=r"$q_{\rm true\ mode}$")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Posterior weight at final step")
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Posterior confidence (final step)")
    ax.legend(loc="best", frameon=False, fontsize=8)

    # =====================================================================
    # Panel (c): gradient norm (same as before, just kept)
    # =====================================================================
    grad = np.array([r["raw_grad_norm"] for r in train_records])
    ax = axes[2]
    ax.plot(iters_t, grad, ".", color="lightsalmon", alpha=0.3, ms=2)
    if len(grad) >= w:
        ax.plot(iters_t[w-1:], moving_avg(grad, w=w),
                "-", color="darkorange", lw=2, label=f"{w}-iter moving avg")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\|\nabla L\|$")
    ax.set_yscale("log")
    ax.set_title("(c) Gradient norm")
    ax.legend(loc="best", frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    
# ----------------------------------------------------------------------------- 
# Figure 2: eval summary
# ----------------------------------------------------------------------------- 

def plot_eval_summary(eval_data: dict, out_path: Path,
                      Delta_g: float = 0.0444, g_lo: float = 9.7806):
    g_true = eval_data["g_true"]
    g_hat = eval_data["g_hat"]
    max_q = eval_data["max_q"]
    err = g_hat - g_true

    mse = np.mean(err**2)
    rmse = np.sqrt(mse)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # --- Left: g_hat vs g_true scatter ---
    ax = axes[0]
    ax.scatter(g_true, g_hat, s=4, alpha=0.4, c=max_q, cmap="viridis")
    lo, hi = g_true.min(), g_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$\hat g = g$")
    ax.set_xlabel(r"True $g$ (m/s$^2$)")
    ax.set_ylabel(r"Estimated $\hat g$ (m/s$^2$)")
    ax.set_title(f"(a) Estimator vs truth (N={len(g_true)})")
    ax.set_aspect("equal")

    # --- Middle: error histogram ---
    ax = axes[1]
    err_norm = err / Delta_g
    ax.hist(err_norm, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(r"$(\hat g - g)/\Delta g$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"(b) Error distribution\n"
        f"MSE/$\\Delta g^2$ = {mse/Delta_g**2:.4f}, "
        f"$\\log_{{10}}$ = {np.log10(mse/Delta_g**2):.3f}"
    )

    # --- Right: max_q histogram (confidence distribution) ---
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
    vis_mean = [np.mean([r["vis"] for r in by_step[s]]) for s in steps]

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
    ax.plot(steps, vis_mean, "o-", color="darkorange")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel("Visibility $\\mathcal{V}$")
    ax.set_ylim(0, 1)
    ax.set_title("(d) Readout visibility over trajectory")

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
                       omega_rad_s: float = 2*math.pi*1e4,
                       gamma_e_rad_s_T: float = 2*math.pi*28e9,
                       kT_to_T: float = 1e3,
                       Delta_g: float = 0.0444,
                       K_modes: int = 4):
    """Visibility, sensitivity, info-per-shot diagnostics."""
    tau = 2 * math.pi / omega_rad_s

    T_arr = np.geomspace(T_min_s, T_max_s, 200)
    Bp_arr = np.linspace(Bp_min_kTm, Bp_max_kTm, 50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # --- (a) visibility(T_s) — decoherence ---
    cycle = 3.5 * tau + 2 * T_arr
    vis = np.exp(-cycle / T2_spin_s)
    ax = axes[0, 0]
    ax.plot(T_arr * 1e6, vis, "-", color="darkorange", lw=2)
    ax.axvline(T_min_s * 1e6, color="gray", ls="--", alpha=0.6, label=f"$T_{{\\min}}={T_min_s*1e6:.0f}\\mu$s")
    ax.axvline(T_max_s * 1e6, color="gray", ls=":", alpha=0.6, label=f"$T_{{\\max}}={T_max_s*1e6:.0f}\\mu$s")
    ax.set_xlabel(r"$T_s$ ($\mu$s)")
    ax.set_ylabel(r"Visibility $\mathcal{V}(T_s) = \exp(-(3.5\tau + 2 T_s)/T_2)$")
    ax.set_xscale("log")
    ax.set_title(f"(a) Spin decoherence (T_2={T2_spin_s*1e3:.1f} ms)")
    ax.legend(loc="best", frameon=False)
    ax.set_ylim(0, 1)

    # --- (b) sensitivity k_g(T_s, B') ---
    T_grid, Bp_grid = np.meshgrid(T_arr, Bp_arr)
    k_g = (2 * gamma_e_rad_s_T / omega_rad_s) * (Bp_grid * kT_to_T) * T_grid**2

    ax = axes[0, 1]
    im = ax.pcolormesh(T_grid * 1e6, Bp_grid, np.log10(k_g + 1e-12),
                       shading="auto", cmap="viridis")
    cb = plt.colorbar(im, ax=ax, label=r"$\log_{10}(k_g)$")
    ax.set_xlabel(r"$T_s$ ($\mu$s)")
    ax.set_ylabel(r"$B'$ (kT/m)")
    ax.set_xscale("log")
    ax.set_title(r"(b) Sensitivity $k_g(T_s, B')$")

    # Overlay V8 alias cap and posterior cap at init
    alias_cap = math.pi / Delta_g
    ax.contour(T_grid * 1e6, Bp_grid, k_g, levels=[alias_cap],
               colors="red", linewidths=1.5)
    ax.text(0.05, 0.95, f"red: $k_g = \\pi/\\Delta g \\approx {alias_cap:.0f}$",
            transform=ax.transAxes, color="red", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # --- (c) info per shot as a function of k_g (at vis=0.65) ---
    def info_per_shot(k, A, K, Delta_g):
        modes = np.linspace(0, Delta_g, K, endpoint=False) + Delta_g / (2*K)
        # Best-case greedy phi = 0 (analytic for K modes can vary; this is illustrative)
        p_per = 0.5 * (1 + A * np.cos(k * modes))
        p_avg = np.mean(p_per)
        H_avg = -(p_avg * np.log2(np.clip(p_avg, 1e-10, 1)) +
                  (1-p_avg) * np.log2(np.clip(1-p_avg, 1e-10, 1)))
        H_per = -np.mean(p_per * np.log2(np.clip(p_per, 1e-10, 1)) +
                         (1-p_per) * np.log2(np.clip(1-p_per, 1e-10, 1)))
        return H_avg - H_per

    k_arr = np.geomspace(20, 1000, 50)
    info_arr = [info_per_shot(k, 0.65, K_modes, Delta_g) for k in k_arr]

    ax = axes[1, 0]
    ax.plot(k_arr, info_arr, "-", color="steelblue", lw=2)
    ax.axvline(alias_cap, color="red", ls="--", lw=1,
               label=f"$\\pi/\\Delta g$ ($\\approx${alias_cap:.0f})")
    ax.axvline(math.pi / (Delta_g/math.sqrt(12)), color="red", ls=":", lw=1,
               label=f"$\\pi/\\sigma_{{\\rm prior}}$ ($\\approx${math.pi/(Delta_g/math.sqrt(12)):.0f})")
    ax.set_xlabel(r"$k_g$")
    ax.set_ylabel("Mutual information per shot (bits)")
    ax.set_xscale("log")
    ax.set_title(f"(c) Info per binary measurement (K={K_modes}, $\\mathcal{{V}}$=0.65)")
    ax.legend(loc="best", frameon=False)

    # --- (d) feasible (T_s, B') region with k_g contours ---
    ax = axes[1, 1]
    k_levels = [50, 100, 200, 300, 500, 1000, 2000]
    CS = ax.contour(T_grid * 1e6, Bp_grid, k_g, levels=k_levels,
                    colors="black", linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
    # Shade feasible operating region (T_s, B' in their ranges and k_g ≤ alias cap)
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

    # Load all data
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
    mse = np.mean(err**2)
    print(f"  MSE          = {mse:.4e} m^2 s^-4")
    print(f"  MSE/Δg²      = {mse/Delta_g**2:.4e}")
    print(f"  log₁₀(MSE/Δg²) = {np.log10(mse/Delta_g**2):.3f}")
    print(f"  mean max_q   = {np.mean(eval_data['max_q']):.3f}")
    print()

    plot_training_curves(train, roll, eval_data, run_dir / "fig1_training_curves.png", Delta_g=Delta_g)
    plot_eval_summary(eval_data, run_dir / "fig2_eval_summary.png", Delta_g=Delta_g)
    plot_trajectory_metrics(roll, run_dir / "fig3_trajectory_metrics.png", Delta_g=Delta_g)
    plot_physics_setup(run_dir / "fig4_physics_setup.png", Delta_g=Delta_g)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()