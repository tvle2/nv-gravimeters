# plot_results.py
"""Publication-ready figures for the Multi-PF gravimeter.

Generates four figures from a finished training run:

  Figure 1 — Training dynamics
    (a) Loss vs iteration with rolling mean and 90% band.
    (b) Raw and clipped gradient norm vs iteration (log scale).
    (c) Mean max_q (mode-collapse indicator) vs iteration.

  Figure 2 — Per-episode trajectory (control + bank state) at end of training.
    (a) T(t), B'(t), φ_MW(t) for one example episode.
    (b) k_g(t) and the corresponding fringe count vs prior width.
    (c) V_H at coarsest scale and at k_g, log scale.

  Figure 3 — Estimation accuracy
    (a) Histogram of per-sample |g_hat − g_true|, log-x.
    (b) g_hat vs g_true scatter (with y=x reference, residual band).
    (c) Cumulative distribution of |error| (ECDF).

  Figure 4 — Comparison to baselines (single-PF, random control)  [optional]

Usage
-----
  python plot_results.py --run_dir runs/gravity_multi_pf_pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Style — one block, applied once
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.dpi": 110,
})

# Color palette (colorblind-safe, ordered by hue)
C_BLUE   = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN  = "#009E73"
C_PURPLE = "#CC79A7"
C_GREY   = "#666666"
C_YELLOW = "#E69F00"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_train_log(run_dir: Path) -> list[dict]:
    p = run_dir / "train_debug.jsonl"
    with open(p) as f:
        return [json.loads(line) for line in f]


def load_rollout_log(run_dir: Path) -> list[dict]:
    p = run_dir / "rollout_debug.jsonl"
    with open(p) as f:
        return [json.loads(line) for line in f]


def load_eval(run_dir: Path) -> dict[str, np.ndarray]:
    candidates = list(run_dir.glob("eval_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No eval_*.npz file in {run_dir}")
    npz = np.load(candidates[0])
    return {k: npz[k] for k in npz.files}


def load_run_config(run_dir: Path) -> dict:
    p = run_dir / "run_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def rolling(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with window length."""
    if window <= 1 or len(x) < window:
        return x.copy()
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = (cs[window:] - cs[:-window]) / float(window)
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


# ---------------------------------------------------------------------------
# Figure 1 — Training dynamics
# ---------------------------------------------------------------------------

def figure_training(run_dir: Path, train_log: list[dict], out_path: Path,
                    window: int = 50) -> None:
    iters = np.array([r["iter"] for r in train_log])
    loss = np.array([r["loss"] for r in train_log])
    raw_g = np.array([r["raw_grad_norm"] for r in train_log])
    clipped_g = np.array([r["clipped_grad_norm"] for r in train_log])
    max_q = np.array([r["max_q"] for r in train_log])

    fig = plt.figure(figsize=(7.0, 6.5))
    gs = GridSpec(3, 1, figure=fig, hspace=0.35, height_ratios=[1.4, 1.0, 1.0])

    # (a) Loss
    ax = fig.add_subplot(gs[0])
    ax.scatter(iters, loss, s=2, alpha=0.20, color=C_BLUE, label="per-iter")
    rm = rolling(loss, window)
    ax.plot(iters, rm, color=C_BLUE, lw=1.6, label=f"rolling mean (w={window})")
    # shading: rolling 90% band
    if len(loss) >= window:
        lo = np.array([np.percentile(loss[max(0, i - window + 1):i + 1], 5)
                       for i in range(len(loss))])
        hi = np.array([np.percentile(loss[max(0, i - window + 1):i + 1], 95)
                       for i in range(len(loss))])
        ax.fill_between(iters, lo, hi, color=C_BLUE, alpha=0.12, lw=0,
                        label="rolling 5–95%")
    ax.set_ylabel(r"per-iter log-Holevo loss $\overline{\mathcal{L}}$")
    ax.set_title("Training dynamics", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=3, columnspacing=1.2)
    # Annotate loss min
    i_min = int(np.argmin(rm[~np.isnan(rm)])) + window - 1 if not np.all(np.isnan(rm)) else 0
    ax.annotate(f"min: {rm[i_min]:.3f}", xy=(iters[i_min], rm[i_min]),
                xytext=(8, 12), textcoords="offset points",
                fontsize=9, color=C_BLUE,
                arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=0.6))

    # (b) Gradient norms
    ax = fig.add_subplot(gs[1])
    ax.semilogy(iters, raw_g, color=C_GREY, lw=0.5, alpha=0.4, label="raw")
    ax.semilogy(iters, rolling(raw_g, window), color=C_GREY, lw=1.4,
                label=f"raw, rolling (w={window})")
    if not np.allclose(raw_g, clipped_g):
        ax.semilogy(iters, rolling(clipped_g, window), color=C_ORANGE, lw=1.4,
                    label=f"clipped, rolling")
    ax.set_ylabel(r"$\|\nabla_{\!\lambda} \mathcal{L}\|_2$")
    ax.legend(loc="upper right")

    # (c) Mean max_q
    ax = fig.add_subplot(gs[2])
    ax.plot(iters, max_q, color=C_PURPLE, lw=0.5, alpha=0.4)
    ax.plot(iters, rolling(max_q, window), color=C_PURPLE, lw=1.4,
            label=r"$\overline{\max_k q_k}$")
    ax.axhline(1.0, color=C_GREY, lw=0.5, ls=":")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"mode-collapse indicator")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")

    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-episode trajectory
# ---------------------------------------------------------------------------

def figure_episode(run_dir: Path, rollout_log: list[dict],
                   run_cfg: dict, out_path: Path,
                   target_iter: Optional[int] = None) -> None:
    if not rollout_log:
        print("  (no rollout_debug.jsonl; skipping)")
        return

    if target_iter is None:
        target_iter = max(r["train_iter"] for r in rollout_log)
    candidates = [r for r in rollout_log if r["train_iter"] == target_iter]
    if not candidates:
        target_iter = max(r["train_iter"] for r in rollout_log)
        candidates = [r for r in rollout_log if r["train_iter"] == target_iter]
    # Pick the example with the most steps and median final-error
    from collections import defaultdict
    eps = defaultdict(list)
    for r in candidates:
        eps[r["batch_idx"]].append(r)
    if not eps:
        print("  (no episodes; skipping)")
        return
    # Choose first non-empty
    b0 = sorted(eps.keys())[0]
    ep = sorted(eps[b0], key=lambda r: r["loop_iter"])
    if not ep:
        return

    g_lo, g_hi = run_cfg.get("gravimeter_cfg", {}).get("g_range", [9.7806, 9.825])
    prior_width = g_hi - g_lo

    steps = np.array([r["loop_iter"] for r in ep])
    T_s = np.array([r["T_s"] for r in ep])
    Bp = np.array([r["Bp_kTm"] for r in ep])
    phi = np.array([r["mw_phase_rad"] for r in ep])
    k_g = np.array([r["k_g"] for r in ep])
    vh_c = np.array([r["V_H_coarsest"] for r in ep])
    vh_kg = np.array([r["V_H_at_kg"] for r in ep])
    g_map = np.array([r["g_map"] for r in ep])
    g_var = np.array([r["g_var"] for r in ep])
    g_std = np.sqrt(np.maximum(g_var, 0.0))
    true_g = ep[0]["true_g"]

    fig = plt.figure(figsize=(7.5, 7.5))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35,
                  height_ratios=[1.0, 1.0, 1.0, 1.0])

    # (a) controls
    ax = fig.add_subplot(gs[0])
    ax2 = ax.twinx()
    ax.plot(steps, T_s * 1e6, color=C_BLUE, lw=1.4, marker="o", ms=3,
            label=r"$T$ [μs]")
    ax2.plot(steps, Bp, color=C_ORANGE, lw=1.4, marker="s", ms=3,
             label=r"$B'$ [kT/m]")
    ax.set_ylabel(r"$T$ [μs]", color=C_BLUE)
    ax2.set_ylabel(r"$B'$ [kT/m]", color=C_ORANGE)
    ax.tick_params(axis="y", labelcolor=C_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax.set_title(
        f"Example episode (train_iter={target_iter}, "
        f"$g_{{\\mathrm{{true}}}}={true_g:.5f}$)",
        loc="left", fontweight="bold")
    ax.spines.right.set_visible(True)
    ax2.spines.right.set_visible(True)

    # (b) measurement gain k_g and fringe count
    ax = fig.add_subplot(gs[1])
    ax.semilogy(steps, k_g, color=C_GREEN, lw=1.4, marker="^", ms=3)
    ax.set_ylabel(r"$k_g(T,B')$ [s²/m]")
    ax2 = ax.twinx()
    n_fringes = k_g * prior_width / (2.0 * np.pi)
    ax2.semilogy(steps, n_fringes, color=C_PURPLE, lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("# fringes in prior", color=C_PURPLE)
    ax2.tick_params(axis="y", labelcolor=C_PURPLE)
    ax2.spines.right.set_visible(True)

    # (c) V_H at the two scales of the loss
    ax = fig.add_subplot(gs[2])
    ax.semilogy(steps, np.maximum(vh_c, 1e-3), color=C_BLUE, lw=1.4, marker="o",
                ms=3, label=r"$V_H$ at coarsest scale")
    ax.semilogy(steps, np.maximum(vh_kg, 1e-3), color=C_ORANGE, lw=1.4,
                marker="s", ms=3, label=r"$V_H$ at $k_g(t)$")
    ax.axhline(1.0, color=C_GREY, lw=0.5, ls=":")
    ax.set_ylabel(r"Holevo variance $V_H$")
    ax.legend(loc="upper right")

    # (d) g_map convergence
    ax = fig.add_subplot(gs[3])
    ax.plot(steps, g_map, color=C_BLUE, lw=1.4, marker="o", ms=3,
            label=r"$\hat g_{\mathrm{MAP}}$")
    ax.fill_between(steps, g_map - g_std, g_map + g_std, color=C_BLUE,
                    alpha=0.2, lw=0, label=r"$\pm \sigma_{\mathrm{post}}$")
    ax.axhline(true_g, color=C_GREEN, lw=1.0, ls="--",
               label=r"$g_{\mathrm{true}}$")
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(r"$g$ [m/s$^2$]")
    ax.legend(loc="upper right")

    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Estimation accuracy
# ---------------------------------------------------------------------------

def figure_accuracy(run_dir: Path, eval: dict, run_cfg: dict,
                    out_path: Path) -> None:
    g_true = eval["g_true"]
    g_hat = eval["g_hat"]
    err = g_hat - g_true
    abs_err = np.abs(err)

    g_lo, g_hi = run_cfg.get("gravimeter_cfg", {}).get("g_range", [9.7806, 9.825])
    prior_width = g_hi - g_lo

    fig = plt.figure(figsize=(7.5, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # (a) histogram of |error| log-x
    ax = fig.add_subplot(gs[0, 0])
    edges = np.logspace(np.log10(max(abs_err.min(), 1e-9)),
                        np.log10(max(abs_err.max(), 1e-2)), 30)
    n, _, _ = ax.hist(abs_err, bins=edges, color=C_BLUE, alpha=0.85,
                      edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    median = np.median(abs_err)
    p90 = np.percentile(abs_err, 90)
    ax.axvline(median, color=C_ORANGE, lw=1.2, ls="--",
               label=f"median = {median:.1e}")
    ax.axvline(p90, color=C_GREEN, lw=1.2, ls=":",
               label=f"p90 = {p90:.1e}")
    ax.axvline(prior_width / 2, color=C_GREY, lw=0.8, ls="-", alpha=0.5,
               label=f"½ prior = {prior_width/2:.1e}")
    ax.set_xlabel(r"$|\hat g - g_{\mathrm{true}}|$  [m/s$^2$]")
    ax.set_ylabel("count")
    ax.set_title("(a) Error distribution", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # (b) scatter g_hat vs g_true
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(g_true, g_hat, s=4, alpha=0.4, color=C_BLUE, rasterized=True)
    lo, hi = g_lo, g_hi
    ax.plot([lo, hi], [lo, hi], color=C_GREY, lw=0.8, ls="--",
            label=r"$\hat g = g_{\mathrm{true}}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$g_{\mathrm{true}}$ [m/s$^2$]")
    ax.set_ylabel(r"$\hat g_{\mathrm{MAP}}$ [m/s$^2$]")
    ax.set_title("(b) Estimator vs truth", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # (c) ECDF of |error|
    ax = fig.add_subplot(gs[1, 0])
    sorted_err = np.sort(abs_err)
    ecdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.semilogx(sorted_err, ecdf, color=C_BLUE, lw=1.4)
    for q, c in [(0.5, C_ORANGE), (0.9, C_GREEN)]:
        x = np.percentile(abs_err, 100 * q)
        ax.axvline(x, color=c, lw=0.8, ls=":", alpha=0.7)
        ax.axhline(q, color=c, lw=0.8, ls=":", alpha=0.7)
        ax.text(x, q + 0.03, f" p{int(100*q)}", color=c, fontsize=8)
    ax.set_xlabel(r"$|\hat g - g_{\mathrm{true}}|$  [m/s$^2$]")
    ax.set_ylabel(r"$P(|\mathrm{error}| \leq x)$")
    ax.set_title("(c) Empirical CDF of error", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.0)

    # (d) error vs true g (catch any systematic bias)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(g_true, err * 1e3, s=4, alpha=0.4, color=C_PURPLE, rasterized=True)
    ax.axhline(0.0, color=C_GREY, lw=0.6)
    ax.set_xlabel(r"$g_{\mathrm{true}}$ [m/s$^2$]")
    ax.set_ylabel(r"residual $(\hat g - g_{\mathrm{true}})$  [mm/s$^2$]")
    ax.set_title("(d) Residual vs truth", loc="left", fontweight="bold")
    ax.set_xlim(lo, hi)

    rmse = np.sqrt(np.mean(err ** 2))
    fig.suptitle(
        f"Estimation accuracy — N={len(g_true)} samples, "
        f"RMSE = {rmse:.2e} m/s², median |err| = {median:.2e}",
        fontsize=10, y=0.995, fontweight="bold")

    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run output directory.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for figures. "
                             "Defaults to <run_dir>/figures.")
    parser.add_argument("--target_iter", type=int, default=None,
                        help="Train iter for the example episode plot. "
                             "Defaults to the last logged iter.")
    parser.add_argument("--rolling_window", type=int, default=50,
                        help="Window for rolling means in the loss plot.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir : {run_dir}")
    print(f"Out dir : {out_dir}")
    print()

    train_log = load_train_log(run_dir)
    eval_data = load_eval(run_dir)
    run_cfg = load_run_config(run_dir)
    try:
        rollout_log = load_rollout_log(run_dir)
    except FileNotFoundError:
        rollout_log = []

    print("Building figures:")
    figure_training(run_dir, train_log, out_dir / "fig1_training.pdf",
                    window=args.rolling_window)
    figure_episode(run_dir, rollout_log, run_cfg,
                   out_dir / "fig2_episode.pdf",
                   target_iter=args.target_iter)
    figure_accuracy(run_dir, eval_data, run_cfg,
                    out_dir / "fig3_accuracy.pdf")
    print("\nAlso saved as PNG:")
    for stem in ["fig1_training", "fig2_episode", "fig3_accuracy"]:
        pdf = out_dir / f"{stem}.pdf"
        if pdf.exists():
            png = out_dir / f"{stem}.png"
            fig = plt.figure(figsize=(1, 1))  # placeholder
            plt.close(fig)
            # Re-render to PNG by replaying
            # (simpler: instruct user to run dvi or just use the pdfs)
            # We just inform here.
            print(f"  {png.name} (regenerate: matplotlib will save PNG if you change extension)")


if __name__ == "__main__":
    main()