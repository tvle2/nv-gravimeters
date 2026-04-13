"""Comprehensive plotting for Multi-PF Bank training and evaluation.

Usage
-----
::
    # After training (reads from the run output directory):
    python plot_multi_pf.py runs/gravity_multi_pf_pilot

    # Or specify individual files:
    python plot_multi_pf.py --history loss_history.csv --eval eval_pilot.npy --config run_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BLUE = "#2563eb"
C_BLUE_LIGHT = "#93c5fd"
C_GREEN = "#10b981"
C_GREEN_LIGHT = "#6ee7b7"
C_RED = "#ef4444"
C_AMBER = "#f59e0b"
C_GRAY = "#6b7280"
C_PURPLE = "#8b5cf6"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def find_files(run_dir: Path) -> dict:
    """Auto-discover output files in a run directory."""
    files = {}
    # History CSV — match the qsensoropt naming pattern
    csvs = list(run_dir.glob("*_history.csv"))
    if csvs:
        files["history"] = csvs[0]
    # Extended eval npz (preferred)
    npzs = list(run_dir.glob("eval_*_extended.npz"))
    if npzs:
        files["eval_ext"] = npzs[0]
    # Eval npy (fallback)
    npys = list(run_dir.glob("eval_*.npy"))
    if npys:
        files["eval"] = npys[0]
    # Config
    jsons = list(run_dir.glob("run_config.json"))
    if jsons:
        files["config"] = jsons[0]
    return files


def load_eval_extended(path: Path) -> dict:
    """Load extended eval data (v_h, mse, rmse, g_true, g_hat)."""
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_loss_history(path: Path) -> np.ndarray:
    """Load checkpoint-averaged loss from the training CSV."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1, filling_values=np.nan)
    if data.ndim == 0:
        return np.array([float(data)])
    return data.flatten()


def load_eval(path: Path) -> np.ndarray:
    return np.load(str(path))


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_training_loss(ax: plt.Axes, losses: np.ndarray, interval_save: int):
    """Training loss curve with rolling average."""
    n = len(losses)
    iters = np.arange(1, n + 1) * interval_save

    ax.plot(iters, losses, "-", color=C_BLUE_LIGHT, linewidth=0.8, alpha=0.7, label="Checkpoint avg")

    # Rolling average (window = 10 checkpoints)
    win = min(10, n)
    if n >= win:
        rolling = np.convolve(losses, np.ones(win) / win, mode="valid")
        rolling_x = iters[win - 1:]
        ax.plot(rolling_x, rolling, "-", color=C_BLUE, linewidth=2.0,
                label=f"Rolling avg (w={win})")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Holevo variance (V_H)")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate start/end
    ax.annotate(f"{losses[0]:.3f}", (iters[0], losses[0]),
                textcoords="offset points", xytext=(8, 8), fontsize=8, color=C_GRAY)
    ax.annotate(f"{losses[-1]:.3f}", (iters[-1], losses[-1]),
                textcoords="offset points", xytext=(-30, 8), fontsize=8, color=C_BLUE)


def plot_training_loss_log(ax: plt.Axes, losses: np.ndarray, interval_save: int):
    """Same as above but log scale — reveals fine convergence."""
    n = len(losses)
    iters = np.arange(1, n + 1) * interval_save

    ax.semilogy(iters, losses, "-", color=C_BLUE_LIGHT, linewidth=0.8, alpha=0.7)

    win = min(10, n)
    if n >= win:
        rolling = np.convolve(losses, np.ones(win) / win, mode="valid")
        rolling_x = iters[win - 1:]
        ax.semilogy(rolling_x, rolling, "-", color=C_BLUE, linewidth=2.0, label=f"Rolling avg")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("V_H (log scale)")
    ax.set_title("Training Loss (log)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, which="both")


def plot_eval_bars(ax: plt.Axes, eval_data: np.ndarray):
    """Per-episode evaluation V_H."""
    n = len(eval_data)
    colors = [C_GREEN if v < 1.0 else (C_AMBER if v < 3.0 else C_RED) for v in eval_data]
    ax.bar(range(n), eval_data, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=np.mean(eval_data), color=C_RED, linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(eval_data):.4f}")
    ax.axhline(y=np.median(eval_data), color=C_AMBER, linestyle=":", linewidth=1.5,
               label=f"median = {np.median(eval_data):.4f}")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Holevo variance (V_H)")
    ax.set_title("Evaluation: Per-Episode V_H")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_eval_histogram(ax: plt.Axes, eval_data: np.ndarray):
    """Histogram of evaluation V_H."""
    bins = np.linspace(0, min(np.max(eval_data) * 1.1, 10.0), 25)
    ax.hist(eval_data, bins=bins, color=C_GREEN, edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(np.mean(eval_data), color=C_RED, linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(eval_data):.4f}")
    ax.axvline(np.median(eval_data), color=C_AMBER, linestyle=":", linewidth=1.5,
               label=f"median = {np.median(eval_data):.4f}")
    ax.set_xlabel("Holevo variance (V_H)")
    ax.set_ylabel("Count")
    ax.set_title("Evaluation: V_H Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")


def plot_eval_quality_buckets(ax: plt.Axes, eval_data: np.ndarray):
    """Quality bucket chart."""
    labels = ["V_H<0.1\nexcellent", "0.1–0.5\ngood", "0.5–1.0\nok",
              "1.0–3.0\npartial", "3.0–10\npoor", "≥10\nfailed"]
    thresholds = [0.1, 0.5, 1.0, 3.0, 10.0, np.inf]
    colors_bar = ["#10b981", "#34d399", "#fbbf24", "#f97316", "#ef4444", "#991b1b"]

    counts = []
    lo = 0.0
    for hi in thresholds:
        counts.append(int(np.sum((eval_data >= lo) & (eval_data < hi))))
        lo = hi

    bars = ax.bar(labels, counts, color=colors_bar, edgecolor="white", linewidth=0.5)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c + 0.3, str(c), ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel(f"Episodes (of {len(eval_data)})")
    ax.set_title("Evaluation: Quality Buckets")
    ax.set_ylim(0, max(max(counts) + 2, 1))
    ax.grid(True, alpha=0.2, axis="y")


def plot_eval_sorted(ax: plt.Axes, eval_data: np.ndarray):
    """Sorted V_H curve — shows the CDF-like shape."""
    sorted_vh = np.sort(eval_data)
    pct = np.linspace(0, 100, len(sorted_vh))
    ax.plot(pct, sorted_vh, "-o", color=C_PURPLE, linewidth=2, markersize=3)
    ax.axhline(y=1.0, color=C_AMBER, linestyle=":", alpha=0.5, label="V_H = 1.0 threshold")
    ax.fill_between(pct, 0, sorted_vh, alpha=0.15, color=C_PURPLE)
    pct_below_1 = 100.0 * np.mean(sorted_vh < 1.0)
    ax.set_xlabel("Percentile of episodes")
    ax.set_ylabel("Holevo variance (V_H)")
    ax.set_title(f"Sorted Eval V_H ({pct_below_1:.0f}% below 1.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


def plot_mse_rmse_bars(ax: plt.Axes, eval_ext: dict):
    """Per-episode RMSE bar chart."""
    rmse = eval_ext["rmse"]
    n = len(rmse)
    colors = [C_BLUE if r < 1e-3 else (C_AMBER if r < 5e-3 else C_RED) for r in rmse]
    ax.bar(range(n), rmse * 1e3, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=np.mean(rmse) * 1e3, color=C_RED, linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(rmse)*1e3:.3f} mm/s\u00b2")
    ax.axhline(y=np.median(rmse) * 1e3, color=C_AMBER, linestyle=":", linewidth=1.5,
               label=f"median = {np.median(rmse)*1e3:.3f} mm/s\u00b2")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("RMSE (mm/s\u00b2)")
    ax.set_title("Evaluation: Per-Episode RMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_g_true_vs_g_hat(ax: plt.Axes, eval_ext: dict):
    """Scatter plot: true g vs estimated g."""
    g_true = eval_ext["g_true"]
    g_hat = eval_ext["g_hat"]
    ax.scatter(g_true, g_hat, s=8, alpha=0.5, color=C_PURPLE, edgecolors="none")
    # Perfect estimation line
    g_min = min(g_true.min(), g_hat.min())
    g_max = max(g_true.max(), g_hat.max())
    margin = (g_max - g_min) * 0.05
    lims = [g_min - margin, g_max + margin]
    ax.plot(lims, lims, "--", color=C_GRAY, linewidth=1, alpha=0.7, label="y = x (perfect)")
    ax.set_xlabel("True g (m/s\u00b2)")
    ax.set_ylabel("Estimated g (m/s\u00b2)")
    ax.set_title("g\u0302 vs g (all samples)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    # Annotate correlation
    corr = np.corrcoef(g_true, g_hat)[0, 1]
    rmse_global = np.sqrt(np.mean((g_true - g_hat) ** 2))
    ax.text(0.05, 0.92,
            f"R\u00b2 = {corr**2:.4f}\nRMSE = {rmse_global:.2e} m/s\u00b2",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def plot_error_histogram(ax: plt.Axes, eval_ext: dict):
    """Histogram of estimation errors (g_hat - g_true)."""
    errors = eval_ext["g_hat"] - eval_ext["g_true"]
    errors_mm = errors * 1e3  # convert to mm/s^2
    ax.hist(errors_mm, bins=30, color=C_PURPLE, edgecolor="white",
            linewidth=0.5, alpha=0.8, density=True)
    ax.axvline(0, color=C_GRAY, linestyle="-", linewidth=1, alpha=0.5)
    ax.axvline(np.mean(errors_mm), color=C_RED, linestyle="--", linewidth=1.5,
               label=f"bias = {np.mean(errors_mm):.3f} mm/s\u00b2")
    ax.set_xlabel("Error: g\u0302 \u2212 g (mm/s\u00b2)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")


def plot_vh_vs_mse_scatter(ax: plt.Axes, eval_ext: dict):
    """Scatter: V_H vs MSE per episode — shows correlation between losses."""
    vh = eval_ext["v_h"]
    mse = eval_ext["mse"]
    ax.scatter(vh, mse, s=30, alpha=0.7, color=C_BLUE, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Holevo variance (V_H)")
    ax.set_ylabel("MSE (m/s\u00b2)\u00b2")
    ax.set_title("V_H vs MSE (per episode)")
    ax.grid(True, alpha=0.2)
    # Fit line
    if len(vh) > 2:
        corr = np.corrcoef(vh, mse)[0, 1]
        ax.text(0.05, 0.92, f"corr = {corr:.3f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def plot_config_summary(ax: plt.Axes, config: dict):
    """Text panel showing key configuration."""
    ax.axis("off")

    profile = config.get("profile", {})
    gcfg = config.get("gravimeter_cfg", {})
    bcfg = config.get("bank_cfg", {})

    lines = [
        f"Profile: {config.get('run_profile', '?')}",
        f"Noise: {config.get('noise_mode', '?')}",
        "",
        "— Training —",
        f"  batchsize: {profile.get('batchsize', '?')}",
        f"  iterations: {profile.get('iterations', '?')}",
        f"  grad_accum: {profile.get('gradient_accumulation', '?')}",
        f"  max_steps: {profile.get('max_steps', '?')}",
        f"  max_resources: {profile.get('max_resources', '?')} s",
        f"  lr: {profile.get('initial_lr', '?')}",
        "",
        "— Bank —",
        f"  N_total: {bcfg.get('n_total', '?')}",
        f"  N_min: {bcfg.get('n_min', '?')}",
        f"  K_max: {bcfg.get('k_max', '?')}",
        f"  V_H_max: {bcfg.get('v_h_max', '?')}",
        "",
        "— Physics —",
        f"  g range: {gcfg.get('g_range', '?')}",
        f"  T range: {gcfg.get('T_range_s', '?')} s",
        f"  Bp range: {gcfg.get('Bp_range_kTm', '?')} kT/m",
        f"  prec: {gcfg.get('prec', '?')}",
    ]
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))


# ---------------------------------------------------------------------------
# Master plot
# ---------------------------------------------------------------------------

def make_full_report(
    losses: Optional[np.ndarray],
    eval_data: Optional[np.ndarray],
    config: Optional[dict],
    interval_save: int,
    eval_ext: Optional[dict] = None,
    out_path: str = "multi_pf_report.png",
):
    """Generate the full multi-panel report figure."""

    has_train = losses is not None and len(losses) > 0
    has_eval = eval_data is not None and len(eval_data) > 0
    has_ext = eval_ext is not None and "mse" in eval_ext
    has_config = config is not None

    n_rows = 3 + (2 if has_ext else 0)  # extra 2 rows for MSE/RMSE panels
    fig = plt.figure(figsize=(18, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, hspace=0.40, wspace=0.30)

    title = "Multi-PF Bank — Training & Evaluation Report"
    if has_config:
        title += f"  [{config.get('run_profile', '')} / {config.get('noise_mode', '')}]"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    row = 0

    # ---- Row 0: Training ----
    if has_train:
        ax1 = fig.add_subplot(gs[row, 0:2])
        plot_training_loss(ax1, losses, interval_save)
        ax2 = fig.add_subplot(gs[row, 2])
        plot_training_loss_log(ax2, losses, interval_save)
    else:
        ax_empty = fig.add_subplot(gs[row, :])
        ax_empty.text(0.5, 0.5, "No training data", ha="center", va="center",
                      fontsize=14, color=C_GRAY)
        ax_empty.axis("off")
    row += 1

    # ---- Row 1: V_H evaluation ----
    if has_eval:
        ax3 = fig.add_subplot(gs[row, 0])
        plot_eval_bars(ax3, eval_data)
        ax4 = fig.add_subplot(gs[row, 1])
        plot_eval_histogram(ax4, eval_data)
        ax5 = fig.add_subplot(gs[row, 2])
        plot_eval_sorted(ax5, eval_data)
    else:
        ax_empty = fig.add_subplot(gs[row, :])
        ax_empty.text(0.5, 0.5, "No evaluation data", ha="center", va="center",
                      fontsize=14, color=C_GRAY)
        ax_empty.axis("off")
    row += 1

    # ---- Row 2-3: MSE / RMSE / scatter (extended eval) ----
    if has_ext:
        ax_rmse = fig.add_subplot(gs[row, 0])
        plot_mse_rmse_bars(ax_rmse, eval_ext)
        ax_scatter = fig.add_subplot(gs[row, 1])
        plot_g_true_vs_g_hat(ax_scatter, eval_ext)
        ax_errhist = fig.add_subplot(gs[row, 2])
        plot_error_histogram(ax_errhist, eval_ext)
        row += 1

        ax_vhmse = fig.add_subplot(gs[row, 0])
        plot_vh_vs_mse_scatter(ax_vhmse, eval_ext)

        # Extended stats panel
        ax_ext_stats = fig.add_subplot(gs[row, 1])
        ax_ext_stats.axis("off")
        rmse_all = eval_ext["rmse"]
        mse_all = eval_ext["mse"]
        g_true_all = eval_ext["g_true"]
        g_hat_all = eval_ext["g_hat"]
        global_rmse = np.sqrt(np.mean((g_true_all - g_hat_all) ** 2))
        global_bias = np.mean(g_hat_all - g_true_all)
        ext_lines = [
            "═══ MSE / RMSE Statistics ═══",
            "",
            f"  Mean MSE:     {np.mean(mse_all):.2e} (m/s\u00b2)\u00b2",
            f"  Mean RMSE:    {np.mean(rmse_all)*1e3:.3f} mm/s\u00b2",
            f"  Median RMSE:  {np.median(rmse_all)*1e3:.3f} mm/s\u00b2",
            f"  Min RMSE:     {np.min(rmse_all)*1e3:.3f} mm/s\u00b2",
            f"  Max RMSE:     {np.max(rmse_all)*1e3:.3f} mm/s\u00b2",
            "",
            f"  Global RMSE:  {global_rmse:.2e} m/s\u00b2",
            f"               ({global_rmse*1e3:.3f} mm/s\u00b2)",
            f"  Global bias:  {global_bias:.2e} m/s\u00b2",
            "",
            f"  N samples:    {len(g_true_all)}",
            f"  g range:      [{g_true_all.min():.4f}, {g_true_all.max():.4f}]",
        ]
        ax_ext_stats.text(
            0.05, 0.95, "\n".join(ext_lines), transform=ax_ext_stats.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#bfdbfe"),
        )
        row += 1

    # ---- Last row: Quality buckets + V_H stats + config ----
    if has_eval:
        ax6 = fig.add_subplot(gs[row, 0])
        plot_eval_quality_buckets(ax6, eval_data)

        ax7 = fig.add_subplot(gs[row, 1])
        ax7.axis("off")
        stats_lines = [
            "═══ V_H Statistics ═══",
            "",
            f"  Episodes:  {len(eval_data)}",
            f"  Mean V_H:  {np.mean(eval_data):.4f}",
            f"  Median:    {np.median(eval_data):.4f}",
            f"  Std:       {np.std(eval_data):.4f}",
            f"  Min:       {np.min(eval_data):.4f}",
            f"  Max:       {np.max(eval_data):.4f}",
            f"  P25:       {np.percentile(eval_data, 25):.4f}",
            f"  P75:       {np.percentile(eval_data, 75):.4f}",
            "",
            f"  % below 0.1:  {100*np.mean(eval_data<0.1):.0f}%",
            f"  % below 0.5:  {100*np.mean(eval_data<0.5):.0f}%",
            f"  % below 1.0:  {100*np.mean(eval_data<1.0):.0f}%",
        ]
        if has_train:
            stats_lines += [
                "",
                "═══ Training ═══",
                f"  First loss:  {losses[0]:.4f}",
                f"  Last loss:   {losses[-1]:.4f}",
                f"  Best loss:   {np.min(losses):.4f}",
                f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%",
            ]
        ax7.text(0.05, 0.95, "\n".join(stats_lines), transform=ax7.transAxes,
                 fontsize=9, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fdf4", edgecolor="#bbf7d0"))
    else:
        ax6 = fig.add_subplot(gs[row, 0:2])
        ax6.axis("off")

    if has_config:
        ax8 = fig.add_subplot(gs[row, 2])
        plot_config_summary(ax8, config)
    else:
        ax8 = fig.add_subplot(gs[row, 2])
        ax8.axis("off")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[ok] Saved report to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Multi-PF Bank results")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Path to run output directory (auto-discovers files)")
    parser.add_argument("--history", default=None, help="Path to loss history CSV")
    parser.add_argument("--eval", default=None, help="Path to eval .npy file")
    parser.add_argument("--eval-ext", default=None, help="Path to eval_*_extended.npz file")
    parser.add_argument("--config", default=None, help="Path to run_config.json")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--interval", type=int, default=50,
                        help="Checkpoint save interval (iterations per CSV row)")
    args = parser.parse_args()

    losses = None
    eval_data = None
    eval_ext = None
    config = None
    interval_save = args.interval

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"[error] Directory not found: {run_dir}")
            sys.exit(1)
        files = find_files(run_dir)
        if "history" in files:
            losses = load_loss_history(files["history"])
            print(f"[ok] Loaded training history: {len(losses)} checkpoints from {files['history'].name}")
        if "eval_ext" in files:
            eval_ext = load_eval_extended(files["eval_ext"])
            eval_data = eval_ext["v_h"]
            print(f"[ok] Loaded extended eval: {len(eval_data)} episodes from {files['eval_ext'].name}")
        elif "eval" in files:
            eval_data = load_eval(files["eval"])
            print(f"[ok] Loaded eval data: {len(eval_data)} episodes from {files['eval'].name}")
        if "config" in files:
            config = load_config(files["config"])
            interval_save = config.get("profile", {}).get("interval_save", interval_save)
            print(f"[ok] Loaded config from {files['config'].name}")
        out_path = args.out or str(run_dir / "report.png")
    else:
        if args.history:
            losses = load_loss_history(Path(args.history))
        if args.eval_ext:
            eval_ext = load_eval_extended(Path(args.eval_ext))
            eval_data = eval_ext["v_h"]
        elif args.eval:
            eval_data = load_eval(Path(args.eval))
        if args.config:
            config = load_config(Path(args.config))
            interval_save = config.get("profile", {}).get("interval_save", interval_save)
        out_path = args.out or "multi_pf_report.png"

    if losses is None and eval_data is None:
        print("[error] No data found. Provide a run directory or --history / --eval paths.")
        sys.exit(1)

    make_full_report(losses, eval_data, config, interval_save, eval_ext=eval_ext, out_path=out_path)


if __name__ == "__main__":
    main()
