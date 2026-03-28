from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Small helpers
# =============================================================================

def _latest_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _load_run_config(out_dir: Path) -> dict:
    cfg_path = out_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_interval_save(run_cfg: dict) -> int:
    try:
        return max(1, int(run_cfg.get("interval_save", 1)))
    except Exception:
        return 1


def _infer_train_baseline(run_cfg: dict) -> Optional[float]:
    """
    For gravity-only training:
        baseline = Var_uniform(g_range) * train_g_loss_scale
    """
    try:
        if str(run_cfg.get("objective_mode", "")).strip().lower() != "gravity_only":
            return None
        g_lo, g_hi = run_cfg["gravimeter_config"]["g_range"]
        scale = float(run_cfg["train_g_loss_scale"])
        var_uniform = (float(g_hi) - float(g_lo)) ** 2 / 12.0
        baseline = var_uniform * scale
        return baseline if isfinite(baseline) else None
    except Exception:
        return None


def _infer_eval_baseline(run_cfg: dict) -> Optional[float]:
    """
    Prior variance baseline for MSE_g:
        Var_uniform(g_range)
    """
    try:
        g_lo, g_hi = run_cfg["gravimeter_config"]["g_range"]
        var_uniform = (float(g_hi) - float(g_lo)) ** 2 / 12.0
        return var_uniform if isfinite(var_uniform) else None
    except Exception:
        return None


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=max(2, span), adjust=False).mean().to_numpy()


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=max(1, window), center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def _cumulative_min(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    cur = np.inf
    for i, v in enumerate(values):
        cur = min(cur, float(v))
        out[i] = cur
    return out


def _maybe_log_y(ax: plt.Axes, values: np.ndarray) -> None:
    if len(values) and np.all(np.asarray(values) > 0.0):
        ax.set_yscale("log")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# File discovery
# =============================================================================

def _find_history_csv(out_dir: Path) -> Optional[Path]:
    # Canonical local copy if you save one manually
    direct = out_dir / "training_history.csv"
    if direct.exists():
        return direct

    # qsensoropt / trainer-style history in run root
    hist = _latest_matching_file(out_dir, "*_history.csv")
    if hist is not None:
        return hist

    # fallback recursive
    candidates = list(out_dir.rglob("*_history.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_eval_csv(out_dir: Path) -> Optional[Path]:
    direct = out_dir / "eval" / "branchbank_eval.csv"
    if direct.exists():
        return direct

    alt = _latest_matching_file(out_dir / "eval", "*_eval.csv") if (out_dir / "eval").exists() else None
    if alt is not None:
        return alt

    # fallback recursive
    candidates = list(out_dir.rglob("*_eval.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_controls_csv(out_dir: Path) -> Optional[Path]:
    direct = out_dir / "controls" / "branchbank_controls.csv"
    if direct.exists():
        return direct

    alt = _latest_matching_file(out_dir / "controls", "*_ext.csv") if (out_dir / "controls").exists() else None
    if alt is not None:
        return alt

    # fallback recursive
    candidates = list(out_dir.rglob("*_ext.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =============================================================================
# Training-loss plotting
# =============================================================================

def _prepare_training_history_df(df: pd.DataFrame, interval_save: int) -> pd.DataFrame:
    out = df.copy()

    if "Loss" not in out.columns:
        raise ValueError("Training history CSV must contain a 'Loss' column.")

    if "Checkpoint" not in out.columns:
        out["Checkpoint"] = np.arange(1, len(out) + 1, dtype=np.int64)

    out["UpdateStep"] = out["Checkpoint"] * int(interval_save)

    span = max(3, min(15, len(out) // 6 if len(out) >= 6 else 3))
    roll = max(3, min(11, span))

    out["Loss_EMA"] = _ema(out["Loss"].to_numpy(dtype=float), span=span)
    out["Loss_RollMedian"] = _rolling_median(out["Loss"].to_numpy(dtype=float), window=roll)
    out["Loss_BestSoFar"] = _cumulative_min(out["Loss"].to_numpy(dtype=float))
    return out


def plot_training_history(out_dir: Path, plots_dir: Path) -> Optional[Path]:
    history_csv = _find_history_csv(out_dir)
    if history_csv is None:
        return None

    run_cfg = _load_run_config(out_dir)
    interval_save = _infer_interval_save(run_cfg)

    df = _safe_read_csv(history_csv)
    if df.empty:
        return None

    dfp = _prepare_training_history_df(df, interval_save)
    dfp.to_csv(plots_dir / "training_loss_summary.csv", index=False)

    # Linear plot: EMA only
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(
        dfp["UpdateStep"],
        dfp["Loss_EMA"],
        linewidth=2.6,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)
    _savefig(fig, plots_dir / "training_loss.png")

    # Log-y plot: EMA only
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(
        dfp["UpdateStep"],
        dfp["Loss_EMA"],
        linewidth=2.6,
    )
    _maybe_log_y(ax, dfp["Loss_EMA"].to_numpy(dtype=float))
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)
    _savefig(fig, plots_dir / "training_loss_log.png")

    return history_csv


# =============================================================================
# Eval plotting
# =============================================================================

def plot_eval_curve(out_dir: Path, plots_dir: Path) -> Optional[Path]:
    eval_csv = _find_eval_csv(out_dir)
    if eval_csv is None:
        return None

    run_cfg = _load_run_config(out_dir)
    eval_baseline = _infer_eval_baseline(run_cfg)

    df = _safe_read_csv(eval_csv)
    if df.empty or "Resources" not in df.columns:
        return None

    y_col = "MSE_g" if "MSE_g" in df.columns else [c for c in df.columns if c != "Resources"][0]
    vals = df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(df["Resources"], vals, linewidth=2.0, marker="o", markersize=2.5)
    if eval_baseline is not None:
        ax.axhline(eval_baseline, linewidth=1.2, linestyle="-.", label=f"Prior baseline ≈ {eval_baseline:.4g}")
        ax.legend(fontsize=8)
    _maybe_log_y(ax, vals)
    ax.set_title(f"{y_col} vs resources")
    ax.set_xlabel("Resources")
    ax.set_ylabel(y_col)
    ax.grid(True, which="both", alpha=0.3)
    _savefig(fig, plots_dir / f"eval_{y_col}_vs_resources.png")

    # Improvement factor if baseline exists
    if eval_baseline is not None and len(vals) > 0 and vals[-1] > 0.0:
        improvement = eval_baseline / vals[-1]
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        ax.plot(df["Resources"], eval_baseline / vals, linewidth=2.0)
        _maybe_log_y(ax, (eval_baseline / vals))
        ax.set_title(f"Improvement factor vs resources (final ≈ {improvement:.3g}×)")
        ax.set_xlabel("Resources")
        ax.set_ylabel("Baseline / MSE")
        ax.grid(True, which="both", alpha=0.3)
        _savefig(fig, plots_dir / "eval_improvement_factor_vs_resources.png")

    return eval_csv


# =============================================================================
# Control-history plotting
# =============================================================================

def _sort_controls_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ResOverMaxRes" in df.columns:
        return df.sort_values(["ResOverMaxRes", "StepOverMaxStep"], kind="mergesort").reset_index(drop=True)
    if "StepOverMaxStep" in df.columns:
        return df.sort_values(["StepOverMaxStep"], kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def plot_controls_vs_progress(df: pd.DataFrame, plots_dir: Path) -> None:
    progress_col = "ResOverMaxRes" if "ResOverMaxRes" in df.columns else (
        "StepOverMaxStep" if "StepOverMaxStep" in df.columns else None
    )
    if progress_col is None:
        return

    for col in ["T_s", "Bp_kTm", "mw_phase_rad"]:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        ax.scatter(df[progress_col], df[col], s=5, alpha=0.25)
        ax.set_title(f"{col} vs {progress_col}")
        ax.set_xlabel(progress_col)
        ax.set_ylabel(col)
        ax.grid(True, which="both", alpha=0.3)
        _savefig(fig, plots_dir / f"controls_{col}_vs_{progress_col}.png")

    # Combined line plot using binned medians
    bins = np.linspace(float(df[progress_col].min()), float(df[progress_col].max()), 41)
    if len(np.unique(bins)) > 2:
        out = df[[progress_col] + [c for c in ["T_s", "Bp_kTm", "mw_phase_rad"] if c in df.columns]].copy()
        out["bin"] = pd.cut(out[progress_col], bins=bins, include_lowest=True, duplicates="drop")
        med = out.groupby("bin", observed=False).median(numeric_only=True)
        ctr = np.array([interval.mid for interval in med.index], dtype=float)

        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        for col in ["T_s", "Bp_kTm", "mw_phase_rad"]:
            if col in med.columns:
                ax.plot(ctr, med[col].to_numpy(dtype=float), linewidth=2.0, label=col)
        ax.set_title(f"Median controls vs {progress_col}")
        ax.set_xlabel(progress_col)
        ax.set_ylabel("Control value")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        _savefig(fig, plots_dir / f"controls_median_vs_{progress_col}.png")


def plot_control_histograms(df: pd.DataFrame, plots_dir: Path, bins: int = 60) -> None:
    cols = [c for c in ["T_s", "Bp_kTm", "mw_phase_rad"] if c in df.columns]
    if not cols:
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(7.2, 2.6 * len(cols)), dpi=180)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.hist(df[col].to_numpy(dtype=float), bins=bins, alpha=0.85)
        ax.set_title(f"{col} histogram")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, which="both", alpha=0.25)

    _savefig(fig, plots_dir / "controls_histograms.png")


def plot_posterior_metrics(df: pd.DataFrame, plots_dir: Path) -> None:
    progress_col = "ResOverMaxRes" if "ResOverMaxRes" in df.columns else (
        "StepOverMaxStep" if "StepOverMaxStep" in df.columns else None
    )
    if progress_col is None:
        return

    metric_groups = [
        ["Std_g", "Std_A"],
        ["BranchEntropy", "BranchDominance", "QGap12"],
        ["Top1Mass", "Top2Mass"],
        ["Mean_g", "Mean_A"],
    ]

    for group in metric_groups:
        cols = [c for c in group if c in df.columns]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        for col in cols:
            ax.scatter(df[progress_col], df[col], s=4, alpha=0.18, label=col)

            # Add binned median trend
            bins = np.linspace(float(df[progress_col].min()), float(df[progress_col].max()), 41)
            tmp = df[[progress_col, col]].dropna().copy()
            tmp["bin"] = pd.cut(tmp[progress_col], bins=bins, include_lowest=True, duplicates="drop")
            med = tmp.groupby("bin", observed=False)[col].median()
            ctr = np.array([interval.mid for interval in med.index], dtype=float)
            ax.plot(ctr, med.to_numpy(dtype=float), linewidth=2.0)

        ax.set_title(f"Posterior metrics vs {progress_col}")
        ax.set_xlabel(progress_col)
        ax.set_ylabel("Value")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        _savefig(fig, plots_dir / f"posterior_{'_'.join(cols)}_vs_{progress_col}.png")


def plot_branch_masses(df: pd.DataFrame, plots_dir: Path) -> None:
    progress_col = "ResOverMaxRes" if "ResOverMaxRes" in df.columns else (
        "StepOverMaxStep" if "StepOverMaxStep" in df.columns else None
    )
    mass_cols = [c for c in df.columns if c.startswith("Branch") and c.endswith("_Mass")]
    if progress_col is None or not mass_cols:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    for col in mass_cols:
        ax.scatter(df[progress_col], df[col], s=4, alpha=0.14, label=col)

        bins = np.linspace(float(df[progress_col].min()), float(df[progress_col].max()), 41)
        tmp = df[[progress_col, col]].dropna().copy()
        tmp["bin"] = pd.cut(tmp[progress_col], bins=bins, include_lowest=True, duplicates="drop")
        med = tmp.groupby("bin", observed=False)[col].median()
        ctr = np.array([interval.mid for interval in med.index], dtype=float)
        ax.plot(ctr, med.to_numpy(dtype=float), linewidth=1.8)

    ax.set_title(f"Branch masses vs {progress_col}")
    ax.set_xlabel(progress_col)
    ax.set_ylabel("Branch mass")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, plots_dir / "branch_masses_vs_progress.png")


def plot_controls_summary(out_dir: Path, plots_dir: Path, bins: int = 60) -> Optional[Path]:
    controls_csv = _find_controls_csv(out_dir)
    if controls_csv is None:
        return None

    df = _safe_read_csv(controls_csv)
    if df.empty:
        return None

    df = _sort_controls_df(df)
    df.to_csv(plots_dir / "controls_summary.csv", index=False)

    plot_controls_vs_progress(df, plots_dir)
    plot_control_histograms(df, plots_dir, bins=bins)
    plot_posterior_metrics(df, plots_dir)
    plot_branch_masses(df, plots_dir)

    return controls_csv


# =============================================================================
# Summary writer
# =============================================================================

def write_plot_summary(out_dir: Path, plots_dir: Path) -> Path:
    run_cfg = _load_run_config(out_dir)

    summary: dict[str, object] = {
        "out_dir": str(out_dir),
        "files": {},
        "metrics": {},
        "config": {},
    }

    history_csv = _find_history_csv(out_dir)
    eval_csv = _find_eval_csv(out_dir)
    controls_csv = _find_controls_csv(out_dir)

    if history_csv is not None and history_csv.exists():
        df = pd.read_csv(history_csv)
        if not df.empty and "Loss" in df.columns:
            summary["files"]["training_history_csv"] = str(history_csv)
            summary["metrics"]["train_loss_first"] = float(df["Loss"].iloc[0])
            summary["metrics"]["train_loss_last"] = float(df["Loss"].iloc[-1])
            summary["metrics"]["train_loss_min"] = float(df["Loss"].min())

    if eval_csv is not None and eval_csv.exists():
        df = pd.read_csv(eval_csv)
        if not df.empty and "Resources" in df.columns:
            y_col = "MSE_g" if "MSE_g" in df.columns else [c for c in df.columns if c != "Resources"][0]
            summary["files"]["eval_csv"] = str(eval_csv)
            summary["metrics"]["eval_metric_name"] = y_col
            summary["metrics"]["eval_first"] = float(df[y_col].iloc[0])
            summary["metrics"]["eval_last"] = float(df[y_col].iloc[-1])
            summary["metrics"]["eval_min"] = float(df[y_col].min())
            summary["metrics"]["eval_resources_last"] = float(df["Resources"].iloc[-1])

            baseline = _infer_eval_baseline(run_cfg)
            if baseline is not None and df[y_col].iloc[-1] > 0.0:
                summary["metrics"]["eval_prior_baseline"] = float(baseline)
                summary["metrics"]["eval_improvement_factor_last"] = float(baseline / float(df[y_col].iloc[-1]))

    if controls_csv is not None and controls_csv.exists():
        df = pd.read_csv(controls_csv)
        if not df.empty:
            summary["files"]["controls_csv"] = str(controls_csv)
            for col in ["T_s", "Bp_kTm", "mw_phase_rad", "Std_g", "BranchEntropy", "BranchDominance", "QGap12"]:
                if col in df.columns:
                    summary["metrics"][f"{col}_mean"] = float(df[col].mean())
                    summary["metrics"][f"{col}_median"] = float(df[col].median())

    # Minimal config echo for quick inspection
    if run_cfg:
        summary["config"]["run_profile"] = run_cfg.get("run_profile")
        summary["config"]["batchsize"] = run_cfg.get("batchsize")
        summary["config"]["iterations"] = run_cfg.get("iterations")
        summary["config"]["interval_save"] = run_cfg.get("interval_save")
        summary["config"]["max_steps"] = run_cfg.get("max_steps")
        summary["config"]["max_resources"] = run_cfg.get("max_resources")
        summary["config"]["objective_mode"] = run_cfg.get("objective_mode")
        summary["config"]["eval_metric_mode"] = run_cfg.get("eval_metric_mode")

    out_path = plots_dir / "plot_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


# =============================================================================
# Public entry point
# =============================================================================

def plot_branchbank_run(out_dir: str | Path, bins: int = 60) -> Path:
    """
    Main plotting entry point used by trainer.py.

    Expected file layout:
      out_dir/
        run_config.json
        *_history.csv
        controls/branchbank_controls.csv
        eval/branchbank_eval.csv

    Returns:
      Path to the plots directory.
    """
    out_dir = Path(out_dir)
    plots_dir = _ensure_dir(out_dir / "plots")

    # Training history
    try:
        plot_training_history(out_dir, plots_dir)
    except Exception as exc:
        print(f"[warn] Could not plot training history: {exc}")

    # Eval curve
    try:
        plot_eval_curve(out_dir, plots_dir)
    except Exception as exc:
        print(f"[warn] Could not plot evaluation curve: {exc}")

    # Controls / posterior summaries
    try:
        plot_controls_summary(out_dir, plots_dir, bins=bins)
    except Exception as exc:
        print(f"[warn] Could not plot controls summary: {exc}")

    # Summary JSON
    try:
        write_plot_summary(out_dir, plots_dir)
    except Exception as exc:
        print(f"[warn] Could not write plot summary: {exc}")

    return plots_dir