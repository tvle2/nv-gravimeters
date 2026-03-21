# gravity_plotting.py
from __future__ import annotations

"""
Plotting utilities for the branch-bank gravimeter runs.

Expected canonical files produced by gravimeter_model.py:
    run_dir/training_history.csv
    run_dir/controls/branchbank_controls.csv
    run_dir/eval/branchbank_eval.csv
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _prefer_file(preferred: Path, fallback_patterns: Iterable[str]) -> Optional[Path]:
    if preferred.exists():
        return preferred
    for pattern in fallback_patterns:
        matches = sorted(preferred.parent.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resource_binned_mean(df: pd.DataFrame, x: str, y: str, bins: int = 40) -> pd.DataFrame:
    if x not in df.columns or y not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=[x, y])

    xvals = df[x].to_numpy(dtype=float)
    yvals = df[y].to_numpy(dtype=float)

    finite = np.isfinite(xvals) & np.isfinite(yvals)
    xvals = xvals[finite]
    yvals = yvals[finite]
    if len(xvals) == 0:
        return pd.DataFrame(columns=[x, y])

    if np.allclose(np.min(xvals), np.max(xvals)):
        return pd.DataFrame({x: [np.mean(xvals)], y: [np.mean(yvals)]})

    edges = np.linspace(np.min(xvals), np.max(xvals), bins + 1)
    ids = np.digitize(xvals, edges) - 1
    ids = np.clip(ids, 0, bins - 1)

    out = pd.DataFrame({x: xvals, y: yvals, "_bin": ids})
    out = out.groupby("_bin")[[x, y]].mean().reset_index(drop=True)
    return out.sort_values(x).reset_index(drop=True)


def _branch_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    out = []
    idx = 1
    while True:
        col = f"Branch{idx}_{suffix}"
        if col in df.columns:
            out.append(col)
            idx += 1
        else:
            break
    return out


def _save_line(df: pd.DataFrame, x: str, y: str, out_path: Path, *, logy: bool = True, title: str = "") -> None:
    if x not in df.columns or y not in df.columns or len(df) == 0:
        return

    plot_df = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=180)
    ax.plot(plot_df[x], plot_df[y], linewidth=1.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_scatter(df: pd.DataFrame, x: str, y: str, c: str, out_path: Path, *, title: str = "") -> None:
    if not {x, y, c}.issubset(df.columns) or len(df) == 0:
        return

    plot_df = df[[x, y, c]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=180)
    sc = ax.scatter(plot_df[x], plot_df[y], c=plot_df[c], s=10, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(c)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_multiline(
    curves: list[pd.DataFrame],
    labels: list[str],
    x: str,
    y: str,
    out_path: Path,
    *,
    title: str = "",
) -> None:
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    plotted_any = False

    for df, label in zip(curves, labels):
        if x in df.columns and y in df.columns and len(df) > 0:
            plot_df = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(plot_df) > 0:
                ax.plot(plot_df[x], plot_df[y], linewidth=1.4, label=label)
                plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_branchbank_run(run_dir: str | Path, *, bins: int = 40) -> Path:
    run_dir = Path(run_dir)
    plot_dir = _ensure_dir(run_dir / "plots")

    history_csv = _prefer_file(run_dir / "training_history.csv", ["*_history.csv"])
    control_csv = _prefer_file(run_dir / "controls" / "branchbank_controls.csv", ["*_ext.csv", "*.csv"])
    eval_csv = _prefer_file(run_dir / "eval" / "branchbank_eval.csv", ["*_eval.csv", "*.csv"])

    # ---------------- history ----------------
    if history_csv is not None and history_csv.exists():
        hist_df = pd.read_csv(history_csv)
        if "Checkpoint" not in hist_df.columns:
            hist_df = hist_df.copy()
            hist_df["Checkpoint"] = np.arange(1, len(hist_df) + 1, dtype=np.int64)
        for candidate in ["Loss", "MeanLoss", hist_df.columns[1] if len(hist_df.columns) > 1 else None]:
            if candidate is not None and candidate in hist_df.columns:
                plot_df = hist_df[["Checkpoint", candidate]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(plot_df) == 0:
                    break

                y = plot_df[candidate].to_numpy(dtype=float)
                x = plot_df["Checkpoint"].to_numpy(dtype=float)

                smooth_window = max(3, min(9, len(plot_df) // 4))
                y_smooth = pd.Series(y).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

                fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=180)
                ax.plot(x, y, linewidth=1.0, alpha=0.35, label="raw")
                ax.plot(x, y_smooth, linewidth=2.0, label=f"{smooth_window}-pt moving avg")
                ax.set_xlabel("Checkpoint")
                ax.set_ylabel(candidate)
                ax.set_title("Training history")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
                fig.savefig(plot_dir / "training_loss.png")
                plt.close(fig)
                break
    # # ---------------- evaluation ----------------
    # if eval_csv is not None and eval_csv.exists():
    #     eval_df = pd.read_csv(eval_csv)
    #     if {"Resources", "Weighted MSE"}.issubset(eval_df.columns):
    #         mean_df = _resource_binned_mean(eval_df, "Resources", "Weighted MSE", bins=bins)
    #         _save_line(
    #             mean_df,
    #             "Resources",
    #             "Weighted MSE",
    #             plot_dir / "precision_vs_resources.png",
    #             logy=False,
    #             title="Weighted MSE vs resources",
    #         )

    # ---------------- evaluation ----------------
    if eval_csv is not None and eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)

        metric_col = None
        for candidate in ["MSE_g", "Weighted MSE"]:
            if {"Resources", candidate}.issubset(eval_df.columns):
                metric_col = candidate
                break

        if metric_col is not None:
            mean_df = _resource_binned_mean(eval_df, "Resources", metric_col, bins=bins)
            _save_line(
                mean_df,
                "Resources",
                metric_col,
                plot_dir / "precision_vs_resources.png",
                logy=True,
            )

    # ---------------- controls / branch summaries ----------------
    if control_csv is not None and control_csv.exists():
        ctrl_df = pd.read_csv(control_csv)

        if {"ResOverMaxRes", "T_s"}.issubset(ctrl_df.columns):
            df_t = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "T_s", bins=bins)
            _save_line(df_t, "ResOverMaxRes", "T_s", plot_dir / "T_vs_resources.png", title="Free-fall time vs resources")

        if {"ResOverMaxRes", "Bp_kTm"}.issubset(ctrl_df.columns):
            df_b = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "Bp_kTm", bins=bins)
            _save_line(df_b, "ResOverMaxRes", "Bp_kTm", plot_dir / "Bp_vs_resources.png", title="MFG vs resources")

        if {"ResOverMaxRes", "mw_phase_rad", "BranchDominance"}.issubset(ctrl_df.columns):
            _save_scatter(
                ctrl_df,
                "ResOverMaxRes",
                "mw_phase_rad",
                "BranchDominance",
                plot_dir / "mw_phase_scatter.png",
                title="Readout phase vs resources",
            )

        if {"ResOverMaxRes", "BranchEntropy"}.issubset(ctrl_df.columns):
            df_ent = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "BranchEntropy", bins=bins)
            _save_line(
                df_ent,
                "ResOverMaxRes",
                "BranchEntropy",
                plot_dir / "branch_entropy_vs_resources.png",
                title="Branch entropy vs resources",
            )

        if {"ResOverMaxRes", "QGap12"}.issubset(ctrl_df.columns):
            df_gap = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "QGap12", bins=bins)
            _save_line(
                df_gap,
                "ResOverMaxRes",
                "QGap12",
                plot_dir / "qgap12_vs_resources.png",
                title="Top-2 branch mass gap vs resources",
            )

        if {"ResOverMaxRes", "Std_g"}.issubset(ctrl_df.columns):
            df_std_g = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "Std_g", bins=bins)
            _save_line(
                df_std_g,
                "ResOverMaxRes",
                "Std_g",
                plot_dir / "std_g_vs_resources.png",
                title="Global posterior std(g) vs resources",
            )

        if {"ResOverMaxRes", "Mean_g"}.issubset(ctrl_df.columns):
            df_mean_g = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "Mean_g", bins=bins)
            _save_line(
                df_mean_g,
                "ResOverMaxRes",
                "Mean_g",
                plot_dir / "mean_g_vs_resources.png",
                title="Global posterior mean(g) vs resources",
            )

        if {"ResOverMaxRes", "Std_A"}.issubset(ctrl_df.columns):
            df_std_A = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "Std_A", bins=bins)
            _save_line(
                df_std_A,
                "ResOverMaxRes",
                "Std_A",
                plot_dir / "std_A_vs_resources.png",
                title="Global posterior std(A) vs resources",
            )

        if {"ResOverMaxRes", "Mean_A"}.issubset(ctrl_df.columns):
            df_mean_A = _resource_binned_mean(ctrl_df, "ResOverMaxRes", "Mean_A", bins=bins)
            _save_line(
                df_mean_A,
                "ResOverMaxRes",
                "Mean_A",
                plot_dir / "mean_A_vs_resources.png",
                title="Global posterior mean(A) vs resources",
            )

        if {"Mean_g", "Std_g", "BranchEntropy"}.issubset(ctrl_df.columns):
            _save_scatter(
                ctrl_df,
                "Mean_g",
                "Std_g",
                "BranchEntropy",
                plot_dir / "ambiguity_map.png",
                title="Ambiguity map",
            )

        if {"Mean_A", "Std_A", "BranchEntropy"}.issubset(ctrl_df.columns):
            _save_scatter(
                ctrl_df,
                "Mean_A",
                "Std_A",
                "BranchEntropy",
                plot_dir / "A_uncertainty_map.png",
                title="Visibility uncertainty map",
            )

        # branch mass curves
        branch_mass_cols = _branch_columns(ctrl_df, "Mass")
        if branch_mass_cols and "ResOverMaxRes" in ctrl_df.columns:
            curves = []
            labels = []
            for col in branch_mass_cols:
                tmp = _resource_binned_mean(
                    ctrl_df[["ResOverMaxRes", col]].rename(columns={col: "Mass"}),
                    "ResOverMaxRes",
                    "Mass",
                    bins=bins,
                )
                curves.append(tmp)
                labels.append(col)
            _save_multiline(
                curves,
                labels,
                "ResOverMaxRes",
                "Mass",
                plot_dir / "branch_masses.png",
                title="Branch masses vs resources",
            )

        # branch mean-g curves
        branch_mean_g_cols = _branch_columns(ctrl_df, "Mean_g")
        if branch_mean_g_cols and "ResOverMaxRes" in ctrl_df.columns:
            curves = []
            labels = []
            for col in branch_mean_g_cols:
                tmp = _resource_binned_mean(
                    ctrl_df[["ResOverMaxRes", col]].rename(columns={col: "Mean_g"}),
                    "ResOverMaxRes",
                    "Mean_g",
                    bins=bins,
                )
                curves.append(tmp)
                labels.append(col)
            _save_multiline(
                curves,
                labels,
                "ResOverMaxRes",
                "Mean_g",
                plot_dir / "branch_mean_g.png",
                title="Branch mean g vs resources",
            )

        # branch mean-A curves
        branch_mean_A_cols = _branch_columns(ctrl_df, "Mean_A")
        if branch_mean_A_cols and "ResOverMaxRes" in ctrl_df.columns:
            curves = []
            labels = []
            for col in branch_mean_A_cols:
                tmp = _resource_binned_mean(
                    ctrl_df[["ResOverMaxRes", col]].rename(columns={col: "Mean_A"}),
                    "ResOverMaxRes",
                    "Mean_A",
                    bins=bins,
                )
                curves.append(tmp)
                labels.append(col)
            _save_multiline(
                curves,
                labels,
                "ResOverMaxRes",
                "Mean_A",
                plot_dir / "branch_mean_A.png",
                title="Branch mean A vs resources",
            )

    return plot_dir