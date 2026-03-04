import os
import pandas as pd
import matplotlib.pyplot as plt

def format_sci_no_leading_zero(x: float, sig: int = 0) -> str:
    # Start from scientific notation like "1e-04"
    s = f"{x:.{sig}e}"
    # Turn "e-04" -> "e-4" and "e+04" -> "e+4"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s

len_episodes = 256
sigma_B_rel = 1e-4

sigma_str = format_sci_no_leading_zero(sigma_B_rel)   # "1e-4"
CSV_PATH = f"logs_{len_episodes}_sigmaB_{sigma_str}/train_metrics.csv"
OUT_DIR  = f"logs_{len_episodes}_sigmaB_{sigma_str}/plots"

# Set to 1 to disable smoothing
SMOOTH_WINDOW = 1

# Plot settings
DPI = 200


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Centered-ish rolling mean (with min periods so early points still show)."""
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def safe_log_yscale(ax, y_raw: pd.Series) -> None:
    """
    Use log scale only if there are positive values.
    (log scale will crash if all values are <= 0)
    """
    if (y_raw > 0).any():
        ax.set_yscale("log")


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    smooth_window: int = 1,
    use_log_y: bool = False,
) -> None:
    if x_col not in df.columns:
        raise ValueError(f"Missing required column '{x_col}' in {CSV_PATH}")
    if y_col not in df.columns:
        raise ValueError(f"Missing required column '{y_col}' in {CSV_PATH}")

    x = df[x_col]
    y_raw = df[y_col]
    y_smooth = rolling_mean(y_raw, smooth_window)

    fig, ax = plt.subplots()

    # Smoothed line
    ax.plot(x, y_smooth, label=f"{y_col} (smoothed)" if smooth_window > 1 else y_col)

    # Optional raw overlay for transparency
    if smooth_window > 1:
        ax.plot(x, y_raw, alpha=0.25, label=f"{y_col} (raw)")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if use_log_y:
        safe_log_yscale(ax, y_raw)

    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Defensive: remove rows with NaN in update (or metrics) if any
    if "update" not in df.columns:
        raise ValueError("CSV must contain an 'update' column.")

    # Sort by update so plots are always consistent
    df = df.sort_values("update").reset_index(drop=True)

    # --- Plot 1: RMSE vs update ---
    plot_metric(
        df=df,
        x_col="update",
        y_col="rmse_g",
        out_path=os.path.join(OUT_DIR, "rmse_g_vs_update.png"),
        title="Training RMSE(g) vs PPO update",
        xlabel="PPO update",
        ylabel="RMSE(g)",
        smooth_window=SMOOTH_WINDOW,
        use_log_y=False,  # set True if you want log-scale
    )

    # --- Plot 2: v_loss vs update ---
    plot_metric(
        df=df,
        x_col="update",
        y_col="v_loss",
        out_path=os.path.join(OUT_DIR, "v_loss_vs_update.png"),
        title="Training value loss (v_loss) vs PPO update",
        xlabel="PPO update",
        ylabel="v_loss",
        smooth_window=SMOOTH_WINDOW,
        use_log_y=True,  # v_loss often spans orders of magnitude
    )

    print("Saved plots:")
    print(f"  {os.path.join(OUT_DIR, 'rmse_g_vs_update.png')}")
    print(f"  {os.path.join(OUT_DIR, 'v_loss_vs_update.png')}")


if __name__ == "__main__":
    main()