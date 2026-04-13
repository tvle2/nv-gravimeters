"""Training script for the Hierarchical Binary Multi-PF quantum gravity sensor.

Wires the :class:`~gravimeter_hierarchical_pf.GravityHierarchicalSimulation`
into a custom eager training loop.

Usage
-----
::

    # Fast smoke test (verifies level transitions and loss decrease)
    RUN_PROFILE = "smoke"
    python trainer_hierarchical.py

    # Pilot run (2000 iterations, tuned for the 4-level binary tree)
    RUN_PROFILE = "pilot"
    python trainer_hierarchical.py

    # Train-only, no evaluation
    RUN_MODE = "train-only"
    python trainer_hierarchical.py

Configuration
-------------
Top-level flags ``RUN_PROFILE``, ``NOISE_MODE``, and ``RUN_MODE`` can be
edited to select the desired configuration without changing any other code.

Architecture summary
--------------------
* Physical model:  :class:`~gravimeter_model.GravityStatelessPhysicalModel`
* Bank:            :class:`~gravimeter_hierarchical_pf.HierarchicalPFBank`
* Simulation:      :class:`~gravimeter_hierarchical_pf.GravityHierarchicalSimulation`
* Controller:      MLP ``Dense(128, tanh) → Dense(128, tanh) → Dense(64, tanh)
  → Dense(3, tanh)`` (6 inputs → 3 controls)
* Loss:            Terminal MSE ``(g_hat - g_true)²``
* Training:        REINFORCE + model-aware gradients, Adam with inverse-sqrt
  learning-rate decay (Belliardo et al. 2024)

References
----------
Belliardo et al. (2024). Physical Review A 109, 062609.
  https://doi.org/10.1103/PhysRevA.109.062609
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Environment setup BEFORE importing TF — TF reads env vars at import time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("TF_DETERMINISTIC_OPS", None)  # deterministic SVD crashes for d=1
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tensorflow as tf
from tqdm.auto import trange


# ---------------------------------------------------------------------------
# Imports from local qsensoropt (workspace copies)
# ---------------------------------------------------------------------------

import importlib.util
import sys
import types


def _load_local_qsensoropt_module(module_name: str):
    """Load a qsensoropt module from the local workspace directory."""
    root = Path(__file__).resolve().parent
    pkg_name = "_qsensoropt_local"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root)]
        sys.modules[pkg_name] = pkg
    full_name = f"{pkg_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    path = root / f"{module_name}.py"
    if not path.exists():
        raise ImportError(f"Cannot find local qsensoropt module: {path}")
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
except Exception:
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay


from gravimeter_model import GravimeterConfig
from gravimeter_hierarchical_pf import (
    HierarchicalPFConfig,
    build_hierarchical_simulation,
)


# ---------------------------------------------------------------------------
# Top-level mode flags
# ---------------------------------------------------------------------------

#: Select training profile: ``"smoke"`` or ``"pilot"``.
RUN_PROFILE: str = "pilot"

#: ``"all"`` | ``"train-only"`` | ``"eval-only"``
RUN_MODE: str = "all"

#: ``"none"`` | ``"paper"``  (paper = realistic sensor noise from Belliardo 2024)
NOISE_MODE: str = "none"


# ---------------------------------------------------------------------------
# Run profile dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunProfile:
    """Complete specification for one hierarchical PF training run.

    Parameters
    ----------
    name : str
        Human-readable name used in log messages and file names.
    out_dir : str
        Output directory for checkpoints, loss history, and weights.
    batchsize : int
        Number of parallel estimation episodes per gradient step.
    iterations : int
        Total number of gradient update steps.
    interval_save : int
        Save a checkpoint every this many iterations.
    max_steps : int
        Maximum measurement steps per episode.
    max_resources : float
        Maximum resource budget per episode (seconds of measurement time).
    initial_lr : float
        Initial learning rate for :class:`~.InverseSqrtDecay`.
    seed : int
        Global random seed.
    gradient_accumulation : int
        Number of ``execute()`` calls per gradient update (effective
        batchsize multiplier via gradient averaging).

    n_particles : int
        Total particles per PF at each level.
    n_levels : int
        Number of binary disambiguation levels.
    n_disambig_per_level : int
        Measurements per disambiguation level.

    cumulative_loss : bool
        If ``True``, accumulate MSE at every step.
    baseline : bool
        If ``True``, subtract batch mean from REINFORCE loss term.
    loss_logl_outcomes : bool
        Mix outcome log-likelihood into the loss (required for REINFORCE).
    stop_gradient_input : bool
        Stop gradient through the NN input (saves memory).
    stop_gradient_pf : bool
        Stop gradient through the particle filter Bayes update.

    eval_iters : int
        Number of evaluation episodes for performance estimation.
    """

    name: str
    out_dir: str
    batchsize: int
    iterations: int
    interval_save: int
    max_steps: int
    max_resources: float
    initial_lr: float
    seed: int
    gradient_accumulation: int

    # Bank settings
    n_particles: int
    n_levels: int
    n_disambig_per_level: int

    # Training flags
    cumulative_loss: bool
    baseline: bool
    loss_logl_outcomes: bool
    stop_gradient_input: bool
    stop_gradient_pf: bool

    # Evaluation
    eval_iters: int


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

#: Smoke test — verifies level transitions and loss decrease quickly.
#: 4 levels × 7 steps = 28 disambiguation steps + 4 refining = 32 total.
#: With 16 smoke steps the agent only needs to learn 2 levels (14 steps).
SMOKE_PROFILE = RunProfile(
    name="smoke",
    out_dir="runs/gravity_hierarchical_smoke",
    batchsize=16,
    iterations=300,
    interval_save=50,
    max_steps=32,
    max_resources=0.04,        # 40 ms — sufficient for ~28 low-to-mid gain shots
    initial_lr=5e-4,
    seed=42,
    gradient_accumulation=1,

    n_particles=512,
    n_levels=4,
    n_disambig_per_level=7,

    cumulative_loss=False,     # terminal loss — stronger RL signal at start
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=16,
)

#: Pilot run — 2000 iterations, enough to see convergence in all 4 levels.
PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_hierarchical_pilot",
    batchsize=32,
    iterations=2000,
    interval_save=100,
    max_steps=32,
    max_resources=0.08,        # 80 ms total — covers 4 levels + refinement
    initial_lr=3e-4,
    seed=42,
    gradient_accumulation=1,

    n_particles=1024,
    n_levels=24,
    n_disambig_per_level=7,

    cumulative_loss=False,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=32,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def set_global_reproducibility(seed: int) -> None:
    """Set all global random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base random seed for Python, NumPy, and TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def get_profile() -> RunProfile:
    """Return the :class:`RunProfile` selected by the ``RUN_PROFILE`` flag.

    Returns
    -------
    RunProfile

    Raises
    ------
    ValueError
        If ``RUN_PROFILE`` is not ``"smoke"`` or ``"pilot"``.
    """
    key = RUN_PROFILE.strip().lower()
    if key == "smoke":
        return SMOKE_PROFILE
    if key == "pilot":
        return PILOT_PROFILE
    raise ValueError(
        f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Choose 'smoke' or 'pilot'."
    )


def make_gravimeter_cfg() -> GravimeterConfig:
    """Construct :class:`~gravimeter_model.GravimeterConfig` from ``NOISE_MODE``.

    Returns
    -------
    GravimeterConfig

    Raises
    ------
    ValueError
        If ``NOISE_MODE`` is not ``"none"`` or ``"paper"``.
    """
    common = dict(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=1.47e-17,
        hbar_J_s=1.054_571_817e-34,
        kT_to_T=1e3,
        g_range=(9.7806, 9.825),
        infer_mfg_bias=False,
        beta_B_range=(-0.10, 0.10),
        T_range_s=(10e-6, 1.2e-3),
        Bp_range_kTm=(0.5, 80.0),
        T2_spin_s=None,
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec="float32",
    )

    mode = NOISE_MODE.strip().lower()
    if mode == "none":
        return GravimeterConfig(
            **common,
            mfg_rel_noise_bound=0.0,
            mfg_noise_quad_points=1,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.0,
            trap_visibility_mode="none",
            trap_noise_quad_points=1,
        )
    if mode == "paper":
        return GravimeterConfig(
            **common,
            mfg_rel_noise_bound=0.025,
            mfg_noise_quad_points=9,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.01,
            trap_visibility_mode="exact_single_delta",
            trap_noise_quad_points=9,
        )
    raise ValueError(
        f"Unknown NOISE_MODE={NOISE_MODE!r}. Choose 'none' or 'paper'."
    )


def make_bank_cfg(profile: RunProfile) -> HierarchicalPFConfig:
    """Construct :class:`~gravimeter_hierarchical_pf.HierarchicalPFConfig`
    from a :class:`RunProfile`.

    Parameters
    ----------
    profile : RunProfile

    Returns
    -------
    HierarchicalPFConfig
    """
    return HierarchicalPFConfig(
        n_particles=profile.n_particles,
        n_levels=profile.n_levels,
        n_disambig_per_level=profile.n_disambig_per_level,
        prec="float32",
        resample_threshold=0.5,
        resample_alpha=0.5,
        resample_beta=0.98,
        scibior_trick=True,
        trim=True,
    )


def make_sim_pars(
    profile: RunProfile, cfg: GravimeterConfig
) -> SimulationParameters:
    """Construct :class:`~.SimulationParameters` from a :class:`RunProfile`.

    Parameters
    ----------
    profile : RunProfile
    cfg : GravimeterConfig

    Returns
    -------
    SimulationParameters
    """
    return SimulationParameters(
        sim_name=f"gravity_hierarchical_{profile.name}",
        num_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=1.0,
        prec=cfg.prec,
        stop_gradient_input=profile.stop_gradient_input,
        loss_logl_outcomes=profile.loss_logl_outcomes,
        loss_logl_controls=False,    # continuous controls only
        cumulative_loss=profile.cumulative_loss,
        baseline=profile.baseline,
        stop_gradient_pf=profile.stop_gradient_pf,
        log_loss=False,
        permutation_invariant=False,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_loss_history(
    out_dir: Path,
    sim_str: str,
    loss_history: list,
    interval_save: int,
) -> None:
    """Save the loss history as a CSV file.

    Parameters
    ----------
    out_dir : Path
    sim_str : str
    loss_history : list of float
    interval_save : int
    """
    import pandas as pd

    arr = np.array(loss_history)
    num_blocks = len(arr) // interval_save
    if num_blocks == 0:
        return
    arr_trimmed = arr[: num_blocks * interval_save]
    blocks = arr_trimmed.reshape(num_blocks, interval_save)
    mean_loss = blocks.mean(axis=1)
    df = pd.DataFrame({"Loss": mean_loss})
    csv_path = out_dir / f"{sim_str}_history.csv"
    df.to_csv(str(csv_path), index=False, float_format="%.4e")
    print(f"[ok] Saved loss history to {csv_path}")


def _save_run_config(
    out_dir: Path,
    *,
    profile: RunProfile,
    cfg: GravimeterConfig,
    bank_cfg: HierarchicalPFConfig,
) -> None:
    """Save a human-readable JSON run configuration file.

    Parameters
    ----------
    out_dir : Path
    profile : RunProfile
    cfg : GravimeterConfig
    bank_cfg : HierarchicalPFConfig
    """
    config = {
        "run_profile": profile.name,
        "run_mode": RUN_MODE,
        "noise_mode": NOISE_MODE,
        "profile": {
            "batchsize": profile.batchsize,
            "iterations": profile.iterations,
            "interval_save": profile.interval_save,
            "max_steps": profile.max_steps,
            "max_resources": profile.max_resources,
            "initial_lr": profile.initial_lr,
            "seed": profile.seed,
            "gradient_accumulation": profile.gradient_accumulation,
            "n_particles": profile.n_particles,
            "n_levels": profile.n_levels,
            "n_disambig_per_level": profile.n_disambig_per_level,
            "cumulative_loss": profile.cumulative_loss,
            "baseline": profile.baseline,
            "loss_logl_outcomes": profile.loss_logl_outcomes,
            "stop_gradient_input": profile.stop_gradient_input,
            "stop_gradient_pf": profile.stop_gradient_pf,
        },
        "gravimeter_cfg": {
            "g_range": list(cfg.g_range),
            "T_range_s": list(cfg.T_range_s),
            "Bp_range_kTm": list(cfg.Bp_range_kTm),
            "mfg_rel_noise_bound": cfg.mfg_rel_noise_bound,
            "sigma_omega_rel": cfg.sigma_omega_rel,
            "trap_visibility_mode": cfg.trap_visibility_mode,
            "T2_spin_s": cfg.T2_spin_s,
            "readout_flip_prob": cfg.readout_flip_prob,
            "prec": cfg.prec,
            "infer_mfg_bias": cfg.infer_mfg_bias,
        },
        "bank_cfg": {
            "n_particles": bank_cfg.n_particles,
            "n_levels": bank_cfg.n_levels,
            "n_disambig_per_level": bank_cfg.n_disambig_per_level,
            "resample_threshold": bank_cfg.resample_threshold,
            "resample_alpha": bank_cfg.resample_alpha,
            "resample_beta": bank_cfg.resample_beta,
        },
    }
    cfg_path = out_dir / "run_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[ok] Saved run config to {cfg_path}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(profile: RunProfile) -> None:
    """Run the full hierarchical PF training pipeline.

    Steps:

    1. Create output directory and save configuration JSON.
    2. Build :class:`~gravimeter_model.GravityStatelessPhysicalModel`,
       :class:`~gravimeter_hierarchical_pf.HierarchicalPFBank`,
       :class:`~gravimeter_hierarchical_pf.GravityHierarchicalSimulation`,
       and the MLP controller.
    3. Run the custom eager training loop:

       * For each iteration: call ``simulation.execute()`` inside
         ``tf.GradientTape``, accumulate gradients over
         ``gradient_accumulation`` steps, apply Adam update.
       * Log per-step loss, moving average, and learning rate to tqdm.
       * Save checkpoints every ``interval_save`` iterations.

    4. Save the best checkpoint and final weights.
    5. Save loss history as CSV.

    Notes
    -----
    We use a custom eager loop instead of ``utils.train()`` because the
    hierarchical bank's Python-level state management (list operations,
    ``tf.Variable`` in-place updates, ``.numpy()`` calls for level
    transitions) is incompatible with ``@tf.function`` tracing.

    Parameters
    ----------
    profile : RunProfile
        Fully specified training configuration.
    """
    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run profile  : {profile.name}")
    print(f"Noise mode   : {NOISE_MODE}")
    print(f"Run mode     : {RUN_MODE}")
    print(f"Output dir   : {out_dir.resolve()}")
    print("=" * 72)

    # --- Configuration ---
    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)

    rangen = tf.random.Generator.from_seed(profile.seed)

    # --- Build all components ---
    simulation, bank, controller = build_hierarchical_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    variables = controller.trainable_variables
    n_params = sum(int(np.prod(v.shape)) for v in variables)
    print(f"[info] Controller parameters: {n_params:,}")
    print(
        f"[info] Bank: N_particles={bank_cfg.n_particles}, "
        f"N_levels={bank_cfg.n_levels}, "
        f"N_disambig/level={bank_cfg.n_disambig_per_level}"
    )
    print(
        f"[info] Simulation: {simpars.num_steps} steps, "
        f"max_resources={simpars.max_resources:.3f} s, "
        f"batch={profile.batchsize}"
    )
    print(
        f"[info] g_range=[{cfg.g_range[0]:.4f}, {cfg.g_range[1]:.4f}], "
        f"width={cfg.g_range[1] - cfg.g_range[0]:.4e} m/s²"
    )
    print(
        f"[info] After {bank_cfg.n_levels} levels: "
        f"interval width ≈ "
        f"{(cfg.g_range[1] - cfg.g_range[0]) / 2**bank_cfg.n_levels:.4e} m/s²"
    )

    # --- Optimizer ---
    lr_schedule = InverseSqrtDecay(
        initial_learning_rate=profile.initial_lr,
        prec=cfg.prec,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # --- Checkpoint directory ---
    sim_str = str(simulation)
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Eager training loop ---
    # The hierarchical bank uses Python-level state manipulation (.numpy()
    # calls, list operations), which is incompatible with tf.function tracing.
    # We therefore run in eager mode, which is fully supported by TensorFlow
    # and still benefits from GPU acceleration for all tensor operations.
    loss_history: list = []

    def single_step_eager() -> float:
        """One gradient accumulation step (eager mode).

        Returns
        -------
        float
            Average loss over accumulation steps.
        """
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        for _ in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                loss_diff, loss = simulation.execute(rangen)
            grads = tape.gradient(loss_diff, variables)
            # Guard against None gradients (unused variables)
            grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, variables)
            ]
            acc_loss += float(loss.numpy())
            acc_grads = [ag + g for ag, g in zip(acc_grads, grads)]
        # Average gradients over accumulation steps
        acc_grads = [g / profile.gradient_accumulation for g in acc_grads]
        optimizer.apply_gradients(zip(acc_grads, variables))
        return acc_loss / profile.gradient_accumulation

    best_loss = float("inf")
    best_ckpt_idx = 1

    pbar = trange(profile.iterations, desc="Training", unit="step")
    for j in pbar:
        step_loss = single_step_eager()
        loss_history.append(step_loss)

        # Compute rolling average over last 10 steps
        recent = np.mean(loss_history[max(0, j - 9): j + 1])

        # Get current learning rate
        try:
            lr_val = float(lr_schedule(optimizer.iterations).numpy())
        except Exception:
            lr_val = float(optimizer.learning_rate.numpy())

        pbar.set_postfix(
            loss=f"{step_loss:.4f}",
            avg10=f"{recent:.4f}",
            best=f"{best_loss:.4f}",
            lr=f"{lr_val:.1e}",
        )

        # Save checkpoint every interval_save iterations
        if (j + 1) % profile.interval_save == 0:
            ckpt_idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / str(ckpt_idx)
            controller.save_weights(str(ckpt_path) + ".weights.h5")

            # Track best checkpoint by window-mean loss
            window_start = max(0, j + 1 - profile.interval_save)
            window_loss = float(
                np.mean(loss_history[window_start: j + 1])
            )
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = ckpt_idx

    pbar.close()
    print("[training] Done.")

    # --- Load best checkpoint ---
    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(
            f"[ok] Loaded best checkpoint (idx={best_ckpt_idx}, "
            f"window_loss={best_loss:.4f}) from {best_path}"
        )

    # Clean up checkpoint directory
    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass  # non-empty directory, leave it

    # --- Save final weights ---
    weights_path = (
        out_dir / f"gravity_hierarchical_{profile.name}_final.weights.h5"
    )
    controller.save_weights(str(weights_path))
    print(f"[ok] Saved controller weights to {weights_path}")

    # --- Save loss history CSV ---
    _save_loss_history(out_dir, sim_str, loss_history, profile.interval_save)

    # --- Save run config ---
    _save_run_config(out_dir, profile=profile, cfg=cfg, bank_cfg=bank_cfg)
    print(f"[done] All outputs saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    profile: RunProfile,
    weights_path: Optional[Path] = None,
) -> None:
    """Load a trained controller and evaluate its performance.

    Runs ``profile.eval_iters`` episodes in deploy mode and reports:

    * Mean MSE (m/s²)²
    * Root-mean-square error (RMSE) in m/s²
    * Final disambiguation level (confirms full hierarchy is traversed)
    * Mean mode-0 weight at end of last episode

    Parameters
    ----------
    profile : RunProfile
    weights_path : Path, optional
        Path to the ``.weights.h5`` file.  Defaults to
        ``{out_dir}/gravity_hierarchical_{name}_final.weights.h5``.
    """
    out_dir = Path(profile.out_dir)
    if weights_path is None:
        weights_path = (
            out_dir / f"gravity_hierarchical_{profile.name}_final.weights.h5"
        )

    if not weights_path.exists():
        print(
            f"[warn] Weights not found at {weights_path}. "
            "Skipping evaluation."
        )
        return

    print(f"\n[eval] Loading weights from {weights_path}")

    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)

    simulation, bank, controller = build_hierarchical_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )
    controller.load_weights(str(weights_path))
    print("[eval] Weights loaded.")

    all_mse: list = []
    all_g_true: list = []
    all_g_hat: list = []
    all_final_levels: list = []

    pbar = trange(profile.eval_iters, desc="Evaluating", unit="ep")
    for i in pbar:
        # Deploy mode returns true_values and history
        result = simulation.execute(rangen, deploy=True)
        true_values_tensor = result[0]  # (bs, 1, d)
        g_true = true_values_tensor[:, 0, 0].numpy()  # (bs,)

        # Bank state after this episode
        g_hat = bank.map_mode_mean().numpy()  # (bs,)
        mse_batch = float(np.mean((g_hat - g_true) ** 2))
        all_mse.append(mse_batch)
        all_g_true.append(g_true)
        all_g_hat.append(g_hat)
        all_final_levels.append(bank.current_level)

        pbar.set_postfix(
            RMSE=f"{np.sqrt(mse_batch):.2e}",
            level=f"{bank.current_level}/{bank_cfg.n_levels}",
            mean_RMSE=f"{np.sqrt(np.mean(all_mse)):.2e}",
        )
    pbar.close()

    all_mse_arr = np.array(all_mse)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)
    global_rmse = float(
        np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat) ** 2))
    )

    print(f"\n[eval] Results over {profile.eval_iters} episodes:")
    print(f"  Mean MSE:        {np.mean(all_mse_arr):.4e} (m/s²)²")
    print(f"  Mean RMSE:       {np.sqrt(np.mean(all_mse_arr)):.4e} m/s²")
    print(f"  Global RMSE:     {global_rmse:.4e} m/s²")
    print(
        f"  Final level:     {np.mean(all_final_levels):.1f} / "
        f"{bank_cfg.n_levels}"
    )

    # Save evaluation results
    eval_path = out_dir / f"eval_{profile.name}_mse.npy"
    np.save(str(eval_path), all_mse_arr)
    print(f"[ok] Saved MSE array to {eval_path}")

    eval_ext_path = out_dir / f"eval_{profile.name}_extended.npz"
    np.savez(
        str(eval_ext_path),
        mse=all_mse_arr,
        rmse=np.sqrt(all_mse_arr),
        g_true=all_g_true_flat,
        g_hat=all_g_hat_flat,
        final_levels=np.array(all_final_levels),
    )
    print(f"[ok] Saved extended eval (MSE, RMSE, g_true, g_hat) to {eval_ext_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: select profile, set reproducibility, run training/eval."""
    profile = get_profile()
    set_global_reproducibility(profile.seed)
    tf.keras.backend.clear_session()

    if RUN_MODE in {"all", "train-only"}:
        run_training(profile)

    if RUN_MODE in {"all", "eval-only"}:
        run_evaluation(profile)

    if RUN_MODE not in {"all", "train-only", "eval-only"}:
        raise ValueError(
            f"Unknown RUN_MODE={RUN_MODE!r}. "
            "Choose 'all', 'train-only', or 'eval-only'."
        )


if __name__ == "__main__":
    main()
