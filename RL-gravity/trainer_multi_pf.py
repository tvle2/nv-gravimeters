"""Training script for the Multi-PF Bank quantum gravity sensor.

Wires the :class:`~gravimeter_multi_pf.GravityMultiPFSimulation` into
qsensoropt's ``utils.train()`` training loop.

Usage
-----
::

    # Pilot run (fast, 2000 iterations, small batch)
    RUN_PROFILE = "pilot"
    python trainer_multi_pf.py

    # Full run (50000 iterations, larger batch)
    RUN_PROFILE = "full"
    python trainer_multi_pf.py

Configuration
-------------
All training hyperparameters are captured in frozen :class:`RunProfile`
dataclasses (:data:`PILOT_PROFILE`, :data:`FULL_PROFILE`).

The top-level flags ``RUN_PROFILE``, ``NOISE_MODE``, and ``RUN_MODE`` can be
edited to select the desired configuration without touching any other code.

Architecture summary
--------------------
* Physical model: :class:`~gravimeter_model.GravityStatelessPhysicalModel`
* Bank:           :class:`~gravimeter_multi_pf.MultiPFBank`
* Simulation:     :class:`~gravimeter_multi_pf.GravityMultiPFSimulation`
* Controller:     MLP ``Dense(128,tanh) → Dense(128,tanh) → Dense(64,tanh)
  → Dense(3,tanh)`` (4 * TOP_K_MODES + 5 → 3 controls)
* Loss:           Holevo variance ``V_H = |μ_H|^{-2} - 1`` (Berry & Sanders 2009)
* Training:       REINFORCE + model-aware gradients, Adam with inverse-sqrt
  learning-rate decay (Belliardo et al. 2024)

References
----------
Belliardo et al. (2024). Physical Review A 109, 062609.
  https://doi.org/10.1103/PhysRevA.109.062609

Berry & Sanders (2009). Physical Review A 80, 052114.
  https://arxiv.org/abs/0907.0014
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm, trange

# Suppress TF info/warning logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# NOTE: TF_DETERMINISTIC_OPS is intentionally NOT set.
# TF's deterministic SVD does not support matrices with 1 column,
# which triggers an UnimplementedError in qsensoropt's Liu-West
# jittering (sqrt_hmatrix → SVD) when d=1 (single-parameter g).
# Reproducibility is ensured via explicit seeds instead.
os.environ.pop("TF_DETERMINISTIC_OPS", None)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------------------
# Imports from qsensoropt and our modules
# ---------------------------------------------------------------------------

import importlib.util
import sys
import types
import json


def _load_local_qsensoropt_module(module_name: str):
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
    from qsensoropt.utils import train
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
except Exception:
    train = _load_local_qsensoropt_module("utils").train
    SimulationParameters = _load_local_qsensoropt_module(
        "simulation_parameters"
    ).SimulationParameters
    InverseSqrtDecay = _load_local_qsensoropt_module("schedulers").InverseSqrtDecay


from gravimeter_model_complete import GravimeterConfig
from gravimeter_multi_pf import (
    MultiPFBankConfig,
    build_gravity_multi_pf_simulation,
)


# ---------------------------------------------------------------------------
# Top-level mode flags
# ---------------------------------------------------------------------------

#: Select training profile: ``"smoke"``, ``"pilot"``, or ``"full"``.
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
    """Complete specification of a training run.

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
        Save checkpoint every this many iterations.
    max_steps : int
        Maximum number of measurement steps per episode.
    max_resources : float
        Maximum resource budget per episode (seconds of measurement time).
    initial_lr : float
        Initial learning rate for :class:`~.InverseSqrtDecay`.
    seed : int
        Global random seed.
    gradient_accumulation : int
        Number of ``execute()`` calls per gradient update (effective batchsize
        multiplier).

    n_total : int
        Total particle budget across all modes in the bank.
    n_min : int
        Minimum particles per active mode.
    k_max : int
        Maximum simultaneous modes.

    top_k_modes : int
        Number of top-weighted modes included in the controller input.
    v_h_max : float
        Holevo variance clipping threshold.

    cumulative_loss : bool
        If ``True``, accumulate loss at every step (recommended for
        coarse-to-fine learning).
    baseline : bool
        If ``True``, subtract the batch mean from the REINFORCE loss term.
    loss_logl_outcomes : bool
        Mix outcome log-likelihood into the loss (required for REINFORCE).
    stop_gradient_input : bool
        Stop gradient through the NN input (saves memory).
    stop_gradient_pf : bool
        Stop gradient through the particle filter Bayes update.

    eval_iters : int
        Number of evaluation batches for performance estimation.
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
    n_total: int
    n_min: int
    k_max: int

    # Controller / loss
    top_k_modes: int
    v_h_max: float

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

SMOKE_PROFILE = RunProfile(
    name="smoke",
    out_dir="runs/gravity_multi_pf_smoke",
    batchsize=8,
    iterations=300,
    interval_save=50,
    max_steps=16,             # short episodes → ~4× faster per iteration
    max_resources=0.02,       # 20 ms — enough for ~16 low-gain shots
    initial_lr=5e-4,          # higher LR for fast convergence
    seed=42,
    gradient_accumulation=1,  # no accumulation → 2× faster per iteration

    n_total=128,              # small PF → fast Bayes updates
    n_min=16,
    k_max=8,                  # budget: 128/16 = 8 modes max

    top_k_modes=4,
    v_h_max=100.0,

    cumulative_loss=False,    # terminal loss — stronger RL signal
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=16,
)

PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_multi_pf_pilot",
    batchsize=8,
    iterations=800,
    interval_save=50,
    max_steps=32,
    max_resources=0.08,       # 80 ms total measurement time
    initial_lr=3e-4,
    seed=42,
    gradient_accumulation=1,

    n_total=512,              # more particles for better per-mode ESS
    n_min=32,
    k_max=8,                  # budget: 512/32 = 16, but 8 is cleaner

    top_k_modes=4,
    v_h_max=100.0,

    cumulative_loss=False,    # terminal loss — stronger RL signal
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=32,
)

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravity_multi_pf_full",
    batchsize=64,
    iterations=50000,
    interval_save=256,
    max_steps=128,
    max_resources=0.16,       # 160 ms total measurement time
    initial_lr=1e-3,
    seed=123,
    gradient_accumulation=4,

    n_total=2048,
    n_min=32,
    k_max=64,

    top_k_modes=4,
    v_h_max=100.0,

    cumulative_loss=True,
    baseline=True,
    loss_logl_outcomes=True,
    stop_gradient_input=True,
    stop_gradient_pf=False,

    eval_iters=256,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _jsonl_to_json(jsonl_path: Path, json_path: Path) -> None:
    if not jsonl_path.exists():
        return
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def set_global_reproducibility(seed: int) -> None:
    """Set all global random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base random seed.
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
        If ``RUN_PROFILE`` is not a recognized key.
    """
    key = RUN_PROFILE.strip().lower()
    if key == "smoke":
        return SMOKE_PROFILE
    if key == "pilot":
        return PILOT_PROFILE
    if key == "full":
        return FULL_PROFILE
    raise ValueError(
        f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Choose 'smoke', 'pilot', or 'full'."
    )


def make_gravimeter_cfg() -> GravimeterConfig:
    """Construct the :class:`~gravimeter_model.GravimeterConfig` from
    the ``NOISE_MODE`` flag.

    Returns
    -------
    GravimeterConfig

    Raises
    ------
    ValueError
        If ``NOISE_MODE`` is not recognized.
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
        T_range_s=(10e-6, 1.0e-3),
        Bp_range_kTm=(0.5, 50.0),
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


def make_bank_cfg(profile: RunProfile) -> MultiPFBankConfig:
    """Construct :class:`~gravimeter_multi_pf.MultiPFBankConfig` from a
    :class:`RunProfile`.

    Parameters
    ----------
    profile : RunProfile

    Returns
    -------
    MultiPFBankConfig
    """
    return MultiPFBankConfig(
        n_total=profile.n_total,
        n_min=profile.n_min,
        k_max=profile.k_max,
        prune_threshold=1e-6,
        split_fringes_threshold=1.5,
        top_k_modes=profile.top_k_modes,
        v_h_max=profile.v_h_max,
        resample_threshold=0.5,
        resample_alpha=0.5,
        resample_beta=0.98,
        scibior_trick=True,
        trim=True,
    )


def make_sim_pars(profile: RunProfile, cfg: GravimeterConfig) -> SimulationParameters:
    """Construct :class:`~.SimulationParameters` from a :class:`RunProfile`.

    Parameters
    ----------
    profile : RunProfile
    cfg : GravimeterConfig
        Used to read the precision string.

    Returns
    -------
    SimulationParameters
    """
    return SimulationParameters(
        sim_name=f"gravity_multi_pf_{profile.name}",
        num_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=1.0,
        prec=cfg.prec,
        stop_gradient_input=profile.stop_gradient_input,
        loss_logl_outcomes=profile.loss_logl_outcomes,
        loss_logl_controls=False,     # continuous controls
        cumulative_loss=profile.cumulative_loss,
        baseline=profile.baseline,
        stop_gradient_pf=profile.stop_gradient_pf,
        log_loss=False,
        permutation_invariant=False,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(profile: RunProfile) -> None:
    """Run the full training pipeline for one :class:`RunProfile`.

    Steps:

    1. Create output directory and save configuration.
    2. Build :class:`~gravimeter_model.GravityStatelessPhysicalModel`,
       :class:`~gravimeter_multi_pf.MultiPFBank`,
       :class:`~gravimeter_multi_pf.GravityMultiPFSimulation`, and
       the MLP controller.
    3. Run the custom eager training loop (calls ``simulation.execute()``
       in ``GradientTape`` context, applies gradients via Adam optimizer).
       We use a custom eager loop instead of ``utils.train()`` because the
       Multi-PF bank's Python-level state management (list operations,
       ``tf.Variable`` in-place updates, ``.numpy()`` calls for split
       decisions) is incompatible with ``@tf.function`` tracing.
    4. Save final controller weights.

    Parameters
    ----------
    profile : RunProfile
        Fully specified training configuration.
    """
    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run profile : {profile.name}")
    print(f"Noise mode  : {NOISE_MODE}")
    print(f"Run mode    : {RUN_MODE}")
    print(f"Output dir  : {out_dir.resolve()}")
    print("=" * 72)

    # --- Configuration ---
    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)

    rangen = tf.random.Generator.from_seed(profile.seed)

    # --- Build all components ---
    simulation, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )

    variables = controller.trainable_variables
    print(
        f"[info] Controller parameters: "
        f"{sum(int(np.prod(v.shape)) for v in variables):,}"
    )
    print(
        f"[info] Bank: N_total={bank_cfg.n_total}, K_max={bank_cfg.k_max}, "
        f"N_min={bank_cfg.n_min}"
    )
    print(
        f"[info] Simulation: {simpars.num_steps} steps, "
        f"max_resources={simpars.max_resources:.3f} s, "
        f"batch={profile.batchsize}"
    )

    # --- Optimizer ---
    lr_schedule = InverseSqrtDecay(
        initial_learning_rate=profile.initial_lr,
        prec=cfg.prec,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # --- Training checkpoint directory ---
    sim_str = str(simulation)
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Eager training loop ---
    # The Multi-PF bank uses Python-level state manipulation (.numpy() calls,
    # list operations), which is incompatible with tf.function tracing.
    # We therefore run the training in eager mode, which is fully supported
    # by TensorFlow and still benefits from GPU acceleration for tensor ops.
    #
    # For XLA-compatible training (Phase 6 of the redesign roadmap), the bank
    # state would need to be flattened into loop-variable tensors as described
    # in Section 6.3 of MULTI_PF_REDESIGN.md.
    loss_history = []

    train_debug_jsonl = out_dir / "train_debug.jsonl"
    rollout_debug_jsonl = out_dir / "rollout_debug.jsonl"

    # start fresh
    for p in (train_debug_jsonl, rollout_debug_jsonl):
        if p.exists():
            p.unlink()

    # def single_step_eager() -> float:
    #     """One gradient accumulation step (eager)."""
    #     acc_loss = 0.0
    #     acc_grads = [tf.zeros_like(v) for v in variables]
    #     for _ in range(profile.gradient_accumulation):
    #         with tf.GradientTape() as tape:
    #             loss_diff, loss = simulation.execute(rangen)
    #         grads = tape.gradient(loss_diff, variables)
    #         # Guard against None gradients (can occur for unused variables)
    #         grads = [g if g is not None else tf.zeros_like(v)
    #                  for g, v in zip(grads, variables)]
    #         acc_loss += float(loss.numpy())
    #         acc_grads = [ag + g for ag, g in zip(acc_grads, grads)]
    #     # Average gradients over accumulation steps
    #     acc_grads = [g / profile.gradient_accumulation for g in acc_grads]
    #     optimizer.apply_gradients(zip(acc_grads, variables))
    #     return acc_loss / profile.gradient_accumulation
    def single_step_eager(debug: bool = False):
        """One gradient accumulation step (eager)."""
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        step_debug_records = []

        for acc_i in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                if debug and acc_i == profile.gradient_accumulation - 1:
                    loss_diff, loss, dbg = simulation.execute(
                        rangen,
                        debug=True,
                        debug_max_examples=3,
                    )
                    step_debug_records.extend(dbg)
                else:
                    loss_diff, loss = simulation.execute(rangen)

            grads = tape.gradient(loss_diff, variables)
            grads = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads, variables)]

            # replace non-finite grads with zero so logging is stable
            grads = [
                tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                for g in grads
            ]

            acc_loss += float(loss.numpy())
            acc_grads = [ag + g for ag, g in zip(acc_grads, grads)]

        acc_grads = [g / profile.gradient_accumulation for g in acc_grads]
        grad_norm = float(tf.linalg.global_norm(acc_grads).numpy())
        optimizer.apply_gradients(zip(acc_grads, variables))

        return (
            acc_loss / profile.gradient_accumulation,
            grad_norm,
            step_debug_records,
        )
    

    best_loss = float("inf")
    best_ckpt_idx = 1

    pbar = trange(profile.iterations, desc="Training", unit="step")
    for j in pbar:
        debug_now = ((j + 1) % profile.interval_save == 0)
        step_loss, grad_norm, step_debug_records = single_step_eager(debug=debug_now)
        loss_history.append(step_loss)

        # Update progress bar with current metrics
        recent = np.mean(loss_history[max(0, j - 9) : j + 1])
        try:
            # TF ≥ 2.11: optimizer.learning_rate is a tensor, use schedule directly
            lr_val = float(lr_schedule(optimizer.iterations).numpy())
        except Exception:
            lr_val = float(optimizer.learning_rate.numpy())
        pbar.set_postfix(
            loss=f"{step_loss:.4f}",
            avg10=f"{recent:.4f}",
            best=f"{best_loss:.4f}",
            lr=f"{lr_val:.1e}",
        )

        train_record = {
            "iter": int(j + 1),
            "loss": float(step_loss),
            "avg10": float(recent),
            "best": float(best_loss) if np.isfinite(best_loss) else None,
            "grad_norm": float(grad_norm),
            "lr": float(lr_val),
            "bank_k_active": int(bank.k_active),
            "mean_q0": float(tf.reduce_mean(bank.mode_weights[:, 0]).numpy()),
            "mean_V_H": float(
                tf.reduce_mean(
                    bank.holevo_variance(
                        tf.cast(
                            2.0 * np.pi / max(simulation.g_hi - simulation.g_lo, 1e-10),
                            simulation.simpars.prec,
                        ) * tf.ones((profile.batchsize,), dtype=simulation.simpars.prec)
                    )
                ).numpy()
            ),
        }
        _append_jsonl(train_debug_jsonl, [train_record])
        if step_debug_records:
            for rec in step_debug_records:
                rec["train_iter"] = int(j + 1)
            _append_jsonl(rollout_debug_jsonl, step_debug_records)

        # Save checkpoint every interval_save iterations
        if (j + 1) % profile.interval_save == 0:
            ckpt_idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / str(ckpt_idx)
            controller.save_weights(
                str(ckpt_path) + ".weights.h5"
            )
            # Track best checkpoint by window-mean loss
            window_start = max(0, j + 1 - profile.interval_save)
            window_loss = float(np.mean(loss_history[window_start : j + 1]))
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = ckpt_idx

    pbar.close()
    print("[training] Done.")

    # --- Load best checkpoint ---
    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(f"[ok] Loaded best checkpoint (idx={best_ckpt_idx}, "
              f"window_loss={best_loss:.4f}) from {best_path}")

    # Clean up checkpoint directory
    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass  # non-empty directory, leave it

    # --- Save final weights ---
    weights_path = out_dir / f"gravity_multi_pf_{profile.name}_final.weights.h5"
    controller.save_weights(str(weights_path))
    print(f"[ok] Saved controller weights to {weights_path}")

    # --- Save loss history CSV ---
    _save_loss_history(out_dir, sim_str, loss_history, profile.interval_save)

    # --- Save run config ---
    _save_run_config(out_dir, profile=profile, cfg=cfg, bank_cfg=bank_cfg)
    print(f"[done] All outputs saved to {out_dir.resolve()}")


def _save_loss_history(
    out_dir: Path,
    sim_str: str,
    loss_history: list,
    interval_save: int,
) -> None:
    """Save loss history as a CSV file (compatible with ``utils.train`` output).

    Parameters
    ----------
    out_dir : Path
    sim_str : str
        Simulation string identifier (from ``str(simulation)``).
    loss_history : list of float
    interval_save : int
        Window size for computing block averages.
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
    bank_cfg: MultiPFBankConfig,
) -> None:
    """Save a human-readable JSON config file to the output directory.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    profile : RunProfile
    cfg : GravimeterConfig
    bank_cfg : MultiPFBankConfig
    """
    import json
    from dataclasses import asdict

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
            "n_total": profile.n_total,
            "n_min": profile.n_min,
            "k_max": profile.k_max,
            "top_k_modes": profile.top_k_modes,
            "v_h_max": profile.v_h_max,
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
            "n_total": bank_cfg.n_total,
            "n_min": bank_cfg.n_min,
            "k_max": bank_cfg.k_max,
            "prune_threshold": bank_cfg.prune_threshold,
            "split_fringes_threshold": bank_cfg.split_fringes_threshold,
            "top_k_modes": bank_cfg.top_k_modes,
            "v_h_max": bank_cfg.v_h_max,
        },
    }

    cfg_path = out_dir / "run_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[ok] Saved run config to {cfg_path}")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_evaluation(profile: RunProfile, weights_path: Optional[Path] = None) -> None:
    """Load a trained controller and evaluate its performance.

    Runs ``profile.eval_iters`` episodes with the trained controller and
    prints the mean Holevo variance as a function of consumed resources.

    Parameters
    ----------
    profile : RunProfile
    weights_path : Path, optional
        Path to the ``.weights.h5`` file.  Defaults to
        ``{out_dir}/gravity_multi_pf_{name}_final.weights.h5``.
    """
    from typing import Optional  # re-import for type hints inside function

    out_dir = Path(profile.out_dir)
    if weights_path is None:
        weights_path = out_dir / f"gravity_multi_pf_{profile.name}_final.weights.h5"

    if not weights_path.exists():
        print(f"[warn] Weights not found at {weights_path}. Skipping evaluation.")
        return

    print(f"\n[eval] Loading weights from {weights_path}")
    cfg = make_gravimeter_cfg()
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile, cfg)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)

    simulation, bank, controller = build_gravity_multi_pf_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
    )
    controller.load_weights(str(weights_path))
    print("[eval] Weights loaded.")

    # Run evaluation episodes — collect V_H, MSE, true g, and estimated g
    # All metrics are from the SAME episode (deploy mode) so they are paired.
    all_vh = []       # Holevo variance per episode
    all_mse = []      # MSE per episode (batch-averaged)
    all_g_true = []   # true g values (bs,) per episode
    all_g_hat = []    # posterior mean estimates (bs,) per episode

    pbar = trange(profile.eval_iters, desc="Evaluating", unit="ep")
    for i in pbar:
        # Execute one episode in deploy mode to get true_values
        result = simulation.execute(rangen, deploy=True)
        true_values_tensor = result[0]  # (bs, 1, d)
        g_true = true_values_tensor[:, 0, 0]  # (bs,)

        # V_H from the bank after this episode (same episode!)
        from math import pi as _pi
        k_ref_fixed = tf.cast(
            2.0 * _pi / max(simulation.g_hi - simulation.g_lo, 1e-10),
            simulation.simpars.prec,
        ) * tf.ones((profile.batchsize,), dtype=simulation.simpars.prec)
        vh_tensor = bank.holevo_variance(k_ref_fixed)  # (bs, 1)
        vh = float(tf.reduce_mean(vh_tensor).numpy())
        all_vh.append(vh)

        # g_hat from MAP mode (highest-weight mode's mean, not mixture mean)
        g_hat, _ = bank.map_mode_mean()  # (bs,)
        mse_batch = float(tf.reduce_mean(tf.square(g_hat - g_true)).numpy())
        all_mse.append(mse_batch)
        all_g_true.append(g_true.numpy())
        all_g_hat.append(g_hat.numpy())

        pbar.set_postfix(
            V_H=f"{vh:.4f}",
            RMSE=f"{np.sqrt(mse_batch):.2e}",
            mean_VH=f"{np.mean(all_vh):.4f}",
        )
    pbar.close()

    all_vh = np.array(all_vh)
    all_mse = np.array(all_mse)
    all_rmse = np.sqrt(all_mse)
    all_g_true_flat = np.concatenate(all_g_true)
    all_g_hat_flat = np.concatenate(all_g_hat)

    print(f"\n[eval] Results over {profile.eval_iters} episodes:")
    print(f"  Mean V_H:  {np.mean(all_vh):.6f}")
    print(f"  Mean MSE:  {np.mean(all_mse):.2e}")
    print(f"  Mean RMSE: {np.mean(all_rmse):.2e} m/s²")
    print(f"  Global RMSE (all samples): "
          f"{np.sqrt(np.mean((all_g_true_flat - all_g_hat_flat)**2)):.2e} m/s²")

    # Save evaluation results — extended format
    eval_path = out_dir / f"eval_{profile.name}.npy"
    np.save(str(eval_path), all_vh)
    print(f"[ok] Saved V_H to {eval_path}")

    # Save extended eval data as npz
    eval_ext_path = out_dir / f"eval_{profile.name}_extended.npz"
    np.savez(
        str(eval_ext_path),
        v_h=all_vh,
        mse=all_mse,
        rmse=all_rmse,
        g_true=all_g_true_flat,
        g_hat=all_g_hat_flat,
    )
    print(f"[ok] Saved extended eval (V_H, MSE, RMSE, g_true, g_hat) to {eval_ext_path}")


# ---------------------------------------------------------------------------
# Type annotation for Optional (used in run_evaluation)
# ---------------------------------------------------------------------------

from typing import Optional  # noqa: E402


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
