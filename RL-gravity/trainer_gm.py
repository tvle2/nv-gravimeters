# trainer_gm.py
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import importlib.util
import json
import math

import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tqdm.auto import trange

def _load_local(module_name: str):
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
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from qsensoropt.simulation_parameters import SimulationParameters
    from qsensoropt.schedulers import InverseSqrtDecay
except Exception:
    SimulationParameters = _load_local("simulation_parameters").SimulationParameters
    InverseSqrtDecay = _load_local("schedulers").InverseSqrtDecay


from gravimeter_model_complete import GravimeterConfig
from gravimeter_gm_bank import GaussianMixtureBankConfig
from gravimeter_gm_simulation import build_gravity_gm_simulation


# ===========================================================================
# Top-level flags
# ===========================================================================

RUN_PROFILE: str = "diag"           # "diag", "diag_long", "pilot", "full"
RUN_MODE: str = "all"               # "all", "train-only", "eval-only"
NOISE_MODE: str = "paper"            # "none" or "paper"

# ===========================================================================
# RunProfile
# ===========================================================================

@dataclass(frozen=True)
class RunProfile:
    name: str
    out_dir: str
    batchsize: int
    iterations: int
    interval_save: int
    max_steps: int
    max_resources: float
    initial_lr: float
    grad_clip_norm: float
    seed: int
    gradient_accumulation: int
    K: int
    top_k_modes: int
    prec: str
    # Framework loss flags
    cumulative_loss: bool
    log_loss: bool
    loss_logl_outcomes: bool
    baseline: bool
    stop_gradient_input: bool
    stop_gradient_pf: bool
    eval_iters: int


# Small, fast diagnostic profile. Use this to check pipeline correctness.
DIAG_PROFILE = RunProfile(
    name="diag",
    out_dir="runs/gravity_gm_v4_noise",
    batchsize=128,
    iterations=3000,
    interval_save=20,
    max_steps=64,
    max_resources=120e-3,
    initial_lr=2e-3,
    grad_clip_norm=100.0,
    seed=42,
    gradient_accumulation=4,
    K=4,
    top_k_modes=4,
    prec="float64",
    cumulative_loss=True,
    log_loss=False,             
    loss_logl_outcomes=True,
    baseline=True,
    stop_gradient_input=False,
    stop_gradient_pf=True,
    eval_iters=32,
)

# Longer diagnostic.
DIAG_LONG_PROFILE = RunProfile(
    name="diag_long",
    out_dir="runs/gravity_gm_diag_long",
    batchsize=128,
    iterations=2000,
    interval_save=100,
    max_steps=32,
    max_resources=100e-3,
    initial_lr=2e-3,
    grad_clip_norm=10.0,
    seed=42,
    gradient_accumulation=1,
    K=64,
    top_k_modes=4,
    prec="float64",
    cumulative_loss=False,
    log_loss=False,
    loss_logl_outcomes=True,
    baseline=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    eval_iters=128,
)

# Production-scale (large batch, long episodes).
PILOT_PROFILE = RunProfile(
    name="pilot",
    out_dir="runs/gravity_gm_pilot",
    batchsize=256,
    iterations=5000,
    interval_save=250,
    max_steps=32,
    max_resources=100e-3,
    initial_lr=3e-3,
    grad_clip_norm=10.0,
    seed=42,
    gradient_accumulation=1,
    K=64,
    top_k_modes=4,
    prec="float64",
    cumulative_loss=True,
    log_loss=False,
    loss_logl_outcomes=True,
    baseline=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    eval_iters=256,
)

FULL_PROFILE = RunProfile(
    name="full",
    out_dir="runs/gravity_gm_full",
    batchsize=512,
    iterations=20000,
    interval_save=500,
    max_steps=64,
    max_resources=200e-3,
    initial_lr=3e-3,
    grad_clip_norm=10.0,
    seed=123,
    gradient_accumulation=2,
    K=128,
    top_k_modes=4,
    prec="float64",
    cumulative_loss=True,
    log_loss=False,
    loss_logl_outcomes=True,
    baseline=True,
    stop_gradient_input=True,
    stop_gradient_pf=True,
    eval_iters=512,
)


def get_profile() -> RunProfile:
    k = RUN_PROFILE.strip().lower()
    return {
        "diag": DIAG_PROFILE,
        "diag_long": DIAG_LONG_PROFILE,
        "pilot": PILOT_PROFILE,
        "full": FULL_PROFILE,
    }[k]


# ===========================================================================
# Configs
# ===========================================================================

def make_gravimeter_cfg(profile: RunProfile) -> GravimeterConfig:
    common = dict(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=1.47e-17,
        hbar_J_s=1.054_571_817e-34,
        kT_to_T=1e3,
        g_range=(9.7806, 9.825),
        infer_mfg_bias=False,
        infer_phi_off=False,
        fixed_phi_off_rad=0.0,
        T_range_s=(10e-6, 5e-4),
        Bp_range_kTm=(0.5, 25.0),
        readout_flip_prob=0.0,
        dead_time_s=0.0,
        mfg_resource_cost_s_at_ref=0.0,
        mfg_resource_ref_kTm=50.0,
        prec=profile.prec,
    )
    mode = NOISE_MODE.strip().lower()
    if mode == "none":
        return GravimeterConfig(
            **common,
            T2_spin_s=1.0e-3,
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
            T2_spin_s=1.0e-3,
            mfg_rel_noise_bound=3.0e-4,
            mfg_noise_quad_points=9,
            fixed_mfg_rel_bias=0.0,
            apply_fixed_mfg_bias_in_model=True,
            sigma_omega_rel=0.01,
            trap_visibility_mode="exact_single_delta",
            trap_noise_quad_points=9,
        )
    raise ValueError(f"Unknown NOISE_MODE={NOISE_MODE!r}")

def make_bank_cfg(profile: RunProfile) -> GaussianMixtureBankConfig:
    return GaussianMixtureBankConfig(
        K=profile.K,
        sigma_min_fraction=1e-4,
        sigma_max_fraction=1.0,
        init_sigma_fraction=0.25,
        posterior_gain_width_multiplier= 0.25,
        min_gain_fringe_fraction=0.25,
        log_sigma_frac_bounds=(-6.0, 0.0),
        revive_min_q=0.10,
    )


def make_sim_pars(profile: RunProfile) -> SimulationParameters:
    return SimulationParameters(
        sim_name=f"gravity_gm_{profile.name}",
        num_steps=profile.max_steps,
        max_resources=profile.max_resources,
        resources_fraction=1.0,
        prec=profile.prec,
        stop_gradient_input=profile.stop_gradient_input,
        loss_logl_outcomes=profile.loss_logl_outcomes,
        loss_logl_controls=False,
        cumulative_loss=profile.cumulative_loss,
        baseline=profile.baseline,
        stop_gradient_pf=profile.stop_gradient_pf,
        log_loss=profile.log_loss,
        permutation_invariant=False,
    )


# ===========================================================================
# Sanity & helpers
# ===========================================================================

def _sanity_checks(cfg: GravimeterConfig, profile: RunProfile) -> None:
    tau = 2.0 * math.pi / cfg.omega_rad_s
    cycle_min = cfg.dead_time_s + 3.5 * tau + 2.0 * cfg.T_range_s[0]
    cycle_max = cfg.dead_time_s + 3.5 * tau + 2.0 * cfg.T_range_s[1]
    max_steps_by_resource = profile.max_resources / cycle_min
    prior_width = cfg.g_range[1] - cfg.g_range[0]
    print(f"[sanity] tau                  = {tau*1e6:.2f} us")
    print(f"[sanity] cycle_min            = {cycle_min*1e6:.2f} us")
    print(f"[sanity] cycle_max            = {cycle_max*1e6:.2f} us")
    print(f"[sanity] max_steps by resource= {max_steps_by_resource:.0f}")
    print(f"[sanity] num_steps            = {profile.max_steps}")
    print(f"[sanity] prior_width          = {prior_width:.5f}")
    print(f"[sanity] one fringe at k_g    = {2*math.pi/prior_width:.2f}")
    print(f"[sanity] K modes              = {profile.K}, "
          f"mode_width = {prior_width/profile.K:.2e}")
    if cfg.T2_spin_s is not None:
        vis_max = math.exp(-cycle_max / cfg.T2_spin_s)
        print(f"[sanity] T2_spin              = {cfg.T2_spin_s*1e3:.2f} ms, "
              f"vis at T=T_max: {vis_max:.3f}")


def _append_jsonl(path: Path, records: List[dict]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def set_global_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ===========================================================================
# Training
# ===========================================================================

def run_training(profile: RunProfile) -> None:
    out_dir = Path(profile.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Profile  : {profile.name}")
    print(f"Noise    : {NOISE_MODE}")
    print(f"Out dir  : {out_dir.resolve()}")
    print(f"Batch    : {profile.batchsize}")
    print(f"K modes  : {profile.K}")
    print(f"Steps    : {profile.max_steps}")
    print(f"Iters    : {profile.iterations}")
    print(f"LR0      : {profile.initial_lr}")
    print(f"sg_pf    : {profile.stop_gradient_pf}")
    print("=" * 72)

    set_global_reproducibility(profile.seed)

    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)
    _sanity_checks(cfg, profile)

    rangen = tf.random.Generator.from_seed(profile.seed)
    sim, bank, controller = build_gravity_gm_simulation(
        batchsize=profile.batchsize,
        cfg=cfg,
        bank_cfg=bank_cfg,
        simpars=simpars,
        rangen=rangen,
        top_k_modes=profile.top_k_modes,
    )
    variables = controller.trainable_variables
    nparams = sum(int(np.prod(v.shape)) for v in variables)
    print(f"[info] Controller params: {nparams:,}")

    lr_schedule = InverseSqrtDecay(
        initial_learning_rate=profile.initial_lr, prec=profile.prec,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    sim_str = f"gravity_gm_{profile.name}"
    ckpt_dir = out_dir / f"{sim_str}_history_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = out_dir / "train_debug.jsonl"
    rollout_jsonl = out_dir / "rollout_debug.jsonl"
    for p in (train_jsonl, rollout_jsonl):
        if p.exists():
            p.unlink()

    def one_grad_step(debug: bool = False):
        acc_loss = 0.0
        acc_grads = [tf.zeros_like(v) for v in variables]
        records: List[dict] = []

        for acc_i in range(profile.gradient_accumulation):
            with tf.GradientTape() as tape:
                if debug and acc_i == profile.gradient_accumulation - 1:
                    loss_diff, loss, dbg = sim.execute(
                        rangen, debug=True, debug_max_examples=3,
                    )
                    records.extend(dbg)
                else:
                    loss_diff, loss = sim.execute(rangen)
            grads = tape.gradient(loss_diff, variables)
            grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, variables)
            ]
            grads = [
                tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                for g in grads
            ]
            acc_loss += float(loss.numpy())
            acc_grads = [a + g for a, g in zip(acc_grads, grads)]

        acc_grads = [g / float(profile.gradient_accumulation) for g in acc_grads]
        raw_norm = float(tf.linalg.global_norm(acc_grads).numpy())
        clipped_grads, _ = tf.clip_by_global_norm(acc_grads, profile.grad_clip_norm)
        clipped_norm = float(tf.linalg.global_norm(clipped_grads).numpy())
        optimizer.apply_gradients(zip(clipped_grads, variables))

        avg_loss = acc_loss / profile.gradient_accumulation
        return avg_loss, raw_norm, clipped_norm, records

    loss_history: List[float] = []
    final_mse_history: List[float] = []

    best_loss = float("inf")
    best_ckpt_idx = 1

    pbar = trange(profile.iterations, desc="Training", unit="step")
    for j in pbar:
        debug_now = ((j + 1) % profile.interval_save == 0)
        step_loss, raw_norm, clip_norm, dbg = one_grad_step(debug=debug_now)
        loss_history.append(step_loss)
        recent = float(np.mean(loss_history[max(0, j - 19): j + 1]))

        try:
            lr_val = float(lr_schedule(optimizer.iterations).numpy())
        except Exception:
            lr_val = float(optimizer.learning_rate.numpy())

        pbar.set_postfix(
            loss=f"{step_loss:+.4f}",
            avg20=f"{recent:+.4f}",
            best=f"{best_loss:+.4f}",
            lr=f"{lr_val:.1e}",
            gn=f"{raw_norm:.1e}",
            maxq=f"{float(tf.reduce_mean(tf.reduce_max(bank.q, axis=1)).numpy()):.3f}",
        )

        train_record = {
            "iter": int(j + 1),
            "loss": float(step_loss),
            # NEW: publication-quality metrics (final-step only, no NLL)
            "final_mse_norm": float(sim._last_final_mse_norm.numpy()),
            "final_max_q":    float(sim._last_final_max_q.numpy()),
            "final_qclose":   float(sim._last_final_qclose.numpy()),
            # Existing fields:
            "avg20": float(recent),
            "best": float(best_loss) if np.isfinite(best_loss) else None,
            "raw_grad_norm": float(raw_norm),
            "clip_grad_norm": float(clip_norm),
            "lr": float(lr_val),
            "K": int(bank.K),
            "max_q": float(tf.reduce_mean(tf.reduce_max(bank.q, axis=1)).numpy()),
        }

        final_mse_history.append(train_record["final_mse_norm"])


        _append_jsonl(train_jsonl, [train_record])
        if dbg:
            for rec in dbg:
                rec["train_iter"] = int(j + 1)
            _append_jsonl(rollout_jsonl, dbg)

        if (j + 1) % profile.interval_save == 0:
            idx = (j + 1) // profile.interval_save
            ckpt_path = ckpt_dir / f"{idx}.weights.h5"
            controller.save_weights(str(ckpt_path))
            window_start = max(0, j + 1 - profile.interval_save)
            window_final_mse = float(
                np.mean(final_mse_history[window_start: j + 1])
            )
            
            # Replace window_loss with the final_mse_norm window mean
            window_loss = window_final_mse
            if window_loss < best_loss:
                best_loss = window_loss
                best_ckpt_idx = idx

    pbar.close()
    print("[training] Done.")

    best_path = ckpt_dir / f"{best_ckpt_idx}.weights.h5"
    if best_path.exists():
        controller.load_weights(str(best_path))
        print(f"[ok] Loaded best ckpt idx={best_ckpt_idx}, window_loss={best_loss:+.4f}")

    final_path = out_dir / f"{sim_str}_final.weights.h5"
    controller.save_weights(str(final_path))
    print(f"[ok] Saved final weights to {final_path}")

    # CSV history
    arr = np.array(loss_history)
    interval = max(profile.interval_save, 1)
    n_blocks = len(arr) // interval
    if n_blocks > 0:
        blocks = arr[: n_blocks * interval].reshape(n_blocks, interval)
        means = blocks.mean(axis=1)
        with open(out_dir / f"{sim_str}_history.csv", "w") as f:
            f.write("Loss\n")
            for v in means:
                f.write(f"{v:.6e}\n")
        print(f"[ok] Wrote loss history CSV")

    # Cleanup checkpoints
    for f in ckpt_dir.glob("*.h5"):
        f.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass


# ===========================================================================
# Evaluation
# ===========================================================================

def run_evaluation(
    profile: RunProfile, weights_path: Optional[Path] = None,
) -> None:
    out_dir = Path(profile.out_dir)
    sim_str = f"gravity_gm_{profile.name}"
    if weights_path is None:
        weights_path = out_dir / f"{sim_str}_final.weights.h5"
    if not weights_path.exists():
        print(f"[warn] No weights at {weights_path}; skipping eval.")
        return

    print(f"\n[eval] Loading weights from {weights_path}")
    cfg = make_gravimeter_cfg(profile)
    bank_cfg = make_bank_cfg(profile)
    simpars = make_sim_pars(profile)
    rangen = tf.random.Generator.from_seed(profile.seed + 1000)

    sim, bank, controller = build_gravity_gm_simulation(
        batchsize=profile.batchsize,
        cfg=cfg, bank_cfg=bank_cfg, simpars=simpars, rangen=rangen,
        top_k_modes=profile.top_k_modes,
    )
    controller.load_weights(str(weights_path))

    all_mse = []
    all_g_true = []
    all_g_hat = []
    all_max_q = []

    prior_width = cfg.g_range[1] - cfg.g_range[0]
    pbar = trange(profile.eval_iters, desc="Evaluating", unit="ep")
    for _ in pbar:
        payload = sim.execute(rangen, deploy=True)
        true_values = payload[0]
        g_true = true_values[:, 0, 0].numpy()
        g_hat, _ = bank.marginal_mean_and_var()
        g_hat_np = g_hat.numpy()
        mse = (g_hat_np - g_true) ** 2
        max_q = tf.reduce_max(bank.q, axis=1).numpy()
        all_mse.extend(mse.tolist())
        all_g_true.extend(g_true.tolist())
        all_g_hat.extend(g_hat_np.tolist())
        all_max_q.extend(max_q.tolist())

    mse_arr = np.array(all_mse)
    rmse = float(np.sqrt(mse_arr.mean()))
    rel_rmse = rmse / prior_width
    mean_max_q = float(np.mean(all_max_q))
    print(f"\n[eval] N = {len(all_mse)} episodes")
    print(f"[eval] RMSE              = {rmse:.4e} m s^-2")
    print(f"[eval] RMSE / Δg         = {rel_rmse:.4e}")
    print(f"[eval] log10(MSE / Δg²)  = {math.log10(mse_arr.mean()/prior_width**2):+.3f}")
    print(f"[eval] mean max_q        = {mean_max_q:.3f}")

    np.savez(
        out_dir / f"{sim_str}_eval.npz",
        g_true=np.array(all_g_true),
        g_hat=np.array(all_g_hat),
        max_q=np.array(all_max_q),
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    profile = get_profile()
    mode = RUN_MODE.strip().lower()
    if mode in ("all", "train-only"):
        run_training(profile)
    if mode in ("all", "eval-only"):
        run_evaluation(profile)


if __name__ == "__main__":
    main()