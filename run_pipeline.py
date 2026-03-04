# run_pipeline.py
import os
import math
import numpy as np
import torch


from pl_brl.env_rbpf import Prior, ActionGrid, ModelConfig, EnvConfig, GravimeterEnvPaperRBPF
from pl_brl.rl_pipeline import (
    ActorCriticMasked,
    PPOConfig,
    generate_expert_dataset,
    behavior_cloning_pretrain,
    ppo_train,
    evaluate_policy,
)

def format_sci_no_leading_zero(x: float, sig: int = 0) -> str:
    # Start from scientific notation like "1e-04"
    s = f"{x:.{sig}e}"
    # Turn "e-04" -> "e-4" and "e+04" -> "e+4"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s

def make_grids(
    n: int = 150,
    low_frac: float = 0.4,
    Bp_min: float = 0.2,
    Bp_split: float = 10.0,
    Bp_max: float = 100.0,
    T_min: float = 100e-6,
    T_split: float = 500e-6,
    T_max: float = 3e-3,
):
    """Hybrid grid: geometric at low values, linear at high values."""
    n_low = int(round(n * low_frac))
    n_high = n - n_low

    Bp_low = np.geomspace(Bp_min, Bp_split, n_low + 1, dtype=np.float32)[:-1]
    Bp_high = np.linspace(Bp_split, Bp_max, n_high, dtype=np.float32)
    Bp_vals = np.concatenate([Bp_low, Bp_high]).astype(np.float32)

    T_low = np.geomspace(T_min, T_split, n_low + 1, dtype=np.float32)[:-1]
    T_high = np.linspace(T_split, T_max, n_high, dtype=np.float32)
    T_vals = np.concatenate([T_low, T_high]).astype(np.float32)
    return Bp_vals, T_vals


def mass_from_radius(r_m: float, rho_kg_m3: float = 3510.0) -> float:
    """Spherical diamond mass from radius and density (rho≈3510 kg/m^3)."""
    return float((4.0 / 3.0) * math.pi * (r_m ** 3) * rho_kg_m3)


def main():
    # ---- priors ----
    prior = Prior(
        g_range=(9.7639, 9.8337),
        dt_range=(-1e-6, 1e-6),
        phi_range=(-np.pi, np.pi),
    )

    # ---- action grids ----
    Bp_vals, T_vals = make_grids(n=150, low_frac=0.4)
    grid = ActionGrid(Bp_vals_kTm=Bp_vals, T_vals_s=T_vals)

    # ---- physics (paper-faithful fixed mass from r=100 nm) ----
    mass = mass_from_radius(r_m=100e-9, rho_kg_m3=3510.0)
    len_episodes = 512
    sigma_B_rel = 1e-2  # ~Paper note 10% MFG control noise (tuning this changes MFG amplitude noise)
    
    model = ModelConfig(
        omega_rad_s=2.0 * np.pi * 10e3,
        gamma_e_rad_s_T=2.0 * np.pi * 28e9,
        mass_kg=mass,
        eps_prob=1e-6,
        sigma_phi_drift=0.03,
        sigma_dt_drift=0.0,
        sigma_B_rel=sigma_B_rel,
    )

    # ---- env config ----
    env_cfg = EnvConfig(
        episode_len=len_episodes, #60
        n_g_grid=512, #2048
        n_nuisance=64, #256
        g_hist_bins=30,
        probe_every=0,
        lambda_dt=0.0,

        alias_sigma_mult=3.0,
        alias_frac=0.75,
        theta_probe_max=np.pi / 4.0,
        resample_ess_frac=0.25,

        wrap_max=1.00,                 # NEW (try 3, 5, 8)
        lambda_cycle=0.0,             # NEW (start 0, then tune up)
    )

    # ---- PPO config ----
    ppo_cfg = PPOConfig(
        device="cpu",
        n_envs=8, #8
        rollout_steps=256, #512
        update_epochs=4, #6
        minibatch_size=512,
        lr=3e-4,
        ent_coef=0.01,
    )

    def env_ctor(seed: int):
        return GravimeterEnvPaperRBPF(prior=prior, grid=grid, model=model, cfg=env_cfg, seed=seed)

    print("Δg_2π at max (T,B') =", model.dg_period_2pi(float(T_vals[-1]), float(Bp_vals[-1])))
    print("Δg_2π at min (T,B') =", model.dg_period_2pi(float(T_vals[0]), float(Bp_vals[0])))
    print(f"Using fixed mass_kg={model.mass_kg:.3e} (r=100 nm, rho=3510 kg/m^3)")
    print(f"Using sigma_B_rel={model.sigma_B_rel:.2e} (shot-to-shot MFG amplitude noise)")

    envs = [env_ctor(1000 + i) for i in range(ppo_cfg.n_envs)]
    obs_dim = envs[0].obs_dim
    nT, nB = envs[0].nT, envs[0].nB

    policy = ActorCriticMasked(obs_dim=obs_dim, nT=nT, nB=nB, hidden=256)

    # scaling for stable BC/PPO
    policy.set_obs_scaling(bins=env_cfg.g_hist_bins, g_range=prior.g_range, dt_range=prior.dt_range)

    # ---- precompute mask tables ----
    kg = np.zeros((nT, nB), dtype=np.float32)
    dg2pi = np.zeros((nT, nB), dtype=np.float32)
    for i in range(nT):
        for j in range(nB):
            k = model.k_g(float(T_vals[i]), float(Bp_vals[j]))
            kg[i, j] = k
            dg2pi[i, j] = float(2.0 * np.pi / max(k, 1e-30))

    dev = torch.device(ppo_cfg.device)
    kg_flat = torch.tensor(kg.reshape(-1), dtype=torch.float32, device=dev)
    dg2pi_flat = torch.tensor(dg2pi.reshape(-1), dtype=torch.float32, device=dev)

    bins = int(env_cfg.g_hist_bins)
    std_g_index = bins + 4
    probe_index = bins + 11  # last element

    policy.set_action_masking(
        dg2pi_flat=dg2pi_flat,
        kg_flat=kg_flat,
        std_g_index=std_g_index,
        probe_index=probe_index,
        alias_sigma_mult=float(env_cfg.alias_sigma_mult),
        alias_frac=float(env_cfg.alias_frac),         # kept but no longer used
        theta_probe_max=float(env_cfg.theta_probe_max),
        wrap_max=float(env_cfg.wrap_max),             # NEW
    )

    # ---- [1] expert dataset ----
    print("\n[1] Generating expert dataset...")
    data = generate_expert_dataset(
        env_ctor=env_ctor,
        episodes=300, #500
        horizon=env_cfg.episode_len,
        seed=0,
        topk=8, #16
    )
    print("Dataset transitions:", len(data["aT"]))

    # ---- [2] BC pretrain ----
    print("\n[2] Behavior cloning pretrain...")
    behavior_cloning_pretrain(policy, data, device=ppo_cfg.device, epochs=12, batch_size=2048, lr=1e-3, csv_path="logs/bc_metrics.csv", run_name="debug_512_64")

    # ---- [3] PPO finetune ----
    print("\n[3] PPO finetune...")
    log_path = f"logs_{len_episodes}_sigmaB_{format_sci_no_leading_zero(sigma_B_rel)}/train_metrics.csv"
    policy = ppo_train(
        envs=envs,
        model=policy,
        cfg=ppo_cfg,
        total_updates = 2000,
        seed=0,
        log_every=20,
        err_thresh_g=0.01,
        csv_path = log_path,
    )

    # ---- save ----
    os.makedirs("checkpoints", exist_ok=True)

    ckpt_path = f"checkpoints{len_episodes}_sigmaB_{format_sci_no_leading_zero(sigma_B_rel)}/rbpf_policy.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "nT": nT,
            "nB": nB,
            "prior": prior,
            "model_cfg": model,
            "env_cfg": env_cfg,
            "Bp_vals_kTm": Bp_vals,
            "T_vals_s": T_vals,
        },
        ckpt_path,
    )
    print(f"\nSaved policy to: {ckpt_path}")

    # ---- evaluation ----
    eval_env = env_ctor(42)
    metrics = evaluate_policy(
        eval_env,
        policy,
        episodes=len_episodes,
        device=ppo_cfg.device,
        err_thresh_g=0.01,     # catastrophic = > 0.01 m/s^2
        top_peaks=5,
        verbose_cats=True,
        policy_mode="sample",   # "greedy" or "sample"
    )
    print("\nEvaluation metrics:", metrics)


if __name__ == "__main__":
    main()
