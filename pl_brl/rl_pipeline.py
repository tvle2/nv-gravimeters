# pl_brl/rl_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActorCriticMasked(nn.Module):
    """
    Actor-Critic with masked joint categorical distribution over (aT,aB).
    Includes fixed feature scaling for the network trunk (mask still uses raw obs).
    """

    def __init__(self, obs_dim: int, nT: int, nB: int, hidden: int = 256):
        super().__init__()
        self.nT = int(nT)
        self.nB = int(nB)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_T = nn.Linear(hidden, self.nT)
        self.pi_B = nn.Linear(hidden, self.nB)
        self.v = nn.Linear(hidden, 1)

        self._mask_ready = False
        self._scale_ready = False

    # ---------------- scaling (for BC/PPO stability) ----------------
    def set_obs_scaling(
        self,
        bins: int,
        g_range: Tuple[float, float],
        dt_range: Tuple[float, float],
    ) -> None:
        """
        Scale only the heavy-magnitude continuous features before feeding the trunk.
        Mask reads raw obs, so constraints remain physically correct.
        """
        gmin, gmax = float(g_range[0]), float(g_range[1])
        dtmin, dtmax = float(dt_range[0]), float(dt_range[1])

        self.bins = int(bins)
        self.g_mid = 0.5 * (gmin + gmax)
        self.g_half = 0.5 * (gmax - gmin)
        self.dt_half = 0.5 * (dtmax - dtmin)

        # indices in the obs vector after hist
        self.mu_g_idx = self.bins + 3
        self.std_g_idx_feat = self.bins + 4
        self.mu_dt_idx = self.bins + 5
        self.std_dt_idx = self.bins + 6

        self._scale_ready = True

    def _preprocess_obs_for_net(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._scale_ready:
            return obs

        x = obs.clone()
        # normalize to O(1)
        x[:, self.mu_g_idx] = (x[:, self.mu_g_idx] - self.g_mid) / max(self.g_half, 1e-12)
        x[:, self.std_g_idx_feat] = x[:, self.std_g_idx_feat] / max(self.g_half, 1e-12)
        x[:, self.mu_dt_idx] = x[:, self.mu_dt_idx] / max(self.dt_half, 1e-12)
        x[:, self.std_dt_idx] = x[:, self.std_dt_idx] / max(self.dt_half, 1e-12)
        return x

    # ---------------- masking ----------------

    def set_action_masking(
        self,
        dg2pi_flat: torch.Tensor,
        kg_flat: torch.Tensor,
        std_g_index: int,
        probe_index: int,
        alias_sigma_mult: float,
        alias_frac: float,
        theta_probe_max: float,
        wrap_max: float = 5.0,
    ) -> None:
        # Allow repeated calls (for evaluation sweeps)
        if hasattr(self, "dg2pi_flat"):
            self.dg2pi_flat = dg2pi_flat
        else:
            self.register_buffer("dg2pi_flat", dg2pi_flat)

        if hasattr(self, "kg_flat"):
            self.kg_flat = kg_flat
        else:
            self.register_buffer("kg_flat", kg_flat)

        self.std_g_index = int(std_g_index)
        self.probe_index = int(probe_index)
        self.alias_sigma_mult = float(alias_sigma_mult)
        self.alias_frac = float(alias_frac)
        self.theta_probe_max = float(theta_probe_max)

        self.wrap_max = float(wrap_max)
        self.sigma_theta_max = float(2.0 * np.pi * self.wrap_max)
        
        self._mask_ready = True

        # recompute fallback each time
        self.fallback_idx = int(torch.argmin(self.kg_flat).item())

    def forward(self, obs_raw: torch.Tensor):
        obs = self._preprocess_obs_for_net(obs_raw)
        h = self.trunk(obs)
        logits_T = self.pi_T(h)
        logits_B = self.pi_B(h)
        v = self.v(h).squeeze(-1)
        return logits_T, logits_B, v

    def _compute_mask(self, obs_raw: torch.Tensor) -> torch.Tensor:
        assert self._mask_ready, "Call set_action_masking(...) first."
        std_g = obs_raw[:, self.std_g_index].clamp_min(0.0)

        kg = self.kg_flat.unsqueeze(0)
        m = self.alias_sigma_mult
        sigma_theta = m * std_g.unsqueeze(1) * kg

        mask = sigma_theta <= self.sigma_theta_max

        row_ok = mask.any(dim=1)
        if not torch.all(row_ok):
            mask[row_ok == 0, self.fallback_idx] = True
        return mask

    def _masked_joint_logits(self, obs_raw: torch.Tensor) -> torch.Tensor:
        logits_T, logits_B, _ = self.forward(obs_raw)
        joint = logits_T.unsqueeze(2) + logits_B.unsqueeze(1)  # [B,nT,nB]
        joint_flat = joint.reshape(obs_raw.shape[0], -1)       # [B,A]
        mask = self._compute_mask(obs_raw)
        return joint_flat.masked_fill(~mask, -1e9)

    def _masked_joint_dist(self, obs_raw: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self._masked_joint_logits(obs_raw))

    @torch.no_grad()
    def act(self, obs_raw: torch.Tensor):
        dist = self._masked_joint_dist(obs_raw)
        idx = dist.sample()
        logp = dist.log_prob(idx)
        _, _, v = self.forward(obs_raw)
        aT = idx // self.nB
        aB = idx % self.nB
        return aT, aB, logp, v

    def logprob_entropy_value(self, obs_raw: torch.Tensor, aT: torch.Tensor, aB: torch.Tensor):
        dist = self._masked_joint_dist(obs_raw)
        idx = aT * self.nB + aB
        logp = dist.log_prob(idx)
        ent = dist.entropy()
        _, _, v = self.forward(obs_raw)
        return logp, ent, v

    @torch.no_grad()
    def greedy(self, obs_raw: torch.Tensor) -> Tuple[int, int]:
        logits = self._masked_joint_logits(obs_raw.unsqueeze(0)).squeeze(0)
        idx = int(torch.argmax(logits).item())
        return idx // self.nB, idx % self.nB


# ----------------------------- dataset (expert) -----------------------------
def expert_scores(env, obs: np.ndarray) -> np.ndarray:
    if getattr(env.cfg, "probe_every", 0) != 0:
        raise RuntimeError("expert_scores(): expected probe_every=0")
    if getattr(env.cfg, "lambda_dt", 0.0) != 0.0:
        raise RuntimeError("expert_scores(): expected lambda_dt=0")

    st = env.belief.stats()
    stdg = max(float(st["std_g"]), 1e-12)
    mu_g = float(st["mu_g"])

    nT, nB = env.nT, env.nB
    scores = np.full((nT, nB), -np.inf, dtype=np.float64)

    m = float(env.cfg.alias_sigma_mult)

    wrap_max = float(env.cfg.wrap_max)
    sigma_theta_max = 2.0 * np.pi * wrap_max

    # --- key knobs (these matter) ---
    theta_target = 1.0     # rad: desired sigma_theta per step (try 0.7–1.5)
    beta = 0.5             # how hard to enforce k≈k_star (try 0.2–2.0)
    lam_edge = 0.2         # penalty near boundary (try 0.05–0.5)
    margin = 0.85          # safety margin vs hard edge (try 0.7–0.9)

    sigma_theta_safe = sigma_theta_max * margin

    # target k and cap by safe boundary
    k_target = theta_target / (m * stdg)
    k_cap = sigma_theta_safe / (m * stdg)
    k_star = min(k_target, k_cap)

    for i in range(nT):
        T_s = float(env.grid.T_vals_s[i])
        for j in range(nB):
            Bp = float(env.grid.Bp_vals_kTm[j])

            k = float(env.model.k_g(T_s, Bp))
            if not np.isfinite(k) or k <= 0.0:
                continue

            sigma_theta = m * stdg * k
            if sigma_theta > sigma_theta_safe:
                continue  # hard gate with margin

            # Visibility mixture under dt particles
            eta = env.model.eta(Bp)
            A_part = env.model.visibility_A(eta, env.belief.dt)
            A_eff = float(np.sum(env.belief.w_n * A_part))

            # MFG amplitude noise proxy dephasing
            if float(env.model.sigma_B_rel) > 0.0:
                phi_nom = float(env.model.delta_phi_scalar(mu_g, T_s, Bp))
                A_eff *= float(np.exp(-0.5 * (float(env.model.sigma_B_rel) * phi_nom) ** 2))

            # Encourage informative but controlled k:
            # base info at k_star (NOT k)
            base = (A_eff ** 2) * (k_star ** 2)

            # penalize deviating from k_star (log is scale-invariant)
            log_ratio = np.log(max(k, 1e-12) / max(k_star, 1e-12))
            pen_k = beta * (log_ratio ** 2)

            # penalize operating close to the safe edge
            edge = sigma_theta / max(sigma_theta_safe, 1e-12)
            pen_edge = lam_edge * (edge ** 4)

            scores[i, j] = base - pen_k - pen_edge

    return scores


def expert_action_and_topk(env, obs: np.ndarray, topk: int = 16):
    scores = expert_scores(env, obs)
    flat = scores.reshape(-1)
    finite = np.isfinite(flat)
    if not np.any(finite):
        idx = env.fallback_joint_idx
        return env.fallback_aT, env.fallback_aB, np.array([idx], dtype=np.int64)

    finite_idx = np.flatnonzero(np.isfinite(flat))
    K = int(min(topk, finite_idx.size))
    top_idx = finite_idx[np.argpartition(flat[finite_idx], -K)[-K:]]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    best = int(top_idx[0])
    aT = best // env.nB
    aB = best % env.nB
    return int(aT), int(aB), top_idx.astype(np.int64)


def generate_expert_dataset(env_ctor, episodes: int, horizon: int, seed: int = 0, topk: int = 16) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs_list, aT_list, aB_list, topk_list = [], [], [], []

    for _ in range(episodes):
        env = env_ctor(int(rng.integers(1_000_000)))
        obs = env.reset()
        for _t in range(horizon):
            aT, aB, top_idx = expert_action_and_topk(env, obs, topk=topk)
            obs_list.append(obs.copy())
            aT_list.append(aT)
            aB_list.append(aB)
            topk_list.append(top_idx)
            obs, _, done, _ = env.step(aT, aB)
            if done:
                break

    K = max(len(x) for x in topk_list)
    topk_arr = np.full((len(topk_list), K), -1, dtype=np.int64)
    for i, x in enumerate(topk_list):
        topk_arr[i, : len(x)] = x

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "aT": np.asarray(aT_list, dtype=np.int64),
        "aB": np.asarray(aB_list, dtype=np.int64),
        "topk": topk_arr,
    }


# ----------------------------- behavior cloning (Top-K) -----------------------------


def behavior_cloning_pretrain(
    model: ActorCriticMasked,
    data: Dict[str, np.ndarray],
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    csv_path: str | None = "logs/bc_metrics.csv",
    run_name: str = "default",
) -> None:
    dev = torch.device(device)
    model.to(dev)
    model.train()

    writer = None
    f_csv = None
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_exists = os.path.exists(csv_path)
        f_csv = open(csv_path, "a", newline="")
        writer = csv.DictWriter(f_csv, fieldnames=["run", "epoch", "topk_nll"])
        if not csv_exists:
            writer.writeheader()

    opt = optim.Adam(model.parameters(), lr=lr)

    obs = torch.tensor(data["obs"], dtype=torch.float32, device=dev)
    topk = torch.tensor(data["topk"], dtype=torch.int64, device=dev)

    N = obs.shape[0]
    idxs = torch.arange(N, device=dev)

    try:
        for ep in range(1, epochs + 1):
            perm = idxs[torch.randperm(N)]
            losses = []
            for start in range(0, N, batch_size):
                mb = perm[start:start + batch_size]
                mb_obs = obs[mb]
                mb_topk = topk[mb]

                logits = model._masked_joint_logits(mb_obs)  # [B,A]
                logp_all = logits - torch.logsumexp(logits, dim=1, keepdim=True)

                mask_valid = mb_topk >= 0
                safe_topk = mb_topk.clamp_min(0)

                gathered = logp_all.gather(1, safe_topk)
                gathered = gathered.masked_fill(~mask_valid, -1e9)

                log_mass = torch.logsumexp(gathered, dim=1)
                loss = (-log_mass).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                losses.append(float(loss.item()))
            print(f"[BC] epoch {ep}/{epochs} | topk_nll={np.mean(losses):.6f}")
            if writer is not None:
                writer.writerow({"run": run_name, "epoch": ep, "topk_nll": float(np.mean(losses))})
                f_csv.flush()
    finally:
        if f_csv is not None:
            f_csv.close()


# ----------------------------- PPO -----------------------------


@dataclass
class PPOConfig:
    device: str = "cpu"
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    n_envs: int = 8
    rollout_steps: int = 512
    update_epochs: int = 6
    minibatch_size: int = 512


class RolloutBuffer:
    def __init__(self, T: int, N: int, obs_dim: int, device: str):
        self.T, self.N, self.device = T, N, device
        self.obs = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)
        self.aT = torch.zeros((T, N), dtype=torch.int64, device=device)
        self.aB = torch.zeros((T, N), dtype=torch.int64, device=device)
        self.logp = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.val = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.rew = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.done = torch.zeros((T, N), dtype=torch.float32, device=device)

    def compute_gae(self, last_val: torch.Tensor, gamma: float, lam: float):
        adv = torch.zeros_like(self.rew)
        last_gae = torch.zeros((self.N,), dtype=torch.float32, device=self.device)

        for t in reversed(range(self.T)):
            nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == self.T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae

        ret = adv + self.val
        adv_flat = adv.reshape(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        return adv, ret


def ppo_train(
    envs: List,
    model: ActorCriticMasked,
    cfg: PPOConfig,
    total_updates: int = 1200,
    seed: int = 0,
    log_every: int = 20,
    err_thresh_g: float = 0.01,
    csv_path: str = "logs/train_metrics.csv",
) -> ActorCriticMasked:
    set_seed(seed)
    dev = torch.device(cfg.device)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    N = cfg.n_envs
    T = cfg.rollout_steps
    obs_dim = envs[0].obs_dim

    # --- logging setup ---
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_exists = os.path.exists(csv_path)
    f_csv = open(csv_path, "a", newline="")
    writer = csv.DictWriter(
        f_csv,
        fieldnames=[
            "update",
            "episodes_finished",
            "mean_return",
            "rmse_g",
            "cat_rate_g",
            "mean_post_std_g",
            "probe_rate",
            "pi_loss", 
            "v_loss", 
            "entropy", 
            "total_loss",
        ],
    )
    if not csv_exists:
        writer.writeheader()

    # per-env episodic return accumulator
    ep_return = np.zeros((N,), dtype=np.float64)

    obs_np = np.stack([e.reset() for e in envs], axis=0)
    obs = torch.tensor(obs_np, dtype=torch.float32, device=dev)

    # windowed episode-end stats (since last log)
    win_returns: List[float] = []
    win_g_errs: List[float] = []
    win_g_stds: List[float] = []

    try:
        for upd in range(1, total_updates + 1):
            buf = RolloutBuffer(T=T, N=N, obs_dim=obs_dim, device=cfg.device)

            probe_hits = 0
            steps = 0

            for t in range(T):
                buf.obs[t] = obs

                aT, aB, logp, v = model.act(obs)
                buf.aT[t] = aT
                buf.aB[t] = aB
                buf.logp[t] = logp
                buf.val[t] = v

                next_obs_np = np.zeros((N, obs_dim), dtype=np.float32)
                rew_np = np.zeros((N,), dtype=np.float32)
                done_np = np.zeros((N,), dtype=np.float32)

                for i, env in enumerate(envs):
                    o, r, d, info = env.step(int(aT[i].item()), int(aB[i].item()))
                    next_obs_np[i] = o
                    rew_np[i] = r
                    done_np[i] = float(d)

                    ep_return[i] += float(r)

                    probe_hits += int(bool(info.get("probe_now", False)))
                    steps += 1

                    if d:
                        # episode-end metrics
                        true_g = float(info["true_g"])
                        g_map = float(env.belief.g_map())
                        g_err = g_map - true_g
                        g_std = float(info["post_std_g"])

                        win_returns.append(float(ep_return[i]))
                        win_g_errs.append(g_err)
                        win_g_stds.append(g_std)

                        ep_return[i] = 0.0
                        next_obs_np[i] = env.reset()

                buf.rew[t] = torch.tensor(rew_np, device=dev)
                buf.done[t] = torch.tensor(done_np, device=dev)
                obs = torch.tensor(next_obs_np, dtype=torch.float32, device=dev)

            with torch.no_grad():
                _, _, last_val = model.forward(obs)

            adv, ret = buf.compute_gae(last_val=last_val, gamma=cfg.gamma, lam=cfg.lam)

            B = T * N
            obs_b = buf.obs.reshape(B, obs_dim)
            aT_b = buf.aT.reshape(B)
            aB_b = buf.aB.reshape(B)
            logp_old_b = buf.logp.reshape(B)
            adv_b = adv.reshape(B)
            ret_b = ret.reshape(B)

            idxs = torch.arange(B, device=dev)

            pi_losses = []
            v_losses = []
            entropies = []
            total_losses = []


            for _ in range(cfg.update_epochs):
                perm = idxs[torch.randperm(B)]
                for start in range(0, B, cfg.minibatch_size):
                    mb = perm[start:start + cfg.minibatch_size]
                    mb_obs = obs_b[mb]
                    mb_aT = aT_b[mb]
                    mb_aB = aB_b[mb]
                    mb_logp_old = logp_old_b[mb]
                    mb_adv = adv_b[mb]
                    mb_ret = ret_b[mb]

                    logp_new, ent, v = model.logprob_entropy_value(mb_obs, mb_aT, mb_aB)
                    ratio = torch.exp(logp_new - mb_logp_old)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    v_loss = 0.5 * (mb_ret - v).pow(2).mean()
                    ent_loss = -ent.mean()

                    loss = pi_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

                    pi_losses.append(float(pi_loss.item()))
                    v_losses.append(float(v_loss.item()))
                    entropies.append(float(ent.mean().item()))
                    total_losses.append(float(loss.item()))

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()

            # --- log every K updates ---
            if upd % log_every == 0:
                probe_rate = float(probe_hits) / max(1, steps)

                if len(win_g_errs) > 0:
                    errs = np.asarray(win_g_errs, dtype=np.float64)
                    stds = np.asarray(win_g_stds, dtype=np.float64)
                    rets = np.asarray(win_returns, dtype=np.float64)

                    rmse_g = float(np.sqrt(np.mean(errs ** 2)))
                    cat_rate_g = float(np.mean(np.abs(errs) > float(err_thresh_g)))
                    mean_post_std_g = float(np.mean(stds))
                    mean_return = float(np.mean(rets))
                    ep_count = int(len(errs))
                else:
                    rmse_g = float("nan")
                    cat_rate_g = float("nan")
                    mean_post_std_g = float("nan")
                    mean_return = float("nan")
                    ep_count = 0

                print(
                    f"upd {upd:04d} | eps={ep_count:3d} | rmse_g={rmse_g:.4e} "
                    f"| cat_rate={cat_rate_g:.3f} | mean_std_g={mean_post_std_g:.4e} "
                    f"| mean_ret={mean_return:.3f} | probe_rate={probe_rate:.3f}"
                )

                writer.writerow(
                    {
                        "update": upd,
                        "episodes_finished": ep_count,
                        "mean_return": mean_return,
                        "rmse_g": rmse_g,
                        "cat_rate_g": cat_rate_g,
                        "mean_post_std_g": mean_post_std_g,
                        "probe_rate": probe_rate,
                        "pi_loss": float(np.mean(pi_losses)) if len(pi_losses) else float("nan"),
                        "v_loss": float(np.mean(v_losses)) if len(v_losses) else float("nan"),
                        "entropy": float(np.mean(entropies)) if len(entropies) else float("nan"),
                        "total_loss": float(np.mean(total_losses)) if len(total_losses) else float("nan"),
                    }
                )
                f_csv.flush()

                # clear window
                win_returns.clear()
                win_g_errs.clear()
                win_g_stds.clear()

        return model

    finally:
        f_csv.close()


@torch.no_grad()
def evaluate_policy(
    env,
    model: ActorCriticMasked,
    episodes: int = 80,
    device: str = "cpu",
    err_thresh_g: float = 0.01,   # <-- catastrophic threshold (m/s^2)
    top_peaks: int = 5,           # <-- how many posterior peaks to print
    verbose_cats: bool = True,    # <-- print details for catastrophic episodes
    A_low_thresh: float = 0.10,        # define “low visibility”
    overconf_std_g_max: float = 0.005, # define “overconfident” (tight posterior)
    policy_mode: str = "sample",  # "sample" or "greedy"
) -> Dict[str, float]:
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    # g_errs, g_stds = [], []
    g_err_map_list, g_err_mean_list, g_err_med_list = [], [], []
    g_stds = []

    dt_errs, dt_stds = [], []
    phi_errs = []

    # --- per-episode action/phase diagnostics ---
    k_max_list, k_mean_list, k_p95_list, k_med_list = [], [], [], []  
    cat_flags = []

    # --- visibility diagnostics (A_true) ---
    A_mean_list, A_min_list, A_frac_low_list = [], [], []            

    # --- posterior diagnostics ---
    ent_g_list, ess_g_list = [], []

    cat_count = 0
    overconf_cat_count = 0

    # --- A) mask pressure diagnostics ---
    valid_frac_mean_list, valid_frac_min_list = [], []
    valid_count_min_list = []       
                 

    for ep in range(episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=dev)
        done = False
        last = None

        # track k during the episode
        k_vals = []
        A_vals = []

        valid_fracs = []
        valid_counts = []

        stdg_trace = []
        probe_trace = []    

        while not done:

            mask = model._compute_mask(obs.unsqueeze(0)).squeeze(0)  # [A]
            valid_count = int(mask.sum().item())
            valid_frac = float(valid_count) / float(mask.numel())

            valid_fracs.append(valid_frac)
            valid_counts.append(valid_count)

            # aT, aB = model.greedy(obs)
            if policy_mode == "greedy":
                aT, aB = model.greedy(obs)
            elif policy_mode == "sample":
                aT_t, aB_t, _, _ = model.act(obs.unsqueeze(0))  # act expects [B, obs_dim]
                aT, aB = int(aT_t.item()), int(aB_t.item())
            else:
                raise ValueError(f"Unknown policy_mode={policy_mode}. Use 'greedy' or 'sample'.")
                                 
            obs_np, _, done, info = env.step(aT, aB)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=dev)
            last = info

            stdg_trace.append(float(info.get("post_std_g", np.nan)))
            probe_trace.append(bool(info.get("probe_now", False)))

            # track A_true (visibility)
            if "A_true" in info:
                A_vals.append(float(info["A_true"]))

            # C) compute k per step (prefer info if you added it; else recompute)
            if "k_g_nom" in info:
                k_vals.append(float(info["k_g_nom"]))
            else:
                k_vals.append(float(env.model.k_g(float(info["T_s"]), float(info["Bp_kTm"]))))

        assert last is not None

        if len(valid_fracs) > 0:
            vf = np.asarray(valid_fracs, dtype=np.float64)
            vc = np.asarray(valid_counts, dtype=np.int64)
            valid_frac_mean_list.append(float(np.mean(vf)))
            valid_frac_min_list.append(float(np.min(vf)))
            valid_count_min_list.append(int(np.min(vc)))
        else:
            valid_frac_mean_list.append(float("nan"))
            valid_frac_min_list.append(float("nan"))
            valid_count_min_list.append(float("nan"))

        # --- baseline metrics ---
        # g_err = float(last["post_mu_g"] - last["true_g"])
        true_g = float(last["true_g"])

        g_mean = float(last["post_mu_g"])
        g_med  = float(env.belief.g_median())
        g_map  = float(env.belief.g_map())
        
        err_mean = g_mean - true_g
        err_med  = g_med  - true_g
        err_map  = g_map  - true_g

    
        g_std = float(last["post_std_g"])

        g_err_mean_list.append(err_mean)
        g_err_med_list.append(err_med)
        g_err_map_list.append(err_map)
        g_stds.append(g_std)

        dt_err = float(last["post_mu_dt"] - last["true_dt"])
        dt_std = float(last["post_std_dt"])

        dphi = float((last["post_mu_phi"] - last["true_phi"] + np.pi) % (2.0 * np.pi) - np.pi)
        dt_errs.append(dt_err)
        dt_stds.append(dt_std)
        phi_errs.append(dphi)

        # --- C) episode k summaries ---
        if len(k_vals) > 0:
            kv = np.asarray(k_vals, dtype=np.float64)
            k_max = float(np.max(kv))
            k_mean = float(np.mean(kv))
            k_p95 = float(np.percentile(kv, 95))
            k_med = float(np.median(kv))
        else:
            k_max, k_mean, k_p95, k_med = 0.0, 0.0, 0.0, 0.0

        k_max_list.append(k_max)
        k_mean_list.append(k_mean)
        k_p95_list.append(k_p95)
        k_med_list.append(k_med)

        # --- A_true visibility summaries ---
        if len(A_vals) > 0:
            Av = np.asarray(A_vals, dtype=np.float64)
            A_mean = float(np.mean(Av))
            A_min = float(np.min(Av))
            A_frac_low = float(np.mean(Av < float(A_low_thresh)))
        else:
            A_mean, A_min, A_frac_low = float("nan"), float("nan"), float("nan")

        A_mean_list.append(A_mean)
        A_min_list.append(A_min)
        A_frac_low_list.append(A_frac_low)

        # --- B) posterior diagnostics at episode end ---
        p = env.belief.w_g_marginal().astype(np.float64)
        p = p / (np.sum(p) + 1e-300)

        ent_g = float(-np.sum(p * np.log(p + 1e-300)))
        ess_g = float(1.0 / (np.sum(p * p) + 1e-300))

        ent_g_list.append(ent_g)
        ess_g_list.append(ess_g)

        # --- A) catastrophic detection ---
        # is_cat = abs(g_err) > float(err_thresh_g)
        true_g = float(last["true_g"])
        g_map  = float(env.belief.g_map())
        is_cat = abs(err_map) > float(err_thresh_g)
        
        is_overconf_cat = bool(is_cat and (g_std < float(overconf_std_g_max)))  # <-- ADD
        cat_flags.append(bool(is_cat))
        if is_cat:
            cat_count += 1

            if verbose_cats:
                # Top posterior peaks
                K = int(min(top_peaks, p.size))
                idx = np.argpartition(p, -K)[-K:]
                idx = idx[np.argsort(p[idx])[::-1]]

                peaks = [(float(env.belief.g_grid[i]), float(p[i])) for i in idx]

                print("\n[CAT EPISODE]")
                print(f"  ep={ep} | true_g={true_g:.6f} | post_mu_g={g_mean:.6f} | err_map={err_map:+.6f}")
                
                print(f"  g_mean={g_mean:.6f} | g_med={g_med:.6f} | g_map={g_map:.6f} | true_g={true_g:.6f}")
                print(f"  err_mean={g_mean-true_g:+.6f} | err_med={g_med-true_g:+.6f} | err_map={g_map-true_g:+.6f}")

                print(f"  post_std_g={g_std:.6f} | entropy_g={ent_g:.3f} | ess_g={ess_g:.1f}")
                print(f"  k_mean={k_mean:.3e} | k_p95={k_p95:.3e} | k_max={k_max:.3e}")
                print("  top posterior peaks (g, prob):")
                for gg, pp in peaks:
                    print(f"    g={gg:.6f}  p={pp:.3e}")
                print("  first steps: t | probe | std_g | k_g_nom | valid_count | valid_frac | A_true")
                Tshow = min(10, len(k_vals), len(stdg_trace), len(probe_trace), len(valid_fracs), len(A_vals))
                for t in range(Tshow):
                    print(
                        f"    {t:02d} | {int(probe_trace[t])} | {stdg_trace[t]:.3e} | "
                        f"{k_vals[t]:.3e} | {valid_counts[t]:5d} | {valid_fracs[t]:.3e} | {A_vals[t]:.3f}"
                    )
                print(f"  A_mean={A_mean:.3f} | A_min={A_min:.3f} | frac(A<{A_low_thresh})={A_frac_low:.3f}")
                print(f"  overconf_cat={is_overconf_cat} (std_g<{overconf_std_g_max})")

        if is_overconf_cat:
            overconf_cat_count += 1

    # convert to arrays
    g_err_map = np.asarray(g_err_map_list, dtype=np.float64)
    g_err_mean = np.asarray(g_err_mean_list, dtype=np.float64)
    g_err_med = np.asarray(g_err_med_list, dtype=np.float64)
    g_stds = np.asarray(g_stds, dtype=np.float64)
    dt_errs = np.asarray(dt_errs, dtype=np.float64)
    phi_errs = np.asarray(phi_errs, dtype=np.float64)
    dt_stds = np.asarray(dt_stds, dtype=np.float64)

    k_max_arr = np.asarray(k_max_list, dtype=np.float64)
    k_mean_arr = np.asarray(k_mean_list, dtype=np.float64)
    k_p95_arr = np.asarray(k_p95_list, dtype=np.float64)
    k_med_arr = np.asarray(k_med_list, dtype=np.float64)           

    A_mean_arr = np.asarray(A_mean_list, dtype=np.float64)         
    A_min_arr = np.asarray(A_min_list, dtype=np.float64)          
    A_frac_low_arr = np.asarray(A_frac_low_list, dtype=np.float64) 
    ent_g_arr = np.asarray(ent_g_list, dtype=np.float64)
    ess_g_arr = np.asarray(ess_g_list, dtype=np.float64)
    cat_arr = np.asarray(cat_flags, dtype=bool)

    valid_frac_mean_arr = np.asarray(valid_frac_mean_list, dtype=np.float64)
    valid_frac_min_arr = np.asarray(valid_frac_min_list, dtype=np.float64)
    valid_count_min_arr = np.asarray(valid_count_min_list, dtype=np.float64)

    # --- C) correlations (safe) ---
    abs_err = np.abs(g_err_map)
    sigma = g_stds + 1e-12
    z = abs_err / sigma

    coverage_1s = float(np.mean(abs_err <= 1.0 * sigma))
    coverage_2s = float(np.mean(abs_err <= 2.0 * sigma))
    coverage_3s = float(np.mean(abs_err <= 3.0 * sigma))

    z_median = float(np.median(z))
    z_p95 = float(np.percentile(z, 95))
    z_mean = float(np.mean(z))
    def safe_corr(x, y):
        if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    corr_abs_err_kmax = safe_corr(abs_err, k_max_arr)
    corr_abs_err_kp95 = safe_corr(abs_err, k_p95_arr)
    corr_abs_err_kmean = safe_corr(abs_err, k_mean_arr)
    
    def nanmean(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")
    
    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x)))

    rmse_g_map  = rmse(g_err_map)
    rmse_g_mean = rmse(g_err_mean)
    rmse_g_med  = rmse(g_err_med)

    cat_rate_map  = float(np.mean(np.abs(g_err_map)  > err_thresh_g))
    cat_rate_mean = float(np.mean(np.abs(g_err_mean) > err_thresh_g))
    cat_rate_med  = float(np.mean(np.abs(g_err_med)  > err_thresh_g))
    
    # compare k between catastrophic vs not
    if np.any(cat_arr) and np.any(~cat_arr):
        kmax_cat = float(np.mean(k_max_arr[cat_arr]))
        kmax_ok = float(np.mean(k_max_arr[~cat_arr]))
        Amean_cat = nanmean(A_mean_arr[cat_arr])
        Amean_ok  = nanmean(A_mean_arr[~cat_arr])
        Alow_cat  = nanmean(A_frac_low_arr[cat_arr])
        Alow_ok   = nanmean(A_frac_low_arr[~cat_arr])
        vfmean_cat = nanmean(valid_frac_mean_arr[cat_arr])
        vfmean_ok  = nanmean(valid_frac_mean_arr[~cat_arr])
        vfmin_cat  = nanmean(valid_frac_min_arr[cat_arr])
        vfmin_ok   = nanmean(valid_frac_min_arr[~cat_arr])
        vcountmin_cat = nanmean(valid_count_min_arr[cat_arr])
        vcountmin_ok  = nanmean(valid_count_min_arr[~cat_arr])
    else:
        kmax_cat = kmax_ok = float("nan")
        Amean_cat = Amean_ok = float("nan")
        Alow_cat = Alow_ok = float("nan")

        vfmean_cat = vfmean_ok = float("nan")
        vfmin_cat = vfmin_ok = float("nan")
        vcountmin_cat = vcountmin_ok = float("nan")

    return {
        # --- g metrics (3 estimators) ---
        "rmse_g_map": rmse_g_map,
        "rmse_g_mean": rmse_g_mean,
        "rmse_g_median": rmse_g_med,

        "cat_thresh_g": float(err_thresh_g),
        "cat_rate_g_map": cat_rate_map,
        "cat_rate_g_mean": cat_rate_mean,
        "cat_rate_g_median": cat_rate_med,

        "rmse_g": rmse_g_map,
        "mean_post_std_g": float(np.mean(g_stds)),
        "rmse_dt": float(np.sqrt(np.mean(dt_errs ** 2))),
        "mean_post_std_dt": float(np.mean(dt_stds)),
        "mean_abs_phi_err_rad": float(np.mean(np.abs(phi_errs))),

        "cat_count_g": int(cat_count),
        "cat_rate_g": cat_rate_map,

        "mean_entropy_g": float(np.mean(ent_g_arr)),
        "mean_ess_g": float(np.mean(ess_g_arr)),

        "corr_abs_err_kmax": corr_abs_err_kmax,
        "corr_abs_err_kp95": corr_abs_err_kp95,
        "corr_abs_err_kmean": corr_abs_err_kmean,
        "kmax_mean_cat": kmax_cat,
        "kmax_mean_ok": kmax_ok,

        "kmax_median": float(np.median(k_max_arr)),
        "kmean_median": float(np.median(k_mean_arr)),
        "kp95_median": float(np.median(k_p95_arr)),
        "kmedian_median": float(np.median(k_med_arr)),

        "overconf_std_g_max": float(overconf_std_g_max),
        "overconf_cat_count_g": int(overconf_cat_count),
        "overconf_cat_rate_g": float(overconf_cat_count) / max(1, int(episodes)),

        "A_low_thresh": float(A_low_thresh),
        "A_mean": nanmean(A_mean_arr),
        "A_min_mean": nanmean(A_min_arr),
        "A_frac_low_mean": nanmean(A_frac_low_arr),
        "A_mean_cat": Amean_cat,
        "A_mean_ok": Amean_ok,
        "A_frac_low_cat": Alow_cat,
        "A_frac_low_ok": Alow_ok,

        "valid_frac_mean": nanmean(valid_frac_mean_arr),
        "valid_frac_min_mean": nanmean(valid_frac_min_arr),
        "valid_count_min_mean": nanmean(valid_count_min_arr),

        "valid_frac_mean_cat": vfmean_cat,
        "valid_frac_mean_ok": vfmean_ok,
        "valid_frac_min_cat": vfmin_cat,
        "valid_frac_min_ok": vfmin_ok,
        "valid_count_min_cat": vcountmin_cat,
        "valid_count_min_ok": vcountmin_ok,

        "coverage_1sigma": coverage_1s,
        "coverage_2sigma": coverage_2s,
        "coverage_3sigma": coverage_3s,
        "z_abs_err_over_std_mean": z_mean,
        "z_abs_err_over_std_median": z_median,
        "z_abs_err_over_std_p95": z_p95,
    }