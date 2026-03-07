#student_policy.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from environment import (
    AdaptiveBayesController,
    GravimeterEnv,
    JointGravityEpsBelief,
    NoiseConfig,
    PriorConfig,
)


# ============================================================
# Small utilities
# ============================================================


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sample_global_true_eps(noise: NoiseConfig, seed: int) -> float:
    b = float(max(noise.mfg_rel_noise_bound, 0.0))
    if b <= 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-b, +b))


def _topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    if logits.ndim != 2:
        return 0.0
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    ok = (topk == y[:, None]).any(dim=1).float().mean().item()
    return float(ok)


def _eps_stats_from_logw(eps_grid: np.ndarray, logw_eps: np.ndarray) -> dict:
    eps_grid = np.asarray(eps_grid, dtype=np.float64).reshape(-1)
    logw_eps = np.asarray(logw_eps, dtype=np.float64).reshape(-1)

    m = float(np.max(logw_eps))
    w = np.exp(logw_eps - m)
    w = w / (np.sum(w) + 1e-300)

    mu = float(np.sum(w * eps_grid))
    ex2 = float(np.sum(w * (eps_grid ** 2)))
    std = float(np.sqrt(max(ex2 - mu * mu, 0.0)))
    eps_map = float(eps_grid[int(np.argmax(w))])

    cdf = np.cumsum(w)
    q05 = float(np.interp(0.05, cdf, eps_grid))
    q50 = float(np.interp(0.50, cdf, eps_grid))
    q95 = float(np.interp(0.95, cdf, eps_grid))

    return {
        "global_eps_mean": mu,
        "global_eps_std": std,
        "global_eps_map": eps_map,
        "global_eps_q05": q05,
        "global_eps_q50": q50,
        "global_eps_q95": q95,
    }


def _metrics_from_infos(infos: list[dict]) -> Dict[str, float]:
    err_mean = np.asarray([x["g_err_mean"] for x in infos], dtype=np.float64)
    err_median = np.asarray([x["g_err_median"] for x in infos], dtype=np.float64)
    err_map = np.asarray([x["g_err_map"] for x in infos], dtype=np.float64)

    eps_err_mean = np.asarray([x["eps_err_mean"] for x in infos], dtype=np.float64)
    eps_err_map = np.asarray([x["eps_err_map"] for x in infos], dtype=np.float64)

    std_g = np.asarray([x["post_std_g"] for x in infos], dtype=np.float64)
    std_eps = np.asarray([x["post_std_eps"] for x in infos], dtype=np.float64)

    ent_g = np.asarray([x["post_entropy_g"] for x in infos], dtype=np.float64)
    ent_joint = np.asarray([x["post_entropy_joint"] for x in infos], dtype=np.float64)
    cyc = np.asarray([x["cycle_time_s"] for x in infos], dtype=np.float64)

    return {
        "rmse_g_mean": float(np.sqrt(np.mean(err_mean ** 2))),
        "rmse_g_median": float(np.sqrt(np.mean(err_median ** 2))),
        "rmse_g_map": float(np.sqrt(np.mean(err_map ** 2))),
        "mean_post_std_g": float(np.mean(std_g)),
        "mean_post_std_eps": float(np.mean(std_eps)),
        "mean_entropy_g": float(np.mean(ent_g)),
        "mean_entropy_joint": float(np.mean(ent_joint)),
        "rmse_eps_mean": float(np.sqrt(np.mean(eps_err_mean ** 2))),
        "rmse_eps_map": float(np.sqrt(np.mean(eps_err_map ** 2))),
        "mean_cycle_time_s_last_shot": float(np.mean(cyc)),
    }


# ============================================================
# Feature extraction
# ============================================================


@dataclass(frozen=True)
class FeatureConfig:
    g_hist_bins: int = 64
    eps_hist_bins: int = 31
    include_peak_features: bool = True
    include_step_features: bool = True


class BeliefFeaturizer:
    """
    Convert JointGravityEpsBelief -> fixed-size feature vector.

    Design:
    - downsampled p(g) marginal
    - downsampled p(eps) marginal
    - compact scalar summaries

    This keeps the student policy simple but still lets it see
    multimodality and uncertainty.
    """

    def __init__(self, prior: PriorConfig, noise: NoiseConfig, cfg: FeatureConfig):
        self.prior = prior
        self.noise = noise
        self.cfg = cfg

        self.g_lo = float(prior.g_range[0])
        self.g_hi = float(prior.g_range[1])
        self.g_mid = 0.5 * (self.g_lo + self.g_hi)
        self.g_halfspan = 0.5 * (self.g_hi - self.g_lo) + 1e-12

        self.eps_bound = float(max(noise.mfg_rel_noise_bound, 0.0))
        self.eps_scale = self.eps_bound if self.eps_bound > 0.0 else 1.0

    @staticmethod
    def _resample_prob(p: np.ndarray, out_n: int) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        out_n = int(max(1, out_n))

        if p.size == out_n:
            q = p.copy()
        elif p.size == 1:
            q = np.full(out_n, p[0], dtype=np.float64)
        else:
            src_x = np.linspace(0.0, 1.0, p.size, dtype=np.float64)
            dst_x = np.linspace(0.0, 1.0, out_n, dtype=np.float64)
            q = np.interp(dst_x, src_x, p)

        q = np.maximum(q, 0.0)
        s = float(np.sum(q))
        if s <= 0.0 or (not np.isfinite(s)):
            q = np.full(out_n, 1.0 / out_n, dtype=np.float64)
        else:
            q = q / s

        return q.astype(np.float32)

    def _norm_g(self, x: float) -> float:
        return float((x - self.g_mid) / self.g_halfspan)

    def _norm_g_width(self, x: float) -> float:
        return float(x / self.g_halfspan)

    def _norm_eps(self, x: float) -> float:
        return float(x / self.eps_scale)

    def transform(
        self,
        belief: JointGravityEpsBelief,
        step_idx: int,
        episode_len: int,
    ) -> np.ndarray:
        pg = belief.w_g_marginal()
        pe = belief.w_eps_marginal()

        pg_ds = self._resample_prob(pg, self.cfg.g_hist_bins)
        pe_ds = self._resample_prob(pe, self.cfg.eps_hist_bins)

        g_q05, g_q95 = belief.credible_interval_g(mass=0.90)
        eps_q05, eps_q95 = belief.credible_interval_eps(mass=0.90)

        g_mean = belief.mean_g()
        g_std = belief.std_g()
        g_median = belief.median_g()
        g_map = belief.map_g()

        eps_mean = belief.mean_eps()
        eps_std = belief.std_eps()
        eps_median = belief.median_eps()
        eps_map = belief.map_eps()

        g_entropy = belief.entropy_g_nats()
        joint_entropy = belief.entropy_joint_nats()

        g_entropy_norm = float(g_entropy / max(math.log(max(2, belief.n_g_grid)), 1e-12))
        joint_entropy_norm = float(
            joint_entropy / max(math.log(max(2, belief.n_g_grid * belief.n_eps_grid)), 1e-12)
        )

        scalar_list = []

        if self.cfg.include_step_features:
            step_frac = float(step_idx / max(1, episode_len))
            remain_frac = float((episode_len - step_idx) / max(1, episode_len))
            scalar_list += [step_frac, remain_frac]

        scalar_list += [
            self._norm_g(g_mean),
            self._norm_g_width(g_std),
            self._norm_g(g_median),
            self._norm_g(g_map),
            self._norm_g_width(0.5 * (g_q95 - g_q05)),
            self._norm_eps(eps_mean),
            self._norm_eps(eps_std),
            self._norm_eps(eps_median),
            self._norm_eps(eps_map),
            self._norm_eps(0.5 * (eps_q95 - eps_q05)),
            g_entropy_norm,
            joint_entropy_norm,
        ]

        if self.cfg.include_peak_features:
            p1, p2, gap = belief.peak_stats_g()
            scalar_list += [float(p1), float(p2), float(gap)]

        scalars = np.asarray(scalar_list, dtype=np.float32)
        x = np.concatenate([pg_ds, pe_ds, scalars], axis=0).astype(np.float32)
        return x

    def feature_dim(self) -> int:
        dummy_prior = self.prior
        dummy_noise = self.noise
        dummy_belief = JointGravityEpsBelief(
            prior=dummy_prior,
            noise=dummy_noise,
            n_g_grid=128,
            n_eps_grid=max(3, self.cfg.eps_hist_bins),
        )
        dummy_belief.reset_factorized()
        x = self.transform(dummy_belief, step_idx=0, episode_len=32)
        return int(x.size)


# ============================================================
# Teacher dataset generation
# ============================================================


@dataclass(frozen=True)
class DatasetConfig:
    n_runs: int = 256
    episodes_per_run: int = 8
    base_seed: int = 0
    log_every_runs: int = 16


def generate_teacher_dataset(
    env_ctor: Callable[[int], GravimeterEnv],
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    n_g_grid: int,
    n_eps_grid: int,
    feature_cfg: FeatureConfig,
    dataset_cfg: DatasetConfig,
) -> dict:
    """
    Generate supervised data from the teacher planner.

    Important:
    - one run = one fixed global_true_eps
    - within a run, the global eps posterior is carried across episodes
    - each training sample = one decision point inside an episode
    """
    featurizer = BeliefFeaturizer(prior=prior, noise=noise, cfg=feature_cfg)

    tmp_belief = JointGravityEpsBelief(
        prior=prior,
        noise=noise,
        n_g_grid=n_g_grid,
        n_eps_grid=n_eps_grid,
    )

    xs: list[np.ndarray] = []
    y_action: list[int] = []
    sample_run_ids: list[int] = []
    run_records: list[dict] = []

    for run_idx in range(int(dataset_cfg.n_runs)):
        global_true_eps = _sample_global_true_eps(
            noise=noise,
            seed=int(dataset_cfg.base_seed + 100_003 * run_idx + 17),
        )
        logw_eps_global = tmp_belief.uniform_logw_eps()

        samples_before = len(xs)

        for ep_idx in range(int(dataset_cfg.episodes_per_run)):
            env = env_ctor(int(dataset_cfg.base_seed + 1_000_000 * run_idx + ep_idx))
            env.global_true_eps = float(global_true_eps)
            env.reset(logw_eps_global=logw_eps_global)

            done = False
            while not done:
                x = featurizer.transform(
                    belief=env.belief,
                    step_idx=env.t,
                    episode_len=env.cfg.episode_len,
                )

                aT, aB, phi, _A, _score = controller.plan_action(env.belief)
                action_idx = env.grid.encode(aT, aB)

                xs.append(x)
                y_action.append(int(action_idx))
                sample_run_ids.append(int(run_idx))

                done, _info = env.step(aT, aB, phi)

            logw_eps_global = env.belief.logw_eps_marginal()

        run_records.append(
            {
                "run_idx": int(run_idx),
                "global_true_eps": float(global_true_eps),
                "n_samples_from_run": int(len(xs) - samples_before),
                **_eps_stats_from_logw(tmp_belief.eps_grid, logw_eps_global),
            }
        )

        if dataset_cfg.log_every_runs > 0 and ((run_idx + 1) % dataset_cfg.log_every_runs == 0):
            print(
                f"[Dataset] run {run_idx + 1}/{dataset_cfg.n_runs} | "
                f"total_samples={len(xs)} | "
                f"last_global_true_eps={global_true_eps:+.6f}"
            )

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(y_action, dtype=np.int64)
    run_ids = np.asarray(sample_run_ids, dtype=np.int64)

    return {
        "X": X,
        "y_action": y,
        "run_ids": run_ids,
        "run_records": run_records,
        "feature_cfg": asdict(feature_cfg),
        "dataset_cfg": asdict(dataset_cfg),
        "n_actions": int(controller.grid.n_actions),
        "input_dim": int(X.shape[1]),
    }


# ============================================================
# Student model
# ============================================================


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 1024
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.05
    val_fraction: float = 0.10
    label_smoothing: float = 0.0
    early_stop_patience: int = 8
    seed: int = 0


class StudentPolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        d_in = int(input_dim)
        depth = int(max(1, depth))

        for _ in range(depth):
            layers += [
                nn.Linear(d_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d_in = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(d_in, int(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits = self.action_head(z)
        return logits


@dataclass
class TrainedStudent:
    model: StudentPolicyNet
    x_mean: np.ndarray
    x_std: np.ndarray
    feature_cfg: FeatureConfig


def _make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    train_cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(train_cfg.seed)

    run_ids = np.asarray(run_ids, dtype=np.int64).reshape(-1)
    if X.shape[0] != run_ids.size:
        raise ValueError("run_ids must have one entry per training sample")

    unique_runs = np.unique(run_ids)
    if unique_runs.size < 2:
        raise ValueError("Need at least two unique runs for a run-level train/val split")

    rng.shuffle(unique_runs)

    n_val_runs = int(max(1, round(train_cfg.val_fraction * unique_runs.size)))
    n_val_runs = min(n_val_runs, unique_runs.size - 1)

    val_runs = unique_runs[:n_val_runs]
    is_val = np.isin(run_ids, val_runs)

    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    x_mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    X_train_n = ((X_train - x_mean) / x_std).astype(np.float32)
    X_val_n = ((X_val - x_mean) / x_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(y_val.astype(np.int64)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        drop_last=False,
    )

    split_meta = {
        "n_train_runs": int(unique_runs.size - n_val_runs),
        "n_val_runs": int(n_val_runs),
        "n_train_samples": int(train_idx.size),
        "n_val_samples": int(val_idx.size),
    }

    return train_loader, val_loader, x_mean, x_std, split_meta


def train_student_policy(
    X: np.ndarray,
    y_action: np.ndarray,
    run_ids: np.ndarray,
    n_actions: int,
    feature_cfg: FeatureConfig,
    train_cfg: TrainConfig,
) -> tuple[TrainedStudent, dict]:
    device = torch.device(train_cfg.device)
    train_loader, val_loader, x_mean, x_std, split_meta = _make_dataloaders(
        X,
        y_action,
        run_ids,
        train_cfg,
    )

    model = StudentPolicyNet(
        input_dim=int(X.shape[1]),
        n_actions=int(n_actions),
        hidden_dim=int(train_cfg.hidden_dim),
        depth=int(train_cfg.depth),
        dropout=float(train_cfg.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.label_smoothing))

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc_top1": [],
        "val_acc_top3": [],
        **split_meta,
    }

    for epoch in range(int(train_cfg.epochs)):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_logits = []
        all_y = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                bs = int(xb.shape[0])
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

                all_logits.append(logits.detach().cpu())
                all_y.append(yb.detach().cpu())

        val_loss = val_loss_sum / max(1, val_count)
        logits_cat = torch.cat(all_logits, dim=0)
        y_cat = torch.cat(all_y, dim=0)

        val_acc1 = _topk_accuracy(logits_cat, y_cat, k=1)
        val_acc3 = _topk_accuracy(logits_cat, y_cat, k=3)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc_top1"].append(float(val_acc1))
        history["val_acc_top3"].append(float(val_acc3))

        print(
            f"[Train] epoch {epoch + 1:03d}/{train_cfg.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc1={val_acc1:.4f} | "
            f"val_acc3={val_acc3:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch + 1)
            no_improve = 0
            best_state = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= int(train_cfg.early_stop_patience):
                print(f"[Train] early stop at epoch {epoch + 1}")
                break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state["model_state_dict"])

    trained = TrainedStudent(
        model=model,
        x_mean=x_mean,
        x_std=x_std,
        feature_cfg=feature_cfg,
    )
    history["best_epoch"] = int(best_epoch)
    history["best_val_loss"] = float(best_val_loss)
    return trained, history


def save_student_checkpoint(
    path: str,
    trained: TrainedStudent,
    n_actions: int,
    train_cfg: TrainConfig,
) -> None:
    payload = {
        "model_state_dict": trained.model.state_dict(),
        "input_dim": int(trained.x_mean.size),
        "n_actions": int(n_actions),
        "x_mean": trained.x_mean,
        "x_std": trained.x_std,
        "feature_cfg": asdict(trained.feature_cfg),
        "train_cfg": asdict(train_cfg),
    }
    torch.save(payload, path)


def load_student_checkpoint(path: str, device: str = "cpu") -> TrainedStudent:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    feature_cfg = FeatureConfig(**ckpt["feature_cfg"])
    train_cfg = ckpt["train_cfg"]

    model = StudentPolicyNet(
        input_dim=int(ckpt["input_dim"]),
        n_actions=int(ckpt["n_actions"]),
        hidden_dim=int(train_cfg["hidden_dim"]),
        depth=int(train_cfg["depth"]),
        dropout=float(train_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return TrainedStudent(
        model=model,
        x_mean=np.asarray(ckpt["x_mean"], dtype=np.float32),
        x_std=np.asarray(ckpt["x_std"], dtype=np.float32),
        feature_cfg=feature_cfg,
    )


# ============================================================
# Teacher / student rollout
# ============================================================


def _student_action_from_belief(
    trained: TrainedStudent,
    prior: PriorConfig,
    noise: NoiseConfig,
    belief: JointGravityEpsBelief,
    step_idx: int,
    episode_len: int,
    device: str = "cpu",
) -> int:
    featurizer = BeliefFeaturizer(prior=prior, noise=noise, cfg=trained.feature_cfg)
    x = featurizer.transform(
        belief=belief,
        step_idx=step_idx,
        episode_len=episode_len,
    )
    x_n = ((x - trained.x_mean) / trained.x_std).astype(np.float32)
    xb = torch.from_numpy(x_n[None, :]).to(device)

    trained.model.eval()
    with torch.no_grad():
        logits = trained.model(xb)
        action_idx = int(torch.argmax(logits, dim=1).item())
    return action_idx


def run_teacher_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    logw_eps_global: np.ndarray,
    store_trace: bool = False,
) -> tuple[dict, list[dict], np.ndarray]:
    env.reset(logw_eps_global=logw_eps_global)
    trace: list[dict] = []
    last_info: dict | None = None

    done = False
    while not done:
        aT, aB, phi, A_plan, score = controller.plan_action(env.belief)
        action_idx = env.grid.encode(aT, aB)

        done, info = env.step(aT, aB, phi)
        last_info = info

        if store_trace:
            trace.append(
                {
                    "step": int(info["t"]),
                    "policy_type": "teacher",
                    "action_idx": int(action_idx),
                    "aT": int(aT),
                    "aB": int(aB),
                    "T_s": float(info["T_s"]),
                    "Bp_nom_kTm": float(info["Bp_nom_kTm"]),
                    "Bp_true_kTm": float(info["Bp_true_kTm"]),
                    "mw_phase_rad": float(phi),
                    "A_plan": float(A_plan),
                    "A_true": float(info["A_true"]),
                    "score": float(score),
                    "outcome_plus": int(info["outcome_plus"]),
                    "true_g": float(info["true_g"]),
                    "true_eps": float(info["true_eps"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_mu_eps": float(info["post_mu_eps"]),
                    "post_std_eps": float(info["post_std_eps"]),
                    "post_map_eps": float(info["post_map_eps"]),
                }
            )

    assert last_info is not None
    new_logw_eps_global = env.belief.logw_eps_marginal()
    return last_info, trace, new_logw_eps_global


def run_student_episode(
    env: GravimeterEnv,
    controller: AdaptiveBayesController,
    trained: TrainedStudent,
    prior: PriorConfig,
    noise: NoiseConfig,
    logw_eps_global: np.ndarray,
    device: str = "cpu",
    store_trace: bool = False,
) -> tuple[dict, list[dict], np.ndarray]:
    env.reset(logw_eps_global=logw_eps_global)
    trace: list[dict] = []
    last_info: dict | None = None

    done = False
    while not done:
        action_idx = _student_action_from_belief(
            trained=trained,
            prior=prior,
            noise=noise,
            belief=env.belief,
            step_idx=env.t,
            episode_len=env.cfg.episode_len,
            device=device,
        )
        aT, aB = env.grid.decode(action_idx)

        T_s = float(env.grid.T_vals_s[aT])
        Bp_nom_kTm = float(env.grid.Bp_vals_kTm[aB])
        phi = controller.choose_phase(env.belief, T_s, Bp_nom_kTm)

        done, info = env.step(aT, aB, phi)
        last_info = info

        if store_trace:
            trace.append(
                {
                    "step": int(info["t"]),
                    "policy_type": "student",
                    "action_idx": int(action_idx),
                    "aT": int(aT),
                    "aB": int(aB),
                    "T_s": float(info["T_s"]),
                    "Bp_nom_kTm": float(info["Bp_nom_kTm"]),
                    "Bp_true_kTm": float(info["Bp_true_kTm"]),
                    "mw_phase_rad": float(phi),
                    "A_true": float(info["A_true"]),
                    "outcome_plus": int(info["outcome_plus"]),
                    "true_g": float(info["true_g"]),
                    "true_eps": float(info["true_eps"]),
                    "post_mu_g": float(info["post_mu_g"]),
                    "post_std_g": float(info["post_std_g"]),
                    "post_map_g": float(info["post_map_g"]),
                    "post_mu_eps": float(info["post_mu_eps"]),
                    "post_std_eps": float(info["post_std_eps"]),
                    "post_map_eps": float(info["post_map_eps"]),
                }
            )

    assert last_info is not None
    new_logw_eps_global = env.belief.logw_eps_marginal()
    return last_info, trace, new_logw_eps_global


# ============================================================
# Policy evaluation over hierarchical runs
# ============================================================


@dataclass(frozen=True)
class EvalConfig:
    n_runs: int = 32
    episodes_per_run: int = 16
    base_seed: int = 1234
    log_every_runs: int = 4

    # new diagnostics
    wrong_branch_peak_abs_err_threshold: float = 5e-3
    unresolved_halfwidth90_threshold: float = 5e-3
    truth_mass_band_abs: float = 1e-3


def _posterior_diagnostics(
    env: GravimeterEnv,
    true_g: float,
    top_k: int = 5,
    truth_mass_band_abs: float = 1e-3,
) -> dict:
    belief = env.belief

    g_q05, g_q95 = belief.credible_interval_g(mass=0.90)
    p1, p2, gap = belief.peak_stats_g()
    peaks = belief.top_peaks_g(k=top_k)
    eps_q05, eps_q95 = belief.credible_interval_eps(mass=0.90)

    wg = belief.w_g_marginal()
    truth_mask = np.abs(belief.g_grid - float(true_g)) <= float(truth_mass_band_abs)
    truth_mass_band = float(np.sum(wg[truth_mask])) if np.any(truth_mask) else 0.0
    nearest_peak_abs_err = float(
        min(abs(float(g) - float(true_g)) for g, _p in peaks)
    ) if len(peaks) > 0 else float("inf")

    truth_in_90ci = float(float(g_q05) <= float(true_g) <= float(g_q95))

    return {
        "g_q05": float(g_q05),
        "g_q50": float(belief.median_g()),
        "g_q95": float(g_q95),
        "g_halfwidth90": float(0.5 * (g_q95 - g_q05)),
        "peak1_prob": float(p1),
        "peak2_prob": float(p2),
        "peak_gap": float(gap),
        "top_peaks_g": [{"g": float(g), "p": float(p)} for g, p in peaks],
        "eps_q05": float(eps_q05),
        "eps_q50": float(belief.median_eps()),
        "eps_q95": float(eps_q95),
        "truth_in_90ci": float(truth_in_90ci),
        "truth_mass_band": float(truth_mass_band),
        "nearest_topk_peak_abs_err": float(nearest_peak_abs_err),
        "coarse_entropy_g": float(belief.coarse_entropy_g(n_bins=16)),
    }


def _diagnostic_metrics(diag_records: list[dict]) -> dict:
    if len(diag_records) == 0:
        return {
            "rmse_g_nearest_topk_peak": float("nan"),
            "truth_in_90ci_rate": float("nan"),
            "mean_truth_mass_band": float("nan"),
            "mean_g_halfwidth90": float("nan"),
            "n_wrong_branch_episodes": 0,
            "n_unresolved_episodes": 0,
            "n_catastrophic_episodes": 0,
        }

    nearest_peak_err = np.asarray(
        [x["nearest_topk_peak_abs_err"] for x in diag_records],
        dtype=np.float64,
    )
    truth_in_90ci = np.asarray(
        [x["truth_in_90ci"] for x in diag_records],
        dtype=np.float64,
    )
    truth_mass_band = np.asarray(
        [x["truth_mass_band"] for x in diag_records],
        dtype=np.float64,
    )
    halfwidth = np.asarray(
        [x["g_halfwidth90"] for x in diag_records],
        dtype=np.float64,
    )
    wrong_branch = np.asarray(
        [x["wrong_branch_flag"] for x in diag_records],
        dtype=np.float64,
    )
    unresolved = np.asarray(
        [x["unresolved_flag"] for x in diag_records],
        dtype=np.float64,
    )
    catastrophic = np.asarray(
        [x["catastrophic_flag"] for x in diag_records],
        dtype=np.float64,
    )

    return {
        "rmse_g_nearest_topk_peak": float(np.sqrt(np.mean(nearest_peak_err ** 2))),
        "truth_in_90ci_rate": float(np.mean(truth_in_90ci)),
        "mean_truth_mass_band": float(np.mean(truth_mass_band)),
        "mean_g_halfwidth90": float(np.mean(halfwidth)),
        "n_wrong_branch_episodes": int(np.sum(wrong_branch > 0.5)),
        "n_unresolved_episodes": int(np.sum(unresolved > 0.5)),
        "n_catastrophic_episodes": int(np.sum(catastrophic > 0.5)),
    }


def evaluate_teacher_policy(
    env_ctor: Callable[[int], GravimeterEnv],
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    eval_cfg: EvalConfig,
    return_first_trace: bool = False,
) -> tuple[dict, list[dict] | None, list[dict]]:
    infos: list[dict] = []
    diag_records: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []
    run_records: list[dict] = []

    for run_idx in range(int(eval_cfg.n_runs)):
        global_true_eps = _sample_global_true_eps(
            noise=noise,
            seed=int(eval_cfg.base_seed + 97_919 * run_idx + 11),
        )

        tmp = env_ctor(int(eval_cfg.base_seed + 10_000_000 + run_idx))
        logw_eps_global = tmp.belief.uniform_logw_eps()

        for ep_idx in range(int(eval_cfg.episodes_per_run)):
            env = env_ctor(int(eval_cfg.base_seed + 1_000_000 * run_idx + ep_idx))
            env.global_true_eps = float(global_true_eps)

            store_trace = bool(return_first_trace and first_trace is None and run_idx == 0 and ep_idx == 0)
            final_info, trace, logw_eps_global = run_teacher_episode(
                env=env,
                controller=controller,
                logw_eps_global=logw_eps_global,
                store_trace=store_trace,
            )
            infos.append(final_info)

            if store_trace:
                first_trace = trace

            diag = _posterior_diagnostics(
                env,
                true_g=float(final_info["true_g"]),
                top_k=5,
                truth_mass_band_abs=eval_cfg.truth_mass_band_abs,
            )

            wrong_branch = bool(
                diag["nearest_topk_peak_abs_err"] > float(eval_cfg.wrong_branch_peak_abs_err_threshold)
            )
            unresolved = bool(
                (diag["g_halfwidth90"] > float(eval_cfg.unresolved_halfwidth90_threshold))
                and (diag["truth_in_90ci"] > 0.5)
            )
            catastrophic = bool(wrong_branch or unresolved)

            diag["wrong_branch_flag"] = float(wrong_branch)
            diag["unresolved_flag"] = float(unresolved)
            diag["catastrophic_flag"] = float(catastrophic)
            diag_records.append(diag)

            if catastrophic:
                catastrophic_records.append(
                    {
                        "run_idx": int(run_idx),
                        "episode_idx": int(ep_idx),
                        "true_g": float(final_info["true_g"]),
                        "true_eps": float(final_info["true_eps"]),
                        "post_mu_g": float(final_info["post_mu_g"]),
                        "post_median_g": float(final_info["post_median_g"]),
                        "post_map_g": float(final_info["post_map_g"]),
                        "post_mu_eps": float(final_info["post_mu_eps"]),
                        "post_map_eps": float(final_info["post_map_eps"]),
                        "abs_err_mean": float(abs(final_info["g_err_mean"])),
                        "abs_err_median": float(abs(final_info["g_err_median"])),
                        "abs_err_map": float(abs(final_info["g_err_map"])),
                        **diag,
                    }
                )

        run_records.append(
            {
                "run_idx": int(run_idx),
                "global_true_eps": float(global_true_eps),
                **_eps_stats_from_logw(tmp.belief.eps_grid, logw_eps_global),
            }
        )

        if eval_cfg.log_every_runs > 0 and ((run_idx + 1) % eval_cfg.log_every_runs == 0):
            cur = _metrics_from_infos(infos)
            print(
                f"[Teacher Eval] run {run_idx + 1}/{eval_cfg.n_runs} | "
                f"episodes_done={(run_idx + 1) * eval_cfg.episodes_per_run} | "
                f"rmse_mean={cur['rmse_g_mean']:.6e} | "
                f"rmse_map={cur['rmse_g_map']:.6e}"
            )

    metrics = _metrics_from_infos(infos)
    metrics.update(_diagnostic_metrics(diag_records))

    eps_mean_err = []
    eps_map_err = []
    eps_std_final = []

    for rec in run_records:
        eps_mean_err.append(abs(rec["global_eps_mean"] - rec["global_true_eps"]))
        eps_map_err.append(abs(rec["global_eps_map"] - rec["global_true_eps"]))
        eps_std_final.append(rec["global_eps_std"])

    metrics["global_eps_abs_err_mean_final"] = float(np.mean(eps_mean_err))
    metrics["global_eps_abs_err_map_final"] = float(np.mean(eps_map_err))
    metrics["global_eps_std_final_mean"] = float(np.mean(eps_std_final))
    metrics["n_total_episodes"] = int(eval_cfg.n_runs * eval_cfg.episodes_per_run)

    return metrics, first_trace, catastrophic_records

def evaluate_student_policy(
    env_ctor: Callable[[int], GravimeterEnv],
    prior: PriorConfig,
    noise: NoiseConfig,
    controller: AdaptiveBayesController,
    trained: TrainedStudent,
    eval_cfg: EvalConfig,
    device: str = "cpu",
    return_first_trace: bool = False,
) -> tuple[dict, list[dict] | None, list[dict]]:
    infos: list[dict] = []
    diag_records: list[dict] = []
    first_trace: list[dict] | None = None
    catastrophic_records: list[dict] = []
    run_records: list[dict] = []

    for run_idx in range(int(eval_cfg.n_runs)):
        global_true_eps = _sample_global_true_eps(
            noise=noise,
            seed=int(eval_cfg.base_seed + 97_919 * run_idx + 11),
        )

        tmp = env_ctor(int(eval_cfg.base_seed + 20_000_000 + run_idx))
        logw_eps_global = tmp.belief.uniform_logw_eps()

        for ep_idx in range(int(eval_cfg.episodes_per_run)):
            env = env_ctor(int(eval_cfg.base_seed + 2_000_000 * run_idx + ep_idx))
            env.global_true_eps = float(global_true_eps)

            store_trace = bool(return_first_trace and first_trace is None and run_idx == 0 and ep_idx == 0)
            final_info, trace, logw_eps_global = run_student_episode(
                env=env,
                controller=controller,
                trained=trained,
                prior=prior,
                noise=noise,
                logw_eps_global=logw_eps_global,
                device=device,
                store_trace=store_trace,
            )
            infos.append(final_info)

            if store_trace:
                first_trace = trace

            diag = _posterior_diagnostics(
                env,
                true_g=float(final_info["true_g"]),
                top_k=5,
                truth_mass_band_abs=eval_cfg.truth_mass_band_abs,
            )

            wrong_branch = bool(
                diag["nearest_topk_peak_abs_err"] > float(eval_cfg.wrong_branch_peak_abs_err_threshold)
            )
            unresolved = bool(
                (diag["g_halfwidth90"] > float(eval_cfg.unresolved_halfwidth90_threshold))
                and (diag["truth_in_90ci"] > 0.5)
            )
            catastrophic = bool(wrong_branch or unresolved)

            diag["wrong_branch_flag"] = float(wrong_branch)
            diag["unresolved_flag"] = float(unresolved)
            diag["catastrophic_flag"] = float(catastrophic)
            diag_records.append(diag)

            if catastrophic:
                catastrophic_records.append(
                    {
                        "run_idx": int(run_idx),
                        "episode_idx": int(ep_idx),
                        "true_g": float(final_info["true_g"]),
                        "true_eps": float(final_info["true_eps"]),
                        "post_mu_g": float(final_info["post_mu_g"]),
                        "post_median_g": float(final_info["post_median_g"]),
                        "post_map_g": float(final_info["post_map_g"]),
                        "post_mu_eps": float(final_info["post_mu_eps"]),
                        "post_map_eps": float(final_info["post_map_eps"]),
                        "abs_err_mean": float(abs(final_info["g_err_mean"])),
                        "abs_err_median": float(abs(final_info["g_err_median"])),
                        "abs_err_map": float(abs(final_info["g_err_map"])),
                        **diag,
                    }
                )

        run_records.append(
            {
                "run_idx": int(run_idx),
                "global_true_eps": float(global_true_eps),
                **_eps_stats_from_logw(tmp.belief.eps_grid, logw_eps_global),
            }
        )

        if eval_cfg.log_every_runs > 0 and ((run_idx + 1) % eval_cfg.log_every_runs == 0):
            cur = _metrics_from_infos(infos)
            print(
                f"[Student Eval] run {run_idx + 1}/{eval_cfg.n_runs} | "
                f"episodes_done={(run_idx + 1) * eval_cfg.episodes_per_run} | "
                f"rmse_mean={cur['rmse_g_mean']:.6e} | "
                f"rmse_map={cur['rmse_g_map']:.6e}"
            )

    metrics = _metrics_from_infos(infos)
    metrics.update(_diagnostic_metrics(diag_records))

    eps_mean_err = []
    eps_map_err = []
    eps_std_final = []

    for rec in run_records:
        eps_mean_err.append(abs(rec["global_eps_mean"] - rec["global_true_eps"]))
        eps_map_err.append(abs(rec["global_eps_map"] - rec["global_true_eps"]))
        eps_std_final.append(rec["global_eps_std"])

    metrics["global_eps_abs_err_mean_final"] = float(np.mean(eps_mean_err))
    metrics["global_eps_abs_err_map_final"] = float(np.mean(eps_map_err))
    metrics["global_eps_std_final_mean"] = float(np.mean(eps_std_final))
    metrics["n_total_episodes"] = int(eval_cfg.n_runs * eval_cfg.episodes_per_run)

    return metrics, first_trace, catastrophic_records