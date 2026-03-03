# Gravimeter RL (paper-faithful fixed-mass version)

This repo contains a cleaned-up version of your RBPF+RL pipeline aligned to:
- Main text phase model: ΔΦ = (2η/y0) g T^2 + 16π η_g η
- Supplement visibility: A(η,δt) from Eq. S65
- Optional MFG amplitude noise: B' -> B'(1+ε), ε~N(0, sigma_B_rel), marginalized in the filter likelihood

Design choices:
- Optimize **accuracy per shot** (reward is log-variance reduction per shot)
- MW analysis phase φ_MW is **not** part of the RL action; it is chosen by a fast analytic phase-lock rule.

## Files
- `run_pipeline.py` : end-to-end dataset generation + BC + PPO + eval
- `pl_brl/env_rbpf.py` : environment + proper RBPF
- `pl_brl/rl_pipeline.py` : masked ActorCritic + BC + PPO

## How to run
From this folder:

```bash
python run_pipeline.py
```

Checkpoints will be saved to `./checkpoints/`.

## Key physics parameters
In `run_pipeline.py`:
- mass is fixed using r=100 nm, rho=3510 kg/m^3 -> M≈1.47e-17 kg
- sigma_B_rel=0.10 enables 10% shot-to-shot MFG amplitude noise
