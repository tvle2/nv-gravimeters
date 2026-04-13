# Multi-Particle-Filter Bank Redesign for Quantum Gravity Sensor
## Replacing the IMM Approach with a Principled Multi-PF Architecture

**Document status:** Architectural design specification  
**Author:** Quantum Sensing Research Group, NC State University  
**Target system:** Levitated NV-center nanodiamond gravimeter  
**Replaces:** `GravityWrapIMMFilter` in `gravimeter_model_imm_wrap.py`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why IMM is Fundamentally Wrong for Phase Wrapping](#2-why-imm-is-fundamentally-wrong-for-phase-wrapping)
3. [Literature Foundation](#3-literature-foundation)
4. [Proposed Architecture: Multi-PF Bank](#4-proposed-architecture-multi-pf-bank)
5. [Detailed Design: Multi-PF Bank](#5-detailed-design-multi-pf-bank)
6. [Code Architecture](#6-code-architecture)
7. [Comparison: IMM vs. Multi-PF Bank](#7-comparison-imm-vs-multi-pf-bank)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [References](#9-references)

---

## 1. Executive Summary

### The Problem

The quantum gravity sensor measures gravitational acceleration `g` via the optical transition probability of a levitated NV-center nanodiamond:

```
p(click=1 | g, T, Bp, φ_MW) = 0.5 * (1 + vis * cos(k_g(T, Bp) * g + φ_MW))
```

The gain `k_g(T, Bp)` — a function of interrogation time `T` and magnetic field gradient `Bp` — can be tuned to be very large, making the cosine likelihood wrap many times over the prior range of `g`. At large gain, the g-prior `[g_lo, g_hi]` spans `K = ceil(k_g * (g_hi - g_lo) / (2π))` complete fringes, yielding a posterior with up to K sharp, nearly identical peaks. This **severe phase wrapping** means the posterior is highly multimodal, making standard single-Gaussian Bayesian inference fail catastrophically.

### The Current Failure

The current `GravityWrapIMMFilter` attempts to handle this with an Interacting Multiple Model (IMM) filter: 128 sub-filters tiling the g-prior, with Set Transformer encoding and a protocol library of fixed-gain settings. **This architecture is built on a category error.** IMM is designed for _dynamical regime switching_ (e.g., a target switching from straight flight to a turn). Phase wrapping is not a dynamical regime — it is epistemic uncertainty about which fringe the true g occupies. Applying IMM conflates a filtering problem (what state is the system in?) with an inference problem (which fringe is the parameter in?). Every piece of IMM-specific machinery — the transition matrix, the mixing step, the mode interaction — has no physical justification and actively degrades performance.

### The Solution

Replace the IMM with a **Bank of Particle Filters (Multi-PF Bank)**, implementing the Multiple Model Adaptive Estimation (MMAE) architecture, combined with:

1. **A single qsensoropt `ParticleFilter` per fringe hypothesis.** Each PF covers one fringe — one contiguous interval `[g_k^lo, g_k^hi]` of width `2π/k_g` where the likelihood has a single peak.
2. **Bayesian weight updates on mode (fringe) probabilities.** Mode weights are updated by Bayes rule: `q_k ∝ q_k * p(y | mode k)`. No mixing step. No transition matrix.
3. **Coarse-to-fine gain scheduling.** The RL policy learns to start with low-gain (small `k_g`) measurements, where the likelihood has few or zero fringes and a standard single-PF works. As the posterior narrows, the policy ramps up gain, and new fringe-specific PFs are created only when needed.
4. **Maximum reuse of qsensoropt.** The physical model (`GravityStatelessPhysicalModel`) is unchanged. The multi-PF bank wraps qsensoropt `ParticleFilter` instances. The training loop is unchanged. Only `generate_input()` and `loss_function()` are subclassed.
5. **Holevo variance as the loss function.** The MSE of a bimodal posterior is the mean of two fringes — a value that may have zero probability mass. Holevo variance `V_H = |μ|⁻² - 1` where `μ = |E[e^{i k_ref g}]|` correctly measures circular estimation error.

This design is mathematically rigorous, backed by the MMAE literature for multi-hypothesis Bayesian inference, the MFL literature for coarse-to-fine adaptive sensing, and directly implements Holevo's theory of circular phase estimation. It requires no new training infrastructure beyond what qsensoropt already provides.

---

## 2. Why IMM is Fundamentally Wrong for Phase Wrapping

### 2.1 IMM Assumes Regime Switching; Phase Wrapping is Epistemic Uncertainty

The Interacting Multiple Model filter was invented by Blom and Bar-Shalom (1988) to track targets whose dynamics switch between a small set of known modes — for example, constant-velocity flight versus a coordinated turn. In IMM, each mode is a different _process model_ for the system's dynamics. The system genuinely transitions between these modes over time, and the mode indicator is itself a latent Markov chain.

Phase wrapping in a gravity sensor is a fundamentally different problem. There is one true value of g. The measurement model `p(click=1 | g, T, Bp, φ_MW) = 0.5*(1 + vis*cos(k_g*g + φ_MW))` has a cosine likelihood that is `2π/k_g`-periodic in g. After a measurement at gain `k_g`, the posterior has K peaks, one in each fringe. This multimodality is **purely epistemic** — it reflects the sensor's inability to distinguish fringes, not any physical switching in the underlying system. The true g is fixed and does not transition between fringes.

Applying IMM to this situation makes the following physical errors:

- **The transition matrix has no meaning.** In IMM, the `K×K` Markov transition matrix `P_ij` encodes the probability that the system switches from mode i to mode j. For phase wrapping, this would mean "the probability that g jumps from fringe i to fringe j between measurements." There is no such jump — g is constant. A transition matrix with non-zero off-diagonal elements actively injects false state-switching noise into the estimator, causing it to assign probability mass to the wrong fringes over time.

- **The problem is identification, not tracking.** The goal is to identify which fringe contains the true g, then measure it precisely within that fringe. MMAE (parallel Bayesian filters, no mixing) is the correct framework. IMM adds interaction between models that is physically unmotivated.

### 2.2 The Mixing Step Destroys Information

The IMM algorithm interleaves a **mixing step** before each update: prior to applying measurement likelihood, each sub-filter's particle set is mixed according to the transition matrix. Formally:

```
μ̃_j^0 = Σ_i μ_ij * x̄_i^{t-1}    (mixed mean for mode j)
P̃_j^0 = Σ_i μ_ij * [P_i + (x̄_i - μ̃_j^0)(x̄_i - μ̃_j^0)^T]   (mixed covariance)
```

where `μ_ij = P(mode j at t | mode i at t-1, y^{1:t-1})` are the mixing weights.

For phase wrapping, this mixing step is devastating:

1. **It blends particles from different fringes.** Sub-filter for fringe 0 (g ≈ g_0) receives a contribution from sub-filter for fringe 1 (g ≈ g_0 + 2π/k_g). The mixed particle cloud spans both fringes — it is no longer localized within the fringe it is supposed to track.
2. **It corrupts the local approximation.** Each sub-filter's value derives entirely from the assumption that it covers one fringe. Injecting particles from neighboring fringes makes the local Gaussian approximation invalid.
3. **It is information-destroying.** MMAE mode weights after mixing lose the sharp contrast between fringes that had low vs. high likelihood. Disambiguation becomes slower.

In contrast, MMAE — the simpler architecture that IMM was designed to improve upon for dynamical systems — simply runs K filters in parallel with no mixing, and weights them by likelihood. For the static parameter estimation problem (constant g, identify the fringe), MMAE is exactly right.

### 2.3 Static Mode Geometry vs. Dynamic Fringe Structure

The current implementation tiles the g-prior with `MAX_MODES=128` fixed-width intervals, splitting when phase span exceeds a threshold. This creates a mode geometry that is independent of the actual likelihood periodicity.

The natural periodicity of the likelihood at gain `k_g` is `2π/k_g`. Modes that do not align with this periodicity will straddle fringe boundaries, putting particles from two adjacent likelihood peaks into the same sub-filter. The sub-filter then applies a Bayes update using its local particles' likelihoods, but those particles span two peaks. The resulting weight update misrepresents the likelihood and the sub-filter cannot converge to either peak cleanly.

The correct mode geometry is **fringe-aligned**: fringe k covers the interval `[g_lo + k * 2π/k_g, g_lo + (k+1) * 2π/k_g]` for k = 0, 1, ..., K-1, where `K = ceil(k_g * (g_hi - g_lo) / (2π))`. This geometry is defined by the measurement gain, not by a static tiling.

Furthermore, the number of fringes changes with each choice of gain. At low gain (short T), K may be 1-3. At high gain, K may be 100+. The mode count must track the gain, not be fixed at 128.

### 2.4 Particle Budget Fragmentation

The current design allocates `particles_per_mode = 16` particles to each of up to 128 modes, for a total of 2048 particles. This 2048-particle budget is divided across modes regardless of their relevance.

The effective sample size (ESS) within each sub-filter is at most 16. With 16 particles per fringe, the local approximation of the within-fringe posterior is very crude. In contrast:

- **If only 1 fringe is plausible** (after disambiguation), all 2048 particles should concentrate in that fringe, giving ESS up to 2048 for fine-grained estimation.
- **If K fringes are equally plausible**, the budget should be divided K ways adaptively: ~2048/K particles per fringe.

The current fixed allocation cannot do this. Even when the RL policy has successfully disambiguated to 2 fringes, 126 × 16 = 2016 particles remain wasted on irrelevant modes. The Multi-PF bank with adaptive allocation corrects this exactly.

### 2.5 Unnecessary Complexity: The Set Transformer Controller

The current design uses a Set Transformer network to encode 128 mode tokens (each a (μ_k, σ_k, q_k) triple) into a permutation-invariant summary for the RL policy. Set Transformers use multi-head attention across the mode set, with quadratic complexity in the number of modes: O(K² × d_model). For K=128, this is a large attention computation at every measurement step, every training iteration.

This complexity adds:
- Larger parameter count → harder to train, more regularization needed
- Longer forward/backward pass → slower iteration cycles
- Architecture mismatch: qsensoropt's standard `StatelessMetrology.generate_input()` produces a simple (μ, Σ, step, resources) input vector and works robustly in practice for single-PF settings

The Multi-PF bank replaces the Set Transformer with a compact augmented summary vector:
- Per-mode quantities aggregated into: number of active modes, entropy of mode weights, top-2 mode weights, Holevo variance of the marginal posterior, mean and log-standard-deviation of each within-mode posterior.
- Total input size: O(1) per mode summary + O(d) for aggregate statistics = small fixed-size vector.
- This plugs directly into qsensoropt's existing MLP controller.

### 2.6 Failure to Reuse qsensoropt Infrastructure

The `GravityWrapIMMFilter` effectively reimplements its own particle management (splitting, merging, mode-level weight arrays) outside of qsensoropt's `ParticleFilter` class. This forfeits:

- **The Ścibior-Wood trick**: differentiable resampling that lets gradients flow through the particle filter. Without this, gradient-based training of the RL policy cannot correctly propagate error signals through the measurement-update-resample chain.
- **ESS-based resampling triggers**: qsensoropt's `full_resampling` and `partial_resampling` intelligently decide when to resample, maintaining particle diversity without unnecessary computational cost.
- **Liu-West jittering**: the Gaussian perturbation step that spreads particles around the posterior mean after resampling, preventing particle collapse.
- **TF `while_loop` compatibility**: qsensoropt's simulation loop runs inside a TensorFlow `while_loop` for efficiency. Custom data structures (Python lists of sub-filters) are not graph-mode compatible.
- **Automated training infrastructure**: `utils.train()`, `performance_evaluation()`, `store_input_control()` all operate on the `Simulation` interface and would need to be reimplemented for a custom filter.

The Multi-PF bank wraps each fringe hypothesis as a native qsensoropt `ParticleFilter` instance, recovering all of these benefits automatically.

---

## 3. Literature Foundation

### 3.1 Joas et al. (2021) — Adaptive Quantum Sensing via MFL

**Paper:** "Online adaptive quantum characterization of a nuclear spin"  
**URL:** https://www.nature.com/articles/s41534-021-00389-z  
**Authors:** T. Joas, A. M. Waeber, G. Braunbeck, F. Reinhard  
**Published:** npj Quantum Information, 2021

**Key contribution:** Demonstrated online adaptive Bayesian estimation of a DC magnetic field using a single NV-center spin, achieving a **110× dynamic range** improvement over fixed-τ Ramsey spectroscopy while maintaining near-theoretical sensitivity (SMS ≈ 132 nT/√Hz).

**Relevance to this redesign:**

The MFL algorithm implements exactly the coarse-to-fine gain scheduling advocated here. In early epochs, the algorithm selects short τ (low gain `k_g ∝ τ`), producing a likelihood that is slowly-varying and has few fringes over the prior range. The posterior updates cleanly using a standard particle filter. As the posterior narrows, τ is adaptively increased, and the likelihood periodicity decreases relative to the (now narrower) posterior. By the time τ approaches T₂*/2, the posterior has converged to a single fringe.

The critical insight: **at low gain, there is no multimodality problem.** A single standard particle filter suffices. The multi-PF bank only becomes necessary when the RL policy deliberately selects high-gain measurements for precision within a nearly-identified fringe. The RL policy learns this schedule automatically from the Holevo variance loss.

Joas et al. also confirm that the MFL approach's adaptive τ selection yields sensitivity close to the theoretical limit (only 2.25× worse than SMS in their experiment), demonstrating that adaptive Bayesian methods can nearly saturate the Cramér-Rao bound for this class of sensing problem.

### 3.2 Yankelev et al. (2020) — Dual-T Disambiguation for Atom Interferometry

**Paper:** "Atom interferometry with thousand-fold increase in dynamic range"  
**URL:** https://www.science.org/doi/10.1126/sciadv.abd0650  
**Authors:** D. Yankelev, C. Avinadav, N. Davidson, O. Firstenberg  
**Published:** Science Advances, 2020

**Key contribution:** Extended the dynamic range of atom interferometric gravity/inertial sensors by a factor of 1000× using simultaneous dual-T (two interrogation times T₁, T₂ = τT₁) measurements in a single shot. The dual-T moiré fringe has a period D × (2π/k₀) where D = (1-τ)⁻¹ ≫ 1, allowing phase unwrapping without compromising single-shot sensitivity.

**Relevance to this redesign:**

Yankelev's approach is a **deterministic multi-scale phase unwrapping** — the antithesis of the stochastic approach here, but it illuminates the physics:

1. **Two measurements at different gains jointly break phase ambiguity.** If k₁ and k₂ are incommensurate, the pair (y₁, y₂) has a combined fringe period much larger than either individually. This is the multi-scale principle.

2. **For the RL-controlled system, the policy can learn to choose (T₁, T₂) pairs** that optimally disambiguate. The multi-PF bank's mode weight update naturally captures this: after two measurements at different gains, only fringes consistent with both measurements survive.

3. **The particle filter on sequential dual-T shots** (Section "Dynamic Tracking" in the paper) is exactly the architecture advocated here: particles represent hypotheses about g, they are propagated and reweighted by measurement likelihood modulo 2π, and particle clusters naturally form around surviving fringe hypotheses.

The paper achieves disambiguation in 2-3 shots across 3 orders of magnitude, confirming that the multi-scale approach is highly efficient. Our RL policy will learn a similar schedule but optimized for the specific sensor characteristics.

### 3.3 Berry et al. (2009) — Holevo Variance for Phase Estimation

**Paper:** "How to perform the most accurate possible phase measurements"  
**URL:** https://arxiv.org/abs/0907.0014  
**Authors:** D. W. Berry, B. L. Sanders  
**Published:** Physical Review A, 2009

**Key contribution:** Established the theoretical framework for optimal adaptive phase estimation, proving that the Holevo variance `V_H = μ⁻² - 1` where `μ = |E[e^{iφ}]|` is the correct metric for circular/wrapped distributions, and derived optimal feedback strategies achieving Heisenberg scaling.

**Relevance to this redesign:**

The standard MSE is `E[(g - ĝ)²]`. For a bimodal posterior with peaks at g₀ and g₀ + 2π/k_ref, the MSE-minimizing estimate is `ĝ = g₀ + π/k_ref` — between the two peaks, in a region of zero posterior probability. Training the RL policy with MSE loss would therefore reward false confidence: the policy might choose measurements that make the marginal posterior appear narrower (smaller MSE of the mean) without actually disambiguating the fringes.

The Holevo variance avoids this:

```
μ = |Σ_k q_k * Σ_j w_{k,j} * exp(i * k_ref * g_{k,j})|

V_H = 1/μ² - 1
```

When the posterior has two equally weighted peaks at g₀ and g₀ + π/k_ref, the complex exponentials `exp(i k_ref g)` at the two peaks point in opposite directions, giving `|μ| ≈ 0` and `V_H → ∞`. The policy is immediately penalized for failing to disambiguate. As disambiguation proceeds and one peak dominates, `|μ| → 1` and `V_H → 0`.

The choice of `k_ref` in the Holevo variance requires care:
- Too small: all phases nearly equal, V_H insensitive to posterior shape.
- Too large: phase wraps multiple times across the posterior, V_H again uninformative.
- **Correct:** `k_ref` should be chosen such that the full posterior width spans approximately one period: `k_ref ≈ 2π / (g_hi - g_lo)` initially, then increased as the posterior narrows.

In practice, use the adaptive `k_ref = 2π / (6σ_g)` where `σ_g` is the current posterior standard deviation, clamped between `2π/(g_hi - g_lo)` and `k_g^max / 2`.

### 3.4 Higgins, Berry et al. (2007) — Heisenberg-Limited Phase Estimation Without Entanglement

**Paper:** "Entanglement-free Heisenberg-limited phase estimation"  
**URL:** https://arxiv.org/abs/0709.2996  
**Authors:** B. L. Higgins, D. W. Berry, S. D. Bartlett, H. M. Wiseman, G. J. Pryde  
**Published:** Nature 450, 393–396, 2007

**Key contribution:** Demonstrated Heisenberg-limited phase estimation using adaptive measurements on individual photons with increasing phase-shift resources (k = 1, 2, 4, 8, ..., 2^{n-1}). This is the canonical example of coarse-to-fine binary phase disambiguation, implementing Kitaev's phase estimation algorithm in a fully adaptive Bayesian setting.

**Relevance to this redesign:**

The binary doubling sequence k₁=1, k₂=2, ..., k_n=2^{n-1} is the optimal deterministic gain schedule for phase disambiguation. At step i, the measurement at gain k_i disambiguates one additional bit of the binary expansion of g. After n steps, the phase is resolved to 1/2^n of the prior range.

For the gravity sensor, the analogous schedule is: start with `k_g = k_min` (few fringes), then ramp up. The RL policy learns a continuous-time generalization of this binary schedule, tuned to the actual noise and decoherence properties of the sensor.

The key lesson: **the problem structure is inherently coarse-to-fine.** Any protocol that randomly interleaves high-gain and low-gain measurements will perform worse than a protocol that consistently resolves from coarse to fine. The IMM approach, with its static mode tiling at all scales simultaneously, misses this structure entirely.

### 3.5 Belliardo et al. (2024) — Model-Aware RL with qsensoropt

**Paper:** "Model-aware reinforcement learning for high-performance Bayesian experimental design in quantum metrology" (arXiv version) / "Applications of model-aware reinforcement learning in Bayesian experimental design in quantum metrology" (journal)  
**URLs:**  
- https://arxiv.org/abs/2312.16985  
- https://link.aps.org/doi/10.1103/PhysRevA.109.062609  
**Authors:** F. Belliardo, F. Zoratti, F. Marquardt, V. Giovannetti  
**Published:** Physical Review A 109, 062609, 2024

**Key contribution:** Introduced the qsensoropt framework for RL-based Bayesian experimental design. The key insight is **model-awareness**: differentiating through the simulated experiment (particle filter Bayes update + physical model) allows gradient-based training of the control policy. The Ścibior-Wood trick makes resampling differentiable. Applied to NV centers, photonic circuits, and optical cavities.

**Relevance to this redesign:**

The Multi-PF bank is designed to be a drop-in replacement for qsensoropt's `ParticleFilter` in the `Simulation` framework. Specifically:

1. **The physical model is unchanged.** `GravityStatelessPhysicalModel` with its `model()`, `perform_measurement()`, `count_resources()` methods is untouched. All derivatives through the physical model remain valid.

2. **The REINFORCE + model-aware gradient** still flows through: NN controls → `perform_measurement()` → `apply_measurement()` → Holevo variance loss. The differentiable chain is preserved.

3. **The `train()` utility function is unchanged.** The `GravityMultiPFSimulation` subclasses `StatelessSimulation` and provides `generate_input()` and `loss_function()`, so `utils.train()` calls `simulation.execute()` inside a `GradientTape` exactly as before.

4. **The RL policy can learn the coarse-to-fine schedule.** The RL agent does not need to be told to start at low gain — it discovers this from the Holevo variance reward signal. A high-gain measurement before disambiguation causes large V_H, while a low-gain measurement allows unambiguous update and smooth V_H reduction. The policy learns to exploit this structure.

### 3.6 van den Berg (2021) — Mixed-Prior Bayesian Phase Estimation

**Paper:** "Efficient Bayesian phase estimation using mixed priors"  
**URL:** https://quantum-journal.org/papers/q-2021-06-07-469/  
**Authors:** E. van den Berg (IBM Quantum)  
**Published:** Quantum, 2021

**Key contribution:** Proposed dynamic switching between Fourier-series and wrapped-Gaussian representations of multimodal phase posteriors. The Fourier representation is exact but grows in complexity with each update; the Gaussian representation is computationally cheap but only valid once the posterior is single-peaked. The algorithm switches representations when the posterior standard deviation falls below a threshold.

**Relevance to this redesign:**

Van den Berg explicitly addresses the same problem: Bayesian estimation with a periodic likelihood function and a prior spanning many periods. The "mixed prior" is a discrete mixture of K wrapped Gaussians (one per fringe), initialized from the Fourier representation of the uniform prior. The weight update rule is precisely the MMAE weight update: `q_k ∝ q_k * p(y | mode k)`.

The particle-filter analogue of van den Berg's algorithm is exactly the Multi-PF bank:
- K particle filters ↔ K Gaussian components
- Mode weight update via Bayes ↔ Gaussian weight update
- Pruning when `q_k < threshold` ↔ dropping components with negligible weight
- Transition from multi-mode to single-mode ↔ switching from Fourier to Gaussian representation

Van den Berg's key observation about initialization is directly applicable: starting K identical Gaussian priors creates a degeneracy — all K update identically, effectively behaving as a single filter. To break symmetry, the K initial particle filters must have **distinct** support (one per fringe). This is the fringe-aligned initialization described in Section 5.2.

### 3.7 Brady et al. (2024) — Differentiable IMM Particle Filter

**Paper:** "Differentiable Interacting Multiple Model Particle Filtering"  
**URL:** https://arxiv.org/abs/2410.00620  
**Authors:** J.-J. Brady, W. Wang, S. Maskell  
**Published:** Signal Processing (preprint 2024, accepted 2025)

**Key contribution:** Proposed the DIMMPF — a differentiable version of the IMM particle filter that allows gradient-based learning of the regime-switching dynamics and individual mode models. The paper proves consistency of IMMPF gradient estimates and demonstrates superior performance on regime-switching benchmark problems.

**Why we do NOT use this:**

Brady et al. carefully motivate DIMMPF for **true regime-switching systems** where the latent mode indicator is a Markov chain with unknown transition dynamics. The paper states:

> "We propose a sequential Monte Carlo algorithm for parameter learning when the studied model exhibits random discontinuous jumps in behaviour."

The gravity sensor does not exhibit random discontinuous jumps. g is constant. The DIMMPF would be the right tool if g itself were switching between discrete values drawn from a Markov chain — but that is a completely different physical system. Applying DIMMPF here would still require learning a meaningless transition matrix for a system that does not transition.

Furthermore, DIMMPF adds the additional complexity of neural-network-parameterized switching dynamics on top of the already complex IMMPF. For a static-parameter estimation problem, this complexity yields no benefit and makes training harder.

The correct takeaway from Brady et al.: differentiable particle filtering is powerful, and gradient-based training through particle filters works. But the model must match the physics. For phase disambiguation of a fixed parameter, use MMAE (parallel independent filters, weight update, pruning), not IMM.

### 3.8 Bonato Group — NV-Center Adaptive Protocols and Fourier Posteriors

**Paper (representative):** "Adaptive Bayesian phase estimation for quantum error correcting codes" and related NV-center sensing protocols  
**Relevant URL:** https://quantum-journal.org/papers/q-2021-06-04-467/  
(Granade et al., "Efficient qubit phase estimation using adaptive measurements," Quantum, 2021)

**Key contribution:** Systematic analysis of phase estimation protocols using covariant estimators and confidence intervals, showing that non-identifiable likelihood functions (i.e., wrapped cosine likelihoods) cause standard MLE methods to fail, and adaptive covariant approaches to succeed. The Fourier posterior representation (keeping the full Fourier transform of the posterior) provides an exact and compact alternative to particle filtering for periodic distributions.

**Relevance to this redesign:**

The Fourier posterior approach is an exact alternative to the particle-filter bank for tracking wrapped posteriors. Each Fourier coefficient `c_k = E[e^{ikφ}]` can be updated analytically after each cosine-likelihood measurement. The computational cost is proportional to the number of retained Fourier coefficients, which grows logarithmically with the number of measurements.

For the RL training setting, however, the particle filter has a decisive advantage: it is differentiable through the Ścibior-Wood trick, while the Fourier update is also differentiable (it is just a sequence of multiplications). Both approaches could in principle be trained with REINFORCE. The particle filter is adopted here because:
1. It plugs directly into qsensoropt's `ParticleFilter` class.
2. It naturally handles non-periodic parameters (g is not periodic — it has hard bounds).
3. It supports the Liu-West jittering step that prevents particle collapse.

A future extension could maintain a Fourier posterior in parallel for rapid analytical updates, switching to particle-filter representation when the posterior is nearly unimodal (mirroring van den Berg's mixed-prior approach).

---

## 4. Proposed Architecture: Multi-PF Bank

### 4.1 Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAVITY MULTI-PF SIMULATION (GravityMultiPFSimulation)   │
│                      (subclasses StatelessSimulation)                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        SIMULATION LOOP (unchanged)                     │ │
│  │   for step in range(num_steps):                                        │ │
│  │     input  = generate_input(bank, step, resources)  ─┐                │ │
│  │     ctrl   = controller_nn(input)                    │                │ │
│  │     y      = perform_measurement(ctrl, g_true)       │ custom         │ │
│  │     bank   = bank.apply_measurement(y, ctrl)         │ code           │ │
│  │     bank   = bank.update_weights()                   │                │ │
│  │     bank   = bank.maybe_prune()                      │                │ │
│  │     bank   = bank.maybe_create_modes(ctrl)           │                │ │
│  │     bank   = bank.reallocate_particles()            ─┘                │ │
│  │     loss  += holevo_variance(bank)                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      MULTI-PF BANK (MultiPFBank)                     │   │
│  │                                                                       │   │
│  │  Mode weights:  q = [q_0, q_1, ..., q_{K-1}]  (K ≤ K_max)          │   │
│  │                                                                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐        │   │
│  │  │  PF[0]   │  │  PF[1]   │  │  PF[2]   │  ...  │  PF[K-1] │        │   │
│  │  │ fringe 0 │  │ fringe 1 │  │ fringe 2 │       │fringe K-1│        │   │
│  │  │ N_0 ptcl │  │ N_1 ptcl │  │ N_2 ptcl │       │ N_{K-1}  │        │   │
│  │  │[g0,g0+Δ] │  │[g0+Δ,   │  │[g0+2Δ,  │       │[..., g1] │        │   │
│  │  │          │  │  g0+2Δ] │  │  g0+3Δ] │       │          │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘       └──────────┘        │   │
│  │      N_k ∝ q_k * N_total       Δ = 2π/k_g                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              PHYSICAL MODEL (unchanged from current code)            │   │
│  │         GravityStatelessPhysicalModel                                │   │
│  │         model(): p(y | g, T, Bp, φ_MW) = 0.5*(1+vis*cos(k_g*g+φ)) │   │
│  │         perform_measurement(): samples binary outcome                │   │
│  │         count_resources(): returns T (interrogation time)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              CONTROLLER (MLP, replaces Set Transformer)              │   │
│  │  input:  [μ_k, log_σ_k, q_k for k in top_K_modes]                   │   │
│  │          + [V_H, H_q, N_active, step_norm, resource_norm]            │   │
│  │  output: [T_s, Bp_kTm, mw_phase_rad]                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Training:  GradientTape → simulation.execute() → Holevo variance loss     │
│             → REINFORCE + model-aware gradients → Adam optimizer            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Multi-PF Bank (MultiPFBank class)

The `MultiPFBank` class is a container that manages K `ParticleFilter` instances, their associated mode weights, and the particle budget allocation across modes.

**Design principle:** Each PF in the bank IS a native qsensoropt `ParticleFilter` instance — not a custom implementation. The bank is purely a meta-level manager that orchestrates how particles are allocated and how mode weights are updated.

**Key state:**

```python
class MultiPFBank:
    """
    Bank of K particle filters for multi-fringe phase disambiguation.
    
    Each mode k corresponds to one fringe: the interval of g values
    [g_lo + k*delta, g_lo + (k+1)*delta] where delta = 2*pi/k_g.
    
    Attributes
    ----------
    particle_filters : list[ParticleFilter]
        One PF per active mode. Each PF has num_particles_k particles
        covering fringe k's g-interval.
    mode_weights : tf.Variable shape (bs, K_max)
        q_k[b, k] = P(true g is in fringe k | data so far), for batch b.
        Normalized: sum_k q_k[b, :] = 1 for each b.
    active_mask : tf.Variable shape (bs, K_max)
        active_mask[b, k] = 1 if mode k is active for batch element b.
    particles_per_mode : list[int]
        Number of particles currently allocated to mode k.
    N_total : int
        Fixed total particle budget across all modes.
    N_min : int
        Minimum particles per active mode (prevents particle starvation).
    phys_model : GravityStatelessPhysicalModel
        Shared physical model used by all PFs.
    """
```

**Mode weight normalization invariant:**

At all times, `sum_k (active_mask[b,k] * mode_weights[b,k]) = 1` for each batch element b. This is maintained after every weight update and pruning step.

**How it wraps qsensoropt:**

Each `PF[k]` is initialized as:

```python
pf_k = ParticleFilter(
    num_particles=N_k,
    phys_model=phys_model,   # same model, but with bounds restricted to fringe k
    resampling_allowed=True,
    resample_threshold=0.5,
    alpha=0.5,
    beta=0.98,
    scibior_trick=True,      # critical: differentiable resampling
    trim=True,
    prec="float64",
)
```

The key parameter is that `phys_model` for `PF[k]` has its `g` parameter bounds restricted to `[g_lo + k*delta, g_lo + (k+1)*delta]`. This is achieved by subclassing `GravityStatelessPhysicalModel` with a modified `Parameter` for g:

```python
class FringeGravityModel(GravityStatelessPhysicalModel):
    def __init__(self, fringe_k, k_g_ref, g_lo, g_hi, **kwargs):
        delta = 2 * np.pi / k_g_ref
        fringe_lo = g_lo + fringe_k * delta
        fringe_hi = g_lo + (fringe_k + 1) * delta
        fringe_hi = min(fringe_hi, g_hi)  # clamp to prior
        g_param = Parameter(bounds=(fringe_lo, fringe_hi), name='g')
        super().__init__(g_param=g_param, **kwargs)
```

**Important:** The fringe-restricted model is used for particle initialization and trimming only. The likelihood function `model()` uses the full cosine formula and is the same for all modes — the distinction is in where particles live, not in the measurement model.

### 4.3 Physical Model (Unchanged)

`GravityStatelessPhysicalModel` remains exactly as implemented. The three required methods are untouched:

```python
class GravityStatelessPhysicalModel(StatelessPhysicalModel):

    def model(self, outcomes, controls, parameters, meas_step, num_systems=1):
        """
        Computes P(y | g, T, Bp, φ_MW).
        
        p(y=1 | g, T, Bp, φ_MW) = 0.5 * (1 + vis * cos(k_g(T, Bp) * g + φ_MW))
        p(y=0 | ...) = 1 - p(y=1 | ...)
        
        Parameters
        ----------
        outcomes : Tensor (bs, num_systems, 1)    # binary: 0 or 1
        controls : Tensor (bs, num_systems, 3)    # [T_s, Bp_kTm, mw_phase_rad]
        parameters : Tensor (bs, num_systems, 1)  # [g]
        
        Returns
        -------
        prob : Tensor (bs, num_systems)            # P(y | g, x)
        """
        # No changes needed here.
        ...

    def perform_measurement(self, controls, parameters, meas_step, rangen):
        """Samples y ~ Bernoulli(p(y=1 | g, T, Bp, φ_MW))."""
        ...

    def count_resources(self, resources, outcomes, controls, true_values, meas_step):
        """Returns T (interrogation time) as the resource spent."""
        ...
```

All gradients through `model()` — which qsensoropt uses for the differentiable Bayes update — remain valid.

### 4.4 Simulation Integration (GravityMultiPFSimulation)

`GravityMultiPFSimulation` subclasses `StatelessSimulation` and provides:
1. A custom `generate_input()` that summarizes the bank state for the NN
2. A custom `loss_function()` implementing Holevo variance

The qsensoropt `Simulation.execute()` loop is reused without modification:

```python
class GravityMultiPFSimulation(StatelessSimulation):
    """
    Multi-PF bank simulation for the gravity sensor.
    
    Subclasses StatelessSimulation, overriding only generate_input()
    and loss_function(). The execute() loop from Simulation is inherited
    and runs without modification.
    
    The particle filter passed to Simulation.__init__() is a "proxy PF"
    (the primary mode's PF). The bank is managed as extra state, threaded
    through generate_input() and loss_function() via instance variables
    updated inside the loop.
    
    Parameters
    ----------
    bank : MultiPFBank
        The multi-PF bank (manages all modes internally).
    phys_model : GravityStatelessPhysicalModel
        Physical model (shared across all modes).
    controller : callable
        Neural network mapping input → (T, Bp, φ_MW).
    simpars : SimulationParameters
    """
    
    def generate_input(self, weights, particles, meas_step, used_resources, rangen):
        """
        Build NN input vector from bank state.
        
        Returns Tensor of shape (bs, input_size).
        See Section 4.5 for the full input vector specification.
        """
        ...

    def loss_function(self, weights, particles, true_values, used_resources, meas_step):
        """
        Holevo variance of the marginal posterior.
        
        Returns Tensor of shape (bs, 1).
        See Section 5.7 for the formula.
        """
        ...
```

**Threading bank state through the qsensoropt loop:**

The qsensoropt `while_loop` tracks `weights, particles, state_ensemble, used_resources, meas_step, sum_log_prob` as loop variables. It does not natively support additional loop variables. The bank state (K mode weights, K × N_k particle arrays) is threaded by:

1. Storing the bank as an instance variable `self.bank`
2. Updating it inside `generate_input()` (which is called at the start of each step, before the Bayes update)
3. Using TensorFlow `tf.Variable` for mode weights so they update in-place

**Alternative approach for full graph-mode compatibility:**

Flatten the bank state into the particle array. Use a shape `(bs, K_max * N_per_mode, 2)` particle tensor where the last dimension holds `(g, mode_id)`. The weight tensor shape `(bs, K_max * N_per_mode)` holds within-mode normalized weights scaled by mode weights. Mode weight updates are derived from the weight tensor by summing within-mode weights. This fully integrates the bank into qsensoropt's particle representation with no external state.

The flattened representation has the advantage of being fully compatible with `tf.while_loop` and XLA compilation, at the cost of some bookkeeping complexity. **This is the recommended implementation for training performance.** See Section 6.3 for the detailed flattened interface.

### 4.5 Controller Network

The controller network is a **standard MLP** (multi-layer perceptron), identical in form to qsensoropt's default controller. No Set Transformer is needed.

**Input vector composition** (input_size = 4*TOP_K + 5):

```
For k in top-K modes (sorted by q_k descending, K=min(N_active, 4)):
  - normalize(μ_k, (g_lo, g_hi))          # normalized mean of mode k
  - log_std(σ_k)                           # log std within mode k (same as StatelessMetrology)
  - q_k                                    # mode weight (already in [0,1])
  - active_k                               # 1 if active, 0 if padded

Global features:
  - V_H                                    # Holevo variance (in [-1, 1] after tanh)
  - H_q = -Σ_k q_k log(q_k)              # entropy of mode weights (measures ambiguity)
  - N_active / K_max                       # fraction of modes active (normalized)
  - meas_step / num_steps                  # normalized step
  - used_resources / max_resources         # normalized resource use
```

**Network architecture:**

```python
controller = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(3, activation='tanh'),  # (T_s, Bp_kTm, mw_phase_rad)
])
```

Output scaling: the tanh output in [-1, 1] is mapped to physical ranges:
- `T_s = denormalize(output[0], (T_min, T_max))`
- `Bp_kTm = denormalize(output[1], (Bp_min, Bp_max))`
- `mw_phase_rad = denormalize(output[2], (0, 2*pi))`

This is identical to qsensoropt's standard controller wiring.

### 4.6 Training

Training reuses qsensoropt's REINFORCE + model-aware training unchanged:

```python
# Standard qsensoropt training setup
simpars = SimulationParameters(
    sim_name='gravity_multi_pf',
    num_steps=NUM_STEPS,
    max_resources=MAX_RESOURCES,
    loss_logl_outcomes=True,   # REINFORCE on binary outcome stochasticity
    loss_logl_controls=False,  # continuous controls
    cumulative_loss=True,      # accumulate Holevo variance at each step
    baseline=True,             # batch-mean baseline for variance reduction
    prec='float64',
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=InverseSqrtDecay(initial_learning_rate=1e-3)
)

utils.train(
    simulation=gravity_multi_pf_sim,
    optimizer=optimizer,
    iterations=50000,
    save_path='./checkpoints/multi_pf',
    gradient_accumulation=4,   # average over 4 batches before updating
    xla_compile=True,
)
```

**Gradient flow:** At each training step, `GradientTape` traces through:
1. `controller_nn(input)` → `(T, Bp, φ_MW)` — NN is differentiable
2. `perform_measurement(controls)` — stochastic (REINFORCE term: `loss * log P(y | x, g)`)
3. `bank.apply_measurement(y, controls)` — differentiable Bayes weight update via `pf.apply_measurement()` and Ścibior trick
4. `holevo_variance(bank)` — differentiable function of bank particle weights
5. Gradients w.r.t. NN weights

No modifications to the training loop are required.

---

## 5. Detailed Design: Multi-PF Bank

### 5.1 Mode Geometry

At a given measurement gain `k_g`, the cosine likelihood `p(y | g) ∝ cos(k_g * g + φ)` has period `Δ = 2π / k_g` in g-space.

The g-prior spans `[g_lo, g_hi]` with width `W = g_hi - g_lo`. The number of complete fringes in the prior is:

```
K = ceil(k_g * W / (2*pi))
```

Fringe k (k = 0, 1, ..., K-1) is the interval:

```
g_k^lo = g_lo + k * 2*pi / k_g
g_k^hi = g_lo + (k+1) * 2*pi / k_g
```

(clamped to `[g_lo, g_hi]` at the boundaries).

**Key properties of this geometry:**
1. The likelihood `p(y=1 | g)` has exactly one peak within each fringe interval (at the fringe center).
2. All fringe intervals have equal width `2π/k_g` (except possibly the boundary fringes which may be narrower due to clamping).
3. The fringe centers are equally spaced by `2π/k_g`.
4. The geometry is fully determined by `k_g` — it is not a free design parameter.

**When k_g changes** (between measurements), the fringe geometry changes:
- Low → high gain: more fringes appear within the prior. New modes must be created.
- High → low gain: fewer fringes. Some modes can be merged (if they fall in the same new fringe).

For the Multi-PF bank, mode creation/destruction is triggered only at the coarse-to-fine transition point (see Section 5.4). During high-gain measurements (which are selected by the RL policy after disambiguation), the mode geometry remains fixed.

### 5.2 Initialization

At initialization (start of each episode), the bank starts with a **single particle filter** covering the full g-prior:

```python
def initialize(self, rangen):
    """
    Start with a single PF covering the full g-prior.
    This is identical to qsensoropt's default initialization.
    """
    self.K_active = 1
    self.mode_weights = tf.Variable([[1.0]])  # shape (bs, 1): all weight in mode 0
    self.active_mask = tf.Variable([[1]])

    # PF[0] covers the full g range
    pf0 = ParticleFilter(
        num_particles=self.N_total,  # all particles in one mode
        phys_model=self.phys_model,  # full g bounds
        ...
    )
    weights0, particles0 = pf0.reset(rangen)
    # particles0 ~ Uniform(g_lo, g_hi), shape (bs, N_total, 1)
    
    self.pf_list = [pf0]
    self.weights_list = [weights0]
    self.particles_list = [particles0]
```

In this state, the bank is exactly a standard qsensoropt `ParticleFilter` with `N_total` particles. The coarse-to-fine structure only activates if/when the RL policy selects a high-gain measurement that causes the posterior to become multimodal.

**Rationale for single-mode start:** At the beginning, the prior is flat over `[g_lo, g_hi]`. The optimal first few measurements (as established by the MFL literature) are low-gain, so the likelihood has at most 1-3 fringes. A single PF with N_total particles handles this perfectly. The multi-mode structure is created lazily, only when needed.

### 5.3 Mode Weight Update

After each measurement `y` with control `x = (T, Bp, φ_MW)`:

**Step 1: Bayes update within each mode**

For each active mode k:
```
# Apply measurement to PF[k]
new_weights_k, _ = pf_k.apply_measurement(
    weights_k, particles_k, state_ensemble_k,
    outcomes=y, controls=x, meas_step=step
)
# new_weights_k has shape (bs, N_k): within-fringe posterior weights
```

**Step 2: Compute per-mode marginal likelihood**

The marginal likelihood of measurement y given mode k is the normalization constant of the Bayes update:

```
p(y | mode k) = Σ_j weights_k[j] * p(y | g_j, x)
             = sum(weights_k * model(y, x, particles_k))
             = sum(pre_update_weights_k * prob_k) / sum(pre_update_weights_k)
```

This is the **total variation** of the unnormalized weight update. In qsensoropt, this is computed during `apply_measurement`:

```python
# Inside apply_measurement for mode k:
prob_k = phys_model.wrapper_model(y_broad, x_broad, particles_k, ...)  # (bs, N_k)
unnorm_weights_k = weights_k * prob_k                                    # (bs, N_k)
Z_k = tf.reduce_sum(unnorm_weights_k, axis=1, keepdims=True)            # (bs, 1)
norm_weights_k = unnorm_weights_k / Z_k                                  # (bs, N_k)
# Z_k is the marginal likelihood: Z_k[b] = p(y | mode k, batch b)
```

**Step 3: Update mode weights**

```python
# new_q_k ∝ q_k * Z_k for each batch element b
for k in range(K_active):
    new_q[:, k] = mode_weights[:, k] * Z_k[:, 0]  # (bs,)

# Normalize
new_q = new_q / tf.reduce_sum(new_q, axis=1, keepdims=True)
mode_weights.assign(new_q)
```

**Step 4: Apply within-mode normalized weights**

Each `PF[k]` now holds `norm_weights_k` — the posterior weights within fringe k, conditioned on mode k being correct. The joint posterior is:

```
p(g | y^{1:t}) = Σ_k q_k * Σ_j weights_k[j] * δ(g - g_{k,j})
```

This is the correct MMAE structure: the marginal posterior is a mixture of fringe-conditional posteriors, weighted by mode probabilities.

**Gradient flow through mode weight update:**

The gradient of `V_H` with respect to NN controls flows through:
1. `V_H` → `mode_weights * within_mode_weights` (differentiable weighted average)
2. `mode_weights` ← `Z_k = Σ_j weights_k[j] * prob_k[j]` (differentiable: `prob_k` is differentiable through `model()`)
3. `within_mode_weights` ← Ścibior trick through `apply_measurement` (differentiable)
4. `prob_k` ← `model(y, x, particles_k)` — gradient flows into `x`, which is the NN output

### 5.4 Adaptive Mode Creation

Mode creation occurs when the RL policy selects a measurement at gain `k_g` such that the single-mode PF's posterior spans multiple fringes of the new likelihood.

**Trigger condition:** After a measurement at gain `k_g_new`, detect multimodality in the posterior:

```python
def _should_split(self, weights, particles, k_g_new):
    """
    Returns True if the posterior under mode k spans more than
    1.5 fringes of the likelihood at gain k_g_new.
    """
    g_mean = compute_mean(weights, particles)  # (bs, 1)
    g_std = tf.sqrt(compute_covariance(weights, particles))  # (bs, 1, 1)
    g_std_scalar = g_std[:, 0, 0]  # (bs,)
    
    delta = 2 * np.pi / k_g_new
    fringes_spanned = (6 * g_std_scalar) / delta  # 6σ width in fringe units
    
    return fringes_spanned > 1.5  # (bs,) bool
```

**Splitting procedure:**

When a single-mode PF needs to be split at gain `k_g_new`:

```python
def _split_mode(self, mode_k, k_g_new):
    """
    Split PF[mode_k] into K_new sub-modes, one per fringe of
    the likelihood at gain k_g_new.
    
    Returns K_new new (weights, particles) pairs.
    """
    weights_k, particles_k = self.weights_list[mode_k], self.particles_list[mode_k]
    
    # Fringe width at new gain
    delta = 2 * np.pi / k_g_new
    g_lo = phys_model.params[0].bounds[0]
    
    # Assign each particle to its fringe
    fringe_idx = tf.floor((particles_k[..., 0] - g_lo) / delta)  # (bs, N_k)
    fringe_idx = tf.cast(fringe_idx, tf.int32)
    
    K_new = int(np.ceil(k_g_new * (g_hi - g_lo) / (2 * np.pi)))
    
    new_modes = []
    new_q = []
    for k in range(K_new):
        mask = (fringe_idx == k)  # (bs, N_k): True where particle in fringe k
        # Per-batch, select particles in fringe k
        fringe_particles = particles_k * tf.cast(mask[..., tf.newaxis], tf.float64)
        fringe_weights = weights_k * tf.cast(mask, tf.float64)
        
        # Weight of mode k = total weight of particles currently in fringe k
        q_new_k = tf.reduce_sum(fringe_weights, axis=1)  # (bs,)
        
        # Normalize within-fringe weights
        safe_sum = tf.maximum(tf.reduce_sum(fringe_weights, axis=1, keepdims=True), 1e-30)
        fringe_weights_norm = fringe_weights / safe_sum  # (bs, N_k)
        
        new_modes.append((fringe_weights_norm, fringe_particles))
        new_q.append(q_new_k)
    
    return new_modes, new_q  # K_new modes with their weights
```

**Mode creation policy:**

- Mode creation is triggered at the **first high-gain measurement step** (when `k_g_new > k_split_threshold`).
- After the split, the K_new new modes each hold `N_k` particles (which may be very few per mode if N_total is not large relative to K_new). Particle reallocation (Section 5.6) immediately follows.
- The original mode `mode_k` is replaced by the K_new new modes in `pf_list`.
- Mode weights of new modes are `q_k_new / Σ_j q_j_new` (normalized particle mass in each fringe).

**Gradients through mode creation:**

Mode creation is a hard thresholding operation (which fringe does each particle fall in?) and is not differentiable in the strict sense. However, gradients flow through:
- The mode weights `q_k_new` (which are sums of particle weights, differentiable through the Ścibior trick)
- The within-mode weight updates after the split

The non-differentiability of the fringe assignment is acceptable because mode creation is a rare event (typically once per episode, at the coarse-to-fine transition), and REINFORCE already handles non-differentiable steps via the log-prob term.

### 5.5 Mode Pruning

Mode k is pruned when its weight falls below a threshold:

```python
PRUNE_THRESHOLD = 1e-6   # mode weight below which mode is deactivated

def prune_modes(self):
    """
    Deactivate modes with q_k < PRUNE_THRESHOLD.
    Redistribute their weight to surviving modes proportionally.
    """
    is_alive = mode_weights > PRUNE_THRESHOLD  # (bs, K_max) bool
    
    # Zero out dead mode weights
    pruned_weights = mode_weights * tf.cast(is_alive, tf.float64)
    
    # Renormalize over surviving modes
    Z = tf.reduce_sum(pruned_weights, axis=1, keepdims=True)
    mode_weights.assign(pruned_weights / tf.maximum(Z, 1e-30))
    active_mask.assign(tf.cast(is_alive, tf.int32))
    
    N_active = tf.reduce_sum(active_mask, axis=1)  # (bs,)
    return N_active
```

**Pruning threshold:** `1e-6` corresponds to a mode that has accumulated 6 decades of likelihood disadvantage relative to the dominant mode. At this point, the mode contains < 1 ppm of the probability mass and contributes negligibly to all observable quantities (mean, variance, Holevo variance). Pruning it does not meaningfully affect estimation quality but does free up particle budget.

**Batch-heterogeneous pruning:** For a batch of `bs` estimation problems, different batch elements may have different numbers of active modes (because the true g differs across the batch, and so different fringes are pruned at different rates). The `active_mask[b, k]` variable tracks this per-batch.

**Minimum mode count:** Even if mode weights drive all modes below threshold simultaneously (which should not happen in a well-functioning system, but can occur with numerical noise), at least 1 mode must remain active. The pruning routine always preserves the argmax mode:

```python
# Never prune the max-weight mode
max_mode = tf.argmax(mode_weights, axis=1)  # (bs,)
active_mask.scatter_nd_update(tf.stack([tf.range(bs), max_mode], axis=1), tf.ones(bs))
```

### 5.6 Particle Reallocation

After pruning and before the next measurement, reallocate the particle budget across active modes proportionally to their weights:

```python
N_TOTAL = 2048    # fixed total particle budget
N_MIN = 32        # minimum particles per active mode

def reallocate_particles(self):
    """
    Redistribute N_TOTAL particles across active modes proportional to mode weights.
    
    For each batch element b and mode k:
        N_k[b] = max(N_MIN, round(q_k[b] * N_TOTAL))
    
    Total may slightly exceed N_TOTAL due to rounding and N_MIN enforcement;
    clip to N_TOTAL by reducing the largest allocation.
    """
    N_k = tf.maximum(
        N_MIN,
        tf.cast(tf.round(mode_weights * N_TOTAL), tf.int32)
    )
    
    # Adjust for budget constraint: reduce largest modes if total > N_TOTAL
    while tf.reduce_sum(N_k) > N_TOTAL:
        largest_mode = tf.argmax(N_k)
        N_k[largest_mode] -= 1
    
    # Resample each mode's particle cloud to its new size
    for k in range(K_active):
        if N_k[k] != current_N_k[k]:
            weights_k, particles_k = self._resample_to_size(k, N_k[k])
            self.weights_list[k] = weights_k
            self.particles_list[k] = particles_k
            self.pf_list[k].num_particles = N_k[k]
    
    self.particles_per_mode = N_k
```

**Resampling to a new size:** When `N_k` increases (a mode gained weight), new particles are drawn from the current within-mode posterior using the soft-resampling procedure. When `N_k` decreases (a mode lost weight), particles are pruned using a systematic resampling that preserves the distribution.

**Practical note:** In TensorFlow graph mode, dynamic resampling to variable sizes requires `tf.TensorArray` or padding to a maximum size. For XLA compilation, padding to `N_max_per_mode = N_TOTAL` (with masking) is recommended.

### 5.7 Holevo Variance Loss

The Holevo variance is computed from the full marginal posterior:

```
μ_H = E[exp(i * k_ref * g)]
    = Σ_k q_k * E_k[exp(i * k_ref * g)]
    = Σ_k q_k * Σ_j w_{k,j} * exp(i * k_ref * g_{k,j})
```

where `k_ref` is a reference gain chosen adaptively.

**Implementation:**

```python
def holevo_variance(bank, k_ref):
    """
    Compute Holevo variance of the marginal posterior.
    
    Parameters
    ----------
    bank : MultiPFBank
    k_ref : float  — reference gain for the circular statistics
    
    Returns
    -------
    V_H : Tensor shape (bs, 1)
    """
    # Complex exponential moments within each mode
    # particles_k shape: (bs, N_k, 1)
    # weights_k shape: (bs, N_k)
    
    mu_k_list = []
    for k in range(bank.K_active):
        particles_k = bank.particles_list[k]   # (bs, N_k, 1)
        weights_k = bank.weights_list[k]       # (bs, N_k)
        q_k = bank.mode_weights[:, k]          # (bs,)
        
        # Within-mode complex moment
        phase_k = k_ref * particles_k[..., 0]  # (bs, N_k)
        exp_k = tf.complex(tf.cos(phase_k), tf.sin(phase_k))  # (bs, N_k)
        mu_k = tf.reduce_sum(
            tf.cast(weights_k, tf.complex128) * exp_k, axis=1
        )  # (bs,)
        
        mu_k_list.append(tf.cast(q_k, tf.complex128) * mu_k)
    
    # Marginal complex moment (mixture of modes)
    mu_H = tf.reduce_sum(tf.stack(mu_k_list, axis=1), axis=1)  # (bs,)
    
    # Holevo variance
    abs_mu = tf.abs(mu_H)  # (bs,)
    V_H = 1.0 / tf.maximum(abs_mu ** 2, 1e-10) - 1.0  # (bs,)
    
    return tf.reshape(V_H, (batch_size, 1))  # (bs, 1)
```

**Adaptive k_ref:**

```python
def adaptive_k_ref(bank, g_lo, g_hi, k_g_current):
    """
    Choose k_ref adaptively based on current posterior width.
    
    k_ref is set so that the posterior 6σ width corresponds to
    approximately one period (2π) at scale k_ref.
    """
    # Compute marginal posterior mean and std
    g_mean, g_var = bank.marginal_mean_and_var()
    g_std = tf.sqrt(g_var)  # (bs,)
    
    # Set k_ref so that 6*sigma = 2*pi / k_ref
    k_ref_adaptive = 2 * np.pi / tf.maximum(6 * g_std, 1e-10)  # (bs,)
    
    # Clamp to physically meaningful range
    k_ref_min = 2 * np.pi / (g_hi - g_lo)   # one period spans full prior
    k_ref_max = k_g_current / 2              # not finer than current measurement
    
    k_ref = tf.clip_by_value(k_ref_adaptive, k_ref_min, k_ref_max)
    return k_ref  # (bs,)
```

**Why Holevo variance and not MSE:**

With a bimodal posterior having equal-weight peaks at `g₀` and `g₀ + π/k_ref`:

- MSE-minimizing estimate: `ĝ = g₀ + π/(2*k_ref)` (between the peaks)  
- MSE at this estimate: `(π/(2*k_ref))²` — appears small if `k_ref` is large
- Holevo variance: `μ_H = 0.5 * e^{i*k_ref*g₀} + 0.5 * e^{i*k_ref*(g₀+π/k_ref)} = 0.5 * e^{i*k_ref*g₀}(1 + e^{iπ}) = 0` → `V_H = ∞`

The Holevo variance is infinite for a symmetric bimodal posterior — it correctly identifies that this posterior carries no usable information about the phase.

**Numerical regularization:** In practice, `V_H = ∞` causes gradient explosion. Cap the loss:

```python
V_H_clipped = tf.minimum(V_H, V_H_MAX)  # V_H_MAX = 100.0
```

This gives the same gradient signal when `V_H` is large (strong signal to disambiguate) without causing NaN gradients.

---

## 6. Code Architecture

### 6.1 File Organization

```
gravimeter/
├── gravimeter_model.py                    # GravityStatelessPhysicalModel (unchanged)
│                                          # Moved from gravimeter_model_imm_wrap.py
│                                          # Remove all IMM-specific code
│
├── gravimeter_multi_pf.py                 # NEW: Multi-PF bank implementation
│   ├── class MultiPFBank                  # Core bank: manages K PF instances
│   ├── class GravityMultiPFSimulation     # StatelessSimulation subclass
│   ├── def build_controller(input_size)   # Factory for MLP controller
│   └── def compute_holevo_variance(bank, k_ref)  # Loss function
│
├── trainer_multi_pf.py                    # NEW: Training script
│   ├── Configuration constants
│   ├── Simulation setup
│   └── Calls utils.train() from qsensoropt
│
├── gravimeter_model_imm_wrap.py           # KEEP for reference, deprecate
│   └── (Do not modify; serves as before/after comparison)
│
└── evaluate_multi_pf.py                   # NEW: Evaluation and benchmarking
    ├── Comparison vs. IMM baseline
    ├── Sensitivity vs. resources curves
    └── Phase disambiguation visualization
```

**Migration plan for gravimeter_model.py:**

Extract `GravityStatelessPhysicalModel` from `gravimeter_model_imm_wrap.py` into a clean `gravimeter_model.py` with no IMM dependencies. This is a pure extraction — no changes to the class.

### 6.2 Class Diagram

```
StatelessPhysicalModel (qsensoropt)
    └── GravityStatelessPhysicalModel       (gravimeter_model.py)
            ↑
            | phys_model (composition)
            |
        MultiPFBank                         (gravimeter_multi_pf.py)
            ├── pf_list: List[ParticleFilter]   ← native qsensoropt PFs
            ├── mode_weights: tf.Variable (bs, K_max)
            ├── active_mask: tf.Variable (bs, K_max)
            ├── particles_list: List[Tensor (bs, N_k, 1)]
            ├── weights_list: List[Tensor (bs, N_k)]
            ├── apply_measurement(y, x, step) → updated bank
            ├── update_mode_weights(Z_k_list) → updated bank
            ├── prune_modes() → N_active
            ├── maybe_create_modes(k_g_new) → updated bank
            ├── reallocate_particles() → updated bank
            ├── marginal_mean_and_var() → (μ, σ²)
            └── holevo_variance(k_ref) → V_H (bs, 1)

Simulation (qsensoropt)
    └── StatelessSimulation (qsensoropt)
            └── GravityMultiPFSimulation    (gravimeter_multi_pf.py)
                    ├── bank: MultiPFBank
                    ├── generate_input(weights, particles, step, resources, rangen)
                    │       → Tensor (bs, input_size)
                    └── loss_function(weights, particles, true_values, resources, step)
                            → Tensor (bs, 1)    [Holevo variance]

ParticleFilter (qsensoropt) ─── [used internally by MultiPFBank] ───────────────────┐
    ├── apply_measurement(weights, particles, state, outcomes, controls, step)       │
    ├── full_resampling(weights, particles, count, rangen)                           │
    ├── compute_mean(weights, particles)                                             │
    └── compute_covariance(weights, particles)                                       │
         [all used unchanged by MultiPFBank to manage each mode's PF]    ───────────┘

controller = tf.keras.Sequential([Dense(128, tanh), Dense(128, tanh),
                                   Dense(64, tanh), Dense(3, tanh)])
```

### 6.3 Key Interfaces — Pseudocode

#### `MultiPFBank.__init__`

```python
def __init__(self, phys_model, N_total, N_min, K_max, prec='float64'):
    """
    Parameters
    ----------
    phys_model : GravityStatelessPhysicalModel
        Physical model (shared). g bounds define the full prior.
    N_total : int
        Total particle budget (e.g., 2048).
    N_min : int
        Minimum particles per active mode (e.g., 32).
    K_max : int
        Maximum number of simultaneous modes (e.g., 64).
    """
    self.phys_model = phys_model
    self.N_total = N_total
    self.N_min = N_min
    self.K_max = K_max
    self.prec = prec
    self.bs = phys_model.batchsize
    
    # g bounds from the physical model's Parameter
    self.g_lo = phys_model.params[0].bounds[0]
    self.g_hi = phys_model.params[0].bounds[1]
    
    # State: initialized in self.reset()
    self.pf_list = []
    self.weights_list = []
    self.particles_list = []
    self.mode_weights = tf.Variable(
        tf.zeros((self.bs, K_max), dtype=prec)
    )
    self.active_mask = tf.Variable(
        tf.zeros((self.bs, K_max), dtype=tf.int32)
    )
```

#### `MultiPFBank.reset`

```python
def reset(self, rangen):
    """
    Initialize bank for a new estimation episode.
    
    Starts with one mode covering the full g prior.
    Equivalent to standard qsensoropt PF initialization.
    """
    # Single PF covering the full g range
    pf0 = ParticleFilter(
        num_particles=self.N_total,
        phys_model=self.phys_model,
        scibior_trick=True,
        trim=True,
        prec=self.prec,
    )
    seed = get_seed(rangen)
    weights0, particles0 = pf0.reset(seed)
    # weights0: (bs, N_total) uniform 1/N_total
    # particles0: (bs, N_total, 1) uniform over [g_lo, g_hi]
    
    self.pf_list = [pf0]
    self.weights_list = [weights0]
    self.particles_list = [particles0]
    
    # Mode 0 has all weight
    new_mw = tf.zeros((self.bs, self.K_max), dtype=self.prec)
    new_mw = tf.tensor_scatter_nd_update(
        new_mw,
        tf.stack([tf.range(self.bs), tf.zeros(self.bs, dtype=tf.int32)], axis=1),
        tf.ones(self.bs, dtype=self.prec)
    )
    self.mode_weights.assign(new_mw)
    
    new_mask = tf.zeros((self.bs, self.K_max), dtype=tf.int32)
    new_mask = tf.tensor_scatter_nd_update(
        new_mask,
        tf.stack([tf.range(self.bs), tf.zeros(self.bs, dtype=tf.int32)], axis=1),
        tf.ones(self.bs, dtype=tf.int32)
    )
    self.active_mask.assign(new_mask)
    self.K_active = 1
```

#### `MultiPFBank.apply_measurement`

```python
def apply_measurement(self, outcomes, controls, meas_step, rangen):
    """
    Apply one measurement to all active modes.
    
    Steps:
    1. For each mode k: call pf_k.apply_measurement() → new weights, marginal likelihood Z_k
    2. Update mode weights: q_k ← q_k * Z_k, renormalize
    3. Optionally trigger mode creation if k_g is high and posterior is multimodal
    4. Optionally prune dead modes
    5. Reallocate particles
    
    Parameters
    ----------
    outcomes : Tensor (bs, 1)    — binary measurement result
    controls : Tensor (bs, 3)    — [T_s, Bp_kTm, mw_phase_rad]
    meas_step : Tensor (bs, 1)   — current step index
    rangen : tf.random.Generator
    """
    Z_k_list = []
    new_weights_list = []
    
    for k in range(self.K_active):
        # Get pre-update weights and particles
        w_k = self.weights_list[k]   # (bs, N_k)
        p_k = self.particles_list[k] # (bs, N_k, 1)
        
        # Compute likelihood under each particle in mode k
        # (equivalent to apply_measurement but extracting Z_k)
        prob_k = self.phys_model.wrapper_model(
            outcomes_broad,  # broadcast outcomes to (bs, N_k, 1)
            controls_broad,  # broadcast controls to (bs, N_k, 3)
            p_k,
            state_ensemble=tf.zeros((self.bs, self.pf_list[k].np, 0), dtype=self.prec),
            meas_step=meas_step_broad,
            num_systems=self.pf_list[k].np,
        )[0]  # prob_k: (bs, N_k)
        
        unnorm_w_k = w_k * prob_k  # (bs, N_k)
        Z_k = tf.reduce_sum(unnorm_w_k, axis=1)  # (bs,): marginal likelihood
        
        # Outlier guard: if Z_k ≈ 0, don't update
        safe_Z_k = tf.maximum(Z_k, 1e-300)
        norm_w_k = unnorm_w_k / safe_Z_k[:, tf.newaxis]  # (bs, N_k)
        
        Z_k_list.append(Z_k)
        new_weights_list.append(norm_w_k)
    
    # Update mode weights
    Z_stack = tf.stack(Z_k_list, axis=1)  # (bs, K_active)
    q_active = self.mode_weights[:, :self.K_active]  # (bs, K_active)
    new_q_unnorm = q_active * Z_stack
    Z_total = tf.reduce_sum(new_q_unnorm, axis=1, keepdims=True)
    new_q = new_q_unnorm / tf.maximum(Z_total, 1e-300)
    
    # Write back normalized mode weights for active modes
    # (inactive modes remain at 0)
    self.mode_weights[:, :self.K_active].assign(new_q)
    
    # Update within-mode weights
    self.weights_list = new_weights_list
    
    # ESS-based resampling for each mode
    for k in range(self.K_active):
        new_w_k, new_p_k, _ = self.pf_list[k].full_resampling(
            new_weights_list[k],
            self.particles_list[k],
            count_for_resampling=tf.ones(self.bs, dtype=tf.int32),
            rangen=rangen,
        )
        self.weights_list[k] = new_w_k
        self.particles_list[k] = new_p_k
    
    # Check for mode creation, pruning, reallocation
    k_g = self._compute_k_g(controls)  # scalar or (bs,)
    if self.K_active == 1 and tf.reduce_any(self._should_split(0, k_g)):
        self._split_all_modes(k_g)
    
    self.prune_modes()
    self.reallocate_particles(rangen)
```

#### `GravityMultiPFSimulation.generate_input`

```python
def generate_input(self, weights, particles, meas_step, used_resources, rangen):
    """
    Build NN input from bank state.
    
    Input vector layout (total size = 4 * TOP_K_MODES + 5):
    
    For each of top-K active modes (padded with zeros if fewer than TOP_K):
        [0] μ_k normalized: 2*(μ_k - g_lo)/(g_hi - g_lo) - 1   ∈ [-1, 1]
        [1] log_std_k: -2/10 * log10(σ_k) - 1                   ∈ [-1, 1]
        [2] q_k                                                   ∈ [0, 1]
        [3] active_k                                              ∈ {0, 1}
    
    Global features:
        [4*TOP_K+0] tanh(log(V_H + 1)) / tanh(log(101))         ∈ [-1, 1]
        [4*TOP_K+1] H_q / log(K_max)                             ∈ [0, 1]
        [4*TOP_K+2] N_active / K_max                             ∈ [0, 1]
        [4*TOP_K+3] meas_step / num_steps * 2 - 1               ∈ [-1, 1]
        [4*TOP_K+4] used_resources / max_resources * 2 - 1      ∈ [-1, 1]
    
    Returns
    -------
    Tensor shape (bs, 4*TOP_K + 5)
    """
    TOP_K = 4
    bank = self.bank
    
    # Get top-K modes by weight
    q = bank.mode_weights  # (bs, K_max)
    top_k_idx = tf.argsort(q, direction='DESCENDING', axis=1)[:, :TOP_K]
    
    mode_features = []
    for i in range(TOP_K):
        mode_k = top_k_idx[:, i]  # (bs,)
        
        # Gather mean and std for this mode
        mu_k = bank.marginal_mode_mean(mode_k)  # (bs,)
        sigma_k = bank.marginal_mode_std(mode_k)  # (bs,)
        q_k = tf.gather(q, mode_k, batch_dims=1)  # (bs,)
        active_k = tf.gather(
            tf.cast(bank.active_mask, self.prec), mode_k, batch_dims=1
        )  # (bs,)
        
        mu_k_norm = 2 * (mu_k - self.g_lo) / (self.g_hi - self.g_lo) - 1
        log_std_k = -2.0 / 10.0 * tf.math.log(
            tf.maximum(sigma_k, 1e-10)
        ) / tf.math.log(10.0) - 1.0
        
        mode_features.extend([mu_k_norm, log_std_k, q_k, active_k])
    
    # Holevo variance
    k_ref = adaptive_k_ref(bank, self.g_lo, self.g_hi)
    V_H = bank.holevo_variance(k_ref)[:, 0]  # (bs,)
    V_H_norm = tf.tanh(tf.math.log(V_H + 1.0)) / tf.tanh(tf.math.log(101.0))
    
    # Mode weight entropy
    q_safe = tf.maximum(q * tf.cast(bank.active_mask, self.prec), 1e-30)
    H_q = -tf.reduce_sum(q_safe * tf.math.log(q_safe), axis=1)  # (bs,)
    H_q_norm = H_q / tf.math.log(tf.cast(self.K_max, self.prec))
    
    # N_active
    N_active = tf.cast(
        tf.reduce_sum(bank.active_mask, axis=1), self.prec
    ) / self.K_max  # (bs,)
    
    # Step and resource (same as StatelessMetrology)
    step_norm = 2.0 * tf.cast(meas_step[:, 0], self.prec) / self.num_steps - 1.0
    res_norm = 2.0 * used_resources[:, 0] / self.max_resources - 1.0
    
    global_features = tf.stack([V_H_norm, H_q_norm, N_active, step_norm, res_norm], axis=1)
    
    all_features = tf.stack(mode_features, axis=1)  # (bs, 4*TOP_K)
    input_vec = tf.concat([all_features, global_features], axis=1)  # (bs, 4*TOP_K+5)
    
    return input_vec
```

#### `GravityMultiPFSimulation.loss_function`

```python
def loss_function(self, weights, particles, true_values, used_resources, meas_step):
    """
    Holevo variance of the marginal posterior.
    
    Parameters
    ----------
    weights : Tensor (bs, N)       — (unused: bank maintains its own state)
    particles : Tensor (bs, N, 1)  — (unused)
    true_values : Tensor (bs, 1, 1) — true g values (for debugging only)
    used_resources : Tensor (bs, 1)
    meas_step : Tensor (bs, 1)
    
    Returns
    -------
    Tensor shape (bs, 1)
    """
    k_ref = adaptive_k_ref(self.bank, self.g_lo, self.g_hi)
    V_H = self.bank.holevo_variance(k_ref)  # (bs, 1)
    
    # Cap to prevent gradient explosion
    V_H_clipped = tf.minimum(V_H, V_H_MAX)  # V_H_MAX = 100.0
    
    return V_H_clipped
```

---

## 7. Comparison: IMM vs. Multi-PF Bank

| Dimension | Current IMM Approach | Multi-PF Bank |
|---|---|---|
| **Mode semantics** | Behavioral regimes (inappropriate for static g) | Fringe hypotheses: epistemic uncertainty about which period contains g |
| **Mode interaction** | Mixing step blends particles across modes before each update | No mixing: modes are independent parallel hypotheses |
| **Transition matrix** | K×K Markov transition matrix (physically meaningless for static g) | None: g does not transition between fringes |
| **Mode geometry** | Static tiling with fixed-width intervals; split heuristically | Fringe-aligned: intervals of width 2π/k_g, determined by measurement gain |
| **Mode count** | Fixed at up to 128 | Dynamic: 1 initially, grows at coarse-to-fine transition, shrinks via pruning |
| **Particle budget** | Fixed 16 particles/mode × 128 modes = 2048 total | Adaptive: proportional to mode weight, with N_min=32 per active mode |
| **Effective particle count per fringe** | 16 (fixed, poor approximation) | Up to N_total (all particles in 1 fringe after disambiguation) |
| **Gradient flow** | Unclear; bypasses Ścibior trick for many operations | Full gradient through Ścibior-Wood differentiable resampling in each PF |
| **Controller complexity** | Set Transformer (O(K²) attention): high parameter count, slow | Standard MLP (O(d_model)): fast, easy to train |
| **Controller input** | 128-token attention over all mode states | Fixed-size vector: top-K mode summaries + global statistics |
| **Loss function** | MSE (wrong for bimodal posteriors) | Holevo variance (correct circular loss, penalizes unresolved ambiguity) |
| **qsensoropt reuse** | Low: custom PF management outside qsensoropt infrastructure | High: each mode IS a native qsensoropt ParticleFilter; training loop unchanged |
| **TF graph-mode compat.** | Problematic: dynamic lists of sub-filters not XLA-compatible | Achievable with flattened particle array and tf.TensorArray |
| **Training difficulty** | High: IMM + Set Transformer + custom code = many failure modes | Low: each component has independent prior art and is well-tested |
| **Physical correctness** | No: IMM assumes regime switching that does not occur | Yes: MMAE with Holevo loss matches the physics |
| **Phase wrapping handling** | Fragile: mode geometry doesn't adapt to measurement gain | Principled: modes created at the fringe periodicity of each measurement |
| **Coarse-to-fine** | Absent: protocol library selects fixed-gain settings | Implicit: RL policy learns to start low-gain (single PF) and ramp gain |
| **Disambiguation mechanism** | Mode weight competition (correct, but corrupted by mixing) | Mode weight competition (correct, no mixing corruption) |
| **Literature backing** | IMM literature (Blom 1988): designed for regime switching, not applicable | MMAE literature, MFL (Joas 2021), Holevo phase estimation (Berry 2009) |

**Summary:** The IMM approach applies approximately 3 layers of physically incorrect structure (mixing step, transition matrix, static tiling) on top of a correct core idea (mode weight competition). The Multi-PF bank removes the incorrect layers and implements only the correct core, using battle-tested qsensoropt infrastructure throughout.

---

## 8. Implementation Roadmap

### Phase 1: Physical Model Extraction (1-2 days)

**Goal:** Create `gravimeter_model.py` with a clean `GravityStatelessPhysicalModel` class.

1. Copy `GravityStatelessPhysicalModel` from `gravimeter_model_imm_wrap.py` to `gravimeter_model.py`.
2. Remove all IMM-specific imports, constants (`MAX_MODES`, `particles_per_mode`, protocol library).
3. Verify `model()`, `perform_measurement()`, `count_resources()` pass existing unit tests.
4. Add a minimal unit test: single PF with 1024 particles on a low-gain measurement sequence → verify posterior converges to the correct g.

**Deliverable:** `gravimeter_model.py` with clean `GravityStatelessPhysicalModel`.

### Phase 2: Multi-PF Bank Core (3-5 days)

**Goal:** Implement `MultiPFBank` with correct MMAE semantics.

1. Implement `MultiPFBank.__init__`, `reset`, `apply_measurement` (Section 6.3).
2. Implement mode weight update (Section 5.3) with correct Bayes normalization.
3. Implement `prune_modes` (Section 5.5).
4. Implement `holevo_variance` (Section 5.7) with adaptive `k_ref`.
5. **Test 1:** Single mode (K=1), low-gain measurement sequence → bank behaves identically to single qsensoropt PF. Verify `holevo_variance` converges as posterior narrows.
6. **Test 2:** Two modes (K=2), one correct fringe, one wrong fringe. Verify wrong-fringe mode weight decays exponentially with each measurement, correct-fringe mode weight → 1.0.
7. **Test 3:** Mode pruning → K=1 after disambiguation. Verify particle budget concentrates.

**Deliverable:** `MultiPFBank` class passing all unit tests. No RL training yet.

### Phase 3: Mode Creation (2-3 days)

**Goal:** Implement adaptive mode creation at the coarse-to-fine transition.

1. Implement `_should_split` (posterior width in fringe units > 1.5).
2. Implement `_split_mode` (assign particles to fringes, create new PF list).
3. Implement `reallocate_particles` (Section 5.6).
4. **Test 4:** Single-mode bank, high-gain measurement → automatic split into K new modes. Verify each new mode receives particles from the correct fringe. Verify mode weights after split sum to 1.0.
5. **Test 5:** Full disambiguation sequence: start low-gain → 1 mode → increase gain → K modes → decrease mode count via pruning → 1 mode. Verify Holevo variance decreases monotonically (on average).

**Deliverable:** Full `MultiPFBank` with creation/pruning/reallocation.

### Phase 4: Simulation Integration (2-3 days)

**Goal:** Wire `GravityMultiPFSimulation` into qsensoropt's training infrastructure.

1. Implement `GravityMultiPFSimulation.generate_input()` (Section 6.3).
2. Implement `GravityMultiPFSimulation.loss_function()` (Section 6.3).
3. Verify `input_size` computation is correct (4 * TOP_K + 5).
4. Implement `build_controller()` factory (Section 4.5).
5. **Test 6:** Run `simulation.execute()` in training mode on a 10-step episode. Verify loss is finite and gradients are non-NaN. Check gradient w.r.t. NN weights is non-zero.
6. **Test 7:** Run 100 training iterations with `utils.train()`. Verify loss decreases.

**Note on `while_loop` compatibility:** The `GravityMultiPFSimulation` must be compatible with `tf.function` (graph mode). The `MultiPFBank` should use `tf.Variable` for all mutable state, and all Python-level loops over modes should be either unrolled (for fixed `K_max`) or replaced with `tf.TensorArray` / vectorized operations.

**Deliverable:** `GravityMultiPFSimulation` running inside qsensoropt's training loop.

### Phase 5: Training and Hyperparameter Tuning (5-10 days)

**Goal:** Train the controller to learn the coarse-to-fine gain schedule.

1. Start with `N_total=512`, `K_max=8`, `num_steps=32`, `batch_size=64`. Train for 10,000 iterations. Verify the policy begins to learn a low→high gain schedule.
2. Scale up: `N_total=2048`, `K_max=32`, `batch_size=128`. Train for 50,000 iterations.
3. Monitor: plot Holevo variance vs. resources curves. Compare to:
   - Flat-gain baseline (fixed T, fixed Bp)
   - MFL-style schedule (analytically designed coarse-to-fine τ sequence)
   - IMM baseline (the current implementation)
4. Tune `PRUNE_THRESHOLD`, `N_MIN`, `V_H_MAX`, `TOP_K` as needed.
5. Analyze mode dynamics: plot number of active modes vs. step, mode weight entropy vs. step. Verify the policy discovers the coarse-to-fine structure.

**Key metric for success:** Holevo variance at final step should scale as `1 / N_total^β` with `β > 0.5` (beating standard quantum limit scaling), ideally approaching `β ≈ 1` (Heisenberg scaling) for the noiseless case.

**Deliverable:** Trained checkpoint and performance evaluation plots.

### Phase 6: TF Graph Compatibility and XLA Optimization (2-3 days)

**Goal:** Make the implementation XLA-compilable for maximal training speed.

1. Replace Python-level mode list with flattened particle array of shape `(bs, K_max * N_per_mode, 1)`. Mode k's particles are at indices `[k*N_per_mode : (k+1)*N_per_mode]`.
2. Replace `tf.Variable` assignments with loop-variable updates inside `tf.while_loop`.
3. Add `@tf.function(jit_compile=True)` to `single_iteration` in the training script.
4. Profile and verify 5-10× speedup from XLA vs. eager mode.

**Deliverable:** XLA-compilable training at near-optimal speed.

### Phase 7: Benchmarking and Documentation (2-3 days)

**Goal:** Validate the redesign and document its performance.

1. Run `performance_evaluation()` from qsensoropt on the trained model.
2. Compare to: (a) single-PF with low-gain measurements only; (b) Cramér-Rao bound; (c) Heisenberg limit; (d) previous IMM approach.
3. Run ablation studies: Holevo vs. MSE loss; adaptive vs. fixed k_ref; mode creation on/off.
4. Document the final architecture in `trainer_multi_pf.py` with inline comments.

---

## 9. References

1. **Blom, H. A. P., Bar-Shalom, Y.** (1988). "The interacting multiple model algorithm for systems with Markovian switching coefficients." *IEEE Transactions on Automatic Control*, 33(8), 780-783. https://doi.org/10.1109/9.1299  
   *(Original IMM paper — defines the algorithm we are replacing)*

2. **Joas, T., Waeber, A. M., Braunbeck, G., Reinhard, F.** (2021). "Online adaptive quantum characterization of a nuclear spin." *npj Quantum Information*, 7(1), 56. https://www.nature.com/articles/s41534-021-00389-z  
   *(MFL: coarse-to-fine adaptive τ scheduling; 110× dynamic range; the direct experimental inspiration for our gain scheduling)*

3. **Yankelev, D., Avinadav, C., Davidson, N., Firstenberg, O.** (2020). "Atom interferometry with thousand-fold increase in dynamic range." *Science Advances*, 6(45), eabd0650. https://www.science.org/doi/10.1126/sciadv.abd0650  
   *(Dual-T moiré disambiguation; particle filter on sequential shots; 1000× dynamic range; multi-scale phase unwrapping)*

4. **Berry, D. W., Sanders, B. L.** (2009). "How to perform the most accurate possible phase measurements." *Physical Review A*, 80(5), 052114. https://arxiv.org/abs/0907.0014  
   *(Holevo variance as the correct loss for circular/wrapped phase distributions; V_H = μ⁻² - 1 where μ = |E[e^{iφ}]|)*

5. **Higgins, B. L., Berry, D. W., Bartlett, S. D., Wiseman, H. M., Pryde, G. J.** (2007). "Entanglement-free Heisenberg-limited phase estimation." *Nature*, 450(7168), 393-396. https://doi.org/10.1038/nature06257 / https://arxiv.org/abs/0709.2996  
   *(Binary coarse-to-fine gain schedule; Kitaev's phase estimation algorithm with adaptive Bayesian feedback; Heisenberg scaling without entanglement)*

6. **Belliardo, F., Zoratti, F., Marquardt, F., Giovannetti, V.** (2024). "Model-aware reinforcement learning for high-performance Bayesian experimental design in quantum metrology." *Physical Review A*, 109, 062609. https://link.aps.org/doi/10.1103/PhysRevA.109.062609 / arXiv preprint: https://arxiv.org/abs/2312.16985  
   *(qsensoropt framework: model-aware RL with particle filter, Ścibior trick, NV center applications — the foundation we build on)*

7. **Belliardo, F., Zoratti, F., Giovannetti, V.** (2024). "Application of Machine Learning to Experimental Design in Quantum Mechanics." *International Journal of Quantum Information*, 22, 2450002. https://doi.org/10.1142/s0219749924500023  
   *(Second qsensoropt paper with additional applications; qsensoropt library on PyPI)*

8. **van den Berg, E.** (2021). "Efficient Bayesian phase estimation using mixed priors." *Quantum*, 5, 469. https://quantum-journal.org/papers/q-2021-06-07-469/  
   *(Mixed-prior Bayesian QPE: K-component mixture of Gaussians for wrapped posteriors; Fourier ↔ Gaussian representation switching; MMAE weight update for discrete phase hypotheses)*

9. **Brady, J.-J., Wang, W., Maskell, S.** (2024). "Differentiable Interacting Multiple Model Particle Filtering." *Signal Processing* (preprint). https://arxiv.org/abs/2410.00620  
   *(DIMMPF: differentiable IMM-PF for regime-switching models — explicitly for dynamical regime switching, not for static-parameter phase disambiguation; cited here to contrast with our MMAE approach)*

10. **Granade, C., Wiebe, N., Ferrie, C., Cory, D.** (2012). "Robust online Hamiltonian learning." *New Journal of Physics*, 14, 103013. https://doi.org/10.1088/1367-2630/14/10/103013  
    *(SMC with Liu-West resampling for quantum Hamiltonian learning; the particle filter algorithm underlying qsensoropt's resampling)*

11. **Wiebe, N., Granade, C., Ferrie, C., Cory, D.** (2016). "Hamiltonian learning and certification using quantum resources." *Physical Review Letters*, 112, 190501. https://arxiv.org/abs/1309.0876  
    *(Quantum bootstrap filter: quantum resources for particle-filter-based Hamiltonian learning)*

12. **Ścibior, A., Wood, F.** (2021). "Differentiable particle filtering without modifying the forward pass." arXiv:2106.10314. https://arxiv.org/abs/2106.10314  
    *(The Ścibior-Wood trick for differentiable resampling: makes the particle filter training-compatible with gradient-based RL)*

13. **Granade, C. E., et al.** (2021). "Efficient qubit phase estimation using adaptive measurements." *Quantum*, 5, 467. https://quantum-journal.org/papers/q-2021-06-04-467/  
    *(Covariant estimators for phase estimation; non-identifiable likelihood functions in phase wrapping; adaptive schemes using confidence intervals)*

14. **Panda, C. D., Tao, M., Ceja, M., Reynoso, A., Müller, H.** (2023). "Atomic gravimeter robust to environmental effects." arXiv:2305.05555. https://arxiv.org/pdf/2305.05555  
    *(Gravity measurement using atom interferometry at multiple hold times for phase disambiguation; relevance to multi-scale gain scheduling in gravimetry)*

15. **Magill, D. T.** (1965). "Optimal adaptive estimation of sampled stochastic processes." *IEEE Transactions on Automatic Control*, 10(4), 434-439.  
    *(Original MMAE paper — the correct framework for our multi-PF bank, as opposed to IMM)*

---

## Appendix A: Notation Summary

| Symbol | Definition |
|---|---|
| g | Gravitational acceleration (unknown parameter) |
| g_lo, g_hi | Prior bounds on g |
| T | Interrogation time (control variable) |
| Bp | Magnetic field gradient (control variable, units kT/m) |
| φ_MW | Microwave phase (control variable, radians) |
| k_g(T, Bp) | Measurement gain = effective phase accumulated per unit g |
| vis | Fringe visibility (fixed physical parameter) |
| K | Number of fringes in prior at gain k_g |
| Δ = 2π/k_g | Fringe width (period of likelihood in g-space) |
| q_k | Mode weight: P(true g in fringe k \| data) |
| w_{k,j} | Within-mode particle weight: P(g = g_{k,j} \| true g in fringe k, data) |
| Z_k | Marginal likelihood of measurement y under mode k |
| μ_H | Complex circular moment: E[exp(i k_ref g)] |
| V_H | Holevo variance: \|μ_H\|⁻² - 1 |
| k_ref | Reference gain for Holevo variance computation |
| N_total | Total particle budget across all modes |
| N_k | Particle count for mode k |
| N_min | Minimum particles per active mode |
| K_max | Maximum number of simultaneous modes |
| ESS | Effective sample size: 1/Σ_j w_j² |

## Appendix B: Physical Intuition for Mode Weights

The mode weight `q_k` has a precise Bayesian interpretation:

```
q_k(t) = P(true g ∈ [g_lo + k*Δ, g_lo + (k+1)*Δ] | y^{1:t}, x^{1:t})
```

After t measurements with controls `x^{1:t}` and outcomes `y^{1:t}`, the posterior probability that g lives in fringe k is `q_k(t)`.

The MMAE update rule:

```
q_k(t) ∝ q_k(t-1) * p(y_t | mode k, x_t)
```

is the correct Bayes rule for this mixture model. The marginal likelihood `p(y_t | mode k, x_t)` is computed as the expectation of `p(y_t | g, x_t)` under the within-fringe posterior:

```
p(y_t | mode k, x_t) = ∫ p(y_t | g, x_t) p(g | mode k, y^{1:t-1}, x^{1:t-1}) dg
                      ≈ Σ_j w_{k,j}(t-1) * p(y_t | g_{k,j}, x_t)
```

The IMM weight update — which adds a mixing term — is NOT a valid Bayes rule for this problem. The IMM weight update is the Bayes rule for a _different_ problem: one where the mode index itself follows a Markov chain. Since the mode index for a fixed-g problem does not follow a Markov chain (there is no mode-switching dynamics), the IMM update introduces systematic bias.

## Appendix C: Gradient Flow Diagram

```
Loss: V_H = 1/|μ_H|² - 1

μ_H = Σ_k q_k * Σ_j w_{k,j} * e^{i*k_ref*g_{k,j}}
        ↑                 ↑
    mode weights    within-mode weights
        ↑                 ↑
    Z_k update      Bayes update (Ścibior trick)
        ↑                 ↑
    Σ_j w_{k,j} * prob_k[j]
                          ↑
               prob_k[j] = model(y, x, g_{k,j})
                                       ↑
                                 control x = NN_output(input)
                                                    ↑
                                             NN weights θ
                                                    ↑
                                             ∂V_H/∂θ ← gradient
```

The gradient signal flows as follows:
- `∂V_H / ∂μ_H`: always nonzero when V_H > 0 (i.e., when the posterior is not perfectly localized)
- `∂μ_H / ∂q_k`: proportional to the per-mode complex moment `Σ_j w_{k,j} e^{i k_ref g_{k,j}}`
- `∂q_k / ∂Z_k ∝ q_k / Σ_j q_j Z_j`: the mode weight gradient w.r.t. its likelihood
- `∂Z_k / ∂x = ∂/∂x Σ_j w_{k,j} * model(y, x, g_{k,j})`: gradient through the likelihood w.r.t. controls
- `∂x / ∂θ`: gradient through the NN
- Ścibior trick: `∂(w_{k,j} post) / ∂(w_{k,j} pre) = prob_{k,j} / stop_gradient(prob_{k,j})`

**REINFORCE augmentation:** The stochasticity of the binary outcome `y ~ Bernoulli(p)` is handled by the REINFORCE term `V_H * log p(y | x, g_true)`, which is added to the loss before taking gradients (as implemented in qsensoropt's `loss_logl_outcomes=True` setting).
```
