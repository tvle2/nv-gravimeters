# Hybrid Hierarchical + Holevo Training Report

## What your artifacts prove about the run

Your run is **numerically stable** in the sense that the loop executes, logs are produced, and evaluation completes; but it is **not behaving correctly as a trainable model-aware controller** because (i) the episode always becomes ill-defined the moment it enters refinement (controls become NaN), and (ii) the coarse (level‑0) posterior does not update at all, so the loss landscape is effectively flat where you most need signal.

These are not “hyperparameter” symptoms; they follow directly from verifiable patterns in your debug JSONL and from specific code paths in your bank + simulation.

### The refinement failure is deterministic and always occurs at the transition

From your debug rollout JSONL, every evaluation episode shows the following pattern:

- Measurement steps progress normally through the 4 disambiguation levels.
- At the first refinement attempt (right after completing 4×7 = 28 disambiguation shots), the controller outputs become **NaN** (`T_s`, `Bp_kTm`, `mw_phase_rad`).
- The resource counter remains frozen and the step counter stops advancing, producing **duplicated `global_step=28` records**: one with finite controls (the last disambiguation shot that triggered the transition) and one with NaNs (the first refinement attempt).

This matches exactly the simulation logic: `continue_flag = (new_used_resources <= max_resources)` will become `False` whenever `new_used_resources` is NaN because comparisons with NaN return False; then the step counter stops increasing and the bank update is masked out, so loss snapshots go to zero on those rows. That behavior is implemented in the execute loop in `gravimeter_hierarchical_local_holevo.py`. fileciteturn0file0

This is why your log reports “control params are NaN when refining is true”: **the transition into refinement produces an invalid controller input → NaN policy output → NaN resource update → masking of learning signal**.

### Level‑0 posterior does not move (so disambiguation doesn’t start)

Across the first 6 shots at level 0 in your eval traces, the posterior statistics are essentially those of the original uniform prior:

- `g_std` stays at the uniform‑prior value (≈ prior_width / √12).
- `q0` remains near 0.5 with tiny variance.
- `local_holevo_current` is **exactly at the clip ceiling** (100) on every logged level‑0 step.

That is exactly what your loss and feature construction would produce if the posterior is not changing: your local Holevo term is clipped and then logged, so it becomes a constant as long as the circular moment is sufficiently small; and your branch BCE term cannot improve if `q0` stays ≈ 0.5.

This is not speculation—the behavior is the *direct numerical consequence* of your loss definition and your nuisance phase structure (explained next). The local Holevo clip and log are in `gravimeter_hierarchical_local_holevo.py`. fileciteturn0file0

## Why level‑0 disambiguation is uninformative in your physics model

Your physical model is (correctly) a cosine-likelihood interferometer:

\[
p(y=1\mid g, T, B', \phi) = \tfrac12\left(1+\mathrm{vis}\cos(k_g(T,B')\,g+\phi+\phi_{\text{off}})\right),
\]

implemented in `gravimeter_model_complete.py`, with a *latent* additive phase offset \(\phi_{\text{off}}\) that is inferred (`infer_phi_off=True` in your run config). fileciteturn0file2 fileciteturn0file4

This latent offset induces a **gauge symmetry**: if \(\phi_{\text{off}}\) is initially broad (nearly uniform on \([-\pi,\pi]\)), then **a single measurement cannot provide any marginal evidence about \(g\)** because integrating over \(\phi_{\text{off}}\) removes the cosine:

\[
\int_{-\pi}^{\pi}\cos(k g + \phi + \phi_{\text{off}})\,\frac{d\phi_{\text{off}}}{2\pi}=0.
\]

Therefore for any fixed \(g\),

\[
\int p(y\mid g,\phi_{\text{off}})\,p(\phi_{\text{off}})\,d\phi_{\text{off}}
= \tfrac12,
\]

so the per-branch evidences \(Z_0,Z_1\) computed by your bank are initially equal and \(q_0\) cannot move (this is exactly what you see at level 0).

Crucially, even **multiple** measurements do not break this symmetry if the policy behaves like “same effective \(k_g\) with varying microwave phase” because the common \(\phi_{\text{off}}\) shift can be absorbed by a variable substitution in the marginal likelihood (translation invariance). The only reliable way to make \(g\) identifiable under an unknown additive phase is to ensure that the sequence includes **controls that change \(k_g\)** (or otherwise couple \(g\) differently across steps), so that \(\phi_{\text{off}}\) can no longer absorb a single global shift.

This is fully consistent with the gravimeter protocol literature: the measured signal is fundamentally an accumulated phase and any unknown phase offset must be calibrated/learned jointly. The gravimeter paper explicitly frames the sensing signal as phase accumulation and analyzes robustness to control-induced phase errors. fileciteturn0file9 fileciteturn0file10

## Two root causes in your current approach that prevent learning

The run failures reduce to two “hard” causes (both are correctness issues, not tuning).

### The Holevo loss is mathematically correct but implemented in a gradient-killing form

Your code defines Holevo variance (locally) as

\[
V_H = | \mu |^{-2} - 1,\quad \mu = \mathbb{E}[\exp(i k_{\text{loc}}(g-m))].
\]

Then you **clip** \(V_H\) to 100 and feed \(\log(1+V_H)\) into the loss. fileciteturn0file0

But note the identity:

\[
\log(1+V_H)=\log(|\mu|^{-2})=-\log(|\mu|^2).
\]

So the correct smooth objective is \(-\log(|\mu|^2)\), which is well-behaved when \(|\mu|\) is small if you use a small \(\varepsilon\) floor.

By clipping \(V_H\) before taking the log, you create a **large flat region** in the loss where the derivative is exactly zero. That is exactly what your logs show at level 0: `local_vh=100` always, so the Holevo component becomes a constant \(\log(101)\) and provides no gradient direction.

This contradicts the assumptions under which the model-aware PF gradient is useful: Belliardo et al. emphasize differentiating through the Bayes update and (approximate) differentiable resampling to optimize controls via a smooth loss functional. If the loss is clipped into a constant, the model-aware part has nothing to optimize. fileciteturn0file11

### Refinement-mode NaNs are caused by a state-definition inconsistency at the transition

The controller input in your simulation includes a “local Holevo” feature computed from the bank state (`bank.current_local_holevo()`), and your bank’s local Holevo computation **always evaluates both branches’ circular moments** and then mixes them. fileciteturn0file0

But at refinement time, branch 1 is no longer semantically meaningful because the bank has committed \(q=[1,0]\). In floating point arithmetic, the expression \(q_1 \cdot \text{moment}_1\) is not guaranteed to be safe if the moment is non-finite, because **\(0\times \mathrm{NaN}=\mathrm{NaN}\)**. So if the inactive branch ever contains a NaN (from any prior numerical artifact), the local Holevo feature becomes NaN, contaminates the controller input, and then the controller outputs become NaN.

Even if you do not *expect* NaNs in branch 1, your transition code explicitly keeps a placeholder branch around (it does not eliminate it), and refinement masks are applied after resource counting—so a single NaN at the first refinement attempt collapses `continue_flag` and stops learning signal exactly as your run shows.

This state inconsistency is a correctness issue: **in refinement mode, your “two-branch ambiguity feature” must be defined in a way that cannot depend on the inactive branch**.

## A technically sound fix that stays within your approach

You asked specifically to fix the current approach, not replace it. The following changes do exactly that:

- Same hierarchical PF bank.
- Same model-aware RL structure.
- Same use of local Holevo information.
- Same residual controller idea.

But they correct the two root causes so the method can actually train.

### Fix the Holevo loss without clipping by using the equivalent smooth form

Replace “clip \(V_H\), then log1p” with the mathematically equivalent:

\[
L_H = -\log(|\mu|^2 + \varepsilon).
\]

This preserves correct ordering, avoids infinities, and **never becomes flat by clipping**. The only “bound” is \(\varepsilon\), which is a numerical floor (not a heuristic tuning knob).

### Make refinement-mode features independent of the inactive branch

When \(q_1=0\), define the branch‑1 circular moment contribution to be exactly zero **before** mixing:

\[
\mu = q_0 \mu_0 + q_1 \mu_1,\quad
\text{but compute }\mu_1 := 0 \text{ wherever } q_1 = 0.
\]

This is mathematically exact (because a zero-weight component contributes nothing), and it eliminates the only robust mechanism by which refinement can create NaN controls.

### Make the disambiguation phase anchor offset-aware

Your phase anchoring logic already uses the correct geometric “left-center” construction for classifying left vs right (good). fileciteturn0file0

But under an inferred \(\phi_{\text{off}}\), the correct anchor must cancel the estimated offset:

\[
\phi_{\text{cls}} = -k_g \, g_{\text{left-center}} - \hat\phi_{\text{off}},
\]
\[
\phi_{\text{quad}} = -k_g \, \hat g - \hat\phi_{\text{off}} - \tfrac{\pi}{2}.
\]

This is not a “patch”; it is the correct control law obtained by setting the *total phase* at a representative \(g\) to a desired value, given your measurement model. It directly addresses the identifiability barrier caused by the latent additive phase.

## Concrete code changes

All of the following edits belong to `gravimeter_hierarchical_local_holevo.py`. fileciteturn0file0  
One additional (optional but recommended) edit belongs to `gravimeter_hierarchical_pf_complete.py`. fileciteturn0file1

### Replace your local Holevo helper with a safe moment-based implementation

Replace the entire `_local_holevo_from_pair()` with:

```python
def _local_holevo_from_pair(
    self,
    weights0: tf.Tensor,
    particles0: tf.Tensor,
    weights1: tf.Tensor,
    particles1: tf.Tensor,
    q0: tf.Tensor,
    q1: tf.Tensor,
    mid: tf.Tensor,
    width: tf.Tensor,
) -> tf.Tensor:
    """
    Returns clipped V_H for *diagnostics* (not for training loss).

    Key correctness property:
    - If qk == 0, that branch's moment is forced to 0 so inactive-branch NaNs
      cannot contaminate the mixture moment (0 * NaN must not appear).
    """
    dtype = self._dtype
    width = tf.maximum(width, tf.cast(1e-8, dtype))
    k_loc = 2.0 * tf.cast(pi, dtype) / width  # (bs,)

    g0 = particles0[:, :, 0]
    g1 = particles1[:, :, 0]

    phase0 = k_loc[:, None] * (g0 - mid[:, None])
    phase1 = k_loc[:, None] * (g1 - mid[:, None])

    re0 = tf.reduce_sum(weights0 * tf.cos(phase0), axis=1)
    im0 = tf.reduce_sum(weights0 * tf.sin(phase0), axis=1)
    re1 = tf.reduce_sum(weights1 * tf.cos(phase1), axis=1)
    im1 = tf.reduce_sum(weights1 * tf.sin(phase1), axis=1)

    # Exact mixture semantics: if qk == 0, that component contributes 0.
    q0_pos = q0 > tf.cast(0.0, dtype)
    q1_pos = q1 > tf.cast(0.0, dtype)
    re0 = tf.where(q0_pos, re0, tf.zeros_like(re0))
    im0 = tf.where(q0_pos, im0, tf.zeros_like(im0))
    re1 = tf.where(q1_pos, re1, tf.zeros_like(re1))
    im1 = tf.where(q1_pos, im1, tf.zeros_like(im1))

    mu_re = q0 * re0 + q1 * re1
    mu_im = q0 * im0 + q1 * im1

    # Abs moment squared in [0,1]. Use a numerical floor only.
    eps = tf.cast(1e-20, dtype)
    abs_mu_sq = tf.maximum(mu_re * mu_re + mu_im * mu_im, eps)

    vh = 1.0 / abs_mu_sq - 1.0
    return tf.clip_by_value(vh, 0.0, tf.cast(self.cfg.local_holevo_clip, dtype))
```

### Add a training-safe local Holevo loss (no clipping)

Add this new method to the bank class:

```python
def current_local_holevo_loss(self) -> tf.Tensor:
    """
    Training-safe Holevo-style loss:  -log(|mu|^2 + eps)

    Exactly equals log(1 + V_H) without constructing/clipping V_H:
        log(1 + V_H) = -log(|mu|^2)
    """
    dtype = self._dtype
    width = tf.maximum(self.g_hi_vec - self.g_lo_vec, tf.cast(1e-8, dtype))
    mid = 0.5 * (self.g_lo_vec + self.g_hi_vec)

    q0 = tf.where(self.refining_mask, tf.ones((self.bs,), dtype=dtype), self.mode_weights[:, 0])
    q1 = tf.where(self.refining_mask, tf.zeros((self.bs,), dtype=dtype), self.mode_weights[:, 1])

    # Compute |mu|^2 using the same moment logic as _local_holevo_from_pair,
    # but without forming V_H (and with inactive-branch gating).
    k_loc = 2.0 * tf.cast(pi, dtype) / width

    g0 = self.particles0[:, :, 0]
    g1 = self.particles1[:, :, 0]
    phase0 = k_loc[:, None] * (g0 - mid[:, None])
    phase1 = k_loc[:, None] * (g1 - mid[:, None])

    re0 = tf.reduce_sum(self.weights0 * tf.cos(phase0), axis=1)
    im0 = tf.reduce_sum(self.weights0 * tf.sin(phase0), axis=1)
    re1 = tf.reduce_sum(self.weights1 * tf.cos(phase1), axis=1)
    im1 = tf.reduce_sum(self.weights1 * tf.sin(phase1), axis=1)

    q0_pos = q0 > tf.cast(0.0, dtype)
    q1_pos = q1 > tf.cast(0.0, dtype)
    re0 = tf.where(q0_pos, re0, tf.zeros_like(re0))
    im0 = tf.where(q0_pos, im0, tf.zeros_like(im0))
    re1 = tf.where(q1_pos, re1, tf.zeros_like(re1))
    im1 = tf.where(q1_pos, im1, tf.zeros_like(im1))

    mu_re = q0 * re0 + q1 * re1
    mu_im = q0 * im0 + q1 * im1

    eps = tf.cast(1e-20, dtype)
    abs_mu_sq = tf.maximum(mu_re * mu_re + mu_im * mu_im, eps)

    return -tf.math.log(abs_mu_sq)
```

### Store the correct Holevo loss in your loss snapshots

In `apply_measurement()` in the disambiguation block, replace:

```python
local_vh = self._local_holevo_from_pair(...)
self._loss_local_holevo_snapshot = tf.where(dis_rows, local_vh, self._loss_local_holevo_snapshot)
```

with:

```python
local_holevo_loss = self.current_local_holevo_loss()
self._loss_local_holevo_snapshot = tf.where(dis_rows, local_holevo_loss, self._loss_local_holevo_snapshot)
```

This makes the stored quantity already equal to \(-\log(|\mu|^2+\varepsilon)\), so the loss no longer clips flat.

### Fix `hierarchical_loss_components()` to use the snapshot directly (no log1p)

In `hierarchical_loss_components()`, replace:

```python
local_holevo = tf.math.log1p(self._loss_local_holevo_snapshot)
```

with:

```python
local_holevo = self._loss_local_holevo_snapshot
```

Because your snapshot is now already the log‑Holevo loss.

### Use the training-safe Holevo loss for the controller input normalization

In `generate_input()` (simulation class), replace the block:

```python
vh_local = bank.current_local_holevo()
vh_local_norm = tf.math.log1p(vh_local) / tf.math.log(tf.cast(1.0 + bank.cfg.local_holevo_clip, self._dtype))
vh_local_norm = tf.clip_by_value(vh_local_norm, 0.0, 1.0)
```

with:

```python
holevo_log = bank.current_local_holevo_loss()
# Normalize by the maximum possible value given the epsilon floor.
holevo_max = -tf.math.log(tf.cast(1e-20, self._dtype))
vh_local_norm = tf.clip_by_value(holevo_log / tf.maximum(holevo_max, tf.cast(1e-8, self._dtype)), 0.0, 1.0)
```

This retains the same 9-input interface but removes saturation caused by the hard clip.

### Add a phase-offset estimate and subtract it in the phase anchors

Add this helper method to your bank:

```python
def phi_off_circular_mean(self) -> tf.Tensor:
    """
    Circular mean of phi_off under the current (mixture) posterior.

    Returns 0 if phi_off is not inferred.
    """
    dtype = self._dtype
    if not self.phys_model.cfg.infer_phi_off:
        return tf.zeros((self.bs,), dtype=dtype)

    idx_phi = 1  # infer_mfg_bias=False in your run config
    phi0 = self.particles0[:, :, idx_phi]
    phi1 = self.particles1[:, :, idx_phi]

    re0 = tf.reduce_sum(self.weights0 * tf.cos(phi0), axis=1)
    im0 = tf.reduce_sum(self.weights0 * tf.sin(phi0), axis=1)
    re1 = tf.reduce_sum(self.weights1 * tf.cos(phi1), axis=1)
    im1 = tf.reduce_sum(self.weights1 * tf.sin(phi1), axis=1)

    q0 = tf.where(self.refining_mask, tf.ones((self.bs,), dtype=dtype), self.mode_weights[:, 0])
    q1 = tf.where(self.refining_mask, tf.zeros((self.bs,), dtype=dtype), self.mode_weights[:, 1])

    re1 = tf.where(q1 > tf.cast(0.0, dtype), re1, tf.zeros_like(re1))
    im1 = tf.where(q1 > tf.cast(0.0, dtype), im1, tf.zeros_like(im1))

    mu_re = q0 * re0 + q1 * re1
    mu_im = q0 * im0 + q1 * im1
    return tf.atan2(mu_im, mu_re)
```

Then, in `_residual_outputs_to_controls()`, modify your phase-anchor definitions:

```python
phi_hat = bank.phi_off_circular_mean()

phi_cls  = wrap_to_pi_tf(-kg_des * left_center - phi_hat)
phi_quad = wrap_to_pi_tf(-kg_des * g_mean - phi_hat - 0.5 * tf.cast(pi, dtype))
```

This is the correct control law under your likelihood model. The likelihood in `gravimeter_model_complete.py` uses `phi_total = phi_off + mw_phase`; subtracting the posterior mean of `phi_off` is the correct way to place the experiment at the intended phase point. fileciteturn0file2

## One small but important optional fix in the base hierarchy

In `gravimeter_hierarchical_pf_complete.py`, when transitioning into refinement (`will_refine=True`), you currently keep `particles1` unchanged (it remains whatever it was before). fileciteturn0file1

Since \(q_1=0\) in refinement, branch 1 is semantically irrelevant and may be set to any finite placeholder without changing inference. The safest is to re-sample it in-bounds so it can’t poison any diagnostic features.

Change the `new_particles1 = ...` line under `_advance_ready_rows()` so that under `will_refine` you sample fresh interval particles instead of reusing the old branch:

```python
new_particles1 = tf.where(
    will_refine[:, None, None],
    self._sample_particles_for_interval(new_lo, new_hi, rangen),
    tf.where(will_split_again[:, None, None], right_p, self.particles1),
)
```

This does not alter the posterior because `mode_weights=[1,0]` in refinement; it only prevents inactive-branch garbage from leaking into any computations.

## Why these fixes are “correct” rather than heuristics

They are forced by algebraic equivalences and model semantics:

- Replacing `log1p(clip(V_H))` with `-log(|μ|^2+ε)` is not a new loss; it is exactly the same Holevo objective without a gradient-killing clip. The equivalence \(\log(1+V_H)=-\log(|\mu|^2)\) is identity-level math.
- For refinement, “inactive branch contributes zero” is literally the definition of a mixture with \(q_1=0\). Forcing its moment to 0 is exactly correct and eliminates \(0\times \mathrm{NaN}\) contamination.
- Subtracting \(\hat\phi_{\text{off}}\) in phase anchoring is required by your own likelihood model structure; the gravimeter paper’s discussion of control errors as phase offsets is consistent with this requirement. fileciteturn0file9 fileciteturn0file10
- Belliardo et al. explicitly motivate model-aware gradients through PF updates and differentiable resampling; a clipped-constant loss breaks that premise by construction. fileciteturn0file11

## What you should see after applying the fixes

The simplest “signs” that training is now actually working:

- The debug rollout JSONL contains **zero NaN** controls (no duplicated frozen `global_step` records at the refinement boundary).
- `grad_norm` becomes **nonzero** in training logs (because gradients are no longer sanitized to all zeros after refinement NaNs).
- At level 0, `g_std` begins to drop below the uniform prior baseline, and `q0` begins to deviate from 0.5 within the first few shots.
- `loss_local_holevo` is no longer pinned at 4.615 (= log(101)); it should start high and then decrease as \(|\mu|\) increases.

Once those hold, optimizing the remaining aspects (shot allocation per level, weight scaling, etc.) becomes meaningful; right now it isn’t, because the model is not even producing finite refinement controls and the main shaping loss is flat where you need it.

