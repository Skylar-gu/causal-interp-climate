*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — flagship retry #2 steering, v2 (repaired readout)

Frozen **2026-08-10, before `fs_retry2_steering_v2.py` was written and before any v2 number
existed.** Supersedes `notes/prereg_flagship_g2_suite.md` §4, whose readout was found
structurally void (`notes/flagship_steering_s0_defect.md`). v1's `PASS` is **withdrawn as VOID**
and is not carried forward in any form.

Flagship-internal throughout. **NOT** comparable to the small-model 0.022 or the SAVAR 0.81.

## 0. What was wrong, and what each change repairs

| # | v1 defect | v2 repair |
|---|---|---|
| 1 | Primary lead `s=0`: patch applied at layer 8, readout reads layer 8 ⇒ `dA` **is** the injected delta. Linear in alpha to 1.2e-07; leakage bit-identical at every dose. | **Primary lead `s=1` (12 h), patch OFF.** `s=0` is never scored. `s=2` (18 h) reported as secondary. |
| 2 | `R_c = <outer(W_c,Q_c)/sigma_c, dA>` is a *correlation* with each mode, not a *projection* onto them; the 7 channels are non-orthogonal (\|cos\| max 0.605, stable rank 3.31/7), manufacturing leakage from geometry alone (floor up to 0.241). | **Feature-space readout.** Communities are a **hard partition** of SAE features ⇒ no basis overlap, **no geometric floor at all**. |
| 3 | `outer(W_c,Q_c)` assumes the community response factorizes rank-1; retry #1 measured identity-R² **0.313**, i.e. ~69% of community structure discarded. | Same repair — no mode approximation is used. |

Repair 2 alone would not have sufficed: at `s=0` in feature space the target feature's own code
change dominates, which biases leakage **favourably**. Repair 1 is independently mandatory.

## 1. Readout (frozen)

At each roll step `s`, for the layer-8 activation `A` (N=40962 mesh nodes x B windows x 512):

    F      = TopK-encode(A)                      # (N*B, 4096), the SAE's own encoder
    dF     = F(perturbed) - F(baseline)
    rel[c] = || dF[:, members(c)] ||_F  /  || F_base[:, members(c)] ||_F

`members(c)` are the features Leiden assigned to community c (`fs_modes.npz`, 7 communities over
977 active features). Membership is **disjoint**, so no feature contributes to two communities.
`rel` is a dimensionless relative code change, so communities of different size and activity are
directly comparable — which is what makes the off/on ratio meaningful.

**Primary metric.** For handle feature f assigned to community t:

    L_feat = mean over c != t of rel[c]   /   rel[t]

averaged over the 5 doses and 4 windows, at `s=1`.

The v1 mode-basis metrics `L` and `L_dual` are **also computed and always reported** at every
step, for continuity and cross-checking. They are companions; `L_feat` is primary.

## 2. Arms (frozen)

- `noise` — noop repeat. Documents that the CPU forward is bit-deterministic.
- `recon` — full SAE substitution (rho=1). The **power reference**, since the noise floor is
  identically 0 and cannot serve as one.
- 4 handles x 5 doses alpha in {-1, -0.5, +0.5, +1, +2}. Handles are the v1 selection, unchanged
  and not re-picked: f628->c5, f3357->c2, f1632->c4, f2073->c1.
- **2 negative controls**: random features, mass-matched to the handle band, steered at alpha=+1,
  each assigned the community of the handle it is matched to. These are the calibration.

24 arms, 4 windows, 3 steps.

## 3. Bars (frozen — no number has been seen)

**PASS** iff **both**:
1. at least one handle has `L_feat < 0.5` at `s=1`, **and**
2. that handle is **powered**: `rel[t] >= 0.1 x rel[t]` of the `recon` arm in the same community.

**INCONCLUSIVE (metric uninformative)** — overrides any PASS — iff the **median negative-control
`L_feat` is itself < 0.5**. If pushing a random feature also looks localized, the metric cannot
distinguish a handle from noise and no handle result may be claimed. This is guardrail #9
(calibrate both sides) applied to the thing v1 lacked entirely.

**MISS** otherwise.

**Vacuity guards, declared in advance.**
- The noise floor will be exactly 0 (bit-deterministic CPU forward). Any "on-target >= 10x noise
  floor" gate is therefore **vacuous** and is replaced by the `recon`-relative power gate above.
  It will be reported as vacuous, never as passed.
- `s=0` is **not scored under any circumstance**, and the scorer refuses to emit a verdict from
  it (exact-linearity detector already in `fs_score.py`).

**Dose-response sanity check (diagnostic, not a bar).** At `s=1`, gain `rel[t]/|alpha|` and
`L_feat` must vary across doses by **> 1e-5** relative spread. A spread at float32 roundoff is
the v1 defect's signature and would mean the readout is again measuring the injection rather
than the network; if it recurs, the run is declared VOID rather than scored.

## 4. Sample size and its justification

4 windows x 3 steps x 24 arms, CPU. The L4 cannot hold a flagship forward (19.2 GB requested vs
18.9 GB limit) and one CPU forward at batch 2 holds ~110-125 GB RSS, so runs are serial. Est.
~3.5-4 h. Windows are the same 4 as v1 so the comparison is like-for-like.

## 4b. AMENDMENT (2026-08-10) — control set extended from 2 to 10, disclosed as post-hoc

**Declared openly: this was decided AFTER seeing the n=2 control result**, so it is a post-hoc
deviation and is labelled as one. It is admissible for one specific reason, which is checkable
rather than rhetorical:

> **Adding controls can only weaken or neutralize the PASS. It can never strengthen it.**

The control set enters the bars in exactly one place — the INCONCLUSIVE override, which fires
when the *median* control `L_feat` falls below 0.5. More controls either (a) pull the median
below 0.5, flipping PASS -> INCONCLUSIVE, or (b) leave it above, in which case the verdict is
unchanged. There is no configuration of new control data that converts a MISS into a PASS or
makes an existing PASS stronger. A post-hoc change with a one-sided, unfavourable-only effect is
not a fishing expedition.

**What prompted it.** At n=2 the controls split 1.519 / **0.282**. The median (0.900) cleared the
override, but one control sat *below the 0.5 handle bar*, and in community 2 the control
(`f2676`, L_feat 0.282) read as **more localized than the real handle** (`f3357`, 0.466). A
two-point median cannot distinguish "handles are specific" from "low L_feat is easy to hit by
chance". The n=2 calibration was simply too weak to license the PASS it permitted.

**What changes:** `--ctrl-only 8 --ctrl-seed 1`, same design as the frozen controls (random
features, mass-matched to the handle band, alpha=+1, assigned a handle's community), excluding
the two already drawn. Controls are **pooled** to n=10 and the override is re-evaluated on the
pooled median. Handles are **not** re-run and no handle number changes.

**What does not change:** the bar (0.5), the power gate (10% of recon), the primary lead (s=1),
the VOID guard, and the handle results. The n=2 verdict is reported alongside the n=10 verdict so
the effect of the amendment is visible rather than absorbed.

### Outcome of the amendment (run 2026-08-10, 99.1 m)

The extension was a genuine risk to the PASS and **the PASS survived it**.

- Control median **0.900 (n=2) -> 0.976 (n=10)**. INCONCLUSIVE did not fire.
- All 8 new controls read **>= 0.5** (0.511, 0.539, 0.554, 0.971, 0.981, 1.331, 1.513, 1.839).
  The n=2 outlier `f2676` (0.282) was not representative. Sorted n=10:
  0.282, 0.511, 0.539, 0.554, 0.971, 0.981, 1.331, 1.513, 1.519, 1.839.
- Random features therefore read `L_feat` ~0.5-1.8, i.e. **the metric is informative** — it does
  not hand out low leakage for free, which is exactly what n=2 could not establish.

**A new limit surfaced in the same data, recorded rather than set aside.** The frozen bar is an
*absolute* 0.5, and that value sits inside the controls' low tail (**3/10 controls in
[0.5, 0.6]**). Referenced against the control distribution instead, f1632 (0.427) is below only
**1 of 10** controls, giving **p = 0.182** (floor 0.091) — the same near-miss structure as
ablation target #3535 (Q 0.383, p 0.182). f1632 is more specific than 9/10 random features:
suggestive, **not significant at n=10**.

This control-referenced statistic is **descriptive**. It is NOT substituted for the frozen bar —
swapping in a stricter test after seeing the result would be goalpost-moving even though it runs
against the favourable direction. The pre-registered verdict is **PASS**; the margin over chance
is modest and is reported as such.

## 5. Ablation extension, pre-registered in the same breath

`prereg_flagship_g2_suite.md` §3 froze `ntarget=1` purely on CPU budget, so the "at least one
grid-locked feature is inert" bar was applied to a sample of **one** (#2954, NOT inert, Q=1.060,
p=0.545). Detection found exactly **two** candidates. Testing the second (**#3535**) *completes
the intended candidate set* rather than extending it opportunistically.

Declared now, before that run: the combined verdict is over **both** candidates and will be
reported as a **2-target test**, with the same 10 mass-matched controls per target and the same
p-floor 1/11 = 0.091. A PASS requires >= 1 of 2 inert. **#2954's existing MISS stands as-is and
is not re-run.** If #3535 is also not inert, the flagship inertness verdict is a clean 0/2 MISS.
It will not be reported as "we found one" under any outcome.
