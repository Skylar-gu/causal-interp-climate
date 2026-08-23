*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Prereg — PCMCI+ proposes, intervention disposes (hurricane features), 2026-08-20

Frozen BEFORE any analysis script is written and before any edge is looked at.
Companion de-risking probe: `flagship_sae/p0_transmission_probe.py` (NOT a scored result —
it decides whether this design runs at all, and its outcome is reported either way).

## Why this design

Observational causal discovery has failed five times in this repo — the concept graph (VOID on
its permutation control), PX_geo (point-mass null, instrument failure), the flagship
watching-graph (anchor gate breached by `qrand` at p=0.028, *more* significant than the
positive control at 0.033), the 6→8→9 chain (absent on the alias-repaired basis with both
anchors clean), and the tropical hub (all consensus lost at τ_max=48). Interventions have
worked every time they were run.

The response is not to run PCMCI+ again and hope. It is to **demote it to a hypothesis
generator and score its output by intervention** — which is possible here and impossible in the
atmosphere, because GraphCast can be intervened on. The claim under test is therefore not
"PCMCI+ recovers causal structure". It is:

> **Edges that PCMCI+ proposes are more likely to survive an interventional test than
> pairs matched to them on every confound we know about.**

That is falsifiable, it is cheap, and a clean negative is the sixth entry in the falsification
log — and the first one obtained *interventionally* rather than by a permutation anchor.

## The statistic, and why it is an asymmetry

The obvious statistic — ablate A, measure B — will not survive. The SAE is an overcomplete
dictionary; feature footprints overlap; this repo has measured edge rate rising from 1.2% at
zero footprint overlap to 14–17% above cosine 0.45. Ablating A mechanically perturbs the field
where B reads, so a raw ΔB is largely reconstruction leakage.

**Footprint overlap is symmetric by construction.** So score

```
asym(A, B, tau) = |dB(t+tau)| / B_base  -  |dA(t+tau)| / A_base
```

where `dB` is the change in B's in-box activation when A is ablated to its restore-to-normal
level, and `dA` is the change in A's when B is ablated. Leakage enters both terms and
differences out. PCMCI+ named a *direction*; the intervention either confirms that direction or
it does not. The reverse arm IS the negative control, so the control cannot be a no-op.

## Design

**Realizations.** 8 NH storms (`skill_conv_storms`) + 5 SH storms (`skill_sh_storms`), each at
IC offsets {−48, −24, 0, +24} h = 52 realizations x 16 steps.

**Feature selection — label-free, fixed before the graph is built.**
- in-box mean activation > 0 in ≥6 of 8 NH storms;
- rank by median in-box activation across storms, take top 20;
- force-include the convection features 2401 / 2067 / 3174 (known interventional effect →
  positive control), dropping the lowest-ranked to hold N = 20;
- **feature 3243 (the TC readout) is excluded from the pair set** and kept as an outcome node
  only. `skill_conv_run.py` asserts against ablating it, and the `mech_atm_river` failure is on
  record as a group that camouflaged itself by containing the readout.
- Concept-library labels are NOT used for selection. Per the 2026-08-17 reconciliation, 24 of
  40 concept features sit poleward of 65°, six of ten "concepts" are polar features wearing
  dynamical names, and the `atm_river` detector (q850·|V850|) selects TC features by
  construction. Selected features are labelled *post hoc* and descriptively, and this note is
  cited when they are.

**Conditioning gate, reported before the graph (guardrail #5).** `cond`, `min eigenvalue`, and
`max |corr|` on the stacked 52 x 16 x 20 series — never a median or an effective rank alone. If
`max |corr| > 0.95`, one of the pair is dropped and the three numbers are re-reported. The
Nyquist clique (mutual |r| 0.94–0.99, cond 4.8 → 757) is exactly this failure mode.

**PCMCI+.** ParCorr, `tau_min=1`, `tau_max=4` (24 h), `pc_alpha=0.05`,
`analysis_mode='multiple'`, `reference_points` excluding the first τ_max steps of each
realization. Effective samples 52 x (16−4) = 624 against N·τ_max = 80, ratio 7.8 — reported,
and not allowed below 5.

**Intervention.** Top 10 edges by |MCI|, plus 10 matched non-edge pairs. Two arms per pair
(ablate A; ablate B), restore-to-normal, on the 8 NH storms.

**Readouts.** (1) `asym` at the PCMCI+-named lag τ; (2) `asym` at τ+2 (lag specificity);
(3) storm ΔMSLP at +96 h, so every edge also carries a forecast consequence.

## The matched non-edge control

Matched on three axes simultaneously, each by **nearest-to-target, never argmax inside a
tolerance band** — that bug pushed `core_control`'s draws to +35% on 9 of 9 picks and handed
the control a head start in the comparison it was refereeing:

1. footprint cosine |cos(A,B)| within ±0.05 — kills the leakage confound;
2. marginal |Pearson r| at lag τ within ±0.05 — kills "edges are just the correlated pairs";
3. parent in-box firing amplitude within ±25% — kills "we ablated something bigger".

### Guardrail #9, all three legs

- **(i) The null VARIES.** Non-edge asymmetries are differences of two noisy ablation effects
  and must scatter around zero with real spread. **The IQR is reported.** If it collapses to a
  point mass the instrument has failed and this is reported as instrument failure, not as a
  pass — that is how PX_geo died.
- **(ii) The bar is ATTAINABLE under the null.** **≥2 of 10 matched non-edges must individually
  exceed the edge-set median.** If 0 of 10 can reach it, the bar sits above the null's ceiling
  (the BSF block-threshold failure) and the result is VOID.
- **(iii) A negative control FAILS it.** Two, neither able to be a no-op: the **reverse-lag
  arm** (same two rollouts read the other way) must not show positive asymmetry, and the
  **wrong-lag arm** (score at τ+2) must be weaker than at τ. A flat lag profile is leakage, not
  mechanism.

## Pre-registered bars

- **B1 (primary).** Median `asym` of PCMCI+ edges > median `asym` of matched non-edges,
  one-sided Wilcoxon signed-rank **p < 0.05**, n ≥ 10 pairs.
- **B2 (direction).** **≥7 of 10** PCMCI+ edges have `asym > 0`.
- **B3 (lag specificity).** `asym(τ) > asym(τ+2)` for **≥7 of 10** edges.
- **B4 (null calibration — must pass or the result is VOID).** Non-edges: `|median asym| < 0.5 x`
  edge median AND IQR > 0 AND ≥2 of 10 exceed the edge median.
- **B5 (positive control).** The convection features must show positive asymmetry toward
  storm-core features. **If B5 fails the instrument is underpowered and the run is reported as
  instrument failure, not as a negative result.**

**FALSIFIED IF** B1 fails, or B2 ≤ 5 of 10, or B4's variance leg fails.

A clean negative is a result and will be reported as one: *PCMCI+ edges are indistinguishable
from footprint-matched non-edges under intervention.*

## P0 — the probe that can kill this in 20 minutes

One storm (`ida2021`), `MECH_TRACK=all`, three arms: baseline, ablate 2067 (known strong),
ablate a known-weak but nonzero in-box feature.

- **KILL CONDITION.** If ablating a strong feature moves no other tracked feature by >10% of
  its baseline in-box amplitude, there is no measurable feature-to-feature transmission, the
  edge test is dead, and the 12 h pivots to the commitment-horizon sweep below plus the
  outstanding multi-draw rotation controls. That negative is reported.
- **CALIBRATION.** Regress |ΔB| on footprint cos(2067, B) across all 4096. R² > 0.8 means
  leakage dominates and the asymmetry statistic is load-bearing rather than merely prudent.

## Companion result — the commitment horizon

Separate claim, same machinery, reported independently of the above.

Ablate the convection group **at a single timestep k only** and measure ΔMSLP at +96 h, for
k ∈ {0,1,2,3,4,6,8,11,15}. `_sched` already accepts a length-H gain schedule.

**Question:** *when* is the model's convection representation load-bearing? A sharply
front-loaded damage curve says the model commits to the storm's fate within the first 24 h and
that later convection is decorative — a statement about how an autoregressive weather model
represents time, not just about which features matter.

**Control.** The identical pulse sweep on the in-box matched control group, which fires at
87–112% of the convection group in these boxes and therefore CAN fail.

**Bar.** Convection pulse damage at k ≤ 3 exceeds damage at k ≥ 8 by **≥2x** in **≥6 of 8**
storms, AND the in-box control shows no such ratio (< 1.3x). If the control also front-loads,
the finding is about early-lead sensitivity in general and is reported as that instead.

## Recorded in advance

`activations/mode_series/feat_traj_3yr_ALL4096.npy` and `feat_traj_3yr.npy` are **97.45%
zero rows** (4263 of 4383; preallocated for 3 years, stopped at n_done=120) with corrupt
`target_times` past row 120. Guardrail #6. Neither file is used by anything in this design, and
this note records why they must not be reused by anything else.
