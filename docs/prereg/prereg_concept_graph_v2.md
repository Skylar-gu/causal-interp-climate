*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — CONCEPT causal graph v2, the PURITY GATE (CG-v2)

**Frozen 2026-08-14, before `flagship_sae/concept_cgraph_v2.py` is run and before any v2
causal number exists.** The v1 numbers it is compared against are already published
(`results/fs_concept_cgraph.npy`, `results/concept_cgraph_score_existing.txt`) and are quoted
here so they cannot be moved later. The v1 prereg (`notes/prereg_concept_graph.md`, frozen
ec90ea7) stands unchanged; this document amends only the **node definition** and the
**readout combiner**, and adds a dose sweep. Every CG-1 bar, the CG-4 battery and the CG-2
NEG bar are carried over **verbatim**.

## Why v1 was VOID, and what the diagnosis was

CG-1 asked whether the concept graph reproduces across disjoint initial-condition sets. It
did: ρ **+0.976**. But the CG-2 NEG control — the SAME 150 features randomly re-partitioned
into 10 meaningless groups of 15 (seed 0) — reproduced at ρ **+0.893** against a < 0.30 bar,
and **beat** the real concepts on sign agreement (0.911 vs 0.711) and top-10 overlap (9/10 vs
7/10). The graph was measuring **dose geometry**, not concept identity. CG-1 was declared
VOID and CG-4 (7/10, INCONCLUSIVE) was not licensed.

The diagnosed cause is **impure exemplars**. v1 took the top 15 features by |z| per concept
with no check that a feature belongs to *one* concept. Margin (`z_top − z_second`) of those
top-15 sets, recomputed from `results/fs_atlas_extra.npy` and reproduced exactly here:

| concept | median margin | min | n at margin ≥ 0.5 |
|---|---|---|---|
| q600 | 1.05 | 0.20 | 337 |
| atm_river | 1.34 | 0.31 | 50 |
| blocking | 0.86 | 0.33 | 73 |
| ascent | 0.85 | 0.22 | 20 |
| t850 | 0.65 | 0.51 | 70 |
| jet250 | 0.62 | 0.13 | 32 |
| vort850 | 0.39 | 0.07 | 4 |
| z500 | 0.34 | 0.07 | 56 |
| **shear** | **0.27** | **0.01** | **6** |
| baroclinicity | 0.25 | 0.02 | 4 |

A margin of 0.01 means the feature is labelled two concepts equally. **`shear` is the worst,
and it carries R7 — one of the two critical negative tests.** Its v1 group was substantially
not shear.

## THE FROZEN SELECTION RULE (v2)

Implemented in `flagship_sae/cgv2_select.py`, run **before** this prereg was committed
because it produces **no causal number** — it is node definition, exactly as v1 declared
`div250` struck before its run.

1. **ALIVE** — `zcnt >= 300`.
2. **LABELLED** — `z_top >= 1.0`, `z_top = max_c |z[:,c]|` over the ten concepts.
3. **PURE** — `z_top − z_second >= 0.5`. *The new gate.* v1 had none.
4. **DECORRELATED** — greedy over the pure candidates in descending `|z_c|`: add a feature
   only if the **absolute Pearson correlation of its ACTIVATION SERIES** with every
   already-chosen feature is `< 0.5`, so the K features span the concept instead of
   repeating it.
   Series basis (frozen): `results/fs_cgv2_actseries.npy` — the already-extracted IID
   activation dump, 160 windows spanning 2016-01 .. 2020-12 over 57 distinct months (the
   same sample the representation atlas was built on), SAE-encoded and summed over all
   40,962 mesh nodes per window.
5. **K = 4**, fixed across all concepts. That is what the purity gate affords everywhere:
   `vort850` and `baroclinicity` have exactly 4 pure candidates.
6. Any concept that cannot fill K is **STRUCK and reported as a coverage gap**, the way
   `div250` was in v1.

### Result of the rule — declared here, BEFORE the run

| concept | alive | +labelled | +pure | chosen | PC1 evr | max &#124;r&#124; within group |
|---|---|---|---|---|---|---|
| vort850 | 35 | 17 | 4 | **4** | 0.319 | 0.198 |
| q600 | 811 | 627 | 337 | **4** | 0.421 | 0.350 |
| ascent | 61 | 44 | 20 | **4** | 0.374 | 0.438 |
| shear | 254 | 88 | 6 | **4** | 0.399 | 0.417 |
| t850 | 668 | 330 | 70 | **4** | 0.535 | 0.480 |
| z500 | 763 | 289 | 56 | **4** | 0.370 | 0.451 |
| jet250 | 285 | 177 | 32 | **4** | 0.397 | 0.335 |
| blocking | 288 | 146 | 73 | **4** | 0.425 | 0.469 |
| atm_river | 414 | 133 | 50 | **4** | 0.369 | 0.294 |
| baroclinicity | 377 | 93 | 4 | **4** | 0.413 | 0.488 |

**STRIKES: none.** All ten concepts fill K = 4. `div250` remains struck from v1 (0 features
at argmax), so R1 remains **NOT TESTABLE** and is not scored. Coverage is therefore 10/10
concepts and 10/11 relations, unchanged from v1.

Not one v1 group survives intact: the purity gate replaces the exemplar sets wholesale. The
frozen feature indices are stored in `results/fs_cgv2_groups.npy`.

## The readout combiner — and a diagnostic that forces a declared change

The intended combiner was: **the first principal component of the K standardized series**,
because a sum keeps what is idiosyncratic to each feature while the first PC keeps only what
they share, and because PC1 is scale-free so group-size effects vanish.

**The selection diagnostic above shows PC1 cannot do that job here, and this is recorded
before any causal number exists.** Step 4 decorrelates the K series by construction
(|r| < 0.5), so there is little shared variance left for a first PC to find:

- PC1 explained-variance ratio, **real** concepts: 0.319 – 0.535 (median 0.398).
- PC1 explained-variance ratio, the **random** perm groups: 0.315 – 0.474 (median 0.360).
  The real groups are **not** distinguishable from random ones on this statistic. Four
  exactly independent series would give 0.25.
- Seven of the ten real PC1 loading vectors have **mixed signs** (e.g. q600
  [+0.57, −0.54, +0.43, −0.45]). PC1 is a **contrast between the group's features**, not
  their shared component.

A mixed-sign contrast is a hostile readout for CG-2 POS: dosing a concept raises all four of
its features, and a contrast can cancel that to near zero or negative, which would void the
run for a reason that has nothing to do with concept identity.

**Declared resolution — three readouts, all computed from ONE GPU pass, all reported
unconditionally, none selected after the fact.** The run stores the full 4,096-vector
activation delta for every (IC, γ, dosed group), so every readout is a post-hoc projection of
the same stored numbers and no arm is re-run:

- **RO-A `pc1`** — projection on the first PC of the K standardized series, exactly as
  originally specified: `A[i,j] = Σ_k w_j[k] · d[f_jk] / sd_jk`, `w_j` unit-norm, sign fixed
  so `Σ_k w_j[k] > 0`.
- **RO-B `zmean`** — the sign-coherent version of the same idea: equal positive unit weights
  on the standardized series, `A[i,j] = (1/K) Σ_k d[f_jk] / sd_jk`. Also scale-free and also
  independent of group size, but it cannot cancel a coherent response.
- **RO-C `sum`** — the raw summed activation change, **identical to v1's readout**, so the
  v1→v2 comparison is like-for-like and the effect of the purity gate alone is isolated.

`sd_jk` is the standard deviation of feature `f_jk` in the frozen 160-window series.

**HEADLINE RULE, declared now and deterministic:** the headline verdict is **RO-A (`pc1`)**,
as originally specified — *unless* RO-A fails the CG-2 POS calibration below, in which case
the headline is **RO-B (`zmean`)**. All three are reported in full either way, with their own
complete CG-1 / CG-2 / CG-3 / CG-4 lines. Nothing is chosen by which result is nicer.

## The intervention — otherwise identical to v1

For source concept *i*: `coef_patch(sae, features(i), γ)` at t, advance **one 6-h step**, read
every concept *j*'s activation change against an undosed baseline. Same doser, same
one-6h-step readout, same ICs, same `fs_common.delta` algebra as v1.

**Two corrections carried in from the fix list, recorded so they are not re-litigated:**

- *"Match injected mass"* was confused. `fs_common.delta` computes `d = (f * coef) @ W_dec.T`
  — the dose is **already** proportional to each feature's own current activation, so
  `coef = 1.0` means "+100 % of what that feature is doing right now". There is nothing to
  fix, and no equal-absolute-injection variant is run. v1's CG-3 already measured the
  magnitude confound at Spearman **+0.110**, which is low.
- *"Dose smaller"* is weakly supported. The flagship lane measured feature interactions at
  ~0.82 non-additive **and scale-invariant across 0.10 ≤ γ ≤ 1.0**, so a smaller dose is not
  expected to make effects more additive. The sweep below is run to **answer** the open
  question, not because it is expected to help.

ICs, reused verbatim from v1 so the reproducibility split matches:
- **Set A (6):** 2018-09-10, 2020-01-05, 2020-04-06, 2020-07-06, 2020-10-05, 2019-06-15
- **Set B (4, disjoint):** 2020-02-20, 2020-05-25, 2020-08-15, 2020-11-20

**Dose sweep (new):** every arm is run at **γ ∈ {0.25, 0.5, 1.0}**. γ = 1.0 is the primary
and the only one that carries a verdict; 0.25 and 0.5 are reported. The pre-registered
question the sweep answers, which is genuinely open: **does the perm-control ρ depend on
dose?** Reported as a three-point table of perm ρ vs γ whatever it shows, including "flat".

## CG-2 CALIBRATION (guardrail #9) — both sides, read BEFORE CG-1

- **POS** — each concept's self-effect (diagonal, computed but excluded from A) must be the
  largest entry in its own row. Bar carried over from v1 unchanged: **≥ 9/10**. If dosing a
  concept does not move that concept most, the readout is broken and there is no verdict.
- **NEG** — `concept_perm`: the SAME 40 selected features randomly re-partitioned into 10
  groups of 4 (seed 0), dosed and read through the identical code path in the same pass. Its
  ρ between IC sets must be **< 0.30**.
  **v1 measured +0.893 here.** If it is still ≥ 0.30, CG-1 is **VOID again**, the purity gate
  did not rescue it, and that is reported as the result — a real negative about whether SAE
  concepts have recoverable causal structure at all.

The NEG bar is not vacuous and is not a point mass: it is a full re-run of the whole
pipeline, it produced +0.893 in v1, and its value is free to land anywhere in [−1, +1].

## CG-1 PRIMARY — is the concept graph reproducible?

Bars carried over from v1 **verbatim**, off-diagonal cells only:

- **REPRODUCIBLE** iff ρ ≥ 0.50 **and** sign agreement ≥ 0.70 **and** top-10 shared ≥ 6/10.
- **NOT REPRODUCIBLE** iff ρ < 0.30.
- Anything between is **PARTIAL**; CG-4 is then descriptive only.

Reference values that cannot move: feature level ρ +0.181 / sign 0.513 / top-20 6/20;
v1 concept level ρ +0.976 / sign 0.711 / top-10 7/10, **VOID** on the NEG.

### Matrix transforms — declared, all three reported

The second load-bearing idea in v2 is that **the perm control cancels label-blind structure
by construction**, and both graphs come out of the same pass, so the cancellation is free.

- **T-raw** — `A` as measured. **This is the transform that carries the CG-2 gate and the
  headline CG-1 verdict**, because it is the only one that is like-for-like with v1 and
  therefore the only one that answers "did the perm ρ move from +0.893?".
- **T-diff** — `D = A_real − A_perm`, entrywise, perm group *r* taken at position *r*.
  Declared limitation, stated before the run: the pairing of a real concept to a perm group
  is arbitrary, so this subtracts a *sample* of the label-blind field rather than its
  conditional expectation. Reported, not used to overturn T-raw.
- **T-dc** — double-centering of the off-diagonal cells,
  `Ã[i,j] = A[i,j] − rowmean_i − colmean_j + grandmean`, applied **identically to real and
  perm**. This removes the multiplicative "how much does i inject × how responsive is j"
  dose geometry without needing any pairing, and is the principled form of the same idea.
  The NEG bar applies here too: if perm ρ stays ≥ 0.30 after double-centering, the
  reproducibility is not interaction structure at all.

## CG-3 — magnitude confound

Spearman(|A[i,j]|, source total effect), reported for every readout alongside every CG-1
number, never omitted. v1: **+0.110**.

## CG-4 — THE ANSWER KEY (unchanged from v1, 10 testable relations)

| # | relation | sign |
|---|---|---|
| R1 | ascent → div250 | + | **STRUCK — div250 has no features** |
| R2 | t850 → z500 (hydrostatic thickness) | + |
| R3 | jet250 → baroclinicity (thermal wind) | + |
| R4 | baroclinicity → vort850 (baroclinic growth) | + |
| R5 | q600 → ascent (latent heating) | + |
| R6 | ascent → q600 (moisture convergence) | + |
| R7 | shear → ascent (shear suppresses organized convection) | **−** |
| R8 | blocking → jet250 (blocking diverts the jet) | **−** |
| R9 | q600 → atm_river (moisture transport) | + |
| R10 | atm_river → ascent (AR-forced ascent) | + |
| R11 | baroclinicity → z500 (baroclinic height response) | + |

Primary score: signed-edge accuracy over the 10 testable relations against a 0.50 chance
line, one-sided binomial. **CONFIRMED ≥ 8/10**, **NULL ≤ 6/10**, 7/10 INCONCLUSIVE. Scored
only if CG-1 is REPRODUCIBLE or PARTIAL; if the CG-2 NEG fails, CG-4 is reported as
descriptive only and carries no verdict.

**R7 and R8 are the sharpest tests in the whole battery, and this is why.** Spatial
co-occurrence — which is how the atlas labels features in the first place — is **positive by
construction**: two feature groups that fire in the same places light each other up. It can
never produce a negative edge. A correctly-signed **negative** is therefore something
co-occurrence cannot fake, while any positive edge can be explained away by it. In v1 both
came back positive (R7 +3.4, R8 +14.3), exactly consistent with the graph reading overlap
rather than physics. **If the purity gate works, R7 is where it will show**, because `shear`
was the least pure group in v1 (median margin 0.27, min 0.01) and is the group the gate
changed most.

Pre-declared downgrade, carried over from v1: if the overall score is CONFIRMED but R7 and R8
are both positive, the result is downgraded to *"co-occurrence recovered, causation not
demonstrated"*.

## CG-5 — the co-occurrence confound

Unchanged. Report asymmetry `|A[i,j] − A[j,i]| / (|A[i,j]| + |A[j,i]|)` for every scored
pair, and feature-set Jaccard for every scored pair (expected 0 by construction, recorded so
the expectation is on file).

## CG-6 — what is NOT claimed

- No propagation, speed or `V_MAX` reading. Concepts have no locations.
- The timescale column is not scored: the run is a single 6-h step and can test sign only.
- Flagship-internal. Not comparable to the mini pool, the mini SAE, or SAVAR.
- v2 does not license the observational extraction; that clause of the v1 prereg is retired
  independently, because the observational route already failed its own anchor gate today.

## What a MISS looks like, declared

If the CG-2 NEG perm ρ is still ≥ 0.30 at γ = 1.0 under T-raw, the verdict is: **the purity
gate did not rescue CG-1; the interventional concept graph is dominated by dose geometry and
does not carry concept identity.** That is reported as the headline, plainly, with the
CG-4 score shown as descriptive only. A clean negative here is a real result about the
recoverability of causal structure from SAE concepts, and it is the expected outcome if the
v1 failure was not about exemplar purity.
