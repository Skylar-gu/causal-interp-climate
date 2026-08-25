*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — the CONCEPT causal graph (CG-1 .. CG-6)

**Frozen 2026-08-13, before `flagship_sae/concept_cgraph.py` is run and before any concept-level
number exists.** The feature-level numbers it is compared against (`fs_cgraph.npy`,
`fs_cgraph_rep.npy`) are already published and are quoted here so they cannot be moved later.

## Why

The project's stated deliverable has been a graph among *modes* — objects with locations —
verified against propagation physics (speed = distance / lag). Flagship SAE features do not
support that: median footprint spread is **7,300 km**, and only **6 of 4,096** are compact
enough (<2,000 km) to carry a propagation reading, all six polar.

But locality was never the point. SAE features are **concepts** — sea ice, arid-morning
boundary layers, atmospheric rivers — and the question worth answering is whether the
*interactions between atmospheric concepts* are replicated inside GraphCast. Concepts need no
locations. This prereg turns that into a falsifiable test.

It also changes the epistemic situation. The whole repo is framed as running an
answer-key-free recipe on GraphCast *because no ground truth exists there*. Concept–concept
relationships **are** a ground truth: directional, signed, and with known timescales. CG-4
below is the first genuine answer key this project has had on a real model.

## Concept definition (frozen)

Source: `results/fs_atlas_extra.npy` (the representation atlas), `alive := zcnt > 300`.

A feature belongs to concept *c* iff `argmax_k |z[:,k]| == c` **and** `|z[:,c]| > 1.0`.
Concept node = the **15 highest-|z| such features**.

- **K = 15 is fixed across all concepts**, so the dose is a comparable intervention. Group
  populations are wildly unbalanced (q600 627, vort850 17); dosing whole groups would confound
  concept identity with group size. Using the top-15 exemplars is a *sample* of each concept
  and is described as such.
- **`div250` is EXCLUDED**: 9 features at |z|>1.0, **0** at argmax. Declared before the run.
  Consequence: relation R1 (ascent → div250, mass continuity) is **NOT TESTABLE** and is struck
  from the battery rather than scored. Recorded as a coverage gap, not a miss.

Ten concepts: `vort850 (17 avail), q600, ascent, shear, t850, z500, jet250, blocking,
atm_river, baroclinicity`.

## The intervention (identical to the feature-level run, one level up)

For source concept *i*: `coef_patch(sae, features(i), GAMMA=1.0)` at t, advance one 6-h step,
read every concept *j*'s summed activation change vs an undosed baseline. `A[i,j]` averaged
over ICs. Same doser, same reader, same GAMMA, same ICs as `fs_cgraph.py` — **only the
aggregation level changes**, so the comparison to the feature-level result is like-for-like.

ICs, reused verbatim so the reproducibility split matches the published one:
- **Set A (6):** 2018-09-10, 2020-01-05, 2020-04-06, 2020-07-06, 2020-10-05, 2019-06-15
- **Set B (4, disjoint):** 2020-02-20, 2020-05-25, 2020-08-15, 2020-11-20

## CG-1 (PRIMARY, the go/no-go) — is the concept graph reproducible?

Feature-level, already published and re-verified from the stored matrices:

| statistic | feature level (41 features) |
|---|---|
| off-diagonal Pearson ρ, A vs B | **+0.181** |
| sign agreement | **0.513** (chance) |
| top-20 edges shared | **6/20** |

Bars, frozen:
- **REPRODUCIBLE** iff ρ ≥ 0.50 **and** sign agreement ≥ 0.70 **and** top-10 shared ≥ 6/10.
- **NOT REPRODUCIBLE** iff ρ < 0.30. Then the program stops here: no static concept graph
  exists, the 10-GPU-hour observational extraction is **not** licensed, and CG-4 is not scored.
- Anything between is **PARTIAL** — reported, and CG-4 scored descriptively only.

Top-10 not top-20, because there are only 90 off-diagonal cells at 10 concepts vs 1,640 at 41
features; top-20 of 90 would be a weak test.

## CG-2 (calibration, guardrail #9) — both sides, before CG-1 is read

- **POS:** each concept's self-effect (diagonal, computed but excluded from A) must be the
  largest entry in its own row. If dosing a concept does not move that concept most, the
  readout is broken and there is no verdict.
- **NEG:** `concept_perm` — the SAME 150 features, randomly re-partitioned into 10 groups of
  15 (seed 0). Its ρ between IC sets must be **< 0.30**. If a scrambled partition reproduces as
  well as the real one, CG-1 measures dose geometry, not concept identity, and is VOID.

## CG-3 — magnitude confound

Report Spearman(|A[i,j]|, dose90 mass of concept *i*). If edge strength is explained by how
much total activation the source injects, the graph is a magnitude readout. Reported alongside
every CG-1 number, never omitted.

## CG-4 — THE ANSWER KEY (frozen battery, 11 relations)

Scored only if CG-1 is REPRODUCIBLE or PARTIAL. Sign and timescale from textbook atmospheric
dynamics, written before any edge was seen.

| # | relation | sign | timescale |
|---|---|---|---|
| R1 | ascent → div250 | + | 0–6 h | **STRUCK — div250 has no features** |
| R2 | t850 → z500 (hydrostatic thickness) | + | instantaneous |
| R3 | jet250 ↔ baroclinicity (thermal wind) | + | instantaneous |
| R4 | baroclinicity → vort850 (baroclinic growth) | + | 1–2 d |
| R5 | q600 → ascent (latent heating) | + | 6–12 h |
| R6 | ascent → q600 (moisture convergence) | + | 6–12 h |
| R7 | shear → ascent (shear suppresses organized convection) | **−** | 1–2 d |
| R8 | blocking → jet250 (blocking diverts the jet) | **−** | days |
| R9 | q600 → atm_river | + | days |
| R10 | atm_river → ascent | + | 6–12 h |
| R11 | baroclinicity → z500 | + | 1–2 d |

Primary score: **signed-edge accuracy** = fraction of the 10 testable relations whose recovered
`A[i,j]` has the predicted sign, against a 0.50 chance line, binomial.
- **CONFIRMED** ≥ 8/10 (p ≤ 0.055)
- **NULL** ≤ 6/10
- 7/10 is INCONCLUSIVE

Secondary, reported but not part of the primary: the two **negative** relations R7 and R8 are
the strongest single tests, because spatial co-occurrence of concepts is positive by
construction and cannot produce a sign flip. If the overall score is CONFIRMED but R7 and R8
both come out positive, the result is downgraded to "co-occurrence recovered, causation not
demonstrated" — declared here, not after.

## CG-5 — the co-occurrence confound, stated as the main threat

Atlas labels are *spatial co-occurrence*: a feature is `ascent` because it fires where ascent is
anomalously strong. Physically-correlated fields (q600 and ascent genuinely co-vary) therefore
produce overlapping feature sets, and an edge between them may recover the labelling rather
than a causal relation.

Three defences, all frozen:
1. `concept_perm` (CG-2 NEG) breaks label identity while preserving dose geometry.
2. **Asymmetry**: co-occurrence is symmetric. Report `|A[i,j] - A[j,i]| / (|A[i,j]| + |A[j,i]|)`
   for every scored pair. R5/R6 are a directed pair on the same concepts and exist precisely to
   probe this.
3. The two negative relations (R7, R8), which co-occurrence cannot generate.

Additionally report **feature-set overlap** (Jaccard) for every scored concept pair. Any pair
with Jaccard > 0.1 is flagged and its relation reported separately — by construction, K=15
argmax-disjoint groups give Jaccard 0, so this is expected to be a formality and is recorded so
that the expectation is on file.

## CG-6 — what is NOT claimed

- No propagation, speed, or `V_MAX` reading. Concepts have no locations; that machinery is not
  applied and its absence is not a miss.
- Timescale column is **not** scored at one step. The run is a single 6-h step, so it can only
  test sign. Lag ordering needs a multi-step version and is deferred; the column is frozen here
  so it cannot be fitted later.
- Flagship-internal. Not comparable to the mini pool, the mini SAE, or SAVAR.
- Nothing here licenses the 10-GPU-hour observational extraction unless CG-1 is REPRODUCIBLE.
