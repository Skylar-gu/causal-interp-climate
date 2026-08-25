*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Amendment 1 to `prereg_hybrid_hurricane.md` — 2026-08-20, before any edge is scored

The calibration required by the parent prereg ran first, as specified, and it **killed the
estimator the parent named**. Nothing has been scored. This amendment is frozen before the graph
is built on real data, and it records what the calibration measured rather than what I hoped.

## A1. The ≥50%-of-windows consensus rule is anti-powered, and this is arithmetic

Injecting a known coupling `X_b[t] += beta * X_a[t-1]` into z-scored real in-box activation
series (80 windows, T=16, ParCorr, tau_min=1, tau_max=2, pc_alpha=0.05):

```
 beta   p(window)  pairs p>=0.5  consensus@13  consensus@52  joint@13  joint@52
  0.2      0.069       0/5           0.002         0.000       0.167     0.600
  0.4      0.112       0/5           0.007         0.000       0.867     1.000
  0.6      0.196       0/5           0.022         0.000       1.000     1.000
  0.8      0.265       1/5           0.136         0.200       1.000     1.000
  1.0      0.358       2/5           0.325         0.400       1.000     1.000
  1.5      0.527       3/5           0.564         0.600       1.000     1.000
```

**Judged where the parent prereg put the bar — at beta ≥ 0.8 — the kill condition is met:** mean
per-window recovery is 0.265, only 1 of 5 injected pairs individually reaches p = 0.5, and
consensus detection is 0.136 at 13 windows and 0.200 at 52.

Consensus detection is `P(Bin(n, p) >= n/2)`, so **while p < 0.5 it falls as n grows** — visible
in the cons@13 vs cons@52 columns at every beta below 0.8. In that regime more windows make a
≥50% consensus rule *stricter*, not more powerful: window count is a penalty, not a lever. The
sign flips once p crosses 0.5, which happens at beta ≈ 1.5.

*Correction, recorded rather than quietly fixed:* an earlier draft of this note said per-window
recovery "never reaches 0.5 at any strength tested up to beta = 1.5". It does reach it, at
beta = 1.5 exactly (p = 0.527), and the consensus rule becomes usable there. That does not
rescue the rule for this purpose — beta = 1.5 is a lag-1 coupling delivering 1.5 SD of the
child per step, and **whether any real feature-to-feature coupling in GraphCast is that strong
is precisely what the intervention was built to find out.** A rule that can only detect what it
was built to assume is not an instrument. But the honest statement is "dead at the
pre-registered bar", not "dead everywhere".

Three independent confirmations that this is the rule and not the data:

1. The arithmetic above, which needs no experiment.
2. On the real, uninjected series, `graph --mode consensus` returns **exactly zero edges from 80
   windows**. There would be nothing to intervene on.
3. A white-noise null suggested "0 false consensus edges at 8 windows". On real, autocorrelated
   windows the same rule gives **2 false consensus edges at 8 windows** (0 at 16). White noise
   was the wrong null and flattered the rule in both directions at once.

**This reaches beyond this experiment.** `pcmci/gint_consensus.py` defaults to `CONS_FRAC = 0.5`,
and the ≥50% consensus rule is what produced `leiden_flag`'s edge set and the audit finding that
"11 of 38 edges are season-specific and the ≥50% consensus rule has been discarding regime-gated
physics all along". That observation now has a mechanism: any effect whose per-window detection
probability is under one half is *systematically* removed by the rule, and adding windows makes
the removal more certain. A regime-gated edge, present in a minority of windows by construction,
is exactly the case the rule cannot keep.

## A2. Amendments

**A2.1 — Estimator.** The graph is built with `analysis_mode='multiple'` over all windows
jointly (as the parent prereg's own PCMCI+ paragraph specifies), NOT by per-window fits with a
≥50% consensus. The consensus result is reported alongside as the negative it is.

**A2.2 — `tau_max` = 2, and 52 windows, because the parent's own ratio floor demands it.** The
parent requires the effective-sample ratio to stay above 5. Measured:

| windows | tau_max | ratio | verdict |
|---|---|---|---|
| 13 | 4 | 1.95 | fails the parent's floor |
| 13 | 2 | 4.55 | fails the parent's floor |
| 52 | 2 | 18.2 | passes |

`tau_max = 2` with 52 windows is the only combination consistent with the prereg as written, so
the IC-offsets battery is run rather than skipped. It is also bought by a second measurement:
under a per-column circular-shift null (autocorrelation preserved, cross-lag destroyed) the
null's ceiling on |MCI| is **0.266 at 13 windows** but **0.131 at 52**, and the number of real
edges clearing that ceiling is **median 10, range 7–14 at n=13** against **median 33, range
30–40 at n=52**. At 13 windows the required top-10 exists with zero margin and some draws yield
only 7. Spending 1.5 h of GPU to make the ranking safe is cheap against the 88 downstream
rollout batteries it selects.

Offsets are `-48, -24, +24` only. The 13 unshifted runs already exist, so `+0` is not repeated;
13 existing + 39 new = 52.

**A2.3 — Selection gains a non-constancy clause.** 40 of 80 calibration windows contained a
selected feature that is identically zero across all 16 steps. ParCorr is undefined there and
tigramite warns rather than failing. The parent's gate ("in-box mean > 0 in ≥6 of 8 storms") does
not prevent it, because a feature can fire in six storms and be flat in a seventh window. Added:
**a feature must be non-constant within every window used**, applied before the graph, with the
census printed. Nothing is dropped silently.

**A2.4 — A new gate the parent did not have: the |MCI| null ceiling.** The joint fit's
false-positive rate sits at nominal alpha — about 10 invented directed lagged edges per 220 at
`pc_alpha = 0.05`, at both 13 and 52 windows. So "top 10 edges by |MCI|" is **not** self-evidently
a set of real edges. An edge is only carried into the intervention stage if its |MCI| exceeds the
circular-shift null's maximum over 10 draws. If fewer than 10 edges clear it, fewer than 10 are
tested and the shortfall is reported — the top-k is not topped up from below the ceiling.

**A2.5 — Non-independence of paired controls, declared.** 6 of 10 drawn matched controls share an
endpoint with the edge they referee (e.g. control `f2401→f2067` for edge `f1232→f2401`). This
costs no extra GPU, since one ablation rollout serves both arms, but the paired `asym` values are
then not independent and the B1 Wilcoxon is anticonservative. Reported per pair. The draw rule is
NOT changed after the fact; the dependence is disclosed and B1 is read with it in view.

## A3. What the footprint census did to the design's premise

The parent's central design decision — score an asymmetry, because footprint overlap is symmetric
and cancels — was built on the repo's dictionary-overlap lesson (edge rate 1.2% → 14–17% above
cosine 0.45). Measured on these features, under the repo's own footprint definition:

- f2067 covers 79 of 40962 mesh nodes; 623 features overlap it at all, **12 above cosine 0.1,
  zero above 0.45**. f2401: 9 nodes, 17 above 0.1, zero above 0.45. f3174: 38 nodes, 6 above 0.1,
  zero above 0.45.
- Among the selected features, **78 of 110 ordered pairs have footprint cosine exactly 0**;
  median 0, max 0.300.

**Storm-box features are unusually local** against the repo's recorded median footprint spread of
7,851 km. So the leakage confound the asymmetry statistic was built to defeat is much weaker here
than assumed, the footprint axis of the matched-control draw is near-vacuous, and the binding
axes are marginal |r| and firing amplitude. The asymmetry statistic is retained — it costs
nothing, the reverse arm is still the negative control, and a statistic that survives a confound
that turned out to be small is not thereby wrong — but the claim "the asymmetry defeats
footprint leakage" is **withdrawn as unnecessary here**, and the locality census is reported as a
finding in its own right.

## A4. Incident

An agent on this task ran `pkill -f "hybrid_pcmci.py calibrate"`; the pattern self-matched and
killed its own shell chain. Guardrail #7, fifth recorded occurrence, first one to cost nothing
(no GPU job was live, no files harmed). The rule stands: never match a process by a string that
can appear in any live command line.

## A5. A guardrail-#9 failure inside the calibration's own verdict logic

`hybrid_calibrate_windows.py` originally computed its KILL verdict from the **maximum**
per-window recovery over the whole beta sweep. Because a strong enough injection always exists,
that verdict is **unfailable** — beta = 1.5 rescued the rule and the tool printed `KILL: NO`,
the opposite of the pre-registered condition. It is the same defect this repo has killed five
results for, appearing this time inside the instrument that was checking for it.

Fixed: the verdict is read off the row at `--kill-beta` (default 0.8, the pre-registered value),
it warns when that beta was not tested, and it reports the beta at which p first reaches 0.5
separately, as context rather than as the verdict. Re-derived from the saved
`results/hybrid_calibrate_windows.json` with the corrected logic, no re-run needed:

```
at beta=0.8, n=13: consensus P(detect)=0.136  joint P(detect)=1.000
at beta=0.8, n=52: consensus P(detect)=0.200  joint P(detect)=1.000
at beta=0.8: mean per-window recovery p=0.265; 1/5 injected pairs individually reach p>=0.5
mean per-window recovery first reaches 0.5 at beta=1.5
KILL for the >=50%-consensus specification, judged at beta=0.8: YES
```

The lesson generalises past this script: **a verdict taken as a max over a sweep is not a bar.**
It is the sweep's ceiling, and the ceiling always passes.

## A6. OPEN CONFLICT between A2.3 and the parent's positive control — flagged, not resolved

Data-gating the 8 NH baseline series (clean: (16, 4096) each, finite, 870–2078 features firing
per box) shows the non-constancy clause A2.3 collides with the parent prereg's B5.

- **294 features** are non-constant in all 8 storm boxes. 586 fire in ≥7 of 8, 791 in ≥6 of 8.
- Of the convection triplet the parent force-includes as the positive control, **only 2067 is
  non-constant in every storm.** 2401 and 3174 are flat in at least one box.
- 3243 (the TC readout, excluded by the parent anyway) is also not non-constant everywhere.

The two clauses cannot both hold: A2.3 bars a feature that is constant in any window used, and
B5 requires the convection triplet to be present. Note the conflict **worsens** at 52 windows,
because "non-constant in every window" is a stricter requirement the more windows there are —
the same direction-of-n trap that killed the consensus rule.

Recorded now, before the graph is built, and deliberately NOT resolved by quietly relaxing
whichever clause is inconvenient. The resolution and its cost will be written as amendment 2
once the 52-window series exists and the count can be measured rather than guessed. The
candidate resolutions, stated in advance so the choice cannot be made to fit the result:

1. Keep A2.3 and let the positive control shrink to 2067 alone. Cost: B5 becomes a
   single-feature test and is correspondingly weaker; this must be stated wherever B5 is cited.
2. Keep the triplet and restrict the window set to those where all three are non-constant.
   Cost: fewer windows, which the |MCI| null ceiling measurement says is the expensive direction.
3. Select nodes from the 294 and treat the convection triplet as an intervention TARGET rather
   than a graph node. Cost: the graph no longer contains the one group with a known
   interventional effect, so B5 cannot be scored on the graph at all.

None of these is free, and which one is taken changes what the positive control means.

### A6.1 RESOLUTION — chosen 2026-08-20 06:15 UTC, before any graph exists on real data

**Resolution 1 is taken: A2.3 stands, and the positive control shrinks to feature 2067 alone.**

Decided on the criteria written in A6, not on any result — no graph has been built on the real
series, and the 52-window set does not exist yet. Recorded now so the choice cannot later be
attributed to whichever option flattered the outcome.

Why 1 and not the others, in the terms already stated:

- **Resolution 2 (restrict windows to where all three fire) is rejected** because it pays in the
  currency the calibration identified as expensive. The circular-shift null's |MCI| ceiling is
  0.266 at 13 windows against 0.131 at 52; windows are what buys a defensible ranking, and
  spending them to keep two extra nodes inverts the trade the offsets battery was queued to make.
- **Resolution 3 (convection as intervention target only) is rejected** because it deletes the
  positive control from the graph entirely. B5 exists to show the instrument has power — that a
  group with a *known* interventional effect registers as one. An instrument with no power check
  is how Job B at N=40 was reported before it was withdrawn.
- **Resolution 1 keeps the power check at zero cost in windows.** 2067 is a convection feature
  with the same known interventional effect as the group; what is lost is redundancy, not the
  logic of the test.

**The cost, to be repeated wherever B5 is cited:** B5 becomes a single-feature test. It can
therefore fail for a reason that has nothing to do with the design — 2067 alone may simply be
too weak a node — and a B5 failure must be read as "instrument underpowered", exactly as the
parent prereg already requires, rather than as evidence against the hybrid claim.

Node counts measured under this resolution (a data gate, not a result): **195 features are
non-constant across all 13 current windows** (294 across the 8 NH storms alone). All 13 files
pass the shape, finiteness and (16, 4096) gates with zero bad files. The count at 52 windows will
be lower and will be reported before the graph, not after.

## A3.1 CORRECTION — A3's footprint census does not transfer to the real node set

A3 withdrew the asymmetry statistic's justification on the grounds that footprint overlap is
nearly absent — "78 of 110 ordered pairs have footprint cosine exactly 0, median 0". **That
census was computed on the 11-feature mechanism-library fixture, not on the features the real
selection actually chose, and it does not hold for them.**

Measured on the 20 nodes selected from the 52-window battery, same repo footprint definition:

```
footprint cosine over all ordered pairs:  median 0.237   mean 0.249
                                          fraction exactly 0:  0.000
                                          fraction > 0.45:     0.047   max 0.500
footprint sizes: 43-59 mesh nodes, tightly clustered
```

Not one pair has zero overlap. The fixture features were an unrepresentative sample — a mix of
tiny, scattered concept features — while the real selection is the top of the in-box firing
ranking, and those features are similar in size and substantially overlapping.

**So A3's withdrawal is itself withdrawn. The asymmetry statistic's original justification
stands.** Footprint leakage is a live confound on this node set, exactly as the repo's
dictionary-overlap lesson predicts, and the statistic built to cancel it is doing the job it was
designed for. The top-k competition finding in `notes/p0_topk_competition_2026_08_20.md` is an
additional, independent reason for differencing — not a replacement for this one.

### And the edges sit preferentially where the overlap is

The ten top-|MCI| edges have footprint cosines 0.406, 0.406, 0.534, 0.608, 0.621, 0.653, 0.703,
0.724, 0.738, 0.746 — **mean 0.614 against an all-pairs median of 0.237**, about 2.6x the typical
overlap between two selected nodes.

That is the repo's documented signature reproducing on a new dataset: edge rate rising from 1.2%
at zero footprint overlap to 14–17% above cosine 0.45, read as redundant co-encoding of one
evolving object rather than as causation. **It is also precisely why this experiment exists.** An
observational graph whose edges concentrate on overlapping footprints is exactly the graph whose
edges need to be tested by intervention rather than believed — and the matched non-edge controls
are matched *on that cosine*, so the comparison holds it fixed instead of arguing about it.

Lesson recorded for the method, not just this run: **a census computed on a convenience fixture
is not a census.** The fixture was chosen because it was on disk, and it gave the opposite answer
to the real data on the one quantity that a central design decision rested on.

## A7. CORRECTION — the parity claim was overstated in my summaries and in commit 18cdec5

That commit's title reads *"The parity product predicts which lag an estimator picks, at rho
0.72-0.91"*, and I repeated it in that form several times. **The second half of that sentence is
not established.**

What IS established, and holds exactly: the parity product `P_a * P_b` predicts the **shape of
the lagged cross-correlation function** — the CCF's own Nyquist component `ALT` — at Spearman
0.722 / 0.720 / 0.905 over the three node sets. That is a statement about the raw correlation
structure, computed with no estimator involved.

What is NOT established is that the estimator's **selected edge lag** follows it. Concordance of
the chosen tau with sign(ALT) is 45/83 (Fisher p = 0.51), 31/54 (p = 0.28) and 27/40 (p = 0.028).
**Only `local_physics` is significant, and it is the most parity-contaminated of the three sets.**
PCMCI+ conditions on other nodes, so the forcing is not deterministic edge by edge.

The parity agent reported this correctly and I compressed it wrongly when summarising. The
defensible claim is:

> On 6-hourly SAE series the sampling grid, not the dynamics, sets the shape of the lagged
> dependence between features. Whether that propagates all the way to which lag a
> conditional-independence estimator selects for a given edge is supported in one of three node
> sets and not in the other two.

That is still enough to void a lag-mechanism claim — a lag read off a CCF whose shape is set by
the grid cannot be evidence of physical timing — but it is weaker than "the estimator picks the
lag parity dictates", and the difference matters for anyone citing it.

Two smaller corrections from the same review:

- **The footprint census figures I quoted were unsourced.** Measured properly
  (`flagship_sae/footprint_census.py`, `results/footprint_census.json`): the locality gate
  (≤200 mesh nodes) admits **68.2%** of the dictionary, compactness (<3,000 km spread) admits
  **4.4%**, and among the ≤200-node features the compact fraction is **4.3%** — *no enrichment
  whatsoever*. Spearman(size, spread) is **0.024 (p=0.14)** under the footprint-mask definition
  and 0.333 under a permissive nonzero-activation definition; the 0.324 I quoted was the latter.
  The corrected version is a **stronger** statement of the same point: selecting on footprint
  size buys literally nothing in compactness.
- **The "+24-42% ceiling" band** is the `hurricane_latent` trio only. The 52-window hybrid set is
  0.190 → 0.298, i.e. **+57%**, and should be quoted separately.
