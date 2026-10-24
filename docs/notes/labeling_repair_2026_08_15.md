*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Mechanism labelling: the repair, and the committed claim it invalidates

2026-08-15. Scripts: `src/label_mechanisms_v2.py` (rotation null),
`src/label_rescore.py` (inference). Result:
`results/fs_mechanisms_v2.npy`. No GPU; ERA5 reads + the existing i.i.d. dump.

## Why

`label_mechanisms.py:100` assigns a feature to a mechanism by `argmax` over
probes with a fixed bar `z > 0.15`, uncalibrated. `z[f,m]` is a MEAN of
per-window standardized field values over the nodes where `f` fires, so its noise
scale is set by the footprint: naive SE = 1/sqrt(n_fire) runs 0.048 at the 5th
percentile of footprint size to 0.004 at the 95th. One fixed bar is therefore an
11x different statistical bar across the dictionary, and it is LOOSEST exactly
where it should be strictest — compact features, which are the good causal
handles. Measured: the labelled fraction rises 78.5% -> 97.9% as the battery grows
4 -> 17 probes, i.e. the bar goes vacuous as more phenomena are added. Same family
as the blocks-v1 vacuous threshold and the SPD point-mass null.

Second defect: `argmax` forces a winner, and the probes are not orthogonal —
q600 vs ascent correlate at r = +0.39 over features, shear vs q600 at -0.32. The
convection-vs-moisture contrast therefore separated two groups that were disjoint
only BECAUSE argmax made them disjoint.

## The instrument

1. **Rotation null.** 127 rigid longitude rotations of the mesh sampling. Footprint
   size, shape and latitude band are preserved exactly and the field keeps its own
   spatial autocorrelation; only zonal alignment is destroyed. Nearest-neighbour
   snap is a median 44 km against ~198 km mesh spacing. Same
   footprint/distance-preserving idea as M1's locality control, the
   distance-preserving PX null and the coverage-matched heat-dome control.
2. **Empirical p**, `(1 + #{|null| >= |obs|}) / (1 + R)`. See the withdrawal below.
3. **BH-FDR** at q = 0.05 over live features x probes.
4. **argmax -> winner must beat the runner-up by more than the null's own 95th
   percentile gap**, else `ambiguous`.
5. **Calibration gate**: the same rule on a held-out rotation draw must collapse
   the labelled rate to about the FDR level.

### WITHDRAWN mid-run: the Gaussian step, killed by its own gate

v2 standardized the null and took Gaussian tail probabilities so p-resolution
would not be capped at 1/R. Its calibration gate rejected it — **32.1% of features
labelled on a pure-null input against a nominal 5%**. The null-shape check printed
the reason in the same run: excess kurtosis p95 **+11.6**, heavy tails, so a
Gaussian tail understates the null probability and manufactures significance.
Replaced by the empirical permutation p.

**Stated limit of the replacement.** The p floor is 1/(1+R) = 0.0078, so no
individual feature resolves below it, and BH needs a rejection set of >= 2,164
before it can reject anything at q = 0.05. The procedure identifies the SET at the
stated FDR; it does not rank within it. Related instability, reported not buried:
across 8 held-out draws the null-input rate is mean 2.7% but ranges **0.0% –
21.6%**, because BH against a p floor is close to all-or-nothing. Conclusions
below rest on gaps of 7–23 sigma against bars of 1.0–1.4 and are far from that
edge.

| | v1 fixed 0.15 | v2 gaussian | v3 empirical |
|---|---|---|---|
| features labelled | 96.9% | 41.0% | **30.0%** |
| on a pure-null input | never tested | 32.1% | **2.7%** (target ≤ 5%) |

Labels among 3,461 live features: ascent 434 (12.5%), vort850 391 (11.3%),
q600 162 (4.7%), shear 50 (1.4%), **ambiguous 2,424 (70.0%)**.

## What survives, and what does not

**Convection arm — CONFIRMED, with the largest margins in the table.**

| feat | vort850 | q600 | ascent | shear | gap | bar |
|---|---|---|---|---|---|---|
| 2401 | +1.5 | +7.9 | **+28.5** | +2.9 | 20.6 | 1.3 |
| 2067 | −0.2 | +7.4 | **+29.1** | −1.9 | 21.7 | 1.4 |
| 3174 | +2.1 | +5.2 | **+21.7** | −1.9 | 16.4 | 1.4 |

The vorticity group is likewise 3/3. **The −41% genesis result's feature selection
survives calibration cleanly.**

**Moisture control arm — the claim it supports is NOT SUPPORTED.**
`run_moisture.sh` used `MECH_FEATS=4006,3501,2900`. All three calibrate as
**ascent**:

| feat | vort850 | q600 | ascent | shear | calibrated | gap |
|---|---|---|---|---|---|---|
| 4006 | +3.6 | +3.4 | **+7.2** | −0.1 | ascent | 3.7 |
| 3501 | −1.4 | +4.4 | **+13.6** | −2.3 | ascent | 9.2 |
| 2900 | +2.6 | +4.7 | **+13.2** | −2.1 | ascent | 8.5 |

So `c752ebb`'s "convection is the only lever — a firing-rate-matched moisture group
costs at most 8%" did not compare convection against moisture. It compared
**strongly ascent-loaded features (21–29 sigma) against weakly ascent-loaded ones
(7–14 sigma)**, and the weak ones did little.

- The claim "convection is the ONLY lever" is **withdrawn**. Moisture was never
  tested and cannot be excluded.
- What the data does support, and is arguably the better finding: **ablation
  effect scales with ascent-loading strength** — a dose-response along the
  labelling axis, which nobody set out to measure.
- Correctly done in that run and worth keeping: it dropped feature 3174 for
  overlapping the convection group, and the disjointness assertion in
  `skill_conv_run.py:40` fired as designed. The defect is upstream, in what the
  q600 group WAS, not in the run's hygiene.

**Shear "built-in control" — not a shear control.** Of `pick['shear']`, two are
`ambiguous` and 1996 labels shear only marginally (gap 1.3 vs bar 1.2) while
firing 254,703 times against a 40,962-node mesh, i.e. near-ubiquitous — ablating
it is not a mechanism intervention. The sentence "shear features fire MORE on Ida
but don't drive it" cannot be made. It remains a valid null ablation, just not a
labelled one.

**Corroboration of an earlier correction.** The old GEOGRAPHIC Ida cast (the −13%
result) comes out entirely `ambiguous` with negative z — independent confirmation
that geographic selection was the wrong rule, which the mechanism pivot already
concluded on other grounds.

## The corrected moisture group

131 calibrated q600 winners sit inside the convection arm's firing band and are
disjoint from it. Suggested re-run, same protocol, same bars:

    MECH_NAME=moisture2 MECH_FEATS=2958,2671,37

| feat | n_fire | vort850 | q600 | ascent | shear | gap | bar |
|---|---|---|---|---|---|---|---|
| 2958 | 3,797 | +0.9 | **+6.7** | +2.1 | +0.7 | 4.6 | 1.4 |
| 2671 | 2,331 | −2.8 | **+8.1** | +3.8 | +1.6 | 4.3 | 1.6 |
| 37 | 1,622 | −0.5 | **+4.8** | +0.5 | −0.8 | 4.0 | 1.7 |

## SETTLED 2026-08-17: the corrected arm ran, and the withdrawal is itself withdrawn

`MECH_FEATS=2958,2671,37` gives **−0.032 hPa** median Δ-deepening over the 7
developing storms — zero, and weaker than every other arm in the library. So
moisture is no longer untested: genuine calibrated q600 features do nothing, and
"ascent is the lever, mid-level moisture is not" is **restored on evidence that
actually bears on it**. The withdrawal above stands as a record of what the
original arm could and could not support; the conclusion it withdrew is back.

The load-bearing null is not this arm, though — `moisture2`'s in-box exposure is
only 2.14 against convection's 39.55. It is `mech_q600` (exposure 8.93 on all 7
storms, +0.222 hPa), which is null *despite* three of its four members calibrating
as ascent, i.e. despite bias toward finding an effect. Full table, the
`mech_atm_river` void, and the six arms that had no exposure at all:
`notes/mechanism_library_2026_08_17.md`.

The dose-response this note proposed as "the better finding" also holds, and now
across five arms: effect size is monotone in calibrated ascent-loading, and so is
whether a group fires on a hurricane at all.

## Standing limit on every p above

The rotation null tests **zonal** alignment. A feature whose footprint and field
are both organised by latitude alone will read p ~ 1 even when the association is
real, because the rotated null reproduces it. Raw z is therefore reported next to
p — the free null over-credits, the constrained null is fair, and both belong in
the table. Same lesson as M1.

## Guardrail earned here

> A labelling rule is an instrument. Before a battery of probes is expanded, run
> the rule on a null input and show the labelled fraction collapses. A rule whose
> labelled fraction RISES with the number of probes is selecting, not measuring.
