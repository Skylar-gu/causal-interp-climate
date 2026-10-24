*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Control audit — what the flagship claim's control was actually doing

Written 2026-08-18 while the GPU queue ran, by auditing existing results rather than by
running anything new. Two of the project's supported claims turn out to rest on controls that
were never calibrated the way guardrail #9 requires.

## The convection result itself is intact

`results/skill/convection/verdict.json`, seven developing storms, ablation vs baseline:

| arm | median | range |
|---|---|---|
| conv-zero (full ablation) | **41%** of baseline deepening lost | 16–65% |
| conv-normal (restore-to-normal) | **19%** | 4–41% |
| rand-normal (matched control) | **0%** | −1 to 3% |

Paired Wilcoxon, convection vs random: **p = 0.0156, 7/7 storms**. That is the smallest
p attainable at n = 7, so the comparison is as strong as this sample allows.

## But the control was near-vacuous inside the storm boxes

`RANDOM_CTRL` is three features matched to the convection group on **global firing rate**,
filtered to a tropical centroid so they "can fire on the storm". Baseline in-box activation
of the random group as a share of the convection group's:

| storm | share |
|---|---|
| michael 2018 | **0%** |
| wilma 2005 | **0%** |
| patricia 2015 | **0%** |
| ida 2021 | 3% |
| haishen 2020 | 11% |
| haiyan 2013 | 19% |
| goni 2020 | 36% |

Median **2%**. In three of seven storms the control ablates *nothing inside the box*, and the
arm's own source comment states the rule this breaks: *"a control that ablates nothing cannot
fail."*

**What this does and does not damage.** The comparison against **baseline** is untouched —
convection ablation costs 4–41% of deepening and that is measured against the unpatched
model. What is weakened is the **specificity** claim: that the effect belongs to the
convection features rather than to ablating any three features. That claim leans on a control
that was largely a no-op, and it is only fully supported in the three storms where the
control genuinely fired (goni 36%, patricia 24%, haiyan 19%) — where convection still cost
17% / 4% / 33% against random 3% / 0% / −1%. Three storms is underpowered.

The mechanism of the failure is that matching was done on **global** firing rate. A feature
with the right global rate has no reason to fire in one particular storm's box, and the
|clat| < 25 centroid filter cannot fix that — SAE features have a median footprint spread of
7,851 km, so a centroid says little about where a feature actually fires.

## The repair

`src/inbox_control.py` redraws the control **per storm**, matched on in-box firing
over that storm's own region, computed from the IID activation dump so it needs no GPU.

| battery | old control (global-rate matched) | new control (in-box matched) |
|---|---|---|
| NH, 8 storms | median **2%** (0–54%) | median **95%** (87–112%) |
| SH, 5 storms | median 14% (9–24%) | median **97%** (93–111%) |

`skill_conv_run.py` takes these via `MECH_INBOX_CTL`; absent the variable nothing changes, so
no committed number moves. The re-run writes to `MECH_NAME=convection_inboxctl` rather than
over `results/skill/convection`.

## The second finding: PS-5 had no control at all

`flagship_sae/ps5_shear_displacement.py` was written this session because the PS-5 numbers
existed only in a note with no reproducible code. It reproduces them — P1 mean angle +40°,
R = 0.38, 5/7 left; P2 displacement vs |S| **r = +0.830, p = 0.021** — and then reports the
result as **UNCONTROLLED**, because the stored snapshots hold `node_conv` and never
`node_rand`. Guardrail #9's negative control was never computable.

By this repo's own rule P2 is therefore **not reportable as it stands**, and the scorer now
prints that in place of the correlation. `skill_conv_run.py` now stores `node_rand` from the
same unpatched baseline forward pass, so the re-run supplies it at no extra cost — and
supplies it with the *repaired* control rather than the near-vacuous one.

## Why this was worth doing before reporting

Five results in this repo have already been voided on their controls: the BSF block
threshold, the SPD alignment null, the flagship `frac_eastward` anchor gate, the concept
graph, and PX_geo. Every one of them died to a control that could not fail, found late. Both
findings here are the same failure mode caught before publication rather than after.

Neither is a void. The convection result's baseline comparison stands as measured; PS-5's
correlation is real arithmetic on real runs. What is missing in both cases is the leg that
shows the statistic could have come out the other way — and both are now queued.

## The rest of the battery, audited the same way

### Heat dome dose sweep — adequate, though not by design

`topk_ablate.npy` carries arms top5..top160 and **no size-matched random arm at any k**. On
its face that is the same defect: ablating more features costs more regardless of which. Two
things rescue it.

| arm | n | mean d_ridge |
|---|---|---|
| top5 | 5 | −6.6 |
| top10 | 10 | −9.8 |
| top20 | 20 | −36.0 |
| top40 | 40 | −52.0 |
| top80 | 80 | −69.4 |
| top160 | 160 | −69.8 |
| **random** | **177** | **−0.7** |
| full_physics | 522 | −41.6 |

The one random control that exists sits at n = 177, close enough to top160's 160 to compare
directly — and it costs **106x less**. And `full_physics` at n = 522 costs *less* than top160
at n = 160. Feature count is therefore not what drives the curve, which is the thing a
size-matched control would have been testing. The gap at k = 5..80 is real but the conclusion
does not hinge on it.

### Response operators — fully calibrated, and a clean negative

`results/respop_score.txt` reports **DETECTED 0/10** with all three legs explicitly checked:
*floor failable True; negative control passes True; specificity null varies True.* Nothing to
repair. A clean negative reported as one.

Worth noting for the labelling work: the spatial-structure table puts 0.396 of t850's
response, 0.308 of z500's and 0.296 of jet250's in the −90..−60 band. That is the same polar
concentration the global-standardisation bug produced in the labels — though the permutation
mean also sits at 0.238 there, so part of it is generic rather than concept-specific.

### Flagship watching-graph — already voided, correctly

`flag_gint_report.json` records `anchor_gate: INSTRUMENT FAILURE`. The `qrand_flag` anchor
reaches geo_p = 0.028 against `leiden_flag`'s 0.033 — the scrambled control is *more*
significant than the thing it controls. Recorded as a failure rather than repaired after the
fact, which is why the surviving results are worth anything.

## Audit summary

| claim | control status |
|---|---|
| Convection necessity, vs baseline | intact as measured |
| Convection **specificity**, vs random | **control fires at 2% in-box; repair queued** |
| PS-5 P1/P2 | **no control existed; repair queued** |
| Concept groups on global RMSE | two confounds checked, both dead |
| Heat dome dose sweep | no per-k control, but 106x separation at matched size |
| Heat dome physics groups | two controls, both near zero |
| Response operators 0/10 | three calibration legs, all pass |
| Flagship watching-graph | already voided on its own anchor gate |
| GraphCast vs HRES | direct comparison, no control applicable |
