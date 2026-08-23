*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# "Normal" is a real quantity on the wrong axis

2026-08-21. Answers a direct challenge to the clamp target: is the quiet-day "normal" level
meaningful, and what does convection actually look like during the storms? Measured over
each storm's most intense 24 h. No new rollouts: `results/skill/fields_conv/run_*.npy`
already stores per-step node-level codes, `nodefeat` (16 steps, 839 in-box nodes,
[TC, 2401, 2067, 3174]), plus the storm centre at every step.

## What "normal" is

`flagship_sae/skill_conv_run.py:276-294`. For each convection feature, take quiet analogue
days at the same calendar date in other years, reject any on which the TC feature sums above
20 inside the box, and average the feature's code **over nodes where it is already firing**.
If it never fires on those days, normal is 0.

## Measured against the storms

Peak-24 h window = the four consecutive 6 h steps with the largest MSLP drop. Disk is
storm-following, 1500 km, recentred each step on the tracked centre.

| storm | peak 24 h | drop (hPa) | firing (node,feature) pairs | storm level | normal | ratio | clamp removes |
|---|---|---|---|---|---|---|---|
| Ida 2021 | +54–78 h | 4.6 | 12.8 | 2.21 | 1.13 | 1.95x | 65.6% |
| Michael 2018 | +6–30 h | 6.4 | 21.8 | 2.74 | 2.26 | 1.21x | 37.2% |
| Haishen 2020 | +0–24 h | 14.9 | 38.8 | 3.04 | 2.13 | 1.43x | 32.8% |
| Goni 2020 | +6–30 h | 4.2 | 14.0 | 2.61 | 1.12 | 2.33x | 53.7% |
| Haiyan 2013 | +0–24 h | 8.1 | 32.8 | 2.45 | 1.37 | 1.79x | 51.7% |
| Patricia 2015 | +42–66 h | 3.0 | 11.8 | 1.86 | 2.10 | **0.89x** | 17.8% |
| Wilma 2005 | +54–78 h | 10.0 | 35.0 | 2.50 | 2.07 | 1.21x | 27.6% |
| **median** | | | | **2.50** | **2.07** | **1.21x** | **37.2%** |

## Five things this settles

1. **A rapidly intensifying hurricane fires these features only 1.2x harder than a quiet
   day**, median over the seven storms. On Patricia the storm level is *below* its own
   normal, 0.89x.

2. **The clamp is therefore weak, and unequally weak.** Restoring to normal removes a median
   37% of the group's in-disk activation over the intense window, ranging 17.8% to 65.6%.
   The same nominal treatment differs by 3.7x in how much it actually takes out. That is
   uncontrolled heterogeneity inside what the paper reports as one arm, and it is the honest
   explanation for why restore and delete differ by only 1.49x (+2.79 vs +4.15 hPa).

3. **Normal rests on very few days.** Ida used **one** surviving analogue day, Goni two;
   the others three to five.

   Ida shows what that buys. Its three features carry 63.2 / 10.6 / 37.8 of activation mass
   over the intense window, and the clamp removes 63.2 / 0.2 / 9.9. So **86% of everything
   the clamp takes out of Ida is feature 2401 alone, deleted outright because its normal is
   0.00** — and it is 0.00 because the single surviving analogue day did not have 2401
   firing. On the two features whose target is a real number, Ida's storm fires at 0.81x and
   1.30x normal, i.e. at normal. The 1.95x group ratio in the table is an artifact of
   averaging a zero into the denominator. Ida is the paper's showcase storm and its 41%
   figure rests mostly on one feature with a degenerate target.

4. **For feature 2401 normal is exactly 0 on Ida, Goni and Haiyan**, so on three of seven
   storms "restore to normal" *is* deletion for that feature. The distinction the paper
   draws between its restore and delete arms does not hold uniformly across the battery.

5. **Clamp strength does not predict effect.** Spearman(removed %, effect) = **+0.21,
   p = 0.64**. Haishen has the second-weakest clamp (32.8%) and the largest effect
   (7.47 hPa); Ida has the strongest clamp (65.6%) and 2.79.


## Ida's normal, recomputed from 11 and 28 quiet days

2026-08-21, same day. The 160-window IID dump carries layer-8 activations for 12 days in
Aug 10 - Sep 10 and 30 in Jul 20 - Sep 30, so the estimate can be redone with the identical
procedure (same box, same 1500 km disk, same TC<=20 screen, same mean-over-firing-nodes) and
no GPU.

| feature | old (1 analogue) | 11 quiet days | 28 quiet days |
|---|---|---|---|
| 2401 | **0.00** | **2.80** | **2.36** |
| 2067 | 1.87 | 2.03 | 2.20 |
| 3174 | 1.53 | 1.55 | 1.73 |
| **clamp removes, over Ida's intense 24 h** | **65.6%** | **12.2%** | **14.4%** |

**The committed Ida arm clamps about five times harder than "restore to normal" should.**
The whole difference is feature 2401, whose target was 0.00 only because the single surviving
analogue day happened not to have it firing anywhere in the disk. Given a real estimate it is
2.4-2.8, close to Ida's own storm level, and the clamp on it becomes small.

Quiet days are not scarce: **11 of 12** candidate days pass the storm screen in this box.
The original found only 1 of 5 because it required the same calendar date, 27 August, and
that date carried a storm in the box in four of the five chosen years. The failure is the
same-date constraint, not the availability of quiet analogues.

**Consequence.** Ida's 41% is the paper's headline number and its showcase figure. It was
produced by an intervention roughly five times stronger than the operator is described as
being. Whether an effect survives at the corrected target is a GPU question and is not
answered here; the arm must be rerun before that figure is quoted again. The same check is
owed to Goni and Haiyan, whose 2401 target is also 0.00.

## Why normal is the wrong axis

What separates a storm from a quiet day in this dictionary is **how many nodes fire, not how
hard they fire**. Code magnitude when firing is tightly bounded on every storm:

| storm | firing nodes / step | code when firing (mean, p10–p90) |
|---|---|---|
| Ida | 17.3 | 2.40, [1.41, 3.72] |
| Michael | 17.3 | 2.67, [1.52, 4.02] |
| Haishen | 31.4 | 2.87, [1.72, 4.70] |
| Goni | 26.4 | 2.56, [1.52, 3.77] |

That narrow range is what TopK with k = 32 produces: magnitudes are set by competition
within a node, so they cannot vary much. The storm signal lives in spatial extent, 12 to 39
firing nodes inside a 547-node disk.

A cap on magnitude can only ever shave the thin top slice off an already narrow
distribution. It cannot touch the extent that actually encodes the storm. So the answer to
"is normal meaningful" is: it is a well-defined quantity, but the intervention manipulates
the axis along which storm and quiet day barely differ, and leaves untouched the axis along
which they differ a lot.

**This does not void the intervention result.** The effect is real against its controls and
the matched-exposure argument stands. It reframes what the operator is doing: the reported
19% is what you get from removing roughly a third of the group's activation magnitude, not
from removing the mechanism.

## The obvious next arm

An **extent** intervention: suppress the group at all nodes in the disk where it fires,
rather than capping its magnitude, and compare. That is one line in the patch (set the
selected features to zero wherever they exceed zero inside the mask, instead of capping at
`ftarget`) and one battery, ~30 GPU-minutes. It is the arm that would separate "convection
magnitude is a lever" from "convective coverage is the lever", and the numbers above say
those are very different edits.

Note this is a fourth predictor of effect size that has failed on this battery, after in-box
exposure, calibrated ascent score, and geometric alignment with the dominant local modes
(`exposure_confound_2026_08_20.md`, `result_local_aggregate_2026_08_21.md`).
