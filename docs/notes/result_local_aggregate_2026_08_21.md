*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# The convection edit is not the dominant local mode — and the storm box does not have one

2026-08-21. Runs the test pre-registered in `notes/prereg_local_aggregate.md`, which was
frozen before any number below was read. No new rollouts: 160 IID windows already on disk
(`fs_iid_dump.npy`, 2016-2020, 40,962 nodes x 512 dims, hook step 9), ~9 minutes of CPU.

## The surrogate is real storms

The TC feature 3243 was used to locate a tropical box in each window, sized like the real
battery's (23 deg lat x 40 deg lon, ~800 mesh nodes). 146 of 160 windows contain one. The
top boxes are named cyclones, which is the validation that the surrogate is not synthetic:

| window | date | centre | storm |
|---|---|---|---|
| 149 | 2020-09-01 | 28.8N 125.5E | Maysak |
| 120 | 2019-10-08 | 17.0N 142.6E | Hagibis |
| 83 | 2018-10-27 | 18.8N 131.2E | Yutu |
| 117 | 2019-09-29 | 26.2N 46.0W | Lorenzo |
| 63 | 2018-02-14 | 22.0S 175.9E | Gita |

## The premise of the question is weaker than it looks

Before measuring any alignment: inside a storm box the layer-8 activations are **not**
dominated by one mode. Over 60 boxes,

- PC1 holds **21.5%** of the local variance [16.8, 27.6]
- top 5 PCs hold **53.6%** [47.6, 58.3]
- participation ratio **12.0** [9.4, 16.2]
- **59 PCs** are needed for 90% of the variance [48, 73]

An effective dimension of ~12 is not a SAVAR mode. There is no single aggregate for the
convection label to be riding on, which limits how much of the local-aggregate story could
have been true in the first place.

## Preregistered verdict: NEITHER

`align1` = fraction of the group's delete-displacement energy along PC1. Medians over 60
boxes, 10-90 across boxes in brackets; null is 200 mass-matched random triples per box
(12,000 draws).

| group | effect (hPa) | align PC1 | align top-5 | subspace top-5 | in-box mass |
|---|---|---|---|---|---|
| convection | +2.79 | 1.8% [0.4, 5.3] | **11.3%** [5.8, 18.8] | 6.9% [4.0, 9.1] | 94.6 |
| asc21 | +3.63 | 1.4% [0.3, 3.0] | 5.9% [3.7, 11.7] | 4.1% [2.7, 5.7] | 61.8 |
| asc17 | +0.02 | 0.9% [0.2, 2.1] | 3.3% [1.6, 5.2] | 3.4% [2.5, 4.7] | 51.7 |
| moisture | −0.03 | 0.7% [0.0, 3.1] | 5.7% [2.7, 9.4] | 4.9% [3.3, 7.3] | 10.3 |
| **TC feature** (pos. ctl) | — | **4.1%** [0.5, 9.0] | **18.2%** [12.0, 28.6] | **18.2%** | 149.7 |
| random (null) | — | 0.9% [0.2, 3.0] | 4.9% [2.1, 10.1] | 4.7% [2.4, 8.3] | 98.3 |

Per box against that box's own null: **LOCAL-AGGREGATE 16/60, ORTHOGONAL 37/60**, below q10
2/60, in the gap 5/60. The rule required 70% for a verdict, so the verdict is **NEITHER**
and is reported as such. Convection sits between the null and the TC feature and does not
resolve to either.

### Guardrail #9, all three sides

1. **Null varies.** `align1` over 12,000 random draws spans 0.001% to 13.5%. Not a point
   mass.
2. **Bar attainable.** 8.5% of random draws clear the LOCAL-AGGREGATE bar, so it is not
   above the statistic's ceiling.
3. **Negative control fails it.** The calibrated moisture group clears it on 6/51 boxes,
   i.e. at the null rate. The statistic is not just reading firing mass.

The positive control behaves: the TC feature is 4.5x the null on `align1` and 3.9x on the
subspace statistic, so the instrument can detect alignment when it is there.

## What the numbers do say

**Convection's excess is in where it fires, not in which directions it owns.** Its
`align top-5` is above its own box's null on **40/60** boxes, but its decoder-subspace
overlap is above null on only **14/60** — at the null rate. The group's decoder directions
are geometrically generic; what is not generic is that they fire on the structured part of
the box. Alignment is inherited from the storm, not from the dictionary.

**Geometry does not separate the strongest lever from the null one.** asc21 (+3.63 hPa, the
largest effect in the library) is at the null on every statistic: above the null q90 on
`align1` on 6/60 boxes, against asc17's (+0.02 hPa) 1/60. Their medians are 1.4% vs 0.9%
and 5.9% vs 3.3%. A 227x difference in causal effect between two groups whose geometry with
respect to the dominant local modes is nearly the same.

## Where this leaves the programme

Three candidate observational predictors of "is this group a causal lever" have now been
tested on the same battery, and all three fail:

| predictor | result |
|---|---|
| in-box exposure | correlates rho 0.72 across arms, but 227x effect spread at matched exposure (`exposure_confound_2026_08_20.md`) |
| calibrated ascent score | non-monotone; hard null at 16.8 sigma with real exposure |
| alignment with dominant local modes | asc21 and asc17 indistinguishable |

That is the sharper version of this repo's standing thesis. It is not only that observational
*causal discovery* on the internals fails; the cheap observational proxies for *which
direction is worth intervening on* fail too. Nothing so far substitutes for running the
intervention.

It also means the asc17 null remains unexplained after a third attempt. That is now the most
informative open question in the lane, because whatever separates asc21 from asc17 is the
thing the labelling instrument is missing.

## Declared limits

- **Not the seven storms.** IID 2016-2020 initial conditions with TC-feature-located boxes.
  Confirming on Ida et al. needs one baseline rollout per storm with an in-box activation
  dump. That is GPU work; the card was occupied by another session and it was not run.
- **Single time step**, so this is the geometry at the moment of the edit, not its
  accumulation over a 96 h rollout.
- **Delete direction** for `align1`/`alignK`; `sub` is clamp-target-free and agrees.
- `align1` is a within-box statistic and says nothing about whether the downstream response
  is mode-like.
