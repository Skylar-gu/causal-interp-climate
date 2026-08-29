# RESULT — Ida genesis knockout under calibrated labels: convection stands, moisture was never a lever, and one storm-core spin feature is a second one

**2026-08-29.** Prereg `notes/prereg_ida_genesis_calibrated.md` (+ amendments 1, 2; all committed
before the arms they govern ran). Scripts `flagship_sae/steer_ida_genesis_v2.py`,
`steer_ida_genesis_v2_followup.py`, scorer `ida_genesis_v2_analyze.py`. Results
`results/fs_ida_genesis_v2.npy`, `fs_ida_genesis_v2_verdict.json`, `fs_ida_genesis_v2_followup.npy`;
logs `out/fs_ida_genesis_v2*.log`. GPU: 5.4 min main + 2.9 min follow-up + ~3 min mechmaps.

## Why this was run

The genesis panel in the artifact (`build_genesis.py`, `art_bars.py`) quoted
**−41 / −17 / −12 / +1 %** for convection / moisture / vorticity / shear, from `39c8e9b`
(2026-08-12), whose groups were chosen with the uncalibrated v1 labeller. The 08-15 repair had
already shown the "moisture" group was two weak-ascent features plus 3174 (shared with
convection) and the "shear" group was 2/3 `ambiguous`. The convection bar was safe; the
comparison was not. The grouped-knockout script was never committed, so the protocol was
rebuilt from `steer_ida_genesis.py` + `ida_mechmaps.py`.

## Protocol

Persistent `coef_patch(±1)` on a feature group through a 48-h rollout from 2021-08-26 (Ida at
formation, TC feature ≈ 0); readout = feature 3243 summed over the Caribbean–Gulf box
(10–33°N, 98–58°W) at +48 h. Groups: convection = the committed triplet; moisture / vorticity /
shear = the three strongest rotation-null-calibrated members of that mechanism (z > 0 on its own
probe) with in-box baseline exposure ≥ half the weakest convection member (amendment 1).
Random control ×3 seeds, exposure-matched to the convection triplet, drawn from non-ascent
features. Hard assert: 3243 in no group.

## Bars (fixed in the prereg)

Baseline +48 h **49.1**, repeat **49.1**, floor 0.08. Random controls: +0.2, −0.0, **−5.9**
(seed 9: 1833/3519/4003). Bar = 3 × max|random| = **17.6 units = 36 %**. Necessary = ablate
≤ −15 % and above bar; sufficient = dose ≥ +15 % and above bar.

## Result

| arm | features (calibrated label, z) | exposure | ablate Δ% | dose Δ% | verdict |
|---|---|---|---|---|---|
| **convection** | 2401, 2067, 3174 (ascent +28.5/+29.1/+21.7) | 31 | **−41** | **+48** | NECESSARY, SUFFICIENT |
| **low-level spin** | 2089, 2514, 3316 (vort850 +27.1/+18.7/+12.5) | 53 | **−54** | **+51** | NECESSARY, SUFFICIENT |
| ascent, second trio | 3901, 553, 866 (ascent +33.1/+22.3/+20.6) | 93 | −31 | +22 | above 15 %, **below the 36 % control bar** |
| moisture (genuine q600) | 2415, 3780, 1829 (q600 +10.9/+9.6/+8.1) | 330 | **+3** | −41 (see caveat) | within bar |
| wind shear | 1996, 3460 (shear +3.8/+3.2; only 2 pass the floor) | 171 | **+3** | −13 | within bar |
| all four ablated | 11 features | — | **−80** | — | NECESSARY |
| convection + moisture dosed | 6 features | — | — | −10 | within bar |
| random ×3 | ambiguous, exposure-matched | 23–46 | 0 / 0 / −12 | — | — |

Old panel for comparison: −41 / −17 / −12 / +1, all-four −41, conv+moist dose +42.

**Follow-up (amendment 2), single features, same protocol** (baseline 48.9):

| feature | label | alone |
|---|---|---|
| 3316 | vort850, +12.5σ, centroid 27°N, exposure 110 | **−45 %** |
| 2514 | vort850, +18.7σ | −10 % |
| 2089 | vort850, +27.1σ, centroid 75°N | 0 % |
| old group 3861/2514/2089 (39c8e9b) | 3/3 vort850, polar | **−11 %** (reproduces the old −12) |
| 2401 | ascent | −27 % |
| 3174 | ascent | −18 % |
| 2067 | ascent | 0 % |

## What this settles

1. **The convection bar is exactly reproduced (−41 %) and is not three special features.** A
   disjoint second trio of calibrated ascent features gives −31 % / +22 %. Within the triplet the
   effect is carried by 2401 and 3174 (−27, −18; 2067 alone does nothing) and is roughly additive.
2. **Moisture was never a lever at genesis.** The old −17 % was weak ascent. Genuine q600
   features, at 10× the convection group's exposure, give +3 % when removed. This matches the
   seven-storm deepening battery (`moisture2` −0.03 hPa, `mech_q600` +0.22 hPa) and closes the
   loophole flagged in `RESULT_convection_lever.md`.
3. **Wind shear does nothing, now with a group that is actually shear.** +3 % ablated, −13 %
   dosed (correct sign, under bar). The old "+1 %" was 2/3 unlabelled features.
4. **Low-level spin is a second handle on Ida, and it is one feature.** 3316 alone gives −45 %,
   83 % of the group effect (≥ the 75 % threshold fixed in amendment 2 → reading: "one
   storm-core spin feature"). The old −12 % came from a group whose members sit at 49–85°N and
   barely fire on Ida; 2089 (the strongest-calibrated vort850 feature in the dictionary, 75°N) is
   literally 0 %. The vorticity label rule was fine; the old group was an *exposure* artifact.
5. **Convection is not the sole gate.** All-four ablated is −80 %, ≈ convection −41 % plus spin
   −54 % less overlap, not "−41 = convection alone" as the old panel said. Ingredients add.

## Caveats, reported as such

- **Dose arms on large-footprint groups are not exposure-matched.** `coef_patch` is global, not
  disk-limited like the seven-storm battery. The q600 features fire on 12–15 % of all nodes and
  1996 on 26 %, against 0.1–0.4 % for the convection triplet, so "double moisture" is a
  whole-tropics perturbation. Moisture-dose −41 % (one-sided; ablate +3 %) and conv+moist-dose
  −10 % are read as generic degradation from a huge global patch, not a moisture mechanism. The
  ablate direction is the exposure-comparable one for those groups. Convection and spin are
  two-sided (ablate down, dose up), the signature of a real handle.
- **The control bar is set by one seed.** Two random groups are at 0 %, the third at −12 %.
  With three seeds, 3 × max is a conservative, high-variance bar; the second ascent trio (−31 %)
  falls under it and is reported as such, not promoted.
- **One storm, one IC, one internal readout.** The seven-storm physical-readout battery is the
  generalization evidence; there, `mech_vort850` (group 2822/2935/1148/2089 — polar, two members
  anti-signed, exposure 0.9–5.4) gave +0.55 hPa. That battery has not been run with 3316; on this
  result it should be. [Done later the same day — see the section at the end.]
- **The paper sentence** "Amplifying convection rather than vorticity also produces the larger
  response in the model's own cyclone representation" (`main.tex:133`, from the
  `ida_dialup_convection_vs_vorticity` figure, +39 % vs +15 %) was measured with the polar
  vorticity group. With a spin group that fires on Ida, dose is +51 % vs +48 %. The sentence is
  group-dependent and should be qualified or dropped. Not edited here.

## Two instrument lessons

- `fs_mechanisms_v2.npy['label']` is assigned on **two-sided |z|**: a feature can be labelled
  "ascent" with z = −5 (fires where ascent is anomalously *weak*). Any group built from `label`
  must also check the sign (amendment 1). The first launch picked three such features as
  "ascent"; they gave −2 %.
- Ranking by raw exposure selects ubiquitous features. Rank by calibrated z with an exposure
  floor, and report exposure per feature.

## Artifact updates

`art_bars.py` now reads `fs_ida_genesis_v2_verdict.json` (bars −41 / −54 / +3 / +3 with the
control band); `build_genesis.py` text, alt-text and methods footnote updated; mechmaps
re-rendered with the calibrated groups (`MECHMAPS_TAG=_v2`, `results/fs_ida_mechmaps_prog_v2.npy`,
`figures/art_mechmaps_v2.png`). `paper_overleaf/main.tex` not touched.

---

## 2026-08-29, later — seven-storm physical battery for the spin group (prereg `notes/prereg_spin3316_battery.md`)

Same protocol as `convection` / `mech_vort850` (1,500 km disk, restore-to-normal, delete-to-zero,
firing-rate-matched random group, 7 developing TCs + `nondev2013`). Runs
`results/skill/mech_spin3316/` (2089/2514/3316) and `results/skill/mech_3316/` (3316 alone),
scored by `skill_conv_analyze.py`; driver `bash_files/run_spin3316.sh`, logs `out/mech_spin3316.log`,
`out/mech_3316.log`. 54 GPU-min total.

| group | median Δ-deepening, → normal | → zero | random | non-dev | median TC-feature suppression |
|---|---|---|---|---|---|
| convection 2401/2067/3174 (committed) | 2.794 | 4.153 | +0.013 | −0.010 | 0.12 |
| **spin 2089/2514/3316** | **3.301** | **7.330** | +0.029 | −0.041 | 0.25 |
| **3316 alone** | **2.953** | **5.019** | −0.083 | +0.016 | 0.23 |
| `mech_vort850` 2822/2935/1148/2089 (paper's "low-level spin") | 0.553 | 0.824 | +0.031 | −0.016 | 0.01 |
| genuine q600 (`moisture2`) | −0.032 | −0.077 | −0.000 | −0.004 | — |

Per storm, spin → normal (hPa; % of baseline deepening): Ida 7.97 (115 %), Wilma 9.18 (40 %),
Goni 4.38 (36 %), Michael 3.30 (29 %), Patricia 2.14 (39 %), Haiyan 3.00 (13 %), Haishen 0.77
(3 %). Same sign on 7/7 for both batteries; every random-control and non-developer cell inside
the ±0.06–0.08 floor. Exposure (`conv_box`) 16–50 for the group, 38–117 for 3316 alone, i.e.
comparable to or above convection's (39.55 median); this is not a low-exposure null and not a
high-exposure artifact (Haishen has the second-lowest exposure *and* the smallest effect, but
Michael at the same exposure gives 29 %).

**Prereg reading: the ≥ 1.0 hPa threshold is cleared 3×.** Low-level spin is a second
multi-storm physical lever, at least as strong as convection on the restore-to-normal
counterfactual (3.30 vs 2.79) and stronger on delete (7.33 vs 4.15). The paper's "much smaller
response (0.55 hPa)" for low-level spin (`main.tex:133`, Figure 2c) was a property of that group, not of the mechanism. 3316 alone carries ~90 % of the group's
restore-to-normal effect.

**Correction (same day).** Earlier text in this note called the paper's spin group "the polar
group 3861/2514/2089". That is the 39c8e9b *Ida-genesis* group. The seven-storm `mech_vort850`
battery (the 0.55 hPa the paper quotes) actually used **2822 / 2935 / 1148 / 2089** (run files'
`conv` key; `figures/main_claims/README.md`). It is worse than "polar": centroids 71–87°,
**two of four with negative z on the vorticity probe (2935 −13.9, 1148 −13.6 — anti-vorticity
features labelled "vort850" by the two-sided rule)**, storm-box exposure 0.9–5.4 against
convection's 22–44. The exposure-artifact conclusion stands and is stronger; the group identity
above is corrected.

**Caveat that must travel with this number.** 3316 is a storm-core feature: it suppresses the
cyclone representation twice as much as convection does (TC-feature suppression 0.23 vs 0.12;
Patricia 0.75). It is not the readout by another name (decoder cosine with 3243 is 0.15,
footprint cosine 0.27, both under the 0.45 redundancy threshold, and the `mech_atm_river`
contamination case sat at 0.985), but it is *more proximal* to the cyclone representation than
the convection features are. "Remove the storm's low-level spin" is a legitimate physical
counterfactual — the seed vortex is a textbook genesis ingredient — but it is a step closer to
"remove the storm". A sharper test would be the lead-time ordering of 3316 vs 3243 at genesis
(does spin fire before the cyclone feature?), which the saved outputs do not resolve; it is the
natural next probe before any paper sentence promotes spin to co-equal with convection.

Not edited: `paper_overleaf/main.tex`. Sentences affected: line 133 ("low-level-spin (vorticity)
group produces a much smaller response (0.55 hPa)" and "Amplifying convection rather than
vorticity also produces the larger response in the model's own cyclone representation"), and
Figure 2(c) if it plots the polar group as "low-level spin".

### Ordering check (same day, CPU, from the battery run files' per-lead `box_feats`)

Lead (6-h steps) at which each feature first reaches 20 % of its own peak, baseline arm:

| storm | 3316 on | 3243 on | 2401 on | TC at IC |
|---|---|---|---|---|
| ida2021 | 0 | 4 | 1 | absent |
| michael2018 | 0 | 2 | 1 | absent |
| patricia2015 | 0 | 3 | 10 | absent |
| wilma2005 | 0 | 4 | 8 | absent |
| goni / haishen / haiyan | 0–1 | 0 | 0 | present (mature-storm ICs) |
| nondev2013 | 1 | never | never | no TC ever |

In every storm that forms *inside* the rollout, 3316 is already firing at lead 0 and the cyclone
feature switches on 12–24 h later; on Ida 3316 then *fades* as 3243 grows (108 → 14 while 3243
goes 0 → 92). 3316 also fires on the non-developing wave, where 3243 never turns on. So 3316 is
a precursor that exists without the storm and precedes it, not the cyclone representation under
another name. The proximity caveat above is downgraded: "remove low-level spin" is an
intervention on a genesis ingredient, and its TC-suppression 0.23 is what a precursor's removal
should look like. (Convection 2401 precedes 3243 on Ida/Michael but lags it on Patricia/Wilma,
which is consistent with the spin group beating convection on the physical readout.)

### Dose–response for 3316 (prereg `notes/prereg_gain_3316.md`; `results/skill/gain_3316/`, 21 GPU-min)

MSLP-minimum RMSE against ERA5 over the intensification window (`make_figures.gain_curve`),
g = 0 restore-to-normal, g > 1 scales the excess above normal in the 1,500 km disk:

| storm | baseline | g=0 | ×1.25 | ×1.5 | ×1.75 | ×2 | ×2.5 | ×3 | convection, same storm (best) |
|---|---|---|---|---|---|---|---|---|---|
| Ida | 7.39 | 11.41 | 6.29 | 5.27 | 4.33 | 3.52 | 2.56 | **2.29** | 7.35 → 3.13 at ×2, back to 7.46 at ×3 |
| Haishen | 4.09 | 4.58 | 4.01 | 3.83 | 3.87 | 3.72 | 3.44 | **3.33** | 4.03 → 3.01 at ×1.25, then 19.3 at ×3 |
| Patricia | 6.69 | 7.74 | 6.40 | 6.19 | 5.97 | 5.71 | 5.32 | **5.08** | 6.64 → 6.22 at ×3 (flat) |

**The prereg prediction ("shallower improvement or none, error rising past ×1.5") was wrong.**
Amplifying the spin feature improves all three forecasts monotonically through ×3 with no
overshoot inside the sweep; convection is sharp and turns over (Haishen catastrophically).
Spin is therefore a *steerable* lever, not only a necessary ingredient — and a better-behaved
one than convection: a gentler, monotone response that reaches a lower error on Ida (2.29 vs
3.13) and moves the convection-null storm Patricia (6.69 → 5.08 vs 6.64 → 6.22). g = 0
reproduces the `mech_3316` ablation direction (error up on all three). Where the spin curve
turns over, if it does, is beyond ×3 and untested.

Figure 2.5 (`figures/main_claims/figure2p5_interventions_notitle.pdf`, copied to
`paper_clean/images/`): Figure 2's layout with 3316 in (a) and (b), and (c) with the spin and
convection bars swapped. Built by `figures/main_claims/build_figure2p5.py` from the Figure 2
HTML; same render pipeline.
