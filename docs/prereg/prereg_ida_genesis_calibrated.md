# PREREG — Ida genesis knockout, re-run under calibrated mechanism labels

**2026-08-29.** Committed before any arm other than the baseline has been run.

## Why

The −41 / −17 / −12 / +1 % genesis panel (`39c8e9b`, quoted by `build_genesis.py:78` and
`art_bars.py:45`) selected its feature groups with the v1 labeller (`label_mechanisms.py`,
fixed bar, argmax, 96.9% of features labelled, never nulled). `notes/labeling_repair_2026_08_15.md`
re-scored those groups under the rotation-null labeller (`results/fs_mechanisms_v2.npy`):

| old arm | features | calibrated |
|---|---|---|
| convection | 2401, 2067, 3174 | 3/3 ascent (+28.5/+29.1/+21.7 σ) — survives |
| vorticity | 3861, 2514, 2089 | 3/3 vort850 — survives |
| moisture | 3501, 845, (3174) | ascent, ambiguous, ascent — **not a moisture group**, and shared 3174 with convection |
| shear | 2349, 1996, 744 | ambiguous, shear (gap 1.3 vs bar 1.2, near-ubiquitous), ambiguous — **not a shear group** |

The old panel also had no random control. The convection bar is safe; the *comparison* is not.
The grouped-knockout script itself was never committed; protocol is reconstructed from
`steer_ida_genesis.py` (persistent ±1 patch, readout) and `ida_mechmaps.py` (groups, IC, box).

## Protocol (unchanged from the original where it was sound)

- Model/SAE/patch: flagship GraphCast bf16, layer-8 TopK SAE, `fc.coef_patch(sae, group, ±1)`
  applied at **every** step (persistent), from IC **2021-08-26** for **H = 8** steps (48 h).
- Readout: feature **3243** activation summed over mesh nodes in BOX lat 10–33, lon −98 to −58,
  at every lead; headline number is +48 h. Δ% = (arm − base)/base at +48 h.
- Guardrail: hard assert that 3243 is in **no** group, including the random controls
  (`mechanism_library_2026_08_17.md` §1).

## Feature selection (the part that changes)

Exposure `E_f` = max over the 8 leads of the in-box activation sum of feature f in the
**baseline** rollout. Selection uses only baseline activations and calibrated labels; no
intervention outcome enters it.

- **convection**: the committed triplet 2401, 2067, 3174 (identical to every other arm in the
  repo). Also run `ascent_byrule` = top-3 calibrated-ascent by E_f excluding the triplet, as a
  check that the triplet is not special.
- **moisture**: top-3 by E_f among `label == q600`, E_f > 0.
- **vorticity**: top-3 by E_f among `label == vort850`, E_f > 0.
- **shear**: top-3 by E_f among `label == shear`, E_f > 0. If fewer than 3 calibrated shear
  features have E_f > 0, run what exists and report the arm as **no exposure**, not null.
- **random control** ×3 seeds (7, 8, 9): for each convection feature, one feature drawn from
  the pool {E_f within [0.5×, 2×] of that feature's E_f, label ≠ ascent, not 3243, not in any
  mechanism group}. If the pool is empty at that tolerance, widen to [0.25×, 4×] and say so.

## Arms

baseline ×2 (nondeterminism floor); ablate and dose for convection, ascent_byrule, moisture,
vorticity, shear; ablate all-four (convection+moisture+vorticity+shear); dose convection+moisture;
ablate random ×3. ~18 rollouts × ~90 s.

## Bars, fixed now

- **Floor**: `floor = |base_run1 − base_run2|` at +48 h. Any |Δ| ≤ 3·floor is reported as
  "within floor".
- **Control bar**: `ctl = max over 3 seeds |Δ_random|`. A mechanism effect is reportable only if
  |Δ| > 3·ctl AND |Δ| > 3·floor.
- **Necessary**: ablate Δ ≤ −15% of baseline (the script's original convention) and passes both
  bars above.
- **Sufficient**: dose Δ ≥ +15% and passes both bars.

## Predictions, written before running

- Convection ablate ≤ −30% (reproduces −41% within floor); dose ≥ +30%.
- Vorticity ablate near −12%.
- Moisture (genuine q600): the 7-storm battery says ~0 on deepening (`moisture2` −0.03 hPa,
  `mech_q600` +0.22). Prediction: |Δ| < 15%. If it comes in ≤ −15% and passes bars, the
  "moisture is not the lever" statement needs a genesis-vs-intensification qualifier.
- Shear: expected no exposure or |Δ| < 15%. Under no exposure the arm says nothing about shear.
- Random: within floor.
- all-four ≈ convection alone (bottleneck) is the old claim; if all-four ≪ convection alone, the
  ingredients are additive rather than convection-gated.

## Outputs

`results/fs_ida_genesis_v2.npy`, `out/fs_ida_genesis_v2.log`, result note
`notes/result_ida_genesis_calibrated_2026_08_29.md`. If the moisture/shear bars change,
`art_bars.py:MECH` and the `build_genesis.py` alt-text/caption are updated to the new numbers.

## AMENDMENT 1 (2026-08-29, after the first run's group printout, before any non-convection arm)

The first launch reproduced convection at **−41%** (baseline 48.8 → 29.0; repeat-baseline floor
0.05) and then printed its other groups:

    ascent_byrule: 1311[z=-5.4,E=2380] 3605[z=-4.4,E=1508] 1129[z=-3.5,E=496]
    moisture:      2153[q600,z=-5.6,E=1620] 2333[q600,z=-3.6,E=1556] 1829[q600,z=+8.1,E=524]
    vorticity:     1334[vort850,z=-4.4,E=759] 3691[vort850,z=-2.7,E=434] 1592[vort850,z=+10.8,E=416]
    shear:         1996[shear,z=+3.8,E=289] 1336[shear,z=-3.1,E=91] 3765[shear,z=-3.1,E=75]

Two defects in the selection rule, both mine, fixed before those arms ran:

1. The calibrated `label` is assigned on a two-sided |z|, so a feature can be labelled "ascent"
   with **negative** z — it fires where ascent is anomalously *weak*. That is a descent feature,
   not a convection feature. Every group in `labeling_repair_2026_08_15.md` has positive z. Rule
   now requires **z[f, m] > 0**.
2. Ranking on raw in-box exposure selects near-ubiquitous features (E 1,500–2,400 against the
   convection triplet's 24–38), the same "1996 fires 254,703 times" problem the repair note
   flagged. Rule now: among `label == m`, `z > 0`, and **E_f ≥ 0.5 · min(E_convection)**
   (the feature must touch Ida at least half as much as the weakest convection member),
   rank by **z** and take the top 3. This is how the convection and battery groups were
   themselves chosen (strongest calibrated members), with an exposure floor added so a null
   cannot be a no-exposure null in disguise. Exposure is reported per feature either way.

Random controls unchanged (exposure-matched to the convection triplet). Run killed after the
convection-ablate arm; relaunched under this rule. The −41% convection reproduction from the
killed run is recorded here and will be re-measured in the relaunch.

## AMENDMENT 2 (2026-08-29, written after the vorticity-ablate arm printed, before the follow-up runs)

Run 2 (amended rule) so far: baseline 49.1 (repeat 49.1, floor 0.08); convection −41%;
ascent_byrule (3901/553/866) −31%; genuine q600 moisture (2415/3780/1829) **+3%**;
vorticity (2089/2514/3316) **−54%** — against a prereg prediction of "near −12%".

The vorticity group differs from the 39c8e9b group in one feature: 3316 (calibrated vort850,
z +12.5, centroid +27°N, in-box exposure 110) replaces 3861 (centroid +85°N, exposure below
floor). Checked before deciding anything: no group feature is a near-duplicate of 3243
(decoder cosine ≤ 0.20, footprint cosine ≤ 0.46 for all; the 0.46 is convection's 2401), so
−54% is not the readout ablated by another name.

Follow-up, same protocol, `steer_ida_genesis_v2_followup.py`: single-feature ablations of
each vorticity member and each convection member, plus the original 39c8e9b vorticity group
(3861/2514/2089). Reading, fixed now: if 3316 alone accounts for ≥ 75% of the group effect,
the vorticity result is "one storm-core spin feature", and the paper's low-level-spin
statement (0.55 hPa over seven storms, `mech_vort850`) stands as the multi-storm number while
Ida genesis gets its own sentence. If the effect is spread across members, low-level spin is a
second genesis lever on Ida and the 39c8e9b −12% was an exposure artifact of polar features.
