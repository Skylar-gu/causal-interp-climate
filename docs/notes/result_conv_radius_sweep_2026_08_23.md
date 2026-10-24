*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# RESULT — The convection lever does not depend on the 1,500 km disk

**2026-08-23.** Prereg: `notes/prereg_conv_radius_sweep.md` (frozen and committed before any
non-1500 number existed). Runs: `results/skill/conv_r{500,750,1000,2500}/`; scored by
`src/conv_radius_sweep_analyze.py` into `results/skill/conv_radius_sweep.json`;
figure `figures/conv_radius_sweep.png`. ~3 GPU-h total, all four radii exit 0, 8/8 storms
at every radius, no arm missing.

## Verdict: SATURATES (pre-registered reading)

Median over the seven RI storms, conv-normal arm, r(R) = Δ(R)/2.794:

| R (km) | Δ-deepening (hPa) | r | Δ-err vs ERA5 | rand ctrl | non-developer |
|---|---|---|---|---|---|
| 500 | +3.433 | 1.23 | +2.43 | −0.009 | −0.010 |
| 750 | +2.510 | 0.90 | +1.78 | +0.073 | −0.028 |
| 1000 | +2.591 | 0.93 | +1.70 | −0.012 | +0.021 |
| 1500 | +2.794 | 1.00 | +1.61 | +0.013 | −0.010 |
| 2500 | +2.109 | 0.75 | +1.18 | +0.006 | +0.019 |

r(1000) = 0.93 ≥ 0.80 and r(2500) = 0.75 ≤ 1.25 → **SATURATES**. The effect does not grow
with disk area; the random control stays ≤ 0.073 hPa and the non-developing wave ≤ 0.028 hPa
at every radius, so the large-radius readings are not perturbation-area effects. The
1,500 km design choice is vindicated: the committed result is a statement about the storm's
convection, not basin-scale convection. Exposure check: r vs exposure Spearman −0.60,
nowhere near the ≥ 0.9 "you get what you ablate" signature.

## The cleaner statement is in the conv-zero arm

conv-normal is non-monotone in R (1.23 → 0.90 → 0.93 → 1.00 → 0.75). That wiggle is a
**normal-reference artifact, not physics**: `ftarget` is the mean positive activation over
the disk's own nodes on quiet analogs, so a 500 km disk finds no positive firing for 2–3 of
the 3 convection features in 6 of 8 storms (target 0 → restore-to-normal degenerates toward
delete → inflated 3.43), while a 2,500 km disk raises the targets (weaker clamp → 2.11).
This is the same failure mode as `notes/result_normal_reference_2026_08_21.md`, now visible
as a function of disk size.

The conv-zero arm has no reference (cap at 0 at every radius) and is therefore the clean
area-dependence readout:

| R (km) | conv-zero Δ-deepening | ratio to 1500 |
|---|---|---|
| 500 | +3.727 | 0.90 |
| 750 | +4.334 | 1.04 |
| 1000 | +4.063 | 0.98 |
| 1500 | +4.153 | 1.00 |
| 2500 | +4.046 | 0.97 |

**Flat from 750 km on, and already 90% saturated at 500 km.** The causally relevant
convection firing sits within ~500 km of the storm centre; everything beyond ~750 km adds
nothing. For the paper: the intensification-relevant scale of the convection lever is
≤ 750 km — inner-core plus immediate rainband environment — and 1,500 km was, if anything,
conservative.

## Instrument failures and caveats, reported as such

- **Determinism gate FAILED as written.** The prereg required the baseline arm to reproduce
  the committed battery bit-for-bit; it does not (max |ΔMSLP| 0.03–0.32 hPa per storm).
  This is the known re-run nondeterminism floor, and Δs are computed arm-minus-baseline
  within each run, so medians are internally consistent — but the cross-radius ratio r
  carries a ~±0.1 floor against a 2.5–4 hPa signal. Reported, not repaired.
- **The conv-normal counterfactual is not radius-stable by construction**, because its
  reference is estimated on the disk itself. Any future extent sweep should freeze
  `ftarget` at the 1,500 km estimate and vary only the mask. The conv-zero table above is
  the arm on which the SATURATES conclusion actually rests quantitatively; the
  pre-registered conv-normal bars were met anyway (0.90/0.93 ≥ 0.80, 0.75 ≤ 1.25).
- Exposure is a +48 h single-snapshot proxy (the storm moves under a fixed disk).
- 500 km with a fixed centre under-covers a moving storm late in the rollout; the 0.90
  zero-ratio at 500 km may be partly track motion, so "≤ 750 km" is the safe scale claim,
  not "≤ 500 km".
