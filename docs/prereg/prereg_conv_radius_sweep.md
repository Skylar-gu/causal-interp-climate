*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# PREREG — Does the convection result depend on the 1,500 km disk?

**Frozen 2026-08-23, before any radius other than 1,500 km has produced a number.**
Motivation: the committed result (`notes/RESULT_convection_lever.md`) restores the convection
group [2401, 2067, 3174] to normal inside a 1,500 km disk. That radius was a design choice
(`spec_convection_skill_necessity.md`) and was never varied. A reviewer can fairly ask whether
1,500 km is "the storm's convection" or "basin-scale convection". An earlier attempt
(`out/conv_r1000.log`, `out/conv_r750.log`) crashed on the overlap guard before running a
single arm; no radius number exists anywhere in the repo.

## Design

Identical operator, features, storms, ICs, analogs, horizon (+96 h) and arms to the committed
battery. The only change is `CONV_RADIUS_KM` ∈ {500, 750, 1000, 2500}; 1,500 is the existing
`results/skill/convection/`. The normal reference `ftarget` is the mean positive activation
over the disk's own nodes on the quiet analogs, exactly as at 1,500 km, so "normal" is
re-estimated per radius under the same definition (it is the same counterfactual, not a new
one). Storms: the seven RI storms + the non-developing wave. Arms: baseline, conv-normal,
conv-zero, rand-normal (frozen global-rate control, `RANDOM_CTRL`).

Output: `results/skill/conv_r{R}/run_<storm>.npy`; analysis writes
`results/skill/conv_radius_sweep.json`.

## Readouts

Per storm and radius: Δ-deepening (baseline minus arm, hPa, as in the committed table),
Δ-MSLP-error vs ERA5 over the intensification window, and **disk exposure** = the fraction
of the convection group's node-level firing at the +48 h baseline snapshot (the only
node-level snapshot the runs store) that lies inside the disk, normalised to the 2,500 km
disk. Single-time proxy; the storm moves, so it is a covariate, not an exact dose. Exposure is the covariate that says how much of the anomalous
convection each radius actually touches.

## Frozen reading of the outcome (median over the seven RI storms, conv-normal arm)

Let `r(R) = median Δ-deepening(R) / 2.794`.

- **SATURATES** — `r(1000) ≥ 0.80` and `r(2500) ≤ 1.25`: the 1,500 km choice is vindicated;
  the intensification-relevant convective environment is ≤ 1,000 km and the paper may say
  "the storm's convection". Report the smallest radius with `r ≥ 0.80` as the scale.
- **STILL GROWING** — `r(2500) ≥ 1.25`: the effect is not confined to the storm; the paper
  must say "convection within the storm's basin-scale environment" and the 1,500 km number
  becomes a lower bound on the relevant scale, not a storm-scale statement.
- **COLLAPSES INWARD** — `r(1000) < 0.80` but `r(2500) ≤ 1.25`: the lever lives in the
  1,000–1,500 km annulus, i.e. the inflow environment rather than the core. Reported as
  such; it does not weaken the causal claim but changes its physical label.
- **Exposure check.** If `r` tracks exposure (Spearman over the five radii ≥ 0.9), the
  result is "you get what you ablate" and scale is a property of the feature footprint,
  not of the mechanism. Stated plainly either way.

## Controls that must hold at every radius (guardrail #9)

- rand-normal median |Δ-deepening| < 10 % of conv-normal at the same radius, and
  < 0.3 hPa absolute. If the random control *grows* with radius, the large-radius effect
  is perturbation-area, not mechanism, and the STILL GROWING reading is void.
- Non-developing wave |Δ| < 0.3 hPa at every radius.
- Baseline arm reproduces the committed baseline MSLP trace bit-for-bit at every radius
  (same IC, same compiled graph; the disk is not used) — a free determinism gate.

## Cost and ops

8 storms × 4 radii × ~6 min ≈ 3.5 GPU-h, serialised behind whatever owns the card.
Launcher waits on `nvidia-smi --query-compute-apps=used_memory` reporting no process, per
CLAUDE.md; no `pgrep`. Crash-safe per storm (existing `run_<storm>.npy` is skipped).
Storms whose baseline already fails to intensify are excluded by the committed battery, not
here; nothing is re-selected.

## What is not allowed afterwards

No changing the 0.80 / 1.25 thresholds, no dropping a radius, no swapping the analog set
(`CONV_ANALOG_SPAN` stays 0 so the medians are comparable to the committed table). If a
storm crashes at one radius it is reported as missing at that radius, not re-run with
different settings.
