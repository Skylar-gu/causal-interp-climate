*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# SPEC — Convection is causally necessary for GraphCast's hurricane-intensity forecast skill

**Advancing the genesis result** (`figures/genesis_writeup.html`) from "one storm, internal feature"
to "across storms, on the real storm, against reality." Design frozen before results (R4b). 2026-08-12.

## Claim under test

GraphCast forecasts tropical-cyclone intensification *because* it represents convection. Remove the
convection features and the model should fail to deepen the real storm — degrading its intensity
forecast vs ERA5 — while a non-developing disturbance and random-feature ablations are unaffected.

## Design

**Storms (rapid-intensification, well-observed, across basins where feature 3243 fires):**
Ida 2021 (Atlantic, ~Aug 26 IC), Ian 2022 (Atlantic, ~Sep 26), Michael 2018 (Atlantic, ~Oct 7),
Haishen 2020 (W-Pac, ~Sep 3), Goni 2020 (W-Pac, ~Oct 29), Haiyan 2013 (W-Pac, ~Nov 5). Verify each
IC has feature 3243 turning on and a real deepening in ERA5 over the following 72–96 h; drop any that
GraphCast doesn't forecast to intensify (a model that never spins the storm up can't have it removed).

**Controls:**
- Non-developing tropical wave (a disturbance that stayed weak in ERA5) — convection-knockout should
  barely change its (already flat) intensity trajectory.
- Random-feature ablation: 3 random features of matched firing rate — should not suppress deepening.

**Intervention (honest counterfactual):** convection features (2401, 2067, 3174) restored to their
NORMAL level within a ~1500 km disk around the storm (reuse `build_apply_cond`/`delta_cond` and the
normal reference from `steer_ida_counterfactual.py`), held persistently through the rollout. Compare
to (a) baseline (untouched) and (b) delete-to-zero, for continuity with the current result.

**Rollout:** from each formation IC, roll to +96 h (16 steps), bf16, flagship 0.25°.

## Readouts (per storm)

1. **Internal:** feature 3243 activation in a storm-tracking box (continuity with the artifact).
2. **Physical:** min MSLP and max 10 m wind in the box vs lead — the actual forecast intensity.
3. **Skill:** RMSE / intensity error of baseline vs convection-ablated forecast against **ERA5 truth**
   (MSLP, wind) over the intensification window. Does removing convection make the forecast *worse*?
4. **Co-location (the open check):** at a mid-intensification step, confirm convection feature 2401
   fires ON the storm (overlay its firing on the TC feature / MSLP low) — the pixel-tight version of
   "the convection feature is active at the hurricane."

## Frozen success criteria

- **Convection is necessary for intensity skill** iff, across the storms: convection-normal-patch
  reduces forecast deepening (less MSLP drop, lower peak wind) AND increases intensity error vs ERA5,
  by significantly more than the random-feature control; AND the non-developing control is ~unaffected.
- **Not necessary / distributed** iff the physical intensity and skill are ~unchanged when convection
  is restored to normal (even if the internal feature moves) — a valid, honest result: the internal
  representation would then be a readout, not a lever on the output.
- Report the split between the INTERNAL effect (feature) and the PHYSICAL/SKILL effect (MSLP/ERA5)
  straight — a large internal effect with a small physical one is itself the finding.

## Compute & ops

- ~6 storms × {baseline, normal-patch, zero, random-ctrl} × 16 steps ≈ ~400 flagship forwards +
  ERA5 truth loads. bf16. **Serialize behind the running GPU jobs** (skill decomp, counterfactual,
  8→9 test) — poll nvidia-smi, start only in a free window. Crash-safe per-storm saves to
  results/skill/convection/. Figures in figures/. Commit code (results gitignored), push.
- The artifact update (folding in physical + multi-storm + skill + co-location) is done by the main
  session, not this agent — the agent produces results + figures.

## Guardrail

Do not tune bars after seeing numbers. If GraphCast doesn't capture a storm's RI in the baseline,
that storm is out (documented), not a knob to turn. A distributed/no-skill-effect result retires the
strong "necessary for skill" claim honestly and is worth reporting.
