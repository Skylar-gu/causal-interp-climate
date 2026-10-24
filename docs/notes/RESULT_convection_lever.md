*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Result: convection features are a causal lever on GraphCast's hurricane intensity forecasts

2026-08-17. This is the project's positive interventional result, stated once with all
of its controls. Everything here is already computed; the note exists because the
evidence was scattered across five files and the claim had never been written down
as a claim.

Artifacts: `results/skill/convection/verdict.json` (treatment + controls),
`results/fs_mechanisms_v2.npy` (calibrated labels),
`results/skill/moisture2/`, `results/skill/xt_*/` (specificity controls).
Scripts: `src/skill_conv_run.py`, `skill_conv_analyze.py`.

## The claim

Ablating three SAE features that calibrate as **ascent** costs GraphCast **2.79 hPa
of median 96-hour deepening across seven hurricanes** and degrades its intensity
forecast skill by **1.61 hPa RMSE** against ERA5, while a firing-rate-matched random
group costs 0.01 hPa and the same ablation on a non-developing storm costs −0.01.

This is a claim about the model's computation, not about the atmosphere: the
intervention is on GraphCast's layer-8 mesh-node representation, and the readout is
GraphCast's own forecast.

## The instrument

- **Model**: flagship GraphCast, 0.25°, 37 levels, 40,962 mesh nodes; SAE (TopK,
  dict 4096, k=32) on layer-8 mesh-node features.
- **Group**: features 2401, 2067, 3174.
- **Counterfactual — restore-to-normal, not delete.** Inside a 1500 km disk around
  the storm, the selected features are *capped* at a normal level measured from five
  quiet same-season analog dates (`fs_common.SAEJax.delta_cond`). This strips the
  anomalous excess and leaves the background, so the model is asked "what if
  convection here were only normal", not "what if this direction were deleted".
  Delete-to-zero is reported alongside and is always larger.
- **Readout**: MSLP minimum in the storm box at 6-hourly leads to +96 h;
  Δ-deepening = (IC − min) baseline minus (IC − min) ablated. Also 10 m wind, and
  intensity-error RMSE against the ERA5 trajectory over the intensification window.

## The result

| storm | ERA5 deepening | GraphCast baseline | Δ ablate→normal | Δ ablate→zero | Δ random ctrl |
|---|---|---|---|---|---|
| haishen2020 | 33.21 | 26.33 | **+7.474** | +17.145 | −0.115 |
| haiyan2013 | 22.29 | 22.54 | **+7.516** | +9.173 | −0.156 |
| wilma2005 | 48.00 | 22.88 | **+4.382** | +10.503 | −0.119 |
| ida2021 | 21.81 | 6.80 | **+2.794** | +3.464 | +0.050 |
| goni2020 | 18.66 | 12.46 | **+2.146** | +3.548 | +0.370 |
| michael2018 | 22.79 | 11.49 | **+2.145** | +4.153 | +0.158 |
| patricia2015 | 19.64 | 5.48 | **+0.203** | +0.896 | +0.013 |
| **median** | | | **+2.794** | **+4.153** | **+0.013** |

Median Δ-error +1.61 hPa (skill gets worse), median TC-feature suppression 0.121.

The spread is real and is part of the result: the effect runs 0.20–7.52 hPa. It is a
median, not a uniform effect, and `patricia2015` is nearly null. Any statement of
this result that quotes only the median is incomplete.

## Four controls, and what each one rules out

**1. Firing-rate-matched random features — rules out perturbation magnitude.**
Three features matched to the convection group's firing rates, with tropical
centroids so they *can* fire on the storm, frozen with seed 7 before results:
**+0.013 hPa** against 2.794, a factor of 215. Perturbing three arbitrary features
in the same disk under the same counterfactual does nothing.

**2. A non-developing storm — rules out generic rollout degradation.**
The same ablation on a tropical wave that stays weak: **−0.010 hPa**. The
intervention does not damage forecasts in general; it costs deepening only where
there is deepening to cost.

**3. Calibrated labels — rules out mislabelled selection.**
Under a rigid-longitude-rotation null (127 rotations; footprint size, shape and
latitude band preserved, only zonal alignment destroyed), all three features label
as **ascent** at **+28.5 / +29.1 / +21.7 σ** with gaps of 16–22 against
null-derived bars of ~1.3–1.4. The rule is calibrated: on a pure-null input it
labels 2.7% against a 5% target, where the uncalibrated predecessor labelled 32%.
The group is what it is claimed to be.

**4. Genuine moisture, and exposure-matched non-mechanisms — rules out
"any strongly-firing group would do this".** This is the control the result most
needed, and it now exists in two forms:

| group | in-box exposure | Δ-deepening |
|---|---|---|
| convection (3/3 ascent) | 39.55 | **+2.794** |
| genuine q600 moisture (3/3 q600) | 2.14 | −0.032 |
| mech_q600 (moisture as launched; 3/4 really ascent) | 8.93 | +0.222 |

and, on a January bomb cyclone where exposure can be matched directly:

| group | in-box exposure | Δ-deepening |
|---|---|---|
| baroclinicity | 24.08 | +0.007 |
| convection | 23.10 | **+0.680** |
| jet250 | 22.05 | +0.235 |

At **matched exposure** a different group costs ~100× less. The effect is not about
how hard a group fires in the storm box.

## Supporting evidence: the effect is graded, not binary

Across five arms the effect is monotone in the *calibrated* ascent-loading of the
ablated group:

| median ascent σ | arm | Δ-deepening |
|---|---|---|
| +28.5 | convection | 2.79 |
| +13.2 | moisture-as-launched | 0.32 |
| +6.0 | mech_q600 | 0.22 |
| +2.1 | moisture2 (pure q600) | −0.03 |

A calibrated label derived from a rotation null on i.i.d. sampled windows predicts
the *magnitude* of an intervention effect in held-out hurricane forecasts. That is a
stronger statement than the labels being descriptive, and nothing in the labelling
pipeline was fitted to intervention outcomes. Three further arms at 20.6 σ, 16.8 σ
and 9.0 σ are running to turn four points into a curve.

## Limitations, stated

- **Seven storms, one model, one readout.** Median over 7 with an 0.20–7.52 spread.
- **Restore-to-normal is a strong intervention.** It removes the anomalous excess
  over a 1500 km disk for the whole rollout. It is the honest version of the
  counterfactual, but it is not small.
- **Regime-specific.** On an extratropical bomb cyclone the same group costs
  +0.680 hPa (z = 4.26 against a repeated-identical-intervention null), 4× weaker,
  while retaining 58% of its exposure. "Convection is the lever" is a statement
  about tropical cyclone intensification.
- **Moisture is excluded only at 600 hPa.** The tested quantity is mid-level
  specific humidity; column moisture transport is untested. The one arm aimed at it
  (`mech_atm_river`, clean re-run) gives +1.393 hPa, but that thread is weaker than
  it first looked: `0b7470f` showed the arm is mislabelled *twice over* — its
  detector is q850·|V850|, moisture FLUX, which is maximal in a hurricane, so
  selecting on it selects TC features, and feature 3243 scored highest on it because
  3243 IS the TC feature. The confound is in the selection instrument, not only in
  the one feature that was dropped, so the surviving group may still be TC-ish rather
  than transport-ish. Open, but it does not currently reopen moisture.
- **Three features.** The claim is about this group, and the dose-response is what
  argues it generalizes along the ascent axis rather than being three special
  features.

## What was withdrawn on the way here, and why it matters

The claim "convection is the ONLY lever" was **withdrawn** on 2026-08-15 when the
calibrated labelling showed the moisture control arm was three *ascent* features —
so the original comparison was strong-ascent against weak-ascent, not convection
against moisture. It was **restored** on 2026-08-16 when a genuine q600 group ran at
−0.032 hPa. The claim in this note is the restored one, resting on the corrected
comparison. `notes/labeling_repair_2026_08_15.md` and
`notes/mechanism_library_2026_08_17.md` carry the full record.
