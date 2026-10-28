*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Result: Paper 199 revision runs on Rorqual (J1/J3, J2, J4, J6, J7), 2026-10-12

Prereg `prereg_cluster_revision.md` (4a7b8d6) + amendments 1 (J6), 2 (J2), 3 (determinism).
H100, deterministic XLA ops (amendment 3). Code/verdicts: fb11347. ~20 GPU-h total.
J5 (80-storm battery) not run (journal scope; needs the i.i.d. dump to draw controls).

## Instrument

| check | result |
|---|---|
| Staged ERA5 vs committed truth | IC MSLP and ERA5 deepening identical for all 8 storms |
| Gate 2, default XLA | **FAILED**: gain-1 vs baseline 0.11 hPa (Ida). Kept in `suff_ctl_nondet/` |
| Gate 2, deterministic ops | passes bit-for-bit, all 8 NH + 5 SH storms |
| Gate 3, Ida restore-to-normal | 2.90 vs 2.68 hPa committed (Δ 0.22 < 0.37) |
| J4 floor (default XLA, 5 copies) | single arm ≤ 0.125 hPa; difference floor **0.25 hPa** |

## J1: controlled gain sweep (bar: ≥ 5/7 at g = 2 beat the in-box control by > 0.25 hPa)

**Passes at threshold, 5/7, and the pass is fragile.** Deepening-error reduction, treatment − control, g = 2:
Ida +10.6, Wilma +8.4, Michael +5.6, Goni +4.7, Patricia +0.6 | Haishen −4.9, Haiyan −10.6.
- The fifth pass is Patricia (+0.6 vs floor 0.25), pre-declared under-resolved.
- The analyzer first scored min-MSLP trajectory RMSE by mistake. On that metric the result is 4/7.
  The bar was then re-scored on the pre-registered metric, deepening error (`result_sufficiency.md`'s |err|).
  Both are in `rev_verdict.json` (`spec`, `spec_rmse`).
- Failures are overshoots, not control wins. The H100 baseline already reaches 100% (Haiyan) and 80% (Haishen)
  of ERA5 deepening, and g = 2 takes them to 160% and 135%. Haishen improves at g = 1.25–1.5 (+3.2, +5.8).
- Guardrail #9: the control can fail. Haiyan's control costs up to 5.35 hPa, Wilma's 0.66, Goni's 0.39.
  In the other storms it moves ≤ 0.19 hPa.
- Non-developer: exactly 0.00 at every gain. The convection features never exceed normal in its disk,
  so the false-alarm test is **vacuous**, not passed.
- Cross-hardware: baselines differ from the L40S box (e.g. Haiyan 100% vs 80% of ERA5). Only within-run differences are used.

## J3: TC feature 3243 readout (Fig 3)

The post-patch leak at +48 h is small (0–2.5 units). Pre-patch at t+1, g = 2 vs the in-box control at g = 2:
Ida 70.2 vs 52.7, Michael 111.1 vs 92.4, Haishen 368.0 vs 337.4, Goni 136.6 vs 120.9,
Wilma 146.9 vs 140.8, Patricia 10.4 vs 7.3. Haiyan 244.3 vs 242.5 (the control itself raises 3243 there).
The Fig 3 rise survives the pre-patch readout in 6/7 storms.

## J7: Southern Hemisphere (bar ≥ 4/5): passes 5/5

Marcus +9.1, Veronica +4.2, Fantala +2.5, Winston +0.85, Harold +0.48.
Winston is thin: +9.9 at g = 1.5, then it overshoots to 150% of ERA5 at g = 2.
The SH control is near-inert: only Marcus moves it beyond the floor (0.58).

## J2: direction baselines (rule: treatment − median(random) at g = 2, bootstrap CI > 0)

**Not established at g = 2.** Median +3.96 hPa, CI [−5.86, +7.26]; the overshooting storms drive the width.
- Not pre-registered as the rule: at g = 1.5, treatment − random is +2.04 [+0.15, +3.21].
- Random directions at matched per-node magnitude do little in most storms (median ≤ 1 hPa). The exception is Haiyan, where everything overshoots.
- **Difference of means matches the SAE direction:** treatment − DoM at g = 2 is +0.55 [−2.11, +2.44].
  Its cosine with the convection decoders is only 0.0–0.25, and it was built from 1–5 quiet analogues per storm.
  AxBench-style finding: a crude storm-minus-climatology direction is as good a lever as the SAE features.

## J6: feature 2954 ablation type (bar: mean ≥ ½ zero, p < 0.05): passes

z500 at +48 h vs control 3404, paired over 6 ICs:
- zero ablation: +1.39 m (p = 0.002, 6/6; reproduces +1.50 within ±0.3)
- mean ablation: +1.14 m (p = 0.002, 6/6), 82% of zero

**Not a zeroing artefact.** Resample ablation was not run (needs a node-varying patch).

## Not done here

- J1 off-manifold flag: in-disk q99.9 and max of the steered codes are stored per arm and lead (`fields.disk_q`).
  They still need comparing with the i.i.d. 99.9th percentile; the dump is on the L40S box / S3.
- J5 (80 storms): not run.
