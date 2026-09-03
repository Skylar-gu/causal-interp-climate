# PREREG — single-feature convection battery (2401), for Figure 2.55 panel (a)

**2026-08-30.** Committed before running. Figure 2.55 pairs a single spin feature (3316) against
a single convection feature on the same lollipop plot (two coloured heads per storm), so panel
(a) needs a seven-storm restore-to-normal battery for ONE convection feature, not the triplet —
that data doesn't exist yet (the existing `convection` battery ablates [2401,2067,3174] together).

## Which feature

The Ida single-feature decomposition already run (`results/fs_ida_genesis_v2_followup.npy`,
`notes/result_ida_genesis_calibrated_2026_08_29.md`) gives, on the genesis-knockout metric:
2401 alone -27%, 3174 alone -18%, 2067 alone ~0%. **2401** dominates the triplet, the same role
3316 plays in the vorticity group (-45% alone vs 2514's -10%, 2089's ~0%). Picking 2401 for
symmetry with how 3316 was picked, not a new criterion.

## Protocol

Identical instrument to `mech_3316` (already-calibrated bars: `rand-normal` firing-matched
control, 3x-control/3x-floor bar, `nondev2013` null) — no new methodology, just
`MECH_NAME=mech_2401 MECH_FEATS=2401` through `skill_conv_run.py`, same 8-storm battery
(ida2021, michael2018, haishen2020, goni2020, haiyan2013, patricia2015, wilma2005, nondev2013).
Cost: ~26 min measured on `mech_3316` (8 storms × ~3.2 min).

## Prediction, written before running

Smaller median deepening-restored effect than the full triplet (some of convection's
seven-storm effect is genuinely distributed across the three features, not concentrated the way
3316 concentrates the vorticity group's effect) but still clears the bar on most storms, since
2401 alone already reproduced -27% of the Ida genesis effect versus the triplet's larger
combined number.

## Output

`results/skill/mech_2401/`, `out/mech_2401.log`. Consumed by a new
`figures/main_claims/build_figure2p55.py`.
