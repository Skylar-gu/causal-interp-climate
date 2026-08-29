# PREREG — dose–response (gain sweep) for the storm-core spin feature 3316

**2026-08-29.** Committed before the run. Purpose: panel (b) of a "Figure 2.5" that mirrors
Figure 2 with feature 3316 in place of the convection triplet.

## Protocol

Identical to `gain_conv` (`bash_files/run_step23.sh`, STEP 3): `MECH_NAME=gain_3316
MECH_FEATS=3316 MECH_GAINS=0,1.25,1.5,1.75,2,2.5,3` through `skill_conv_run.py` on
`haishen2020 ida2021 patricia2015`. g = 0 is restore-to-normal (the committed ablation arm, a
regression check against `mech_3316` 7.36 / 0.98 / 1.89 hPa); g > 1 scales the excess above
normal inside the 1,500 km disk. Readout: MSLP-minimum RMSE against ERA5 over the
intensification window, exactly `make_figures.gain_curve`. The random control arms
(`rand-normal`, `rand-gain-3`) run as in `gain_conv`.

## Predictions, written before running

- g = 0 reproduces the `mech_3316` ablation within the ±0.15 hPa floor.
- Convection's curve is U-shaped on Ida (7.35 → 3.13 at ×2) and Haishen (4.03 → 3.01 at ×1.25)
  and rises steeply beyond. For a precursor feature the open question is whether amplification
  helps at all: prediction is a *shallower* improvement than convection's, or none, with error
  rising monotonically past ×1.5, because doubling a seed vortex that has already been consumed
  by +48 h (Ida 108 → 14) should have less to act on than doubling the fuel. If 3316's curve
  improves the forecast as much as or more than convection's, spin is a steerable lever, not
  only a necessary ingredient, and the paper's dose–response claim generalizes.
- Patricia (convection ablation null, 0.20 hPa; spin ablation 1.89) is the asymmetry probe.

## Outputs

`results/skill/gain_3316/run_*.npy`, `out/gain_3316.log`; consumed by
`figures/main_claims/build_figure2p5.py` → `figure2p5_web_notitle_print.html` →
`figure2p5_interventions_notitle.pdf` (also copied to `paper_clean/images/`).
