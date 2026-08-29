# PREREG — seven-storm battery for the Ida spin group (2089 / 2514 / 3316)

**2026-08-29.** Committed before the run. Follows directly from
`notes/result_ida_genesis_calibrated_2026_08_29.md`.

## Why

On Ida's internal cyclone feature, the calibrated low-level-spin group 2089/2514/3316 is a
handle as strong as convection (−54 % / +51 %), and 83 % of that is feature 3316 alone. The
multi-storm *physical* number the paper quotes for low-level spin (`mech_vort850`, +0.553 hPa
median deepening removed) was measured with the polar group 3861/2514/2089, which on Ida gives
−11 % — i.e. the paper's spin number is an exposure-limited group. Whether spin is a second
physical-readout lever across storms is untested.

## Protocol

Identical to `mech_vort850` and `convection` (1,500 km disk, restore-to-normal from quiet
same-season analogs, delete-to-zero alongside, firing-rate-matched random group, seven
developing TCs + `nondev2013`): `MECH_NAME=mech_spin3316 MECH_FEATS=2089,2514,3316` through
`skill_conv_run.py`, scored by `skill_conv_analyze.py`. Group is disjoint from the convection
triplet (asserted by the script) and from 3243 (asserted). Nothing else changes.

If time allows, a second battery `MECH_NAME=mech_3316 MECH_FEATS=3316` (the single feature),
same protocol.

## Bars (the mechanism library's, unchanged)

- Run-to-run floor of this protocol: ±0.06 hPa on the random-control and non-developer cells
  (`mechanism_library_2026_08_17.md`). An arm is readable only if its random control and
  non-developer cells stay inside ±0.06.
- Reference points, same protocol: convection **+2.794**, mech_ascent +2.377, mech_vort850
  (polar group) **+0.553**, moisture2 −0.032.

## Predictions, written before running

- If the Ida result generalizes: `D_norm` for mech_spin3316 **> 0.553** (beats the polar group)
  and > 3 × 0.06 = 0.18. The interesting threshold is **≥ 1.0 hPa**, at which spin joins
  convection as a multi-storm lever and the paper's "much smaller response (0.55 hPa)" needs
  rewriting.
- If `D_norm` ≤ 0.553: the Ida spin effect is Ida-specific (or internal-readout-specific) and the
  paper's sentence stands with a one-storm caveat.
- Per-storm exposure (`conv_box`) is reported alongside; a null with exposure < 5 is "no
  exposure", not null. Sign consistency across the seven storms is reported.

## Outputs

`results/skill/mech_spin3316/` (+ `mech_3316/`), `out/mech_spin3316.log`, result appended to
`notes/result_ida_genesis_calibrated_2026_08_29.md` as a dated section.

## CORRECTION (2026-08-29, after the run; the prereg text above is left as written)

"the polar group 3861/2514/2089" in *Why* is wrong: that is the Ida-genesis group. The
`mech_vort850` battery used 2822/2935/1148/2089 (two anti-signed, all polar, exposure 0.9–5.4).
Bars and predictions are unaffected (the 0.553 reference value is the same battery).
