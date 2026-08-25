*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# The mechanism library, scored: one axis, one void, and six arms that were never tested

2026-08-17. Thirteen ablation arms in `results/skill/*/verdict.json`, run 2026-08-15
16:05–20:03 (`out/mechlib_status.txt`) plus `moisture2` (2026-08-16 13:28–13:55) and
a clean `mech_atm_river` re-run (2026-08-17). Scored by
`flagship_sae/skill_conv_analyze.py` with `MECH_RES=<arm>`. Labels from
`results/fs_mechanisms_v2.npy` (`notes/labeling_repair_2026_08_15.md`).

Protocol per arm, unchanged from the convection arm so the numbers are comparable:
7 developing TCs + 1 non-developing control, four arms each (baseline,
group→normal, group→0, random-group→normal), the counterfactual being
restore-to-normal inside a 1500 km disk. `D_norm` is the median over the 7
developing storms of Δ-deepening in hPa under group→normal.

## The table

`box` is the arm's own exposure: peak-in-time, group-mean activation of the
ablated features **inside the storm box** at baseline
(`skill_conv_analyze.py:78`), median over the 7 storms. It answers "did the
intervention touch the storm at all", and it is the column that reorganises
everything below.

| arm | box | D_norm | D_zero | D_rand | nondev | Δ/box | calibrated identity |
|---|---|---|---|---|---|---|---|
| ~~mech_atm_river as run~~ | 67.90 | ~~+8.191~~ | ~~+8.272~~ | −0.040 | −0.016 | — | **VOID — contains the outcome variable** |
| convection | 39.55 | **+2.794** | +4.153 | +0.013 | −0.010 | 0.071 | 3/3 ascent (+21 to +29σ) |
| mech_ascent | 20.27 | **+2.377** | +3.075 | −0.036 | +0.007 | 0.117 | 3/4 ascent |
| **mech_atm_river, clean** | 25.45 | **+1.393** | +1.657 | −0.005 | +0.012 | 0.055 | 1/3 ascent, 2/3 ambiguous — **identity unverified** |
| mech_vort850 | 3.07 | **+0.553** | +0.824 | +0.031 | −0.016 | 0.180 | 4/4 vort850 |
| moisture | 13.64 | +0.318 | +0.320 | +0.015 | −0.039 | 0.023 | 0/3 — all three are ascent |
| mech_q600 | 8.93 | +0.222 | +0.386 | +0.047 | −0.000 | 0.025 | 3/4 ascent, 1 ambiguous, 0 q600 |
| moisture2 | 2.14 | **−0.032** | −0.077 | −0.000 | −0.004 | −0.015 | **3/3 genuine q600** |
| mech_blocking | 0.00 | +0.040 | +0.026 | −0.059 | −0.021 | — | no exposure (7/7 storms) |
| mech_jet250 | 0.00 | +0.021 | −0.018 | −0.016 | +0.041 | — | no exposure (7/7) |
| mech_shear | 0.00 | +0.001 | +0.063 | −0.026 | +0.021 | — | no exposure (7/7) |
| mech_baroclinicity | 0.33 | −0.026 | −0.094 | −0.005 | −0.023 | — | no exposure (4/7) |
| mech_t850 | 0.00 | −0.027 | +0.010 | +0.042 | −0.015 | — | no exposure (7/7) |
| mech_z500 | 0.00 | −0.031 | −0.013 | +0.006 | −0.002 | — | no exposure (7/7) |

Every random-feature control and every non-developing-storm cell sits inside
±0.06 hPa, which is the run-to-run noise floor of this protocol. That floor is
what makes the bottom six rows readable as zero and the top three as real.

## 1. `mech_atm_river` is VOID: its feature list contained the outcome variable

`MECH_FEATS=3243,1232,1178,1403`, and **3243 is the TC feature** — the readout
the verdict scores (`S.TC`, confirmed as `tc=3243` in every `run_*.npy` in the
repo). The arm therefore ablated the thing it then measured. It produced the
largest effect in the whole library by a factor of three (+8.19 hPa, `tc_supp`
0.985 — near-total suppression of the TC representation) for exactly that
reason. The directory is preserved as `results/skill/mech_atm_river_CONTAMINATED`
and must never be quoted.

This is not a small bookkeeping slip. `skill_conv_run.py:40` asserts the ablation
group is disjoint from the *convection* group, and that assertion fired correctly
all day; nothing asserted disjointness from the **outcome feature**.

The clean re-run on `1232,1178,1403` now quantifies exactly what the contamination
bought: **+8.191 → +1.393 hPa** and `tc_supp` **0.985 → 0.030**. Deleting the
readout supplied 83% of the headline effect and essentially all of the apparent TC
suppression. A guard is now in place (`skill_conv_run.py`, hard assert with no
override, covering both the ablation group and the random control; verified to
fire on `3243,1232` and pass `1232,1178,1403`).

> Guardrail. Assert the ablation group is disjoint from the outcome variable, not
> only from the other treatment groups. An arm that ablates its own readout will
> produce the strongest result in any battery, so this defect is
> self-camouflaging: it looks like the best arm.

The surviving +1.393 hPa is real against its controls (rand −0.005, nondev
+0.012) and is the third-largest effect in the library — but see §3, because it
does not fit the axis that explains everything else, and its group identity is
not verifiable with the current four probes.

## 2. The moisture withdrawal is itself WITHDRAWN — "convection is the lever" is rescued

`labeling_repair_2026_08_15.md` withdrew "convection is the only lever" because
the moisture control arm was three ascent features. The corrected arm has now
run. Genuine calibrated q600 features (2958 / 2671 / 37, +6.7 / +8.1 / +4.8σ on
q600, gaps 4.0–4.6 against bars 1.4–1.7) give **−0.032 hPa** — indistinguishable
from zero and from the random control.

So the conclusion survives, on evidence that now actually bears on it. What the
withdrawn version had was a comparison of strong ascent against weak ascent; what
this is, is a comparison of ascent against moisture, and moisture does nothing.
The claim is restored in the corrected form: **among the four genesis ingredients
GraphCast's dictionary encodes, ascent is the lever on 96 h intensity; mid-level
moisture is not.**

Two honest caveats, both of which limit how far that sentence travels:

- **moisture2's exposure is thin.** box 2.14 against convection's 39.55, and 2 of
  7 storms have no exposure at all. A null with low exposure is weak evidence on
  its own. The load-bearing moisture null is `mech_q600` — box 8.93, real
  exposure on all 7 storms, +0.222 hPa — and it is null *despite* three of its four
  features being ascent-contaminated, i.e. despite being biased toward finding an
  effect.
- **"Moisture" here means specific humidity at 600 hPa.** Not IVT, not
  precipitable water, not θe. The tested quantity is a single-level humidity
  value, so this says nothing about column moisture transport — and that caveat is
  no longer hypothetical. The clean `mech_atm_river` arm, the one group in the
  library aimed at moisture *transport*, gives **+1.393 hPa**, six times
  `mech_q600`. Whether that is really transport or ascent under another name cannot
  be settled with the present four probes (§3), but it is enough to block the loose
  reading. The defensible sentence is the narrow one: **mid-level humidity is
  inert; column moisture transport is not, and is not yet identified.**

## 3. Ascent-loading explains the moisture arms — and fails on two others

Rank the arms by their **median calibrated ascent σ** — not by the mechanism name
they were launched under — and the effect follows monotonically:

| median ascent σ | arm | labels | D_norm |
|---|---|---|---|
| **+28.8** | mech_ascent | 3/4 ascent | 2.38 |
| **+28.5** | convection | 3/3 ascent | 2.79 |
| +13.2 | moisture (the voided arm) | 3/3 ascent | 0.32 |
| +6.0 | mech_q600 | 3/4 ascent, 0/4 q600 | 0.22 |
| +2.1 | moisture2 | 3/3 q600 | −0.03 |
| +3.0 | mech_vort850 | 4/4 vort850 | 0.55 ← off-axis |

Read the first two rows as one point, not two: `mech_ascent` shares features 2401
and 2067 with the convection group (2 of 4, permitted by `MECH_ALLOW_OVERLAP` on
the stated grounds that ascent *is* convection under its purified name). Its 2.38
is therefore a consistency check, not an independent replication — the same
mechanism at the same loading landing within 0.4 hPa of itself.

The informative span is the rest: 13.2 → 0.32, 6.0 → 0.22, 2.1 → −0.03. Effect
decays to nothing as ascent-loading does, across three groups that were launched
as moisture arms and differ from each other only in how much ascent contamination
they carry. Within that family the dose-response is clean.

**But two arms break it, and the clean `mech_atm_river` breaks it hard.** Its
features (1232 / 1178 / 1403) carry median ascent +4.4σ and max +10.1σ — *less*
ascent loading than either moisture arm on either summary statistic — yet it gives
**+1.393 hPa**, four to six times their 0.22–0.32. Whatever is doing that work,
ascent-loading does not predict it. `mech_vort850` is the second exception, at
+3.0σ ascent and +0.553 hPa on 4/4 clean vorticity labels.

So the honest scope of the axis: it explains the moisture family completely, and
it is *not* a general law of the library. Two mechanisms move 96 h intensity
without being ascent-loaded.

Within the moisture family, exposure tracks the same axis, which is the tell.
`mech_q600` fires in the storm box (8.93) **because** three of its four members are
really ascent features; strip them and the exposure collapses to 2.14. So across
those groups ascent-loading predicts both whether a group fires on a hurricane and
how much ablating it costs — two independently measured quantities, one ordering.
That is the strongest evidence so far that the calibrated labels track something
real, and it is exactly why the two exceptions above are worth chasing rather than
explaining away.

`mech_vort850` is the interpretable exception: 4/4 clean vorticity labels, no
ascent, and a real +0.553 hPa. Its Δ/box of 0.180 is the highest in the table —
per unit of activation it is the most efficient lever found — but at box 3.07 that
ratio rests on a small denominator and should be treated as suggestive. The safe
statement is that vorticity is a genuine secondary lever, ~5× weaker in absolute
terms than convection, and the only mechanism besides ascent with any effect
outside the noise floor.

## 4. Six arms are UNTESTED, not null

`z500`, `blocking`, `jet250`, `shear`, `t850` and `baroclinicity` all read |Δ| ≤
0.04 hPa — and all have **zero in-box activation in 7 of 7 storms** (baroclinicity
4 of 7). The ablation was a no-op: it removed nothing, because those features
never fire inside a tropical-cyclone box. Their features are alive globally
(`n_fire` 95–9,141 on a 40,962-node mesh), just silent on TCs.

Reporting these as "not levers" would be the vacuous-control failure again, in a
new costume — a treatment that cannot move the readout is not a negative result.
They are recorded as **no exposure**, and 6 of 13 arms in the library are
therefore uninformative as run. Three GPU-hours of the 2026-08-15 matrix bought
no information about those mechanisms.

That the extratropical probes selected features which are silent on hurricanes is
not a defect — it is the vocabulary behaving correctly, and it locates the fix
precisely. Two things are needed before those six arms mean anything, and they are
independent:

**DIAGNOSED, same day, in the other lane.** `0b7470f` found the cause: global
standardisation in `label_expanded.py:74` makes "most anomalous" mean "polar" for
nearly every field, so six of the ten concepts are Arctic/Antarctic features wearing
mid-latitude names (baroclinicity centroids −84/73/−85/80, jet250 −69/66/−80/−67,
z500 −78/−84/−82/−86, t850 −81/−80/−80/−80). The zero in-box exposure recorded here
is the same fact as that commit's "all six are SILENT inside a hurricane", reached
independently by exposure measurement rather than by centroid. So these six arms did
not merely lack exposure — **they never contained the mechanisms they were named
for**, and blocking/z500/jet250/t850/baroclinicity remain untested as physics.

**SETTLED, same day.** Prerequisite 1 was run on `eastcoast2018`
(`notes/xt_battery_2026_08_17.md`). Four of the six DO gain exposure in a bomb
cyclone — baroclinicity 0.33 → 24.57, jet250 0.00 → 22.08, shear 0.00 → 8.73,
blocking 0.00 → 3.47 — and **all four are null once exposed**, baroclinicity most
cleanly at +0.007 hPa on the largest exposure in the battery. `t850` and `z500`
are unexposed even there and remain untested. Convection stays the only lever
(+0.680, z = 4.26) though 4× weaker than in the tropics. So "untested, not null"
was the right correction and the retest resolves it: exposed, and null.

1. **An extratropical testbed.** `flagship_sae/event_screen.py` already has it:
   `eastcoast2018` at 58.0 hPa/24 h tracked (3.4 Bergeron) and `greatlakes2010`
   at 26.0 hPa (1.3 B), both continuity-clean, against 1.0 B for Ida and Michael.
   Baroclinicity, z500 and jet250 have real exposure in a bomb cyclone by
   construction. This is now the main reason to run the event battery, ahead of
   the vocabulary argument that originally motivated it.
2. **An extended probe battery.** The calibrated instrument carries only
   `[vort850, q600, ascent, shear]`, so `ambiguous` on a z500 or blocking feature
   means "none of those four" — which is the correct answer for a z500 feature,
   not an indictment of it. Those six arms' *selection* is currently unaudited in
   either direction. Extending the rotation null to the extratropical probes (the
   A–H families) is a prerequisite for quoting them, and per the labelling
   guardrail the extension must be run on a null input first.

## What is safe to quote

- Convection/ascent ablation costs 2.79 hPa of median 96 h deepening across 7
  hurricanes, against +0.013 for random controls and −0.010 on the
  non-developing storm. Calibrated 3/3 ascent.
- Vorticity is a real secondary lever at 0.553 hPa, calibrated 4/4.
- Mid-level (600 hPa) specific humidity is not a lever: 0.222 hPa with genuine
  exposure and ascent contamination working in its favour, −0.032 hPa for a pure
  q600 group.
- Within the moisture family, effect size scales with calibrated ascent-loading.
- The clean `mech_atm_river` group moves intensity by 1.393 hPa against controls
  at −0.005 and +0.012 — as an *effect*. Not as a mechanism (see below).

## What is not

- The `+8.191` figure from `mech_atm_river` as originally run (void — outcome
  variable ablated; 83% of it was the readout deleting itself).
- The clean `mech_atm_river` +1.393 as an *atmospheric-river* or
  *moisture-transport* result. The effect is real; the label is not established —
  2 of 3 features are `ambiguous` under the four available probes and the third is
  ascent at +10.1σ. It is currently "an unidentified group that works".
- Anything about blocking, z500, jet250, t850, baroclinicity or shear as
  mechanisms (no exposure on this testbed; selection unaudited).
- Any claim about moisture beyond 600 hPa specific humidity, in either direction.
- "Ascent-loading explains the library" as a general law — it explains the moisture
  family, and two arms violate it.
- Ida-specific numbers as a stand-in for the median: Ida alone gives the
  convection arm 2.794 and the moisture arm 0.536, and the two coincide with the
  medians only by accident.
