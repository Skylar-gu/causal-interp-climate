*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Rotation confirms the positional reading; the convection specificity claim does not survive

Three runs finished overnight. Two strengthen the grid-lock conclusion; one weakens the
hurricane conclusion.

## 1. Rotation test — decisive, and the calibration passed

Roll every input field, **including the static land-sea mask and orography**, 180° in
longitude: a rigid rotation of the whole planet onto a different part of the mesh. A feature
reading content rotates with the atmosphere; a feature encoding position stays on the mesh.

| feature | stays on mesh | follows the air | verdict |
|---|---|---|---|
| **f2954** (full lattice) | **0.872** | 0.008 | **positional** |
| **f2075** (bowties) | **0.802** | **−0.026** | **positional** |
| f656 (lattice dots) | 0.556 | 0.019 | positional |
| f2235 (polar band) | 0.798 | **0.979** | content |
| f553 (ascent) | −0.002 | 0.478 | content |
| f2474 (land mask) | −0.024 | 0.392 | content |
| f577 | −0.001 | 0.171 | content |
| f2109 | −0.001 | 0.014 | **ambiguous — instrument fails here** |

The calibration features behave as required: the land mask and the ascent feature follow the
rotated planet and do not stay on the mesh. So the instrument works, and **f2075's activation
is uncorrelated with the rotated atmosphere (−0.026) while retaining 0.80 correlation with
where it was.** Rotate the entire planet and it does not move.

That is the test the earlier "coordinate system" claim needed and did not have. It is now
supported. Two caveats: f2235 scores high on **both** because a zonal band is unchanged by a
longitudinal roll, so the test has little power for zonally symmetric features; and f2109
comes out uncorrelated with either, meaning the rotated state was unphysical enough that it
simply fired elsewhere.

## 2. Matched controls — the lattice class is load-bearing beyond its controls

17 features against controls matched on **coverage and connectivity**, 6 paired ICs, +48 h.
Floor arm (empty ablation): +0.0003 ± 0.0171 m.

| class | feature | feature − baseline | control − baseline | difference | p |
|---|---|---|---|---|---|
| B | f2954 | +1.504 | +0.002 | **+1.502** | 0.001 |
| A | f3680 | +1.204 | −0.012 | **+1.216** | 0.005 |
| A | f560 | +0.724 | −0.012 | **+0.736** | 0.001 |
| A | f3191 | +0.225 | +0.004 | **+0.220** | 0.011 |
| A | f656 | +0.129 | +0.009 | **+0.119** | 0.013 |
| C | f2075 | +0.039 | −0.025 | **+0.064** | 0.004 |
| C | f2235 | +0.657 | +2.456 | **−1.799** | 0.006 |
| D | eight features | +0.006 median | — | +0.005 median | 0.14–0.99 |

**Five of six class-A features plus class B are significantly more costly than a
coverage- and connectivity-matched control**, with the controls sitting at zero, inside the
floor. The matching removes the area-dilution confound that made the previous comparison
unreadable.

`f2235` runs the other way: its control is **3.7× more damaging than the feature itself**
(p = 0.006). Consistent with the rotation test calling it content.

Class D remains near-null against matched controls — median +0.005 m, 7 of 8 non-significant.
With dilution now equal on both arms that comparison is fair, so the earlier retraction can be
narrowed: **class D features are individually removable at globally-detectable scale.** For
real weather features that is itself informative — they are redundantly encoded, so no single
one is load-bearing.

**The combined picture.** The three features the rotation test calls positional (f2954, f2075,
f656) are all significantly load-bearing against matched controls. The content features are
not. Positional information appears to have no redundant backup; weather information does.

## 3. Core-matched control — the convection specificity claim fails

The convection result's control had failed three times: matched on global firing rate it fired
at 2% of the treatment's in-box level; matched on climatological in-box firing it fired at 0%
on the actual storm day in four of seven storms. `core_scan.py` measured activation inside
300 km of the storm centre for all 4,096 features, and `core_control.py` drew a control
matched on that, excluding the ascent family and the TC readout.

Only **3 of 7 storms** could be fully matched — no non-ascent feature fires in the core at the
convection group's rate in the other four, which is itself a finding.

| storm | convection | core-matched control |
|---|---|---|
| goni 2020 | 21.4% | **54.8%** |
| haishen 2020 | 29.7% | −7.7% |
| haiyan 2013 | 38.2% | **69.2%** |

**1 of 3 storms has convection above its control.** Median: convection 29.7%, control 54.8%.

When the control actually fires in the storm core it does **as much or more damage** than the
convection group in two of three storms. So the effect is substantially explained by *something
firing in the core*, not by *convection specifically*. This is what
spearman(effect, core activation) = +0.844 across sixteen concept groups already implied, and
it now has a direct test behind it.

**What survives:** the ablation-versus-baseline result (20–40% of deepening lost, 7/7 storms),
the sufficiency result (7/7 deepen, 6/7 move toward ERA5, no-storm control at +0.03 hPa), and
the HRES gap. Those never depended on the random control.

**What does not:** the claim that the effect belongs to the *convection* features rather than
to any core-active direction. At n = 3 this is not a refutation either — it is an
underpowered result pointing the wrong way, and it should be reported as one.

## Power check on the rotation test

A 180° longitudinal roll leaves latitude untouched and maps any 180°-periodic pattern onto
itself. For such a feature "stays on the mesh" is high **for free** and the test proves
nothing. Self-similarity of each original pattern under the same shift:

| feature | self-similarity | stays | follows | |
|---|---|---|---|---|
| f2075 | **−0.017** | 0.802 | −0.026 | test informative |
| f2954 | **0.015** | 0.872 | 0.008 | test informative |
| f656 | **0.012** | 0.556 | 0.019 | test informative |
| f2474, f553, f577, f2109 | 0.005–0.099 | ~0 | 0.17–0.48 | test informative |
| **f2235** | **0.733** | 0.798 | 0.979 | **NO POWER on "stays"** |

The three features the test calls positional all have self-similarity within ±0.02 of zero, so
their patterns are **not** invariant under the roll and "stays = 0.80–0.87" is a real
measurement rather than an artifact of symmetry.

`f2235` is the one exception and it is the zonal-band case anticipated above: its pattern is
73% self-similar under a longitudinal shift, so its "stays = 0.798" was free and carries no
information. Its verdict rests entirely on "follows = 0.979", which remains valid — it does
track the rotated atmosphere.

## What "positional" does and does not mean here

**Supported:** f2075, f2954 and f656 fire as a function of position on the mesh and not of the
atmospheric content sitting there. Rotate the planet, land and orography included, and they do
not move.

**Still not shown:** that they constitute a coordinate *system* — a basis that spans position
and that other features read. That would need the set of such features to tile the sphere with
limited redundancy, and to be shown that downstream computation depends on them as
coordinates. Neither has been tested. "Positional feature" is established; "coordinate
system" remains the interpretation, not the finding.
