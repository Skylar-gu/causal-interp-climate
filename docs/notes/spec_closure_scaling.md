*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Spec — DYNAMICAL CLOSURE (DC) and PARAMETRIC SCALING (PS)

**Written 2026-08-17. Frozen before any of its own numbers exist.**

Necessity says a direction matters. Neither necessity nor sufficiency says the response is
*physics*. These two tests do, and they are the hardest for an artifact to pass, because an
arbitrary direction has no reason to produce a response that satisfies balance relations or
that tracks a physical parameter across its range.

---

## What already exists, measured, before this spec

**Dose scaling is already covered, in two places.** The impulse instrument carries amps
1 / 3 / 9 at a fixed location, and `skill_conv_run.py --MECH_GAINS` sweeps the feature dose
(g=0 ablate-to-normal, g=1 baseline, g>1 amplify) with a matched-gain random control. So the
*intervention-strength* axis is done. What is missing everywhere is the **ambient-parameter**
axis, which is the one theory makes sharp predictions about.

**From the existing impulse latitude sweep (map arms, -60..+60 at 4 longitudes, amp 3):**

| lat | -60 | -45 | -30 | -15 | 0 | +15 | +30 | +45 | +60 |
|---|---|---|---|---|---|---|---|---|---|
| speed m/s | 27.3 | 27.2 | 27.3 | 9.5 | 10.2 | 18.2 | 35.6 | 25.5 | 14.6 |
| r2 | .93 | .96 | .95 | .62 | .33 | .62 | .90 | .94 | .79 |
| peak_amp | 188 | 170 | 89 | 53 | 41 | 50 | 70 | 128 | 196 |

Two readings, and the difference between them is the point of PS below:

- **Coherence tracks |f| and this DOES discriminate.** r2 collapses to 0.33 at the equator and
  recovers to 0.90-0.96 poleward of 30 deg (r = +0.556 vs |sin lat|, p = 0.0004). Coherent
  balanced packets exist where f is large and not where it vanishes.
- **Amplitude rises with latitude but the predictors are COLLINEAR and this does NOT
  discriminate.** Over -60..+60, |sin lat| gives r = +0.898, plain |lat| gives r = +0.921 and
  cos lat gives r = -0.945. The no-physics predictor fits marginally BEST. The trend is real;
  the attribution to Coriolis is not established, and must not be claimed from this sweep.
- **The response is weakly NONLINEAR.** log-log slope of peak amplitude on dose = 1.118, and
  packet speed rises 23.9 -> 27.5 m/s (+15%) across a 9x dose. A linear wave has
  amplitude-independent phase speed, so nonlinear advection is present.

---

## DC — Dynamical closure

**DC-0 (the blocker).** Balance relations need `u`, `v` and `z` on the SAME level from the same
run. Nothing on disk has that: `respop`'s `COARSE` stores exactly two fields per arm (z500 and
the arm's own field), and for `jet250` those are z500 and u250 -- different levels, so nothing
closes. Every DC test below requires a re-run whose only change is **storing the full field
set**, not new machinery.

Required per arm, per lead, full-resolution response (patched minus unpatched):
`z500, u500, v500, t850, z850, w500, q700`, plus `t500` for hydrostatic.

**DC-1 Geostrophic balance.** In the extratropics (|lat| > 25), the balanced part of the
response must satisfy `f·v' ~ (g)·dZ'/dx` and `f·u' ~ -(g)·dZ'/dy`.
*Statistic:* pattern correlation between the response wind and the geostrophic wind implied by
the response geopotential, cos-lat weighted, as a function of lead.
*Bar, declared now:* a real balanced response should reach `rho >= 0.6` by +24 h in the
extratropics. **Control that must FAIL it:** the same statistic computed on the numeric-floor
arm (nf0 vs nf1), which is pure non-determinism and has no reason to be geostrophic. If the
floor also reaches 0.6 the statistic is measuring the ambient state, not the response, and DC-1
is void.
*Second control:* the same statistic in the deep tropics (|lat| < 10), where geostrophy does
NOT hold and rho should be markedly lower. A statistic that is high everywhere is measuring
something else.

**DC-2 Hydrostatic balance.** `dPhi'/d ln p = -R T'`. Between 850 and 500 hPa the thickness
response must match the layer-mean temperature response.
*Statistic:* regression slope of `(z500' - z850')` on `T'_layer`; theory fixes the slope to
`R ln(850/500)/g`. Report the slope with CI, not just a correlation, since the slope is the
falsifiable quantity and the correlation is nearly guaranteed.

**DC-3 Matsuno-Gill.** The Hakim & Masanam test, done from inside. Impulse a tropical heat
source and check for the canonical response: a Kelvin wave propagating EAST of the heating and
twin off-equatorial Rossby gyres to the WEST.
*Statistic:* east/west asymmetry of the z500 response and the latitude of the two extrema
west of the source. *This is the single most diagnostic test in the spec* -- the Gill pattern
is specific enough that no artifact reproduces it by accident.
*Note:* the mode-free impulse instrument is the right vehicle, not a feature group. It has no
mode basis and is already calibrated both sides, so it is immune to every labelling problem in
this project. Features come second, and only if DC-3 passes on the instrument.

**DC-4 (descriptive).** Energy partition between balanced (rotational) and unbalanced
(divergent) components of the response vs lead. A physical response should shed gravity waves
early and settle onto the balanced manifold. No bar; the trajectory is the result.

---

## PS — Parametric scaling

**PS-1 Break the latitude collinearity.** The existing sweep cannot separate `f` from any
monotone function of |lat|. Extend the impulse map to **|lat| = 70, 75, 80, 85**, where
`sin(lat)` saturates (0.94 -> 1.00) while `|lat|` keeps rising linearly and `cos(lat)` collapses
(0.34 -> 0.09). In that band the three predictors diverge sharply and the fit identifies which
one the model is actually following.
*Pre-registered:* if amplitude tracks `|sin lat|` it saturates poleward of 70; if it tracks
`|lat|` it keeps rising. Both outcomes are informative; the current data cannot tell them apart
and this is the cheapest way to make it able to.

**PS-2 Rossby radius, not amplitude.** The sharper prediction is about SCALE, not magnitude:
`L_R = sqrt(gH)/f`, so the response's spatial decay length should fall as `1/|sin lat|`.
*Statistic:* e-folding radius of the response envelope vs latitude, fitted for the equivalent
depth `H`. A recovered `H` in the physical range (roughly 10-500 m for the leading baroclinic
modes) is a strong quantitative fingerprint; scale is much harder to fake than amplitude
because it has a dimensional prediction attached.

**PS-3 Rossby dispersion.** For barotropic Rossby waves `c = u_bar - beta/K^2`, with
`beta = 2*Omega*cos(lat)/a`. Regress measured packet speed on `beta` at fixed wavenumber and on
`K^-2` at fixed latitude. The existing sweep already shows speed is NOT a simple monotone
function of latitude (35.6 m/s at +30 vs 14.6 at +60), which is what a `u_bar` contribution
would do -- so `u_bar` must be measured from the ambient state and removed before the fit,
otherwise the test is confounded by the jet.

**PS-4 Shear dependence for the convection feature group.** The mechanism-side test.
NOTE the correct physics, which the first draft of this line got wrong: shear does NOT
extinguish convection -- shear can organise convection. What it does to a CYCLONE is tilt and
ventilate the vortex so displaced heating no longer projects efficiently onto spin-up. So the
prediction is about EFFICACY, not amount: the ablation should bite LESS at high ambient shear
while the features' ACTIVATION should be roughly unchanged. Both halves must be reported; the
activation null is what makes the efficacy result non-trivial.
*Free first step:* the 7 storms already span a shear range; regress the measured ablation
effect on ambient shear at the initial condition. That is a rescore of existing runs, no GPU.
*Then:* a controlled sweep, choosing restore-to-normal analog days by ambient shear rather than
by date -- the harness already selects analogs, so this is a change of selection criterion.
*Caveat to state up front:* n = 7 is thin. The 7-storm convection-grip hypothesis already
failed at that n (rho = +0.32, p = 0.48), and PS-4 must not be reported as supported on a
similar correlation.

**PS-5 Downshear-left displacement — the directional test.** *(added after PS-4's free half;
it is a stronger test than PS-4 and should be run first.)*

PS-4's free half found the efficacy of the convection ablation falls with ambient shear
(r = -0.734, p = 0.060, n = 7) while the convection features' ACTIVATION does not
(r = -0.236, p = 0.610). So the model reproduces the correct physics: shear does not
extinguish convection, it makes the same convection worth less to the storm. That is the
textbook mechanism -- displaced heating spins up a vortex inefficiently because efficiency is
highest inside the radius of maximum wind, plus vortex tilt and low-entropy ventilation of the
core (the Tang-Emanuel ventilation index).

The observed signature of that mechanism is SPATIAL and DIRECTIONAL: in a sheared storm,
convection concentrates **downshear-left in the Northern Hemisphere** (downshear-right in the
Southern). This is among the most robust composites in TC observations.

*Statistic.* For each storm, take the ambient shear vector `S = V200 - V850` over the storm
disk at the initial condition, and the centroid of the convection features' activation
relative to the storm centre. Report the displacement magnitude and the angle between the
displacement and `S`, measured positive counter-clockwise.

*Pre-registered predictions, all three of which an artifact has no way to satisfy:*
  1. the displacement angle clusters near **+90 deg** (downshear-LEFT) for NH storms
  2. the displacement MAGNITUDE grows with |S|
  3. the sign of the angle FLIPS in the Southern Hemisphere

*Why this beats PS-4.* PS-4 predicts a magnitude and is a 7-point correlation at p = 0.06.
PS-5 predicts a **direction set by an external vector**, and an artifact has no access to the
shear vector's orientation at all. Prediction 3 is the strongest available: a statistic that
reverses sign with hemisphere cannot be produced by any latitude-dependent or magnitude-only
confound.

*Cost.* The shear vectors are already fetched, and the feature footprints are in each run's
stored snapshot, so predictions 1 and 2 are a rescore with no GPU. Prediction 3 needs at least
two SH storms added to the battery -- the current seven are all Northern Hemisphere, so the
hemisphere flip is UNTESTED and must be reported as such until those runs exist.

*Caveat.* n = 7 and all NH. PS-4 already sits at the same n where the convection-grip
hypothesis died (r = +0.32, p = 0.48); neither PS-4 nor PS-5 predictions 1-2 should be reported
as established on seven Northern-Hemisphere storms alone.


---

## Order of work (informativeness per unit time)

1. **PS-4 free half** -- DONE: r = -0.734, p = 0.060 for efficacy vs shear, while activation
   vs shear is r = -0.236, p = 0.610. Correct mechanism, thin sample.
2. **PS-5 predictions 1-2** -- rescore, no GPU. Directional, so far stronger than PS-4, and it
   is the cheapest remaining test in this spec.
3. **DC-1 + DC-2 re-run** -- one job storing the full field set at 4 ICs. Geostrophic and
   hydrostatic closure are the cheapest real closure tests and share one run.
4. **DC-3 Matsuno-Gill on the impulse instrument** -- the most diagnostic single test here.
5. **PS-1 high-latitude extension** -- 4 new latitudes on the existing map machinery.
6. **PS-2 Rossby radius** -- needs the same fields as DC-1, so it rides along with step 3.
7. **PS-5 prediction 3** -- add >=2 Southern-Hemisphere storms; the hemisphere flip is the
   single strongest prediction in this spec and currently cannot be tested at all.
8. **PS-3** last: it needs the ambient jet removed before the fit and is the most confounded.

## What is NOT claimed

- Balance relations holding does not prove the model "understands" physics; it shows the
  response lies on the manifold that physics occupies.
- All of this is one model at one resolution.
- DC and PS test the RESPONSE, not the feature labels. A correctly-labelled feature can produce
  an unbalanced response and an artifact could in principle produce a balanced one; that is why
  the floor and tropical controls in DC-1 are mandatory.
