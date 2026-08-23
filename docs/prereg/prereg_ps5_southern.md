*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — PS-5 prediction 3, the hemisphere flip

**Written 2026-08-18 before any Southern-Hemisphere storm has been located, run, or scored.**
Committed before the selection script runs.

## The prediction

PS-5 (`notes/spec_closure_scaling.md`) pre-registered three predictions about where the
convection features' activation sits relative to the storm centre, given the ambient shear
vector `S = V200 - V850`:

1. displacement angle clusters near **+90°** (downshear-LEFT) for NH storms — **FAILED**
   (mean +37°, R = 0.36, 5/7 left)
2. displacement magnitude grows with `|S|` — **r = +0.837, p = 0.019** on 7 NH storms,
   leaning on Wilma at `|S| = 0.5`
3. the sign of the angle **FLIPS** in the Southern Hemisphere — **UNTESTABLE**, all seven
   storms are Northern Hemisphere

Prediction 3 is the strongest test in the whole spec. A statistic that reverses sign with
hemisphere cannot be produced by any latitude-dependent or magnitude-only confound, and an
artifact has no access to the shear vector's orientation at all. It is the one test here that
a scrambled control provably cannot pass.

## Selection criteria, frozen now

Storms enter the battery if and only if they satisfy all four. No outcome quantity — no
displacement, no angle, no ablation effect — is computed before the set is fixed.

1. **Southern Hemisphere tropical cyclone**, centre latitude at IC between **8°S and 25°S**.
   The lower bound keeps `f` away from the equatorial band where the impulse instrument
   already showed coherence collapses (r² = 0.33 at 0°, 0.90–0.96 poleward of 30°); the upper
   bound matches the NH battery's 7–22°N.
2. **Agency best-track minimum ≤ 930 hPa**, matching the NH battery's range
   (872–929 hPa) so the convection features are as active as they are there.
3. **IC + 96 h fully inside the WB2 ERA5 zarr**, which ends 2021-12-31.
4. **Basin diversity** — the accepted storms may not all come from one basin.

## Centre and box are taken from ERA5, not from memory

The storm centre and tracking box are located by scanning a wide basin box for the MSLP
minimum at the IC and at +96 h, the same statistic the harness itself uses, rather than typed
from recall. A candidate is **rejected** if ERA5 shows no closed deepening minimum inside the
window — that is a data gate on the input, not a look at the outcome.

## Candidates entering the gate

All six have ERA5 coverage for IC−6 h .. +96 h (verified):

| storm | IC | basin | best-track min |
|---|---|---|---|
| winston 2016 | 2016-02-16 | S Pacific | 884 |
| harold 2020 | 2020-04-01 | S Pacific | 920 |
| fantala 2016 | 2016-04-14 | S Indian | 910 |
| ambali 2019 | 2019-12-03 | SW Indian | 916 |
| marcus 2018 | 2018-03-17 | Australian | 905 |
| veronica 2019 | 2019-03-19 | Australian | 928 |

## The bar

Prediction 3 passes if the mean displacement angle over the accepted SH storms is
**negative** (downshear-RIGHT) where the NH mean is positive, with the two hemispheres'
angle distributions separated.

**Calibration on both sides, as guardrail #9 requires:**
- The null **varies**: the NH angles already span a wide range (R = 0.36, so the NH
  distribution is barely clustered at all). The statistic is not a point mass.
- The bar is **attainable**: a sign flip is one of two outcomes and the SH storms are drawn
  without reference to it.
- A negative control must **FAIL**: the same displacement computed for the firing-rate-matched
  random control group must show no hemisphere dependence. If the random group also flips
  sign, the statistic is reading the ambient flow rather than the convection features, and
  prediction 3 is void.

**Stated in advance:** NH prediction 1 already failed. If the NH angles are only weakly
clustered (R = 0.36), a two-sample test on the hemisphere flip has little power at n = 7 vs
n ≈ 2–3, and a null result will be reported as underpowered rather than as a refutation.
