*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — CONCEPT RESPONSE OPERATORS (RESPOP-1 .. RESPOP-6)

**Frozen 2026-08-14, before `flagship_sae/respop.py` exists and before any response number
exists.** Uses the purified K = 4 concept groups frozen in `results/fs_cgv2_groups.npy`
(prereg `notes/prereg_concept_graph_v2.md`, d93b4ff). Independent of the CG-v2 verdict: the
groups are a node definition, not a result, so this runs whether or not CG-1 survives.

## Why this shape, and why it is not a graph

Graph discovery is skipped entirely. Instead: perturb concept *i* once, roll GraphCast
forward, and measure the **full spatial response field** in a physical variable over lead
time — a **Green's function**, not a binary edge.

That shape is chosen because of three failures already on record:

- It needs **no conditioning**, so the collinearity that broke PCMCI+ here is irrelevant (the
  flagship SAE basis has 39 modes spanning ~16 effective dimensions, condition number
  592–650).
- It needs **no mode basis** for the readout — the response is read in physical space, on the
  model's own grid.
- It returns **magnitude, timescale and spatial structure** instead of one bit, so a null is
  informative rather than empty.

`probe/impulse_instrument.py` already does exactly this for **locations** and is calibrated
(POS storm track 26.0 m/s R² 0.96; NEG deep tropics 2.2 m/s R² 0.15; estimator-null
false-positive rate 0/500; dose-response linear, 1/3/9 K → 23.9/26.0/27.5 m/s). Its machinery
is for the mini stack and a temperature bump, so it is **reused as discipline, not as code**:
the response-field readout, the *measured* NF-1 numeric floor, and SNR-against-a-floor are
carried over verbatim in spirit. In particular the lesson that **the amp-0 arm is NOT zero** —
the GPU forward is not bit-deterministic (segment-sum atomics) and that float-level difference
grows chaotically through the roll, with a floor that varies ~4× with latitude — is designed
in from the start, not discovered afterwards.

## The intervention (RESPOP-1)

For concept *i*: `coef_patch(sae, features(i), γ = 1.0)` applied at the **first forward only**
— an impulse — then a free-running autoregressive rollout of **S = 10 steps (60 h)** with the
patch off. The response field is the difference against an **unpatched roll from the identical
initial condition**. `coef = 1.0` is +100 % of what those four features are currently doing;
`fs_common.delta` makes the dose proportional to each feature's own activation, so this is the
same intervention the concept graph used, at the same γ.

**Initial conditions (frozen, 4, the seasonal quartet of the concept-graph Set A):**
2020-01-05, 2020-04-06, 2020-07-06, 2020-10-05.

## Arms (RESPOP-2) — 23 rolls per window

| arm | n | what it is |
|---|---|---|
| `base` | 1 | unpatched roll; the reference every difference is taken against |
| `nf0`, `nf1` | 2 | **also unpatched**, identical code path, patch = no-op. Their difference from `base` IS the numeric noise floor, measured at every lead. |
| concept | 10 | the ten purified K = 4 groups |
| `perm` | 10 | the CG-v2 NEG control: the SAME 40 features re-partitioned at random into 10 groups of 4 (seed 0). Matched dose, scrambled labels. |

## Readout fields (RESPOP-3, frozen)

**Primary, common yardstick for all concepts:** `geopotential` at 500 hPa, reported in
geopotential metres (÷ 9.80665), so every concept is measured on the same scale and the
numbers are directly comparable to the repo's existing z500 impulse work.

**Secondary, the concept's own governing field** — fixed map, declared now so it cannot be
chosen later:

| concept | field |
|---|---|
| vort850 | v-wind 850 hPa |
| q600 | specific humidity 600 hPa |
| ascent | vertical velocity 500 hPa |
| shear | u-wind 250 hPa **minus** u-wind 850 hPa |
| t850 | temperature 850 hPa |
| z500 | geopotential 500 hPa |
| jet250 | u-wind 250 hPa |
| blocking | geopotential 500 hPa |
| atm_river | specific humidity 850 hPa |
| baroclinicity | temperature 850 hPa |

## Statistics (RESPOP-4, frozen), all cos-latitude weighted

Per arm, per window, per lead:

1. **Magnitude** — area-weighted RMS of the response field (full 0.25° resolution).
2. **Spatial structure** — `A50` and `A90`: the smallest **area fraction of the globe** that
   contains 50 % / 90 % of the weighted squared response, computed on a 0.5° block-mean.
   A perfectly uniform response gives A50 = 0.50; a point response gives A50 → 0.
3. **Latitude profile** — weighted mean |response| in the bands
   90–60 S, 60–30 S, 30–15 S, 15 S–15 N, 15–30 N, 30–60 N, 60–90 N, plus the |response|
   weighted centroid latitude.
4. **SNR** — RMS(arm) / RMS(numeric floor) at the same lead, floor = mean of `nf0`, `nf1`.

Window-mean signed response fields are stored coarsened to 1.5° so the spatial structure can
be inspected and plotted without a 3 GB file.

## Bars (RESPOP-5) — declared before the run, each with a control that can fail it

- **DETECTED** iff SNR ≥ 3 at some lead ≤ 60 h.
  *Negative control that must FAIL this bar:* `nf1` scored against `nf0` as if it were an arm.
  It is an unpatched roll, so its SNR is ≈ 1 by construction and it must come out
  NOT DETECTED. If `nf1` reaches SNR ≥ 3, the floor is mis-estimated and no arm is read.
  *Vacuity check:* the floor RMS must be **non-zero** at every lead. A zero floor makes the
  bar unfailable and the run is reported as vacuous, exactly as guardrail #9 requires.
- **LOCAL** iff A50 ≤ 0.10 at the lead of maximum SNR; **GLOBAL** iff A50 ≥ 0.35;
  otherwise **INTERMEDIATE**.
- **CONCEPT-SPECIFIC** iff |z| ≥ 2 on either magnitude or A50, where z is taken against the
  **ten perm arms** at the same lead. The perm arms use the same 40 features at the same
  dose, so they hold dose geometry fixed and vary only the labelling.
  *The perm null must be shown to VARY:* its standard deviation across the ten arms is
  reported at every lead. If that spread is ~0 the z-scores are meaningless and the
  specificity bar is reported as vacuous rather than passed.
- **TIMESCALE** — the lead at which SNR is maximal, and the RMS growth ratio between 6 h and
  60 h. Descriptive; no bar, because a single dose cannot separate growth from saturation.

## RESPOP-6 — what is NOT claimed

- No edge, no direction, no graph. A response operator says *concept i moves the atmosphere
  like this*; it does not say concept *i* causes concept *j*.
- Not a forecast-skill claim. The response is measured against the model's own unpatched
  roll, not against ERA5.
- Single dose (γ = 1.0) and single sign (+). No dose-response and no ablation direction; the
  flagship lane already established scale-invariance of interactions over 0.10 ≤ γ ≤ 1.0, and
  this run does not re-test it.
- Flagship-internal, four initial conditions, 60 h. Seasonality of the response is not
  claimed; the four ICs are one per season so that no single season drives the answer, not so
  that season can be estimated.
- A concept whose response is indistinguishable from the perm arms is reported as
  NOT CONCEPT-SPECIFIC. That is a real result about SAE concepts, not a failed run.
