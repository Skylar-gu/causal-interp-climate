*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Sufficiency — amplifying the convection features deepens the storm, and reduces the error

**Run 2026-08-18, 16:31 UTC.** `MECH_GAINS=0,1,2,3` on the frozen convection group
[2401, 2067, 3174], eight storms, +96 h. `results/skill/suff_convection/`.

Necessity says a direction matters. This asks the other half: if those features are *amplified*
rather than removed, does the storm deepen — and does it deepen **toward ERA5** rather than
merely more?

## The dose-response is monotone and one-sided

Change in deepening vs baseline, seven developing storms:

| gain | median | range | storms deepened |
|---|---|---|---|
| g = 0 (ablate to normal) | **−2.84 hPa** | −6.85 .. −0.10 | 0/7 |
| g = 1 (must be a no-op) | **−0.10** | −0.22 .. +0.10 | 1/7 |
| g = 2 | **+8.37** | +0.50 .. +17.21 | **7/7** |
| g = 3 | **+18.79** | +1.42 .. +27.89 | **7/7** |
| rand-normal | −0.02 | −0.28 .. +0.71 | 3/7 |
| rand-gain-3 | −0.02 | −0.78 .. +0.88 | 2/7 |

`g = 1` returning −0.10 hPa is the instrument calibrating itself: the gain arm at unity is
bit-equivalent to baseline, so the ladder is measuring dose and not a coding artifact.

## It moves the forecast toward ERA5, not just further

More deepening is not better if it overshoots. Deepening as a fraction of ERA5's, and the
absolute error it implies:

| storm | ERA5 | base % | g2 % | \|err\| base | \|err\| g2 | |
|---|---|---|---|---|---|---|
| ida 2021 | 21.8 | 42% | 88% | 12.55 | **2.51** | helps |
| wilma 2005 | 48.0 | 47% | 64% | 25.41 | **17.04** | helps (g3 → 0.49) |
| goni 2020 | 18.7 | 49% | 72% | 9.59 | **5.26** | helps |
| michael 2018 | 22.8 | 49% | 71% | 11.59 | **6.56** | helps |
| haishen 2020 | 33.2 | 67% | 119% | 10.83 | **6.37** | helps |
| patricia 2015 | 19.6 | 30% | 33% | 13.75 | **13.25** | barely |
| haiyan 2013 | 22.3 | 80% | 128% | 4.51 | 6.31 | **overshoots** |

**g = 2 reduces the deepening error in 6 of 7 storms, median 4.46 hPa.** The one failure,
haiyan, is the storm that was already closest at 80% — a flat gain necessarily over-corrects
where the model is nearly right. That is the same argument that motivated the lead-dependent
ramps, and it is visible here as a pattern rather than an anecdote: the four storms starting
at 42–49% all improve, and the two starting at 67–80% both end up past 100%.

Wilma is the sharpest single case: 47% of ERA5 at baseline, 99% at g = 3, error 25.41 → 0.49.

## Two controls, and only one of them is any good

**Strong — the non-developing storm.** `nondev2013` is a tropical Atlantic wave that stayed
weak, run through the identical pipeline in a real box with real features:

    baseline 0.85   g=0 0.86   g=1 0.82   g=2 0.84   g=3 0.88

**+0.03 hPa at g = 3.** Amplifying the convection features three-fold over a region with no
storm to intensify produces nothing. This control is immune to the in-box defect below,
because the features are the same ones — what differs is that there is no cyclone.

**Weak — the matched-gain random arm.** `rand-gain-3` moves the storms by a median −0.02 hPa,
which looks decisive until you check what it ablates. That group is matched on **global**
firing rate and fires at a median 2% of the convection group's level inside these storm boxes
— 3% in ida, 0% in michael, wilma and patricia. Amplifying features that barely fire in the
core cannot move the storm at any gain, so **this control could not have gone the other way**
and by guardrail #9 it is not evidence. `results/inbox_control_*.json` redraws it at 87–112%
in-box and the re-controlled sweep is queued as `suff_inboxctl`.

## What is established and what is not

**Established:** the direction is *sufficient* as well as necessary. Amplification produces
a monotone, one-sided, 7/7 deepening response, it is absent over a storm-free region, and at
g = 2 it reduces the error against ERA5 in 6 of 7 storms.

**Not established:** that g = 2 is a usable gain. It was picked as the best of three
non-trivial rungs *after* seeing the sweep, so "6 of 7 improved" is an in-sample statement
about the ladder, not an out-of-sample forecast improvement. What the sweep genuinely shows is
that the error minimum lies between g = 1 and g = 3 and that its location is **storm-dependent**
— wilma wants g = 3, haiyan wants less than 2. A fixed gain cannot serve both, which is what
the lead-dependent ramps now running are for.

**Also not established:** anything beyond box-minimum deepening at 0.25° against ERA5, which
is itself 40–60 hPa too shallow for these storms. Patricia responds barely at all (30% → 33%
at g = 2, 37% at g = 3) and is the standing counterexample — it also lost only 4% under
ablation, so it is consistently the storm this lever does not reach.
