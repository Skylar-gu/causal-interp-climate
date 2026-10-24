*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Where GraphCast loses to a numerical model, and which features hold the lever

Goal 3 of the three stated for this project — *places / features responsible for lesser
performance than numerical models* — measured, not asserted.

`src/hres_compare.py`, `results/hres_compare.npy`. GraphCast, IFS HRES and ERA5
scored by the **same** storm-box MSLP-minimum statistic over the same +0..96 h window from
the same calendar IC. Four storms: three of the seven predate the 2016 start of the
WeatherBench2 HRES archive.

## The gap

| storm | best track | ERA5 min | GraphCast min | HRES min | GC short by | HRES short by |
|---|---|---|---|---|---|---|
| ida 2021 | 929 | 981.9 | 996.9 | 968.1 | **+67.9** | +39.1 |
| michael 2018 | 919 | 982.2 | 993.5 | 953.7 | **+74.5** | +34.7 |
| haishen 2020 | 910 | 950.1 | 956.9 | 916.9 | **+46.9** | +6.9 |
| goni 2020 | 905 | 986.7 | 992.9 | 954.8 | **+87.9** | +49.8 |

**HRES beats GraphCast on the storm minimum by 36.7 hPa on average (29–40 hPa, 4 of 4).**

## It is not inherited from the initial condition

The obvious objection is that HRES starts from ECMWF's own 9 km analysis, which resolves a
cyclone far better than 0.25° ERA5, so some of its advantage is handed to it at t = 0. The
data closes that off:

| storm | ERA5 IC | HRES IC | IC gap | ERA5 deepening | GC | HRES | GC/ERA5 | HRES/ERA5 |
|---|---|---|---|---|---|---|---|---|
| ida | 1003.7 | 1004.0 | +0.4 | 21.8 | 6.8 | 35.9 | **31%** | 165% |
| michael | 1005.0 | 1005.5 | +0.5 | 22.8 | 11.5 | 51.8 | **50%** | 227% |
| haishen | 983.3 | 978.2 | −5.0 | 33.2 | 26.3 | 61.4 | **79%** | 185% |
| goni | 1005.3 | 1005.7 | +0.4 | 18.7 | 12.5 | 51.0 | **67%** | 273% |

**The two initial conditions differ by −0.9 hPa on average** (−5.0 to +0.5). HRES's box
minimum at t = 0 is essentially as shallow as ERA5's, and it deepens anyway. So the 36.7 hPa
is **forecast, not inherited**.

On deepening measured from each model's own t = 0 — the fair column, since it removes the IC
entirely — GraphCast reaches a median **59% of ERA5's** deepening while HRES reaches **206%**,
a factor **3.5**. GraphCast under-deepens; HRES over-deepens relative to ERA5 and lands much
closer to best track.

## ERA5 is not truth here, and scoring against it inverts the ranking

Best-track minima are 929/919/910/905; ERA5's are 982/982/950/987 — **40–60 hPa too shallow**.
So an RMSE scored against ERA5 rewards a model for being wrong in ERA5's direction. Haishen
makes this concrete: HRES's error *against ERA5* is 27.6 hPa, the worst number in the table,
because HRES forecast 917 hPa when best track was 910 and ERA5 said 950. **The metric
penalises HRES for being right.** Any headline "GraphCast is competitive with HRES" computed
against reanalysis inherits this.

## The link to the intervention results

This is the deficit the convection lever acts on:

- GraphCast captures ~59% of ERA5's deepening, itself well short of reality.
- Ablating the convection features costs **16–65%** of that remaining deepening across seven
  hurricanes (median 19% under restore-to-normal), against ~2% for a firing-rate-matched
  moisture group and ~0% for random controls.
- So the features causally responsible for the deepening GraphCast *does* produce are
  identified, and the model's shortfall against HRES is a shortfall in exactly that channel.

That is the through-line from goal 1 (decompose features, see effects) to goal 3: the failure
mode is a **specific, localisable, causally-tested representational deficit**, not diffuse
blur. Whether *amplifying* those same features closes any of the gap is the sufficiency
question — the asc21 8-storm gain sweep now queued.

## Limits

- **n = 4 storms.** Three of the seven predate the HRES archive. Nothing here is a
  distribution; four of four agreeing is the whole strength of it.
- One statistic (box MSLP minimum), one window, TC intensity only. GraphCast's well-known
  wins are on large-scale fields at medium range, which this does not touch and does not
  contradict.
- HRES is 9 km; GraphCast is 0.25°. Some of the gap is resolution, and this cannot separate
  resolution from representation.
