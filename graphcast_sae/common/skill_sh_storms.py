"""Southern-Hemisphere storm battery, for PS-5 prediction 3 -- the hemisphere flip.

SEPARATE MODULE ON PURPOSE. The extratropical battery lives in `skill_xt_storms` so that
appending a storm can never silently move the committed TC medians; the same applies here.
Nothing in this file can change a number already reported from `skill_conv_storms`.

WHY THESE FIVE. Selection criteria were frozen in `docs/prereg/prereg_ps5_southern.md` and
committed BEFORE any candidate was located: Southern Hemisphere, centre
8-25 S at IC, agency best-track minimum <= 930 hPa, IC+96 h inside the WB2 ERA5 zarr (which
ends 2021-12-31), and basin diversity.

TWO AMENDMENTS, both made before any outcome quantity was computed and both recorded here:

  A1  IC DATE. The prereg fixed centre and box by an ERA5 MSLP scan but said nothing about
      the IC. My first-guess ICs were mistimed relative to the storms' rapid intensification
      -- Winston deepened only 6.6 hPa from 2016-02-16 and Fantala 2.1 hPa from 2016-04-14.
      A uniform rule was applied to ALL SIX candidates identically: IC = the 00Z date within
      +-5 days of the first guess that maximises ERA5 +96 h deepening. This reads MSLP only.

  A2  DEEPENING MATCH. The NH battery spans 18.7-33.2 hPa of ERA5 deepening. Since PS-4 and
      PS-5 both concern behaviour in an ACTIVELY deepening storm, an SH storm whose ERA5
      deepening falls far below that range is not a matched test. ambali2019 was REJECTED at
      9.4 hPa -- half the NH floor -- and fantala2016 at 15.3 hPa is carried as a
      PRE-DECLARED SECONDARY, reported separately so it can never be swapped into the primary
      set after results exist. It is kept at all because it is the only S-Indian candidate
      and dropping it would leave the primary set spanning two basins.

CENTRE AND BOX come from the ERA5 MSLP-minimum track, not from recall. Box = track extent
+ 6 deg lat / 8 deg lon, clipped to the basin box.

DATELINE. Winston's track extent would put its east edge at 183.2 deg E. `run_storm` tests
mesh nodes with a raw comparison against box["lon"] on a -180..180 grid, which returns an
EMPTY box for any dateline-crossing box. Winston's storm sits at 174-175 E, so the east edge
is trimmed to 179.9 -- this loses no part of the track and avoids the wrap entirely.

Paper: not in the paper (Southern-Hemisphere registry)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.common.skill_sh_storms
"""
import numpy as np

from graphcast_sae.common.skill_conv_storms import CONV, RADIUS_KM, RANDOM_CTRL, TC, norm_lon  # noqa: F401

H = 16  # +96h, same as the NH battery

STORMS = {
    # --- primary set: ERA5 deepening inside the NH battery's 18.7-33.2 hPa range ---
    "winston2016": dict(
        ic="2016-02-21", center=(-17.2, 175.2), box=dict(lat=(-25.0, -11.2), lon=(170.0, 179.9)),
        analogs=["2006-02-21", "2009-02-21", "2013-02-21", "2014-02-21", "2015-02-21"],
        basin="spac", bt=884, era5_deepen=20.7),
    "harold2020": dict(
        ic="2020-04-01", center=(-9.5, 155.0), box=dict(lat=(-21.5, -8.0), lon=(150.0, 173.2)),
        analogs=["2006-04-01", "2009-04-01", "2013-04-01", "2014-04-01", "2015-04-01"],
        basin="spac", bt=920, era5_deepen=23.4),
    "marcus2018": dict(
        ic="2018-03-18", center=(-14.2, 128.0), box=dict(lat=(-21.2, -8.0), lon=(110.0, 140.0)),
        analogs=["2006-03-18", "2009-03-18", "2013-03-18", "2014-03-18", "2015-03-18"],
        basin="aus", bt=905, era5_deepen=27.9),
    "veronica2019": dict(
        ic="2019-03-19", center=(-14.0, 121.0), box=dict(lat=(-24.8, -8.0), lon=(108.8, 129.0)),
        analogs=["2006-03-19", "2009-03-19", "2013-03-19", "2014-03-19", "2015-03-19"],
        basin="aus", bt=928, era5_deepen=26.7),
    # --- pre-declared SECONDARY: below the NH deepening range, reported separately ---
    "fantala2016": dict(
        ic="2016-04-17", center=(-11.0, 53.5), box=dict(lat=(-18.0, -8.0), lon=(45.0, 63.0)),
        analogs=["2006-04-17", "2009-04-17", "2013-04-17", "2014-04-17", "2015-04-17"],
        basin="sind", bt=910, era5_deepen=15.3, secondary=True),
}

PRIMARY = [k for k, v in STORMS.items() if not v.get("secondary")]
SECONDARY = [k for k, v in STORMS.items() if v.get("secondary")]
REJECTED = {"ambali2019": "ERA5 deepening 9.4 hPa, half the NH battery's 18.7 hPa floor"}
