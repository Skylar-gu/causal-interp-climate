"""Mega tropical-cyclone battery -- 80 ERA5-gated storms, all basins, 1986-2020.

SEPARATE MODULE ON PURPOSE, for the same reason skill_sh_storms is separate: appending a
storm here can never silently move a median already reported from skill_conv_storms or
skill_sh_storms. Nothing in this file can change a committed number.

WHY IT EXISTS. The convection-necessity design had 13 storms. IC offsets do not buy
independent draws -- the same storm at -48 h and +0 h shares its synoptic history -- so the
only way to power the design is more storms.

GENERATED, NOT WRITTEN. This file is emitted by graphcast_sae/legacy/mega_registry.py from
results/mega_storm_gate.json, which is produced by graphcast_sae/legacy/mega_sweep.py. Do not edit
it by hand; edit the sweep and re-emit. Every storm below was FOUND by an ERA5 MSLP sweep
and gated on ERA5 alone. No storm was placed from recall, and no name was assigned from
memory -- keys are basin + IC date, because a misremembered name is a silent error and a
date is checkable.

THE GATE, in full (graphcast_sae/legacy/mega_sweep.py):
  detection   MSLP minimum of its +-2.5 deg neighbourhood on a 0.5 deg subgrid, < 1008 hPa
              and >= 2 hPa below that neighbourhood's maximum -- a depression, not a trough
  track       greedy nearest-neighbour linking at <= 400 km / 6 h; >= 96 h of track required
  IC          the 00Z time on the track maximising 96-h deepening (amendment A1 of the SH
              battery, applied uniformly), with IC-48 h .. IC+96 h inside the WB2 zarr
  tropical    genesis latitude 2-28 deg, genesis and IC centres over ocean (land_sea_mask),
              |lat| <= 42 anywhere in the window, centre >= 1.5 deg inside the basin domain
  developing  full-resolution box deepening >= 18.7 hPa (>= 12.0 hPa admitted as SECONDARY),
              ERA5 minimum <= 995 hPa, centre translates > 100 km
  continuity  the full-resolution box minimum stays within 500 km of the tracked centre at
              every lead, and its deepening does not exceed the tracked deepening by > 8 hPa
              -- the event_screen.py lesson, that an advective low moving into the box reads
              as spectacular development
  box         track extent + 5 deg lat / 6 deg lon, grown until it holds >= 140 mesh nodes,
              never crossing the dateline (a wrapping box selects ZERO mesh nodes under
              skill_conv_run.py's raw -180..180 comparison -- the winston2016 failure mode).
              box_nodes below is measured with that exact comparison.
  no overlap  rejected if within 10 days and 2000 km of any of the 13 existing entries, and
              de-duplicated against each other at 7 days / 1500 km

WHAT WAS CALIBRATED AND FAILED. A warm-core gate (300-850 hPa thickness anomaly, centre
minus a 500-1000 km ring) was built to exclude subtropical and hybrid lows, and calibrated
on both sides first. The 12 tropical cyclones already in the battery span 26.4-110.1 m
five explosive EXTRATROPICAL cyclones -- the negative control -- span -182.3 to 101.6, with
three of five ABOVE the TC median. No threshold separates them, so the gate was DROPPED
rather than tuned. `warm_core_m` is recorded per storm in the gate JSON so the question
stays auditable. The tropical requirement rests on genesis latitude, ocean genesis and the
basin domains instead.

ANALOGS. Quiet same-calendar-date dates in other years, which set the restore-to-normal
level. The model-side screen (TC feature sum > 20 in the box) cannot be run without a GPU,
so candidates are pre-screened on ERA5 box-minimum MSLP >= 1004 hPa and offered
QUIETEST-FIRST. Calibration on the 65 analog dates of the shipped runs: every analog the
model accepted had ERA5 box-min >= 1001.1 hPa, and every one at <= 999 hPa was rejected --
but between 1001 and 1008 the model still rejects some, which is how ida2021 ended with 1
surviving analog of 5. Two mitigations, both applied here: boxes are far smaller than
ida2021's 839 nodes, and 8 analogs are offered per storm instead of 5.

The cost of quietest-first, stated rather than hidden: picking the 8 quietest dates of a
~42-year pool is a mild selection toward suppressed conditions, which would lower the
restore-to-normal level and so RAISE the apparent ablation dose relative to the existing
battery's arbitrary fixed years. It is accepted deliberately, because the failure it trades
against -- a storm reaching the GPU with one surviving analog, as ida2021 did -- is worse.
Every candidate's ERA5 box-min is in `analog_mslp` in the gate file, so a later run can
re-draw analogs on a different rule without re-sweeping.

ERA5 deepening in this battery: 13.9 - 67.9 hPa (median 33.8); the NH
battery spans 18.7 - 33.2 and the SH primary set 20.7 - 27.9.

Generated 2026-08-20 20:52 from results/mega_storm_gate.json.

Paper: not in the paper; kept for provenance only
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.skill_mega_storms
"""
import numpy as np  # noqa: F401

from graphcast_sae.common.skill_conv_storms import CONV, RADIUS_KM, RANDOM_CTRL, TC, norm_lon  # noqa: F401

H = 16  # +96 h, same as every other battery

STORMS = {
    # ERA5: 999.8 -> 955.9 hPa at +90 h, deepening 43.9 hPa; genesis 18.0,-87.0; box 275 mesh nodes; 8 analogs offered
    "atl1995_1001": dict(
        ic="1995-10-01", center=(21.5, -90.0),
        box=dict(lat=(16.0, 35.0), lon=(-98.5, -81.0)),
        analogs=["1984-10-01", "2007-10-01", "2016-10-01", "2020-10-01", "1988-10-01", "1989-10-01", "2003-10-01", "1991-10-01"],
        basin="atlantic", era5_deepen=43.9, box_nodes=275),
    # ERA5: 1005.5 -> 962.5 hPa at +96 h, deepening 43.0 hPa; genesis 14.0,-50.5; box 344 mesh nodes; 8 analogs offered
    "atl1997_0906": dict(
        ic="1997-09-06", center=(17.5, -59.0),
        box=dict(lat=(12.5, 34.0), lon=(-69.5, -53.0)),
        analogs=["1985-09-06", "2019-09-06", "1999-09-06", "2001-09-06", "1984-09-06", "1998-09-06", "2009-09-06", "2000-09-06"],
        basin="atlantic", era5_deepen=43.0, box_nodes=344),
    # ERA5: 1002.2 -> 956.5 hPa at +96 h, deepening 45.7 hPa; genesis 16.5,-51.0; box 437 mesh nodes; 8 analogs offered
    "atl1999_0910": dict(
        ic="1999-09-10", center=(18.5, -57.5),
        box=dict(lat=(13.5, 29.5), lon=(-80.0, -51.5)),
        analogs=["2019-09-10", "2001-09-10", "2020-09-10", "2012-09-10", "1993-09-10", "1983-09-10", "2016-09-10", "1998-09-10"],
        basin="atlantic", era5_deepen=45.7, box_nodes=437),
    # ERA5: 1000.5 -> 935.6 hPa at +84 h, deepening 64.9 hPa; genesis 26.0,-79.0; box 285 mesh nodes; 8 analogs offered
    "atl2005_0826": dict(
        ic="2005-08-26", center=(26.0, -80.0),
        box=dict(lat=(19.5, 35.0), lon=(-95.5, -74.0)),
        analogs=["1983-08-26", "1990-08-26", "1994-08-26", "1987-08-26", "1991-08-26", "1984-08-26", "2018-08-26", "1980-08-26"],
        basin="atlantic", era5_deepen=64.9, box_nodes=285),
    # ERA5: 1004.0 -> 961.9 hPa at +96 h, deepening 42.1 hPa; genesis 14.0,-68.0; box 421 mesh nodes; 8 analogs offered
    "atl2008_0828": dict(
        ic="2008-08-28", center=(19.0, -75.0),
        box=dict(lat=(12.5, 31.5), lon=(-93.5, -69.0)),
        analogs=["1993-08-28", "1992-08-28", "2004-08-28", "1985-08-28", "1989-08-28", "1996-08-28", "2002-08-28", "2018-08-28"],
        basin="atlantic", era5_deepen=42.1, box_nodes=421),
    # ERA5: 1003.5 -> 961.7 hPa at +96 h, deepening 41.8 hPa; genesis 12.0,-31.5; box 648 mesh nodes; 8 analogs offered
    "atl2009_0816": dict(
        ic="2009-08-16", center=(11.5, -35.0),
        box=dict(lat=(6.5, 25.0), lon=(-64.5, -29.0)),
        analogs=["1991-08-16", "2014-08-16", "1999-08-16", "2019-08-16", "2018-08-16", "1996-08-16", "1997-08-16", "2002-08-16"],
        basin="atlantic", era5_deepen=41.8, box_nodes=648),
    # ERA5: 994.6 -> 944.1 hPa at +96 h, deepening 50.4 hPa; genesis 16.5,-32.5; box 326 mesh nodes; 8 analogs offered
    "atl2010_0913": dict(
        ic="2010-09-13", center=(17.5, -47.5),
        box=dict(lat=(12.5, 27.0), lon=(-64.5, -41.5)),
        analogs=["2021-09-13", "1983-09-13", "1991-09-13", "2011-09-13", "2015-09-13", "1993-09-13", "1994-09-13", "2005-09-13"],
        basin="atlantic", era5_deepen=50.4, box_nodes=326),
    # ERA5: 1002.9 -> 959.4 hPa at +96 h, deepening 43.5 hPa; genesis 17.0,-62.0; box 456 mesh nodes; 8 analogs offered
    "atl2011_0822": dict(
        ic="2011-08-22", center=(18.0, -65.0),
        box=dict(lat=(13.0, 32.5), lon=(-83.5, -59.0)),
        analogs=["2012-08-22", "1999-08-22", "1995-08-22", "2014-08-22", "2019-08-22", "2004-08-22", "2018-08-22", "1985-08-22"],
        basin="atlantic", era5_deepen=43.5, box_nodes=456),
    # ERA5: 1003.5 -> 959.2 hPa at +84 h, deepening 44.2 hPa; genesis 16.5,-61.0; box 373 mesh nodes; 8 analogs offered
    "atl2014_1014": dict(
        ic="2014-10-14", center=(18.0, -63.0),
        box=dict(lat=(13.0, 35.0), lon=(-75.0, -57.0)),
        analogs=["2018-10-14", "1993-10-14", "2002-10-14", "1991-10-14", "1986-10-14", "1989-10-14", "1988-10-14", "1999-10-14"],
        basin="atlantic", era5_deepen=44.2, box_nodes=373),
    # ERA5: 1007.0 -> 963.2 hPa at +96 h, deepening 43.8 hPa; genesis 11.5,-52.0; box 440 mesh nodes; 8 analogs offered
    "atl2017_0917": dict(
        ic="2017-09-17", center=(12.5, -53.5),
        box=dict(lat=(7.5, 24.0), lon=(-73.5, -47.5)),
        analogs=["1981-09-17", "1991-09-17", "2007-09-17", "2021-09-17", "2018-09-17", "2014-09-17", "1992-09-17", "1993-09-17"],
        basin="atlantic", era5_deepen=43.8, box_nodes=440),
    # ERA5: 1006.7 -> 963.7 hPa at +84 h, deepening 43.0 hPa; genesis 27.5,-77.0; box 277 mesh nodes; 8 analogs offered
    "atl2019_0915": dict(
        ic="2019-09-15", center=(27.5, -77.0),
        box=dict(lat=(22.5, 35.0), lon=(-84.0, -59.5)),
        analogs=["2014-09-15", "1986-09-15", "1991-09-15", "2002-09-15", "2015-09-15", "1997-09-15", "1987-09-15", "1994-09-15"],
        basin="atlantic", era5_deepen=43.0, box_nodes=277),
    # ERA5: 995.8 -> 957.5 hPa at +90 h, deepening 38.4 hPa; genesis -6.0,130.0; box 334 mesh nodes; 8 analogs offered
    "aus1989_0419": dict(
        ic="1989-04-19", center=(-12.5, 122.0),
        box=dict(lat=(-26.0, -7.5), lon=(110.0, 128.0)),
        analogs=["1997-04-19", "1981-04-19", "2005-04-19", "1992-04-19", "1993-04-19", "2017-04-19", "2004-04-19", "2021-04-19"],
        basin="aus", era5_deepen=38.4, box_nodes=334),
    # ERA5: 1005.0 -> 979.7 hPa at +78 h, deepening 25.3 hPa; genesis -9.0,129.0; box 260 mesh nodes; 8 analogs offered
    "aus1991_0411": dict(
        ic="1991-04-11", center=(-10.5, 125.5),
        box=dict(lat=(-19.5, -5.5), lon=(113.5, 131.5)),
        analogs=["1997-04-11", "2003-04-11", "2004-04-11", "1995-04-11", "2020-04-11", "1988-04-11", "2001-04-11", "2015-04-11"],
        basin="aus", era5_deepen=25.3, box_nodes=260),
    # ERA5: 1003.6 -> 967.1 hPa at +90 h, deepening 36.5 hPa; genesis -12.5,144.5; box 177 mesh nodes; 8 analogs offered
    "aus1993_0204": dict(
        ic="1993-02-04", center=(-15.0, 150.0),
        box=dict(lat=(-23.0, -9.5), lon=(143.5, 158.5)),
        analogs=["1998-02-04", "2013-02-04", "1983-02-04", "1992-02-04", "1995-02-04", "2000-02-04", "2011-02-04", "1991-02-04"],
        basin="aus", era5_deepen=36.5, box_nodes=177),
    # ERA5: 994.7 -> 968.8 hPa at +84 h, deepening 25.9 hPa; genesis -8.0,131.0; box 405 mesh nodes; 8 analogs offered
    "aus1996_0407": dict(
        ic="1996-04-07", center=(-14.0, 122.0),
        box=dict(lat=(-29.5, -8.5), lon=(108.5, 128.0)),
        analogs=["2002-04-07", "2004-04-07", "2012-04-07", "1982-04-07", "2016-04-07", "1991-04-07", "1997-04-07", "2001-04-07"],
        basin="aus", era5_deepen=25.9, box_nodes=405),
    # ERA5: 1005.5 -> 978.1 hPa at +96 h, deepening 27.4 hPa; genesis -10.0,133.5; box 202 mesh nodes; 8 analogs offered
    "aus1998_1205": dict(
        ic="1998-12-05", center=(-9.5, 133.0),
        box=dict(lat=(-17.0, -5.0), lon=(122.5, 139.0)),
        analogs=["1989-12-05", "2019-12-05", "2008-12-05", "2015-12-05", "1982-12-05", "1997-12-05", "2009-12-05", "1979-12-05"],
        basin="aus", era5_deepen=27.4, box_nodes=202),
    # ERA5: 1003.4 -> 962.6 hPa at +90 h, deepening 40.9 hPa; genesis -11.0,121.5; box 245 mesh nodes; 8 analogs offered
    "aus1999_1211": dict(
        ic="1999-12-11", center=(-14.0, 119.5),
        box=dict(lat=(-25.5, -8.5), lon=(111.5, 125.5)),
        analogs=["2002-12-11", "2017-12-11", "1979-12-11", "1987-12-11", "1997-12-11", "2012-12-11", "1992-12-11", "2006-12-11"],
        basin="aus", era5_deepen=40.9, box_nodes=245),
    # ERA5: 994.9 -> 970.2 hPa at +90 h, deepening 24.8 hPa; genesis -16.0,121.5; box 271 mesh nodes; 8 analogs offered
    "aus2008_0214": dict(
        ic="2008-02-14", center=(-15.0, 120.5),
        box=dict(lat=(-24.5, -10.0), lon=(108.5, 127.0)),
        analogs=["1998-02-14", "1990-02-14", "1983-02-14", "2019-02-14", "2005-02-14", "2006-02-14", "2012-02-14", "2020-02-14"],
        basin="aus", era5_deepen=24.8, box_nodes=271),
    # ERA5: 997.5 -> 968.5 hPa at +90 h, deepening 29.0 hPa; genesis -13.0,113.0; box 261 mesh nodes; 8 analogs offered
    "aus2012_0313": dict(
        ic="2012-03-13", center=(-18.5, 115.0),
        box=dict(lat=(-24.0, -10.5), lon=(106.5, 125.5)),
        analogs=["2010-03-13", "2002-03-13", "2003-03-13", "1992-03-13", "2019-03-13", "2021-03-13", "1997-03-13", "1998-03-13"],
        basin="aus", era5_deepen=29.0, box_nodes=261),
    # ERA5: 1000.9 -> 961.9 hPa at +96 h, deepening 39.0 hPa; genesis -10.0,128.0; box 345 mesh nodes; 8 analogs offered
    "aus2013_0107": dict(
        ic="2013-01-07", center=(-11.0, 122.0),
        box=dict(lat=(-22.5, -6.0), lon=(107.5, 128.0)),
        analogs=["1995-01-07", "2016-01-07", "1998-01-07", "1988-01-07", "2014-01-07", "1982-01-07", "1987-01-07", "2012-01-07"],
        basin="aus", era5_deepen=39.0, box_nodes=345),
    # ERA5: 1004.1 -> 980.0 hPa at +90 h, deepening 24.1 hPa; genesis -14.5,152.0; box 181 mesh nodes; 8 analogs offered
    "aus2017_0324": dict(
        ic="2017-03-24", center=(-16.0, 151.5),
        box=dict(lat=(-25.0, -11.0), lon=(143.0, 158.0)),
        analogs=["1992-03-24", "1991-03-24", "2015-03-24", "2016-03-24", "1983-03-24", "2003-03-24", "1981-03-24", "1993-03-24"],
        basin="aus", era5_deepen=24.1, box_nodes=181),
    # ERA5: 996.6 -> 972.2 hPa at +90 h, deepening 24.5 hPa; genesis -10.0,146.5; box 241 mesh nodes; 8 analogs offered
    "aus2019_0319": dict(
        ic="2019-03-19", center=(-13.5, 144.0),
        box=dict(lat=(-21.0, -7.5), lon=(131.5, 150.0)),
        analogs=["1995-03-19", "2003-03-19", "2020-03-19", "1992-03-19", "1993-03-19", "2002-03-19", "1981-03-19", "1998-03-19"],
        basin="aus", era5_deepen=24.5, box_nodes=241),
    # ERA5: 1004.4 -> 984.0 hPa at +96 h, deepening 20.4 hPa; genesis 11.5,-142.5; box 474 mesh nodes; 8 analogs offered
    "epa1992_0908": dict(
        ic="1992-09-08", center=(14.5, -144.0),
        box=dict(lat=(7.5, 26.5), lon=(-166.0, -138.0)),
        analogs=["1988-09-08", "1993-09-08", "2017-09-08", "2006-09-08", "1990-09-08", "2021-09-08", "2010-09-08", "1998-09-08"],
        basin="epac", era5_deepen=20.4, box_nodes=474),
    # ERA5: 1004.1 -> 976.3 hPa at +78 h, deepening 27.8 hPa; genesis 13.5,-105.5; box 520 mesh nodes; 8 analogs offered
    "epa1993_0810": dict(
        ic="1993-08-10", center=(14.0, -108.5),
        box=dict(lat=(9.0, 23.0), lon=(-143.5, -102.5)),
        analogs=["2014-08-10", "1989-08-10", "2004-08-10", "1998-08-10", "2009-08-10", "2011-08-10", "2016-08-10", "2015-08-10"],
        basin="epac", era5_deepen=27.8, box_nodes=520),
    # ERA5: 1003.7 -> 970.6 hPa at +96 h, deepening 33.1 hPa; genesis 12.5,-101.0; box 190 mesh nodes; 8 analogs offered
    "epa1997_0917": dict(
        ic="1997-09-17", center=(13.0, -102.0),
        box=dict(lat=(8.0, 21.0), lon=(-112.5, -96.0)),
        analogs=["2011-09-17", "2008-09-17", "2012-09-17", "2020-09-17", "1981-09-17", "2007-09-17", "2021-09-17", "2005-09-17"],
        basin="epac", era5_deepen=33.1, box_nodes=190),
    # ERA5: 1006.2 -> 979.2 hPa at +84 h, deepening 27.0 hPa; genesis 15.5,-106.0; box 272 mesh nodes; 8 analogs offered
    "epa2000_0906": dict(
        ic="2000-09-06", center=(15.5, -106.0),
        box=dict(lat=(8.5, 25.0), lon=(-118.5, -100.0)),
        analogs=["2006-09-06", "2019-09-06", "1985-09-06", "1980-09-06", "2021-09-06", "2012-09-06", "1992-09-06", "2017-09-06"],
        basin="epac", era5_deepen=27.0, box_nodes=272),
    # ERA5: 1003.4 -> 977.6 hPa at +96 h, deepening 25.8 hPa; genesis 13.0,-98.5; box 290 mesh nodes; 8 analogs offered
    "epa2001_0923": dict(
        ic="2001-09-23", center=(14.0, -99.5),
        box=dict(lat=(9.0, 24.0), lon=(-116.0, -93.5)),
        analogs=["1987-09-23", "1990-09-23", "1996-09-23", "1983-09-23", "2021-09-23", "1981-09-23", "2020-09-23", "1995-09-23"],
        basin="epac", era5_deepen=25.8, box_nodes=290),
    # ERA5: 1002.1 -> 975.6 hPa at +96 h, deepening 26.5 hPa; genesis 13.5,-99.5; box 300 mesh nodes; 8 analogs offered
    "epa2008_1006": dict(
        ic="2008-10-06", center=(13.5, -103.0),
        box=dict(lat=(8.5, 23.5), lon=(-119.5, -97.0)),
        analogs=["1984-10-06", "2010-10-06", "1986-10-06", "1992-10-06", "1985-10-06", "2001-10-06", "1980-10-06", "1988-10-06"],
        basin="epac", era5_deepen=26.5, box_nodes=300),
    # ERA5: 1002.2 -> 978.6 hPa at +72 h, deepening 23.6 hPa; genesis 10.0,-99.5; box 444 mesh nodes; 8 analogs offered
    "epa2011_0801": dict(
        ic="2011-08-01", center=(11.5, -101.5),
        box=dict(lat=(6.5, 22.0), lon=(-126.5, -95.5)),
        analogs=["1979-08-01", "2002-08-01", "2003-08-01", "1982-08-01", "1990-08-01", "2013-08-01", "1985-08-01", "2018-08-01"],
        basin="epac", era5_deepen=23.6, box_nodes=444),
    # ERA5: 1000.7 -> 1000.7 hPa at +0 h, deepening 0.0 hPa; genesis 12.5,-142.0; box 354 mesh nodes; 8 analogs offered
    "ndepa2014_1017": dict(
        ic="2014-10-17", center=(15.0, -151.5),
        box=dict(lat=(10.0, 25.5), lon=(-170.0, -145.5)),
        analogs=["2016-10-17", "1990-10-17", "1995-10-17", "2010-10-17", "2015-10-17", "1994-10-17", "2021-10-17", "1993-10-17"],
        basin="epac", era5_deepen=0.0, box_nodes=354, nondev=True),
    # ERA5: 999.8 -> 967.2 hPa at +84 h, deepening 32.6 hPa; genesis 13.5,-104.0; box 257 mesh nodes; 8 analogs offered
    "epa2015_0603": dict(
        ic="2015-06-03", center=(12.5, -104.5),
        box=dict(lat=(7.0, 23.5), lon=(-116.0, -98.5)),
        analogs=["2000-06-03", "2006-06-03", "1984-06-03", "1996-06-03", "1987-06-03", "1989-06-03", "2011-06-03", "1990-06-03"],
        basin="epac", era5_deepen=32.6, box_nodes=257),
    # ERA5: 1004.1 -> 1004.1 hPa at +0 h, deepening 0.0 hPa; genesis 7.5,-149.0; box 393 mesh nodes; 8 analogs offered
    "ndepa2015_0822": dict(
        ic="2015-08-22", center=(13.0, -154.0),
        box=dict(lat=(8.0, 24.0), lon=(-173.5, -148.0)),
        analogs=["1999-08-22", "2004-08-22", "1981-08-22", "1989-08-22", "1983-08-22", "2008-08-22", "2012-08-22", "2020-08-22"],
        basin="epac", era5_deepen=0.0, box_nodes=393, nondev=True),
    # ERA5: 1005.1 -> 977.1 hPa at +84 h, deepening 28.1 hPa; genesis 14.5,-106.5; box 290 mesh nodes; 8 analogs offered
    "epa2018_0925": dict(
        ic="2018-09-25", center=(14.5, -106.5),
        box=dict(lat=(9.5, 23.0), lon=(-124.0, -100.5)),
        analogs=["1987-09-25", "1981-09-25", "1985-09-25", "2021-09-25", "1983-09-25", "1996-09-25", "2020-09-25", "1982-09-25"],
        basin="epac", era5_deepen=28.1, box_nodes=290),
    # ERA5: 1004.5 -> 983.8 hPa at +96 h, deepening 20.7 hPa; genesis 11.5,-111.0; box 407 mesh nodes; 8 analogs offered
    "epa2020_0930": dict(
        ic="2020-09-30", center=(14.5, -109.5),
        box=dict(lat=(9.5, 24.0), lon=(-134.0, -103.5)),
        analogs=["2016-09-30", "1988-09-30", "1980-09-30", "1979-09-30", "2013-09-30", "1993-09-30", "2021-09-30", "1984-09-30"],
        basin="epac", era5_deepen=20.7, box_nodes=407),
    # ERA5: 997.2 -> 974.5 hPa at +84 h, deepening 22.8 hPa; genesis 5.0,100.0; box 263 mesh nodes; 8 analogs offered
    "nin1988_1125": dict(
        ic="1988-11-25", center=(11.5, 92.5),
        box=dict(lat=(6.5, 24.0), lon=(81.5, 98.5)),
        analogs=["2018-11-25", "2003-11-25", "1992-11-25", "1989-11-25", "1991-11-25", "1986-11-25", "1985-11-25", "1983-11-25"],
        basin="nind", era5_deepen=22.8, box_nodes=263),
    # ERA5: 1002.8 -> 964.5 hPa at +96 h, deepening 38.3 hPa; genesis 8.5,87.5; box 236 mesh nodes; 8 analogs offered
    "nin1991_0425": dict(
        ic="1991-04-25", center=(10.5, 87.5),
        box=dict(lat=(5.0, 23.5), lon=(81.0, 95.0)),
        analogs=["1981-04-25", "1992-04-25", "2013-04-25", "1979-04-25", "2019-04-25", "2009-04-25", "2015-04-25", "1998-04-25"],
        basin="nind", era5_deepen=38.3, box_nodes=236),
    # ERA5: 1005.5 -> 982.9 hPa at +90 h, deepening 22.5 hPa; genesis 6.5,91.0; box 334 mesh nodes; 8 analogs offered
    "nin1995_1121": dict(
        ic="1995-11-21", center=(6.5, 91.0),
        box=dict(lat=(5.0, 25.0), lon=(79.0, 97.5)),
        analogs=["1997-11-21", "1986-11-21", "2009-11-21", "1985-11-21", "1983-11-21", "2007-11-21", "1980-11-21", "1984-11-21"],
        basin="nind", era5_deepen=22.5, box_nodes=334),
    # ERA5: 1001.3 -> 987.4 hPa at +96 h, deepening 13.9 hPa; genesis 16.0,84.5; box 272 mesh nodes; 8 analogs offered
    "nin1997_0923": dict(
        ic="1997-09-23", center=(15.5, 82.5),
        box=dict(lat=(10.5, 25.0), lon=(76.5, 97.5)),
        analogs=["1994-09-23", "2000-09-23", "1980-09-23", "2019-09-23", "1986-09-23", "2004-09-23", "1984-09-23", "2001-09-23"],
        basin="nind", era5_deepen=13.9, box_nodes=272, secondary=True),
    # ERA5: 1004.5 -> 986.3 hPa at +96 h, deepening 18.2 hPa; genesis 9.0,94.0; box 215 mesh nodes; 8 analogs offered
    "nin2007_1110": dict(
        ic="2007-11-10", center=(9.5, 94.0),
        box=dict(lat=(5.0, 19.5), lon=(83.5, 100.0)),
        analogs=["2006-11-10", "1997-11-10", "2010-11-10", "2016-11-10", "1991-11-10", "2005-11-10", "1982-11-10", "2004-11-10"],
        basin="nind", era5_deepen=18.2, box_nodes=215, secondary=True),
    # ERA5: 1002.7 -> 1002.2 hPa at +12 h, deepening 0.5 hPa; genesis 12.0,84.0; box 199 mesh nodes; 8 analogs offered
    "ndnin2013_1022": dict(
        ic="2013-10-22", center=(14.5, 81.0),
        box=dict(lat=(8.5, 22.0), lon=(72.5, 87.5)),
        analogs=["2018-10-22", "1993-10-22", "2009-10-22", "1997-10-22", "2006-10-22", "1985-10-22", "2021-10-22", "1979-10-22"],
        basin="nind", era5_deepen=0.5, box_nodes=199, nondev=True),
    # ERA5: 1004.6 -> 984.9 hPa at +96 h, deepening 19.7 hPa; genesis 7.5,83.5; box 156 mesh nodes; 8 analogs offered
    "nin2013_1204": dict(
        ic="2013-12-04", center=(6.5, 83.5),
        box=dict(lat=(5.0, 17.0), lon=(77.5, 90.5)),
        analogs=["1983-12-04", "1989-12-04", "1991-12-04", "1990-12-04", "1979-12-04", "2004-12-04", "1982-12-04", "1995-12-04"],
        basin="nind", era5_deepen=19.7, box_nodes=156),
    # ERA5: 998.6 -> 969.5 hPa at +96 h, deepening 29.1 hPa; genesis 10.0,96.0; box 272 mesh nodes; 8 analogs offered
    "nin2014_1008": dict(
        ic="2014-10-08", center=(12.5, 92.0),
        box=dict(lat=(7.5, 22.5), lon=(78.0, 98.0)),
        analogs=["1997-10-08", "1979-10-08", "1996-10-08", "1982-10-08", "2019-10-08", "2011-10-08", "1980-10-08", "1987-10-08"],
        basin="nind", era5_deepen=29.1, box_nodes=272),
    # ERA5: 1002.0 -> 1001.0 hPa at +12 h, deepening 1.0 hPa; genesis 18.0,84.0; box 174 mesh nodes; 8 analogs offered
    "ndnin2016_0911": dict(
        ic="2016-09-11", center=(17.0, 83.5),
        box=dict(lat=(10.5, 23.0), lon=(76.0, 90.5)),
        analogs=["2018-09-11", "1997-09-11", "1981-09-11", "1991-09-11", "1988-09-11", "2017-09-11", "1979-09-11", "1995-09-11"],
        basin="nind", era5_deepen=1.0, box_nodes=174, nondev=True),
    # ERA5: 998.8 -> 982.9 hPa at +96 h, deepening 15.9 hPa; genesis 6.5,97.0; box 268 mesh nodes; 8 analogs offered
    "nin2016_1208": dict(
        ic="2016-12-08", center=(10.5, 92.0),
        box=dict(lat=(5.5, 18.5), lon=(76.0, 98.0)),
        analogs=["1991-12-08", "1989-12-08", "1979-12-08", "2020-12-08", "1982-12-08", "1995-12-08", "1986-12-08", "2004-12-08"],
        basin="nind", era5_deepen=15.9, box_nodes=268, secondary=True),
    # ERA5: 1003.8 -> 977.2 hPa at +96 h, deepening 26.6 hPa; genesis 8.0,60.0; box 276 mesh nodes; 8 analogs offered
    "nin2018_0521": dict(
        ic="2018-05-21", center=(8.5, 59.5),
        box=dict(lat=(5.0, 20.5), lon=(48.5, 65.5)),
        analogs=["1991-05-21", "2019-05-21", "1982-05-21", "1979-05-21", "1995-05-21", "2014-05-21", "2003-05-21", "1981-05-21"],
        basin="nind", era5_deepen=26.6, box_nodes=276),
    # ERA5: 998.3 -> 974.3 hPa at +96 h, deepening 24.0 hPa; genesis 14.5,64.5; box 240 mesh nodes; 8 analogs offered
    "nin2019_1024": dict(
        ic="2019-10-24", center=(15.0, 70.5),
        box=dict(lat=(10.0, 23.0), lon=(59.5, 78.0)),
        analogs=["1984-10-24", "1992-10-24", "1979-10-24", "2021-10-24", "2018-10-24", "1994-10-24", "1997-10-24", "1982-10-24"],
        basin="nind", era5_deepen=24.0, box_nodes=240),
    # ERA5: 1003.6 -> 967.8 hPa at +96 h, deepening 35.8 hPa; genesis -14.0,75.0; box 273 mesh nodes; 8 analogs offered
    "sin1988_0316": dict(
        ic="1988-03-16", center=(-13.0, 72.0),
        box=dict(lat=(-21.5, -8.0), lon=(57.0, 78.0)),
        analogs=["2015-03-16", "1997-03-16", "1981-03-16", "2010-03-16", "1985-03-16", "2012-03-16", "2001-03-16", "1987-03-16"],
        basin="sind", era5_deepen=35.8, box_nodes=273),
    # ERA5: 991.2 -> 946.1 hPa at +96 h, deepening 45.2 hPa; genesis -9.5,86.0; box 358 mesh nodes; 8 analogs offered
    "sin1993_0122": dict(
        ic="1993-01-22", center=(-13.5, 81.0),
        box=dict(lat=(-22.0, -8.5), lon=(57.5, 87.0)),
        analogs=["1983-01-22", "2007-01-22", "2005-01-22", "2014-01-22", "2015-01-22", "1992-01-22", "1979-01-22", "1982-01-22"],
        basin="sind", era5_deepen=45.2, box_nodes=358),
    # ERA5: 999.3 -> 966.1 hPa at +78 h, deepening 33.2 hPa; genesis -14.0,71.0; box 300 mesh nodes; 8 analogs offered
    "sin1996_0403": dict(
        ic="1996-04-03", center=(-15.0, 70.0),
        box=dict(lat=(-25.5, -9.5), lon=(57.0, 77.5)),
        analogs=["1982-04-03", "2004-04-03", "2017-04-03", "1991-04-03", "1992-04-03", "2011-04-03", "2012-04-03", "2008-04-03"],
        basin="sind", era5_deepen=33.2, box_nodes=300),
    # ERA5: 1000.9 -> 967.0 hPa at +96 h, deepening 33.9 hPa; genesis -13.0,65.0; box 282 mesh nodes; 8 analogs offered
    "sin1998_0208": dict(
        ic="1998-02-08", center=(-14.0, 62.0),
        box=dict(lat=(-28.0, -8.5), lon=(53.0, 68.0)),
        analogs=["1999-02-08", "1993-02-08", "2006-02-08", "2015-02-08", "2010-02-08", "2018-02-08", "1996-02-08", "2000-02-08"],
        basin="sind", era5_deepen=33.9, box_nodes=282),
    # ERA5: 1000.3 -> 1000.3 hPa at +0 h, deepening 0.0 hPa; genesis -5.0,89.5; box 155 mesh nodes; 8 analogs offered
    "ndsin2002_1227": dict(
        ic="2002-12-27", center=(-10.0, 89.0),
        box=dict(lat=(-15.0, -5.0), lon=(83.0, 99.5)),
        analogs=["2001-12-27", "1987-12-27", "1995-12-27", "1997-12-27", "2005-12-27", "1992-12-27", "1994-12-27", "1996-12-27"],
        basin="sind", era5_deepen=0.0, box_nodes=155, nondev=True),
    # ERA5: 999.9 -> 953.1 hPa at +90 h, deepening 46.9 hPa; genesis -15.0,78.0; box 392 mesh nodes; 8 analogs offered
    "sin2007_0221": dict(
        ic="2007-02-21", center=(-14.0, 72.0),
        box=dict(lat=(-23.5, -9.0), lon=(49.0, 78.0)),
        analogs=["1988-02-21", "1993-02-21", "1987-02-21", "2001-02-21", "2019-02-21", "2014-02-21", "1980-02-21", "2003-02-21"],
        basin="sind", era5_deepen=46.9, box_nodes=392),
    # ERA5: 1000.1 -> 1000.1 hPa at +36 h, deepening 0.0 hPa; genesis -18.0,100.0; box 225 mesh nodes; 8 analogs offered
    "ndsin2011_0119": dict(
        ic="2011-01-19", center=(-20.0, 98.0),
        box=dict(lat=(-25.0, -11.5), lon=(85.5, 104.0)),
        analogs=["1992-01-19", "1979-01-19", "2007-01-19", "1983-01-19", "1998-01-19", "2003-01-19", "1985-01-19", "2004-01-19"],
        basin="sind", era5_deepen=0.0, box_nodes=225, nondev=True),
    # ERA5: 998.6 -> 959.1 hPa at +90 h, deepening 39.5 hPa; genesis -11.5,75.0; box 360 mesh nodes; 8 analogs offered
    "sin2012_0210": dict(
        ic="2012-02-10", center=(-14.5, 63.5),
        box=dict(lat=(-24.0, -9.5), lon=(42.5, 69.5)),
        analogs=["2006-02-10", "2001-02-10", "1991-02-10", "2000-02-10", "1996-02-10", "2015-02-10", "2018-02-10", "1993-02-10"],
        basin="sind", era5_deepen=39.5, box_nodes=360),
    # ERA5: 1002.1 -> 963.5 hPa at +96 h, deepening 38.7 hPa; genesis -10.0,87.0; box 325 mesh nodes; 8 analogs offered
    "sin2013_0210": dict(
        ic="2013-02-10", center=(-11.0, 86.0),
        box=dict(lat=(-27.0, -6.0), lon=(73.5, 92.0)),
        analogs=["2019-02-10", "1998-02-10", "2006-02-10", "1989-02-10", "1991-02-10", "1987-02-10", "2009-02-10", "2005-02-10"],
        basin="sind", era5_deepen=38.7, box_nodes=325),
    # ERA5: 993.6 -> 958.4 hPa at +96 h, deepening 35.2 hPa; genesis -18.5,53.5; box 236 mesh nodes; 8 analogs offered
    "sin2015_0112": dict(
        ic="2015-01-12", center=(-17.5, 55.5),
        box=dict(lat=(-24.5, -12.0), lon=(49.5, 70.0)),
        analogs=["1995-01-12", "1979-01-12", "1981-01-12", "1983-01-12", "2019-01-12", "2013-01-12", "2004-01-12", "2005-01-12"],
        basin="sind", era5_deepen=35.2, box_nodes=236),
    # ERA5: 1001.5 -> 964.8 hPa at +96 h, deepening 36.6 hPa; genesis -13.0,55.5; box 295 mesh nodes; 8 analogs offered
    "sin2018_0302": dict(
        ic="2018-03-02", center=(-13.5, 54.0),
        box=dict(lat=(-28.5, -8.0), lon=(44.5, 60.0)),
        analogs=["1997-03-02", "1984-03-02", "2010-03-02", "1981-03-02", "2016-03-02", "2014-03-02", "1985-03-02", "2001-03-02"],
        basin="sind", era5_deepen=36.6, box_nodes=295),
    # ERA5: 1005.2 -> 969.3 hPa at +84 h, deepening 35.9 hPa; genesis -13.5,68.5; box 509 mesh nodes; 8 analogs offered
    "sin2020_0401": dict(
        ic="2020-04-01", center=(-13.5, 68.5),
        box=dict(lat=(-29.0, -7.0), lon=(62.0, 88.5)),
        analogs=["2017-04-01", "1980-04-01", "1982-04-01", "2004-04-01", "1985-04-01", "1998-04-01", "2012-04-01", "2003-04-01"],
        basin="sind", era5_deepen=35.9, box_nodes=509),
    # ERA5: 986.1 -> 940.2 hPa at +90 h, deepening 45.9 hPa; genesis -10.0,165.5; box 148 mesh nodes; 8 analogs offered
    "spa1992_1229": dict(
        ic="1992-12-29", center=(-14.5, 172.5),
        box=dict(lat=(-21.0, -9.5), lon=(166.0, 179.9)),
        analogs=["1995-12-29", "1980-12-29", "1985-12-29", "1989-12-29", "2020-12-29", "2005-12-29", "2010-12-29", "2017-12-29"],
        basin="spac", era5_deepen=45.9, box_nodes=148),
    # ERA5: 998.4 -> 965.8 hPa at +96 h, deepening 32.6 hPa; genesis -15.5,173.0; box 186 mesh nodes; 8 analogs offered
    "spa1994_0122": dict(
        ic="1994-01-22", center=(-15.0, 164.0),
        box=dict(lat=(-22.5, -9.0), lon=(155.5, 171.5)),
        analogs=["1985-01-22", "1992-01-22", "2009-01-22", "2020-01-22", "2005-01-22", "2007-01-22", "1988-01-22", "2004-01-22"],
        basin="spac", era5_deepen=32.6, box_nodes=186),
    # ERA5: 1002.1 -> 969.6 hPa at +96 h, deepening 32.4 hPa; genesis -13.0,173.0; box 274 mesh nodes; 8 analogs offered
    "spa1997_0103": dict(
        ic="1997-01-03", center=(-13.5, 168.0),
        box=dict(lat=(-23.5, -8.5), lon=(152.5, 174.0)),
        analogs=["1984-01-03", "1982-01-03", "1999-01-03", "2006-01-03", "2000-01-03", "1985-01-03", "1991-01-03", "2012-01-03"],
        basin="spac", era5_deepen=32.4, box_nodes=274),
    # ERA5: 1000.7 -> 967.1 hPa at +90 h, deepening 33.6 hPa; genesis -11.0,177.0; box 231 mesh nodes; 8 analogs offered
    "spa1998_0103": dict(
        ic="1998-01-03", center=(-12.5, 173.5),
        box=dict(lat=(-23.0, -7.5), lon=(163.5, 179.5)),
        analogs=["2006-01-03", "1999-01-03", "1982-01-03", "2018-01-03", "1984-01-03", "2000-01-03", "2012-01-03", "2009-01-03"],
        basin="spac", era5_deepen=33.6, box_nodes=231),
    # ERA5: 1000.6 -> 970.5 hPa at +90 h, deepening 30.1 hPa; genesis -16.5,161.0; box 303 mesh nodes; 8 analogs offered
    "spa2008_0115": dict(
        ic="2008-01-15", center=(-16.0, 164.0),
        box=dict(lat=(-26.0, -10.0), lon=(158.0, 179.9)),
        analogs=["2004-01-15", "1991-01-15", "1993-01-15", "2017-01-15", "1987-01-15", "1998-01-15", "1982-01-15", "1980-01-15"],
        basin="spac", era5_deepen=30.1, box_nodes=303),
    # ERA5: 997.2 -> 966.4 hPa at +96 h, deepening 30.8 hPa; genesis -15.5,172.0; box 282 mesh nodes; 8 analogs offered
    "spa2011_0219": dict(
        ic="2011-02-19", center=(-16.0, 169.0),
        box=dict(lat=(-30.0, -11.0), lon=(163.0, 179.9)),
        analogs=["1982-02-19", "2012-02-19", "2007-02-19", "2004-02-19", "2021-02-19", "2008-02-19", "2009-02-19", "1986-02-19"],
        basin="spac", era5_deepen=30.8, box_nodes=282),
    # ERA5: 1004.0 -> 974.5 hPa at +96 h, deepening 29.5 hPa; genesis -16.0,171.5; box 288 mesh nodes; 8 analogs offered
    "spa2014_0309": dict(
        ic="2014-03-09", center=(-15.5, 168.5),
        box=dict(lat=(-27.0, -9.5), lon=(160.5, 179.0)),
        analogs=["2006-03-09", "2021-03-09", "2001-03-09", "1988-03-09", "2005-03-09", "2016-03-09", "1987-03-09", "1994-03-09"],
        basin="spac", era5_deepen=29.5, box_nodes=288),
    # ERA5: 999.0 -> 968.5 hPa at +90 h, deepening 30.5 hPa; genesis -8.0,169.5; box 162 mesh nodes; 8 analogs offered
    "spa2015_0308": dict(
        ic="2015-03-08", center=(-8.0, 171.0),
        box=dict(lat=(-17.5, -5.0), lon=(163.5, 177.0)),
        analogs=["2006-03-08", "2021-03-08", "1983-03-08", "1988-03-08", "2016-03-08", "2008-03-08", "1987-03-08", "1989-03-08"],
        basin="spac", era5_deepen=30.5, box_nodes=162),
    # ERA5: 1006.0 -> 973.7 hPa at +96 h, deepening 32.2 hPa; genesis -10.5,163.5; box 191 mesh nodes; 8 analogs offered
    "spa2017_0503": dict(
        ic="2017-05-03", center=(-13.0, 171.0),
        box=dict(lat=(-19.0, -8.0), lon=(158.0, 177.0)),
        analogs=["1984-05-03", "1998-05-03", "1987-05-03", "2013-05-03", "1983-05-03", "1982-05-03", "2005-05-03", "2018-05-03"],
        basin="spac", era5_deepen=32.2, box_nodes=191),
    # ERA5: 997.5 -> 966.0 hPa at +96 h, deepening 31.5 hPa; genesis -19.0,155.0; box 176 mesh nodes; 8 analogs offered
    "spa2019_0211": dict(
        ic="2019-02-11", center=(-16.0, 161.0),
        box=dict(lat=(-21.0, -9.0), lon=(155.0, 172.0)),
        analogs=["1998-02-11", "2000-02-11", "1990-02-11", "2009-02-11", "1995-02-11", "1983-02-11", "1996-02-11", "2013-02-11"],
        basin="spac", era5_deepen=31.5, box_nodes=176),
    # ERA5: 993.8 -> 932.6 hPa at +90 h, deepening 61.2 hPa; genesis 13.5,153.0; box 349 mesh nodes; 8 analogs offered
    "wpa1986_0705": dict(
        ic="1986-07-05", center=(15.5, 141.0),
        box=dict(lat=(10.5, 23.5), lon=(116.5, 147.0)),
        analogs=["1998-07-05", "1987-07-05", "1989-07-05", "2003-07-05", "2017-07-05", "2009-07-05", "2020-07-05", "1984-07-05"],
        basin="wpac", era5_deepen=61.2, box_nodes=349),
    # ERA5: 998.6 -> 941.7 hPa at +96 h, deepening 56.9 hPa; genesis 14.0,144.5; box 176 mesh nodes; 8 analogs offered
    "wpa1987_0908": dict(
        ic="1987-09-08", center=(15.5, 141.0),
        box=dict(lat=(10.5, 23.5), lon=(130.5, 147.0)),
        analogs=["2010-09-08", "2020-09-08", "1979-09-08", "1995-09-08", "1981-09-08", "2004-09-08", "1984-09-08", "1997-09-08"],
        basin="wpac", era5_deepen=56.9, box_nodes=176),
    # ERA5: 998.5 -> 950.1 hPa at +84 h, deepening 48.4 hPa; genesis 13.0,146.0; box 310 mesh nodes; 8 analogs offered
    "wpa1993_1002": dict(
        ic="1993-10-02", center=(15.0, 140.5),
        box=dict(lat=(10.0, 28.5), lon=(126.5, 146.5)),
        analogs=["1997-10-02", "2015-10-02", "1995-10-02", "2019-10-02", "1983-10-02", "1999-10-02", "2002-10-02", "1984-10-02"],
        basin="wpac", era5_deepen=48.4, box_nodes=310),
    # ERA5: 989.0 -> 934.6 hPa at +96 h, deepening 54.4 hPa; genesis 10.0,150.0; box 392 mesh nodes; 8 analogs offered
    "wpa1996_1106": dict(
        ic="1996-11-06", center=(11.0, 150.5),
        box=dict(lat=(6.0, 21.5), lon=(128.0, 156.5)),
        analogs=["2015-11-06", "2010-11-06", "1980-11-06", "1984-11-06", "2004-11-06", "2020-11-06", "1995-11-06", "1981-11-06"],
        basin="wpac", era5_deepen=54.4, box_nodes=392),
    # ERA5: 1003.7 -> 935.9 hPa at +96 h, deepening 67.9 hPa; genesis 13.5,160.0; box 442 mesh nodes; 8 analogs offered
    "wpa1997_1016": dict(
        ic="1997-10-16", center=(14.0, 157.0),
        box=dict(lat=(9.0, 25.0), lon=(131.5, 163.0)),
        analogs=["2011-10-16", "1993-10-16", "2018-10-16", "2014-10-16", "2007-10-16", "1980-10-16", "2000-10-16", "2019-10-16"],
        basin="wpac", era5_deepen=67.9, box_nodes=442),
    # ERA5: 1001.0 -> 946.8 hPa at +90 h, deepening 54.3 hPa; genesis 7.5,138.0; box 296 mesh nodes; 8 analogs offered
    "wpa2004_0614": dict(
        ic="2004-06-14", center=(9.5, 137.0),
        box=dict(lat=(4.5, 23.5), lon=(126.0, 143.5)),
        analogs=["1980-06-14", "1993-06-14", "2020-06-14", "1998-06-14", "2008-06-14", "1999-06-14", "1989-06-14", "1983-06-14"],
        basin="wpac", era5_deepen=54.3, box_nodes=296),
    # ERA5: 1006.9 -> 945.9 hPa at +90 h, deepening 60.9 hPa; genesis 23.0,150.0; box 417 mesh nodes; 8 analogs offered
    "wpa2005_0713": dict(
        ic="2005-07-13", center=(23.0, 149.0),
        box=dict(lat=(14.0, 28.0), lon=(119.5, 155.0)),
        analogs=["1996-07-13", "1997-07-13", "1992-07-13", "1991-07-13", "1995-07-13", "1981-07-13", "1988-07-13", "2021-07-13"],
        basin="wpac", era5_deepen=60.9, box_nodes=417),
    # ERA5: 1003.6 -> 1002.6 hPa at +18 h, deepening 1.0 hPa; genesis 27.5,174.5; box 172 mesh nodes; 8 analogs offered
    "ndwpa2007_1002": dict(
        ic="2007-10-02", center=(29.0, 168.0),
        box=dict(lat=(22.5, 34.5), lon=(162.0, 177.5)),
        analogs=["1997-10-02", "1989-10-02", "2011-10-02", "1995-10-02", "2017-10-02", "1985-10-02", "1983-10-02", "2009-10-02"],
        basin="wpac", era5_deepen=1.0, box_nodes=172, nondev=True),
    # ERA5: 1003.7 -> 1003.5 hPa at +6 h, deepening 0.2 hPa; genesis 13.0,110.0; box 162 mesh nodes; 8 analogs offered
    "ndwpa2010_1004": dict(
        ic="2010-10-04", center=(17.0, 107.5),
        box=dict(lat=(11.5, 24.5), lon=(101.5, 116.0)),
        analogs=["1986-10-04", "1982-10-04", "1996-10-04", "1999-10-04", "1987-10-04", "1994-10-04", "2019-10-04", "1984-10-04"],
        basin="wpac", era5_deepen=0.2, box_nodes=162, nondev=True),
    # ERA5: 990.3 -> 943.1 hPa at +78 h, deepening 47.2 hPa; genesis 19.0,158.5; box 359 mesh nodes; 8 analogs offered
    "wpa2011_0714": dict(
        ic="2011-07-14", center=(20.0, 147.0),
        box=dict(lat=(15.0, 32.0), lon=(127.0, 153.0)),
        analogs=["1997-07-14", "1992-07-14", "2004-07-14", "2010-07-14", "2013-07-14", "1981-07-14", "2017-07-14", "2001-07-14"],
        basin="wpac", era5_deepen=47.2, box_nodes=359),
    # ERA5: 1001.1 -> 948.4 hPa at +90 h, deepening 52.7 hPa; genesis 12.0,155.5; box 469 mesh nodes; 8 analogs offered
    "wpa2014_0930": dict(
        ic="2014-09-30", center=(15.5, 148.5),
        box=dict(lat=(10.5, 30.5), lon=(126.5, 154.5)),
        analogs=["1997-09-30", "2000-09-30", "2017-09-30", "2015-09-30", "1999-09-30", "1991-09-30", "1982-09-30", "1983-09-30"],
        basin="wpac", era5_deepen=52.7, box_nodes=469),
    # ERA5: 994.9 -> 943.9 hPa at +90 h, deepening 51.0 hPa; genesis 9.5,161.0; box 412 mesh nodes; 8 analogs offered
    "wpa2018_1023": dict(
        ic="2018-10-23", center=(11.5, 151.5),
        box=dict(lat=(6.5, 22.5), lon=(128.5, 157.5)),
        analogs=["2005-10-23", "2021-10-23", "1981-10-23", "1990-10-23", "1999-10-23", "1986-10-23", "1979-10-23", "2008-10-23"],
        basin="wpac", era5_deepen=51.0, box_nodes=412),
}

PRIMARY = [k for k, v in STORMS.items() if not v.get("secondary") and not v.get("nondev")]
SECONDARY = [k for k, v in STORMS.items() if v.get("secondary")]
NONDEV = [k for k, v in STORMS.items() if v.get("nondev")]

# Every rejected candidate, with its ERA5 reason, is in results/mega_storm_gate.json.
N_REJECTED = 7089
