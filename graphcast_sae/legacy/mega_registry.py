"""Emit graphcast_sae/legacy/skill_mega_storms.py from results/mega_storm_gate.json.

The registry is GENERATED, never hand-edited: every field in it traces to an ERA5 number
in the gate file, so a storm cannot enter the battery by recall. Re-running this after a
new sweep batch rewrites the module in place.

Paper: not in the paper; kept for provenance only
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: rewrites graphcast_sae/legacy/skill_mega_storms.py from results/mega_storm_gate.json
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.mega_registry
"""
import json
import os
import sys
import time

import numpy as np

from graphcast_sae.paths import REPO_ROOT, MESH_GEOM
ROOT = str(REPO_ROOT)
GATE = os.path.join(ROOT, "results", "mega_storm_gate.json")
MOD = os.path.join(ROOT, "graphcast_sae", "legacy", "skill_mega_storms.py")

HEAD = '''"""Mega tropical-cyclone battery -- {n} ERA5-gated storms, all basins, {y0}-{y1}.

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
  developing  full-resolution box deepening >= {dp} hPa (>= {ds} hPa admitted as SECONDARY),
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
ida2021's 839 nodes, and {na} analogs are offered per storm instead of 5.

The cost of quietest-first, stated rather than hidden: picking the {na} quietest dates of a
~42-year pool is a mild selection toward suppressed conditions, which would lower the
restore-to-normal level and so RAISE the apparent ablation dose relative to the existing
battery's arbitrary fixed years. It is accepted deliberately, because the failure it trades
against -- a storm reaching the GPU with one surviving analog, as ida2021 did -- is worse.
Every candidate's ERA5 box-min is in `analog_mslp` in the gate file, so a later run can
re-draw analogs on a different rule without re-sweeping.

ERA5 deepening in this battery: {dmin:.1f} - {dmax:.1f} hPa (median {dmed:.1f}); the NH
battery spans 18.7 - 33.2 and the SH primary set 20.7 - 27.9.

Generated {stamp} from results/mega_storm_gate.json.
"""
import numpy as np  # noqa: F401

from graphcast_sae.common.skill_conv_storms import CONV, RADIUS_KM, RANDOM_CTRL, TC, norm_lon  # noqa: F401

H = 16  # +96 h, same as every other battery

STORMS = {{
'''

TAIL = '''}}

PRIMARY = [k for k, v in STORMS.items() if not v.get("secondary") and not v.get("nondev")]
SECONDARY = [k for k, v in STORMS.items() if v.get("secondary")]
NONDEV = [k for k, v in STORMS.items() if v.get("nondev")]

# Every rejected candidate, with its ERA5 reason, is in results/mega_storm_gate.json.
N_REJECTED = {nrej}
'''

def main():
    g = json.load(open(GATE))
    acc = g["accepted"]
    if not acc:
        print("no accepted storms in the gate file")
        return
    deep = [c["era5_deepen"] for c in acc if not c.get("nondev")]
    na = max(len(c.get("analogs", [])) for c in acc)
    yrs = sorted({c["ic"][:4] for c in acc})
    src = HEAD.format(n=len(acc), y0=yrs[0], y1=yrs[-1], dp=18.7, ds=12.0, na=na,
                      dmin=min(deep), dmax=max(deep), dmed=float(np.median(deep)),
                      stamp=time.strftime("%Y-%m-%d %H:%M"))
    for c in sorted(acc, key=lambda r: (r["basin"], r["ic"])):
        flags = ""
        if c.get("secondary"):
            flags += ", secondary=True"
        if c.get("nondev"):
            flags += ", nondev=True"
        an = ", ".join(f'"{a}"' for a in c.get("analogs", []))
        src += (
            f'    # ERA5: {c["mslp_ic"]:.1f} -> {c["mslp_min"]:.1f} hPa at +{c["t_min_h"]} h, '
            f'deepening {c["era5_deepen"]:.1f} hPa; genesis '
            f'{c["genesis"][0]:.1f},{c["genesis"][1]:.1f}; box {c["box_nodes"]} mesh nodes; '
            f'{len(c.get("analogs", []))} analogs offered\n'
            f'    "{c["name"]}": dict(\n'
            f'        ic="{c["ic"]}", center=({c["center"][0]:.1f}, {c["center"][1]:.1f}),\n'
            f'        box=dict(lat=({c["box"]["lat"][0]}, {c["box"]["lat"][1]}), '
            f'lon=({c["box"]["lon"][0]}, {c["box"]["lon"][1]})),\n'
            f'        analogs=[{an}],\n'
            f'        basin="{c["basin"]}", era5_deepen={c["era5_deepen"]}, '
            f'box_nodes={c["box_nodes"]}{flags}),\n')
    src += TAIL.format(nrej=len(g.get("rejected", [])))
    open(MOD, "w").write(src)
    print(f"wrote {MOD}  ({len(acc)} storms)")

    # data gate on the emitted module itself: it must import, and every box must be legal

    import importlib
    S = importlib.import_module("graphcast_sae.legacy.skill_mega_storms")
    importlib.reload(S)
    geom = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(geom["lat"], float)
    mlon = np.asarray(geom["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    bad = []
    for k, v in S.STORMS.items():
        b = v["box"]
        assert b["lon"][0] < b["lon"][1] and b["lat"][0] < b["lat"][1], k
        n = int(((mlat >= b["lat"][0]) & (mlat <= b["lat"][1]) &
                 (mlon >= b["lon"][0]) & (mlon <= b["lon"][1])).sum())
        if n != v["box_nodes"] or n < 120:
            bad.append((k, "box_nodes", n, v["box_nodes"]))
        if len(v["analogs"]) < 3:
            bad.append((k, "analogs", len(v["analogs"])))
    print(f"import OK: {len(S.STORMS)} storms, {len(S.PRIMARY)} primary, "
          f"{len(S.SECONDARY)} secondary, {len(S.NONDEV)} nondev")
    print("box-node re-check:", "ALL PASS" if not bad else f"FAILURES {bad}")
    print("box_nodes range:", min(v["box_nodes"] for v in S.STORMS.values()),
          "-", max(v["box_nodes"] for v in S.STORMS.values()))

if __name__ == "__main__":
    main()
