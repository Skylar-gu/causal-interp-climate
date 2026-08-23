"""Configurable overlay on the frozen TC registry. Nothing in `skill_conv_storms` changes.

Two knobs, both env-driven, both added 2026-08-21 in response to
`docs/notes/result_normal_reference_2026_08_21.md`:

CONV_ANALOG_SPAN=<days>
    Widen the `normal` reference from five same-calendar-date analogues to every date
    within +/- span days of that calendar date, across CONV_ANALOG_YEARS. The committed
    battery found only 1 of 5 quiet days for Ida because it insisted on 27 August, and
    that date carried a storm in the box in four of the five chosen years; 11 of 12 nearby
    days are quiet. A target of 0.00 from a single day is what made Ida's clamp about five
    times stronger than the operator is described as being.

CONV_RADIUS_KM=<km>
    Override the 1500 km intervention disk. `skill_conv_run.run_storm` reads S.RADIUS_KM at
    call time, so this is a pure-configuration extent sweep: identical operator, identical
    features, identical storms, smaller area. Restore-to-normal shaves magnitude at fixed
    area; this varies area at fixed magnitude treatment, which is the complement the
    magnitude-vs-extent question needs.

Everything else -- TC, CONV, RANDOM_CTRL, H, norm_lon, the boxes, the centres, the ICs --
is re-exported unchanged so the medians stay comparable.

    SKILL_STORMS=skill_conv_storms_cfg CONV_ANALOG_SPAN=9 ...

Paper: Table tab:mechanism-interventions (env overlay on the registry)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.common.skill_conv_storms_cfg
"""
import os

import numpy as np

from graphcast_sae.common.skill_conv_storms import (TC, CONV, RADIUS_KM as _R0, H, RANDOM_CTRL,
                               STORMS as _BASE, norm_lon)

RADIUS_KM = float(os.environ.get("CONV_RADIUS_KM", _R0))

# WB2 ERA5 runs 1959-2022; these years are far enough from every storm's own year that the
# storm itself can never be an analogue, and each is screened for a storm in the box anyway.
YEARS = [int(y) for y in os.environ.get(
    "CONV_ANALOG_YEARS", "2006,2009,2013,2014,2015,2016").split(",") if y.strip()]
SPAN = int(os.environ.get("CONV_ANALOG_SPAN", "0"))
STEP = int(os.environ.get("CONV_ANALOG_STEP", "3"))

def _widen(ic, span, step, years):
    """Every date within +/- span days of the IC's calendar date, in each analogue year."""
    base = np.datetime64(ic, "D")
    offs = list(range(-span, span + 1, step))
    out = []
    for y in years:
        anchor = base + np.timedelta64(int(y) - int(str(base)[:4]), "Y") \
            if False else np.datetime64(f"{y}{str(base)[4:]}", "D")
        for o in offs:
            d = anchor + np.timedelta64(o, "D")
            if str(d)[:4] == str(y):          # do not spill into an adjacent year
                out.append(str(d))
    return sorted(set(out))

STORMS = {}
for _name, _cfg in _BASE.items():
    _c = dict(_cfg)
    if SPAN > 0:
        _c["analogs"] = _widen(_c["ic"], SPAN, STEP, YEARS)
    STORMS[_name] = _c

if SPAN > 0 or RADIUS_KM != _R0:
    _n = len(next(iter(STORMS.values()))["analogs"])
    print(f"[storms_cfg] radius {RADIUS_KM:.0f} km (base {_R0:.0f}); "
          f"analogue span +/-{SPAN} d step {STEP} over {YEARS} -> {_n} candidates/storm",
          flush=True)
