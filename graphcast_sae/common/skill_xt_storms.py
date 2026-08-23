"""Storm configuration for the EXTRATROPICAL battery (explosive cyclogenesis).

Why this is a separate module rather than more entries in `skill_conv_storms.py`.
The TC battery's medians are committed numbers (convection = 2.794 hPa over 7
developing storms). Appending an extratropical case to `STORMS` would silently
change every one of them the next time an arm is re-scored. Selected with
`SKILL_STORMS=skill_xt_storms`; the TC registry is untouched.

WHAT THIS BATTERY IS FOR. Six arms of the mechanism library -- blocking, z500,
jet250, t850, baroclinicity, shear -- returned |delta| <= 0.04 hPa on the TC
battery with ZERO in-box activation in 7 of 7 storms. They were not null, they
were unexposed: the ablation removed nothing because those features never fire
inside a tropical-cyclone box. A bomb cyclone is where they should have exposure
by construction, so this is the testbed that makes those six arms answerable.
See `docs/notes/mechanism_library_2026_08_17.md` section 4.

DESIGN NOTES, each of which is a departure from the TC battery and deliberate:

1. `center` is the TRACK MIDPOINT, not the position at IC. The TC convention
   (disk centred on the storm at IC) does not transfer: this system travels
   ~2,000 km in 72 h, so an IC-centred 1500 km disk would drop the mature phase.
   (38, -70) sits ~1,150 km from both the genesis region (30, -78) and the mature
   position (45, -60), so the disk covers the whole track. The random control uses
   the same disk, so this is not a treatment advantage.

2. `H = 12` (+72 h), not 16. The screener's continuity gate was applied over 72 h
   and the storm is out of the box after that; a longer window risks scoring a
   different system that has moved in.

3. Analogs were chosen by an ERA5 MSLP screen (`graphcast_sae/storms/xt_locate.py`), not
   by the calendar. The runner's built-in analog gate rejects a candidate when
   feature 3243 (TC) fires in the box, and 3243 is INERT in a January North
   Atlantic box -- so that gate would have accepted anything, including another
   bomb. All five analogs are measured at <= 6.9 hPa of 72 h deepening and
   <= 11.1 hPa in any 24 h sub-window. Their IC box minima run 981-996 hPa, which
   is January North Atlantic climatology rather than a quiet field; that is what
   "normal" should mean here.

4. The non-developing control is `nondev_xt2010`, and it was chosen to have a
   967 hPa low PRESENT at IC that never deepens (72 h drop 0.0, worst 24 h
   -3.7). A control on an empty field could not fail -- no features fire, so no
   ablation can cost anything, which is the vacuous-control trap that voided the
   six arms in the first place. This one has exposure and still must return zero.

TC = 3243 is retained for tracking continuity only. `tc_supp` is MEANINGLESS for
an extratropical cyclone and must not be quoted from this battery.

Paper: Sec. 3 (extratropical battery registry)
Inputs: none beyond the arguments above
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.common.skill_xt_storms
"""
import numpy as np

from graphcast_sae.common.skill_conv_storms import CONV, RADIUS_KM, TC, norm_lon  # noqa: F401  (shared)

H = 12  # +72h

# EXTRATROPICAL RANDOM CONTROL, drawn from the exposure probe (2026-08-17).
# `skill_conv_storms.RANDOM_CTRL` was drawn with "tropical centroid |clat|<25 so
# they CAN fire on the storm". At 40 N those three features cannot fire, so
# reusing them here would install a control that cannot fail -- and the probe
# CONFIRMED it: [3667, 2875, 2850] have in-box activation 0.00 on eastcoast2018.
#
# Selection, frozen before any treatment arm was scored: label == 'ambiguous'
# under the calibrated rule (so not one of the four mechanisms), live
# (n_fire > 500), disjoint from every treatment group, measured in-box peak
# activation inside the treatment features' own range (13.9-70.0 -> pool bound
# 7.9-88.3) and n_fire inside theirs (1,181-6,298 -> pool bound 451-6,298).
# 149 features qualified; 4 drawn with seed 7, matching the TC battery's
# convention. Group in-box exposure 18.06, i.e. inside the treatment groups'
# 3.47-24.57 -- so this control CAN fail, which the tropical one could not.
RANDOM_CTRL = [2487, 2820, 3757, 3819]

# Measured in-box exposure on eastcoast2018 (peak-in-time group-mean activation,
# baseline arm) against the same statistic on the TC battery. This is the table
# that decides which arms are answerable here; t850 and z500 stay at zero even in
# a bomb cyclone and are NOT run, because a no-op ablation cannot be scored.
XT_EXPOSURE = {          # group: (extratropical box, TC-battery box)
    "baroclinicity": (24.57, 0.33),    # 74x -- the biggest gain in the library
    "convection":    (23.32, 39.55),   # still exposed, 41% down from the tropics
    "jet250":        (22.08, 0.00),    # 0 -> 22, answerable for the first time
    "atm_river":     (10.60, 25.45),
    "vort850":       (8.81, 3.07),     # 2.9x
    "shear":         (8.73, 0.00),     # answerable
    "blocking":      (3.47, 0.00),     # answerable but thin
    "moisture2":     (1.07, 2.14),
    "q600":          (0.00, 8.93),     # LOSES all exposure outside the tropics
    "t850":          (0.00, 0.00),     # unexposed here too -> not run
    "z500":          (0.00, 0.00),     # unexposed here too -> not run
    "TROPICAL_rand": (0.00, None),     # the vacuous-control prediction, confirmed
}

STORMS = {
    # --- the treatment: January 2018 East Coast bomb cyclone ("Grayson") -------
    # 1010.8 -> 953.3 hPa in the box over 72 h; 58.0 hPa in the worst 24 h,
    # = 3.4 Bergeron at this latitude, continuity-clean (0 jumps > 500 km/6 h).
    # The strongest cyclone in the 13-event screen by a factor of two.
    "eastcoast2018": dict(
        ic="2018-01-03", center=(38.0, -70.0), box=dict(lat=(30, 46), lon=(-77, -58)),
        analogs=["2013-01-03", "2002-01-08", "2021-01-08", "2001-01-08", "1994-01-15"],
        basin="wnatl"),
    # --- non-developing control: deep low present, zero development -----------
    "nondev_xt2010": dict(
        ic="2010-01-03", center=(42.0, -65.0), box=dict(lat=(30, 46), lon=(-77, -58)),
        analogs=["2013-01-03", "2002-01-08", "2021-01-08", "2001-01-08", "1994-01-15"],
        basin="wnatl", nondev=True),
}

# Measured by graphcast_sae/storms/xt_locate.py, kept here so the selection is auditable.
ANALOG_SCREEN = {           # date: (72h drop hPa, worst 24h drop hPa, IC box min hPa)
    "2013-01-03": (0.0, 2.7, 989.2),
    "2002-01-08": (6.9, 6.8, 995.8),
    "2021-01-08": (0.5, 7.4, 988.9),
    "2001-01-08": (0.0, 9.8, 981.6),
    "1994-01-15": (0.0, 11.1, 981.4),
    "2010-01-03": (0.0, -3.7, 967.1),   # the nondev control, not an analog
}
