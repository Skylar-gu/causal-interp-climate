"""Storm configuration for the convection-skill-necessity experiment.

Each storm: IC date (00Z formation-ish), approximate center at IC (lat, lon in -180..180),
a storm-tracking box (lat/lon in -180..180), and analog dates (same season, other years)
for the NORMAL-convection reference. Longitudes are stored in -180..180 here; the ERA5
grid is 0..360 so all selection code must normalize (past bug).

Rollout: from each IC to +96h (16 steps). Feature 3243 = TC; convection = [2401,2067,3174].

Paper: Table tab:mechanism-interventions (the frozen seven-storm registry)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.common.skill_conv_storms
"""
import numpy as np

TC = 3243
CONV = [2401, 2067, 3174]
RADIUS_KM = 1500.0
H = 16  # +96h

# Random-feature control: 3 features of matched firing rate to CONV (0.00118, 0.00426, 0.00315),
# NON-joint-coupling (CONV/TC are all 'joint-coupling'), tropical centroid |clat|<25 so they can
# fire on the storm, drawn with seed 7 within 15% firing-rate. Frozen before results.
#   3667 climatology/clock fr0.00133 ; 2875 physics(single) fr0.00445 ; 2850 physics(single) fr0.00269
RANDOM_CTRL = [3667, 2875, 2850]

STORMS = {
    # --- developing RI storms ---
    "ida2021": dict(
        ic="2021-08-26", center=(22.0, -84.0), box=dict(lat=(10, 33), lon=(-98, -58)),
        analogs=["2014-08-27", "2013-08-27", "2009-08-27", "2006-08-27", "2015-08-27"],
        basin="atlantic"),
    # ian2022 DROPPED: WB2 ERA5 zarr ends 2021-12-31, no truth available for 2022.
    "michael2018": dict(
        ic="2018-10-07", center=(21.0, -86.0), box=dict(lat=(17, 32), lon=(-93, -80)),
        analogs=["2014-10-07", "2013-10-07", "2015-10-07", "2009-10-07", "2006-10-07"],
        basin="atlantic"),
    "haishen2020": dict(
        ic="2020-09-03", center=(25.0, 135.0), box=dict(lat=(18, 36), lon=(125, 150)),
        analogs=["2014-09-03", "2013-09-03", "2015-09-03", "2009-09-03", "2006-09-03"],
        basin="wpac"),
    "goni2020": dict(
        ic="2020-10-29", center=(14.0, 130.0), box=dict(lat=(8, 22), lon=(120, 142)),
        analogs=["2014-10-29", "2013-10-29", "2015-10-29", "2009-10-29", "2006-10-29"],
        basin="wpac"),
    "haiyan2013": dict(
        ic="2013-11-05", center=(7.0, 138.0), box=dict(lat=(3, 16), lon=(125, 148)),
        analogs=["2014-11-05", "2015-11-05", "2016-11-05", "2009-11-05", "2006-11-05"],
        basin="wpac"),
    # --- extreme rapid-intensification anchors (biggest expected convection signal) ---
    # Patricia (E-Pacific 2015): peaked 872 hPa ~Oct 23, the strongest E-Pac hurricane on record.
    "patricia2015": dict(
        ic="2015-10-20", center=(13.0, -95.0), box=dict(lat=(9, 22), lon=(-112, -90)),
        analogs=["2013-10-20", "2014-10-20", "2016-10-20", "2009-10-20", "2006-10-20"],
        basin="epac"),
    # Wilma (Atlantic/Caribbean 2005): explosive RI, peaked 882 hPa ~Oct 19, lowest Atlantic MSLP on record.
    "wilma2005": dict(
        ic="2005-10-17", center=(16.5, -79.0), box=dict(lat=(12, 24), lon=(-88, -73)),
        analogs=["2013-10-17", "2014-10-17", "2016-10-17", "2009-10-17", "2006-10-17"],
        basin="atlantic"),
    # --- non-developing control: a tropical Atlantic wave that stayed weak ---
    "nondev2013": dict(
        ic="2013-07-15", center=(13.0, -40.0), box=dict(lat=(6, 20), lon=(-52, -28)),
        analogs=["2014-07-15", "2015-07-15", "2009-07-15", "2006-07-15", "2016-07-15"],
        basin="atlantic", nondev=True),
}

def norm_lon(lon):
    """-180..180 -> 0..360 for ERA5 selection."""
    return np.where(np.asarray(lon) < 0, np.asarray(lon) + 360.0, np.asarray(lon))
