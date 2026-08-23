"""ERA5-only detection + gating sweep for a large tropical-cyclone battery.

WHY THIS EXISTS. The convection-necessity design has 13 storms. IC offsets do not buy
independent draws; storms do. This sweeps the WB2 ERA5 record for closed, deepening,
translating tropical MSLP minima and gates each one on ERA5 alone, so that a candidate
enters the registry only if ERA5 itself shows it developing.

NOTHING HERE READS THE MODEL. No GPU, no SAE, no forecast. Every quantity is ERA5 MSLP
(plus land_sea_mask for the ocean-genesis check). That is deliberate: this is a data gate
on the input, in the sense of the data-gate rule, not an outcome measurement.

METHOD, in the shape of graphcast_sae/storms/locate_sh_storms.py -- wide basin boxes, and ERA5
places the storm rather than recall placing it:

  1. DETECT. Per basin x season, take MSLP on a 0.5 deg subgrid. A candidate centre is a
     grid point that is the minimum of its +-2.5 deg neighbourhood, below 1008 hPa, and at
     least 2 hPa below the neighbourhood maximum (a depression, not a flat trough).
  2. TRACK. Greedy nearest-neighbour linking at <= 400 km per 6 h. Tracks shorter than
     96 h are dropped -- the design needs a storm that exists for the whole window.
  3. IC. Among the 00Z times on the track with >= 96 h of track remaining, the IC is the one
     maximising 96-h deepening. This is amendment A1 of the SH battery applied uniformly.
     Under --nondev the sign flips: the IC MINIMISING deepening, i.e. the quietest window
     of a persistent tropical low, which is the same object as the nondev2013 control.
  4. GATE. See gate() -- deepening, tropical latitude, ocean genesis, translation,
     interior-of-domain, headroom in the zarr, and no dateline crossing.
  5. BOX. Track extent + 5 deg lat / 6 deg lon, grown until it holds >= MIN_NODES mesh
     nodes, clipped to the basin, east edge trimmed to 179.9 if it would touch the dateline.
     box_nodes is measured with the SAME raw comparison skill_conv_run.py uses.
  6. ANALOGS. Same calendar date in other years, screened on ERA5 box-min MSLP. The model
     screen (TC feature sum > 20) cannot be run here, so this is a PROXY, calibrated below.

ANALOG PROXY CALIBRATION (measured, results/skill/*/run_*.npy, 65 offered analog dates):
every analog the model accepted had ERA5 box-min MSLP >= 1001.1 hPa, and every analog with
box-min <= 999 was rejected. Between 1001 and 1008 the model still rejects some -- ida2021's
box is 839 mesh nodes over the whole Caribbean and Gulf in peak season, and its TC feature
sum clears 20 on disturbances with no MSLP signature at all, which is how it ended with 1
analog of 5. Two consequences, both applied here: boxes are kept far smaller than ida's, and
EIGHT analog dates are offered per storm instead of five, quietest first, so that >= 3
surviving is robust to a proxy miss.

Paper: not in the paper; kept for provenance only
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/mega_storm_gate.json (or results/mega_storm_gate_<tag>.json per batch) + a run log
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.mega_sweep --years 2010-2021
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter

from graphcast_sae.paths import REPO_ROOT, MESH_GEOM
ROOT = str(REPO_ROOT)
ZARR = "weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"
MESH = MESH_GEOM
OUT = os.path.join(ROOT, "results", "mega_storm_gate.json")

STEP = np.timedelta64(6, "h")
NLEAD = 17                      # IC .. +96 h inclusive
H = 16

# --- detection ---
STRIDE = 2                      # 0.25 deg -> 0.5 deg for the centre search
NEIGH_DEG = 2.5                 # half-width of the local-minimum neighbourhood
P_CAND = 1008.0                 # a centre must be at least this deep to be a candidate
DEPRESSION = 2.0                # centre must be >= this far below its neighbourhood max
LINK_KM = 400.0                 # max 6-h displacement when linking centres into a track

# --- gate bars (the NH battery spans 18.7 - 33.2 hPa of ERA5 deepening) ---
DEEP_PRIMARY = 18.7
DEEP_SECONDARY = 12.0
NONDEV_MAX = 5.0                # a non-developing control must deepen LESS than this
P_MIN_REQ = 995.0               # ERA5 must take the storm at least this deep in the window
# GENESIS, not IC, carries the "is this tropical" test. Katrina's 96-h-deepening-optimal IC
# is 2005-08-26 at 26.0 N, which an IC-latitude gate of 5-25 rejects outright; its TRACK
# begins in the deep tropics, which is the physical statement the gate is trying to make.
LAT_GEN = (2.0, 28.0)           # |lat| where the track's closed low FIRST appears
LAT_IC = (5.0, 35.0)            # |lat| at IC
LAT_MAX = 42.0                  # |lat| anywhere in the window
# CONTINUITY (the event_screen.py lesson): the full-resolution box minimum must be the
# tracked storm at every lead, not a second low that moved into the box.
CONT_KM = 500.0
CONT_DEEP_SLACK = 8.0
DRIFT_KM = 100.0                # the minimum must translate; a pinned trough is not a TC
MIN_NODES = 140                 # mesh nodes in the box (the brief's floor is ~120)
EDGE_DEG = 1.5                  # centre must stay this far inside the basin domain

# --- zarr headroom: IC-48 h (planned offsets) .. IC+96 h must be inside the record ---
T_LO = np.datetime64("1979-01-05")
T_HI = np.datetime64("2021-12-20")

# --- analogs ---
ANALOG_MIN_MSLP = 1004.0        # ERA5 proxy for "no storm in the box"
N_ANALOG = 8
ANALOG_YEARS = list(range(1979, 2022))

BASINS = {
    # name:  (lat0, lat1, lon0, lon1) in 0..360, season as (month_start, month_end)
    "atlantic": dict(dom=(5, 35, 262, 350), season=(6, 11), hemi=1),
    "epac":     dict(dom=(5, 30, 190, 262), season=(6, 11), hemi=1),
    "wpac":     dict(dom=(3, 35, 105, 180), season=(5, 12), hemi=1),
    "nind":     dict(dom=(5, 25, 52, 100), season=(4, 12), hemi=1),
    "sind":     dict(dom=(-30, -5, 35, 100), season=(-10, 5), hemi=-1),
    "aus":      dict(dom=(-30, -5, 100, 160), season=(-10, 5), hemi=-1),
    # spac stops at 179 deg E on purpose: skill_conv_run.py compares raw -180..180
    # longitudes, so any box crossing the dateline yields ZERO mesh nodes. A storm east
    # of the line cannot be represented and is rejected rather than silently emitted.
    "spac":     dict(dom=(-30, -5, 155, 179), season=(-10, 5), hemi=-1),
}

# the 13 already in the battery -- name, ic, centre (lat, lon in -180..180)
EXISTING = [
    ("ida2021", "2021-08-26", 22.0, -84.0), ("michael2018", "2018-10-07", 21.0, -86.0),
    ("haishen2020", "2020-09-03", 25.0, 135.0), ("goni2020", "2020-10-29", 14.0, 130.0),
    ("haiyan2013", "2013-11-05", 7.0, 138.0), ("patricia2015", "2015-10-20", 13.0, -95.0),
    ("wilma2005", "2005-10-17", 16.5, -79.0), ("nondev2013", "2013-07-15", 13.0, -40.0),
    ("winston2016", "2016-02-21", -17.2, 175.2), ("harold2020", "2020-04-01", -9.5, 155.0),
    ("marcus2018", "2018-03-18", -14.2, 128.0), ("veronica2019", "2019-03-19", -14.0, 121.0),
    ("fantala2016", "2016-04-17", -11.0, 53.5),
]

def km(lat1, lon1, lat2, lon2):
    dla = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlo = np.radians((np.asarray(lon2) - np.asarray(lon1) + 180.0) % 360.0 - 180.0)
    ml = np.radians(0.5 * (np.asarray(lat1) + np.asarray(lat2)))
    return 6371.0 * np.hypot(dla, dlo * np.cos(ml))

def to180(x):
    x = np.asarray(x, float)
    return np.where(x > 180.0, x - 360.0, x)

def open_zarr():
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(ZARR), consolidated=True)
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    return ds

# ------------------------------------------------------------------ detection ----
def detect_centres(v, la, lo):
    """v (T,Y,X) hPa on the 0.5 deg subgrid -> list per time of (lat, lon, p)."""
    dl = abs(float(la[1] - la[0]))
    n = max(1, int(round(NEIGH_DEG / dl)))
    size = (1, 2 * n + 1, 2 * n + 1)
    lo_f = minimum_filter(v, size=size, mode="nearest")
    hi_f = maximum_filter(v, size=size, mode="nearest")
    m = (v == lo_f) & (v < P_CAND) & (hi_f - v >= DEPRESSION)
    out = []
    ti, yi, xi = np.nonzero(m)
    for t in range(v.shape[0]):
        s = ti == t
        out.append([(float(la[y]), float(lo[x]), float(v[t, y, x]))
                    for y, x in zip(yi[s], xi[s])])
    return out

def link_tracks(cent, times):
    """Greedy nearest-neighbour linking. Returns list of dicts with time-indexed track."""
    tracks, live = [], []
    for t, cs in enumerate(cent):
        taken = set()
        nxt = []
        for tr in live:
            best, bd = None, LINK_KM
            for i, c in enumerate(cs):
                if i in taken:
                    continue
                d = km(tr["pos"][-1][0], tr["pos"][-1][1], c[0], c[1])
                if d < bd:
                    best, bd = i, d
            if best is None:
                tracks.append(tr)
                continue
            taken.add(best)
            tr["pos"].append(cs[best])
            nxt.append(tr)
        for i, c in enumerate(cs):
            if i not in taken:
                nxt.append(dict(t0=t, pos=[c]))
        live = nxt
    tracks.extend(live)
    for tr in tracks:
        tr["times"] = times[tr["t0"]:tr["t0"] + len(tr["pos"])]
    return tracks

# ----------------------------------------------------------------------- gate ----
def gate(tr, dom, lsm_at, want_nondev=False):
    """Pick the best IC on one track and gate it. Returns (rec, reason) -- rec is None
    when the track cannot yield an acceptable IC."""
    pos = np.array(tr["pos"], float)                 # (n, 3) lat, lon, p
    ts = tr["times"]
    n = len(pos)
    if n < NLEAD:
        return None, "track shorter than 96 h"
    best = None
    for i in range(n - NLEAD + 1):
        t0 = ts[i]
        if (t0.astype("datetime64[h]").astype(int) % 24) != 0:
            continue                                  # ICs are 00Z, as in the registry
        if not (T_LO <= t0 <= T_HI):
            continue                                  # zarr headroom for -48 h .. +96 h
        seg = pos[i:i + NLEAD]
        deep = float(seg[0, 2] - seg[:, 2].min())
        # developers: the IC that maximises 96-h deepening (SH amendment A1). Non-developing
        # controls: the IC that MINIMISES it, i.e. the quietest 96-h window of a persistent
        # tropical low -- the same object as nondev2013, "a wave that stayed weak".
        if best is None or (deep < best[1] if want_nondev else deep > best[1]):
            best = (i, deep)
    if best is None:
        return None, "no 00Z IC with 96 h of record and track remaining"
    i, deep = best
    seg = pos[i:i + NLEAD]
    ic = str(ts[i].astype("datetime64[D]"))
    la0, la1, lo0, lo1 = dom
    imin = int(np.argmin(seg[:, 2]))
    drift = float(km(seg[0, 0], seg[0, 1], seg[imin, 0], seg[imin, 1]))
    rec = dict(ic=ic, center=[float(seg[0, 0]), float(to180(seg[0, 1]))],
               era5_deepen=round(deep, 1), mslp_ic=round(float(seg[0, 2]), 1),
               mslp_min=round(float(seg[:, 2].min()), 1), t_min_h=6 * imin,
               drift_km=round(drift), track=[[round(a, 2), round(b, 2), round(c, 1)]
                                             for a, b, c in seg])
    rec["genesis"] = [float(pos[0, 0]), float(to180(pos[0, 1])), int(tr["t0"])]
    lat_ic = abs(seg[0, 0])
    if not (LAT_IC[0] <= lat_ic <= LAT_IC[1]):
        return rec, f"IC latitude {seg[0,0]:.1f} outside {LAT_IC}"
    # genesis test: where the closed low FIRST appears. If the track was already alive at
    # the first loaded time its genesis is unobserved here, so fall back to the IC.
    glat = abs(pos[0, 0]) if tr["t0"] > 0 else lat_ic
    if not (LAT_GEN[0] <= glat <= LAT_GEN[1]):
        return rec, f"genesis latitude {glat:.1f} outside {LAT_GEN} (not a tropical low)"
    if np.abs(seg[:, 0]).max() > LAT_MAX:
        return rec, f"track reaches |lat| {np.abs(seg[:,0]).max():.1f} > {LAT_MAX}"
    edge = ((np.abs(seg[:, 0] - la0) < EDGE_DEG).any() or (np.abs(seg[:, 0] - la1) < EDGE_DEG).any()
            or (np.abs(seg[:, 1] - lo0) < EDGE_DEG).any() or (np.abs(seg[:, 1] - lo1) < EDGE_DEG).any())
    if edge:
        return rec, "track touches the basin-domain edge (centre may be outside)"
    if lsm_at(pos[0, 0], pos[0, 1]) >= 0.5 or lsm_at(seg[0, 0], seg[0, 1]) >= 0.5:
        return rec, "genesis / IC centre over land"
    if want_nondev:
        if deep > NONDEV_MAX:
            return rec, f"nondev candidate deepens {deep:.1f} hPa > {NONDEV_MAX}"
        if seg[:, 2].min() < 1000.0:
            return rec, f"nondev candidate reaches {seg[:,2].min():.1f} hPa, too deep"
        rec["nondev"] = True
        return rec, None
    if drift < DRIFT_KM:
        return rec, f"minimum drifts only {drift:.0f} km (pinned, not a TC)"
    if seg[:, 2].min() > P_MIN_REQ:
        return rec, f"ERA5 min {seg[:,2].min():.1f} hPa never reaches {P_MIN_REQ}"
    if deep < DEEP_SECONDARY:
        return rec, f"ERA5 deepening {deep:.1f} hPa below the {DEEP_SECONDARY} hPa floor"
    rec["secondary"] = bool(deep < DEEP_PRIMARY)
    return rec, None

# ------------------------------------------------------------------------ box ----
def make_box(rec, dom, mlat, mlon):
    """Track extent + margin, grown to >= MIN_NODES mesh nodes. -180..180, dateline-safe."""
    seg = np.array(rec["track"], float)
    la0, la1, lo0, lo1 = dom
    tlat0, tlat1 = seg[:, 0].min(), seg[:, 0].max()
    tlon0, tlon1 = seg[:, 1].min(), seg[:, 1].max()
    if tlon1 - tlon0 > 180:
        return None, "track crosses the dateline (box would be empty on a raw -180..180 test)"
    mlat_pad, mlon_pad = 5.0, 6.0
    for _ in range(12):
        b0 = max(la0, tlat0 - mlat_pad)
        b1 = min(la1, tlat1 + mlat_pad)
        c0 = max(lo0 - 5.0, tlon0 - mlon_pad)
        c1 = min(lo1 + 5.0, tlon1 + mlon_pad)
        # c0, c1 are 0..360 with c0 < c1. The box wraps in -180..180 exactly when the
        # interval straddles 180. skill_conv_run.py tests `mlon >= lo0 and mlon <= lo1` on
        # raw -180..180 longitudes, so such a box silently selects ZERO mesh nodes -- the
        # winston2016 failure mode. Trim the east edge when the track allows it, else reject.
        if c0 < 180.0 < c1:
            if tlon1 <= 179.9:
                c1 = 179.9
            else:
                return None, "box crosses the dateline (raw -180..180 test yields 0 nodes)"
        d0, d1 = float(to180(c0)), float(to180(c1))
        if d0 >= d1:
            return None, "box wraps in -180..180"
        # count on the ROUNDED edges -- those are what the registry emits, and what
        # skill_conv_run.py will actually compare against. Counting on the unrounded ones
        # made box_nodes a number the shipped box does not reproduce.
        r = [round(float(b0), 1), round(float(b1), 1), round(d0, 1), round(d1, 1)]
        inb = ((mlat >= r[0]) & (mlat <= r[1]) & (mlon >= r[2]) & (mlon <= r[3]))
        nn = int(inb.sum())
        if nn >= MIN_NODES:
            return dict(lat=[r[0], r[1]], lon=[r[2], r[3]], box_nodes=nn), None
        mlat_pad += 1.5
        mlon_pad += 2.5
    return None, f"box cannot reach {MIN_NODES} mesh nodes inside the basin domain"

def warm_core(ds, lat, lon, t):
    """300-850 hPa thickness at the centre minus a 500-1000 km ring, in metres.

    REPORTED, NOT GATED -- and that is a measured decision, not laziness.

    It was intended as the gate separating a tropical-cyclone battery from a
    maritime-deepening-low one, and it was calibrated on both sides first
    (graphcast_sae/legacy/mega_calibrate.py, guardrail #9). It FAILED: the 12 tropical cyclones
    already in the battery span 26.4 - 110.1 m (median 48.3), while five explosive
    EXTRATROPICAL cyclones -- the negative control -- span -182.3 to 101.6 (median 68.9).
    eastcoast2018 (101.6), greatlakes2010 (82.3) and dennis2020 (68.9) all score ABOVE the
    TC median, because a mature occluded cyclone has a warm seclusion and the 500-1000 km
    ring straddles a baroclinic zone. No threshold separates the two populations, so no
    threshold is applied. The tropical requirement is carried instead by genesis latitude,
    ocean genesis, the basin domains and LAT_MAX. The value is stored per storm so the
    contamination question stays auditable rather than being quietly dropped.
    """
    lon360 = float(lon) % 360.0
    z = ds["geopotential"].sel(time=np.datetime64(t), level=[300, 850],
                               lat=slice(lat - 12, lat + 12))
    if lon360 - 12 < 0 or lon360 + 12 > 360:
        z = z.sel(lon=(ds.lon >= (lon360 - 12) % 360) | (ds.lon <= (lon360 + 12) % 360))
    else:
        z = z.sel(lon=slice(lon360 - 12, lon360 + 12))
    z = z.load()
    la = np.asarray(z.lat.values, float)
    lo = np.asarray(z.lon.values, float)
    th = (z.sel(level=300).values - z.sel(level=850).values) / 9.80665
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    d = km(LA, LO, float(lat), lon360)
    core = float(np.nanmean(th[d < 300]))
    ring = float(np.nanmean(th[(d > 500) & (d < 1000)]))
    return core - ring

def confirm_box(ds, rec):
    """Recompute deepening at FULL resolution in the emitted box, the way
    skill_conv_verify_era5.py will, and enforce continuity against the tracked centre.

    The 0.5 deg detection number is not the number the pipeline reports, so gating on it
    would leave the registry's `era5_deepen` unverified. This reads exactly what the
    downstream readout reads: min MSLP in the box at leads 0 .. +96 h.
    """
    box = rec["box"]
    lo = np.asarray(box["lon"], float) % 360.0
    t = np.datetime64(rec["ic"]) + np.arange(NLEAD) * STEP
    sub = ds["mean_sea_level_pressure"].sel(time=t, lat=slice(*box["lat"]))
    sub = (sub.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1]
           else sub.sel(lon=(ds.lon >= lo[0]) | (ds.lon <= lo[1])))
    sub = sub.load()
    v = np.asarray(sub.values, float) / 100.0
    la = np.asarray(sub.lat.values, float)
    ln = to180(np.asarray(sub.lon.values, float))
    pos = []
    for i in range(v.shape[0]):
        j = np.unravel_index(np.nanargmin(v[i]), v[i].shape)
        pos.append((la[j[0]], float(ln[j[1]]), v[i][j]))
    pos = np.array(pos, float)
    trk = np.array(rec["track"], float)
    trk_lon = to180(trk[:, 1])
    d = km(pos[:, 0], pos[:, 1], trk[:, 0], trk_lon)
    deep = float(pos[0, 2] - pos[:, 2].min())
    rec["era5_deepen_track"] = rec["era5_deepen"]
    rec["era5_deepen"] = round(deep, 1)
    rec["mslp_ic"] = round(float(pos[0, 2]), 1)
    rec["mslp_min"] = round(float(pos[:, 2].min()), 1)
    rec["t_min_h"] = int(6 * np.argmin(pos[:, 2]))
    rec["boxmin_offtrack_km_max"] = round(float(d.max()))
    rec["box_track"] = [[round(a, 2), round(b, 2), round(c, 1)] for a, b, c in pos]
    ipk = int(np.argmin(pos[:, 2]))
    try:    # reported, never gated -- see warm_core's docstring
        rec["warm_core_m"] = round(warm_core(ds, pos[ipk, 0], pos[ipk, 1], t[ipk]), 1)
    except Exception:
        rec["warm_core_m"] = None
    if d.max() > CONT_KM:
        return f"box minimum leaves the tracked storm by {d.max():.0f} km (advective, not developmental)"
    if deep - rec["era5_deepen_track"] > CONT_DEEP_SLACK:
        return (f"box deepening {deep:.1f} exceeds tracked {rec['era5_deepen_track']:.1f} hPa "
                f"by more than {CONT_DEEP_SLACK} (a second low is in the box)")
    if rec.get("nondev"):
        if deep > NONDEV_MAX:
            return f"nondev box deepening {deep:.1f} hPa > {NONDEV_MAX}"
        return None
    if deep < DEEP_SECONDARY:
        return f"full-resolution box deepening {deep:.1f} hPa below the {DEEP_SECONDARY} hPa floor"
    if rec["mslp_min"] > P_MIN_REQ:
        return f"full-resolution box min {rec['mslp_min']:.1f} hPa never reaches {P_MIN_REQ}"
    rec["secondary"] = bool(deep < DEEP_PRIMARY)
    return None

# -------------------------------------------------------------------- analogs ----
def analog_dates(ic):
    y0 = int(ic[:4])
    out = []
    for y in ANALOG_YEARS:
        if y == y0:
            continue
        try:
            d = np.datetime64(f"{y}-{ic[5:]}")
        except Exception:
            continue
        if T_LO <= d <= np.datetime64("2021-12-30"):
            out.append(str(d))
    # prefer years far from the storm year, then spread; ordering is only a tie-break,
    # the real ranking is by ERA5 quietness below.
    return out

def screen_analogs(ds, box, cands):
    """ERA5 box-min MSLP at 00Z on each candidate date. Returns sorted (date, mslp)."""
    if not cands:
        return []
    lo = np.asarray(box["lon"], float) % 360.0
    t = np.array([np.datetime64(c) for c in cands])
    sub = ds["mean_sea_level_pressure"].sel(time=t, lat=slice(*box["lat"]))
    sub = (sub.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1]
           else sub.sel(lon=(ds.lon >= lo[0]) | (ds.lon <= lo[1])))
    v = np.asarray(sub.load().values, float) / 100.0
    mn = np.nanmin(v.reshape(v.shape[0], -1), axis=1)
    return sorted(zip(cands, [round(float(x), 1) for x in mn]), key=lambda r: -r[1])

# ---------------------------------------------------------------------- sweep ----
def season_slices(basin, year):
    m0, m1 = BASINS[basin]["season"]
    if m0 > 0:
        return (np.datetime64(f"{year}-{m0:02d}-01"),
                np.datetime64(f"{year}-{m1:02d}-28") + np.timedelta64(3, "D"))
    # SH: Oct(year) .. May(year+1), loaded contiguously so a New-Year storm is not split
    return (np.datetime64(f"{year}-{-m0:02d}-01"),
            np.datetime64(f"{year+1}-{m1:02d}-28") + np.timedelta64(3, "D"))

def sweep_cell(ds, lsm_at, mlat, mlon, basin, year, log, want_nondev=False):
    dom = BASINS[basin]["dom"]
    t0, t1 = season_slices(basin, year)
    if t1 < np.datetime64("1979-01-01") or t0 > np.datetime64("2021-12-31"):
        return [], []
    t = time.time()
    sub = ds["mean_sea_level_pressure"].sel(
        time=slice(t0, t1), lat=slice(dom[0], dom[1]), lon=slice(dom[2], dom[3]))
    sub = sub.isel(lat=slice(None, None, STRIDE), lon=slice(None, None, STRIDE)).load()
    v = np.asarray(sub.values, np.float32) / 100.0
    la = np.asarray(sub.lat.values, float)
    lo = np.asarray(sub.lon.values, float)
    times = np.asarray(sub.time.values)
    tload = time.time() - t

    cent = detect_centres(v, la, lo)
    tracks = [tr for tr in link_tracks(cent, times) if len(tr["pos"]) >= NLEAD]

    accepted, rejected = [], []
    for tr in tracks:
        rec, why = gate(tr, dom, lsm_at, want_nondev=want_nondev)
        if rec is None:
            continue
        rec["basin"] = basin
        if why:
            rec["reject"] = why
            rejected.append(rec)
            continue
        bx, why = make_box(rec, dom, mlat, mlon)
        if bx is None:
            rec["reject"] = why
            rejected.append(rec)
            continue
        rec["box"] = dict(lat=tuple(bx["lat"]), lon=tuple(bx["lon"]))
        rec["box_nodes"] = bx["box_nodes"]
        accepted.append(rec)
    log(f"  {basin:<9}{year}  load {tload:5.1f}s  {v.shape[0]:4d}t  tracks {len(tracks):3d}"
        f"  accept {len(accepted):2d}  reject {len(rejected):3d}")
    return accepted, rejected

def dedupe(cands, days=7, dist=1500.0, nondev=False):
    """Strongest-first greedy: two ICs within `days` and `dist` are the same storm-week."""
    out = []
    for c in sorted(cands, key=(lambda r: r["era5_deepen"]) if nondev
                    else (lambda r: -r["era5_deepen"])):
        t = np.datetime64(c["ic"])
        dup = None
        for o in out:
            if abs((t - np.datetime64(o["ic"])) / np.timedelta64(1, "D")) <= days and \
               km(c["center"][0], c["center"][1], o["center"][0], o["center"][1]) < dist:
                dup = o["name"] if "name" in o else o["ic"]
                break
        if dup:
            c["reject"] = f"duplicate of the same storm-week as {dup}"
            continue
        out.append(c)
    return out

def clash_with_existing(c):
    t = np.datetime64(c["ic"])
    for nm, ic, la, lo in EXISTING:
        if abs((t - np.datetime64(ic)) / np.timedelta64(1, "D")) <= 10 and \
           km(c["center"][0], c["center"][1], la, lo) < 2000.0:
            return nm
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2010-2021")
    ap.add_argument("--basins", default=",".join(BASINS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--no-analogs", action="store_true")
    ap.add_argument("--nondev", action="store_true",
                    help="sweep for NON-developing controls instead of developers")
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    basins = a.basins.split(",")

    ds = open_zarr()
    lsm = ds["land_sea_mask"].load()
    lsmv = np.asarray(lsm.values, float)
    if lsmv.ndim == 3:
        lsmv = lsmv[0]
    lsm_lat = np.asarray(lsm.lat.values, float)
    lsm_lon = np.asarray(lsm.lon.values, float)

    def lsm_at(lat, lon):
        i = int(np.abs(lsm_lat - lat).argmin())
        j = int(np.abs(lsm_lon - (lon % 360.0)).argmin())
        return float(lsmv[i, j])

    geom = np.load(MESH, allow_pickle=True).item()
    mlat = np.asarray(geom["lat"], float)
    mlon = np.where(np.asarray(geom["lon"], float) > 180.0,
                    np.asarray(geom["lon"], float) - 360.0, np.asarray(geom["lon"], float))

    logf = open(os.path.join(ROOT, "out", f"mega_sweep{a.tag}.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== sweep {a.years} basins={basins} at {time.strftime('%H:%M:%S')} ===")
    acc, rej = [], []
    for basin in basins:
        for year in range(y0, y1 + 1):
            try:
                A, R = sweep_cell(ds, lsm_at, mlat, mlon, basin, year, log,
                                  want_nondev=a.nondev)
            except Exception as e:
                log(f"  {basin:<9}{year}  ERROR {type(e).__name__}: {e}")
                continue
            for r in A:
                r["year"] = year
            acc.extend(A)
            rej.extend(R)

    # ORDER MATTERS. Drop the storms already in the battery FIRST, then de-duplicate, then
    # cap per basin-year. Capping first would let a duplicate of an existing entry consume
    # a slot and silently discard a good new storm (Atlantic 2005: Wilma is already in the
    # registry, so capping at 2 first would have thrown Rita away).
    keep = []
    for c in acc:
        nm = clash_with_existing(c)
        if nm:
            c["reject"] = f"same storm-week as the existing registry entry {nm}"
            rej.append(c)
        else:
            keep.append(c)
    keep = dedupe(keep, nondev=a.nondev)
    per = {}
    capped = []
    rank = (lambda r: r["era5_deepen"]) if a.nondev else (lambda r: -r["era5_deepen"])
    for c in sorted(keep, key=rank):
        k = (c["basin"], c["year"])
        if per.get(k, 0) >= a.per_cell:
            c["reject"] = f"beyond the {a.per_cell}-per-basin-year cap"
            rej.append(c)
            continue
        per[k] = per.get(k, 0) + 1
        capped.append(c)
    keep = capped

    # full-resolution confirmation in the emitted box (this is the number that ships)
    conf = []
    for c in keep:
        try:
            why = confirm_box(ds, c)
        except Exception as e:
            why = f"confirmation read failed: {type(e).__name__}: {e}"
        if why:
            c["reject"] = why
            rej.append(c)
        else:
            conf.append(c)
    keep = conf
    keep.sort(key=lambda r: (r["basin"], r["ic"]))
    for c in keep:
        c["name"] = (("nd" if c.get("nondev") else "") +
                     f"{c['basin'][:3]}{c['ic'][:4]}_{c['ic'][5:7]}{c['ic'][8:10]}")

    if not a.no_analogs:
        for c in keep:
            t = time.time()
            sc = screen_analogs(ds, c["box"], analog_dates(c["ic"]))
            quiet = [d for d, p in sc if p >= ANALOG_MIN_MSLP]
            c["analogs"] = quiet[:N_ANALOG]
            c["analog_mslp"] = dict(sc[:N_ANALOG + 4])
            c["analogs_quiet_n"] = len(c["analogs"])
            log(f"  analogs {c['name']:<16} {c['analogs_quiet_n']} offered "
                f"(pool {len(sc)} quiet {len(quiet)})  {time.time()-t:.1f}s")

    path = OUT if not a.tag else OUT.replace(".json", f"{a.tag}.json")
    json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M"), years=a.years,
                   basins=basins, accepted=keep, rejected=rej),
              open(path, "w"), indent=1)
    log(f"\n{len(keep)} accepted, {len(rej)} rejected -> {path}")
    for c in keep:
        log(f"  {c['name']:<16} {c['basin']:<9} ic {c['ic']} centre "
            f"({c['center'][0]:6.1f},{c['center'][1]:7.1f})  deep {c['era5_deepen']:5.1f} "
            f"min {c['mslp_min']:6.1f}  nodes {c['box_nodes']:4d} "
            f"analogs {c.get('analogs_quiet_n','-')}"
            f"{'  SECONDARY' if c.get('secondary') else ''}")

if __name__ == "__main__":
    main()
