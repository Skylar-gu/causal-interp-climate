"""PS-5 -- where the convection features sit relative to the ambient shear vector.

THE TEST. In a sheared tropical cyclone, convection concentrates DOWNSHEAR-LEFT in the
Northern Hemisphere and downshear-RIGHT in the Southern. This is among the most robust
composites in TC observation. `docs/notes/spec_closure_scaling.md` pre-registered three
predictions:

  P1  the displacement angle clusters near +90 deg (downshear-left) for NH storms
  P2  the displacement MAGNITUDE grows with |S|
  P3  the sign of the angle FLIPS in the Southern Hemisphere

P3 is the strongest test in the spec: an artifact has no access to the shear vector's
orientation at all, and no latitude-dependent or magnitude-only confound can reverse a sign
with hemisphere. It was untestable until `skill_sh_storms` existed.

WHY THIS FILE EXISTS. The P1/P2 numbers were produced ad hoc and the script was not kept, so
the result was a number in a note with no reproducible code behind it. This is that code.
It re-derives P1 and P2 from the stored runs and extends to P3.

FOUR DEFINITIONS THAT DECIDE THE ANSWER, all fixed here and all corrections to the first
ad-hoc pass, which got a sign-flipping result from getting them wrong:

  1. shear is measured at +48 h, NOT at the initial condition -- the displacement is at +48 h
     and pairing it with the IC shear compares two different atmospheres
  2. the storm centre is the MSLP minimum at +48 h, NOT the configured IC centre -- centre
     relocation over 48 h runs 114-689 km, so using the IC centre measures storm MOTION
  3. shear is averaged over 0-500 km of the centre, NOT 1500 km -- the 1500 km disk averages
     the storm's own outflow into the "ambient" flow
  4. shear is a VECTOR difference of the area-mean winds, NOT a mean of scalar shear
     magnitudes, which has no direction and cannot be used for an angle at all

PS-4 was WITHDRAWN over exactly this: its shear-vs-efficacy correlation is -0.734 under the
first definition and +0.449 under this one. The sign flips with the definition, so the
quantity was never measured. Anything read off this script inherits that warning.

    SKILL_STORMS=skill_sh_storms MECH_RES=sh_convection \

Paper: not in the paper (results shipped: results/skill/convection/ps5_displacement.json)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.ps5_shear_displacement
"""
import importlib
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
RES = fc.ROOT / "results/skill" / os.environ.get("MECH_RES", "convection")
MID = 7                     # the captured snapshot: h=7 -> +48 h
R_SHEAR = 500.0             # km, ambient-shear disk
R_CONV = 300.0              # km, convection-centroid disk
STEP = np.timedelta64(6, "h")

def wrap180(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0

def gc_km(la, lo, la0, lo0):
    p1, p2 = np.deg2rad(la), np.deg2rad(la0)
    d = np.deg2rad(wrap180(np.asarray(lo, float) - lo0))
    return 6371.0 * np.arccos(
        np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(d), -1, 1))

def local_xy(la, lo, la0, lo0):
    """East/north offset in km on a local tangent plane at (la0, lo0)."""
    x = wrap180(np.asarray(lo, float) - lo0) * 111.32 * np.cos(np.deg2rad(la0))
    y = (np.asarray(la, float) - la0) * 110.57
    return x, y

def shear_vector(ds, when, clat, clon):
    """Vector 200-850 hPa shear, area-averaged over R_SHEAR km of the centre.

    A mean of scalar |shear| has no direction; the angle test needs the vector.
    """
    sub = ds[["u_component_of_wind", "v_component_of_wind"]].sel(
        time=when, level=[200, 850]).load()
    la = np.asarray(sub.lat.values, float)
    lo = np.asarray(sub.lon.values, float)
    LO, LA = np.meshgrid(lo, la)
    m = gc_km(LA, LO, clat, clon) < R_SHEAR
    w = np.cos(np.deg2rad(LA)) * m
    out = {}
    for v, nm in (("u_component_of_wind", "u"), ("v_component_of_wind", "v")):
        a = np.asarray(sub[v].values, float)              # (level, lat, lon)
        out[nm] = [float((w * a[i]).sum() / w.sum()) for i in range(2)]
    return np.array([out["u"][0] - out["u"][1], out["v"][0] - out["v"][1]]), int(m.sum())

def score(name, cfg, ds, feat_key="node_conv"):
    f = RES / f"run_{name}.npy"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True).item()
    sn = d.get("snap", {}).get("baseline_mid")
    if sn is None or feat_key not in sn:
        return None

    # --- centre: MSLP minimum at +48 h, not the configured IC centre ---------
    g = np.asarray(sn["mslp_grid"], float)
    glat, glon = np.asarray(sn["mslp_lat"], float), np.asarray(sn["mslp_lon"], float)
    j, i = np.unravel_index(int(np.nanargmin(g)), g.shape)
    clat, clon = float(glat[j]), float(glon[i])
    reloc = gc_km(cfg["center"][0], S.norm_lon(cfg["center"][1]), clat, clon)

    # --- convection centroid within R_CONV of that centre --------------------
    mlat = np.asarray(sn["mlat"], float)
    mlon = S.norm_lon(np.asarray(sn["mlon"], float))
    a = np.asarray(sn[feat_key], float)
    a = a.sum(1) if a.ndim == 2 else a
    a = np.maximum(a, 0.0)
    disk = gc_km(mlat, mlon, clat, clon) < R_CONV
    if disk.sum() < 10 or a[disk].sum() <= 0:
        return None
    x, y = local_xy(mlat[disk], mlon[disk], clat, clon)
    w = a[disk]
    dx, dy = float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())

    # --- shear at the SAME time as the displacement --------------------------
    when = np.datetime64(cfg["ic"]) + (MID + 1) * STEP
    Sv, npt = shear_vector(ds, when, clat, clon)

    smag = float(np.hypot(*Sv))
    # angle FROM the shear vector TO the displacement, positive counter-clockwise.
    ang = float(np.degrees(np.arctan2(Sv[0] * dy - Sv[1] * dx, Sv[0] * dx + Sv[1] * dy)))
    return dict(name=name, clat=clat, clon=clon, reloc=float(reloc), dx=dx, dy=dy,
                disp=float(np.hypot(dx, dy)), S=Sv.tolist(), smag=smag, angle=ang,
                hemi="S" if clat < 0 else "N", n_shear=npt, n_disk=int(disk.sum()),
                secondary=bool(cfg.get("secondary", False)))

def circ(angles):
    """Mean direction and resultant length R of a set of angles in degrees."""
    a = np.deg2rad(np.asarray(angles, float))
    C, Sn = np.cos(a).mean(), np.sin(a).mean()
    return float(np.degrees(np.arctan2(Sn, C))), float(np.hypot(C, Sn))

def main():
    ds, _ = fc.open_wb2()
    rows = [r for r in (score(k, v, ds) for k, v in S.STORMS.items()
                        if not v.get("nondev", False)) if r is not None]
    if not rows:
        print(f"no scorable runs in {RES}"); return
    print(f"battery {S.__name__}   results {RES}   snapshot +{6*(MID+1)} h\n")
    print(f"{'storm':<14}{'hemi':>5}{'centre':>16}{'reloc km':>10}{'|S| m/s':>9}"
          f"{'disp km':>9}{'angle deg':>11}{'side':>7}")
    for r in rows:
        side = "LEFT" if r["angle"] > 0 else "RIGHT"
        tag = " (secondary)" if r["secondary"] else ""
        print(f"{r['name']:<14}{r['hemi']:>5}{r['clat']:>8.1f},{r['clon']:>7.1f}"
              f"{r['reloc']:>10.0f}{r['smag']:>9.1f}{r['disp']:>9.0f}"
              f"{r['angle']:>+11.0f}{side:>7}{tag}")

    # --- the control-must-be-able-to-fail rule: the control that must FAIL -----------------------------
    # If the firing-rate-matched RANDOM group shows the same displacement geometry, the
    # statistic is reading the storm's own asymmetry rather than the convection features,
    # and every prediction here is void. node_rand exists only in runs made after
    # 2026-08-18; older runs report it as unavailable rather than silently skipping it.
    ctl = [r for r in (score(k, v, ds, feat_key="node_rand")
                       for k, v in S.STORMS.items() if not v.get("nondev", False))
           if r is not None]
    print(f"\n{'CONTROL (random group, must show no structure)':<46}"
          f"{len(ctl)}/{len(rows)} runs carry node_rand")
    if ctl:
        for r in ctl:
            print(f"  {r['name']:<14}{r['smag']:>9.1f}{r['disp']:>9.0f}{r['angle']:>+11.0f}")
        mu, R = circ([r["angle"] for r in ctl])
        print(f"  control mean angle {mu:+.0f} deg, R = {R:.2f}  "
              f"({sum(1 for r in ctl if r['angle']>0)}/{len(ctl)} left)")
        print("  the control FAILS (as required) if its R is low and its mean angle does "
              "not track the convection group's")
    else:
        print("  NOT COMPUTABLE on these runs -- rerun with MECH_ARMS=baseline to refresh "
              "the snapshot. Until then every prediction below is UNCONTROLLED and, by "
              "guardrail #9, not reportable.")

    prim = [r for r in rows if not r["secondary"]]
    for lbl, sel in (("PRIMARY", prim), ("ALL (incl. secondary)", rows)):
        if len(sel) < 2 or len(sel) == len(prim) and lbl.startswith("ALL"):
            if lbl.startswith("ALL") and len(sel) == len(prim):
                continue
        mu, R = circ([r["angle"] for r in sel])
        n = len(sel)
        print(f"\n{lbl}  n={n}")
        print(f"  P1 mean angle {mu:+.0f} deg, resultant R = {R:.2f}   "
              f"({sum(1 for r in sel if r['angle']>0)}/{n} left)")
        if n >= 3:
            from scipy import stats
            rho, p = stats.pearsonr([r["smag"] for r in sel], [r["disp"] for r in sel])
            print(f"  P2 disp vs |S|:  r = {rho:+.3f}, p = {p:.3f}")
    json.dump(dict(storms=rows, control=ctl),
              open(RES / "ps5_displacement.json", "w"), indent=1)
    print(f"\n-> {RES/'ps5_displacement.json'}")

if __name__ == "__main__":
    main()
