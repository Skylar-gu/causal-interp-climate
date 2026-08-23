"""CONCEPT RESPONSE OPERATORS — the Green's function of each purified concept.

Implements docs/prereg/prereg_response_operator.md (RESPOP-1..RESPOP-6), frozen before this file existed. No graph discovery: impulse concept i once, roll GraphCast free for 60 h,
and read the full spatial response field in physical space.

Arms per window (23 rolls): base, nf0, nf1 (both unpatched -> the MEASURED numeric floor),
10 purified concepts, 10 perm groups (same 40 features, scrambled labels).

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Fig. fig:contrast (a)/(c): the RESPOP run (results/fs_respop.npy -> results/fs_contrast_inputs.npy)
Inputs: results/fs_cgv2_groups.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: out/respop_status.txt; results/fs_respop.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.concepts.respop
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

GAMMA = 1.0
S = int(os.environ.get("RP_S", 10))      # 10 x 6 h = 60 h (prereg); RP_S is for smoke tests
ICS = ["2020-01-05", "2020-04-06", "2020-07-06", "2020-10-05"]
if os.environ.get("RP_NIC"):             # smoke test only
    ICS = ICS[:int(os.environ["RP_NIC"])]
SMOKE = os.environ.get("RP_SMOKE", "")   # comma list of arm names, smoke tests only
OUT = os.environ.get("RP_OUT", "results/fs_respop.npy")
STATUS = fc.ROOT / "out/respop_status.txt"
G0 = 9.80665
CO = 2                                   # 0.5 deg block-mean for A50/A90
CF = 6                                   # 1.5 deg block-mean for the stored fields

# frozen readout fields (prereg RESPOP-3)
FIELDS = [("z500", "geopotential", 500), ("q600", "specific_humidity", 600),
          ("q850", "specific_humidity", 850), ("w500", "vertical_velocity", 500),
          ("t850", "temperature", 850), ("u250", "u_component_of_wind", 250),
          ("u850", "u_component_of_wind", 850), ("v850", "v_component_of_wind", 850)]
FNAMES = [f[0] for f in FIELDS]
OWN = {"vort850": "v850", "q600": "q600", "ascent": "w500", "shear": "SHEAR",
       "t850": "t850", "z500": "z500", "jet250": "u250", "blocking": "z500",
       "atm_river": "q850", "baroclinicity": "t850"}
LAT_EDGES = [-90, -60, -30, -15, 15, 30, 60, 90]

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc, s):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=s + 2)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*s}h"), **dataclasses.asdict(tc))

def block_mean(a, f):
    n = (a.shape[0] // f) * f
    return a[:n].reshape(-1, f, a.shape[1] // f, f).mean((1, 3))

def mag(d, wlat):
    """Full-res cos-lat-weighted RMS and abs-max of a signed response field."""
    w = wlat[:, None]
    return (float(np.sqrt((w * d * d).sum() / (w.sum() * d.shape[1]))),
            float(np.abs(d).max()))

def stats_field(d, lat, wlat):
    """Spatial-structure stats. Cost is dominated by the A50 sort, so this is called
    only for the primary field (z500) and for the arm's own governing field."""
    # spatial concentration on a 0.5 deg block-mean
    dc = block_mean(d, CO)
    latc = block_mean(lat[:, None] * np.ones((1, CO)), CO)[:, 0]
    wc = np.cos(np.deg2rad(latc))[:, None] * np.ones((1, dc.shape[1]))
    e = (dc * dc).ravel(); ww = wc.ravel()
    o = np.argsort(e)[::-1]
    ce = np.cumsum(e[o] * ww[o]); cw = np.cumsum(ww[o])
    a50 = float(cw[min(np.searchsorted(ce, 0.5 * ce[-1]), len(cw) - 1)] / cw[-1])
    a90 = float(cw[min(np.searchsorted(ce, 0.9 * ce[-1]), len(cw) - 1)] / cw[-1])
    ad = np.abs(d)
    prof = []
    for lo, hi in zip(LAT_EDGES[:-1], LAT_EDGES[1:]):
        m = (lat >= lo) & (lat < hi)
        prof.append(float((ad[m] * wlat[m, None]).sum() / (wlat[m].sum() * d.shape[1])))
    cen = float((ad * (wlat * lat)[:, None]).sum() / max((ad * wlat[:, None]).sum(), 1e-30))
    return a50, a90, prof, cen

def main():
    G = np.load(fc.ROOT / "results/fs_cgv2_groups.npy", allow_pickle=True).item()
    names = list(G["concepts"])
    members = [G["groups"][n] for n in names]
    perm = [list(map(int, g)) for g in G["perm_groups"]]
    arms = ([("nf0", None), ("nf1", None)] +
            [(n, m) for n, m in zip(names, members)] +
            [(f"perm{i}", g) for i, g in enumerate(perm)])
    if SMOKE:
        keep = set(SMOKE.split(","))
        arms = [a for a in arms if a[0] in keep]
        print(f"  ** SMOKE TEST: arms subset to {[a for a, _ in arms]}, "
              f"S={S}, ICs={ICS} -- NOT the pre-registered run **")
    print("CONCEPT RESPONSE OPERATORS — prereg docs/prereg/prereg_response_operator.md")
    print(f"  {len(names)} concepts x K={G['K']}, {len(arms)} arms + base, "
          f"S={S} steps (60 h), {len(ICS)} ICs", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    nA, nF = len(arms), len(FIELDS)
    RMS = np.zeros((len(ICS), nA, S, nF + 1))          # +1 = the SHEAR combination
    AMAX = np.zeros_like(RMS)
    A50 = np.zeros((len(ICS), nA, S, nF + 1))
    A90 = np.zeros_like(A50)
    PROF = np.zeros((len(ICS), nA, S, nF + 1, len(LAT_EDGES) - 1))
    CEN = np.zeros((len(ICS), nA, S, nF + 1))
    COARSE = None
    t0 = time.time(); nfwd = 0

    for wi, ic in enumerate(ICS):
        inp, tgt, frc = build_io(ic, tc, S)
        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)
        lat = np.asarray(inp["lat"].values, np.float64)
        wlat = np.cos(np.deg2rad(lat))
        nlt, nln = len(lat), len(inp["lon"].values)
        if COARSE is None:
            COARSE = np.zeros((nA, S, 2, nlt // CF, nln // CF), np.float32)

        def roll(patch):
            """One S-step free rollout; patch applied at the FIRST forward only.
            Returns (S, nF+1, nlat, nlon) float32 of the frozen readout fields."""
            cur = inp
            out = np.zeros((S, nF + 1, len(lat), nln), np.float32)
            for s in range(S):
                tg = tgt.isel(time=slice(s, s + 1)).assign_coords(time=tct)
                fr = frc.isel(time=slice(s, s + 1)).assign_coords(time=tct)
                # apply() returns (preds, acts); numpyify takes the Dataset only
                pr = numpyify(apply(cur, tg, fr, patch if s == 0 else noop)[0])
                for fi, (_, var, lev) in enumerate(FIELDS):
                    out[s, fi] = pr[var].isel(batch=0, time=0).sel(level=lev) \
                        .transpose("lat", "lon").values
                out[s, nF] = out[s, FNAMES.index("u250")] - out[s, FNAMES.index("u850")]
                cur = rollout._get_next_inputs(cur, xr.merge([pr, fr])).assign_coords(
                    time=inp.coords["time"])
            return out

        base = roll(noop); nfwd += S
        base[:, 0] /= G0                                   # z500 in geopotential metres
        for ai, (an, feats) in enumerate(arms):
            patch = noop if feats is None else fc.coef_patch(
                sae, [int(f) for f in feats], GAMMA)
            arm = roll(patch); nfwd += S
            arm[:, 0] /= G0
            own = OWN.get(an)
            oi = nF if own == "SHEAR" else (FNAMES.index(own) if own else 0)
            for s in range(S):
                for fi in range(nF + 1):
                    d = (arm[s, fi] - base[s, fi]).astype(np.float64)
                    RMS[wi, ai, s, fi], AMAX[wi, ai, s, fi] = mag(d, wlat)
                    if fi in (0, oi):
                        (A50[wi, ai, s, fi], A90[wi, ai, s, fi],
                         PROF[wi, ai, s, fi], CEN[wi, ai, s, fi]) = \
                            stats_field(d, lat, wlat)
                COARSE[ai, s, 0] += block_mean(arm[s, 0] - base[s, 0], CF) / len(ICS)
                COARSE[ai, s, 1] += block_mean(arm[s, oi] - base[s, oi], CF) / len(ICS)
            el = (time.time() - t0) / 60
            msg = (f"  IC {ic} ({wi+1}/{len(ICS)})  arm {an} "
                   f"({ai+1}/{nA})  {el:.1f}m  {nfwd} forwards")
            print(msg, flush=True); STATUS.write_text(msg + "\n")

    np.save(fc.ROOT / OUT, dict(
        RMS=RMS, AMAX=AMAX, A50=A50, A90=A90, PROF=PROF, CEN=CEN, COARSE=COARSE,
        arms=[a for a, _ in arms], arm_feats=[list(map(int, f)) if f else [] for _, f in arms],
        names=names, fields=FNAMES + ["shear"], own=OWN, lat_edges=LAT_EDGES,
        ics=ICS, S=S, gamma=GAMMA, K=int(G["K"]), coarsen=CF,
        axes="RMS[ic, arm, lead, field]; COARSE[arm, lead, 0=z500/1=own, lat, lon]",
        prereg="docs/prereg/prereg_response_operator.md @ 9bc745e"), allow_pickle=True)
    print(f"\n-> {OUT}  ({nfwd} forwards, {(time.time()-t0)/60:.1f}m)")
    STATUS.write_text(f"DONE {nfwd} forwards {(time.time()-t0)/60:.1f}m\n")

if __name__ == "__main__":
    main()
