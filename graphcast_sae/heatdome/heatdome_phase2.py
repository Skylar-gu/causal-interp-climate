"""Phase 2 (GPU, bf16) — causal knockout of the heat-dome feature(s).

Restore the Phase-1 heat-dome feature set to NORMAL within a ~1500 km disk around the ridge
centre (from ERA5 truth), held persistently through the 6-day rollout. Four arms share ONE
compiled graph (runtime patch arrays; build_apply_cond / delta_cond, the 'cap to normal inside
a disk' counterfactual):

  baseline      : untouched
  block-normal  : heat-dome feature(s) capped at NORMAL in the disk (honest CF)
  block-zero    : heat-dome feature(s) capped at 0 in the disk (delete)
  rand-normal   : matched-firing-rate W-NA random control features capped at NORMAL in the disk

Records per lead: (INTERNAL) heat-dome feature box firing per arm; (PHYSICAL) z500 ridge
anomaly max & 2m-T max over the box; (SKILL) z500(gpm) & 2m-T box fields for RMSE vs ERA5.
Also a NON-BLOCK-DATE control (2021-05-15): baseline vs block-normal, where the feature
should do little because there is no ridge.

Serialize behind other GPU jobs. Crash-safe: results/heatdome/phase2.npy.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/heatdome
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.heatdome.heatdome_phase2
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.common.signature_physics import gc_km
import graphcast_sae.heatdome.heatdome_config as C

OUT = fc.ROOT / "results/heatdome"; OUT.mkdir(parents=True, exist_ok=True)

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, H, task_config):
    blk, times, st = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(task_config))

def box_diag(p, box):
    """From a numpyified single-time pred: ridge anom max, 2m-T max(C), and box fields."""
    la0, la1 = box["lat"]; lo = C.norm_lon(box["lon"])
    z = p["geopotential"].isel(batch=0, time=0).sel(level=500) / C.G
    zlat = z.sel(lat=slice(la0, la1))
    anom = zlat - zlat.mean("lon")
    t2 = p["2m_temperature"].isel(batch=0, time=0).sel(lat=slice(la0, la1))
    zb = zlat.sel(lon=slice(lo[0], lo[1]))
    ab = anom.sel(lon=slice(lo[0], lo[1]))
    tb = t2.sel(lon=slice(lo[0], lo[1]))
    return (float(np.nanmax(ab.values)), float(np.nanmax(tb.values)) - 273.15,
            np.asarray(zb.values, np.float32), np.asarray(tb.values, np.float32))

def pick_random_ctrl(cat, hd_set, n):
    """n W-NA-firing features matched to hd_set firing rates, seed=7, not candidates. Frozen."""
    fr = cat["firerate"]; clat = cat["clat"]
    clon = np.where(cat["clon"] > 180, cat["clon"] - 360, cat["clon"])
    wna = (clat >= 40) & (clat <= 66) & (clon >= -150) & (clon <= -95)
    pool = np.array([f for f in range(len(fr)) if wna[f] and f not in C.CANDS])
    rng = np.random.default_rng(7)
    tgt_rates = [fr[f] for f in hd_set]
    picked = []
    for tr in tgt_rates:
        cand = [f for f in pool if abs(fr[f] - tr) <= 0.4 * tr and f not in picked]
        if not cand:  # widen if needed
            cand = [f for f in pool if f not in picked]
        picked.append(int(rng.choice(cand)))
    return picked

def measure_normal(codes_at, feats, inbox, nmask, event_peak_box):
    """Normal feature level in the disk across quiet analogs; skip analogs with a ridge present."""
    acc = {f: [] for f in feats}; used = []
    for a in C.ANALOGS:
        try:
            c = codes_at(a)
        except Exception as e:
            print(f"  analog {a}: load ERROR {e}", flush=True); continue
        ridge_fire = float(c[inbox][:, list(feats)].sum())
        if ridge_fire > 0.4 * event_peak_box:
            print(f"  analog {a}: ridgefire={ridge_fire:.0f} block present, SKIP", flush=True); continue
        for f in feats:
            v = c[nmask.astype(bool), f]; acc[f].extend(v[v > 0].tolist())
        used.append(a); print(f"  analog {a}: ridgefire={ridge_fire:.0f} quiet, used", flush=True)
    lvl = {f: (float(np.mean(acc[f])) if acc[f] else 0.0) for f in feats}
    return lvl, used

def run_ic(t0, H, arms, apply, sae, mlat, inbox, box, tc, capture_fields=True):
    inp, tgt, frc = build_io(t0, H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)
    hd = arms["_hd"]; rc = arms["_rc"]
    track = sorted(set(list(hd) + list(rc)))

    def roll(patch):
        cur = inp; ridge = []; heat = []; zf = []; tf = []
        per = {f: [] for f in track}
        pj = tuple(jnp.asarray(x) for x in patch)
        for h in range(H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, pj)
            X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
            Cc = np.asarray(sae.codes(X))
            for f in track: per[f].append(float(Cc[inbox, f].sum()))
            p = numpyify(p)
            rr, hh, zb, tb = box_diag(p, box)
            ridge.append(rr); heat.append(hh)
            if capture_fields: zf.append(zb); tf.append(tb)
            if h < H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        r = dict(ridge=np.array(ridge), heat=np.array(heat),
                 box_feats={f: np.array(per[f]) for f in track})
        if capture_fields:
            r["z500_box"] = np.stack(zf).astype(np.float32)
            r["t2m_box"] = np.stack(tf).astype(np.float32)
        return r

    res = {}
    for aname, patch in arms.items():
        if aname.startswith("_"): continue
        t = time.time()
        res[aname] = roll(patch)
        print(f"  [{aname}] ridge {np.array2string(res[aname]['ridge'],precision=0,max_line_width=250)}", flush=True)
        print(f"  [{aname}] heat  {np.array2string(res[aname]['heat'],precision=1,max_line_width=250)}  ({time.time()-t:.0f}s)", flush=True)
        hdbox = np.max([res[aname]['box_feats'][f] for f in hd], axis=0)
        print(f"  [{aname}] hd_feat_box {np.array2string(hdbox,precision=0,max_line_width=250)}", flush=True)
    return res

def main():
    p1 = np.load(OUT / "phase1.npy", allow_pickle=True).item()
    truth = np.load(OUT / "era5_truth.npy", allow_pickle=True).item()
    hd_set = list(p1["hd_set"]); event_peak_box = float(max(p1["peak"][f] for f in hd_set))
    center = truth["ridge_center"]
    print(f"heat-dome feature set: {hd_set}; ridge centre {center}; event peak box {event_peak_box:.0f}", flush=True)

    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    rand = pick_random_ctrl(cat, hd_set, len(hd_set))
    print(f"random control (W-NA, matched firing): {rand}", flush=True)

    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    la0, la1 = C.BOX["lat"]; blo0, blo1 = C.BOX["lon"]
    inbox = (mlat >= la0) & (mlat <= la1) & (mlon >= blo0) & (mlon <= blo1)
    nmask = (gc_km(mlat, mlon, center[0], center[1]) < C.RADIUS_KM).astype(np.float32)
    print(f"disk nodes: {int(nmask.sum())}; box nodes: {int(inbox.sum())}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)

    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        z = jnp.zeros(sae.n_features, jnp.float32)
        _, acts = apply(inp, tg, fr, (z, z, np.zeros(len(mlat), np.float32)))
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    print("measuring NORMAL feature levels from quiet late-June analogs:", flush=True)
    lvl, used = measure_normal(codes_at, hd_set + rand, inbox, nmask, event_peak_box)
    print(f"  normal levels hd {[round(lvl[f],2) for f in hd_set]} rand {[round(lvl[f],2) for f in rand]}", flush=True)

    fsel_hd = np.zeros(sae.n_features, np.float32); fsel_hd[hd_set] = 1.0
    fsel_rc = np.zeros(sae.n_features, np.float32); fsel_rc[rand] = 1.0
    ftarget = np.zeros(sae.n_features, np.float32)
    for f in hd_set + rand: ftarget[f] = lvl[f]
    zeroF = np.zeros(sae.n_features, np.float32); zeroN = np.zeros(len(mlat), np.float32)

    arms = {
        "_hd": hd_set, "_rc": rand,
        "baseline":     (zeroF, zeroF, zeroN),
        "block-normal": (fsel_hd, ftarget, nmask),
        "block-zero":   (fsel_hd, zeroF, nmask),
        "rand-normal":  (fsel_rc, ftarget, nmask),
    }

    print("\n=== EVENT IC rollout (4 arms) ===", flush=True)
    res_event = run_ic(C.IC, C.H, arms, apply, sae, mlat, inbox, C.BOX, tc, capture_fields=True)

    print("\n=== NON-BLOCK IC control (baseline vs block-normal) ===", flush=True)
    arms_nb = {"_hd": hd_set, "_rc": rand,
               "baseline": (zeroF, zeroF, zeroN), "block-normal": (fsel_hd, ftarget, nmask)}
    res_nb = run_ic(C.NONBLOCK_IC, C.H, arms_nb, apply, sae, mlat, inbox, C.BOX, tc, capture_fields=False)

    out = dict(ic=C.IC, box=C.BOX, center=center, hd_set=hd_set, rand=rand,
               normal_levels={f: float(lvl[f]) for f in hd_set + rand}, analogs_used=used,
               disk_nodes=int(nmask.sum()), box_nodes=int(inbox.sum()),
               leads_h=(np.arange(C.H) + 1) * 6,
               res=res_event, res_nonblock=res_nb, nonblock_ic=C.NONBLOCK_IC)
    np.save(OUT / "phase2.npy", out, allow_pickle=True)
    print("\n-> results/heatdome/phase2.npy", flush=True)

if __name__ == "__main__":
    main()
