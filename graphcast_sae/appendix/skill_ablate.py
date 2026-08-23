"""Phase 5 (GPU): causal necessity of the skill-features for GC's advantage.

On the top-quartile-advantage cases, persistently ablate (coef -1) the top-K skill-features
through the 7-day rollout and recompute adv. Control: ablate K random matched-FIRING features.
Necessity => skill-ablation shrinks GC's advantage toward IFS; control leaves it ~unchanged.

Paper: Appendix app:taxonomy (skill decomposition)
Inputs: results/fs_atlas.npy (not shipped, see docs/REPRODUCE.md); results/skill/cases.npy (not shipped, see docs/REPRODUCE.md); results/skill/decompose.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/ablate.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.appendix.skill_ablate
"""
import os, sys, time, functools, argparse
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

HRES = "weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr"
G = 9.80665
H = 28
LEADS = [72, 120, 168]
PROD_H = {72: 11, 120: 19, 168: 27}
TARGET_LEADS = [120, 168]

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc):
    """Fast loader: download only the 2 input frames of prognostic vars + forcings for
    all frames (target prognostic values are tiled; unused since the model predicts them).
    Validated ~identical to the full loader; loader choice cancels in the paired
    skill-vs-control comparison. Same signature as before."""
    ds, statics = fc.open_wb2()
    times = np.datetime64(t0) - fc.STEP + np.arange(2 + H) * fc.STEP
    prog = list(fc.SURFACE_VARS) + list(fc.ATMOS_VARS)
    frcv = list(fc.FORCING_VARS)
    inp2 = ds[prog].sel(time=times[:2]).load()
    frc_all = ds[frcv].sel(time=times).load()
    last = inp2.isel(time=1)
    tile = xr.concat([last.assign_coords(time=tt) for tt in times[2:]], dim="time")
    blk = xr.merge([xr.concat([inp2, tile], dim="time"), frc_all]).sel(time=times)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def wrmse(a, b, latmask, w):
    d = (a - b) ** 2
    ww = np.broadcast_to(w[:, None], d.shape)
    m = latmask[:, None] & np.ones((1, d.shape[1]), bool)
    return float(np.sqrt((d * ww)[m].sum() / ww[m].sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=30, help="top-quartile case count")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    dec = np.load(fc.ROOT / "results/skill/decompose.npy", allow_pickle=True).item()
    top_feats = list(np.asarray(dec["top_feats"])[: args.topk])
    y = np.asarray(dec["y"]); cis = np.asarray(dec["cis"])
    C = np.load(fc.ROOT / "results/skill/cases.npy", allow_pickle=True).item()
    dates = C["dates"]

    # top-quartile advantage cases
    ord_ = np.argsort(-y)
    q = ord_[: args.nq]
    q_ci = cis[q]
    print(f"top-{args.nq} advantage cases; skill-features (K={args.topk}): {top_feats}", flush=True)

    # matched-firing random control features
    atl = np.load(fc.ROOT / "results/fs_atlas.npy", allow_pickle=True).item()
    fr = np.asarray(atl["firerate"])
    rng = np.random.default_rng(0)
    skillset = set(int(f) for f in top_feats)
    ctrl = []
    NF = getattr(fc, "NF", None) or 4096
    pool = [i for i in range(NF) if i not in skillset]
    for f in top_feats:
        target = fr[int(f)]
        cand = sorted(pool, key=lambda i: abs(fr[i] - target))[:40]
        cand = [c for c in cand if c not in ctrl]
        pick = int(rng.choice(cand[:20]))
        ctrl.append(pick)
    print(f"matched-firing control features: {ctrl}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)
    p_skill = fc.coef_patch(sae, top_feats, -1.0)
    p_ctrl = fc.coef_patch(sae, ctrl, -1.0)

    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    hres = xr.open_zarr(fs.get_mapper(HRES), consolidated=True).rename(
        {"latitude": "lat", "longitude": "lon"})
    if hres.lat.values[0] > hres.lat.values[-1]:
        hres = hres.reindex(lat=hres.lat.values[::-1])
    era5, _ = fc.open_wb2()

    def roll_and_score(inp, tgt, frc, patch, init, latm, w):
        tct = tgt.time.isel(time=slice(0, 1))
        t2, f2 = tgt, frc
        for c in ("datetime",):
            if c in t2.coords: t2 = t2.drop_vars(c)
            if c in f2.coords: f2 = f2.drop_vars(c)
        cur = inp; fields = {}
        for h in range(H):
            ct = t2.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            cf = f2.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            preds, _ = apply(cur, ct, cf, patch)
            preds = numpyify(preds)
            for lead, hh in PROD_H.items():
                if h == hh:
                    fields[lead] = np.asarray(
                        preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / G
            if h < H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([preds, cf])).assign_coords(
                    time=cur.coords["time"])
        adv = []
        for L in TARGET_LEADS:
            vt = np.datetime64(init) + np.timedelta64(L, "h")
            tr = np.asarray(era5["geopotential"].sel(time=vt, level=500).values) / G
            hf = np.asarray(hres["geopotential"].sel(time=np.datetime64(init),
                            prediction_timedelta=int(L), level=500).values) / G
            rg = wrmse(fields[L], tr, latm, w); rh = wrmse(hf, tr, latm, w)
            adv.append(rh - rg)
        return float(np.mean(adv))

    rows = []
    t0 = time.time()
    for k, ci in enumerate(q_ci):
        init = dates[int(ci)]
        inp, tgt, frc = build_io(init, tc)
        lat = np.asarray(inp.lat.values); lon = np.asarray(inp.lon.values)
        latm = (lat >= 30) & (lat <= 75); w = np.cos(np.deg2rad(lat))
        a_base = roll_and_score(inp, tgt, frc, noop, init, latm, w)
        a_skill = roll_and_score(inp, tgt, frc, p_skill, init, latm, w)
        a_ctrl = roll_and_score(inp, tgt, frc, p_ctrl, init, latm, w)
        rows.append((int(ci), a_base, a_skill, a_ctrl))
        print(f"[{k+1}/{args.nq}] ci{int(ci)} {str(init)[:13]}  base_adv={a_base:+.1f} "
              f"skillabl={a_skill:+.1f} ctrlabl={a_ctrl:+.1f}  "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
        np.save(fc.ROOT / "results/skill/ablate.npy", dict(
            rows=np.array(rows), top_feats=top_feats, ctrl=ctrl,
            cols=["ci", "base_adv", "skill_abl_adv", "ctrl_abl_adv"]), allow_pickle=True)

    R = np.array(rows)
    base, sk, ct = R[:, 1], R[:, 2], R[:, 3]
    from scipy import stats
    print("\n===== Phase 5: causal necessity (Z500 NHext adv, mean 120/168h) =====")
    print(f"  baseline adv        = {base.mean():+.2f} gpm")
    print(f"  skill-ablated adv   = {sk.mean():+.2f} gpm   (drop {(base-sk).mean():+.2f})")
    print(f"  control-ablated adv = {ct.mean():+.2f} gpm   (drop {(base-ct).mean():+.2f})")
    t, p = stats.ttest_rel(base - sk, base - ct)
    print(f"  skill-drop vs control-drop: paired t p={p:.2e}  "
          f"(gap {(base-sk).mean()-(base-ct).mean():+.2f} gpm)")

if __name__ == "__main__":
    main()
