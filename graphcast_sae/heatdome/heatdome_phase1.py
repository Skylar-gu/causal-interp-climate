"""Phase 1 (GPU, bf16) — identify the heat-dome feature(s).

Roll the flagship from IC 2021-06-24 through the ridge peak (+144h). At each lead:
  - encode layer-8 -> SAE codes over the full mesh
  - for each atlas blocking candidate [1789,492,2930,1703,1036], record its firing in the
    W-NA box (sum / max / active-node count) and the node-level firing map (for the ridge
    overlay)
  - record the model's own z500 ridge anomaly max over the box (does baseline reproduce the ridge?).
Then rank candidates by peak box-firing -> "the heat-dome feature(s)"; frozen rule: the set is
every candidate whose peak box-firing >= 0.5 * the top candidate's. Sanity: their firing must
rise as the ridge builds (corr with the ridge trajectory) and sit spatially on the ridge.

Serialize behind other GPU jobs. Crash-safe: results/heatdome/phase1.npy.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/heatdome
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.heatdome.heatdome_phase1
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
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

def ridge_anom_max(p, box):
    """z500 zonal-anomaly (vs full-circle zonal mean) max over the box, from a numpyified pred."""
    la0, la1 = box["lat"]; lo = C.norm_lon(box["lon"])
    z = p["geopotential"].isel(batch=0, time=0).sel(level=500) / C.G
    z = z.sel(lat=slice(la0, la1))
    zonal = z.mean("lon")
    anom = z - zonal
    anom = anom.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1] else \
           anom.sel(lon=(p.lon >= lo[0]) | (p.lon <= lo[1]))
    return float(np.nanmax(anom.values))

def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    la0, la1 = C.BOX["lat"]; blo0, blo1 = C.BOX["lon"]
    inbox = (mlat >= la0) & (mlat <= la1) & (mlon >= blo0) & (mlon <= blo1)
    print(f"box mesh nodes: {int(inbox.sum())}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = (jnp.zeros(sae.n_features, jnp.float32), jnp.zeros(sae.n_features, jnp.float32),
            jnp.zeros(len(mlat), jnp.float32))

    inp, tgt, frc = build_io(C.IC, C.H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    box_sum = {f: [] for f in C.CANDS}      # sum of code over box nodes
    box_max = {f: [] for f in C.CANDS}
    box_cnt = {f: [] for f in C.CANDS}      # active nodes in box
    ridge = []
    node_maps = {f: [] for f in C.CANDS}    # full-mesh firing per lead (for spatial track)

    cur = inp
    t0 = time.time()
    for h in range(C.H):
        ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
        cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
        p, a = apply(cur, ct, cf, noop)
        X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
        Cc = np.asarray(sae.codes(X))       # (n_mesh, F)
        for f in C.CANDS:
            col = Cc[:, f]
            box_sum[f].append(float(col[inbox].sum()))
            box_max[f].append(float(col[inbox].max()))
            box_cnt[f].append(int((col[inbox] > 0).sum()))
            node_maps[f].append(col.astype(np.float32))
        p = numpyify(p)
        ridge.append(ridge_anom_max(p, C.BOX))
        if h < C.H - 1:
            cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        if h % 4 == 0:
            print(f"  +{(h+1)*6:3d}h ridge={ridge[-1]:.0f}m  "
                  + "  ".join(f"{f}:{box_sum[f][-1]:.0f}" for f in C.CANDS)
                  + f"   ({time.time()-t0:.0f}s)", flush=True)

    box_sum = {f: np.array(v) for f, v in box_sum.items()}
    box_max = {f: np.array(v) for f, v in box_max.items()}
    box_cnt = {f: np.array(v) for f, v in box_cnt.items()}
    ridge = np.array(ridge)
    peak = {f: float(box_sum[f].max()) for f in C.CANDS}
    top = max(peak, key=peak.get)
    hd_set = [f for f in C.CANDS if peak[f] >= 0.5 * peak[top]]
    # correlation of each candidate's box-firing with the ridge trajectory
    corr = {}
    for f in C.CANDS:
        v = box_sum[f]
        corr[f] = float(np.corrcoef(v, ridge)[0, 1]) if v.std() > 0 else 0.0

    print("\n=== candidate firing over the rollout ===", flush=True)
    print(f"ridge anom (m): {np.array2string(ridge, precision=0, max_line_width=250)}", flush=True)
    for f in C.CANDS:
        print(f"  feat {f}: peak_box={peak[f]:.0f}  peak_cnt={int(box_cnt[f].max())}  "
              f"corr_ridge={corr[f]:+.2f}  box_sum={np.array2string(box_sum[f], precision=0, max_line_width=250)}",
              flush=True)
    print(f"\ntop candidate: {top}  ->  heat-dome feature set (>=50% of top): {hd_set}", flush=True)
    print(f"  {'ONE feature' if len(hd_set)==1 else str(len(hd_set))+' features'}", flush=True)

    out = dict(ic=C.IC, box=C.BOX, cands=C.CANDS, leads_h=(np.arange(C.H)+1)*6,
               ridge=ridge, box_sum=box_sum, box_max=box_max, box_cnt=box_cnt,
               peak=peak, top=top, hd_set=hd_set, corr=corr,
               node_maps={f: np.stack(node_maps[f]) for f in C.CANDS},
               mlat=mlat.astype(np.float32), mlon=mlon.astype(np.float32))
    np.save(OUT / "phase1.npy", out, allow_pickle=True)
    print("-> results/heatdome/phase1.npy", flush=True)

if __name__ == "__main__":
    main()
