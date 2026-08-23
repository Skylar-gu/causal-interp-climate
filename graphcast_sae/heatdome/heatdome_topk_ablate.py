"""Clean dose-response: ablate the TOP-K ridge features BY ACTIVATION (the test the physics-set
run skipped). Ranks features by peak box activation, keeps only those spatially on the ridge
(centroid < RADIUS of ridge centre), builds nested sets top-5..top-160, restores each to NORMAL
in the localized disk through the 6-day rollout, measures ridge collapse & heat.

Decisive: if top-20 already collapses the ridge -> a ~20-feature lever exists. If the effect only
appears at large K -> the code is genuinely distributed (redundant), no small lever.

Reuses heatdome_physics_ablate machinery. Serialize behind GPU jobs. -> results/heatdome/topk_ablate.npy

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/heatdome
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 python -m graphcast_sae.heatdome.heatdome_topk_ablate
"""
import os, sys, time, json
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr
import graphcast_sae.common.fs_common as fc
from graphcast import rollout
from graphcast_sae.common.signature_physics import gc_km
import graphcast_sae.heatdome.heatdome_config as C
from graphcast_sae.heatdome.heatdome_phase1 import numpyify, build_io
from graphcast_sae.heatdome.heatdome_phase2 import box_diag
from graphcast_sae.heatdome.heatdome_physics_ablate import disk_mask

OUT = fc.ROOT / "results/heatdome"
KS = [5, 10, 20, 40, 80, 160]
PEAK_LEAD = 12   # +78h, ridge peak

def select_topk_ridge():
    d = np.load(OUT / "scan.npy", allow_pickle=True).item()
    peak = np.asarray(d["peak_sum"], float)
    cl = np.asarray(d["cen_lat"])[PEAK_LEAD]; co = np.asarray(d["cen_lon"])[PEAK_LEAD]
    rc = np.asarray(d["ridge_center"], float)
    order = np.argsort(peak)[::-1]
    ranked = []
    for f in order:
        la, lo = float(cl[f]), float(co[f])
        if abs(la) < 1 and abs(lo) < 1: continue          # degenerate global feature
        if gc_km(la, lo, rc[0], rc[1]) > C.RADIUS_KM: continue  # off the ridge
        ranked.append((int(f), la, lo, float(peak[f])))
    feats = [r[0] for r in ranked]
    cents = {str(r[0]): [r[1], r[2]] for r in ranked}
    return feats, cents

def main():
    feats, centroids = select_topk_ridge()
    sets = {f"top{k}": feats[:k] for k in KS}
    allf = sorted(set(feats[:max(KS)]))
    print("ridge-coherent-by-activation feats (top 20):", feats[:20], flush=True)
    print("set sizes:", {k: len(v) for k, v in sets.items()}, flush=True)

    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    la0, la1 = C.BOX["lat"]; blo0, blo1 = C.BOX["lon"]
    inbox = (mlat >= la0) & (mlat <= la1) & (mlon >= blo0) & (mlon <= blo1)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)

    big_mask = disk_mask(mlat, mlon, centroids, allf)
    umask_b = big_mask.astype(bool)

    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        z = jnp.zeros(sae.n_features, jnp.float32)
        _, acts = apply(inp, tg, fr, (z, z, np.zeros(len(mlat), np.float32)))
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    print("measuring NORMAL levels from non-2021 late-June analogs:", flush=True)
    acc = {f: [] for f in allf}; used = []
    for a in C.ANALOGS:
        try:
            c = codes_at(a)
        except Exception as e:
            print(f"  analog {a}: ERROR {e}", flush=True); continue
        for f in allf:
            v = c[umask_b, f]; acc[f].extend(v[v > 0].tolist())
        used.append(a); print(f"  analog {a}: used", flush=True)
    ftarget = np.zeros(sae.n_features, np.float32)
    for f in allf: ftarget[f] = float(np.mean(acc[f])) if acc[f] else 0.0

    def fsel_of(fs):
        v = np.zeros(sae.n_features, np.float32); v[fs] = 1.0; return v
    zeroF = np.zeros(sae.n_features, np.float32); zeroN = np.zeros(len(mlat), np.float32)
    arms = {"baseline": (zeroF, zeroF, zeroN)}
    masks = {}
    for k, fs in sets.items():
        m = disk_mask(mlat, mlon, centroids, fs); masks[k] = m
        arms[k] = (fsel_of(fs), ftarget, m)
    print("disk nodes:", {k: int(masks[k].sum()) for k in sets}, flush=True)

    inp, tgt, frc = build_io(C.IC, C.H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    def roll(patch):
        cur = inp; ridge = []; heat = []
        pj = tuple(jnp.asarray(x) for x in patch)
        for h in range(C.H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, pj)
            p = numpyify(p)
            rr, hh, _, _ = box_diag(p, C.BOX)
            ridge.append(rr); heat.append(hh)
            if h < C.H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        return dict(ridge=np.array(ridge), heat=np.array(heat))

    res = {}
    for aname, patch in arms.items():
        t = time.time(); res[aname] = roll(patch)
        print(f"  [{aname:>10}] ridge_pk {res[aname]['ridge'].max():.0f}  heat_pk {res[aname]['heat'].max():.1f}C  ({time.time()-t:.0f}s)", flush=True)

    base = res["baseline"]["ridge"].max()
    print("\n=== ridge collapse vs K ===", flush=True)
    for k in sets:
        pk = res[k]["ridge"].max(); print(f"  {k:>8} ({len(sets[k]):3d} feats): {pk:.0f}  ({100*(base-pk)/base:+.1f}%)", flush=True)

    out = dict(ic=C.IC, box=C.BOX, ks=KS, sets={k: [int(x) for x in v] for k, v in sets.items()},
               feats_ranked=[int(f) for f in feats], allf=[int(f) for f in allf],
               normal_levels={int(f): float(ftarget[f]) for f in allf}, analogs_used=used,
               disk_nodes={k: int(masks[k].sum()) for k in sets},
               leads_h=(np.arange(C.H) + 1) * 6, res=res, centroids=centroids)
    np.save(OUT / "topk_ablate.npy", out, allow_pickle=True)
    print("\n-> results/heatdome/topk_ablate.npy", flush=True)

if __name__ == "__main__":
    main()
