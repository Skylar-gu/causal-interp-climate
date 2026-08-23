"""Physics-guided collective ablation of the heat-dome ridge (GPU, bf16).

Reads results/heatdome/scan_sets.json. For each physics-motivated feature SET
  core -> core+flank -> core+jet -> full_physics -> union_all (strongest test)
restore ALL member features to NORMAL within the localized region (union of ~1500 km disks
around each member's event firing centroid), held persistently through the 6-day rollout.
Arms share ONE compiled graph (build_apply_cond / delta_cond). Controls: baseline and a
random matched-count W-NA feature set. Records per lead: z500 ridge-anomaly max, 2m-T max,
z500 & 2m-T box fields (skill vs ERA5), and each member's box firing (internal suppression).

Does any physically-motivated combination collapse the ridge & heat and worsen skill? The
union of ALL strong ridge-firing features is the decisive test for 'genuinely distributed'.

Serialize behind other GPU jobs. Crash-safe: results/heatdome/physics_ablate.npy.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/heatdome
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.heatdome.heatdome_physics_ablate
"""
import os, sys, time, json
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.common.signature_physics import gc_km
import graphcast_sae.heatdome.heatdome_config as C
from graphcast_sae.heatdome.heatdome_phase1 import numpyify, build_io
from graphcast_sae.heatdome.heatdome_phase2 import box_diag

OUT = fc.ROOT / "results/heatdome"
RADIUS = C.RADIUS_KM
ARMS = ["core", "core_flank", "core_jet", "full_physics", "union_all"]

def disk_mask(mlat, mlon, centroids, feats):
    m = np.zeros(len(mlat), np.float32)
    for f in feats:
        clat, clon = centroids[str(f)]
        if not np.isfinite(clat): continue
        m = np.maximum(m, (gc_km(mlat, mlon, clat, clon) < RADIUS).astype(np.float32))
    return m

def pick_random(cat, n, exclude, seed=11):
    fr = cat["firerate"]; clat = cat["clat"]
    clon = np.where(cat["clon"] > 180, cat["clon"] - 360, cat["clon"])
    wna = (clat >= 35) & (clat <= 70) & (clon >= -155) & (clon <= -90)
    pool = [f for f in range(len(fr)) if wna[f] and f not in exclude]
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False))

def main():
    S = json.load(open(OUT / "scan_sets.json"))
    truth = np.load(OUT / "era5_truth.npy", allow_pickle=True).item()
    sets = {k: [int(f) for f in v] for k, v in S["sets"].items()}
    centroids = S["centroids"]
    union = sets["union_all"]
    print("set sizes:", {k: len(v) for k, v in sets.items()}, flush=True)

    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    la0, la1 = C.BOX["lat"]; blo0, blo1 = C.BOX["lon"]
    inbox = (mlat >= la0) & (mlat <= la1) & (mlon >= blo0) & (mlon <= blo1)

    catlg = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    rand = pick_random(catlg, len(union), set(union))
    # random centroids from atlas
    rc_lon = np.where(catlg["clon"] > 180, catlg["clon"] - 360, catlg["clon"])
    for f in rand: centroids[str(f)] = [float(catlg["clat"][f]), float(rc_lon[f])]
    print(f"union n={len(union)}; random control n={len(rand)}: {rand}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)

    allf = sorted(set(union + rand))
    union_mask = disk_mask(mlat, mlon, centroids, union)

    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        z = jnp.zeros(sae.n_features, jnp.float32)
        _, acts = apply(inp, tg, fr, (z, z, np.zeros(len(mlat), np.float32)))
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    # normal levels for all involved features (quiet analogs; skip ridge-present)
    # NORMAL reference = each feature's TYPICAL late-June firing level in the region, from
    # non-2021 June-27 analog years. These analogs ARE the quiet normal (2021 is not among them)
    # we do NOT skip on ridge-firing, because the 632-feature union fires every late-June (many
    # are generic climatological features) and an event-skip would nuke the reference. Capping the
    # event firing at this level strips only the EVENT-ANOMALOUS excess -> honest restore-to-normal.
    print("measuring NORMAL levels from non-2021 late-June analogs (no event-skip):", flush=True)
    acc = {f: [] for f in allf}; used = []
    umask_b = union_mask.astype(bool)
    for a in C.ANALOGS:
        try:
            c = codes_at(a)
        except Exception as e:
            print(f"  analog {a}: ERROR {e}", flush=True); continue
        rf_now = float(c[inbox][:, union].sum())
        for f in allf:
            v = c[umask_b, f]; acc[f].extend(v[v > 0].tolist())
        used.append(a); print(f"  analog {a}: union_ridgefire={rf_now:.0f} used", flush=True)
    ftarget = np.zeros(sae.n_features, np.float32)
    for f in allf: ftarget[f] = float(np.mean(acc[f])) if acc[f] else 0.0
    ev_union = float(np.load(OUT / "scan.npy", allow_pickle=True).item()["box_sum"][:, union].sum(1).max())
    print(f"  event union box peak {ev_union:.0f}; normal levels measured for {len(allf)} features "
          f"(mean ftarget over union {np.mean([ftarget[f] for f in union]):.2f})", flush=True)

    # build arm patches
    zeroF = np.zeros(sae.n_features, np.float32); zeroN = np.zeros(len(mlat), np.float32)
    def fsel_of(feats):
        v = np.zeros(sae.n_features, np.float32); v[feats] = 1.0; return v
    arms = {"baseline": (zeroF, zeroF, zeroN)}
    masks = {}
    for k in ARMS:
        m = disk_mask(mlat, mlon, centroids, sets[k]); masks[k] = m
        arms[k] = (fsel_of(sets[k]), ftarget, m)
    rmask = disk_mask(mlat, mlon, centroids, rand)
    arms["random"] = (fsel_of(rand), ftarget, rmask)
    print("disk nodes:", {k: int(masks[k].sum()) for k in ARMS}, "random:", int(rmask.sum()), flush=True)

    inp, tgt, frc = build_io(C.IC, C.H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    def roll(patch):
        cur = inp; ridge = []; heat = []; zf = []; tf = []
        per = {f: [] for f in allf}
        pj = tuple(jnp.asarray(x) for x in patch)
        for h in range(C.H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, pj)
            X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
            Cc = np.asarray(sae.codes(X))
            for f in allf: per[f].append(float(Cc[inbox, f].sum()))
            p = numpyify(p)
            rr, hh, zb, tb = box_diag(p, C.BOX)
            ridge.append(rr); heat.append(hh); zf.append(zb); tf.append(tb)
            if h < C.H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        return dict(ridge=np.array(ridge), heat=np.array(heat),
                    z500_box=np.stack(zf).astype(np.float32), t2m_box=np.stack(tf).astype(np.float32),
                    box_feats={f: np.array(per[f]) for f in allf})

    res = {}
    for aname, patch in arms.items():
        t = time.time(); res[aname] = roll(patch)
        setfire = np.sum([res[aname]["box_feats"][f] for f in (union if aname != "baseline" else union)], axis=0)
        print(f"  [{aname:>12}] ridge {np.array2string(res[aname]['ridge'],precision=0,max_line_width=250)}", flush=True)
        print(f"  [{aname:>12}] heat_pk {res[aname]['heat'].max():.1f}C  union_box_pk {setfire.max():.0f}  ({time.time()-t:.0f}s)", flush=True)

    out = dict(ic=C.IC, box=C.BOX, sets=sets, rand=rand, allf=[int(f) for f in allf],
               normal_levels={int(f): float(ftarget[f]) for f in allf}, analogs_used=used,
               disk_nodes={k: int(masks[k].sum()) for k in ARMS}, rand_disk=int(rmask.sum()),
               leads_h=(np.arange(C.H) + 1) * 6, res=res, centroids=centroids)
    np.save(OUT / "physics_ablate.npy", out, allow_pickle=True)
    print("\n-> results/heatdome/physics_ablate.npy", flush=True)

if __name__ == "__main__":
    main()
