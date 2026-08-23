"""Store real footprints for a chosen set of features, count connected components, and map them.

WHY. `frag = spread / nearest-neighbour distance` turned out to be a fake two-part statistic:
spearman(frag, spread) = +0.942 but spearman(frag, nn) = +0.031. The denominator does nothing,
because nn is pinned at the mesh floor for essentially every feature (p1 111 km, p50 115 km,
p95 146 km, against a level-6 spacing of 112 km). Activations are contiguous for ALL features
-- random placement of ~320 nodes would give 631 km -- so nn cannot discriminate between them.

What `frag` therefore could NOT see is the distinction that was actually asked about: one
sprawling band versus twenty scattered islands. Both give nn = 114 km and spread = 10,000 km.
Telling them apart needs CONNECTED COMPONENTS on the mesh graph, which is what this computes:
adjacency by great-circle distance < 1.6 x mesh spacing, then a union-find over active nodes.

    n_comp        how many disconnected pieces the feature lights up
    frac_largest  share of active nodes in the biggest piece
                  -> one object: n_comp small, frac_largest near 1
                  -> scattered islands: n_comp large, frac_largest small

It also writes the footprints so they can be plotted and checked by eye, because a component
count is only as good as the adjacency threshold and the map shows immediately whether the
number matches what is actually there.

Paper: figures/paper_fig_maps.py / paper_fig_gain.py (results/fs_footprints.npy)
Inputs: results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md); results/fs_mesh_detectors.npy (not shipped, see docs/REPRODUCE.md); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_footprints.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.gridlock.footprint_inspect
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import MESH_GEOM

NW = 8                              # windows to accumulate footprints over
SAE_NPZ = fc.WEIGHTS / "sae_k32_lat4096_lay08.npz"
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
L = META["n_mesh"]
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
OUT = fc.ROOT / "results/fs_footprints.npy"
R = 6371.0
SPACING = np.sqrt(4 * np.pi * R * R / 40962)          # 112 km

def encode(A, Wenc, bpre, k=32):
    xn = A - A.mean(1, keepdims=True)
    xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre)
    r = np.arange(len(A))[:, None]
    out[r, idx] = pre[r, idx]
    return out

def main():
    from scipy.spatial import cKDTree

    M = np.load(fc.ROOT / "results/fs_mesh_detectors.npy", allow_pickle=True).item()
    a = np.load(fc.ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
    fr = np.asarray(a["firerate"], float)
    mb, sp, nw = M["mesh_bias"], M["spread_km"], M["nwin"]
    ok = (nw >= 12) & (fr > 0.0015)

    mesh_top = np.where(ok)[0][np.argsort(-mb[ok])][:4]
    spread_top = np.where(ok & (mb < 0.5))[0][
        np.argsort(-sp[ok & (mb < 0.5)])][:4]
    mid = np.where(ok & (mb < 0.5) & (sp < np.percentile(sp[ok], 50)))[0]
    typical = mid[np.argsort(-fr[mid])][:2]
    known = [553, 3174]                                # screened `ascent` features
    SEL = list(dict.fromkeys([int(x) for x in
                              list(mesh_top) + list(spread_top) +
                              list(typical) + known]))
    print(f"inspecting {len(SEL)} features: {SEL}", flush=True)

    g = np.load(MESH_GEOM, allow_pickle=True).item()
    xyz = np.asarray(g["xyz"], float)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    lon = np.where(lon > 180, lon - 360, lon)

    tree = cKDTree(xyz)
    chord = 2 * np.sin(min(1.6 * SPACING / R, np.pi) / 2)
    pairs = tree.query_pairs(chord, output_type="ndarray")
    print(f"mesh adjacency: {len(pairs)} edges at <{1.6*SPACING:.0f} km "
          f"({len(pairs)/L:.1f} per node; icosahedral mesh has ~3)", flush=True)

    z = np.load(SAE_NPZ)
    Wenc, bpre = z["W_enc"], z["b_pre"]
    starts_all = list(META["starts"])
    starts = np.array(starts_all)[np.linspace(0, META["n_windows"] - 1, NW).astype(int)]
    X = np.load(DUMP, mmap_mode="r")

    acc = {f: np.zeros(L, np.int32) for f in SEL}
    for wi, s in enumerate(starts):
        j = starts_all.index(str(s))
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        act = encode(A, Wenc, bpre) > 0
        for f in SEL:
            acc[f] += act[:, f]
        print(f"  window {wi+1}/{NW}", flush=True)

    def components(mask):
        """union-find over active nodes using the precomputed mesh adjacency."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0, 0.0, []
        pos = -np.ones(L, np.int64)
        pos[idx] = np.arange(len(idx))
        parent = np.arange(len(idx))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in pairs:
            if mask[u] and mask[v]:
                ru, rv = find(pos[u]), find(pos[v])
                if ru != rv:
                    parent[ru] = rv
        roots = np.array([find(i) for i in range(len(idx))])
        _, cnt = np.unique(roots, return_counts=True)
        return len(cnt), float(cnt.max() / cnt.sum()), sorted(cnt.tolist(), reverse=True)

    res = {}
    print(f"\n{'feat':>6}{'mesh_bias':>11}{'spread km':>11}{'n_active':>10}"
          f"{'n_comp':>8}{'frac_largest':>14}   biggest pieces")
    for f in SEL:
        mask = acc[f] >= max(1, NW // 4)               # fires in >=25% of windows
        nc, fl, sizes = components(mask)
        res[f] = dict(n_active=int(mask.sum()), n_comp=nc, frac_largest=fl,
                      sizes=sizes[:6], mesh_bias=float(mb[f]), spread=float(sp[f]),
                      nodes=np.where(mask)[0].astype(np.int32))
        print(f"{f:>6}{mb[f]:>11.2f}{sp[f]:>11.0f}{int(mask.sum()):>10}"
              f"{nc:>8}{fl:>14.2f}   {sizes[:6]}")

    np.save(OUT, dict(res=res, sel=SEL, lat=lat, lon=lon, nw=NW,
                      spacing_km=float(SPACING)), allow_pickle=True)
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
