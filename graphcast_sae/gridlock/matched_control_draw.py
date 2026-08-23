"""Draw a control for each feature matched on COVERAGE and CONNECTIVITY.

Global RMSE is an area-weighted mean, so it partly reports how much of the planet a feature
covers: across the 17 features already ablated, spearman(global effect, footprint area) =
+0.547, p = 0.023. Comparing a 10% feature against a 0.1% feature therefore compares coverage,
not importance. A control matched on coverage makes the dilution identical on both sides so
the difference is meaningful.

Connectivity is the second axis and it is what separates the visual classes: a weather blob is
one connected object, a lattice feature is hundreds of isolated nodes. Matching it stops the
control from being "same area, completely different shape".

  coverage      cos-lat weighted share of the sphere the footprint covers
  n_comp        connected components of the footprint on the mesh graph (adjacency at
                1.6x the 112 km level-6 spacing, as in footprint_inspect.py)
  frac_largest  share of active nodes in the biggest component
  degree        mean multi-mesh degree of the active nodes

Controls are drawn from features NOT in the candidate set, matched within tolerance on
coverage and on frac_largest, then ranked by closeness in n_comp and degree.

Paper: Sec. 3 coverage/connectivity-matched controls (results/matched_controls.json)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_connectivity.npy; results/matched_controls.json
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.matched_control_draw
"""
import json
import os
import pathlib
import sys

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH, MESH_GEOM
WEIGHTS = ROOT / "graphcast_sae/weights"
NW = 8
THRESH = 0.25
OUT = ROOT / "results/matched_controls.json"
SEED = 31

def main():
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    cnt = np.zeros((L, 4096), np.float32)
    for wi, j in enumerate(np.linspace(0, META["n_windows"] - 1, NW).astype(int)):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, 32, axis=1)[:, :32]
        a = np.zeros_like(pre); r = np.arange(len(A))[:, None]; a[r, idx] = 1.0
        cnt += a
        print(f"  window {wi+1}/{NW}", flush=True)
    fires = cnt >= NW * THRESH

    g = np.load(MESH_GEOM, allow_pickle=True).item()
    xyz = np.asarray(g["xyz"], float); xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    w = np.cos(np.deg2rad(np.asarray(g["lat"], float)))
    R, SP = 6371.0, np.sqrt(4 * np.pi * 6371.0 ** 2 / L)
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(min(1.6 * SP / R, np.pi) / 2),
                                     output_type="ndarray")
    deg = np.bincount(pairs.ravel(), minlength=L).astype(float)
    print(f"mesh adjacency: {len(pairs):,} edges at <{1.6*SP:.0f} km, "
          f"mean degree {deg.mean():.1f}", flush=True)

    cov = np.zeros(4096); ncomp = np.zeros(4096); fl = np.zeros(4096); dg = np.zeros(4096)
    for f in range(4096):
        on = fires[:, f]
        n = int(on.sum())
        cov[f] = (w * on).sum() / w.sum()
        if n == 0:
            ncomp[f] = 0; fl[f] = 0; dg[f] = 0; continue
        dg[f] = deg[on].mean()
        keep = on[pairs[:, 0]] & on[pairs[:, 1]]
        idx = np.where(on)[0]
        pos = -np.ones(L, np.int64); pos[idx] = np.arange(n)
        e = pos[pairs[keep]]
        m = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        k, lab = connected_components(m, directed=False)
        ncomp[f] = k
        fl[f] = np.bincount(lab).max() / n
        if f % 1000 == 0:
            print(f"  feature {f}/4096", flush=True)

    T = json.load(open("/tmp/gridlock_types.json"))
    SEL = [f for v in T.values() for f in v]
    alive = (cov > 0) & (fires.sum(0) >= 10)
    banned = set(SEL)
    rng = np.random.default_rng(SEED)
    out, report = {}, []
    for f in SEL:
        for tol in (0.30, 0.50, 0.80, 2.0):
            ok = np.where(alive
                          & (np.abs(cov - cov[f]) <= tol * max(cov[f], 1e-9))
                          & (np.abs(fl - fl[f]) <= 0.20)
                          & ~np.isin(np.arange(4096), list(banned)))[0]
            if len(ok) >= 3:
                break
        if len(ok) == 0:
            report.append((f, None)); continue
        score = (np.abs(np.log(np.maximum(ncomp[ok], 1)) - np.log(max(ncomp[f], 1)))
                 + np.abs(dg[ok] - dg[f]) / max(dg[f], 1e-9))
        c = int(ok[np.argsort(score)[:3][rng.integers(0, min(3, len(ok)))]])
        banned.add(c)
        out[f"f{f}"] = [f]; out[f"ctrl_f{f}"] = [c]
        report.append((f, c))
    print(f"\n{'feat':>6}{'cov':>8}{'ncomp':>7}{'fracL':>7}{'deg':>6}"
          f"   ->{'ctrl':>6}{'cov':>8}{'ncomp':>7}{'fracL':>7}{'deg':>6}")
    for f, c in report:
        if c is None:
            print(f"{f:>6}   NO MATCH"); continue
        print(f"{f:>6}{100*cov[f]:>7.2f}%{int(ncomp[f]):>7}{fl[f]:>7.2f}{dg[f]:>6.1f}"
              f"   ->{c:>6}{100*cov[c]:>7.2f}%{int(ncomp[c]):>7}{fl[c]:>7.2f}{dg[c]:>6.1f}")
    out["floor"] = []
    json.dump(out, open(OUT, "w"), indent=1)
    np.save(ROOT / "results/fs_connectivity.npy",
            dict(cov=cov, ncomp=ncomp, frac_largest=fl, degree=dg), allow_pickle=True)
    print(f"\n-> {OUT}   ({len([k for k in out if k.startswith('ctrl_')])} pairs)")
    print("ARMS=" + ",".join(k for k in out))

if __name__ == "__main__":
    main()
