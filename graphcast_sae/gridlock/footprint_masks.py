"""Project each feature/group footprint from the mesh onto the 0.25 deg RMSE grid.

WHY. Global RMSE is an average over the sphere. A feature that damages the forecast intensely
inside its own footprint can vanish in it -- so "not removable at globally-detectable scale"
is not the same claim as "never matters anywhere". This builds the masks that let the same
rollout be scored locally.

COVERAGE IS THE PREMISE, and it was checked before any GPU time (median footprint spread is
7,851 km, so the union of a large group can be most of the globe):

    group           n feat   union area   median per-feature area
    mesh_locked         27        23.7%                     0.2%
    ctrl_mesh           27        43.6%                     0.2%
    scatter_blob       127        82.8%                     1.2%
    ctrl_blob          127        98.3%                     2.2%

Jaccard overlap with its own control: mesh_locked 15.9%, scatter_blob 81.4%. So the GROUP
footprint test is meaningful for mesh_locked and close to vacuous for scatter_blob -- its
union covers five sixths of the planet and nearly coincides with its control's. Individual
features are local (0.2-1.2%), so the scatter class has to be tested ONE FEATURE AT A TIME.

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_footprint_masks.npz
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.gridlock.footprint_masks
"""
import json
import os
import sys

import numpy as np

import pathlib

class fc:                       # paths only; importing fs_common drags in haiku/jax
    from graphcast_sae.paths import REPO_ROOT as ROOT
    WEIGHTS = ROOT / "graphcast_sae/weights"
    from graphcast_sae.paths import SCRATCH

NW = 12
THRESH = 0.25                      # fires in >=25% of windows
NEAR_KM = float(os.environ.get("FM_NEAR_KM", "60"))   # grid cell counts as in-footprint
# within this distance of an active mesh node. Level-6 mesh spacing is 112 km, so 60 km is
# half a spacing -- the tightest halo that still tiles without holes. 150 km (1.34 spacings)
# was the first try and it dilated mesh_locked to 93.8% of the globe, which turned out not to
# be a threshold artifact: see below.
OUT = fc.ROOT / "results/fs_footprint_masks.npz"

def main():
    from scipy.spatial import cKDTree
    z = np.load(fc.WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(fc.SCRATCH / "fs_iid_dump.npy", mmap_mode="r")

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

    g = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    xyz = np.asarray(g["xyz"], float)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 359.75, 1440)
    LO, LA = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    P = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
    P = P.reshape(-1, 3)
    chord = 2 * np.sin(min(NEAR_KM / 6371.0, np.pi) / 2)

    G = json.load(open(os.environ.get("FM_GROUPS", "/tmp/artifact_groups.json")))
    singles = [int(x) for x in os.environ.get("FM_SINGLES",
                                          "2075,656,2235,586,683,1850").split(",")]
    for f in singles:
        G[f"f{f}"] = [f]

    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    masks = {}
    print(f"\n{'mask':<14}{'n feat':>7}{'mesh nodes':>12}{'grid cells':>12}{'area %':>9}")
    for k, v in G.items():
        act = fires[:, np.array(v)].any(1)
        if act.sum() == 0:
            print(f"{k:<14}{len(v):>7}   EMPTY -- skipped"); continue
        tree = cKDTree(xyz[act])
        near = tree.query_ball_point(P, chord, return_length=True) > 0
        m = near.reshape(len(lat), len(lon))
        masks[k] = m
        print(f"{k:<14}{len(v):>7}{int(act.sum()):>12}{int(m.sum()):>12}"
              f"{100*(w*m).sum()/w.sum():>8.1f}%")

    np.savez_compressed(OUT, lat=lat, lon=lon,
                        **{k: v for k, v in masks.items()})
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
