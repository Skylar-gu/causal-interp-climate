"""Footprints for an arbitrary feature list, on the flagship mesh. CPU only.

`footprint_inspect.py` stores footprints for the twelve features its detectors selected.
The paper's map figure needs one more thing those twelve do not contain: the CONVECTION
group itself (2401, 2067, 3174), so that the panel contrasting "the model looking at its
own grid" against "the model looking at the atmosphere" uses the group every intervention
in the paper acts on, rather than a stand-in.

Same encoder, same windows, same accumulation as footprint_inspect, so the counts are
directly comparable and can be merged into one figure. Writes a SEPARATE file rather than
extending results/fs_footprints.npy, which another lane owns.

Paper: figures/paper_fig_maps.py (convection-group footprints)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_footprints_extra.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.footprints_extra            # default: 2401,2067
    FP_FEATS=2401,2067,3243 python -m graphcast_sae.gridlock.footprints_extra
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import MESH_GEOM

NW = 8
SAE_NPZ = fc.WEIGHTS / "sae_k32_lat4096_lay08.npz"
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
L = META["n_mesh"]
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
OUT = fc.ROOT / "results/fs_footprints_extra.npy"
FEATS = [int(x) for x in os.environ.get("FP_FEATS", "2401,2067,3243").split(",")]

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
    g = np.load(MESH_GEOM, allow_pickle=True).item()
    lat = np.asarray(g["lat"], float)
    lon = np.asarray(g["lon"], float)
    lon = np.where(lon > 180, lon - 360, lon)

    z = np.load(SAE_NPZ)
    Wenc, bpre = z["W_enc"], z["b_pre"]
    starts_all = list(META["starts"])
    starts = np.array(starts_all)[np.linspace(0, META["n_windows"] - 1, NW).astype(int)]
    X = np.load(DUMP, mmap_mode="r")
    print(f"features {FEATS}; {NW} windows of {L} mesh nodes", flush=True)

    cnt = {f: np.zeros(L, np.int32) for f in FEATS}
    amp = {f: np.zeros(L, np.float64) for f in FEATS}
    for wi, s in enumerate(starts):
        j = starts_all.index(str(s))
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        act = encode(A, Wenc, bpre)
        for f in FEATS:
            cnt[f] += (act[:, f] > 0)
            amp[f] += act[:, f]
        print(f"  window {wi+1}/{NW}", flush=True)

    res = {}
    for f in FEATS:
        nodes = np.where(cnt[f] > 0)[0]
        res[f] = dict(nodes=nodes.astype(np.int32), count=cnt[f].astype(np.int32),
                      mean_amp=(amp[f] / NW).astype(np.float32), n_active=int(len(nodes)))
        print(f"  f{f}: {len(nodes)} active nodes "
              f"({100*len(nodes)/L:.2f}% of mesh), peak count {cnt[f].max()}/{NW}", flush=True)
    np.save(OUT, dict(res=res, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                      nw=NW, feats=FEATS), allow_pickle=True)
    print("->", OUT, flush=True)

if __name__ == "__main__":
    main()
