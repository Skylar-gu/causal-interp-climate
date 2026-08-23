"""Grid-lockedness for all 4,096 features, so 'the grid-locked features' becomes a real set.

Every group ablated so far was selected by the MESH-PREFERENCE statistic, which turns out not
to predict grid-lockedness: of the 27-feature "lattice group", only 2 of the 16 with a
measured score are grid-locked at all. So no run to date has actually ablated the grid-locked
features as a class.

Score = mean Jaccard overlap of a feature's active node set between pairs of dates, over
NW dates spanning a year. Computed in feature chunks so the masks fit in memory.

Paper: Sec. 3 grid-lock score (results/fs_gridlock_all.npy)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_gridlock_all.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.gridlock_score_all
"""
import itertools
import json
import os
import pathlib

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH
WEIGHTS = ROOT / "graphcast_sae/weights"
NW = int(os.environ.get("GA_NW", "12"))
CH = int(os.environ.get("GA_CHUNK", "512"))
OUT = ROOT / "results/fs_gridlock_all.npy"

def main():
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L, NF = META["n_mesh"], 4096
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    wins = np.linspace(0, META["n_windows"] - 1, NW).astype(int)

    masks = np.zeros((NW, L, NF), np.bool_)
    amps = np.zeros((NW, NF), np.float64)
    for wi, j in enumerate(wins):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, 32, axis=1)[:, :32]
        C = np.zeros_like(pre); r = np.arange(L)[:, None]; C[r, idx] = pre[r, idx]
        masks[wi] = C > 0
        amps[wi] = C.sum(0)
        print(f"  window {wi+1}/{NW}", flush=True)

    inter = np.zeros(NF); union = np.zeros(NF); npair = 0
    for a, b in itertools.combinations(range(NW), 2):
        for s in range(0, NF, CH):
            e = min(s + CH, NF)
            ma, mb = masks[a][:, s:e], masks[b][:, s:e]
            inter[s:e] += (ma & mb).sum(0)
            union[s:e] += (ma | mb).sum(0)
        npair += 1
    score = inter / np.maximum(union, 1)
    alive = masks.sum(1).min(0) >= 5          # fires somewhere in every window
    cv = amps.std(0) / np.maximum(amps.mean(0), 1e-9)

    np.save(OUT, dict(score=score, alive=alive, amp_cv=cv, nw=NW), allow_pickle=True)
    a = score[alive]
    print(f"\n{int(alive.sum())} live features of {NF}, {npair} date pairs")
    print(f"score: median {np.median(a):.3f}  p90 {np.percentile(a,90):.3f}  "
          f"p99 {np.percentile(a,99):.3f}  max {a.max():.3f}")
    for cut in (0.30, 0.45, 0.60):
        n = int(((score >= cut) & alive).sum())
        print(f"  >= {cut:.2f}: {n:>4} features ({100*n/alive.sum():.1f}% of live)")
    top = np.where(alive)[0][np.argsort(-score[alive])][:25]
    print("\ntop 25:", ", ".join(f"f{f}({score[f]:.2f})" for f in top))

if __name__ == "__main__":
    main()
