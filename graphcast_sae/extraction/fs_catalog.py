"""Flagship SAE suite — step 2: encode every window through the PUBLISHED SAE.

Produces the shared cache the rest of the suite reads, and the suite's gate number.

Accumulators (same shape/meaning as the small-model `retry1_encode.npz`, so the
retry ports below are line-for-line comparable):
    featmap (F, N)  mean code of feature f at mesh node n over windows
    P       (F, W)  per-(feature, window) node-sum
    G       (F, F)  Gram, sum_{w,n} a_f a_g
    fire    (F,)    number of (window, node) firings
    mass    (F,)    total activation mass
    fvu     (W,)    reconstruction FVU vs the RAW activation, per window

GATE (prereg §0): median FVU <= 0.20 at processor step 9, else the suite is BLOCKED.

Paper: infrastructure: encoded catalog for the retry suite (legacy)
Inputs: $GC_SCRATCH/fs_acts (extraction/fs_extract.py)
Outputs: $GC_SCRATCH/fs_catalog.npz (--out); results/graphcast_sae_catalog.json (--json)
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.fs_catalog
"""
import argparse
import json
import time

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import FS_CATALOG

GATE_FVU = 0.20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FS_CATALOG))
    ap.add_argument("--json", default="results/graphcast_sae_catalog.json")
    args = ap.parse_args()

    meta = json.load(open(fc.ACTS_DIR / "meta.json"))
    starts = [s for s in meta["starts"] if fc.act_path(np.datetime64(s)).exists()]
    W = len(starts)
    sae = fc.sae_numpy()
    F = sae["W_enc"].shape[0]
    N = meta["n_mesh"]
    print(f"{W} windows x {N} nodes = {W*N:,} tokens; dict {F}, k {sae['k']}", flush=True)

    featmap = np.zeros((F, N), np.float64)
    P = np.zeros((F, W), np.float64)
    G = np.zeros((F, F), np.float64)
    fire = np.zeros(F, np.float64)
    fvu = np.zeros(W)
    t0 = time.time()
    for w, s in enumerate(starts):
        X = np.load(fc.act_path(np.datetime64(s))).astype(np.float32)
        z, recon = fc.encode_np(X, sae, want_recon=True)
        fvu[w] = fc.fvu_raw(X, recon)
        zd = z.astype(np.float64)
        featmap += zd.T
        P[:, w] = zd.sum(0)
        G += zd.T @ zd
        fire += (z > 0).sum(0)
        print(f"  [{w+1}/{W}] {s} FVU={fvu[w]:.4f} L0={(z>0).sum(1).mean():.1f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    featmap /= W
    mass = P.sum(1)
    rate = fire / (W * N)
    alive = int((fire > 0).sum())

    np.savez(args.out, featmap=featmap.astype(np.float32), P=P, G=G, fire=fire,
             fvu=fvu, mass=mass, rate=rate, starts=np.array(starts), n_mesh=N, F=F)

    med = float(np.median(fvu))
    gate = med <= GATE_FVU
    summary = dict(
        program="flagship-SAE suite (0.25/37lev + published SAE) -- NOT cross-comparable to small-model G1",
        sae_repo="theodoremacmillan/sae-graphcast-k32-lat4096-lay08",
        hook="processor step 9 (1-indexed) == authors' 0-indexed layer0008, post-residual",
        n_windows=W, n_mesh=N, dict=F, k=int(sae["k"]),
        fvu_median=med, fvu_min=float(fvu.min()), fvu_max=float(fvu.max()),
        fvu_note="scored against the RAW activation: the SAE normalizes its input per "
                 "token but its training loss targeted the un-normalized input",
        gate_fvu=GATE_FVU, gate_passed=bool(gate),
        alive_features=alive, dead_features=int(F - alive),
        rate_median=float(np.median(rate[fire > 0])),
        source=meta["source"], windows=starts)
    json.dump(summary, open(fc.ROOT / args.json, "w"), indent=1)
    print(f"\nFVU median {med:.4f} (min {fvu.min():.4f} max {fvu.max():.4f})  "
          f"GATE<= {GATE_FVU} -> {'PASS' if gate else 'BLOCKED'}")
    print(f"alive {alive}/{F}; saved -> {args.out} and {args.json}")

if __name__ == "__main__":
    main()
