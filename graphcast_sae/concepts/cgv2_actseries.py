"""CG-v2 step 0: the per-feature ACTIVATION SERIES used by the decorrelation gate.

The v2 selection rule (docs/prereg/prereg_concept_graph_v2.md, step 4) needs, for every SAE
feature, a series it can be correlated against. This encodes the already-extracted IID
activation dump (160 windows x 40,962 mesh nodes x 512, fp16, spanning 2016-01..2020-12,
57 distinct months -- the SAME sample the representation atlas was built on) through the
flagship SAE and sums each window's codes over mesh nodes.

Output: results/fs_cgv2_actseries.npy  {series (160, 4096) float32, starts (160,)}

This produces NO causal number. It is node definition, run before the prereg is frozen so
that the surviving group sizes and the strike list can be declared in the prereg itself --
exactly as v1 declared div250 struck before its run.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Fig. fig:contrast (a)/(c): concept response operators on generic initial states
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_cgv2_actseries.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.concepts.cgv2_actseries
"""
import os, sys, json, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp
import graphcast_sae.common.fs_common as fc

DUMP = fc.SCRATCH / "fs_iid_dump.npy"
META = fc.SCRATCH / "fs_iid_meta.json"
OUT = fc.ROOT / "results/fs_cgv2_actseries.npy"

def main():
    meta = json.load(open(META))
    n, nm, dim = meta["n_windows"], meta["n_mesh"], meta["dim"]
    print(f"backend={jax.default_backend()}  dump {n} x {nm} x {dim}", flush=True)
    X = np.load(DUMP, mmap_mode="r")
    assert X.shape == (n * nm, dim), X.shape

    sae = fc.SAEJax()
    sum_codes = jax.jit(lambda x: sae.codes(x).sum(0))

    series = np.zeros((n, sae.n_features), np.float32)
    t0 = time.time()
    CH = 8192
    for i in range(n):
        acc = np.zeros(sae.n_features, np.float64)
        blk = np.asarray(X[i * nm:(i + 1) * nm], np.float32)
        for s in range(0, nm, CH):
            acc += np.asarray(sum_codes(jnp.asarray(blk[s:s + CH])), np.float64)
        series[i] = acc
        if i % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {(time.time()-t0)/60:.1f}m", flush=True)

    np.save(OUT, dict(series=series, starts=meta["starts"], n_mesh=nm,
                      note="per-window SAE code sum over 40962 mesh nodes; "
                           "basis for the CG-v2 decorrelation gate"),
            allow_pickle=True)
    print(f"-> {OUT}  series {series.shape}  "
          f"features never active: {(series.max(0) == 0).sum()}", flush=True)

if __name__ == "__main__":
    main()
