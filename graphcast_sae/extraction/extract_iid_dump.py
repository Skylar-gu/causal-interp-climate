"""Flagship pool build — step 0: an i.i.d. layer-8 activation dump for clustering.

Samples N frozen i.i.d. 3-frame windows across a multi-year span, runs the flagship forward,
captures layer-8 mesh acts (40,962×512), and stacks them into one fp16 dump
(N*40962, 512) — the flagship analogue of the mini `layer8_train.npy`. Feeds leiden/vmax/km
clustering and the SAE-cluster member. GPU, ~4 s/window.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: infrastructure: the 160-window i.i.d. layer-8 dump (GC_SCRATCH/fs_iid_dump.npy) every atlas/gridlock/concept script reads
Inputs: GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: $GC_SCRATCH/fs_iid_dump.npy + fs_iid_meta.json; status out/fs_iid_status.txt
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.extraction.extract_iid_dump --n 160
"""
import argparse, json, os, sys, time
os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import graphcast_sae.common.fs_common as fc

SCRATCH = fc.SCRATCH
DUMP = SCRATCH / "fs_iid_dump.npy"
META = SCRATCH / "fs_iid_meta.json"

def frozen_starts(n, seed=0, y0=2016, y1=2020):
    """n i.i.d. 6-h-grid window centres across [y0, y1], seeded (frozen sample)."""
    lo = np.datetime64(f"{y0}-01-05T00"); hi = np.datetime64(f"{y1}-12-25T00")
    span_h = int((hi - lo) / np.timedelta64(1, "h"))
    rng = np.random.default_rng(seed)
    hrs = np.sort(rng.choice(span_h // 6, size=n, replace=False)) * 6
    return (lo + hrs.astype("timedelta64[h]")).astype("datetime64[ns]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", default="out/fs_iid_status.txt")
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"backend={jax.default_backend()} {jax.devices()}", flush=True)

    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=None, bf16=True)
    apply = fc.make_apply(params, rf, patched=False)

    starts = frozen_starts(args.n, args.seed)
    dump = np.lib.format.open_memmap(DUMP, mode="w+", dtype=np.float16,
                                     shape=(args.n * fc.N_MESH, fc.D_IN))
    status = fc.ROOT / args.status
    t0 = time.time()
    for i, c in enumerate(starts):
        blk = fc.load_block(c, nframes=fc.INPUT_WINDOW)
        inp, tgt, frc = fc.build_batch_inputs([blk], 0, tc)
        _, acts = apply(inp, tgt * np.nan, frc)
        A = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        dump[i * fc.N_MESH:(i + 1) * fc.N_MESH] = A.astype(np.float16)
        if i % 20 == 0 or i == args.n - 1:
            el = time.time() - t0
            msg = (f"[{i+1}/{args.n}] {str(c)[:13]}  acts{A.shape} "
                   f"mean={A.mean():+.4f} std={A.std():.4f}  "
                   f"{el/60:.1f}m  eta={(args.n-i-1)*el/(i+1)/60:.0f}m")
            print(msg, flush=True); status.write_text(msg + "\n")
    dump.flush()
    json.dump(dict(n_windows=args.n, seed=args.seed, n_mesh=fc.N_MESH, dim=fc.D_IN,
                   starts=[str(s) for s in starts], hook_step=fc.HOOK_STEP,
                   source=fc.ZARR, dtype="float16", dump=str(DUMP)),
              open(META, "w"), indent=1)
    print(f"DONE {args.n} windows in {(time.time()-t0)/60:.1f}m -> {DUMP} "
          f"({dump.shape}, {DUMP.stat().st_size/1e9:.1f} GB)", flush=True)
    status.write_text(f"DONE {args.n} windows -> {DUMP}\n")

if __name__ == "__main__":
    main()
