"""Flagship SAE suite — step 1: extract layer-8 mesh-node activations.

24 teacher-forced windows evenly spaced across 2021 (prereg §1), streamed from the
authors' own WB2 0.25°/37-lev derived zarr, one window per forward (CPU, ~3 min).
Saved in the authors' file naming so their tooling reads them unchanged:
    layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t<YYYY-MM-DDTHH>.npy   (N,512) fp16

Paper: infrastructure: 24-window 2021 extraction for the retry suite (legacy)
Inputs: GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed); $GC_SCRATCH/fs_acts (extraction/fs_extract.py)
Outputs: $GC_SCRATCH/fs_acts/<window>.npy (fp16, one per window) + fs_acts/meta.json; status out/fs_extract_status.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.fs_extract --n 24 --year 2021 --status out/fs_extract_status.txt
"""
import argparse
import json
import time

import numpy as np

import graphcast_sae.common.fs_common as fc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--status", default="out/fs_extract_status.txt")
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    import jax
    fc.ACTS_DIR.mkdir(parents=True, exist_ok=True)
    status = fc.ROOT / args.status
    starts = fc.seasonal_starts(args.n, args.year)
    print(f"backend={jax.default_backend()}  {args.n} windows {starts[0]} .. {starts[-1]}",
          flush=True)

    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=None)
    apply = fc.make_apply(params, rf, patched=False)

    t0 = time.time()
    done = []
    for i, c in enumerate(starts):
        out = fc.act_path(c)
        if out.exists() and not args.overwrite:
            print(f"[{i+1}/{args.n}] {c} cached", flush=True)
            done.append(str(c))
            continue
        blk = fc.load_block(c)
        inp, tgt, frc = fc.build_batch_inputs([blk], 0, tc)
        _, A = apply(inp, tgt * np.nan, frc)
        A = np.asarray(A, np.float32).reshape(-1, fc.D_IN)
        np.save(out, A.astype(np.float16))
        done.append(str(c))
        msg = (f"[{i+1}/{args.n}] {c}  acts{A.shape} "
               f"mean={A.mean():+.4f} std={A.std():.4f}  {(time.time()-t0)/60:.1f}m")
        print(msg, flush=True)
        status.write_text(msg + "\n")
        json.dump(dict(starts=[str(s) for s in starts], done=done, year=args.year,
                       n_mesh=int(A.shape[0]), dim=int(A.shape[1]),
                       hook_step=fc.HOOK_STEP, source=fc.ZARR),
                  open(fc.ACTS_DIR / "meta.json", "w"), indent=1)
    msg = f"DONE {len(done)}/{args.n} windows in {(time.time()-t0)/60:.1f}m -> {fc.ACTS_DIR}"
    print(msg, flush=True)
    status.write_text(msg + "\n")

if __name__ == "__main__":
    main()
