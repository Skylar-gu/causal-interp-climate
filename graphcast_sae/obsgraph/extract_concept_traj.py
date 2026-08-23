"""Concept-feature trajectory extraction — observational PCMCI+ input.

Contiguous teacher-forced trajectory (block-streaming, crash-safe, resume-able) through
the flagship model over real ERA5. Instead of projecting pool-member footprints
(extract_traj_flag.py), we capture the SELECTED concept features: per 6-h step, encode the
layer-8 mesh-node activations with `sae.codes`, and record each selected feature's activation
SUMMED over the 40,962 mesh nodes -> (steps, n_features) series + target_times.

Nodes: results/fs_pcmci_nodes.npy  (dict 'sel': concept -> [feature ids], 30 features total).

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS=--xla_gpu_autotune_level=0 \

Paper: Sec. 4 / Appendix app:null (observational PCMCI+ input series)
Inputs: results/fs_pcmci_nodes.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: --out series (default activations/mode_series/feat_traj_3yr.npy) and --out-all (all 4096 features); status out/extract_concept_traj_status.txt
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.obsgraph.extract_concept_traj --start 2016-01-01 --n-steps 4383 --block 120 --out activations/mode_series/feat_traj_3yr.npy
"""
import argparse, os, sys, time
os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp
import graphcast_sae.common.fs_common as fc
from pathlib import Path

ROOT = fc.ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--n-steps", type=int, default=4383)
    ap.add_argument("--block", type=int, default=120)
    ap.add_argument("--nodes", default=str(ROOT / "results/fs_pcmci_nodes.npy"))
    ap.add_argument("--out", default=str(ROOT / "activations/mode_series/feat_traj_3yr.npy"))
    ap.add_argument("--out-all", default=str(ROOT / "activations/mode_series/feat_traj_3yr_ALL4096.npy"),
                    help="full 4096-feature per-step summed codes (float16), for future reselection")
    ap.add_argument("--status", default="out/extract_concept_traj_status.txt")
    args = ap.parse_args()
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp_all = Path(args.out_all); outp_all.parent.mkdir(parents=True, exist_ok=True)

    nd = np.load(args.nodes, allow_pickle=True).item()
    sel = nd["sel"]; concepts = list(nd["concepts"])
    # flat ordered list of (concept, feature_id); concept order = nd['concepts']
    feat_ids, feat_concept = [], []
    for c in concepts:
        for f in sel[c]:
            feat_ids.append(int(f)); feat_concept.append(c)
    feat_ids = np.asarray(feat_ids, np.int64)
    n_feat = len(feat_ids)
    print(f"{n_feat} selected features over {len(concepts)} concepts", flush=True)

    series = np.zeros((args.n_steps, n_feat), np.float32)
    series_all = np.zeros((args.n_steps, 4096), np.float32)   # all-feature summed codes
    target_times = np.empty(args.n_steps, "datetime64[ns]")
    done = 0
    # resume from a prior crash-safe save (needs the ALL-4096 file to restore both)
    if outp_all.exists():
        prev = np.load(outp_all, allow_pickle=True).item()
        if prev.get("start") == args.start:
            pn = int(prev.get("n_done", 0))
            series_all[:pn] = prev["series"][:pn]
            series[:pn] = series_all[:pn][:, feat_ids]
            target_times[:pn] = prev["target_times"][:pn]
            done = pn
            print(f"RESUME from n_done={done}", flush=True)

    params, mc, tc, stats = fc.load_model()
    sae = fc.SAEJax()
    rf, cap = fc.build_apply(mc, tc, stats, sae=None, bf16=True)
    apply = fc.make_apply(params, rf, patched=False)
    @jax.jit
    def encode_all(acts):
        A = acts.reshape(-1, fc.D_IN).astype(jnp.float32)     # (40962,512)
        codes = sae.codes(A)                                  # (40962,4096)
        return codes.sum(0).astype(jnp.float32)               # (4096,) summed over mesh

    print(f"backend={jax.default_backend()}; trajectory {args.start} x {args.n_steps} steps",
          flush=True)

    t_start = np.datetime64(args.start); status = ROOT / args.status
    t0 = time.time(); b = 0; sess0 = done
    while done < args.n_steps:
        n_win = min(args.block, args.n_steps - done)
        blk_start = t_start + done * fc.STEP
        block = fc.load_block(blk_start + fc.STEP, nframes=n_win + 2)   # frames [blk_start..]
        for s in range(n_win):
            inp, tgt, frc = fc.build_batch_inputs([block], s, tc)
            _, acts = apply(inp, tgt * np.nan, frc)
            allsum = np.asarray(encode_all(acts), np.float32)   # (4096,)
            series_all[done] = allsum
            series[done] = allsum[feat_ids]
            target_times[done] = blk_start + (s + 1) * fc.STEP
            done += 1
        b += 1; el = time.time() - t0; dsess = max(done - sess0, 1)
        msg = (f"block {b}: {done}/{args.n_steps}  last={str(target_times[done-1])[:13]}  "
               f"{el/60:.1f}m  {el/dsess:.2f}s/win  "
               f"eta={(args.n_steps-done)*el/dsess/60:.0f}m")
        print(msg, flush=True); status.write_text(msg + "\n")
        np.save(outp, dict(series=series, target_times=target_times,
                feat_ids=feat_ids, feat_concept=feat_concept, concepts=concepts,
                sel=sel, start=args.start, n_done=done, n_feat=n_feat), allow_pickle=True)
        # float32 (not float16): summed codes over 40962 nodes exceed float16 max (65504)
        # for strongly-firing features -> float16 overflows to inf. 3yr*4096*f32 ~ 72 MB.
        np.save(outp_all, dict(series=series_all.astype(np.float32),
                target_times=target_times, start=args.start, n_done=done,
                note="per-step SAE code summed over 40962 mesh nodes, all 4096 features (float32)"),
                allow_pickle=True)
    status.write_text(f"DONE {done} in {(time.time()-t0)/60:.1f}m -> {args.out}\n")
    print(f"DONE -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
