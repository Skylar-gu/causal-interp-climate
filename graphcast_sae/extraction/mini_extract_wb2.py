"""Scaled layer-8 extraction over WB2-streamed ERA5 for SAE training (design §5, G2).

Streams evenly-spaced teacher-forced windows from WeatherBench 2 (see
`wb2_stream.py`), captures processor step-8 mesh-node embeddings, and writes them
as fp16 to a crash-safe on-disk memmap. Train and val come from disjoint time
ranges (val = a held-out year) so the SAE's validation FVU measures genuine
temporal generalization, not memorization of neighboring tokens/timesteps.

"Project early, store small" applies to the mode series, not to SAE training: the
SAE needs raw activations, so this is the one deliberately large dump -- fp16 and
node-subsampled if asked, landed under activations/raw/.

Run (JAX/graphcast env), in background:

Paper: graphcast_small lane; not in the paper
Inputs: GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed); $GC_SCRATCH/fs_acts (extraction/fs_extract.py)
Outputs: $GC_SCRATCH/mini_acts/ fp16 memmaps + layer<step>_meta.json (--out-dir); status out/extract_wb2_status.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.mini_extract_wb2 --n-train 480 --n-val 96 --train-range 2007-01-01 2018-12-27 --val-range 2020-01-01 2020-12-27 --out-dir $GC_SCRATCH/mini_acts --status out/extract_wb2_status.txt
"""
import argparse
import dataclasses
import functools
import json
import pathlib
import time

import jax
import numpy as np

from graphcast import data_utils

import sys

import graphcast_sae.extraction.mini_extract_layer8 as ex          # noqa: E402
import graphcast_sae.extraction.mini_wb2_stream as wb              # noqa: E402
from graphcast_sae.paths import MINI_ACTS_DIR

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_MESH = 10242
DIM = 512

def even_starts(t0, t1, n):
    """n evenly-spaced valid window-start datetimes across [t0, t1]."""
    cand = wb.valid_start_times(t0, t1, stride_steps=1)
    if n >= len(cand):
        return cand
    idx = np.linspace(0, len(cand) - 1, n).round().astype(int)
    return cand[np.unique(idx)]

def run_split(name, starts, apply, captured, tc, step, out_dir, status_path, t_start):
    n = len(starts)
    mm = np.lib.format.open_memmap(
        out_dir / f"layer{step}_{name}.npy", mode="w+",
        dtype=np.float16, shape=(n * N_MESH, DIM))
    done = 0
    for i, st in enumerate(starts):
        win = wb.build_window(st)
        inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
            win, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(tc))
        captured["count"] = 0
        captured["acts"] = {}
        apply(inp, tgt * np.nan, frc)
        assert captured["count"] == ex.N_MESH_GNN_STEPS, captured["count"]
        emb = np.squeeze(captured["acts"][step], axis=1).astype(np.float16)  # (10242,512)
        mm[i * N_MESH:(i + 1) * N_MESH] = emb
        done += 1
        if i % 10 == 0 or i == n - 1:
            mm.flush()
            el = time.time() - t_start
            msg = (f"[{name}] {done}/{n} windows  ({done*N_MESH:,} tokens)  "
                   f"last={str(st)[:13]}  elapsed={el/60:.1f}m  "
                   f"rate={el/max(done,1):.1f}s/win")
            status_path.write_text(msg + "\n")
            print(msg, flush=True)
    mm.flush()
    return n * N_MESH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=480)
    ap.add_argument("--n-val", type=int, default=96)
    ap.add_argument("--train-range", nargs=2, default=["2007-01-01", "2018-12-27"])
    ap.add_argument("--val-range", nargs=2, default=["2020-01-01", "2020-12-27"])
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--out-dir", default=str(MINI_ACTS_DIR))
    ap.add_argument("--status", default="out/extract_wb2_status.txt")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / args.status
    status_path.parent.mkdir(parents=True, exist_ok=True)

    params, mc, tc, stats = ex.load_model()
    run_forward, captured = ex.build_apply(mc, tc, stats, {args.step})
    apply = functools.partial(run_forward.apply, params, {}, jax.random.PRNGKey(0))
    print(f"loaded graphcast_small; capturing processor step {args.step}", flush=True)

    train_starts = even_starts(*args.train_range, args.n_train)
    val_starts = even_starts(*args.val_range, args.n_val)
    print(f"train: {len(train_starts)} windows {str(train_starts[0])[:10]}..{str(train_starts[-1])[:10]}", flush=True)
    print(f"val:   {len(val_starts)} windows {str(val_starts[0])[:10]}..{str(val_starts[-1])[:10]}", flush=True)

    meta = dict(step=args.step, dim=DIM, n_mesh=N_MESH,
                train_range=args.train_range, val_range=args.val_range,
                n_train_windows=len(train_starts), n_val_windows=len(val_starts),
                train_starts=[str(s) for s in train_starts],
                val_starts=[str(s) for s in val_starts],
                source=wb.WB2_URL, tisr="analytic-1h")
    (out_dir / f"layer{args.step}_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    n_tr = run_split("train", train_starts, apply, captured, tc, args.step, out_dir, status_path, t0)
    n_va = run_split("val", val_starts, apply, captured, tc, args.step, out_dir, status_path, t0)
    done = (f"DONE  train={n_tr:,} tokens  val={n_va:,} tokens  "
            f"total_elapsed={(time.time()-t0)/60:.1f}m -> {out_dir}")
    status_path.write_text(done + "\n")
    print(done, flush=True)

if __name__ == "__main__":
    main()
