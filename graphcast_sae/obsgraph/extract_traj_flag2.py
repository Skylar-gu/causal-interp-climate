"""Flagship mode-series extraction, v2 — q-AGNOSTIC and pipelined.

Two changes vs `extract_traj_flag.py` (which is left untouched):

1. **`--dump-pooled DIR`** — also write the (T, N, 512) fp16 *pooled* tensor per member,
   BEFORE the channel projection. The scalar series bakes q_c in
   (`s_c(t) = (p_c^T A_t)·q_c - mbar_c·q_c`), so every new channel-direction hypothesis
   otherwise costs another full multi-hour GPU run. With the pooled tensor on disk a
   refit is a CPU dot product (`reproject_flag.py` (not in the release)).
   Cost: T·N·512·2 bytes/member  (T=8768, N=39 -> 350 MB/member).

2. **Pipelined prefetch.** The v1 loop was strictly serial:
   block download (~2.9 s/step amortised) -> window assembly (~2.0 s) -> GPU forward
   (~2.1 s) = ~7.0 s/window measured. Those three use different resources, so they are
   now overlapped with two producer threads and bounded queues:
       thread A: fc.load_block  ->  block queue (maxsize 1)
       thread B: fc.build_batch_inputs -> step queue (maxsize 4)
       main:     jitted forward + projection
   Steady-state rate becomes max(download, assembly, forward) instead of their sum.
   Memory is bounded by the queue sizes: ~3 blocks in flight at 947 MB/frame.
   **Keep --block small (<=24) when prefetching** — a 122-frame block is ~115 GB.

Everything else (crash-safe per-block incremental saves, output format, the projection
arithmetic) is byte-identical to v1; `--smoke-ref` asserts that against a v1 output.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: candidates/pool_flag_v2_candidates.npy (not shipped, see docs/REPRODUCE.md); candidates/pool_flag_v2_chandirs.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: --out series (default activations/mode_series/traj_flag2.npy); <--dump-pooled>/pooled_meta.json + per-member pooled tensors; status out/extract_traj_flag2_status.txt
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.obsgraph.extract_traj_flag2 --start 2007-01-01 --n-steps 8768 --block 20 --cands candidates/pool_flag_v2_candidates.npy --chandirs candidates/pool_flag_v2_chandirs.npy --members leiden_flag,sae_flag,sae_sel_flag,vmax_flag,km_flag,shift_flag --dump-pooled $GC_SCRATCH/pooled --out activations/mode_series/traj_flag2.npy
"""
import argparse, json, os, queue, sys, threading, time
os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax
import graphcast_sae.common.fs_common as fc
from pathlib import Path

ROOT = fc.ROOT
_SENTINEL = object()

def block_producer(t_start, n_steps, block, bq, stop):
    """Thread A — download contiguous ERA5 blocks ahead of the consumer."""
    done = 0
    try:
        while done < n_steps and not stop.is_set():
            n_win = min(block, n_steps - done)
            blk_start = t_start + done * fc.STEP
            blk = fc.load_block(blk_start + fc.STEP, nframes=n_win + 2)
            while not stop.is_set():
                try:
                    bq.put((done, n_win, blk_start, blk), timeout=5); break
                except queue.Full:
                    continue
            done += n_win
    except BaseException as e:                                   # surface, don't hang
        bq.put(("ERR", e, None, None))
        return
    bq.put(_SENTINEL)

def step_producer(bq, sq, tc, stop):
    """Thread B — assemble GraphCast inputs for each 6-h window ahead of the GPU."""
    try:
        while not stop.is_set():
            item = bq.get()
            if item is _SENTINEL:
                break
            if item[0] == "ERR":
                sq.put(item); return
            done0, n_win, blk_start, blk = item
            for s in range(n_win):
                if stop.is_set():
                    return
                inp, tgt, frc = fc.build_batch_inputs([blk], s, tc)
                ttime = blk_start + (s + 1) * fc.STEP
                while not stop.is_set():
                    try:
                        sq.put((done0 + s, inp, tgt, frc, ttime), timeout=5); break
                    except queue.Full:
                        continue
            del blk
    except BaseException as e:
        sq.put(("ERR", e, None, None, None))
        return
    sq.put(_SENTINEL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--n-steps", type=int, default=8768)
    ap.add_argument("--block", type=int, default=20,
                    help="frames per download; 3 blocks are in flight, ~947 MB/frame")
    ap.add_argument("--cands", default=str(ROOT / "candidates/pool_flag_v2_candidates.npy"))
    ap.add_argument("--chandirs", default=str(ROOT / "candidates/pool_flag_v2_chandirs.npy"))
    ap.add_argument("--members", default="", help="comma list; default = every pool member")
    ap.add_argument("--dump-pooled", default=None)
    ap.add_argument("--out", default=str(ROOT / "activations/mode_series/traj_flag2.npy"))
    ap.add_argument("--status", default="out/extract_traj_flag2_status.txt")
    ap.add_argument("--log-every", type=int, default=0,
                    help="log/save every N windows (0 = every 600 s)")
    ap.add_argument("--smoke-ref", default=None,
                    help="v1 trajectory .npy to check the first steps against, then exit")
    ap.add_argument("--smoke-steps", type=int, default=24)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cd = np.load(args.cands, allow_pickle=True).item()
    ch = np.load(args.chandirs, allow_pickle=True).item()
    names = [n for n in (args.members.split(",") if args.members else list(cd["cands"])) if n]
    miss = [n for n in names if n not in cd["cands"] or n not in ch]
    assert not miss, f"members absent from pool/chandirs: {miss}"
    Wm = {n: cd["cands"][n].astype(np.float32) for n in names}
    Q = {n: ch[n]["q"].astype(np.float32) for n in names}
    mbar_proj = {n: (ch[n]["mbar"] * ch[n]["q"]).sum(1).astype(np.float32) for n in names}
    series = {n: np.zeros((args.n_steps, Wm[n].shape[0]), np.float32) for n in names}
    target_times = np.zeros(args.n_steps, "datetime64[ns]")

    pooled_mm = None
    if args.dump_pooled:
        pdir = Path(args.dump_pooled); pdir.mkdir(parents=True, exist_ok=True)
        pooled_mm = {n: np.lib.format.open_memmap(
            pdir / f"pooled_{n}.npy", mode="w+", dtype=np.float16,
            shape=(args.n_steps, Wm[n].shape[0], fc.D_IN)) for n in names}
        json.dump(dict(n_steps=args.n_steps, start=args.start, names=names,
                       cands=args.cands, dim=fc.D_IN, dtype="float16",
                       note="s_c(t) = (pooled[t,c] . q_c) - (mbar_c . q_c); q NOT baked in"),
                  open(pdir / "pooled_meta.json", "w"), indent=1)

    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=None, bf16=True)
    apply = fc.make_apply(params, rf, patched=False)
    print(f"backend={jax.default_backend()}; members={names} N={Wm[names[0]].shape[0]}; "
          f"{args.start} x {args.n_steps} steps; block={args.block}; "
          f"pooled_dump={'yes' if pooled_mm else 'no'}", flush=True)

    t_start = np.datetime64(args.start)
    status = ROOT / args.status
    bq, sq, stop = queue.Queue(maxsize=1), queue.Queue(maxsize=4), threading.Event()
    ta = threading.Thread(target=block_producer,
                          args=(t_start, args.n_steps, args.block, bq, stop), daemon=True)
    tb = threading.Thread(target=step_producer, args=(bq, sq, tc, stop), daemon=True)
    ta.start(); tb.start()

    WARM = 8                      # windows to drop before quoting a steady-state rate
    done, t0, last_save, t_warm = 0, time.time(), time.time(), None
    try:
        while done < args.n_steps:
            item = sq.get()
            if item is _SENTINEL:
                break
            if item[0] == "ERR":
                raise RuntimeError("producer failed") from item[1]
            idx, inp, tgt, frc, ttime = item
            _, acts = apply(inp, tgt * np.nan, frc)
            A = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)          # (40962,512)
            for n in names:
                pooled = Wm[n] @ A                                          # (N,512)
                series[n][idx] = (pooled * Q[n]).sum(1) - mbar_proj[n]
                if pooled_mm is not None:
                    pooled_mm[n][idx] = pooled.astype(np.float16)
            target_times[idx] = ttime
            done = idx + 1
            if done == WARM:
                t_warm = time.time()
            if args.smoke_ref and done >= args.smoke_steps:
                break
            due = (done % args.log_every == 0) if args.log_every else \
                  (time.time() - last_save > 600)
            if due or done == args.n_steps:
                el = time.time() - t0
                ss = ((time.time() - t_warm) / max(done - WARM, 1)
                      if t_warm and done > WARM else el / done)
                msg = (f"{done}/{args.n_steps}  last={str(ttime)[:13]}  {el/60:.1f}m  "
                       f"{el/done:.2f}s/win (steady {ss:.2f})  "
                       f"eta={(args.n_steps-done)*ss/3600:.1f}h")
                print(msg, flush=True); status.write_text(msg + "\n")
                np.save(ROOT / args.out, dict(
                    series=series, target_times=target_times, names=names,
                    start=args.start, n_done=done, Ns={n: Wm[n].shape[0] for n in names},
                    cands=args.cands, chandirs=args.chandirs), allow_pickle=True)
                if pooled_mm is not None:
                    for n in names:
                        pooled_mm[n].flush()
                last_save = time.time()
    finally:
        stop.set()

    if args.smoke_ref:
        ref = np.load(ROOT / args.smoke_ref, allow_pickle=True).item()
        print(f"\nSMOKE vs {args.smoke_ref} on the first {done} steps:")
        ok = True
        # the forward is bf16 and XLA autotuning is not bitwise reproducible across
        # runs, so agreement is demanded at 5e-3 of a mode's own SD (a semantic bug —
        # an off-by-one window, a wrong q — is O(1) at this scale, not O(1e-3)).
        TOL = 5e-3
        for n in names:
            if n not in ref["series"]:
                print(f"  {n:<14} not in reference — skipped"); continue
            a, b = series[n][:done], ref["series"][n][:done]
            rel = float(np.abs(a - b).max() / (b.std() + 1e-12))
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            ok &= rel < TOL
            print(f"  {n:<14} max|Δ|/sd = {rel:.2e}  corr = {1-r:.1e} from 1  "
                  f"{'OK' if rel < TOL else '** MISMATCH **'}")
        tref = ref["target_times"][:done]
        same_t = bool((target_times[:done] == tref).all())
        print(f"  target_times identical: {same_t}")
        if pooled_mm is not None:
            for n in names:
                P = np.asarray(pooled_mm[n][:done], np.float32)
                rec = np.einsum("bnd,nd->bn", P, Q[n]) - mbar_proj[n]
                rel = float(np.abs(rec - series[n][:done]).max() / (series[n][:done].std() + 1e-12))
                ok &= rel < 5e-3
                print(f"  {n:<14} pooled->series rel err {rel:.2e}  "
                      f"{'OK' if rel < 5e-3 else '** FAIL **'}")
        print(f"SMOKE {'PASS' if (ok and same_t) else 'FAIL'}")
        return

    np.save(ROOT / args.out, dict(series=series, target_times=target_times, names=names,
            start=args.start, n_done=done, Ns={n: Wm[n].shape[0] for n in names},
            cands=args.cands, chandirs=args.chandirs), allow_pickle=True)
    el = time.time() - t0
    status.write_text(f"DONE {done}/{args.n_steps} in {el/3600:.2f}h "
                      f"({el/max(done,1):.2f}s/win) -> {args.out}\n")
    print(f"DONE {done} steps in {el/3600:.2f}h -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
