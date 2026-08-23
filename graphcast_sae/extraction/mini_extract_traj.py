"""Extract contiguous 6-hourly mode SERIES from GraphCast layer-8 activations.

The i.i.d. 480-window dump cannot carry lagged dynamics; PCMCI needs a contiguous
teacher-forced trajectory. This streams a contiguous ERA5 block from WB2 once
(amortized I/O), runs graphcast_small teacher-forced on each consecutive 3-step
window, captures layer-8 mesh-node activations A_t (10242,512), and projects them
through every candidate's rank-1 spatio-channel mode:

    s_c(t) = (p_c^T A_t - mbar_c) . q_c

("project early, store small": only the scalar series (T, N) per candidate lands).

Run (graphcast/JAX env), background:

Paper: graphcast_small lane; not in the paper
Inputs: candidates/pool_v2_candidates.npy (not shipped, see docs/REPRODUCE.md); candidates/pool_v2_channel_dirs.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed)
Outputs: --out series .npy (required); <--dump-pooled>/pooled_meta.json when asked; status out/extract_traj_status.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.mini_extract_traj --start 2011-01-01 --n-steps 8760 --block 120 --out activations/mode_series/traj_2011.npy --status out/extract_traj_status.txt
"""
import argparse, dataclasses, functools, json, pathlib, time
import jax, numpy as np, xarray as xr
from graphcast import data_utils, solar_radiation as sr

import sys

import graphcast_sae.extraction.mini_extract_layer8 as ex        # noqa: E402
import graphcast_sae.extraction.mini_wb2_stream as wb            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_MESH, DIM, STEP = 10242, 512, np.timedelta64(6, "h")

def build_batches(block_ds, statics, abs_times, tc, bs):
    """Yield (inputs, targets, forcings, target_times) batching `bs` 3-step windows."""
    n = len(abs_times) - 2
    for i0 in range(0, n, bs):
        idxs = list(range(i0, min(i0 + bs, n)))
        wins = []
        for i in idxs:
            at = abs_times[i:i + 3]
            w = block_ds.sel(time=at)
            w = w.assign_coords(time=(at - at[0]).astype("timedelta64[ns]"))
            wins.append(w)
        big = xr.concat(wins, dim="batch")
        for v in wb.STATIC_VARS:
            big[v] = statics[v]
        dts = np.stack([abs_times[i:i + 3] for i in idxs]).astype("datetime64[ns]")
        big = big.assign_coords(datetime=(("batch", "time"), dts))
        inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
            big, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(tc))
        yield inp, tgt, frc, np.array([abs_times[i + 2] for i in idxs])

def stream_block(ds, statics, t_start, n_times):
    """Load a contiguous block of n_times steps + analytic TISR, once."""
    times = t_start + np.arange(n_times) * STEP
    blk = ds[list(wb.SURFACE_VARS) + list(wb.ATMOS_VARS)].sel(time=times).load()
    templ = blk["2m_temperature"].assign_coords(datetime=("time", times))
    tisr = sr.get_toa_incident_solar_radiation_for_xarray(
        templ, integration_period="1h", num_integration_bins=360)
    blk["toa_incident_solar_radiation"] = tisr
    return blk, times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--n-steps", type=int, required=True, help="number of 6h windows")
    ap.add_argument("--block", type=int, default=120)
    ap.add_argument("--bs", type=int, default=12, help="GPU batch size (windows/forward)")
    ap.add_argument("--cands", default=str(ROOT / "candidates/pool_v2_candidates.npy"),
                    help="POOL_V2 is authoritative: {leiden,vmax,km,shift,sae,qperm}. "
                         "act_candidates.npy is the superseded v1 pool (has blur, "
                         "lacks sae/qperm) and will NOT reproduce published results.")
    ap.add_argument("--chandirs", default=str(ROOT / "candidates/pool_v2_channel_dirs.npy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", default="out/extract_traj_status.txt")
    ap.add_argument("--dump-pooled", default=None,
                    help="ALSO write the q-AGNOSTIC pooled tensor (T,N,512) fp16 per member "
                         "to this directory. q_c is otherwise baked into the stored scalar "
                         "series, so every new channel-direction hypothesis costs another "
                         "full ~15 h of forwards. With this, a refit becomes a CPU "
                         "operation: s_c(t) = (pooled[t,c] . q_c) - (mbar_c . q_c). "
                         "Cost ~466 MB/member at T=17532, N=26 (2.8 GB for the 6-member pool).")
    args = ap.parse_args()

    cd = np.load(args.cands, allow_pickle=True).item()
    ch = np.load(args.chandirs, allow_pickle=True).item()
    names = list(cd["cands"])
    Wm = {n: cd["cands"][n].astype(np.float32) for n in names}       # (N,10242)
    Q = {n: ch[n]["q"].astype(np.float32) for n in names}            # (N,512)
    mbar_proj = {n: (ch[n]["mbar"] * ch[n]["q"]).sum(1).astype(np.float32) for n in names}  # (N,) = mbar.q
    series = {n: np.empty((args.n_steps, cd["cands"][n].shape[0]), np.float32) for n in names}

    # q-agnostic pooled tensor (see --dump-pooled). Memmapped so a crash keeps what landed.
    pooled_mm = None
    if args.dump_pooled:
        pd_dir = pathlib.Path(args.dump_pooled); pd_dir.mkdir(parents=True, exist_ok=True)
        pooled_mm = {n: np.lib.format.open_memmap(
            pd_dir / f"pooled_{n}.npy", mode="w+", dtype=np.float16,
            shape=(args.n_steps, cd["cands"][n].shape[0], 512)) for n in names}
        json.dump(dict(n_steps=args.n_steps, start=args.start, names=names,
                       cands=args.cands, dim=512, dtype="float16",
                       note="s_c(t) = (pooled[t,c] . q_c) - (mbar_c . q_c); q NOT baked in"),
                  open(pd_dir / "pooled_meta.json", "w"), indent=1)

    params, mc, tc, stats = ex.load_model()
    run_forward, captured = ex.build_apply(mc, tc, stats, {8})
    apply = functools.partial(run_forward.apply, params, {}, jax.random.PRNGKey(0))
    ds, statics = wb.open_wb2()
    print(f"loaded graphcast_small; trajectory {args.start} x {args.n_steps} steps", flush=True)

    t_start = np.datetime64(args.start)
    status_path = ROOT / args.status
    target_times = np.empty(args.n_steps, dtype="datetime64[ns]")
    done = 0
    t0 = time.time()
    b = 0
    while done < args.n_steps:
        n_win = min(args.block, args.n_steps - done)
        blk_start = t_start + done * STEP
        blk, _ = stream_block(ds, statics, blk_start, n_win + 2)
        abs_times = blk_start + np.arange(n_win + 2) * STEP
        for inp, tgt, frc, ttimes in build_batches(blk, statics, abs_times, tc, args.bs):
            captured["count"] = 0; captured["acts"] = {}
            apply(inp, tgt * np.nan, frc)
            Ab = captured["acts"][8].astype(np.float32)   # (10242, B, 512)
            for bi in range(Ab.shape[1]):
                A = Ab[:, bi, :]                          # (10242,512)
                for n in names:
                    pooled = Wm[n] @ A                    # (N,512)
                    series[n][done] = (pooled * Q[n]).sum(1) - mbar_proj[n]
                    if pooled_mm is not None:
                        pooled_mm[n][done] = pooled.astype(np.float16)
                target_times[done] = ttimes[bi]
                done += 1
            ttime = ttimes[-1]
        b += 1
        el = time.time() - t0
        msg = (f"block {b}: {done}/{args.n_steps} windows  last={str(ttime)[:13]}  "
               f"elapsed={el/60:.1f}m  rate={el/done:.2f}s/win  "
               f"eta={(args.n_steps-done)*el/done/60:.0f}m")
        status_path.write_text(msg + "\n")
        print(msg, flush=True)
        if pooled_mm is not None:
            for n in names:
                pooled_mm[n].flush()
        # incremental save (crash-safe)
        np.save(ROOT / args.out, dict(series=series, target_times=target_times,
                names=names, start=args.start, n_done=done,
                Ns={n: cd["cands"][n].shape[0] for n in names}), allow_pickle=True)

    out = dict(series=series, target_times=target_times, names=names,
               start=args.start, n_done=done,
               Ns={n: cd["cands"][n].shape[0] for n in names})
    np.save(ROOT / args.out, out, allow_pickle=True)
    status_path.write_text(f"DONE {done} windows in {(time.time()-t0)/60:.1f}m -> {args.out}\n")
    print(f"DONE -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
