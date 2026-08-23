"""Finalise the flagship trajectory: derive the zero-GPU anchors, then run the DATA GATE.

`qperm_flag` and `qrand_flag` share `vmax_flag`'s footprints exactly, so their series are
a CPU dot product against the SAME pooled tensor — they cost no GPU. This script reads the
q-agnostic pooled dump written by `extract_traj_flag2.py --dump-pooled`, adds those two
members, and then applies the pre-registered data gate (prereg §5, DG-1..DG-6) before any
analysis is allowed to touch the file.

The gate exists because a local trajectory in this repo was once 98.6% zeros and silently
corrupted three analyses before a reproduction gate caught it.

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: candidates/pool_flag_v2_chandirs.npy (not shipped, see docs/REPRODUCE.md)
Outputs: --out finalized series (required); results/flag_gint_datagate.json (--report)
Run:   # JAX env, CPU
    OMP_NUM_THREADS=8 python -m graphcast_sae.obsgraph.finalize_traj_flag --traj activations/mode_series/traj_flag2.npy --pooled $GC_SCRATCH/pooled --chandirs candidates/pool_flag_v2_chandirs.npy --out activations/mode_series/traj_flag2_full.npy
"""
import argparse, json
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
# Anchors that share another member's FOOTPRINTS and therefore its pooled tensor: their
# series are a CPU dot product, no GPU. qrandc_flag is the amendment-A1 Nyquist-clean
# anchor and belongs here too — it was omitted on the first pass, which silently dropped
# it from the finalized trajectory while it was present in the pool and the chandirs.
ALIAS = {"qperm_flag": "vmax_flag", "qrand_flag": "vmax_flag",
         "qrandc_flag": "vmax_flag"}
STEP_NS = np.timedelta64(6, "h").astype("timedelta64[ns]").astype(np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--chandirs", default="candidates/pool_flag_v2_chandirs.npy")
    ap.add_argument("--expect-steps", type=int, default=0)
    ap.add_argument("--truncate-years", action="store_true",
                    help="prereg §5 hard-stop rule: if the extraction was stopped early, "
                         "cut to the largest WHOLE year boundary so the windows-as-"
                         "realisations split stays one-window-per-year, and log the drop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require", default="",
                    help="comma list of members that MUST end up in the trajectory. A "
                         "member present in the pool but neither forward-projected nor "
                         "derivable would otherwise vanish silently and only surface as a "
                         "KeyError hours later, mid-analysis.")
    ap.add_argument("--report", default="results/flag_gint_datagate.json")
    args = ap.parse_args()

    tr = np.load(ROOT / args.traj, allow_pickle=True).item()
    ch = np.load(ROOT / args.chandirs, allow_pickle=True).item()
    pdir = Path(args.pooled)
    n_done = int(tr["n_done"])
    requested = int(args.expect_steps or n_done)
    dropped = None
    if args.truncate_years and n_done < requested:
        # 1461 = 365.25 d/yr in 6-h steps; keep whole years only
        nyr = int(n_done // 1461)
        keep = nyr * 1461
        assert keep > 0, f"only {n_done} steps — less than one whole year"
        dropped = dict(requested=requested, extracted=n_done, kept=keep,
                       whole_years=nyr, dropped_steps=int(n_done - keep),
                       dropped_vs_requested=int(requested - keep))
        print(f"HARD-STOP RULE (prereg §5): requested {requested}, extracted {n_done}, "
              f"keeping {keep} = {nyr} whole years; dropping {n_done-keep} trailing steps "
              f"and {requested-keep} vs the requested span.")
        n_done = keep
    series = {n: np.asarray(v)[:n_done] for n, v in tr["series"].items()}
    times = tr["target_times"][:n_done]
    names = list(tr["names"])

    # ── zero-GPU anchors from the shared pooled tensor ──────────────────────
    for a, src in ALIAS.items():
        f = pdir / f"pooled_{src}.npy"
        if a in series or not f.exists() or a not in ch:
            continue
        P = np.load(f, mmap_mode="r")
        q = np.asarray(ch[a]["q"], np.float32)
        off = (np.asarray(ch[a]["mbar"], np.float32) * q).sum(1)
        out = np.empty((n_done, q.shape[0]), np.float32)
        for i in range(0, n_done, 2048):
            blk = np.asarray(P[i:min(i + 2048, n_done)], np.float32)
            out[i:i + blk.shape[0]] = np.einsum("bnd,nd->bn", blk, q) - off
        series[a] = out; names.append(a)
        print(f"  derived {a:<12} from pooled_{src} ({n_done} steps x {q.shape[0]} modes)")

    # ── DG-0: every required member actually made it ────────────────────────
    req = [m for m in args.require.split(",") if m]
    missing = [m for m in req if m not in series]
    if missing:
        print(f"\n** MEMBERS MISSING FROM THE TRAJECTORY: {missing} **")
        print(f"   present: {sorted(series)}")
        print(f"   a member must be either forward-projected by the extractor "
              f"(--members) or derivable from a shared pooled tensor (ALIAS={ALIAS}).")
        raise SystemExit("DG-0 FAILED — stopping before analysis")

    # ── DATA GATE ───────────────────────────────────────────────────────────
    g, fails = {}, []
    g["DG0_members"] = dict(required=req, present=sorted(series), ok=True)
    exp = n_done if args.truncate_years else (args.expect_steps or n_done)
    g["DG1_n_done"] = dict(n_done=n_done, requested=requested, expected=exp,
                           truncation=dropped, ok=bool(n_done == exp))

    dt = np.diff(times.astype("datetime64[ns]").astype(np.int64))
    g["DG5_time"] = dict(first=str(times[0]), last=str(times[-1]),
                         uniform_6h=bool(len(dt) and (dt == STEP_NS).all()),
                         n_years=round(float(n_done * 6 / 24 / 365.25), 2),
                         ok=bool(len(dt) and (dt == STEP_NS).all()))

    per = {}
    for n in names:
        S = series[n]
        zrow = int((np.abs(S).sum(1) == 0).sum())
        nan = int((~np.isfinite(S)).sum())
        sd = S.std(0)
        per[n] = dict(shape=list(S.shape), all_zero_rows=zrow, n_nonfinite=nan,
                      min_std=float(sd.min()), max_std=float(sd.max()),
                      frac_zero_entries=float((S == 0).mean()))
    g["per_member"] = per
    g["DG2_no_zero_rows"] = dict(ok=all(v["all_zero_rows"] == 0 for v in per.values()))
    g["DG3_finite"] = dict(ok=all(v["n_nonfinite"] == 0 for v in per.values()))
    g["DG4_min_std"] = dict(ok=all(v["min_std"] > 0 for v in per.values()))

    # DG-6: the pooled tensor must reconstruct the stored scalar series
    rec = {}
    for n in tr["names"]:
        f = pdir / f"pooled_{n}.npy"
        if not f.exists():
            continue
        k = min(2000, n_done)
        P = np.asarray(np.load(f, mmap_mode="r")[:k], np.float32)
        q = np.asarray(ch[n]["q"], np.float32)
        r = np.einsum("bnd,nd->bn", P, q) - (np.asarray(ch[n]["mbar"], np.float32) * q).sum(1)
        rel = float(np.abs(r - series[n][:k]).max() / (series[n][:k].std() + 1e-12))
        rec[n] = rel
    g["DG6_pooled_reconstruct"] = dict(rel_err=rec, ok=all(v < 5e-3 for v in rec.values()))

    print("\n" + "=" * 78)
    print(f"DATA GATE — {args.traj}")
    print("=" * 78)
    print(f"{'member':<15}{'T':>7}{'N':>4}{'zero-rows':>11}{'nonfinite':>11}"
          f"{'min sd':>10}{'max sd':>10}{'frac==0':>9}")
    for n, v in per.items():
        print(f"{n:<15}{v['shape'][0]:>7}{v['shape'][1]:>4}{v['all_zero_rows']:>11}"
              f"{v['n_nonfinite']:>11}{v['min_std']:>10.4g}{v['max_std']:>10.4g}"
              f"{v['frac_zero_entries']:>9.4f}")
    for k in ("DG1_n_done", "DG5_time", "DG2_no_zero_rows", "DG3_finite",
              "DG4_min_std", "DG6_pooled_reconstruct"):
        ok = g[k]["ok"]
        if not ok:
            fails.append(k)
        extra = {kk: vv for kk, vv in g[k].items() if kk != "ok"}
        print(f"  {k:<24} {'PASS' if ok else '** FAIL **'}   {extra}")
    g["all_pass"] = not fails
    Path(ROOT / args.report).parent.mkdir(exist_ok=True)
    json.dump(g, open(ROOT / args.report, "w"), indent=1, default=str)
    print(f"\nGATE {'PASS' if not fails else 'FAIL ' + str(fails)}  -> {args.report}")
    if fails:
        raise SystemExit(f"DATA GATE FAILED: {fails} — stopping before analysis")

    np.save(ROOT / args.out, dict(series=series, target_times=times, names=names,
            start=tr["start"], n_done=n_done,
            Ns={n: series[n].shape[1] for n in names},
            provenance=dict(traj=args.traj, pooled=str(pdir), chandirs=args.chandirs,
                            derived=list(ALIAS))), allow_pickle=True)
    print(f"wrote {args.out}  members={names}")

if __name__ == "__main__":
    main()
