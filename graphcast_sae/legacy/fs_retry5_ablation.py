"""Flagship retry #5, ABLATION half — is a grid-locked feature causally inert?

Port of `probe/retry5_ablation.py`. Error-preserving ablation (coef = -1 on one
feature) applied at layer 8 at EVERY step of a teacher-forced delta roll, against
mass-matched random single-feature controls, a `noise` arm (numerical floor) and a
`recon` arm (the SAE's own reconstruction error in forecast space).

Sample sizes are the CPU-budgeted ones frozen in
`docs/prereg/prereg_flagship_g2_suite.md` §3: 4 windows, 4 steps, 10 controls, 1 target.
The p-floor at 10 controls is 1/11 = 0.091 and is reported as such — never rounded
to "significant at 0.05".

Run (~7 h on CPU):

Paper: not in the paper; kept for provenance only
Inputs: results/graphcast_sae_gridlock.json (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/graphcast_sae_ablation.npy (--out); status out/fs_ablation_status.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.fs_retry5_ablation --nwin 4 --nsteps 4 --nctrl 10
"""
import argparse
import json
import time

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import FS_CATALOG

def matched_controls(mass, target, exclude, n, rng):
    """Alive features within a factor 2 of the target's activation mass."""
    lm = np.log(np.clip(mass, 1e-12, None))
    ok = np.where((mass > 0) & ~np.isin(np.arange(mass.size), exclude))[0]
    pool = ok[np.abs(lm[ok] - lm[target]) <= np.log(2)]
    if pool.size >= 4 * n:
        return rng.choice(pool, n, replace=False), "random-in-band"
    order = ok[np.argsort(np.abs(lm[ok] - lm[target]))]
    return order[:n], "nearest-neighbour-fallback"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwin", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=4)
    ap.add_argument("--nctrl", type=int, default=10)
    ap.add_argument("--ntarget", type=int, default=1)
    # Which detection candidates to test. Default 0 = the top one (#2954, already run and
    # MISS). --tstart 1 tests the second candidate without re-running the first; the
    # combined verdict is a 2-target test, declared in docs/prereg/prereg_flagship_steering_v2.md §5.
    ap.add_argument("--tstart", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--cat", default=str(FS_CATALOG))
    ap.add_argument("--out", default="results/graphcast_sae_ablation.npy")
    ap.add_argument("--status", default="out/fs_ablation_status.txt")
    args = ap.parse_args()

    gl = json.load(open(fc.ROOT / "results/graphcast_sae_gridlock.json"))
    if not gl["passed_detection"]:
        print("detection MISS -> ablation not run (prereg §3)"); return
    targets = [c["feature"] for c in
               gl["candidates"][args.tstart:args.tstart + args.ntarget]]
    if not targets:
        print(f"no candidate at index {args.tstart} (detection found "
              f"{len(gl['candidates'])}) -> nothing to run"); return
    z = np.load(args.cat, allow_pickle=True)
    mass, rate = z["mass"], z["rate"]
    all_cand = [c["feature"] for c in gl["candidates"]]

    rng = np.random.default_rng(0)
    ctrl, ctrl_mode = {}, {}
    for t in targets:
        ctrl[t], ctrl_mode[t] = matched_controls(mass, t, all_cand, args.nctrl, rng)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=sae)
    apply = fc.make_apply(params, rf)

    arms = [("ref", fc.noop_patch(sae)), ("noise", fc.noop_patch(sae))]
    for t in targets:
        arms.append((f"gl{t}", fc.coef_patch(sae, [t], -1.0)))
        for i, f in enumerate(ctrl[t]):
            arms.append((f"ctrl{t}_{i}", fc.coef_patch(sae, [int(f)], -1.0)))
    arms.append(("recon", fc.recon_patch(sae)))
    names = [a[0] for a in arms]
    print(f"{len(arms)} arms; targets {targets}; controls {args.nctrl} "
          f"({[ctrl_mode[t] for t in targets]})", flush=True)

    starts = fc.seasonal_starts(args.nwin * 6, 2021)[::6][:args.nwin]
    nframes = args.nsteps + 2
    E = np.full((len(arms), args.nwin, args.nsteps), np.nan)
    relA = np.full((len(arms), args.nwin), np.nan)
    scorer = None
    t0 = time.time()

    for ci in range(0, args.nwin, args.chunk):
        cs = starts[ci:ci + args.chunk]
        blocks = [fc.load_block(c, nframes) for c in cs]
        base_inputs = [fc.build_batch_inputs(blocks, s, tc) for s in range(args.nsteps)]

        base_pred, base_A0 = [], None
        for s in range(args.nsteps):
            inp, tgt, frc = base_inputs[s]
            p, A = apply(inp, tgt * np.nan, frc, arms[0][1])
            if scorer is None:
                scorer = fc.Scorer(stats, p)
            base_pred.append({v: np.asarray(p[v].data, np.float64) for v in fc.PROG})
            if s == 0:
                base_A0 = np.asarray(A, np.float32)
        idims = {v: base_inputs[0][0][v].dims for v in fc.PROG}
        pdims = {v: scorer.dims[v] for v in fc.PROG}
        roll_perm = {v: [[d for d in pdims[v] if d != "time"].index(d)
                         for d in [d for d in idims[v] if d != "time"]] for v in fc.PROG}
        base_inp_np = [{v: np.asarray(base_inputs[s][0][v].values, np.float64)
                        for v in fc.PROG} for s in range(args.nsteps)]

        for ai, (name, patch) in enumerate(arms):
            if name == "ref":
                E[ai, ci:ci + len(cs), :] = 0.0
                relA[ai, ci:ci + len(cs)] = 0.0
                continue
            delta = {v: np.zeros_like(base_inp_np[0][v]) for v in fc.PROG}
            for s in range(args.nsteps):
                inp, tgt, frc = base_inputs[s]
                pert = inp.copy(deep=False)
                if s > 0:
                    for v in fc.PROG:
                        pert[v] = inp[v] + delta[v]
                p, A = apply(pert, tgt * np.nan, frc, patch)
                dy = {v: np.asarray(p[v].data, np.float64) - base_pred[s][v]
                      for v in fc.PROG}
                E[ai, ci:ci + len(cs), s] = scorer(dy)
                if s == 0:
                    dA = np.asarray(A, np.float32) - base_A0
                    relA[ai, ci:ci + len(cs)] = (
                        np.linalg.norm(dA, axis=(0, 2)) /
                        np.linalg.norm(base_A0, axis=(0, 2)))
                for v in fc.PROG:
                    tpos = pdims[v].index("time")
                    dy0 = np.take(dy[v], 0, axis=tpos)
                    delta[v][:, 0] = delta[v][:, 1]
                    delta[v][:, 1] = np.transpose(dy0, roll_perm[v])
            msg = (f"windows {ci+len(cs)}/{args.nwin} arm {ai+1}/{len(arms)} "
                   f"({name}) {(time.time()-t0)/60:.1f}m")
            print("  " + msg, flush=True)
            (fc.ROOT / args.status).write_text(msg + "\n")
            np.save(fc.ROOT / args.out, dict(
                names=names, E=E, relA=relA, targets=targets,
                ctrl={int(t): np.asarray(ctrl[t]).tolist() for t in targets},
                ctrl_mode=ctrl_mode, nsteps=args.nsteps, nwin=args.nwin,
                starts=[str(s) for s in starts], mass=mass, rate=rate,
                prereg="docs/prereg/prereg_flagship_g2_suite.md §3"), allow_pickle=True)
    print(f"saved -> {args.out} ({(time.time()-t0)/60:.1f}m)", flush=True)

if __name__ == "__main__":
    main()
