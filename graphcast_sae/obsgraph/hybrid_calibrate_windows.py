"""Does the window count buy power? Consensus vs joint, n=13 vs n=52.

The question is whether the 2 h IC-offset battery (52 realizations) is worth
buying over the 13 storms already queued. That question has a different answer for the two
estimators the design could use, and the difference is arithmetic, not empirical:

  * `>=50% of windows` CONSENSUS. Detection probability is P(Bin(n, p) >= n/2) where p is the
    single-window recovery rate. If p < 0.5 this probability *falls* as n grows. More windows
    make a consensus rule STRICTER, not more powerful. n is not a lever here.
  * `analysis_mode='multiple'` JOINT fit (what the prereg actually specifies). Effective
    samples are n x (T - tau_max), so power rises with n. n IS a lever here.

So this script measures, per injected coupling strength beta:
  - p, the per-window recovery rate (52 real windows, exact (a,b,tau) link)
  - the consensus detection probability at n=13 and n=52, by direct subsampling of the
    measured hit vectors AND by the binomial, which agree by construction
  - joint-mode recovery at n=13 and n=52, over repeated random window subsets
  - the joint-mode FALSE-EDGE rate under a per-column circular-shift null that keeps each
    feature's own autocorrelation and destroys only the cross-feature lag structure. That is
    the control-must-be-able-to-fail rule leg (iii) for the joint estimator, and it is the leg the consensus rule was
    only ever calibrated on white noise.

CPU only, multiprocessing over windows.

Paper: Appendix app:null (consensus vs joint power, Shape 5 of app:taxonomy)
Inputs: none beyond the arguments above
Outputs: results/hybrid_calibrate_windows<tag>.json
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.obsgraph.hybrid_calibrate_windows --source mech
    python -m graphcast_sae.obsgraph.hybrid_calibrate_windows --source real --resdir results/skill/hyb_series
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import graphcast_sae.obsgraph.hybrid_pcmci as H                                          # noqa: E402

def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

def _one(job):
    X, a, b, tau, tau_max, pc_alpha = job
    r = H._pcmci_single(X, tau_max, pc_alpha)
    d = H.dets_from_graph(r["graph"])
    return int((a, b, tau) in d), len(d)

def circshift_null(X, rng):
    """Independent circular shift per column: keeps marginal + autocorrelation, kills cross-lag."""
    Y = np.empty_like(X)
    for j in range(X.shape[1]):
        Y[:, j] = np.roll(X[:, j], int(rng.integers(1, X.shape[0])))
    return Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mech")
    ap.add_argument("--resdir", default=None)
    ap.add_argument("--betas", default="0.2,0.4,0.6,0.8,1.0,1.5,2.0")
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--n-windows", type=int, default=52)
    ap.add_argument("--window-grid", default="13,52")
    ap.add_argument("--tau-max", type=int, default=2)
    ap.add_argument("--tau-inject", type=int, default=1)
    ap.add_argument("--pc-alpha", type=float, default=0.05)
    ap.add_argument("--n-select", type=int, default=20)
    ap.add_argument("--n-sub", type=int, default=400, help="subsample draws per window count")
    ap.add_argument("--n-joint-sub", type=int, default=12)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--kill-beta", type=float, default=0.8,
                    help="coupling strength at which the pre-registered kill "
                         "condition is evaluated")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from multiprocessing import Pool

    ids, W = H.load_windows(a.source, a.resdir)
    cols, _ = H.select_features(ids, W, a.n_select, verbose=False)
    med = np.array([np.median([w["X"][:, j].mean() for w in W]) for j in range(len(ids))])
    cols, rep = H.drop_collinear(ids, W, cols, 0.95, med)
    sel = [ids[j] for j in cols]
    mats_all = [H.zscore(w["X"][:, cols]) for w in W]
    N, T = len(cols), mats_all[0].shape[0]
    H.log(f"[cw] N={N} T={T} windows_available={len(mats_all)} tau_max={a.tau_max} "
          f"pc_alpha={a.pc_alpha}")
    H.log(f"[cw] features: {sel}")

    # per-window constant-column census -- ParCorr is undefined on a constant column
    A = np.array([w["X"][:, cols] for w in W])
    const = (A.std(1) < 1e-12)
    H.log(f"[cw] per-window constant (identically-zero) columns: med "
          f"{np.median(const.sum(1)):.1f} max {const.sum(1).max()} of {N}; "
          f"{int((const.sum(1) == 0).sum())}/{len(W)} windows have none")

    rng = np.random.default_rng(0)
    nw = min(a.n_windows, len(mats_all))
    widx = rng.choice(len(mats_all), nw, replace=False)
    mats = [mats_all[i] for i in widx]
    grid = [int(x) for x in a.window_grid.split(",") if int(x) <= nw]
    betas = [float(x) for x in a.betas.split(",")]

    pairs = []
    while len(pairs) < a.n_pairs:
        i, j = rng.choice(N, 2, replace=False)
        if (int(i), int(j)) not in pairs:
            pairs.append((int(i), int(j)))
    H.log(f"[cw] injected pairs (indices into the feature list): {pairs}")

    pool = Pool(a.nproc, initializer=_init)
    out = dict(features=sel, N=N, T=T, n_windows=nw, tau_max=a.tau_max,
               tau_inject=a.tau_inject, pc_alpha=a.pc_alpha, grid=grid, cond=rep, rows=[])

    # ---- NULL: joint-mode false edges under the circular-shift null ------------
    H.log("")
    H.log("[cw] JOINT-MODE NULL (per-column circular shift; autocorrelation kept, "
          "cross-lag destroyed)")
    for n in grid:
        cnts = []
        for rep_i in range(a.n_joint_sub):
            r2 = np.random.default_rng(1000 + rep_i)
            sub = [circshift_null(mats[k], r2) for k in r2.choice(nw, n, replace=False)]
            rj = H.pcmci_multiple(sub, a.tau_max, a.pc_alpha)
            cnts.append(len(H.dets_from_graph(rj["graph"])))
        H.log(f"[cw]   n={n:>3}: false directed lagged edges per joint fit "
              f"med={np.median(cnts):.1f} range=[{min(cnts)},{max(cnts)}] "
              f"of {N*(N-1)*a.tau_max} possible ({np.mean(cnts)/(N*(N-1)*a.tau_max)*100:.2f}%)")
        out[f"joint_null_n{n}"] = cnts
    # consensus null on real windows, no injection
    res = pool.map(_one, [(mats[k], -1, -1, 1, a.tau_max, a.pc_alpha) for k in range(nw)])
    per_win_edges = [c for _, c in res]
    H.log(f"[cw] CONSENSUS NULL (no injection, real windows): edges/window med "
          f"{np.median(per_win_edges):.1f} max {max(per_win_edges)}")
    out["null_edges_per_window"] = per_win_edges

    # ---- RECOVERY -------------------------------------------------------------
    H.log("")
    H.log("[cw] recovery of an injected X_b[t] += beta * X_a[t-tau] coupling")
    hdr = (f"  {'beta':>5} {'p(win)':>7} "
           + " ".join(f"{'cons@'+str(n):>9}" for n in grid)
           + " " + " ".join(f"{'joint@'+str(n):>10}" for n in grid))
    H.log(hdr)
    for beta in betas:
        p_list, cons = [], {n: [] for n in grid}
        joint = {n: [] for n in grid}
        for (i, j) in pairs:
            inj = []
            for X0 in mats:
                X = X0.copy()
                X[a.tau_inject:, j] += beta * X0[:-a.tau_inject, i]
                inj.append(X)
            r = pool.map(_one, [(X, i, j, a.tau_inject, a.tau_max, a.pc_alpha) for X in inj])
            hits = np.array([h for h, _ in r], bool)
            p_list.append(hits.mean())
            for n in grid:
                r3 = np.random.default_rng(7)
                draws = [hits[r3.choice(nw, n, replace=False)].sum() >= np.ceil(n / 2)
                         for _ in range(a.n_sub)]
                cons[n].append(float(np.mean(draws)))
                jh = []
                for rep_i in range(a.n_joint_sub):
                    r4 = np.random.default_rng(2000 + rep_i)
                    sub = [inj[k] for k in r4.choice(nw, n, replace=False)]
                    rj = H.pcmci_multiple(sub, a.tau_max, a.pc_alpha)
                    jh.append(int((i, j, a.tau_inject) in H.dets_from_graph(rj["graph"])))
                joint[n].append(float(np.mean(jh)))
        p = float(np.mean(p_list))
        line = (f"  {beta:>5.1f} {p:>7.3f} "
                + " ".join(f"{np.mean(cons[n]):>9.3f}" for n in grid)
                + " " + " ".join(f"{np.mean(joint[n]):>10.3f}" for n in grid))
        H.log(line)
        H.log(f"        per-pair p: {[round(x,3) for x in p_list]}")
        for n in grid:
            H.log(f"        n={n}: consensus P(detect) per pair {[round(x,2) for x in cons[n]]}"
                  f" | joint P(detect) per pair {[round(x,2) for x in joint[n]]}")
        out["rows"].append(dict(beta=beta, p_win=p, p_win_per_pair=p_list,
                                consensus={str(n): cons[n] for n in grid},
                                joint={str(n): joint[n] for n in grid}))
    pool.close()

    # ---- verdict --------------------------------------------------------------
    H.log("")
    # The pre-registered kill condition is stated AT a coupling strength ("if even a strong
    # injected coupling, beta >= KILL_BETA, cannot reach >=50% window recovery"). It is NOT
    # a max over the whole sweep: quoting the best beta tested would let an arbitrarily
    # strong injection rescue the rule and make the condition unfailable. So the verdict is
    # read off the row AT beta = KILL_BETA, and the rest of the sweep is context.
    kb = a.kill_beta
    row = min(out["rows"], key=lambda r: abs(r["beta"] - kb))
    if abs(row["beta"] - kb) > 1e-9:
        H.log(f"[cw] WARNING: beta={kb} not in the sweep; verdict read at the nearest "
              f"tested beta={row['beta']}")
    for n in grid:
        H.log(f"[cw] at beta={row['beta']}, n={n}: consensus P(detect)="
              f"{np.mean(row['consensus'][str(n)]):.3f}  "
              f"joint P(detect)={np.mean(row['joint'][str(n)]):.3f}")
    npass = sum(1 for x in row["p_win_per_pair"] if x >= 0.5)
    H.log(f"[cw] at beta={row['beta']}: mean per-window recovery p={row['p_win']:.3f}; "
          f"{npass}/{len(row['p_win_per_pair'])} injected pairs individually reach p>=0.5")
    first50 = next((r["beta"] for r in out["rows"] if r["p_win"] >= 0.5), None)
    H.log(f"[cw] mean per-window recovery first reaches 0.5 at beta={first50} "
          f"(None = never, over the tested sweep "
          f"{[r['beta'] for r in out['rows']]})")
    kill = row["p_win"] < 0.5
    out["kill"] = bool(kill)
    out["kill_beta"] = kb
    out["beta_p_first_50"] = first50
    H.log(f"[cw] KILL for the >=50%-consensus specification, judged at beta={row['beta']}: "
          f"{'YES' if kill else 'NO'}")
    p = os.path.join(H.OUTDIR, f"hybrid_calibrate_windows{a.tag}.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    H.log(f"[cw] -> {p}")

if __name__ == "__main__":
    main()
