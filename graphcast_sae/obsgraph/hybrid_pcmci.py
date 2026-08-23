"""Hybrid design: PCMCI+ proposes feature-to-feature edges, intervention disposes.

CPU only. Implements `docs/prereg/prereg_hybrid_hurricane.md` (frozen 2026-08-20).
Nothing here touches the GPU and nothing here writes into a file another lane owns; every
output goes to results/hybrid_*.

Development fixture (no GPU dependency):

Real data (13 windows = 8 NH + 5 SH, MECH_ARMS=baseline MECH_TRACK=all):

    R=results/skill/hyb_series,results/skill/hyb_series_sh

WINDOWS AND WHERE THEY COME FROM.
The prereg's 52 realizations (13 storms x 4 IC offsets, MECH_TRACK=all, 4096 features) do
not exist yet -- the only 4096-tracked run on disk is results/skill/xt_probe (1 storm,
H=12). Until the real extraction lands, `--source mech` uses the ten 43-feature mech_*
batteries, whose `baseline` arms are ten independent unperturbed rollouts of each of the 8
NH storms (they differ; max |diff| 0.9-5.3 in in-box units, i.e. GPU nondeterminism gives a
free 10-member ensemble). That is 80 real (16, 43) in-box activation matrices with the
right autocorrelation, the right zero-inflation and the right cross-feature covariance --
exactly what a power calibration needs. `--source real --resdir a,b` reads the real thing
the moment it exists and every subcommand switches over unchanged.

WHAT THE CALIBRATION FOUND (see hybrid_calibrate_windows.py, results/hybrid_calibrate*.json).
The `>=50% of windows` consensus rule is DEAD at T=16 and must not be used: a single window
of 16 steps recovers a *known injected* lag-1 coupling at rate p=0.07 (beta=0.2) to 0.33
(beta=1.5), never above 0.5, so P(Bin(n,p) >= n/2) is ~0 and FALLS as n grows. On the real
uninjected series `graph --mode consensus` returns literally zero edges from 80 windows.
The prereg's own `analysis_mode='multiple'` joint fit is the estimator that works: it
recovers beta=0.4 couplings at 0.87 with 13 windows and 1.00 with 52.

GUARDRAIL #6 is enforced in one place, `load_windows`: finiteness, the expected H, all-zero
column removal, and a printed record of what was dropped. No subcommand may bypass it.

Paper: Appendix app:null (PCMCI+ proposes, intervention disposes)
Inputs: results/skill (shipped); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/hybrid_footprint_fires.npz (footprint cache), results/hybrid_calibrate.json, results/hybrid_pairs.json and the per-subcommand results/hybrid_*.json reports
Run:   # JAX env, CPU
    python -m graphcast_sae.obsgraph.hybrid_pcmci select     --source mech
    python -m graphcast_sae.obsgraph.hybrid_pcmci condition  --source mech
    python -m graphcast_sae.obsgraph.hybrid_pcmci graph      --source mech --mode both
    python -m graphcast_sae.obsgraph.hybrid_pcmci nullgraph  --source mech --n-null 10
    python -m graphcast_sae.obsgraph.hybrid_pcmci pairs      --source mech --mode multiple
    python -m graphcast_sae.obsgraph.hybrid_pcmci arms       --source mech
    python -m graphcast_sae.obsgraph.hybrid_pcmci select    --source real --resdir $R
    python -m graphcast_sae.obsgraph.hybrid_pcmci graph     --source real --resdir $R --mode multiple
    python -m graphcast_sae.obsgraph.hybrid_pcmci nullgraph --source real --resdir $R
    python -m graphcast_sae.obsgraph.hybrid_pcmci pairs     --source real --resdir $R --mode multiple
    python -m graphcast_sae.obsgraph.hybrid_pcmci arms      --source real --resdir $R
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

from graphcast_sae.paths import REPO_ROOT, SCRATCH as _SCRATCH
ROOT = str(REPO_ROOT)
OUTDIR = os.path.join(ROOT, "results")
SCRATCH = str(_SCRATCH)

TC_READOUT = 3243          # excluded from the pair set: skill_conv_run.py asserts against it
CONV = [2401, 2067, 3174]  # force-included positive control
NH_STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013",
             "patricia2015", "wilma2005", "nondev2013"]
H_EXPECT = 16
N_SELECT = 20

MECH_DIRS = ["mech_ascent", "mech_atm_river", "mech_baroclinicity", "mech_blocking",
             "mech_jet250", "mech_q600", "mech_shear", "mech_t850", "mech_vort850",
             "mech_z500"]

def log(*a):
    print(*a, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# data layer + the data-gate rule
# ─────────────────────────────────────────────────────────────────────────────
def _read_run(path, arm="baseline"):
    r = np.load(path, allow_pickle=True).item()
    bf = r["res"][arm]["box_feats"]
    ids = sorted(int(k) for k in bf)
    X = np.stack([np.asarray(bf[i], float) for i in ids], axis=1)   # (H, N)
    return r["name"], ids, X, r

def load_windows(source="mech", resdir=None, h_expect=H_EXPECT, arm="baseline",
                 storms=None, verbose=True):
    """Return (ids, windows) where windows is a list of dicts {label, X:(H,N)}.

    Guardrail #6: assert finiteness, assert H, drop columns that are identically zero in
    EVERY window, and print what was dropped.
    """
    raw = []
    if source == "mech":
        for d in MECH_DIRS:
            for f in sorted(glob.glob(os.path.join(ROOT, "results/skill", d, "run_*.npy"))):
                name, ids, X, _ = _read_run(f, arm)
                raw.append((f"{d}/{name}", name, ids, X))
    elif source == "real":
        if resdir is None:
            raise SystemExit("--source real needs --resdir (comma-separated dirs allowed)")
        for one in str(resdir).split(","):
            rd = one if os.path.isabs(one) else os.path.join(ROOT, one)
            got = sorted(glob.glob(os.path.join(rd, "run_*.npy")))
            if not got:
                log(f"  WARN: no run_*.npy in {rd}")
            for f in got:
                name, ids, X, _ = _read_run(f, arm)
                raw.append((f"{os.path.basename(rd.rstrip('/'))}/{name}", name, ids, X))
    else:
        raise SystemExit(f"unknown source {source}")
    if not raw:
        raise SystemExit(f"no runs found for source={source} resdir={resdir}")

    if storms is not None:
        raw = [t for t in raw if t[1] in storms]

    ids0 = raw[0][2]
    bad_ids = [lab for lab, _, ids, _ in raw if ids != ids0]
    if bad_ids:
        raise SystemExit(f"feature-id mismatch across windows: {bad_ids[:3]}")

    n_bad_h = [lab for lab, _, _, X in raw if X.shape[0] != h_expect]
    if n_bad_h:
        raise SystemExit(f"H != {h_expect} in {len(n_bad_h)} windows, e.g. {n_bad_h[:3]}")
    n_bad_f = [lab for lab, _, _, X in raw if not np.isfinite(X).all()]
    if n_bad_f:
        raise SystemExit(f"non-finite values in {len(n_bad_f)} windows, e.g. {n_bad_f[:3]}")
    if (np.asarray([X for _, _, _, X in raw]) < 0).any():
        log("  WARN: negative activations present (SAE codes should be >= 0)")

    A = np.asarray([X for _, _, _, X in raw])                    # (W, H, N)
    nz_any = (A != 0).any(axis=(0, 1))
    dropped = [ids0[i] for i in np.where(~nz_any)[0]]
    keep = np.where(nz_any)[0]
    ids = [ids0[i] for i in keep]
    windows = [dict(label=lab, storm=name, X=X[:, keep]) for lab, name, _, X in raw]

    if verbose:
        log(f"[gate] source={source} resdir={resdir} arm={arm}")
        log(f"[gate] windows={len(windows)}  H={h_expect} (asserted)  finite=OK")
        log(f"[gate] tracked features={len(ids0)}  all-zero-everywhere dropped="
            f"{len(dropped)}  kept={len(ids)}")
        if dropped:
            log(f"[gate] dropped ids (first 40): {dropped[:40]}")
        per_win_zero = [int((X == 0).all(axis=0).sum()) for _, _, _, X in raw]
        log(f"[gate] per-window all-zero columns: min={min(per_win_zero)} "
            f"med={int(np.median(per_win_zero))} max={max(per_win_zero)} "
            f"(of {len(ids0)})")
        log(f"[gate] storms present: {sorted(set(w['storm'] for w in windows))}")
    return ids, windows

def zscore(X, eps=1e-9):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    return (X - m) / np.maximum(s, eps)

# ─────────────────────────────────────────────────────────────────────────────
# PCMCI+ wrappers
# ─────────────────────────────────────────────────────────────────────────────
def _pcmci_single(series, tau_max, pc_alpha):
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    pc = PCMCI(dataframe=DataFrame(series), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    return res

def dets_from_graph(g):
    N = g.shape[0]
    return [(c, e, tau) for c in range(N) for e in range(N) if c != e
            for tau in range(1, g.shape[2]) if g[c, e, tau] == "-->"]

def pcmci_per_window(mats, tau_max, pc_alpha, nproc=1):
    """Run PCMCI+ independently on each (T,N) matrix. Returns list of (dets, val_matrix)."""
    out = []
    for X in mats:
        try:
            r = _pcmci_single(X, tau_max, pc_alpha)
            out.append((dets_from_graph(r["graph"]), r["val_matrix"]))
        except Exception as e:                                   # noqa: BLE001
            out.append((None, f"{type(e).__name__}: {e}"))
    return out

def pcmci_multiple(mats, tau_max, pc_alpha, ref_drop=None):
    """Prereg mode: analysis_mode='multiple', one joint fit over all realizations."""
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    data = {k: np.asarray(X, float) for k, X in enumerate(mats)}
    ref_drop = tau_max if ref_drop is None else ref_drop
    T = mats[0].shape[0]
    reference_points = np.arange(ref_drop, T)
    df = DataFrame(data, analysis_mode="multiple", reference_points=reference_points)
    pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    return res

def consensus(dets_per_win, N, frac=0.5):
    lag_cnt, pair_cnt = {}, np.zeros((N, N))
    n = len(dets_per_win)
    for det in dets_per_win:
        for (c, e) in {(c, e) for (c, e, _) in det}:
            pair_cnt[c, e] += 1
        for k in det:
            lag_cnt[k] = lag_cnt.get(k, 0) + 1
    thr = frac * n
    lag_edges = sorted([k for k, v in lag_cnt.items() if v >= thr])
    pair_edges = [(c, e) for c in range(N) for e in range(N)
                  if c != e and pair_cnt[c, e] >= thr]
    return lag_edges, pair_edges, lag_cnt, pair_cnt

# ─────────────────────────────────────────────────────────────────────────────
# selection (prereg)
# ─────────────────────────────────────────────────────────────────────────────
def in_box_stats(ids, windows):
    """Per-storm in-box mean activation, averaged over that storm's windows."""
    storms = sorted(set(w["storm"] for w in windows))
    M = np.zeros((len(storms), len(ids)))
    for si, s in enumerate(storms):
        ws = [w["X"] for w in windows if w["storm"] == s]
        M[si] = np.mean([X.mean(0) for X in ws], axis=0)
    return storms, M                                              # (S, N)

def select_features(ids, windows, n_select=N_SELECT, verbose=True):
    storms, M = in_box_stats(ids, windows)
    nh = [i for i, s in enumerate(storms) if s in NH_STORMS]
    if not nh:
        raise SystemExit(
            f"selection gate needs the 8 NH storms; none of {storms} is in "
            f"skill_conv_storms.STORMS ({NH_STORMS}). Point --resdir at the NH battery.")
    Mnh = M[nh]
    n_fire = (Mnh > 0).sum(0)
    med = np.median(Mnh, axis=0)

    # A2.3 (amendment 1) + A6.1: a feature must be NON-CONSTANT IN EVERY WINDOW USED, because
    # ParCorr is undefined on a flat column and tigramite warns rather than failing. This is
    # enforced here, not reported and stepped over -- and it is what forces the positive control
    # down to 2067 alone: at 52 windows 2401 is flat in 6 and 3174 in 4, so neither can be a
    # node. Resolution 1 in A6.1 was chosen for that reason, on the criteria written down before
    # the graph existed, and force-including them anyway would quietly undo it.
    A_all = np.stack([w["X"] for w in windows])                   # (W, H, N)
    nflat_all = (A_all.std(1) < 1e-12).sum(0)                     # windows where each col is flat
    nonconst = nflat_all == 0

    elig = np.array([(n_fire[j] >= 6) and (ids[j] != TC_READOUT) and bool(nonconst[j])
                     for j in range(len(ids))])
    order = [j for j in np.argsort(-med) if elig[j]]
    chosen = list(order[:n_select])
    forced = [ids.index(f) for f in CONV if f in ids and nonconst[ids.index(f)]]
    dropped_forced = [f for f in CONV if f in ids and not nonconst[ids.index(f)]]
    missing_forced = [f for f in CONV if f not in ids]
    if verbose and dropped_forced:
        log(f"[select] A2.3 BARS forced convection features {dropped_forced} "
            f"(flat in {[int(nflat_all[ids.index(f)]) for f in dropped_forced]} of "
            f"{len(windows)} windows). Positive control B5 is now "
            f"{[f for f in CONV if f in ids and nonconst[ids.index(f)]]} only -- "
            f"a B5 failure must be read as INSTRUMENT UNDERPOWERED, per A6.1.")
    for j in forced:
        if j not in chosen:
            # drop the lowest-ranked non-forced member to hold N
            for k in reversed(chosen):
                if k not in forced:
                    chosen.remove(k)
                    break
            chosen.append(j)
    chosen = sorted(chosen, key=lambda j: -med[j])

    if verbose:
        log(f"[select] NH storms used: {[storms[i] for i in nh]}")
        log(f"[select] eligible (in-box mean>0 in >=6 of {len(nh)} NH storms, "
            f"3243 excluded): {int(elig.sum())} of {len(ids)}")
        if missing_forced:
            log(f"[select] WARNING forced convection features not tracked: {missing_forced}")
        A = np.asarray([w["X"] for w in windows])                  # (W,H,Nall)
        nconst = (A.std(1) < 1e-12).sum(0)                         # windows where col is flat
        log(f"[select] chosen {len(chosen)}:")
        log(f"  {'rank':>4} {'feat':>6} {'med_inbox':>10} {'mean_inbox':>10} "
            f"{'n_fire/NH':>10} {'max_inbox':>10} {'n_flat_win':>11}  note")
        for r, j in enumerate(chosen):
            note = "FORCED(conv)" if ids[j] in CONV else ""
            log(f"  {r:>4} {ids[j]:>6} {med[j]:>10.4f} {Mnh[:, j].mean():>10.4f} "
                f"{n_fire[j]:>10d} {Mnh[:, j].max():>10.4f} "
                f"{nconst[j]:>5d}/{len(windows):<5d} {note}")
        cc = (A[:, :, chosen].std(1) < 1e-12).sum(1)
        log(f"[select] windows in which >=1 chosen feature is identically flat "
            f"(ParCorr undefined): {int((cc > 0).sum())}/{len(windows)}; "
            f"median flat cols per window {np.median(cc):.1f} of {len(chosen)}")
    return chosen, dict(storms=storms, M=M, med=med, n_fire=n_fire, ids=ids)

# ─────────────────────────────────────────────────────────────────────────────
# conditioning (the conditioning rule)
# ─────────────────────────────────────────────────────────────────────────────
def condition_report(ids, windows, cols, tag=""):
    S = np.concatenate([w["X"][:, cols] for w in windows], axis=0)   # (W*H, n)
    Sz = zscore(S)
    C = np.corrcoef(Sz, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    off = C - np.eye(len(cols))
    k = np.unravel_index(np.argmax(np.abs(off)), off.shape)
    cond = np.linalg.cond(C)
    log(f"[cond{tag}] n={len(cols)} samples={S.shape[0]}")
    log(f"[cond{tag}] cond={cond:.2f}  min_eigenvalue={ev.min():.6f}  "
        f"max|corr|={abs(off[k]):.4f} between f{ids[cols[k[0]]]} and f{ids[cols[k[1]]]}")
    return dict(cond=float(cond), min_eig=float(ev.min()), max_abs_corr=float(abs(off[k])),
                pair=(ids[cols[k[0]]], ids[cols[k[1]]])), C

def drop_collinear(ids, windows, cols, thresh=0.95, med=None):
    """Guardrail #5 repair: while max|corr| > thresh, drop the weaker-firing member."""
    cols = list(cols)
    while True:
        rep, C = condition_report(ids, windows, cols, tag="")
        if rep["max_abs_corr"] <= thresh:
            return cols, rep
        a, b = rep["pair"]
        ja, jb = ids.index(a), ids.index(b)
        drop = a if (med is not None and med[ja] < med[jb]) else b
        log(f"[cond] max|corr|={rep['max_abs_corr']:.4f} > {thresh}: dropping f{drop}")
        cols.remove(ids.index(drop))

# ─────────────────────────────────────────────────────────────────────────────
# footprints (repo definition, footprint_inspect.py: 8 windows, fires in >=2 of 8)
# ─────────────────────────────────────────────────────────────────────────────
FP_CACHE = os.path.join(OUTDIR, "hybrid_footprint_fires.npz")
FP_NW = 8
FP_THRESH = 2                # max(1, NW//4) == 2, exactly footprint_inspect.py

def footprint_fires(force=False):
    """(L, 4096) bool: feature fires at that mesh node in >= 2 of 8 IID windows."""
    if os.path.exists(FP_CACHE) and not force:
        z = np.load(FP_CACHE)
        return z["fires"], int(z["nw"]), int(z["thresh"])
    meta = json.load(open(os.path.join(SCRATCH, "fs_iid_meta.json")))
    L, NWtot = meta["n_mesh"], meta["n_windows"]
    zw = np.load(os.path.join(ROOT, "graphcast_sae/weights/sae_k32_lat4096_lay08.npz"))
    Wenc, bpre = zw["W_enc"], zw["b_pre"]
    X = np.load(os.path.join(SCRATCH, "fs_iid_dump.npy"), mmap_mode="r")
    idxs = np.linspace(0, NWtot - 1, FP_NW).astype(int)
    cnt = np.zeros((L, 4096), np.int16)
    for wi, j in enumerate(idxs):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        top = np.argpartition(-pre, 32, axis=1)[:, :32]
        r = np.arange(len(A))[:, None]
        act = np.zeros_like(pre, dtype=bool)
        act[r, top] = pre[r, top] > 0
        cnt += act
        log(f"  [footprint] window {wi+1}/{FP_NW}")
    fires = cnt >= FP_THRESH
    np.savez_compressed(FP_CACHE, fires=fires, nw=FP_NW, thresh=FP_THRESH, idxs=idxs)
    log(f"  [footprint] -> {FP_CACHE}")
    return fires, FP_NW, FP_THRESH

def footprint_cos(fires, a, b):
    fa, fb = fires[:, a].astype(np.float64), fires[:, b].astype(np.float64)
    na, nb = np.sqrt(fa.sum()), np.sqrt(fb.sum())
    if na == 0 or nb == 0:
        return np.nan
    return float(fa @ fb / (na * nb))

# ─────────────────────────────────────────────────────────────────────────────
# marginal lagged correlation
# ─────────────────────────────────────────────────────────────────────────────
def lag_corr(windows, cols, tau):
    """Pooled Pearson r between col a at t-tau and col b at t, over all windows."""
    n = len(cols)
    xs, ys = [], []
    for w in windows:
        Z = zscore(w["X"][:, cols])
        xs.append(Z[:-tau]); ys.append(Z[tau:])
    Xp = np.concatenate(xs, 0); Yp = np.concatenate(ys, 0)
    Xp = (Xp - Xp.mean(0)) / np.maximum(Xp.std(0), 1e-9)
    Yp = (Yp - Yp.mean(0)) / np.maximum(Yp.std(0), 1e-9)
    return (Xp.T @ Yp) / len(Xp)                                   # R[a,b] = corr(a_{t-tau}, b_t)

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATE — the step that can kill the design (the control-must-be-able-to-fail rule leg ii)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_calibrate(args):
    t0 = time.time()
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h)
    cols, _ = select_features(ids, windows, args.n_select, verbose=False)
    med = np.array([np.median([w["X"][:, j].mean() for w in windows]) for j in range(len(ids))])
    cols, rep = drop_collinear(ids, windows, cols, thresh=0.95, med=med)
    sel_ids = [ids[j] for j in cols]
    log(f"[calibrate] N={len(cols)} features, {len(windows)} windows available, "
        f"T={windows[0]['X'].shape[0]}")
    log(f"[calibrate] features: {sel_ids}")

    mats_all = [zscore(w["X"][:, cols]) for w in windows]
    rng = np.random.default_rng(args.seed)

    # --- timing probe first (time one run before launching many) ---
    tt = time.time()
    _ = _pcmci_single(mats_all[0], args.tau_max, args.pc_alpha)
    per_run = time.time() - tt
    log(f"[calibrate] one PCMCI+ run: {per_run:.2f}s "
        f"(N={len(cols)}, T={mats_all[0].shape[0]}, tau_max={args.tau_max})")

    # --- leg (iii)-adjacent: false-edge rate on the REAL data with no injection ---
    n_null_win = min(args.n_windows_small * 2, len(mats_all))
    null_idx = rng.choice(len(mats_all), n_null_win, replace=False)
    null_dets = []
    for i in null_idx:
        r = _pcmci_single(mats_all[i], args.tau_max, args.pc_alpha)
        null_dets.append(dets_from_graph(r["graph"]))
    n_per = [len(d) for d in null_dets]
    N = len(cols)
    n_possible = N * (N - 1) * args.tau_max
    log(f"[calibrate] NO-INJECTION false edges per window over {n_null_win} real windows: "
        f"min={min(n_per)} med={np.median(n_per):.1f} max={max(n_per)} "
        f"(of {n_possible} possible directed lagged links; "
        f"rate={np.mean(n_per)/n_possible*100:.2f}%)")
    le8, _, lag_cnt8, _ = consensus(null_dets[:args.n_windows_small], N, 0.5)
    le_all, _, lag_cnt_all, _ = consensus(null_dets, N, 0.5)
    log(f"[calibrate] NO-INJECTION consensus>=50%: {len(le8)} edges over "
        f"{args.n_windows_small} windows, {len(le_all)} edges over {n_null_win} windows")
    if lag_cnt_all:
        top = sorted(lag_cnt_all.items(), key=lambda kv: -kv[1])[:5]
        log(f"[calibrate] most-repeated no-injection link counts (of {n_null_win}): "
            f"{[(f'f{sel_ids[c]}->f{sel_ids[e]}@{t}', int(v)) for (c, e, t), v in top]}")

    # --- recovery of an INJECTED coupling ---
    betas = [float(b) for b in args.betas.split(",")]
    nwin_grid = sorted(set([args.n_windows_small, min(args.n_windows_big, len(mats_all))]))
    log("")
    log("[calibrate] injected-coupling recovery: X_b[t] += beta * X_a[t-tau] on "
        "z-scored series, then PCMCI+; 'recovery' = fraction of windows in which the exact "
        "(a,b,tau) directed link is returned.")
    log(f"[calibrate] {args.n_pairs} random (a,b) pairs x {max(nwin_grid)} windows x "
        f"{len(betas)} betas, tau_inject={args.tau_inject}, tau_max={args.tau_max}, "
        f"pc_alpha={args.pc_alpha}")

    pairs = []
    while len(pairs) < args.n_pairs:
        a, b = rng.choice(N, 2, replace=False)
        if (int(a), int(b)) not in pairs:
            pairs.append((int(a), int(b)))

    big = max(nwin_grid)
    widx = rng.choice(len(mats_all), big, replace=False)
    log("")
    rows = []
    for beta in betas:
        per_pair_rec = []
        per_pair_joint = []
        extra_counts = []
        for (a, b) in pairs:
            hits = 0
            inj = []
            for i in widx:
                X = mats_all[i].copy()
                X[args.tau_inject:, b] += beta * mats_all[i][:-args.tau_inject, a]
                inj.append(X)
            for X in inj:
                r = _pcmci_single(X, args.tau_max, args.pc_alpha)
                d = dets_from_graph(r["graph"])
                hits += int((a, b, args.tau_inject) in d)
                extra_counts.append(len(d))
            per_pair_rec.append(hits / big)
            if args.joint:
                try:
                    rj = pcmci_multiple(inj, args.tau_max, args.pc_alpha)
                    per_pair_joint.append(
                        int((a, b, args.tau_inject) in dets_from_graph(rj["graph"])))
                except Exception as e:                            # noqa: BLE001
                    per_pair_joint.append(-1)
                    log(f"    joint failed b={beta}: {type(e).__name__}: {e}")
        rec = np.array(per_pair_rec)
        row = dict(beta=beta, rec_mean=float(rec.mean()), rec_min=float(rec.min()),
                   rec_max=float(rec.max()),
                   n_pairs_over_50=int((rec >= 0.5).sum()), n_pairs=len(pairs),
                   per_pair=[float(x) for x in rec],
                   joint=[int(x) for x in per_pair_joint],
                   extra_med=float(np.median(extra_counts)))
        rows.append(row)
        jt = (f"{sum(1 for x in per_pair_joint if x == 1)}/{len(per_pair_joint)}"
              if args.joint else "-")
        log(f"  beta={beta:>4}  recovery mean={rec.mean():.3f} "
            f"[min {rec.min():.3f}, max {rec.max():.3f}]  "
            f"pairs with recovery>=50%: {int((rec>=0.5).sum())}/{len(pairs)}  "
            f"joint(all {big} windows) hits: {jt}  "
            f"median total edges/window={np.median(extra_counts):.0f}")

    # verdict
    log("")
    strong = [r for r in rows if r["beta"] >= 0.8]
    best = max(rows, key=lambda r: r["rec_mean"])
    thr_beta = next((r["beta"] for r in rows if r["rec_mean"] >= 0.5), None)
    log(f"[calibrate] beta at which mean per-window recovery first reaches 50%: {thr_beta}")
    if strong:
        s = max(strong, key=lambda r: r["rec_mean"])
        log(f"[calibrate] strongest tested coupling beta={s['beta']}: "
            f"mean recovery {s['rec_mean']:.3f}, "
            f"{s['n_pairs_over_50']}/{s['n_pairs']} pairs clear the >=50% consensus bar")
        kill = s["rec_mean"] < 0.5
    else:
        kill = best["rec_mean"] < 0.5
    log(f"[calibrate] VERDICT: {'KILL — consensus bar sits above the ceiling of a strong real coupling' if kill else 'NO KILL — a strong coupling clears the >=50% consensus bar'}")
    out = dict(rows=rows, sel_ids=sel_ids, n_windows=big, tau_inject=args.tau_inject,
               tau_max=args.tau_max, pc_alpha=args.pc_alpha, per_run_seconds=per_run,
               null_edges_per_window=n_per, null_consensus_small=len(le8),
               null_consensus_all=len(le_all), n_possible_links=n_possible,
               source=args.source, resdir=args.resdir, kill=bool(kill),
               beta_at_50=thr_beta, cond=rep, seconds=time.time() - t0)
    p = os.path.join(OUTDIR, "hybrid_calibrate.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"[calibrate] -> {p}   ({time.time()-t0:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
def cmd_select(args):
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h)
    cols, info = select_features(ids, windows, args.n_select)
    json.dump(dict(features=[ids[j] for j in cols], storms=info["storms"],
                   source=args.source, resdir=args.resdir),
              open(os.path.join(OUTDIR, "hybrid_select.json"), "w"), indent=1)
    log(f"[select] -> {os.path.join(OUTDIR, 'hybrid_select.json')}")

def cmd_condition(args):
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h)
    cols, _ = select_features(ids, windows, args.n_select, verbose=False)
    med = np.array([np.median([w["X"][:, j].mean() for w in windows]) for j in range(len(ids))])
    log("[cond] BEFORE any drop:")
    cols2, rep = drop_collinear(ids, windows, cols, thresh=args.corr_thresh, med=med)
    log(f"[cond] final N={len(cols2)}: {[ids[j] for j in cols2]}")
    json.dump(dict(features=[ids[j] for j in cols2], **rep),
              open(os.path.join(OUTDIR, "hybrid_condition.json"), "w"), indent=1, default=float)

def cmd_graph(args):
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h)
    cols, _ = select_features(ids, windows, args.n_select, verbose=False)
    med = np.array([np.median([w["X"][:, j].mean() for w in windows]) for j in range(len(ids))])
    cols, rep = drop_collinear(ids, windows, cols, thresh=args.corr_thresh, med=med)
    sel_ids = [ids[j] for j in cols]
    mats = [zscore(w["X"][:, cols]) for w in windows]
    N = len(cols)
    T = mats[0].shape[0]
    log(f"[graph] N={N} windows={len(mats)} T={T} tau_max={args.tau_max} "
        f"pc_alpha={args.pc_alpha}")
    eff = len(mats) * (T - args.tau_max)
    log(f"[graph] effective samples (joint) = {len(mats)}x({T}-{args.tau_max}) = {eff}; "
        f"N*tau_max = {N*args.tau_max}; ratio = {eff/(N*args.tau_max):.2f} "
        f"(prereg floor 5)  |  per-window samples = {T-args.tau_max}, "
        f"ratio = {(T-args.tau_max)/(N*args.tau_max):.2f}")
    out = dict(features=sel_ids, cond=rep, tau_max=args.tau_max, pc_alpha=args.pc_alpha,
               n_windows=len(mats), ratio_joint=eff / (N * args.tau_max))

    if args.mode in ("consensus", "both"):
        dets, vals = [], []
        for X in mats:
            r = _pcmci_single(X, args.tau_max, args.pc_alpha)
            dets.append(dets_from_graph(r["graph"]))
            vals.append(np.abs(r["val_matrix"]))
        lag_edges, pair_edges, lag_cnt, _ = consensus(dets, N, args.cons_frac)
        V = np.mean(vals, 0)
        rows = sorted([(sel_ids[c], sel_ids[e], t, float(V[c, e, t]), int(lag_cnt[(c, e, t)]))
                       for (c, e, t) in lag_edges], key=lambda r: -r[3])
        log(f"[graph:consensus] edges at >={args.cons_frac:.0%} of {len(mats)} windows: "
            f"{len(rows)}")
        for a, b, t, v, n in rows:
            log(f"    f{a} -> f{b} @ tau={t}   |MCI|={v:.3f}   in {n}/{len(mats)} windows")
        out["consensus_edges"] = [dict(a=a, b=b, tau=t, mci=v, n_win=n) for a, b, t, v, n in rows]
        out["edges_per_window"] = [len(d) for d in dets]
        log(f"[graph:consensus] per-window edge count: med="
            f"{np.median([len(d) for d in dets]):.0f} max={max(len(d) for d in dets)}")

    if args.mode in ("multiple", "both"):
        t0 = time.time()
        r = pcmci_multiple(mats, args.tau_max, args.pc_alpha)
        d = dets_from_graph(r["graph"])
        V = np.abs(r["val_matrix"])
        rows = sorted([(sel_ids[c], sel_ids[e], t, float(V[c, e, t])) for (c, e, t) in d],
                      key=lambda x: -x[3])
        log(f"[graph:multiple] (prereg mode) {len(rows)} directed lagged edges "
            f"in {time.time()-t0:.0f}s")
        for a, b, t, v in rows[:40]:
            log(f"    f{a} -> f{b} @ tau={t}   |MCI|={v:.3f}")
        out["multiple_edges"] = [dict(a=a, b=b, tau=t, mci=v) for a, b, t, v in rows]

    p = os.path.join(OUTDIR, "hybrid_graph.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"[graph] -> {p}")

def cmd_nullgraph(args):
    """Guardrail #9 for the JOINT estimator: how many edges, and at what |MCI|, does
    analysis_mode='multiple' invent when only the cross-feature lag structure is destroyed?

    Null: independent circular shift of every column within every window. Each feature keeps
    its own marginal and its own autocorrelation; only the between-feature lag relations die.
    A shuffle that also destroyed autocorrelation would be a vacuous control here, because
    PCMCI+'s false positives on short series come precisely from autocorrelation.
    """
    rng = np.random.default_rng(args.seed)
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h)
    cols, _ = select_features(ids, windows, args.n_select, verbose=False)
    med = np.array([np.median([w["X"][:, j].mean() for w in windows]) for j in range(len(ids))])
    cols, rep = drop_collinear(ids, windows, cols, args.corr_thresh, med)
    mats = [zscore(w["X"][:, cols]) for w in windows]
    N = len(cols)
    n_poss = N * (N - 1) * args.tau_max

    r = pcmci_multiple(mats, args.tau_max, args.pc_alpha)
    V = np.abs(r["val_matrix"])
    real = sorted([float(V[c, e, t]) for c, e, t in dets_from_graph(r["graph"])], reverse=True)
    log(f"[nullgraph] REAL joint fit: {len(real)} directed lagged edges of {n_poss}; "
        f"max|MCI|={real[0]:.3f}; 10th largest={real[9]:.3f}" if len(real) > 9 else
        f"[nullgraph] REAL joint fit: {len(real)} edges; max|MCI|="
        f"{real[0] if real else float('nan'):.3f}")
    cnts, maxes, tenths = [], [], []
    for s in range(args.n_null):
        sub = []
        for X in mats:
            Y = np.empty_like(X)
            for j in range(X.shape[1]):
                Y[:, j] = np.roll(X[:, j], int(rng.integers(1, X.shape[0])))
            sub.append(Y)
        rj = pcmci_multiple(sub, args.tau_max, args.pc_alpha)
        Vn = np.abs(rj["val_matrix"])
        v = sorted([float(Vn[c, e, t]) for c, e, t in dets_from_graph(rj["graph"])],
                   reverse=True)
        cnts.append(len(v))
        maxes.append(v[0] if v else 0.0)
        tenths.append(v[9] if len(v) > 9 else float("nan"))
    log(f"[nullgraph] NULL over {args.n_null} circular-shift draws: edges/fit med "
        f"{np.median(cnts):.1f} range [{min(cnts)},{max(cnts)}] "
        f"({np.mean(cnts)/n_poss*100:.2f}% of {n_poss}, nominal alpha "
        f"{args.pc_alpha*100:.0f}%)")
    log(f"[nullgraph] NULL max|MCI| per draw: med {np.median(maxes):.3f} "
        f"max {max(maxes):.3f}")
    if len(real) > 9:
        exceed = sum(1 for m in maxes if m >= real[9])
        log(f"[nullgraph] the null's single largest |MCI| reaches the real top-10 floor "
            f"({real[9]:.3f}) in {exceed}/{args.n_null} draws  -> "
            f"{'BAR NOT SEPARATED' if exceed > args.n_null*0.05 else 'separated'}")
    json.dump(dict(real_mci=real, null_counts=cnts, null_max=maxes, null_tenth=tenths,
                   n_possible=n_poss, cond=rep),
              open(os.path.join(OUTDIR, "hybrid_nullgraph.json"), "w"), indent=1, default=float)
    log(f"[nullgraph] -> {os.path.join(OUTDIR, 'hybrid_nullgraph.json')}")

def cmd_pairs(args):
    g = json.load(open(os.path.join(OUTDIR, "hybrid_graph.json")))
    key = "consensus_edges" if args.mode == "consensus" else "multiple_edges"
    edges = g.get(key) or []
    if not edges:
        raise SystemExit(f"no {key} in hybrid_graph.json — run `graph --mode {args.mode}` first")
    edges = edges[:args.top_k]
    ids, windows = load_windows(args.source, args.resdir, h_expect=args.h, verbose=False)
    sel = g["features"]
    cols = [ids.index(f) for f in sel]
    fires, nw, thr = footprint_fires()
    log(f"[pairs] footprints: repo definition (footprint_inspect.py) — fires in >={thr} "
        f"of {nw} IID windows, mesh-node binary mask, cosine on that mask")

    R = {t: lag_corr(windows, cols, t) for t in sorted({e["tau"] for e in edges})}
    amp = np.array([np.median([w["X"][:, j].mean() for w in windows]) for j in cols])
    edgeset = {(e["a"], e["b"], e["tau"]) for e in edges}
    edgepairs = {(e["a"], e["b"]) for e in edges} | {(e["b"], e["a"]) for e in edges}

    fcos = np.full((len(sel), len(sel)), np.nan)
    for i in range(len(sel)):
        for j in range(len(sel)):
            fcos[i, j] = footprint_cos(fires, sel[i], sel[j])

    out = []
    used = set()
    for e in edges:
        ia, ib = sel.index(e["a"]), sel.index(e["b"])
        tgt = dict(fcos=fcos[ia, ib], r=abs(R[e["tau"]][ia, ib]), amp=amp[ia])
        best, bestd = None, None
        for i in range(len(sel)):
            for j in range(len(sel)):
                if i == j:
                    continue
                if (sel[i], sel[j]) in edgepairs or (sel[i], sel[j], e["tau"]) in edgeset:
                    continue
                if (sel[i], sel[j]) in used:
                    continue
                d_f = abs(fcos[i, j] - tgt["fcos"])
                d_r = abs(abs(R[e["tau"]][i, j]) - tgt["r"])
                d_a = abs(amp[i] - tgt["amp"]) / max(tgt["amp"], 1e-9)
                if not np.isfinite(d_f):
                    continue
                # NEAREST-TO-TARGET on all three axes jointly (normalised by the
                # prereg tolerances), never argmax inside a band.
                dist = (d_f / 0.05) ** 2 + (d_r / 0.05) ** 2 + (d_a / 0.25) ** 2
                if bestd is None or dist < bestd:
                    bestd, best = dist, (i, j, d_f, d_r, d_a)
        if best is None:
            log(f"[pairs] NO MATCH for f{e['a']}->f{e['b']}@{e['tau']}")
            out.append(dict(edge=e, control=None, matched=False))
            continue
        i, j, d_f, d_r, d_a = best
        used.add((sel[i], sel[j]))
        ok = (d_f <= 0.05) and (d_r <= 0.05) and (d_a <= 0.25)
        shared = sorted({sel[i], sel[j]} & {e["a"], e["b"]})
        if shared:
            log(f"        NOTE: control shares endpoint(s) {shared} with its edge — the "
                f"same ablation rollout serves both arms (no extra GPU cost, but the two "
                f"asym values are not independent).")
        log(f"[pairs] edge f{e['a']}->f{e['b']}@{e['tau']}  "
            f"target fcos={tgt['fcos']:.3f} |r|={tgt['r']:.3f} amp={tgt['amp']:.4f}")
        log(f"        ctrl f{sel[i]}->f{sel[j]}       "
            f"fcos={fcos[i,j]:.3f} (d={d_f:.3f}) |r|={abs(R[e['tau']][i,j]):.3f} "
            f"(d={d_r:.3f}) amp={amp[i]:.4f} (d={100*d_a:.1f}%)  "
            f"{'MATCHED' if ok else 'OUT OF TOLERANCE — flagged, not substituted'}")
        out.append(dict(edge=e, control=dict(a=sel[i], b=sel[j], tau=e["tau"],
                                             fcos=float(fcos[i, j]),
                                             r=float(abs(R[e["tau"]][i, j])),
                                             amp=float(amp[i])),
                        target=dict(fcos=float(tgt["fcos"]), r=float(tgt["r"]),
                                    amp=float(tgt["amp"])),
                        d_fcos=float(d_f), d_r=float(d_r), d_amp_frac=float(d_a),
                        shared_endpoints=shared, matched=bool(ok)))
    n_ok = sum(1 for o in out if o["matched"])
    log(f"[pairs] {n_ok}/{len(out)} controls inside all three tolerances")
    p = os.path.join(OUTDIR, "hybrid_pairs.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    log(f"[pairs] -> {p}")

def cmd_arms(args):
    pairs = json.load(open(os.path.join(OUTDIR, "hybrid_pairs.json")))
    storms = NH_STORMS
    arms = []
    seen = set()
    for o in pairs:
        for kind in ("edge", "control"):
            p = o.get(kind)
            if not p:
                continue
            for who in ("a", "b"):
                f = p[who]
                if f == TC_READOUT:
                    raise SystemExit("refusing to emit an ablation of the 3243 TC readout")
                k = (f, kind, p["a"], p["b"], p["tau"])
                if k in seen:
                    continue
                seen.add(k)
                arms.append(dict(ablate=int(f), role=who, pair_kind=kind,
                                 pair=[int(p["a"]), int(p["b"])], tau=int(p["tau"]),
                                 storms=storms, mode="restore-to-normal"))
    uniq = sorted({a["ablate"] for a in arms})
    log(f"[arms] {len(arms)} arm records, {len(uniq)} distinct features to ablate: {uniq}")
    log(f"[arms] x {len(storms)} NH storms = {len(uniq)*len(storms)} GPU rollout batteries "
        f"(plus one shared baseline per storm)")
    p = os.path.join(OUTDIR, "hybrid_arms.json")
    json.dump(dict(arms=arms, distinct_features=uniq, storms=storms,
                   note="MECH_TRACK=all required; 3243 never ablated"),
              open(p, "w"), indent=1)
    log(f"[arms] -> {p}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["calibrate", "select", "condition", "graph",
                                    "pairs", "arms", "footprints", "nullgraph"])
    ap.add_argument("--source", default="mech", choices=["mech", "real"])
    ap.add_argument("--resdir", default=None)
    ap.add_argument("--n-select", type=int, default=N_SELECT)
    ap.add_argument("--h", type=int, default=H_EXPECT, help="expected rollout length")
    ap.add_argument("--tau-max", type=int, default=2)
    ap.add_argument("--tau-inject", type=int, default=1)
    ap.add_argument("--pc-alpha", type=float, default=0.05)
    ap.add_argument("--cons-frac", type=float, default=0.5)
    ap.add_argument("--corr-thresh", type=float, default=0.95)
    ap.add_argument("--betas", default="0.1,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--n-windows-small", type=int, default=8)
    ap.add_argument("--n-windows-big", type=int, default=52)
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--mode", default="both", choices=["consensus", "multiple", "both"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=10)
    args = ap.parse_args()
    dict(calibrate=cmd_calibrate, select=cmd_select, condition=cmd_condition,
         graph=cmd_graph, pairs=cmd_pairs, arms=cmd_arms, nullgraph=cmd_nullgraph,
         footprints=lambda a: footprint_fires(force=True))[args.cmd](args)

if __name__ == "__main__":
    main()
