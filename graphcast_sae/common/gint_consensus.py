"""Phase 2 — Ĝ_int: PCMCI+ consensus causal graph on deseasonalized mode series,
plus the physics-plausibility read (pre-registered bars in docs/prereg/prereg_phase1_2.md).

Pipeline per candidate:
  1. deseasonalize + detrend each mode series (harmonic regression: linear trend +
     annual K=3 + diurnal K=1 harmonics on the full contiguous span). P2.1 check.
  2. chop residual into W windows-as-realisations.
  3. PCMCI+ (RobustParCorr, tau_max=8, pc_alpha=0.05) per window; consensus edge
     kept if detected in >=CONS_FRAC of windows (pair-level and lag-level).
  4. physics: mode footprint centroids (lat/lon) -> P2.3 eastward-propagation among
     extratropical zonally-separated pairs; P2.4 lag realism.

Run (savar venv):

Paper: shared: deseasonalisation and consensus helpers for the observational graph
Inputs: candidates/pool_v2_candidates.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/litext_gc_gint.npy (--out, when run as a script)
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.common.gint_consensus --traj activations/mode_series/traj_2011_6yr.npy --nwin 8
"""
import argparse, os, json
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
TAU_MAX = int(os.environ.get("TAU_MAX", 8))
PC_ALPHA = float(os.environ.get("PC_ALPHA", 0.05))
CONS_FRAC = float(os.environ.get("CONS_FRAC", 0.5))

# ── deseasonalization + detrend ──────────────────────────────────────────────
def harmonic_design(times):
    """times: datetime64[ns] -> design matrix [1, t_lin, annual(K=3), diurnal(K=1)]."""
    t = times.astype("datetime64[s]").astype(np.float64)      # seconds
    t_lin = (t - t.mean()) / (t.std() + 1e-9)
    # annual phase
    yr = 365.2422 * 86400.0
    day = 86400.0
    cols = [np.ones_like(t), t_lin]
    for k in range(1, 4):                                     # annual + 2 harmonics
        cols += [np.sin(2 * np.pi * k * t / yr), np.cos(2 * np.pi * k * t / yr)]
    for k in range(1, 2):                                     # diurnal
        cols += [np.sin(2 * np.pi * k * t / day), np.cos(2 * np.pi * k * t / day)]
    return np.stack(cols, 1)

def deseason(series, times):
    """series (T,N) -> residual (T,N) after OLS removal of harmonic+trend design."""
    D = harmonic_design(times)
    beta, *_ = np.linalg.lstsq(D, series, rcond=None)
    return series - D @ beta

def power_at(freq_per_sec, series, times):
    """|DFT| power at a target frequency (cycles/sec), per column, via least-squares fit."""
    t = times.astype("datetime64[s]").astype(np.float64)
    c = np.cos(2 * np.pi * freq_per_sec * t)
    s = np.sin(2 * np.pi * freq_per_sec * t)
    B = np.stack([c, s], 1)
    coef, *_ = np.linalg.lstsq(B, series - series.mean(0), rcond=None)
    return np.sqrt((coef ** 2).sum(0))                        # amplitude per column

# ── PCMCI+ per window ────────────────────────────────────────────────────────
def _wi():
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"

def pcmci_one(args):
    wi, series = args
    from tigramite.data_processing import DataFrame
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.robust_parcorr import RobustParCorr
    pc = PCMCI(dataframe=DataFrame(series), cond_ind_test=RobustParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=1, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    g = res["graph"]
    N, _, T1 = g.shape
    det = [(c, e, tau) for c in range(N) for e in range(N) if c != e
           for tau in range(1, T1) if g[c, e, tau] == "-->"]
    return wi, det

def consensus(dets_per_win, N, frac):
    pair_cnt = np.zeros((N, N)); lag_cnt = {}
    for wi, det in dets_per_win.items():
        pairs = {(c, e) for (c, e, _) in det}
        for (c, e) in pairs:
            pair_cnt[c, e] += 1
        for (c, e, tau) in det:
            lag_cnt[(c, e, tau)] = lag_cnt.get((c, e, tau), 0) + 1
    thr = frac * len(dets_per_win)
    pair_edges = [(c, e) for c in range(N) for e in range(N)
                  if c != e and pair_cnt[c, e] >= thr]
    lag_edges = [(c, e, tau) for (c, e, tau), n in lag_cnt.items() if n >= thr]
    return pair_edges, lag_edges, pair_cnt

# ── mode footprint centroids (circular-safe) ─────────────────────────────────
def centroids(What, xyz):
    C = (What[:, :, None] * xyz[None]).sum(1)                 # (N,3)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    lat = np.degrees(np.arcsin(np.clip(C[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(C[:, 1], C[:, 0]))
    return lat, lon

def physics_read(pair_edges, lag_edges, lat, lon):
    """P2.3 eastward propagation among extratropical, zonally-separated same-hemi pairs."""
    def dlon_east(a, b):  # signed eastward lon displacement a->b in (-180,180]
        d = (lon[b] - lon[a] + 180) % 360 - 180
        return d
    considered, eastward = [], 0
    for (c, e) in pair_edges:
        if abs(lat[c]) > 25 and abs(lat[e]) > 25 and np.sign(lat[c]) == np.sign(lat[e]):
            d = dlon_east(c, e)
            if abs(d) > 15:
                considered.append((c, e, round(float(d), 1)))
                if d > 0:                                     # source west of target
                    eastward += 1
    frac_east = eastward / len(considered) if considered else float("nan")
    lags = [tau for (_, _, tau) in lag_edges]
    med_lag = float(np.median(lags)) if lags else float("nan")
    return dict(n_extratrop_zonal=len(considered), frac_eastward=frac_east,
                pairs=considered, median_lag=med_lag,
                lag_hist={int(t): int(sum(1 for x in lags if x == t)) for t in set(lags)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--cands", default=str(ROOT / "candidates/pool_v2_candidates.npy"),
                    help="POOL_V2 is authoritative: {leiden,vmax,km,shift,sae,qperm}. "
                         "act_candidates.npy is the superseded v1 pool (has blur, "
                         "lacks sae/qperm) and will NOT reproduce published results.")
    ap.add_argument("--nwin", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of candidates")
    ap.add_argument("--out", default=str(ROOT / "results/litext_gc_gint.npy"))
    args = ap.parse_args()

    tr = np.load(args.traj, allow_pickle=True).item()
    cd = np.load(args.cands, allow_pickle=True).item()
    xyz = cd["xyz"]
    n_done = tr.get("n_done", len(tr["target_times"]))
    times = tr["target_times"][:n_done]
    names = tr["names"] if not args.only else args.only.split(",")
    print(f"trajectory: {n_done} steps  {str(times[0])[:13]}..{str(times[-1])[:13]}")

    results = {}
    for name in names:
        S = tr["series"][name][:n_done]                       # (T,N)
        N = S.shape[1]
        res_series = deseason(S, times)
        # P2.1
        amp_raw_yr = power_at(1 / (365.2422 * 86400), S, times)
        amp_res_yr = power_at(1 / (365.2422 * 86400), res_series, times)
        amp_raw_dy = power_at(1 / 86400, S, times)
        amp_res_dy = power_at(1 / 86400, res_series, times)
        red_yr = 1 - float((amp_res_yr / (amp_raw_yr + 1e-12)).mean())
        red_dy = 1 - float((amp_res_dy / (amp_raw_dy + 1e-12)).mean())
        # standardize residual per mode
        Z = (res_series - res_series.mean(0)) / (res_series.std(0) + 1e-9)
        # windows-as-realisations
        edges = np.linspace(0, n_done, args.nwin + 1).astype(int)
        jobs = [(wi, Z[edges[wi]:edges[wi + 1]]) for wi in range(args.nwin)]
        dets = {}
        with ProcessPoolExecutor(max_workers=min(args.nwin,
                max(1, (os.cpu_count() or 8) - 2)), initializer=_wi) as ex:
            for wi, det in ex.map(pcmci_one, jobs):
                dets[wi] = det
        pair_edges, lag_edges, pair_cnt = consensus(dets, N, CONS_FRAC)
        lat, lon = centroids(cd["cands"][name], xyz)
        phys = physics_read(pair_edges, lag_edges, lat, lon)
        maxE = 0.5 * N * (N - 1)
        p22 = 0 < len(pair_edges) < maxE
        results[name] = dict(
            pair_edges=pair_edges, lag_edges=lag_edges, pair_cnt=pair_cnt,
            dets_per_win={k: v for k, v in dets.items()},
            centroid_lat=lat, centroid_lon=lon,
            p21_red_annual=red_yr, p21_red_diurnal=red_dy,
            p22_nonempty_unsat=bool(p22), n_pair_edges=len(pair_edges),
            physics=phys, N=N, nwin=args.nwin)
        print(f"\n[{name}] N={N}  |pair_edges|={len(pair_edges)}  |lag_edges|={len(lag_edges)}")
        print(f"  P2.1 deseason: annual power -{red_yr*100:.0f}%  diurnal -{red_dy*100:.0f}%  "
              f"(bar >=90%: {'PASS' if red_yr>=.9 and red_dy>=.9 else 'CHECK'})")
        print(f"  P2.2 non-degenerate (0<{len(pair_edges)}<{maxE:.0f}): {'PASS' if p22 else 'FAIL'}")
        print(f"  P2.4 median lag={phys['median_lag']} steps  hist={phys['lag_hist']}")
        print(f"  P2.3 extratrop zonal pairs={phys['n_extratrop_zonal']}  "
              f"frac_eastward={phys['frac_eastward']}  (bar >0.60)")

    Path(args.out).parent.mkdir(exist_ok=True)
    np.save(args.out, dict(results=results, tau_max=TAU_MAX, pc_alpha=PC_ALPHA,
            cons_frac=CONS_FRAC, nwin=args.nwin, n_done=int(n_done),
            span=(str(times[0]), str(times[-1]))), allow_pickle=True)
    print(f"\nsaved -> {args.out}")

if __name__ == "__main__":
    main()
