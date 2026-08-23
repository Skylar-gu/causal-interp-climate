"""Signature-FIRST physics read (prereg block "SIGNATURE-FIRST physics", 2026-07-20 rev2).

Supersedes the catalog-first own-terms test (pcmci/ownterms_physics.py). Instead of
pre-committing speed-indexed phenomena and force-matching the nearest band, we READ what
each causal chain actually ENCODES (SG-2 fingerprint), CLASSIFY its propagation type
(SG-3), THEN apply only the falsifiable test that matches that type (SG-4). Rigor is in the
pre-registered PROTOCOL, not a pre-registered answer. Fixes the polar loophole (|lat|>80
excluded from zonal speed) and judges each chain on its OWN footprint coherence — no
guilt-by-association.

Run (savar venv, CPU only):

Paper: shared: storm-track / great-circle physics used by the verdicts
Inputs: candidates/pool_v2_candidates.npy (not shipped, see docs/REPRODUCE.md); results/litext_gc_gint_v2.npy (not shipped, see docs/REPRODUCE.md); results/litext_gc_ownterms_physics.npy (not shipped, see docs/REPRODUCE.md); results/litext_gc_trust_table.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/litext_gc_signature_physics.npy (--out, when run as a script)
Run:   # JAX env, CPU
    python -m graphcast_sae.common.signature_physics
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from graphcast_sae.common.gint_consensus import deseason

V_MAX = 50.0        # SG-1 plausibility gate (m/s)
R_MIN = 0.80        # SG-2 footprint-coherence threshold (mean resultant length)
POLAR = 80.0        # SG-2 polar-exclusion latitude
CANDIDATES = ["leiden_act", "sae_act", "km_act", "vmax_act", "shift_act", "qperm_act"]
ANCHORS = {"shift_act", "qperm_act"}

# SG-4 phenomenon bands (each with an independently-predicted number)
PHEN = {
    "storm_track": dict(label="extratropical storm-track / RWP group velocity",
                        band=(25.0, 40.0), obs="25-35 m/s (Chang 1993/1999; Berbery & Vera 1996)"),
    "easterly_wave": dict(label="tropical easterly wave", band=(5.0, 11.0),
                          obs="~8 m/s (Kiladis 2009)"),
    "mjo": dict(label="MJO", band=(3.0, 8.0), obs="~5 m/s (Zhang 2005)"),
    "teleconnection": dict(label="stationary Rossby-wave teleconnection",
                           band=(0.0, 8.0), obs="~0 phase speed, day-week lag (Jin & Hoskins 1995)"),
}

def gc_km(la0, lo0, la1, lo1):
    la0, lo0, la1, lo1 = map(np.radians, (la0, lo0, la1, lo1))
    return 6371.0 * np.arccos(np.clip(np.sin(la0) * np.sin(la1) +
                              np.cos(la0) * np.cos(la1) * np.cos(lo1 - lo0), -1, 1))

def sdlon(lo0, lo1):
    return (lo1 - lo0 + 180) % 360 - 180

# ── SG-2 per-mode footprint fingerprint ──────────────────────────────────────
def mode_fp(W, xyz, lat, lon, m):
    w = np.clip(W[m].astype(float), 0, None)
    s = w.sum()
    if s <= 0:
        return None
    C = (w[:, None] * xyz).sum(0) / s
    R = float(np.linalg.norm(C))
    Cn = C / (R + 1e-12)
    clat = float(np.degrees(np.arcsin(np.clip(Cn[2], -1, 1))))
    clon = float(np.degrees(np.arctan2(Cn[1], Cn[0])))
    d = gc_km(lat, lon, clat, clon)
    spread = float(np.sqrt((w * d ** 2).sum() / s))
    pr = float(s ** 2 / (w ** 2).sum())
    ftop = float(np.sort(w)[::-1][:20].sum() / s)
    return dict(mode=int(m), clat=round(clat, 1), clon=round(clon, 1), R=round(R, 3),
                spread_km=round(spread, 0), PR=round(pr, 0), frac_top20=round(ftop, 3),
                coherent=bool(R >= R_MIN), polar=bool(abs(clat) > POLAR))

# ── consensus lag helper (min consensus lag, else modal window-detection lag) ─
def build_lags(r):
    lagcons = {}
    for (c, e, t) in r["lag_edges"]:
        lagcons.setdefault((c, e), []).append(t)
    detlag = {}
    for _, det in r["dets_per_win"].items():
        for (c, e, tau) in det:
            detlag.setdefault((c, e), []).append(tau)

    def lag(c, e):
        if (c, e) in lagcons:
            return min(lagcons[(c, e)])
        taus = detlag.get((c, e), [])
        if not taus:
            return None
        cnt = Counter(taus)
        top = max(cnt.values())
        return min(t for t, n in cnt.items() if n == top)
    return lag

# ── SG-1 gated max-consensus simple path (≥3 nodes); optional coherence filter ─
def best_path(r, lat, lon, lag_of, fp, coherent_only=False):
    pc = r["pair_cnt"]
    edges = []
    for (c, e) in r["pair_edges"]:
        lg = lag_of(c, e)
        if lg is None:
            continue
        spd = gc_km(lat[c], lon[c], lat[e], lon[e]) * 1000 / (lg * 6 * 3600)
        if spd > V_MAX:
            continue
        if coherent_only and not (fp[c]["coherent"] and fp[e]["coherent"]
                                  and not fp[c]["polar"] and not fp[e]["polar"]):
            continue
        edges.append((c, e))
    adj = {}
    for (c, e) in edges:
        adj.setdefault(c, []).append(e)
    best = [None]

    def dfs(path, tot):
        node = path[-1]
        if len(path) >= 3:
            key = (tot, len(path) - 1, tuple(-x for x in path))
            if best[0] is None or key > best[0][0]:
                best[0] = (key, list(path))
        for nx in adj.get(node, []):
            if nx not in path:
                dfs(path + [nx], tot + int(pc[node, nx]))

    for s in list(adj):
        dfs([s], 0)
    return None if best[0] is None else best[0][1]

# ── SG-2 seasonality proxy on the deseasonalized mode series ──────────────────
def seasonality(traj, name, times, c, e, lag, tgt_lat):
    S = traj["series"][name]
    n = min(len(times), S.shape[0])
    Z = deseason(S[:n], times[:n])
    months = times[:n].astype("datetime64[M]").astype(int) % 12 + 1
    # cold season by target-mode hemisphere
    cold = {5, 6, 7, 8, 9} if tgt_lat < 0 else {11, 12, 1, 2, 3}
    warm = {11, 12, 1, 2, 3} if tgt_lat < 0 else {5, 6, 7, 8, 9}
    xc, xe = Z[:n - lag, c], Z[lag:n, e]
    mth = months[lag:n]
    mc = np.isin(mth, list(cold)); mw = np.isin(mth, list(warm))

    def corr(mask):
        a, b = xc[mask], xe[mask]
        if len(a) < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    rc, rw = corr(mc), corr(mw)
    ratio = abs(rc) / (abs(rw) + 1e-9) if np.isfinite(rc) and np.isfinite(rw) else np.nan
    return dict(r_cold=None if not np.isfinite(rc) else round(rc, 3),
                r_warm=None if not np.isfinite(rw) else round(rw, 3),
                ratio=None if not np.isfinite(ratio) else round(ratio, 2))

# ── SG-2/SG-3 fingerprint a given path ───────────────────────────────────────
def fingerprint_chain(path, r, lat, lon, lag_of, fp, traj, name, times):
    if path is None:
        return None
    edges = list(zip(path[:-1], path[1:]))
    modes = [fp[m] for m in path]
    edge_fp = []
    scoreable = []
    for (c, e) in edges:
        lg = lag_of(c, e)
        tot = gc_km(lat[c], lon[c], lat[e], lon[e])
        zon = gc_km(lat[c], lon[c], lat[c], lon[e])
        polar_edge = abs(lat[c]) > POLAR or abs(lat[e]) > POLAR
        spd = tot * 1000 / (lg * 6 * 3600)
        score = (fp[c]["coherent"] and fp[e]["coherent"] and not polar_edge)
        seas = seasonality(traj, name, times, c, e, int(lg), float(lat[e]))
        d = dict(c=int(c), e=int(e), lag=int(lg), timescale_h=int(lg * 6),
                 gc_km=round(float(tot), 0), dlon=round(float(sdlon(lon[c], lon[e])), 1),
                 dlat=round(float(lat[e] - lat[c]), 1),
                 zonal_frac=(None if polar_edge else round(float(zon / (tot + 1e-9)), 2)),
                 speed=(None if polar_edge else round(float(spd), 1)),
                 polar_excluded=bool(polar_edge), scoreable=bool(score), seasonality=seas)
        edge_fp.append(d)
        if score:
            scoreable.append(d)
    # SG-3 classify
    lats = [m["clat"] for m in modes]
    net_disp = gc_km(lat[path[0]], lon[path[0]], lat[path[-1]], lon[path[-1]])
    if len(scoreable) >= 2:
        dsign = {np.sign(s["dlon"]) for s in scoreable}
        zf = np.median([s["zonal_frac"] for s in scoreable])
        if len(dsign) == 1 and zf > 0.6:
            ctype = "zonal-propagating"
        elif zf < 0.4:
            ctype = "meridional"
        else:
            ctype = "mixed-coherent"
    elif len(scoreable) == 1 and net_disp < 500:
        ctype = "standing/quasi-stationary"
    else:
        ctype = "incoherent/artifact"
    med_speed = (float(np.median([s["speed"] for s in scoreable])) if scoreable else None)
    net_dlon = sum(s["dlon"] for s in scoreable) if scoreable else 0.0
    return dict(path=[int(p) for p in path], modes=modes, edges=edge_fp,
                n_coherent=int(sum(m["coherent"] for m in modes)), n_modes=len(modes),
                n_scoreable=len(scoreable), chain_type=ctype,
                med_scoreable_speed=(None if med_speed is None else round(med_speed, 1)),
                net_dlon=round(float(net_dlon), 1),
                direction=("E" if net_dlon > 0 else ("W" if net_dlon < 0 else "-")),
                med_abslat=round(float(np.median([abs(x) for x in lats])), 0),
                same_hemi=bool(len({np.sign(x) for x in lats}) == 1),
                net_disp_km=round(float(net_disp), 0))

# ── SG-4 type-matched falsifiable test ───────────────────────────────────────
def classify_verdict(fpc):
    if fpc_is_none(fpc):
        return "MISS", "no >=3-node chain survives the plausibility gate", None
    t = fpc["chain_type"]
    lat = fpc["med_abslat"]; spd = fpc["med_scoreable_speed"]; dr = fpc["direction"]
    if t == "incoherent/artifact":
        why = (f"incoherent: {fpc['n_coherent']}/{fpc['n_modes']} modes coherent, "
               f"{fpc['n_scoreable']} scoreable edges")
        return "MISS", why, None
    if t == "zonal-propagating" and spd is not None:
        if lat > 25 and fpc["same_hemi"] and dr == "E":
            ph = PHEN["storm_track"]; lo, hi = ph["band"]
            ok = lo <= spd <= hi
            return ("MATCH" if ok else "unidentified"), \
                (f"{ph['label']}: {spd} m/s vs {lo:.0f}-{hi:.0f} ({ph['obs']})" if ok
                 else f"zonal eastward extratrop but {spd} m/s outside group-vel band"), \
                ("storm_track" if ok else None)
        if lat < 20 and dr == "W":
            ph = PHEN["easterly_wave"]; lo, hi = ph["band"]; ok = lo <= spd <= hi
            return ("MATCH" if ok else "unidentified"), \
                (f"{ph['label']}: {spd} m/s vs {lo:.0f}-{hi:.0f} ({ph['obs']})" if ok
                 else f"zonal westward tropical but {spd} m/s outside easterly-wave band"), \
                ("easterly_wave" if ok else None)
        if lat < 15 and dr == "E":
            ph = PHEN["mjo"]; lo, hi = ph["band"]; ok = lo <= spd <= hi
            return ("MATCH" if ok else "unidentified"), \
                (f"{ph['label']}: {spd} m/s vs {lo:.0f}-{hi:.0f} ({ph['obs']})" if ok
                 else f"zonal eastward equatorial but {spd} m/s outside MJO band"), \
                ("mjo" if ok else None)
        return "unidentified coherent structure", \
            f"coherent zonal {dr} chain at {lat:.0f}deg lat, {spd} m/s (no catalog match)", None
    if t == "standing/quasi-stationary":
        return "INCONCLUSIVE", ("standing/quasi-stationary; teleconnection test needs "
                                "day-week lags but tau_max=48h cannot confirm"), None
    if t in ("meridional", "mixed-coherent"):
        return "unidentified coherent structure", \
            f"coherent {t} chain, {fpc['n_scoreable']} scoreable edges, med|lat|={lat:.0f}", None
    return "INCONCLUSIVE", f"type={t}", None

def fpc_is_none(x):
    return x is None

def main():
    global CANDIDATES, ANCHORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="results/litext_gc_gint_v2.npy")
    ap.add_argument("--cands", default="candidates/pool_v2_candidates.npy")
    ap.add_argument("--traj", default="activations/mode_series/traj_v2.npy")
    ap.add_argument("--only", default="", help="comma list; default = all CANDIDATES present")
    ap.add_argument("--anchors", default="shift_act,qperm_act")
    ap.add_argument("--out", default="results/litext_gc_signature_physics.npy")
    args = ap.parse_args()

    g = np.load(ROOT / args.graph, allow_pickle=True).item()["results"]
    cd = np.load(ROOT / args.cands, allow_pickle=True).item()
    traj = np.load(ROOT / args.traj, allow_pickle=True).item()
    times = traj["target_times"][:traj.get("n_done", len(traj["target_times"]))]
    xyz, latN, lonN = cd["xyz"], cd["lat"], cd["lon"]
    ANCHORS = {a for a in args.anchors.split(",") if a}
    sel = [c for c in args.only.split(",") if c] or CANDIDATES
    CANDIDATES = [c for c in sel if c in g and c in cd["cands"]]
    skipped = [c for c in sel if c not in CANDIDATES]
    if skipped:
        print(f"NOTE: not in this graph/pool, skipped: {skipped}")
    print(f"graph={args.graph}  traj={args.traj}  n_steps={len(times)}")

    # context columns from the published run; absent for a re-run on a new basis
    def _opt(path, fn, what):
        try:
            return fn(np.load(ROOT / path, allow_pickle=True).item())
        except Exception:
            print(f"NOTE: {what} unavailable for this basis -> column shows n/a")
            return {}
    old_ef = _opt("results/litext_gc_trust_table.npy",
                  lambda d: {r[0]: (r[4], r[6]) for r in d["rows"]}, "eastward-fraction table")
    ot = _opt("results/litext_gc_ownterms_physics.npy",
              lambda d: d["per_candidate"], "own-terms catalog")

    out = {}
    print(f"SIGNATURE-FIRST PHYSICS  (R_MIN={R_MIN}, polar-excl |lat|>{POLAR:.0f}, gate {V_MAX:.0f} m/s)\n")
    for name in CANDIDATES:
        r = g[name]
        W = cd["cands"][name]
        lat, lon = r["centroid_lat"], r["centroid_lon"]
        fp = {m: mode_fp(W, xyz, latN, lonN, m) for m in range(W.shape[0])}
        lag_of = build_lags(r)
        pA = best_path(r, lat, lon, lag_of, fp, coherent_only=False)
        pB = best_path(r, lat, lon, lag_of, fp, coherent_only=True)
        fA = fingerprint_chain(pA, r, lat, lon, lag_of, fp, traj, name, times)
        fB = fingerprint_chain(pB, r, lat, lon, lag_of, fp, traj, name, times)
        vA = classify_verdict(fA)
        vB = classify_verdict(fB) if fB and (pB != pA) else None
        out[name] = dict(chainA=fA, verdictA=dict(status=vA[0], why=vA[1], phenom=vA[2]),
                         chainB=fB, verdictB=(None if vB is None else
                                              dict(status=vB[0], why=vB[1], phenom=vB[2])),
                         is_anchor=name in ANCHORS)
        tag = "ANCHOR" if name in ANCHORS else "      "
        print(f"[{name:<11}] {tag}")
        if fA is None:
            print(f"   most-prominent chain: (none)  -> {vA[0]}: {vA[1]}")
        else:
            print(f"   most-prominent chain {'->'.join(map(str, fA['path']))}  type={fA['chain_type']}"
                  f"  coherent {fA['n_coherent']}/{fA['n_modes']}  scoreable-edges {fA['n_scoreable']}")
            for m in fA["modes"]:
                pl = " POLAR" if m["polar"] else ""
                co = "coh" if m["coherent"] else "DEGEN"
                print(f"       mode {m['mode']:2d} [{m['clat']:+5.1f},{m['clon']:+6.1f}] "
                      f"R={m['R']:.2f} spread={m['spread_km']:.0f}km top20={m['frac_top20']:.2f} {co}{pl}")
            for ed in fA["edges"]:
                sp = "excl(polar)" if ed["polar_excluded"] else f"{ed['speed']}m/s zf={ed['zonal_frac']}"
                sc = "scoreable" if ed["scoreable"] else "not-scoreable"
                se = ed["seasonality"]
                print(f"       edge {ed['c']:2d}->{ed['e']:2d} lag={ed['lag']}({ed['timescale_h']}h) "
                      f"dlon={ed['dlon']:+.0f} dlat={ed['dlat']:+.0f} {sp} [{sc}] "
                      f"seas cold/warm={se['r_cold']}/{se['r_warm']} ratio={se['ratio']}")
            print(f"   -> VERDICT {vA[0]}: {vA[1]}")
        if vB is not None and fB is not None:
            print(f"   [coherent-only chain {'->'.join(map(str, fB['path']))} type={fB['chain_type']}"
                  f"  -> {vB[0]}: {vB[1]}]")
        print()

    # ── FAIR TABLE: signature-first vs eastward-fraction vs interim OT ────────
    print("=" * 118)
    print("FAIR SIGNATURE-FIRST TABLE  (vs old eastward-fraction AND interim catalog-speed OT)")
    print("=" * 118)
    print(f"{'candidate':<12}{'chain':<12}{'type':<22}{'signature verdict':<30}"
          f"{'east-frac':<14}{'OT(catalog)':<20}")
    print("-" * 118)
    rows = []
    for name in CANDIDATES:
        o = out[name]; fA = o["chainA"]; vA = o["verdictA"]
        ch = "(none)" if fA is None else "->".join(map(str, fA["path"]))
        tp = "-" if fA is None else fA["chain_type"]
        verd = f"{vA['status']}"
        if vA["phenom"]:
            verd = f"MATCH {vA['phenom']}"
        ef = old_ef.get(name); efs = f"{ef[0]:.2f} {ef[1]}" if ef else "n/a"
        oti = ot.get(name, {})
        otm = oti.get("matched_label")
        pd = oti.get("chain", {}).get("polar_degenerate") if oti.get("chain") else None
        ots = ("MISS" if not otm else ("storm(polar!)" if pd else "storm-track"))
        anchor = " [ANCHOR]" if name in ANCHORS else ""
        print(f"{name:<12}{ch:<12}{tp:<22}{verd:<30}{efs:<14}{ots:<20}{anchor}")
        rows.append((name, ch, tp, vA["status"], vA["phenom"], efs, ots, name in ANCHORS))
    print("-" * 118)

    np.save(ROOT / args.out,
            dict(per_candidate=out, table_rows=rows, R_MIN=R_MIN, V_MAX=V_MAX, polar=POLAR,
                 graph=args.graph, traj=args.traj,
                 prereg="SG-1..SG-6 (docs/prereg/prereg_phase1_2.md 2026-07-20 rev2)"),
            allow_pickle=True)
    print(f"\nsaved -> {args.out}")

    # controls
    la = out["leiden_act"]["verdictA"]
    print(f"\nCONTROL leiden (positive): {la['status']} / {la['why']}")
    for a in sorted(ANCHORS):
        va = out[a]["verdictA"]
        print(f"CONTROL {a} (negative): {va['status']} / {va['why']}  "
              f"{'-> clean MISS' if va['status']=='MISS' else '-> NOT a clean miss, FLAG'}")

if __name__ == "__main__":
    main()
