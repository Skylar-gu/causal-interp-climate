"""FG-6 — the one table, printed unconditionally for every member including the anchors.

Pulls together the pre-flight basis diagnostics, the trajectory conditioning (re-measured
here on the real 12-year series, which is what counts), the graph, both edge-count nulls,
the geography-permutation null and the signature-physics verdict, and then applies the
pre-registered anchor gate (§7) and physics bars (§8) mechanically.

Nothing here decides anything the prereg did not already fix; it only reads the bars off.

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: candidates/pool_flag_v2_candidates.npy (not shipped, see docs/REPRODUCE.md); results/flag_gint.npy (not shipped, see docs/REPRODUCE.md); results/flag_gint_nulls.npy (not shipped, see docs/REPRODUCE.md); results/flag_gint_preflight.json (not shipped, see docs/REPRODUCE.md); results/flag_signature_physics.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/flag_gint_report.json (--out)
Run:   # JAX env, CPU
    OMP_NUM_THREADS=8 python -m graphcast_sae.obsgraph.report_flag_gint
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT

from graphcast_sae.common.gint_consensus import deseason                                    # noqa: E402

BAR_FE = 0.60
BAR_P = 0.05
ANCHORS = ("shift_flag", "qperm_flag", "qrand_flag", "qrandc_flag")

def traj_conditioning(traj, name, times, n_done):
    S = traj["series"][name][:n_done]
    R = deseason(S, times)
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    C = np.corrcoef(Z.T)
    ev = np.linalg.eigvalsh(C)
    offd = np.abs(C - np.eye(C.shape[0]))
    t = times.astype("datetime64[s]").astype(np.float64)
    nyq = np.cos(2 * np.pi * 2 * t / 86400.0)
    nyq = (nyq - nyq.mean()) / nyq.std()
    nf = ((nyq @ Z) / len(nyq)) ** 2
    return dict(cond=float(ev.max() / max(ev.min(), 1e-12)), min_eig=float(ev.min()),
                max_abs_corr=float(offd.max()), med_abs_corr=float(np.median(offd)),
                nyq_max=float(nf.max()), n_nyq_over_50=int((nf > 0.5).sum()))

def eff_nodes(W):
    return float(np.median([w.sum() ** 2 / (w ** 2).sum() if (w ** 2).sum() > 0 else 0
                            for w in W]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="results/flag_gint.npy")
    ap.add_argument("--nulls", default="results/flag_gint_nulls.npy")
    ap.add_argument("--sig", default="results/flag_signature_physics.npy")
    ap.add_argument("--traj", default="activations/mode_series/traj_flag2_full.npy")
    ap.add_argument("--cands", default="candidates/pool_flag_v2_candidates.npy")
    ap.add_argument("--preflight", default="results/flag_gint_preflight.json")
    ap.add_argument("--out", default="results/flag_gint_report.json")
    args = ap.parse_args()

    G = np.load(ROOT / args.graph, allow_pickle=True).item()
    res = G["results"]
    NL = np.load(ROOT / args.nulls, allow_pickle=True).item()["nulls"]
    cd = np.load(ROOT / args.cands, allow_pickle=True).item()
    tr = np.load(ROOT / args.traj, allow_pickle=True).item()
    pf = json.load(open(ROOT / args.preflight))
    try:
        # signature_physics saves {per_candidate: {member: {...}}, table_rows: [...], ...}
        SG = np.load(ROOT / args.sig, allow_pickle=True).item()["per_candidate"]
    except Exception as e:
        print(f"NOTE: signature-physics results unavailable ({e}) -> SG column shows None")
        SG = {}
    n_done = int(tr.get("n_done", len(tr["target_times"])))
    times = tr["target_times"][:n_done]

    order = [n for n in ("leiden_flag", "sae_flag", "sae_sel_flag", "vmax_flag", "km_flag")
             if n in res] + [a for a in ANCHORS if a in res]
    rows = {}
    for n in order:
        r = res[n]
        nl = NL.get(n, {})
        gp = nl.get("geo_perm", {})
        sb = nl.get("surrogate", {})
        sc = nl.get("collinear", {})
        cond = traj_conditioning(tr, n, times, n_done)
        sg = SG.get(n, {})
        vA = (sg.get("verdictA") or {}).get("status")
        fe = r["physics"]["frac_eastward"]
        p = gp.get("p_value", float("nan"))
        rows[n] = dict(
            is_anchor=n in ANCHORS, N=int(r["N"]),
            eff_nodes=eff_nodes(cd["cands"][n]), edges=int(r["n_pair_edges"]),
            lag_edges=len(r["lag_edges"]), median_lag=r["physics"]["median_lag"],
            frac_eastward=(None if not np.isfinite(fe) else float(fe)),
            n_pairs=int(r["physics"]["n_extratrop_zonal"]),
            geo_p=(None if not np.isfinite(p) else float(p)),
            geo_null_mean=gp.get("null_mean"), geo_null_sd=gp.get("null_sd"),
            geo_null_max=gp.get("null_max"), geo_null_varies=gp.get("null_varies"),
            geo_bar_attainable=gp.get("bar_attainable"),
            nullB_mean=sb.get("null_mean"), nullB_sd=sb.get("null_sd"),
            nullB_p=sb.get("p_value"),
            nullC_mean=sc.get("null_mean"), nullC_sd=sc.get("null_sd"),
            nullC_p=sc.get("p_value"),
            **cond,
            preflight_cond=pf["rep"].get(n, {}).get("cond"),
            sg_verdict=vA, sg_why=(sg.get("verdictA") or {}).get("why"),
            clears_FE=bool(np.isfinite(fe) and fe > BAR_FE),
            clears_p=bool(np.isfinite(p) and p < BAR_P))
        rows[n]["clears_physics"] = bool(rows[n]["clears_FE"] and rows[n]["clears_p"])

    W = 132
    print("=" * W)
    print(f"FG-6  FLAGSHIP Ĝ_int — span {G['span'][0][:13]}..{G['span'][1][:13]}  "
          f"n={G['n_done']}  nwin={G['nwin']}  tau_max={G['tau_max']}  "
          f"pc_alpha={G['pc_alpha']}  cons_frac={G['cons_frac']}")
    print("=" * W)
    print(f"{'member':<14}{'eff.nod':>8}{'edges':>7}{'fr_east':>9}{'npair':>6}"
          f"{'geo p':>8}{'nullB':>12}{'nullC':>12}{'cond':>9}{'mineig':>8}"
          f"{'max|r|':>8}{'SG verdict':>22}")
    for n in order:
        v = rows[n]
        tag = "*" if v["is_anchor"] else " "
        fe = "  n/a" if v["frac_eastward"] is None else f"{v['frac_eastward']:.3f}"
        gp = "  n/a" if v["geo_p"] is None else f"{v['geo_p']:.4f}"
        nb = ("n/a" if v["nullB_mean"] is None
              else f"{v['nullB_mean']:.1f}±{v['nullB_sd']:.1f}")
        nc = ("n/a" if v["nullC_mean"] is None
              else f"{v['nullC_mean']:.1f}±{v['nullC_sd']:.1f}")
        print(f"{tag}{n:<13}{v['eff_nodes']:>8.0f}{v['edges']:>7}{fe:>9}{v['n_pairs']:>6}"
              f"{gp:>8}{nb:>12}{nc:>12}{v['cond']:>9.1f}{v['min_eig']:>8.4f}"
              f"{v['max_abs_corr']:>8.3f}{str(v['sg_verdict'])[:21]:>22}")
    print("  * = anchor (negative control). Anchor edge COUNTS are not comparable to "
          "candidate counts (prereg A1).")

    # ── the control-must-be-able-to-fail rule, all three legs, printed ───────────────────────────────
    print("\nGUARDRAIL #9 — two-sided calibration of the frac_eastward bar (>0.60, p<0.05)")
    varies = [rows[n]["geo_null_varies"] for n in order if rows[n]["geo_null_varies"] is not None]
    sds = [rows[n]["geo_null_sd"] for n in order if rows[n]["geo_null_sd"] is not None]
    att = [rows[n]["geo_null_max"] for n in order if rows[n]["geo_null_max"] is not None]
    if varies:
        print(f"  (i)   null VARIES: {all(varies)}  "
              f"(SD range {min(sds):.3f}-{max(sds):.3f} across members)")
    else:
        print("  (i)   null VARIES: n/a")
    if att:
        print(f"  (ii)  bar ATTAINABLE under the null: {all(x > BAR_FE for x in att)}  "
              f"(null max {min(att):.3f}-{max(att):.3f})")
    else:
        print("  (ii)  bar ATTAINABLE: n/a")
    anc = [n for n in order if rows[n]["is_anchor"]]
    breach = [n for n in anc if rows[n]["clears_physics"]]
    detail = ", ".join("{}={}".format(a, rows[a]["frac_eastward"]) for a in anc)
    print(f"  (iii) negative controls FAIL the bar: {not breach}  ({detail})")

    print("\nFG-4  ANCHOR GATE")
    for a in anc:
        v = rows[a]
        print(f"  {a:<14} frac_eastward={v['frac_eastward']} (n={v['n_pairs']}) "
              f"geo p={v['geo_p']}  edges={v['edges']}  "
              f"-> {'** BREACH **' if v['clears_physics'] else 'clean'}")
    gate = "INSTRUMENT FAILURE" if breach else "PASS"
    print(f"  VERDICT: {gate}" + (f"  breached by {breach}" if breach else
          "  — both/all anchors fail the physics bar, as required"))

    print("\nFG-5  PHYSICS BARS (only licensed if the anchor gate PASSES)")
    for n in order:
        if rows[n]["is_anchor"]:
            continue
        v = rows[n]
        print(f"  {n:<14} FE {v['frac_eastward']} {'>' if v['clears_FE'] else '<='} 0.60  "
              f"| geo p {v['geo_p']} {'<' if v['clears_p'] else '>='} 0.05  "
              f"| edges {v['edges']} vs NULL C {v['nullC_mean']}±{v['nullC_sd']} "
              f"(p={v['nullC_p']})  -> {'CLEARS' if v['clears_physics'] else 'FAILS'}")

    json.dump(dict(rows=rows, anchor_gate=gate, breached=breach,
                   span=[str(x) for x in G["span"]], n_done=int(G["n_done"]),
                   nwin=int(G["nwin"]), bar_frac_eastward=BAR_FE, bar_p=BAR_P),
              open(ROOT / args.out, "w"), indent=1, default=float)
    print(f"\n-> {args.out}")

if __name__ == "__main__":
    main()
