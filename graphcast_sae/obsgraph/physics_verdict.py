"""Flagship storm-track verdict on a watching graph — the decisive SAE test.

For each member: count storm-track-consistent edges (25-40 m/s, eastward, |lat|>25, same-hemi),
score vs (a) a FREE geography-permutation null and (b) the FAIR locality-controlled null
(edge lengths held). Verdict logic with a positive-control GATE:
  * leiden_flag (positive control) MUST show a storm-track signal, else the pipeline is
    mis-wired -> INCONCLUSIVE, fix before trusting anything.
  * both anchors MUST be clean (fair p >= 0.05).
  * sae_flag PASSES iff it clears the fair null and separates from the anchors.

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: results/litext_gc_flag_physics_verdict.json (not shipped, see docs/REPRODUCE.md)
Outputs: results/litext_gc_flag_physics_verdict.json
Run:   # JAX env, CPU
    python -m graphcast_sae.obsgraph.physics_verdict --gint <gint.npy> --cands <pool.npy>
"""
import argparse, sys
from pathlib import Path
import numpy as np
from graphcast_sae.paths import REPO_ROOT as ROOT
from graphcast_sae.common.signature_physics import gc_km, sdlon, build_lags
BAND = (25.0, 40.0); NPERM = 20000; WIN = 0.35

def stc_edges(r):
    lat, lon = np.asarray(r["centroid_lat"]), np.asarray(r["centroid_lon"]); lagf = build_lags(r)
    edges = [(c, e, lagf(c, e)) for (c, e) in r["pair_edges"] if lagf(c, e) is not None]
    def ok(c, e, lg):
        v = gc_km(lat[c], lon[c], lat[e], lon[e]) * 1000 / (lg * 6 * 3600)
        return (BAND[0] <= v <= BAND[1] and sdlon(lon[c], lon[e]) > 0
                and abs(lat[c]) > 25 and abs(lat[e]) > 25 and np.sign(lat[c]) == np.sign(lat[e]))
    S = sum(ok(c, e, lg) for c, e, lg in edges)
    return edges, S, lat, lon, ok

def p_geo(r, edges, S, lat, lon, seed=0):
    """FREE null: permute mode->location labels, keep edges+lags."""
    rng = np.random.default_rng(seed); N = len(lat); ge = 0
    for _ in range(NPERM):
        p = rng.permutation(N); la, lo = lat[p], lon[p]; s = 0
        for c, e, lg in edges:
            v = gc_km(la[c], lo[c], la[e], lo[e]) * 1000 / (lg * 6 * 3600)
            if (BAND[0] <= v <= BAND[1] and sdlon(lo[c], lo[e]) > 0
                    and abs(la[c]) > 25 and abs(la[e]) > 25 and np.sign(la[c]) == np.sign(la[e])): s += 1
        ge += s >= S
    return (1 + ge) / (1 + NPERM)

def p_local(r, edges, S, lat, lon, seed=0):
    """FAIR null: hold each edge's LENGTH (+/-WIN), randomize only its target's direction."""
    rng = np.random.default_rng(seed); N = len(lat)
    D = np.array([[gc_km(lat[i], lon[i], lat[j], lon[j]) for j in range(N)] for i in range(N)])
    cand = []
    for c, e, lg in edges:
        L = D[c, e]; cs = [j for j in range(N) if j != c and (1-WIN)*L <= D[c, j] <= (1+WIN)*L]
        cand.append((c, lg, np.array(cs if cs else [e])))
    ge = 0
    for _ in range(NPERM):
        s = 0
        for c, lg, cs in cand:
            e2 = cs[rng.integers(len(cs))]
            v = gc_km(lat[c], lon[c], lat[e2], lon[e2]) * 1000 / (lg * 6 * 3600)
            if (BAND[0] <= v <= BAND[1] and sdlon(lon[c], lon[e2]) > 0
                    and abs(lat[c]) > 25 and abs(lat[e2]) > 25 and np.sign(lat[c]) == np.sign(lat[e2])): s += 1
        ge += s >= S
    return (1 + ge) / (1 + NPERM)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gint", required=True)
    ap.add_argument("--pos", default="leiden_flag"); ap.add_argument("--cand", default="sae_flag")
    ap.add_argument("--anchors", default="shift_flag,qperm_flag")
    args = ap.parse_args()
    anchors = args.anchors.split(",")
    g = np.load(args.gint, allow_pickle=True).item()
    G = g["results"]; nwin = g.get("nwin", "?"); n_done = g.get("n_done", "?")
    print(f"gint={Path(args.gint).name}  nwin={nwin}  n_done={n_done}\n")
    print(f"{'member':<12}{'|edges|':>8}{'STC':>5}{'p_geo(free)':>13}{'p_local(fair)':>15}   role")
    res = {}
    for m in G:
        r = G[m]; edges, S, lat, lon, _ = stc_edges(r)
        pg = p_geo(r, edges, S, lat, lon) if S > 0 else 1.0
        pl = p_local(r, edges, S, lat, lon) if S > 0 else 1.0
        role = "POS-CTRL" if m == args.pos else ("ANCHOR" if m in anchors else ("CANDIDATE" if m == args.cand else ""))
        res[m] = dict(n=len(r["pair_edges"]), S=S, p_geo=pg, p_local=pl, role=role)
        print(f"{m:<12}{len(r['pair_edges']):>8}{S:>5}{pg:>13.4f}{pl:>15.4f}   {role}")

    pos, cand = res[args.pos], res[args.cand]
    anch_sig = [a for a in anchors if res[a]["p_local"] < 0.05]
    print("\n=== VERDICT (fair, locality-controlled null) ===")
    if not (pos["S"] > 0 and pos["p_local"] < 0.05):
        print(f"INCONCLUSIVE — positive control {args.pos} shows no storm-track signal "
              f"(STC={pos['S']}, p_local={pos['p_local']:.3f}). Pipeline suspect; fix before trusting SAE.")
    elif anch_sig:
        print(f"INSTRUMENT FAILURE — anchor(s) {anch_sig} significant under the fair null. "
              f"Test too loose at this setting; do not read the SAE.")
    else:
        cand_sig = cand["p_local"] < 0.05
        print(f"positive control {args.pos}: STC={pos['S']} p_local={pos['p_local']:.4f} -> PASS (pipeline sound)")
        anc = [f"{a}:{res[a]['S']}/{res[a]['p_local']:.2f}" for a in anchors]
        print(f"anchors clean: {anc}")
        print(f"CANDIDATE {args.cand}: STC={cand['S']} p_local={cand['p_local']:.4f} -> "
              f"{'SAE ENCODES STORM-TRACK PHYSICS (positive)' if cand_sig else 'SAE physics-empty, sits with the controls (negative)'}")
    import json
    json.dump({m: {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in d.items()}
               for m, d in res.items()},
              open(str(ROOT / "results/litext_gc_flag_physics_verdict.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
