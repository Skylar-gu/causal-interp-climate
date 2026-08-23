"""PC alignment of the convection edit, mid-rollout, on the seven named storms.

`docs/notes/result_local_aggregate_2026_08_21.md` measured this on TC-located boxes from the IID
dump. Those are real cyclones but each is a single initial condition -- t = 0 of a forecast,
not +54 h with the clamp already running for nine steps. This runs the identical statistic on
`results/skill/actdump/act_<storm>.npz`, which carries the raw layer-8 activations inside the
box at every step of the baseline rollout.

Same statistics, same null, same pre-registered rule (docs/prereg/prereg_local_aggregate.md):

  align1(G)  fraction of the group's delete-displacement energy along PC1 of the in-disk
             activations at that step
  alignK(G)  cumulative over the top K = 5
  sub(G)     mean squared cosine of the principal angles between span{d_i} and the top-K PCs

Reported over each storm's most intense 24 h (the four consecutive steps with the largest
MSLP drop) and, separately, over the whole rollout, so the lead dependence is visible rather
than assumed. The disk is storm-following, recentred each step on the tracked centre.

Paper: Appendix app:mesh (local-aggregate geometry mid-rollout)
Inputs: results/skill/actdump (not shipped, see docs/REPRODUCE.md)
Outputs: results/skill/actdump/geom_midrollout.json
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.geom_midrollout
"""
import json
import sys
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
ACT = ROOT / "results/skill/actdump"
z = np.load(ROOT / "graphcast_sae/weights/sae_k32_lat4096_lay08.npz")
Wenc, bpre, Wdec = z["W_enc"], z["b_pre"], z["W_dec"]

TC = 3243
GROUPS = {
    "convection  (+2.79)": [2401, 2067, 3174],
    "asc21       (+3.63)": [553, 866, 1981],
    "asc17       (+0.02)": [3357, 1033, 3314],
    "moisture    (-0.03)": [2958, 2671, 37],
    "TC feature   (pos.)": [TC],
}
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020",
          "haiyan2013", "patricia2015", "wilma2005"]
K, NDRAW, RADIUS = 5, 200, 1500.0
rng = np.random.default_rng(0)

def codes(A, k=32):
    xn = A - A.mean(1, keepdims=True)
    xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre); r = np.arange(len(A))[:, None]
    out[r, idx] = pre[r, idx]
    return out

def align(f, G, V):
    D = -(f[:, G] @ Wdec[:, G].T)
    tot = float((D * D).sum())
    if tot <= 0:
        return None
    e = ((D @ V) ** 2).sum(0)
    return float(e[0] / tot), float(e[:K].sum() / tot)

def subspace(G, V):
    Q, _ = np.linalg.qr(Wdec[:, G])
    s = np.linalg.svd(Q.T @ V[:, :K], compute_uv=False)
    return float((s ** 2).mean())

def gc_km(lat, lon, la0, lo0):
    la, lo = np.radians(lat.astype(np.float64)), np.radians(np.mod(lon, 360).astype(np.float64))
    a0, o0 = np.radians(float(la0)), np.radians(np.mod(float(lo0), 360))
    return 6371.0 * np.arccos(np.clip(np.sin(a0) * np.sin(la)
                                      + np.cos(a0) * np.cos(la) * np.cos(lo - o0), -1, 1))

def main():
    acc = {g: {"a1": [], "aK": [], "sub": []} for g in GROUPS}
    acc["random (null)"] = {"a1": [], "aK": [], "sub": []}
    acc_all = {g: {"a1": [], "aK": []} for g in list(GROUPS) + ["random (null)"]}
    spec, per_storm = [], []

    for name in STORMS:
        p = ACT / f"act_{name}.npz"
        if not p.exists():
            print(f"[{name}] MISSING {p.name}, skipped", flush=True)
            continue
        d = np.load(p)
        A, blat, blon = d["act"], d["box_lat"], d["box_lon"]
        mslp, clat, clon = d["mslp_min"], d["clat"], d["clon"]
        Hn = A.shape[0]
        drops = [mslp[t] - mslp[t + 4] for t in range(Hn - 4)]
        t0 = int(np.argmax(drops))
        win = set(range(t0, t0 + 4))
        rows = []
        for t in range(Hn):
            m = gc_km(blat, blon, clat[t], clon[t]) <= RADIUS
            if m.sum() < 50:
                continue
            Ab = np.asarray(A[t][m], np.float32)
            Ac = Ab - Ab.mean(0, keepdims=True)
            _, Sv, Vt = np.linalg.svd(Ac, full_matrices=False)
            V = Vt.T
            lam = Sv ** 2
            f = codes(Ab)
            mass = f.sum(0)
            row = {}
            for g, G in GROUPS.items():
                r = align(f, G, V)
                if r is None:
                    continue
                row[g] = r
                acc_all[g]["a1"].append(r[0]); acc_all[g]["aK"].append(r[1])
                if t in win:
                    acc[g]["a1"].append(r[0]); acc[g]["aK"].append(r[1])
                    acc[g]["sub"].append(subspace(G, V))
            live = np.where(mass > 0)[0]
            order = live[np.argsort(mass[live])]
            base = GROUPS["convection  (+2.79)"]
            pos = {int(fi): int(np.searchsorted(mass[order], mass[fi])) for fi in base}
            for _ in range(NDRAW if t in win else 20):
                pick = []
                for fi, q in pos.items():
                    lo, hi = max(0, q - 40), min(len(order), q + 40)
                    cand = [c for c in order[lo:hi] if c not in pick and c != TC]
                    pick.append(int(rng.choice(cand)) if cand else int(rng.choice(order)))
                r = align(f, pick, V)
                if r is None:
                    continue
                acc_all["random (null)"]["a1"].append(r[0])
                acc_all["random (null)"]["aK"].append(r[1])
                if t in win:
                    acc["random (null)"]["a1"].append(r[0])
                    acc["random (null)"]["aK"].append(r[1])
                    acc["random (null)"]["sub"].append(subspace(pick, V))
            if t in win:
                spec.append((float(lam[0] / lam.sum()), float(lam[:K].sum() / lam.sum()),
                             float(lam.sum() ** 2 / (lam ** 2).sum())))
            rows.append((t, int(m.sum()), row))
        cw = [r[2].get("convection  (+2.79)") for r in rows if r[0] in win and "convection  (+2.79)" in r[2]]
        per_storm.append((name, t0, np.median([c[0] for c in cw]) if cw else float("nan"),
                          np.median([c[1] for c in cw]) if cw else float("nan")))
        print(f"[{name}] peak 24 h = +{t0*6}-{(t0+4)*6} h, drop {max(drops):.1f} hPa, "
              f"convection align PC1 {per_storm[-1][2]*100:.1f}%  top-5 {per_storm[-1][3]*100:.1f}%",
              flush=True)

    if not spec:
        print("no activation dumps found; run run_actdump.sh first")
        return
    s = np.array(spec)
    print(f"\nIn-disk spectrum during the intense window: PC1 {np.median(s[:,0])*100:.1f}%, "
          f"top-5 {np.median(s[:,1])*100:.1f}%, participation ratio {np.median(s[:,2]):.1f}")
    print(f"\n{'group':22s} {'align PC1':>18s} {'align top-5':>18s} {'subspace top-5':>18s}")
    for g in list(GROUPS) + ["random (null)"]:
        v = acc[g]
        if not v["a1"]:
            continue
        fm = lambda x: f"{np.median(x)*100:5.1f}% [{np.percentile(x,10)*100:4.1f},{np.percentile(x,90)*100:5.1f}]"
        print(f"{g:22s} {fm(v['a1']):>18s} {fm(v['aK']):>18s} {fm(v['sub']):>18s}")

    q90 = np.percentile(acc["random (null)"]["a1"], 90)
    q10 = np.percentile(acc["random (null)"]["a1"], 10)
    cm = np.median(acc["convection  (+2.79)"]["a1"])
    tm = np.median(acc["TC feature   (pos.)"]["a1"])
    verdict = ("LOCAL-AGGREGATE" if cm >= q90 and cm >= 0.5 * tm else
               "ORTHOGONAL" if q10 <= cm <= q90 else "NEITHER")
    print(f"\npre-registered rule: convection {cm*100:.2f}%, null q10 {q10*100:.2f}% "
          f"q90 {q90*100:.2f}%, TC {tm*100:.2f}% -> {verdict}")
    json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()} for k, v in acc.items()},
              open(ROOT / "results/skill/actdump/geom_midrollout.json", "w"))
    print("-> results/skill/actdump/geom_midrollout.json")

main()
