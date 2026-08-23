"""What is the convection-vs-control contrast actually measuring?

THE WORRY. The treatment ablates three features that fire in the storm core; every control
drawn so far fires ~0% there at +48 h. For a storm-intensity readout that contrast is close
to tautological -- anything absent from the core cannot move the core's pressure -- so it may
be measuring LOCALITY rather than MECHANISM IDENTITY.

THE TEST. Sixteen concept groups have been run on the same seven storms with the same
restore-to-normal arm. Each run's snapshot stores its own group's node activation at +48 h,
so both quantities are already on disk: how much the group fires inside 300 km of the storm
centre, and how much deepening its ablation costs. If core firing predicts the effect, the
contrast is mostly locality. If groups with equal core firing give very different effects,
identity matters on top of it.

Paper: Appendix app:taxonomy, shape 1 (does core firing predict the effect)
Inputs: results/skill/convection (shipped)
Outputs: printed report
Run:   # demo env (numpy/scipy/matplotlib)
    FS_DEVICE=cpu python -m graphcast_sae.storms.core_firing_vs_effect
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

import graphcast_sae.common.skill_conv_storms as S

R = 6371.0
CORE_KM = 300.0

def gc_km(la, lo, la0, lo0):
    d = (np.asarray(lo, float) - lo0 + 180) % 360 - 180
    return R * np.arccos(np.clip(
        np.sin(np.deg2rad(la)) * np.sin(np.deg2rad(la0)) +
        np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(la0)) * np.cos(np.deg2rad(d)), -1, 1))

def scan(d):
    """(median % deepening lost, median core activation, median core share, n storms)."""
    loss, core, share = [], [], []
    for f in sorted(glob.glob(f"{d}/run_*.npy")):
        st = os.path.basename(f)[4:-4]
        if S.STORMS.get(st, {}).get("nondev"):
            continue
        r = np.load(f, allow_pickle=True).item()
        res = r["res"]
        if "baseline" in res and "conv-normal" in res:
            b = np.asarray(res["baseline"]["mslp_min"])
            a = np.asarray(res["conv-normal"]["mslp_min"])
            db, da = b[0] - b.min(), a[0] - a.min()
            if db > 1:
                loss.append(100 * (db - da) / db)
        sn = r.get("snap", {}).get("baseline_mid")
        if not sn or "node_conv" not in sn:
            continue
        box = S.STORMS[st]["box"]
        g = np.asarray(sn["mslp_grid"], float)
        glat, glon = np.asarray(sn["mslp_lat"], float), np.asarray(sn["mslp_lon"], float)
        j, i = np.unravel_index(int(np.nanargmin(g)), g.shape)
        mlat, mlon = np.asarray(sn["mlat"], float), np.asarray(sn["mlon"], float)
        inbox = ((mlat >= box["lat"][0]) & (mlat <= box["lat"][1]) &
                 (mlon >= box["lon"][0]) & (mlon <= box["lon"][1]))
        dist = gc_km(mlat, np.where(mlon < 0, mlon + 360, mlon), glat[j], glon[i])
        v = np.asarray(sn["node_conv"], float)
        v = v.sum(1) if v.ndim == 2 else v
        v = np.where(inbox, v, 0.0)
        core.append(v[dist < CORE_KM].sum())
        share.append(100 * v[dist < CORE_KM].sum() / v.sum() if v.sum() > 0 else 0.0)
    med = lambda x: float(np.median(x)) if x else np.nan
    return med(loss), med(core), med(share), len(loss)

def main():
    dirs = ([d for d in glob.glob("results/skill/mech_*") if "CONTAM" not in d] +
            glob.glob("results/skill/moisture*") + ["results/skill/convection"])
    rows = []
    for d in dirs:
        L, C, Sh, n = scan(d)
        if n and not np.isnan(L):
            rows.append((os.path.basename(d), L, C, Sh, n))
    rows.sort(key=lambda r: -r[1])
    print(f"{'group':<20}{'% deepening lost':>18}{'core activation':>17}"
          f"{'core share':>12}{'effect/core':>13}{'n':>4}")
    for k, L, C, Sh, n in rows:
        eff = L / C if C > 0 else np.nan
        print(f"{k:<20}{L:>17.1f}%{C:>17.1f}{Sh:>11.1f}%{eff:>13.2f}{n:>4}")
    L = np.array([r[1] for r in rows]); C = np.array([r[2] for r in rows])
    rho, p = stats.spearmanr(L, C)
    print(f"\nspearman(effect, core activation) = {rho:+.3f}  p = {p:.5f}  n = {len(rows)}")
    nz = C > 0
    print(f"among groups that DO fire in the core (n={nz.sum()}), effect per unit of core "
          f"activation spans {np.nanmin(L[nz]/C[nz]):.2f} to {np.nanmax(L[nz]/C[nz]):.2f} "
          f"-- a {np.nanmax(L[nz]/C[nz])/max(np.nanmin(L[nz]/C[nz]),1e-9):.0f}x range, "
          f"so locality does not explain everything.")

if __name__ == "__main__":
    main()
