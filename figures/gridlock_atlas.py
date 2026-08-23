"""Every grid-lock candidate on a map, ordered by visual class, titled with its ABLATION COST.

The classes A-D were assigned by eye from an earlier unordered render of this same figure.
That is not laziness: mesh_bias reads refinement level only, face concentration is confounded
with geography, and fill completeness tracks footprint size, so all three fail to reproduce
the distinction (notes/result_detector_failure.md).

mesh_bias is deliberately NOT shown here. It is the selection statistic, and putting it in the
title invites reading it as the answer -- when for class C it is 5-6%, i.e. pure chance, on
features that draw unmistakable bowties. What belongs in the title is the thing actually being
tested: what removing that single feature does to the forecast.

Inputs: the i.i.d. dump $GC_SCRATCH/fs_iid_dump.npy + fs_iid_meta.json -- NOT shipped (6.7 GB); regenerate with `FS_DEVICE=gpu python -m graphcast_sae.extraction.extract_iid_dump --n 160`. results/fs_perfeat_rmse.npy and data/mesh_2to6_geom.npy are shipped.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from graphcast_sae.paths import SCRATCH, MESH_GEOM  # noqa: E402
WEIGHTS = ROOT / "graphcast_sae/weights"
NCOL = 4


def encode(A, Wenc, bpre, k=32):
    xn = A - A.mean(1, keepdims=True)
    xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre)
    r = np.arange(len(A))[:, None]
    out[r, idx] = pre[r, idx]
    return out


CLASSCOL = {"A": "#e35d4b", "B": "#e8a33d", "C": "#c15fd4", "D": "#3ba776"}
ORDER = ["A_sparse_lattice", "B_full_globe_lattice",
         "C_geometric_shape", "D_lattice_textured_weather"]


def load_effects(field="z500"):
    """(mean change in global RMSE per feature, floor sd) or (None, None) if not run yet."""
    from scipy import stats                                    # noqa: F401
    f = ROOT / "results/fs_perfeat_rmse.npy"
    if not f.exists():
        return None, None
    d = np.load(f, allow_pickle=True).item()
    acc, F, arms = d["acc"], d["fields"], d["arms"]
    if "baseline" not in acc or np.isnan(acc["baseline"][:, -1, F.index(field)]).all():
        return None, None
    fi = F.index(field)
    B = acc["baseline"][:, -1, fi]
    eff = {}
    for a in arms:
        if not a.startswith("f") or a == "floor":
            continue
        v = acc[a][:, -1, fi] - B
        if not np.isnan(v).all():
            eff[int(a[1:])] = float(np.nanmean(v))
    fl = None
    if "floor" in acc:
        fv = acc["floor"][:, -1, fi] - B
        if not np.isnan(fv).all():
            fl = float(np.nanstd(fv, ddof=1))
    return eff, fl


def main():
    TYPES = json.load(open("/tmp/gridlock_types.json"))
    eff, floor_sd = load_effects()
    feats, cls = [], {}
    for key in ORDER:
        tag = key[0]
        block = TYPES[key]
        if eff:                       # within a class, worst-first
            block = sorted(block, key=lambda f: -eff.get(f, 0.0))
        for f in block:
            feats.append(f); cls[f] = tag
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    # accumulate over several windows: one snapshot can under-show a sparse feature
    sel = np.linspace(0, META["n_windows"] - 1, 6).astype(int)
    acc = None
    for j in sel:
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        C = encode(A, Wenc, bpre)
        acc = C if acc is None else acc + C
    acc /= len(sel)

    g = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.asarray(g["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    nrow = int(np.ceil(len(feats) / NCOL))
    fig = plt.figure(figsize=(4.6 * NCOL, 2.5 * nrow))
    fig.patch.set_facecolor("#0b0d10")
    for i, f in enumerate(feats):
        ax = fig.add_subplot(nrow, NCOL, i + 1, projection=ccrs.PlateCarree())
        ax.set_global(); ax.set_facecolor("#0b0d10")
        ax.coastlines(resolution="110m", linewidth=0.3, color="#4b5560")
        col = CLASSCOL[cls[f]]
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.6)
        v = acc[:, f]; on = v > 0
        if on.sum():
            ax.scatter(mlon[on], mlat[on], c=v[on], s=1.8, cmap="magma",
                       vmin=0, vmax=np.percentile(v[on], 99),
                       transform=ccrs.PlateCarree(), linewidths=0)
        if eff is None:
            score = "ablation pending"
        else:
            e = eff.get(f)
            score = "not run" if e is None else (
                f"{e:+.3f} m" + ("  (within noise)" if floor_sd and abs(e) < 2 * floor_sd
                                 else ""))
        ax.set_title(f"{cls[f]} · f{f}    {score}", fontsize=9.5, color=col, pad=3)
    note = ("titles show the change in global 500 hPa forecast error at +48 h from ablating "
            "that ONE feature — positive = the forecast got WORSE"
            if eff else "ablation running — titles fill in when it lands")
    fig.suptitle("Grid-lock candidates by visual class   "
                 "A sparse lattice · B full-globe lattice · C geometric shape · "
                 "D lattice-textured weather\n" + note,
                 color="#e6edf3", fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = ROOT / "figures/gridlock_atlas.png"
    fig.savefig(p, dpi=118, facecolor=fig.get_facecolor())
    print(f"-> {p}   ({len(feats)} features)")


if __name__ == "__main__":
    main()
