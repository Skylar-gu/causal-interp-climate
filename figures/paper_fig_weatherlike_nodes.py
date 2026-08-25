"""Four weather-like (NOT grid-locked) SAE features as discrete mesh nodes, 2x2, own PDF.

Same rendering as the top row of paper_fig_gridlock.py (nearest-node raster of the
feature's firing nodes, shade = mean amplitude), but a pink/purple ramp instead of the
green/blue one, and no axes frames. Each panel is titled with the feature's grid-lock
score (mean Jaccard self-overlap of the active node set across 12 dates; grid-locked
features sit at ~0.45+, weather features at 0.01-0.12) and the forecast cost of ablating
it: extra z500 RMSE at +48 h against the paired baseline over 6 ICs, with its
coverage- and connectivity-matched control alongside.

Inputs: data/mesh_2to6_geom.npy, the i.i.d. layer-8 dump (atlas-style mean code over
6 windows, all firing nodes, shade saturating at the 99th percentile -- same selection as
figures/gridlock_atlas.py; cached in results/),
results/fs_gridlock_all.npy, ../causal-graphcast/results/fs_matched_rmse.npy.

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_weatherlike_nodes.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/home/ec2-user/causal-graphcast")
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED  # noqa: E402,F401

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
NODE_CMAP = LinearSegmentedColormap.from_list(
    "pink_purple_node",
    ["#ec6fb8", "#d63aa5", "#b3126c", "#7b1a7a", "#4a1670", "#1c0a3a"])   # starts at a mid pink so weak nodes read on white

FEATS = [2014, 1089, 2109, 2789]
NAMES = {2014: "scattered storm cells", 1089: "land-surface scatter",
         2109: "tropical convection belt", 2789: "Maritime Continent convection"}


def atlas_activation(feats, nwin=6):
    """Mean TopK code per mesh node over `nwin` windows of the i.i.d. layer-8 dump, with
    the authors' per-token centring + L2 normalisation -- exactly what gridlock_atlas.py
    renders. Cached to results/fs_atlas_acc6_<feats>.npy so the 6.7 GB dump is read once."""
    import json
    tag = "_".join(map(str, feats))
    cache = ROOT / f"results/fs_atlas_acc6_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    SCRATCH = Path("/home/ec2-user/gc_flagship")
    z = np.load(ROOT / "flagship_sae/weights/sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    sel = np.linspace(0, META["n_windows"] - 1, nwin).astype(int)
    acc = np.zeros((L, len(feats)), np.float32)
    for j in sel:
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        thr = -np.partition(-pre, 32, axis=1)[:, 32:33]        # TopK-32 per token
        code = np.where(pre >= thr, pre, 0.0)
        acc += code[:, feats]
    acc /= len(sel)
    np.save(cache, acc)
    return acc


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)
    acc = atlas_activation(FEATS)          # (L, len(FEATS)) mean code over 6 dump windows
    score = np.asarray(np.load(ROOT / "results/fs_gridlock_all.npy",
                               allow_pickle=True).item()["score"], float)

    # nearest-node raster (identical to paper_fig_gridlock.py)
    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    # ablation cost at +48 h vs paired baseline, with matched control
    M = np.load(MAIN / "results/fs_matched_rmse.npy", allow_pickle=True).item()
    A, iz, Lz = M["acc"], M["fields"].index("z500"), M["S"] - 1
    base = np.asarray(A["baseline"])[:, Lz, iz]
    cost = {}
    for f in FEATS:
        df = np.asarray(A[f"f{f}"])[:, Lz, iz] - base
        dc = np.asarray(A[f"ctrl_f{f}"])[:, Lz, iz] - base
        cost[f] = dict(df=df.mean(), dc=dc.mean(), p=stats.ttest_rel(df, dc).pvalue,
                       pct=100 * df.mean() / base.mean())

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), facecolor=BG,
                             subplot_kw=dict(projection=PC))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04, hspace=0.42)
    for ax, f, tag in zip(axes.ravel(), FEATS, "abcd"):
        amp = acc[:, FEATS.index(f)].astype(float)
        on = amp > 0                                   # every node the feature fires on
        amp = np.clip(amp / max(np.percentile(amp[on], 99), 1e-9), 0, 1)   # saturate at p99, as the atlas does
        v = np.where(on, amp, np.nan)
        ax.imshow(np.ma.masked_invalid(v[nn]) ** 0.7, origin="lower",
                  extent=[-180, 180, -90, 90], cmap=NODE_CMAP, vmin=0, vmax=1,
                  transform=PC, interpolation="nearest")
        print(f"f{f}: {on.sum()} firing nodes")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)   # thin frame, as in the grid-locked figure
        c = cost[f]
        c = {k: (0.0 if isinstance(v, float) and abs(v) < 0.005 and k in ('df', 'dc') else v) for k, v in c.items()}
        ax.set_title(f"{tag}   f{f} — {NAMES[f]}", fontsize=9.6, color=INK, weight="bold",
                     loc="left", pad=24)
        ax.annotate(f"grid-lock score {score[f]:.2f}\n"
                    f"ablation {c['df']:+.2f} m z500 at +48 h ({c['pct']:.1f}% of error)",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=7.6, color=MUTED, ha="left",
                    va="bottom", annotation_clip=False)
        print(f"f{f}: score {score[f]:.3f}  +48h z500 +{c['df']:.3f} m ({c['pct']:.1f}%)  "
              f"ctrl +{c['dc']:.3f} m  p={c['p']:.2f}")

    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/paper_fig_weatherlike_nodes.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG, bbox_inches="tight",
                    pad_inches=0.05)
    print("-> figures/paper_fig_weatherlike_nodes.pdf / .png")


if __name__ == "__main__":
    main()
