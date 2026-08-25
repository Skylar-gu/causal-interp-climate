"""f2954 (grid-locked, green/blue) beside f2109 (weather-like, pink/purple), one row.

Each panel is lifted verbatim from its current generating script:
  left  = paper_fig_gridlock.py panel (a): firing nodes (>= 2 of 8 IID windows) from
          results/hybrid_footprint_fires.npz, shade = accumulated amplitude
          (data/fs_footprint_fires_nw12.npz), green->blue NODE_CMAP, gamma 0.7.
  right = paper_fig_weatherlike_nodes.py panel (c): all firing nodes of the 6-window atlas
          mean code (cached results/fs_atlas_acc6_*.npy), saturating at p99, pink/purple
          ramp, gamma 0.7.
Both use the same nearest-node raster, coastline, and frame. Titles carry each feature's
grid-lock score and +48 h ablation cost from results/fs_matched_rmse.npy.

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_pair_2954_2109.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/home/ec2-user/causal-graphcast")
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED  # noqa: E402
from paper_fig_weatherlike_nodes import atlas_activation, FEATS as WL_FEATS  # noqa: E402

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
GREEN_BLUE = LinearSegmentedColormap.from_list(
    "green_blue_node",
    ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])
PINK_PURPLE = LinearSegmentedColormap.from_list(
    "pink_purple_node",
    ["#ec6fb8", "#d63aa5", "#b3126c", "#7b1a7a", "#4a1670", "#1c0a3a"])


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)

    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    # ---- left: f2954 exactly as paper_fig_gridlock.py draws it ----------------------
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    acc12 = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")["acc"]
    amp = acc12[:, 2954].astype(float); amp = amp / max(amp.max(), 1e-9)
    left = np.where(fires[:, 2954], amp, np.nan)[nn]

    # ---- right: f2109 exactly as paper_fig_weatherlike_nodes.py draws it -------------
    acc6 = atlas_activation(WL_FEATS)
    a = acc6[:, WL_FEATS.index(2109)].astype(float)
    on = a > 0
    a = np.clip(a / max(np.percentile(a[on], 99), 1e-9), 0, 1)
    right = np.where(on, a, np.nan)[nn]

    # ---- scores and costs for the subtitles -----------------------------------------
    score = np.asarray(np.load(ROOT / "results/fs_gridlock_all.npy",
                               allow_pickle=True).item()["score"], float)
    M = np.load(MAIN / "results/fs_matched_rmse.npy", allow_pickle=True).item()
    A, iz, Lz = M["acc"], M["fields"].index("z500"), M["S"] - 1
    base = np.asarray(A["baseline"])[:, Lz, iz]

    def cost(f):
        df = (np.asarray(A[f"f{f}"])[:, Lz, iz] - base).mean()
        return df, 100 * df / base.mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9), facecolor=BG,
                             subplot_kw=dict(projection=PC))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.05)
    panels = [(axes[0], left, GREEN_BLUE, 2954, "a", "f2954 — full lattice (grid-locked)"),
              (axes[1], right, PINK_PURPLE, 2109, "b", "f2109 — tropical convection belt")]
    for ax, img, cmap, f, tag, title in panels:
        ax.imshow(np.ma.masked_invalid(img) ** 0.7, origin="lower",
                  extent=[-180, 180, -90, 90], cmap=cmap, vmin=0, vmax=1, transform=PC,
                  interpolation="nearest")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        df, pct = cost(f)
        ax.set_title(f"{tag}   {title}", fontsize=9.6, color=INK, weight="bold", loc="left",
                     pad=24)
        ax.annotate(f"grid-lock score {score[f]:.2f}\n"
                    f"ablation {df:+.2f} m z500 at +48 h ({pct:.1f}% of error)",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=7.6, color=MUTED, ha="left",
                    va="bottom", annotation_clip=False)

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_pair_2954_2109.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG, bbox_inches="tight",
                    pad_inches=0.05)
    print("-> figures/paper_fig_pair_2954_2109.png / .pdf")


if __name__ == "__main__":
    main()
