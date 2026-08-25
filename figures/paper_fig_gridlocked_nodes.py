"""The four grid-locked features from the top row of paper_fig_gridlock.py, as their own
2x2 PDF. Rendering is identical to that figure (nearest-node raster of the footprint nodes,
shade = mean accumulated amplitude, green/blue ramp, white ground); only the labels differ:

    f<id> — <pattern>
    grid-lock score <s>
    ablation +<d> m z500 at +48 h vs matched control (<p>% of error)
      (feature minus its coverage/connectivity-matched control, the quantity the appendix
       and paper_fig_gridlock.py report; f3404 is itself a control arm, so baseline-relative)

Inputs: data/mesh_2to6_geom.npy, data/fs_footprint_fires_nw12.npz (acc),
../causal-graphcast/results/hybrid_footprint_fires.npz (footprint nodes),
results/fs_gridlock_all.npy, ../causal-graphcast/results/fs_matched_rmse.npy.

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_gridlocked_nodes.py
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

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
NODE_CMAP = LinearSegmentedColormap.from_list(
    "green_blue_node",
    ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])

SHOW = [(2954, "full-globe lattice"),
        (3404, "land-locked scatter"),
        (2075, "bowties"),
        (656,  "lattice dots")]


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    acc = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")["acc"]
    score = np.asarray(np.load(ROOT / "results/fs_gridlock_all.npy",
                               allow_pickle=True).item()["score"], float)

    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    M = np.load(MAIN / "results/fs_matched_rmse.npy", allow_pickle=True).item()
    A, iz, Lz = M["acc"], M["fields"].index("z500"), M["S"] - 1
    base = np.asarray(A["baseline"])[:, Lz, iz]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), facecolor=BG,
                             subplot_kw=dict(projection=PC))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04, hspace=0.42)
    for ax, (f, name), tag in zip(axes.ravel(), SHOW, "abcd"):
        amp = acc[:, f].astype(float)
        amp = amp / max(amp.max(), 1e-9)
        v = np.where(fires[:, f], amp, np.nan)
        ax.imshow(np.ma.masked_invalid(v[nn]) ** 0.7, origin="lower",
                  extent=[-180, 180, -90, 90], cmap=NODE_CMAP, vmin=0, vmax=1,
                  transform=PC, interpolation="nearest")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        arm = next(k for k, g in M["groups"].items() if list(g) == [f])   # f3404 was run as ctrl_f2954
        df = (np.asarray(A[arm])[:, Lz, iz] - base).mean()
        if f"ctrl_{arm}" in A:      # feature MINUS its matched control, as paper_fig_gridlock.py
            df -= (np.asarray(A[f"ctrl_{arm}"])[:, Lz, iz] - base).mean()   # and the appendix report
        df = 0.0 if abs(df) < 0.005 else df
        ax.set_title(f"{tag}   f{f} — {name}", fontsize=9.6, color=INK, weight="bold",
                     loc="left", pad=24)
        ax.annotate(f"grid-lock score {score[f]:.2f}\n"
                    f"ablation {df:+.2f} m z500 at +48 h vs matched control "
                    f"({100 * df / base.mean():.1f}% of error)",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=7.6, color=MUTED, ha="left",
                    va="bottom", annotation_clip=False)
        print(f"f{f}: score {score[f]:.3f}  ablation {df:+.3f} m ({100 * df / base.mean():.1f}%)")

    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/paper_fig_gridlocked_nodes.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG, bbox_inches="tight",
                    pad_inches=0.05)
    print("-> figures/paper_fig_gridlocked_nodes.pdf / .png")


if __name__ == "__main__":
    main()
