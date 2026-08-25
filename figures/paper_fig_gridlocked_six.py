"""The six grid-locked features (stay on the mesh under rotation at all five angles tested:
45/90/135/180/270 deg; notes/rotation_angles_2026_08_25.md in causal-graphcast), drawn exactly
as paper_fig_gridlocked_nodes.py draws its panels: nearest-node raster of the footprint nodes,
shade = mean accumulated amplitude, green->blue ramp, white ground. Titles are the feature id
only. 3 rows x 2 columns. Features firing on fewer than 300 nodes (f407, f2989: the coarsest
icosahedron vertices) are drawn as node markers, since a nearest-node raster reduces them to specks.

With --controls, a fourth row adds the two content controls (f3404 land-shaped, f2109 tropical
convection) in the pink/purple ramp of paper_fig_pair_2954_2109.py.

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_gridlocked_six.py [--controls]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in __import__("glob").glob(__import__("os").path.expanduser("~/.fonts/IBMPlexSans-*.ttf")):
    _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "IBM Plex Sans"   # the clean-paper figures' face
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/home/ec2-user/causal-graphcast")
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK  # noqa: E402

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
GREEN_BLUE = LinearSegmentedColormap.from_list(
    "green_blue_node", ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])
PINK_PURPLE = LinearSegmentedColormap.from_list(
    "pink_purple_node", ["#ec6fb8", "#d63aa5", "#b3126c", "#7b1a7a", "#4a1670", "#1c0a3a"])

SIX = [2954, 2075, 407, 2585, 2989, 3535]
CONTROLS = [3404, 2109]
WITH_CONTROLS = "--controls" in sys.argv


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    acc = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")["acc"]

    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    def img_fires(f):
        amp = acc[:, f].astype(float); amp = amp / max(amp.max(), 1e-9)
        return np.where(fires[:, f], amp, np.nan)[nn]

    def nodes_of(f):
        """(lon, lat, amp) of the firing nodes -- used for features too sparse to raster."""
        amp = acc[:, f].astype(float); amp = amp / max(amp.max(), 1e-9)
        on = fires[:, f]
        return lon[on], lat[on], amp[on]
    SPARSE = 300      # firing nodes below this are drawn as markers, not a nearest-node raster

    panels = [(f, img_fires(f), GREEN_BLUE) for f in SIX]
    if WITH_CONTROLS:
        from paper_fig_weatherlike_nodes import atlas_activation, FEATS as WL_FEATS  # noqa
        panels.append((3404, img_fires(3404), PINK_PURPLE))
        acc6 = atlas_activation(WL_FEATS)
        a = acc6[:, WL_FEATS.index(2109)].astype(float); on = a > 0
        a = np.clip(a / max(np.percentile(a[on], 99), 1e-9), 0, 1)
        panels.append((2109, np.where(on, a, np.nan)[nn], PINK_PURPLE))
    nrow = len(panels) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(9.2, 2.45 * nrow), facecolor=BG,
                             subplot_kw=dict(projection=PC))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01, wspace=0.04, hspace=0.22)
    for ax, (f, img, cmap) in zip(axes.ravel(), panels):
        if f in SIX and int(fires[:, f].sum()) < SPARSE:
            lo, la, am = nodes_of(f)
            ax.scatter(lo, la, c=am ** 0.7, cmap=cmap, vmin=0, vmax=1, s=26, marker="o",
                       edgecolors=INK, linewidths=0.35, transform=PC, zorder=5)
            print(f"f{f}: drawn as {len(lo)} node markers")
        else:
            ax.imshow(np.ma.masked_invalid(img) ** 0.7, origin="lower",
                      extent=[-180, 180, -90, 90], cmap=cmap, vmin=0, vmax=1,
                      transform=PC, interpolation="nearest")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        ax.set_title(f"Feature {f}", fontsize=9.6, color=INK, weight="bold", loc="left", pad=4)
        print(f"f{f}: {int(np.isfinite(img).sum())} raster cells")
    stem = "paper_fig_gridlocked_six" + ("_controls" if WITH_CONTROLS else "")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/{stem}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    print(f"-> figures/{stem}.pdf / .png")


if __name__ == "__main__":
    main()
