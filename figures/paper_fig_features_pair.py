"""Grid-locked features (left) beside physical features (right), 2x2 each.

Left block reuses the renderer of paper_fig_gridlock.py row 1: footprint nodes (fires in
>= 2 of 8 IID windows) shaded by accumulated amplitude, drawn as DISCRETE MESH NODES via a
nearest-node raster, green->blue on white. Right block reuses paper_fig_graphmap.py panel
(a): accumulated activation over 12 dates, linearly gridded and drawn as a smooth field
with the same colour ramp and PowerNorm gamma. Nothing is re-derived; only the layout is new.

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_features_pair.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.interpolate import griddata
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
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "green_blue_field",
    ["#ffffff", "#e4f3e1", "#a8d9b5", "#5cb9a6", "#2f8fae", "#22588a"])
FIELD_GAMMA = 0.28

GRIDLOCKED = [(2954, "f2954 — full lattice"),
              (3404, "f3404 — its matched control"),
              (2075, "f2075 — bowties"),
              (656,  "f656 — lattice dots")]
PHYSICAL = [(3243, "f3243 — tropical cyclone"),
            (2401, "f2401 — convection"),
            (3861, "f3861 — low-level vorticity"),
            (553,  "f553 — ascent")]


def unit(la, lo):
    a, o = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)


def frame(ax, title):
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
    ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=9, color=INK, pad=4)


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    acc = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")["acc"]

    # nearest-node raster (grid-locked block), as in paper_fig_gridlock.py
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    # linear grid (physical block), as in paper_fig_graphmap.py
    def gridmap(vals, step=1.2):
        glon = np.arange(-180, 180 + .01, step); glat = np.arange(-90, 90 + .01, step)
        G = griddata((lon, lat), vals, tuple(np.meshgrid(glon, glat)), method="linear")
        return np.nan_to_num(G, nan=0.0)

    fig = plt.figure(figsize=(12.4, 3.9), facecolor=BG)
    outer = gridspec.GridSpec(1, 2, wspace=0.10, left=0.01, right=0.99, top=0.84, bottom=0.02)
    gl = outer[0].subgridspec(2, 2, wspace=0.05, hspace=0.30)
    gr = outer[1].subgridspec(2, 2, wspace=0.05, hspace=0.30)

    for j, (fid, title) in enumerate(GRIDLOCKED):
        ax = fig.add_subplot(gl[j // 2, j % 2], projection=PC)
        amp = acc[:, fid].astype(float); amp = amp / max(amp.max(), 1e-9)
        v = np.where(fires[:, fid], amp, np.nan)[nn]
        ax.imshow(np.ma.masked_invalid(v) ** 0.7, origin="lower",
                  extent=[-180, 180, -90, 90], cmap=NODE_CMAP, vmin=0, vmax=1,
                  transform=PC, interpolation="nearest")
        frame(ax, title)

    for j, (fid, title) in enumerate(PHYSICAL):
        ax = fig.add_subplot(gr[j // 2, j % 2], projection=PC)
        G = gridmap(acc[:, fid])
        ax.imshow(G, origin="lower", extent=[-180, 180, -90, 90], cmap=FIELD_CMAP,
                  norm=PowerNorm(FIELD_GAMMA), transform=PC, interpolation="bilinear")
        frame(ax, title)

    fig.text(0.01, 0.94, "a   Grid-locked features", fontsize=11, color=INK, weight="bold")
    fig.text(0.01, 0.905, "activation locked to the icosahedral mesh; drawn as discrete mesh nodes",
             fontsize=8, color=MUTED)
    fig.text(0.535, 0.94, "b   Physical features", fontsize=11, color=INK, weight="bold")
    fig.text(0.535, 0.905, "activation follows the atmosphere; drawn as a smooth field",
             fontsize=8, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_features_pair.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG, bbox_inches="tight")
    print("-> figures/paper_fig_features_pair.png / .pdf")


if __name__ == "__main__":
    main()
