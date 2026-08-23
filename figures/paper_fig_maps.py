"""The footprint map panel, rendered as maps rather than as scatter clouds.

Conventions follow MacMillan & Ouellette (arXiv:2512.24440) and the repo's own
figures/meshmap.py: equirectangular (PlateCarree) with Natural Earth 50 m coastlines;
grid-locked features drawn as DISCRETE MESH NODES, because smoothing is exactly what
erases the node-level contrast that defines grid-lock; physical features drawn as a
smooth field, because that is what they are. Theme matched to the Ida panels of Figure 1:
near-black ocean, pale coastlines, magma for activation.

Needs cartopy -> python figures/paper_fig_maps.py
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
PC = ccrs.PlateCarree()
# Two layouts from one script. The paper gets a 2x2 in a 5.25 in column. A web page renders
# the same block at ~664 CSS px, where a 2x2 leaves each world map 312 px across and the
# node structure is invisible no matter what the source resolution is -- so there it gets a
# single column, and each map is the full width.
TALL = os.environ.get("MAPS_TALL") == "1"
W = 6.64 if TALL else 5.25

# Two palettes: the paper keeps its plum-black; the web page uses the cool blue-black of
# the companion artifact, so the figure sits flush on the page instead of on a slab.
if os.environ.get("MAPS_TALL") == "1":
    BG, OCEAN, COAST = "#0a0e15", "#070a10", "#7d93aa"
    INK, MUTED, FAINT = "#eaf0f6", "#98a8ba", "#5e6f81"
    EMBER, GRIDC = "#ff8347", "#1e2937"
else:
    # Paper: monochrome on white. Feature geometry is categorical -- where a feature fires --
    # so hue carries nothing here, and black on white survives grayscale printing.
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from paper_palette import INK as _I, MUTED as _M, FAINT as _F, GRIDC as _G, YELLOW as _Y
    BG, OCEAN, COAST = "#ffffff", "#fbfbfb", "#a8a8a8"
    INK, MUTED, FAINT = _I, _M, _F
    # warm for the model's own scaffolding, cool for the atmosphere it encodes
    EMBER, GRIDC = _Y, _G
if TALL:
    FIELD_CMAP = "magma"
    NODE_CMAP = "magma"
else:
    # One cold family for all four panels, because all four now show the same quantity:
    # mean activation amplitude per mesh node. The smooth panel starts at white so the
    # page shows through; the node panels start at a visible tint, since an isolated
    # near-white square on white is not a mark.
    # Light green -> medium blue. Two hues, not one: a single-hue blue ramp separates the
    # low end from the high end by lightness alone, which is exactly the difference a small
    # printed mark loses. Crossing hue as well makes the amplitude gradient legible at
    # figure size.
    FIELD_CMAP = LinearSegmentedColormap.from_list(
        "green_blue_field",
        ["#ffffff", "#e4f3e1", "#a8d9b5", "#5cb9a6", "#2f8fae", "#22588a"])
    NODE_CMAP = LinearSegmentedColormap.from_list(
        "green_blue_node",
        ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])
# on white, weak activity is pale rather than dark, so it needs a stronger gamma
FIELD_GAMMA = 0.5 if TALL else 0.28
TITLE, SUB, NOTE = 7.6, 6.2, 5.9
_SC = 1.0   # set after the layout is known

# ---- mesh geometry and the two footprint stores -----------------------------------
fp = np.load(ROOT / "results/fs_footprints.npy", allow_pickle=True).item()
xt = np.load(ROOT / "results/fs_footprints_extra.npy", allow_pickle=True).item()
LAT = np.asarray(fp["lat"], float)
LON = np.where(np.asarray(fp["lon"], float) > 180, np.asarray(fp["lon"], float) - 360,
               np.asarray(fp["lon"], float))
L = len(LAT)


def nodes_of(fid):
    if fid in fp["res"]:
        return np.asarray(fp["res"][fid]["nodes"], int)
    return np.asarray(xt["res"][fid]["nodes"], int)


AMBER = LinearSegmentedColormap.from_list(
    "amber", ["#40230f", "#a3521f", EMBER, "#ffcfae"])


def counts_of(fid):
    """How often the feature fired at each node, as a fraction of the windows sampled."""
    if fid in xt["res"]:
        c = np.asarray(xt["res"][fid]["count"], float)
        return c / max(c.max(), 1.0)
    w = np.zeros(L)
    w[nodes_of(fid)] = 1.0
    return w


def strength_of(fid):
    """Per-node mean activation over the 8 windows, normalised to 0..1.

    Amplitude, not an on/off count: a convection feature fires weakly over a lot of the
    tropics and hard in a few places, and the count throws that away.
    """
    if fid in xt["res"]:
        a = np.asarray(xt["res"][fid]["mean_amp"], float)
        return a / max(a.max(), 1e-9)
    w = np.zeros(L)
    w[nodes_of(fid)] = 1.0
    return w


# ---- mesh -> 0.5 deg grid, inverse distance over 4 great-circle neighbours ---------
GLON = np.arange(-180.0, 180.0, 0.5) + 0.25
GLAT = np.arange(-89.75, 90.0, 0.5)
_MG2, _MG1 = np.meshgrid(GLON, GLAT)


def _unit(la, lo):
    a, o = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)


_TREE = cKDTree(_unit(LAT, LON))
_D, _I = _TREE.query(_unit(_MG1.ravel(), _MG2.ravel()), k=8)


def to_grid(vals, power=1.4):
    """Smooth field, for a feature that tracks a physical quantity."""
    w = 1.0 / np.maximum(_D, 1e-9) ** power
    return ((np.asarray(vals)[_I] * w).sum(1) / w.sum(1)).reshape(_MG1.shape)


# ---- fine grid + nearest-node rasterisation, for the grid-locked features ----------
# Drawing these as scatter markers makes the rendering depend on marker size rather than on
# the data: too small and the pattern breaks up, too large and neighbouring nodes merge. A
# nearest-node raster is resolution-independent and, unlike interpolation, preserves the
# node-level structure exactly -- each mesh node simply becomes the cells closest to it,
# which is what the mesh actually is.
FLON = np.arange(-180.0, 180.0, 0.25) + 0.125
FLAT = np.arange(-89.875, 90.0, 0.25)
_FG2, _FG1 = np.meshgrid(FLON, FLAT)
_, _NN = _TREE.query(_unit(_FG1.ravel(), _FG2.ravel()), k=1)
_NN = _NN.reshape(_FG1.shape)


def to_nodes(vals):
    return np.asarray(vals)[_NN]


def worldmap(ax, title, sub):
    ax.set_global()
    ax.set_facecolor(OCEAN)
    ax.coastlines(resolution="50m", linewidth=0.28, color=COAST, alpha=0.9, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.22, color=GRIDC, alpha=0.9, zorder=2,
                 xlocs=[-120, -60, 0, 60, 120], ylocs=[-60, -30, 0, 30, 60])
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["geo"].set_edgecolor(GRIDC)
    ax.spines["geo"].set_linewidth(0.6)
    # both offsets in POINTS, not axes fractions: the two layouts have very different axes
    # heights, and an axes-fraction offset moved the subtitle up into the title in the tall one.
    ax.set_title(title, fontsize=TITLE, color=INK, weight="bold",
                 pad=SUB + 9, loc="left")
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", fontsize=SUB, color=MUTED,
                ha="left", va="bottom", annotation_clip=False)


PANELS = [
    dict(fid=2075, kind="nodes", title="f2075",
         sub="three bowties laid across the equator"),
    dict(fid=2235, kind="nodes", title="f2235",
         sub="bands where the mesh converges at the poles"),
    dict(fid=656, kind="nodes", title="f656",
         sub="a regular lattice, 213 disconnected pieces"),
    dict(fid=(2401, 2067), kind="field", title="convection",
         sub="the group the interventions act on"),
]

if TALL:
    TITLE, SUB = 10.2, 8.4
nrow, ncol = (4, 1) if TALL else (2, 2)
fh = (4 * 3.34 + 0.55) if TALL else 2.72
fig, axes = plt.subplots(nrow, ncol, figsize=(W, fh), facecolor=BG,
                         subplot_kw=dict(projection=PC))
for ax, spec in zip(axes.ravel(), PANELS):
    worldmap(ax, spec["title"], spec["sub"])
    if spec["kind"] == "nodes":
        # Node membership comes from fs_footprints.npy, which is what the component and
        # spread statistics quoted in the text were computed on. The SHADE comes from the
        # per-node mean amplitude recomputed in footprints_extra.py -- these features were
        # previously drawn as flat on/off marks, which threw that amplitude away.
        n = nodes_of(spec["fid"])
        a = strength_of(spec["fid"])[n]
        a = a / max(a.max(), 1e-9)
        # discrete nodes: MacMillan Fig 3 convention -- smoothing would erase the very
        # node-level structure that identifies a grid-locked feature.
        sz = float(np.clip(2600.0 / max(len(n), 1), 0.30, 3.2))
        order = np.argsort(a)                      # strongest nodes drawn last
        ax.scatter(LON[n][order], LAT[n][order], s=sz, c=a[order] ** 0.7, cmap=NODE_CMAP,
                   vmin=0.0, vmax=1.0, marker="s", linewidths=0, transform=PC, zorder=4)
    else:
        v = np.zeros(L)
        for f in spec["fid"]:
            v = np.maximum(v, strength_of(f))
        g = to_grid(v)
        hi = float(np.percentile(g[g > 0], 99.0))
        g = np.ma.masked_less(g, 0.035 * hi)
        ax.pcolormesh(GLON, GLAT, (g / hi) ** FIELD_GAMMA, cmap=FIELD_CMAP, vmin=0.0, vmax=1.0,
                      shading="auto", transform=PC, zorder=4, rasterized=True)

if not TALL:
    fig.text(0.012, 0.020, "colour = mean activation amplitude, each panel scaled to its own peak",
             fontsize=NOTE - 0.4, color=FAINT, ha="left", va="bottom")
if TALL:
    fig.subplots_adjust(top=0.955, bottom=0.028, left=0.004, right=0.996, hspace=0.30)
else:
    fig.subplots_adjust(top=0.885, bottom=0.078, left=0.012, right=0.988,
                        wspace=0.055, hspace=0.42)
name = "paper_fig_footprints_tall.png" if TALL else "paper_fig_footprints.png"
fig.savefig(ROOT / "figures" / name, dpi=340 if TALL else 520, facecolor=BG)
print("wrote figures/" + name)
for spec in PANELS:
    fid = spec["fid"]
    if spec["kind"] == "nodes":
        print(f"  f{fid}: {len(nodes_of(fid))} nodes")
    else:
        print(f"  {fid}: {sum(len(nodes_of(f)) for f in fid)} nodes (union drawn as field)")
