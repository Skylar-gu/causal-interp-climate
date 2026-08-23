"""Map figure for the observational graph: what an SAE feature looks like, where the
inferred arrows go, and the one directional claim that survives matched surrogates.

Panels
  (a) four representative SAE activation maps (layer 8, accumulated over 12 dates),
      inferno on a pale coastline -- the same rendering as the Ida panels.
  (b) the `leiden_flag` consensus graph (38 pair-edges, 39 hard-partition footprints)
      drawn geographically: arrow from source centroid to target centroid. Ochre arrows
      are the 24 "residual" edges no matched surrogate draw (S4c, 12 draws) reproduced;
      grey dashed arrows the 14 the surrogate also found. Footprints shaded GnBu.
  (c) frac_eastward -- fraction of scoreable extratropical zonal pairs whose arrow
      points east -- for the real graph and its parts against the surrogate ladder.

Why leiden_flag and not sae_flag for the arrows: SAE features are concepts, not places
(median footprint spread 7,851 km); their centroids sit at the poles and a bearing is
undefined. The arrows therefore come from the hard-partition member, and the SAE maps
in (a) show why that is the honest choice.

Inputs are bundled in results/fs_graphmap_inputs.npy (built once from the main repo's
results/flag_gint.npy, candidates/pool_flag_v2_candidates.npy and the Q8 lane-A/B
result files); see build_inputs() below.

Needs cartopy -> python figures/paper_fig_graphmap.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, BLUE, YELLOW, GREY, PALE  # noqa: E402

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"          # paper white-theme coastline (paper_fig_maps.py)
INPUTS = ROOT / "results/fs_graphmap_inputs.npy"
# GRAPHMAP_FILL=color -> GnBu-shaded regions, output *_color; default -> thin outlines.
import os  # noqa: E402
FILL = os.environ.get("GRAPHMAP_FILL", "outline")
SUFFIX = "_color" if FILL == "color" else ""

# Paper convention (paper_fig_maps.py): activation on WHITE, light green -> medium blue,
# so the page shows through and the figure survives grayscale printing.
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "green_blue_field",
    ["#ffffff", "#e4f3e1", "#a8d9b5", "#5cb9a6", "#2f8fae", "#22588a"])
FIELD_GAMMA = 0.28         # on white, weak activity is pale, so it needs a stronger gamma

# representative SAE features (ids from the Ida mechanism groups + the cyclone readout)
SHOW = [(3243, "f3243 — tropical cyclone (readout)"),
        (2401, "f2401 — convection"),
        (3861, "f3861 — low-level vorticity"),
        (553,  "f553 — ascent")]

# frac_eastward table, from the Q8 surrogate ladder (lane A/B, member leiden_flag)
FE_ROWS = [  # label, value, n scored, kind
    ("Surviving edges",                     0.929, "real+"),
    ("Whole real graph",                    0.810, "real"),
    ("Injected-edge positive control",      0.769, "ctrl"),
    ("Matched surrogates",                  0.420, "surro"),
]


def build_inputs():
    """One-off: gather the (unshipped) source arrays into the compact bundle."""
    import json
    # None of these three inputs is shipped; the bundle results/fs_graphmap_inputs.npy IS.
    #   results/flag_gint.npy               <- graphcast_sae.obsgraph.report_flag_gint chain
    #   candidates/pool_flag_v2_candidates  <- graphcast_sae.obsgraph.build_pool_flag_v2
    #   results/q8_laneB_gate.json          <- the Q8 surrogate-ladder gate (residual edges)
    G = np.load(ROOT / "results/flag_gint.npy", allow_pickle=True).item()["results"]["leiden_flag"]
    cd = np.load(ROOT / "candidates/pool_flag_v2_candidates.npy", allow_pickle=True).item()
    q8 = json.load(open(ROOT / "results/q8_laneB_gate.json"))["q8_residual"]
    out = dict(
        centroid_lat=np.asarray(G["centroid_lat"], np.float32),
        centroid_lon=np.asarray(G["centroid_lon"], np.float32),
        pair_edges=[tuple(e) for e in G["pair_edges"]],
        residual_edges=[tuple(e) for e in q8["residual_edges"]],
        footprints=np.asarray(cd["cands"]["leiden_flag"], np.float32),
        mesh_lat=np.asarray(cd["lat"], np.float32),
        mesh_lon=np.asarray(cd["lon"], np.float32),
        frac_eastward_whole=float(G["physics"]["frac_eastward"]),
        n_zonal_pairs=int(G["physics"]["n_extratrop_zonal"]),
        provenance="flag_gint.npy leiden_flag; Q8 lane-B gate.json residual set (S4c, 12 draws)",
    )
    np.save(INPUTS, out, allow_pickle=True)
    return out


def gridmap(lat, lon, vals, step=1.0):
    glon = np.arange(-180, 180 + .01, step)
    glat = np.arange(-90, 90 + .01, step)
    G = griddata((lon, lat), vals, tuple(np.meshgrid(glon, glat)), method="linear")
    return glon, glat, np.nan_to_num(G, nan=0.0)


def base_map(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST, linewidth=0.45)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(0.5)   # thin ink frame on white


def main():
    d = np.load(INPUTS, allow_pickle=True).item() if INPUTS.exists() else build_inputs()
    mlat, mlon = d["mesh_lat"], np.where(d["mesh_lon"] > 180, d["mesh_lon"] - 360, d["mesh_lon"])
    clat, clon = d["centroid_lat"], d["centroid_lon"]
    W = d["footprints"]
    residual = set(map(tuple, d["residual_edges"]))
    edges = [tuple(e) for e in d["pair_edges"]]

    fp = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    glat = geom["lat"]; glon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    ACC = fp["acc"]

    # Equal-aspect maps: each panel-(a) map is ~3.3 in wide -> 1.65 in tall at 2:1, the
    # panel-(b) map ~10 in wide -> 5 in tall. The old [1, 2.15] ratio squashed row (a);
    # aspect="auto" then stretched the maps to fill it.
    fig = plt.figure(figsize=(14, 8.2), facecolor=BG)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 3.0], width_ratios=[1, 1, 1, 1.05],
                           hspace=0.24, wspace=0.06, left=0.02, right=0.985, top=0.905,
                           bottom=0.06)

    # ---------------- (a) SAE activation maps ----------------
    for j, (f, name) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, j], projection=PC)
        lo, la, G = gridmap(glat, glon, ACC[:, f], step=1.2)
        ax.imshow(G, origin="lower", extent=[-180, 180, -90, 90], cmap=FIELD_CMAP,
                  norm=PowerNorm(FIELD_GAMMA), transform=PC, interpolation="bilinear")
        base_map(ax)
        ax.set_title(name, fontsize=9.5, color=INK, pad=4)
    fig.text(0.02, 0.955, "a   SAE activation maps — layer 8, accumulated over 12 dates "
             "(darker = fires more often there)",
             fontsize=11.5, color=INK, weight="bold")

    # ---------------- (b) geographic arrows ----------------
    ax = fig.add_subplot(gs[1, 0:3], projection=PC)
    if FILL == "color":
        cover = W.max(0)                               # hard partition: one mode per node
        lo, la, G = gridmap(mlat, mlon, cover, step=1.0)
        ax.imshow(G, origin="lower", extent=[-180, 180, -90, 90], cmap="GnBu", vmin=0,
                  vmax=np.percentile(G[G > 0], 97), transform=PC, alpha=0.75,
                  interpolation="bilinear")
    else:
        # Region OUTLINES only, thin ink. Fills carried no information: within a region the
        # pooling weights are uniform to within a few percent, and between regions the
        # sum-to-one normalisation just encodes 1/region size. Nearest-node raster on a
        # 0.5-degree grid so boundaries follow the mesh partition exactly.
        from scipy.spatial import cKDTree
        def _unit(la, lo):
            a, o = np.deg2rad(la), np.deg2rad(lo)
            return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
        rlon = np.arange(-180.0, 180.0, 0.5) + 0.25
        rlat = np.arange(-89.75, 90.0, 0.5)
        R2, R1 = np.meshgrid(rlon, rlat)
        _, nn = cKDTree(_unit(mlat, mlon)).query(_unit(R1.ravel(), R2.ravel()), k=1)
        region = np.where(W.max(0) > 0, W.argmax(0), -1)[nn].reshape(R1.shape)
        for c in range(W.shape[0]):
            ax.contour(rlon, rlat, (region == c).astype(float), levels=[0.5], colors=[INK],
                       linewidths=0.45, transform=PC, zorder=3)
    base_map(ax)
    ax.scatter(clon, clat, s=14, color=INK, zorder=6, transform=PC, linewidths=0)

    def arrow(s, t, color, lw, ls, z):
        lo0, la0, lo1, la1 = clon[s], clat[s], clon[t], clat[t]
        # shortest-way longitude, so a dateline pair does not sweep the whole map
        dlon = ((lo1 - lo0 + 180) % 360) - 180
        ax.plot([lo0, lo0 + dlon], [la0, la1], color=color, lw=lw, ls=ls, zorder=z,
                transform=PC, solid_capstyle="round")
        if abs(lo0 + dlon) > 180:                      # draw the wrapped copy too
            off = -360 if lo0 + dlon > 180 else 360
            ax.plot([lo0 + off, lo0 + dlon + off], [la0, la1], color=color, lw=lw, ls=ls,
                    zorder=z, transform=PC)
        # arrowhead: last 10% of the segment, in plotting coordinates
        x0, x1 = lo0 + 0.9 * dlon, lo0 + dlon
        y0, y1 = la0 + 0.9 * (la1 - la0), la1
        if abs(x1) > 180:
            x0 += off; x1 += off
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=PC, color=color,
                                     arrowstyle="-|>", mutation_scale=11, lw=lw, zorder=z + 1,
                                     shrinkA=0, shrinkB=0))

    for (s, t) in edges:
        if (s, t) in residual:
            arrow(s, t, YELLOW, 1.7, "-", 5)
        else:
            arrow(s, t, GREY, 1.1, (0, (3, 2)), 4)
    from matplotlib.patches import Patch
    handles = [
        plt.Line2D([], [], color=YELLOW, lw=1.7, label="surviving edge"),
        plt.Line2D([], [], color=GREY, lw=1.1, ls=(0, (3, 2)),
                   label="surrogate-reproduced edge"),
        (Patch(facecolor="#a8d9b5", edgecolor="none", alpha=0.75,
               label="39 spatial pooling regions") if FILL == "color" else
         Patch(facecolor="none", edgecolor=INK, linewidth=0.6,
               label="39 spatial pooling regions")),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True, facecolor=BG,
              edgecolor=PALE, framealpha=0.92)
    fig.text(0.02, 0.700, "b   Inferred lagged edges on the map — 39-region spatial "
             "partition, 38 consensus edges (source → target centroid)",
             fontsize=11.5, color=INK, weight="bold")

    # ---------------- (c) frac_eastward ----------------
    axc = fig.add_subplot(gs[1, 3])
    _p = axc.get_position()                            # 20% shorter, anchored at the top
    axc.set_position([_p.x0, _p.y0 + 0.2 * _p.height, _p.width, 0.8 * _p.height])
    axc.set_facecolor(BG)
    n = len(FE_ROWS); yy = np.arange(n)[::-1] * 1.0
    for y, (lab, v, kind) in zip(yy, FE_ROWS):
        col = {"real+": YELLOW, "real": BLUE, "real-": GREY, "surro": PALE, "ctrl": BG}[kind]
        edge = BLUE if kind == "ctrl" else col
        axc.barh(y, v, height=0.46, color=col, edgecolor=edge, lw=1.1, zorder=3)
        axc.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=9, color=INK, weight="bold")
        axc.text(0.0, y + 0.33, lab, va="bottom", ha="left", fontsize=8.2, color=INK)
    axc.axvline(0.5, color=FAINT, lw=0.8, ls=":", zorder=2)
    axc.axvline(0.6, color=MUTED, lw=0.9, zorder=2)
    axc.text(0.49, n - 0.15, "chance", fontsize=7.5, color=FAINT, ha="right", va="bottom")
    axc.text(0.61, n - 0.15, "prereg bar", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    axc.set_xlim(0, 1.0); axc.set_ylim(-0.5, n + 0.15)
    axc.set_yticks([])
    axc.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    axc.set_xlabel("Fraction Eastward", fontsize=9, color=MUTED)
    for sp in ("top", "right", "left"):
        axc.spines[sp].set_visible(False)
    axc.spines["bottom"].set_color(PALE)
    axc.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    axc.set_title("c   Real vs matched surrogate: eastward asymmetry\n"
                  "     lives in the surviving edges",
                  fontsize=10.5, color=INK, weight="bold", loc="left", pad=6)
    axc.text(0, -0.95, "Fraction of scoreable extratropical zonal pairs whose arrow points east.\n"
             "Surrogates match spectrum, sparsity, footprint overlap and same-time\n"
             "covariance (S4c); only lead–lag asymmetry is left, and it points east.",
             fontsize=7.6, color=FAINT, ha="left", va="top", transform=axc.transData)

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_graphmap{SUFFIX}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor=BG, bbox_inches="tight")
    print(f"-> figures/paper_fig_graphmap{SUFFIX}.png / .pdf")


if __name__ == "__main__":
    main()
