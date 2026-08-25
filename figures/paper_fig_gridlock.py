"""Grid-locked features can be causally load-bearing without a physical reading.

Row 1 (a-d): four features drawn as DISCRETE MESH NODES (nearest-node raster on a 0.25 deg
grid; smoothing would erase the node-level structure that defines grid-lock), white ground,
green->blue amplitude, the paper convention of paper_fig_maps.py.
  f2954  the full-lattice feature: +1.50 m z500 at +48 h, 19% of the baseline error
  f3404  its coverage- and connectivity-matched control, at least as grid-locked: +0.00 m
  f2075  bowties (positional by the 180-degree rotation test)
  f656   lattice dots (positional)
Row 2:
  (e) all 17 single-feature ablations vs their matched controls (+48 h z500, 6 paired ICs),
      plus the 27-feature mesh-locked CLASS against its firing-rate-matched control.
  (f) grid-lock score vs causal cost: grid-lockedness does not predict cost.

Data: results/fs_matched_rmse.npy (matched controls), results/fs_global_rmse_perIC.npy
(class ablation), results/fs_gridlock_all.npy (score), results/hybrid_footprint_fires.npz
(footprint nodes: fires in >=2 of 8 IID windows), data/fs_footprint_fires_nw12.npz
(per-node accumulated amplitude), data/mesh_2to6_geom.npy (mesh lat/lon).

Needs cartopy -> .venv-jax/bin/python figures/paper_fig_gridlock.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/home/ec2-user/causal-graphcast")
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREEN, YELLOW, GREY, PALE  # noqa

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
NODE_CMAP = LinearSegmentedColormap.from_list(
    "green_blue_node",
    ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])

SHOW = [(2954, "f2954 — full lattice", "load-bearing: +1.50 m z500, 19% of error"),
        (3404, "f3404 — its matched control", "equally grid-locked, costs +0.00 m"),
        (2075, "f2075 — bowties", "positional (rotation test), +0.06 m"),
        (656,  "f656 — lattice dots", "positional, +0.12 m")]


def main():
    geom = np.load(ROOT / "data/mesh_2to6_geom.npy", allow_pickle=True).item()
    lat = np.asarray(geom["lat"], float)
    lon = np.asarray(geom["lon"], float); lon = np.where(lon > 180, lon - 360, lon)
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    acc = np.load(ROOT / "data/fs_footprint_fires_nw12.npz")["acc"]

    # nearest-node raster
    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(lat, lon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    M = np.load(MAIN / "results/fs_matched_rmse.npy", allow_pickle=True).item()
    A, groups, iz, Lz = M["acc"], M["groups"], M["fields"].index("z500"), M["S"] - 1
    base = np.asarray(A["baseline"])[:, Lz, iz]
    score = np.asarray(np.load(MAIN / "results/fs_gridlock_all.npy",
                               allow_pickle=True).item()["score"])
    rows = []
    for k in groups:
        if not k.startswith("f") or k == "floor":
            continue
        f, c = groups[k][0], groups["ctrl_" + k][0]
        df = np.asarray(A[k])[:, Lz, iz] - base
        dc = np.asarray(A["ctrl_" + k])[:, Lz, iz] - base
        rows.append(dict(f=f, c=c, df=df.mean(), dc=dc.mean(),
                         p=stats.ttest_rel(df, dc).pvalue, sf=score[f], sc=score[c]))
    rows.sort(key=lambda r: -r["df"])
    rho = stats.spearmanr([r["sf"] for r in rows], [r["df"] - r["dc"] for r in rows])

    # class-level arm (27 mesh-locked features vs firing-rate-matched control), same lead
    P = np.load(MAIN / "results/fs_global_rmse_perIC.npy", allow_pickle=True).item()
    pz = P["fields"].index("z500")
    li = int(round(48 / (120 / P["S"]))) - 1 if P["S"] else 7
    pb = np.asarray(P["acc"]["baseline"])[:, li, pz]
    cls = dict(df=(np.asarray(P["acc"]["mesh_locked"])[:, li, pz] - pb).mean(),
               dc=(np.asarray(P["acc"]["ctrl_mesh"])[:, li, pz] - pb).mean(),
               p=stats.ttest_rel(np.asarray(P["acc"]["mesh_locked"])[:, li, pz],
                                 np.asarray(P["acc"]["ctrl_mesh"])[:, li, pz]).pvalue)

    # Row 0: four equal-aspect maps (~2.9 in wide -> ~1.45 in tall); row 1: the two charts.
    fig = plt.figure(figsize=(12.6, 6.6), facecolor=BG)
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 2.35], hspace=0.30,
                              left=0.05, right=0.985, top=0.90, bottom=0.09)
    gs0 = outer[0].subgridspec(1, 4, wspace=0.08)
    gs1 = outer[1].subgridspec(1, 2, wspace=0.28)

    # ---------------- (a-d) node maps ----------------
    for j, (fid, title, sub) in enumerate(SHOW):
        ax = fig.add_subplot(gs0[0, j], projection=PC)
        amp = acc[:, fid].astype(float)
        amp = amp / max(amp.max(), 1e-9)
        v = np.where(fires[:, fid], amp, np.nan)
        img = v[nn]
        ax.imshow(np.ma.masked_invalid(img) ** 0.7, origin="lower",
                  extent=[-180, 180, -90, 90], cmap=NODE_CMAP, vmin=0, vmax=1,
                  transform=PC, interpolation="nearest")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST,
                       linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        ax.set_title(f"{'abcd'[j]}   {title}", fontsize=9.6, color=INK, weight="bold",
                     loc="left", pad=14)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="left",
                    va="bottom", annotation_clip=False)
    fig.text(0.05, 0.965, "Grid-locked features: activation locked to the icosahedral "
             "mesh, drawn as discrete mesh nodes (shade = mean amplitude)",
             fontsize=11, color=INK, weight="bold")

    # ---------------- (e) matched-control dumbbells ----------------
    ax = fig.add_subplot(gs1[0, 0])
    labels = [f"f{r['f']}" for r in rows] + ["27-feature mesh-locked class"]
    yy = np.arange(len(labels))[::-1]
    for y, r in zip(yy[:-1], rows):
        col = YELLOW if r["f"] == 2954 else (BLUE if (r["p"] < 0.05 and r["df"] > r["dc"]) else GREY)
        ax.plot([r["dc"], r["df"]], [y, y], color=GRIDC, lw=1.2, zorder=1)
        ax.scatter([r["df"]], [y], s=40, color=col, edgecolors=INK, linewidths=0.4, zorder=4)
        ax.scatter([r["dc"]], [y], s=28, facecolors="none", edgecolors=MUTED, linewidths=1.1,
                   zorder=3)
    y = yy[-1]
    ax.plot([cls["dc"], cls["df"]], [y, y], color=GRIDC, lw=1.2, zorder=1)
    ax.scatter([cls["df"]], [y], s=40, color=GREY, edgecolors=INK, linewidths=0.4, zorder=4,
               marker="D")
    ax.scatter([cls["dc"]], [y], s=28, facecolors="none", edgecolors=MUTED, linewidths=1.1,
               zorder=3, marker="D")
    ax.axhline(y + 0.5, color=GRIDC, lw=0.8)
    ax.axvline(0, color=PALE, lw=0.9, zorder=0)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=7.6, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.set_xlabel("Δ z500 RMSE vs baseline at +48 h (m); 6 paired ICs", fontsize=8.6,
                  color=MUTED)
    ax.grid(axis="x", color=GRIDC, lw=0.5, zorder=0)
    ax.scatter([], [], s=40, color=BLUE, edgecolors=INK, linewidths=0.4,
               label="feature ablated (p < 0.05 vs its control)")
    ax.scatter([], [], s=40, color=GREY, edgecolors=INK, linewidths=0.4,
               label="feature ablated (n.s.)")
    ax.scatter([], [], s=28, facecolors="none", edgecolors=MUTED, linewidths=1.1,
               label="coverage- & connectivity-matched control")
    ax.legend(loc="center right", fontsize=7.4, frameon=False)
    ax.text(cls["df"] + 0.06, y, f"class ≈ its control (p = {cls['p']:.2f})", fontsize=7.4,
            color=MUTED, va="center")
    ax.set_title("e   One grid-locked feature is load-bearing; the class is not",
                 fontsize=10.5, color=INK, weight="bold", loc="left", pad=6)

    # ---------------- (f) score vs cost ----------------
    ax = fig.add_subplot(gs1[0, 1])
    for r in rows:
        col = YELLOW if r["f"] == 2954 else (BLUE if (r["p"] < 0.05 and r["df"] > r["dc"]) else GREY)
        ax.scatter(r["sf"], r["df"] - r["dc"], s=46, color=col, edgecolors=INK,
                   linewidths=0.4, zorder=4)
        ax.scatter(r["sc"], 0.0, s=22, facecolors="none", edgecolors=MUTED, linewidths=1.0,
                   zorder=3)
        if r["f"] in (2954, 3680, 560, 2235, 2075):
            ax.annotate(f"f{r['f']}", (r["sf"], r["df"] - r["dc"]), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.6, color=INK)
    r2954 = [r for r in rows if r["f"] == 2954][0]
    ax.annotate("f3404, its control", (r2954["sc"], 0.0), xytext=(-8, -14),
                textcoords="offset points", fontsize=7.4, color=MUTED, ha="right")
    ax.axhline(0, color=PALE, lw=0.9, zorder=0)
    ax.set_xlabel("grid-lock score (mesh-skeleton firing excess)", fontsize=8.6, color=MUTED)
    ax.set_ylabel("causal cost: feature − matched control (m z500, +48 h)", fontsize=8.6,
                  color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    ax.grid(color=GRIDC, lw=0.5, zorder=0)
    ax.text(0.03, 0.06, f"Spearman ρ = {rho.statistic:+.2f}, p = {rho.pvalue:.2f}\n"
            "grid-lockedness does not predict cost", transform=ax.transAxes, fontsize=8.4,
            color=INK, ha="left", va="bottom")
    ax.scatter([], [], s=22, facecolors="none", edgecolors=MUTED, linewidths=1.0,
               label="matched controls (score shown, cost ≈ 0)")
    ax.legend(loc="lower right", fontsize=7.4, frameon=False)
    ax.set_title("f   Being mesh-locked is not what makes a feature matter",
                 fontsize=10.5, color=INK, weight="bold", loc="left", pad=6)
    for a in fig.axes:
        if a.name == "rectilinear":
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color(PALE)

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_gridlock.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG, bbox_inches="tight")
    print("-> figures/paper_fig_gridlock.png / .pdf")
    print(f"   f2954: +{r2954['df']:.3f} m = {100*r2954['df']/base.mean():.1f}% of baseline "
          f"{base.mean():.2f} m; control f3404 {r2954['dc']:+.3f}; scores {r2954['sf']:.2f} / "
          f"{r2954['sc']:.2f}")
    print(f"   class: mesh_locked {cls['df']:+.2f} vs ctrl {cls['dc']:+.2f}, p={cls['p']:.2f}; "
          f"rho={rho.statistic:+.2f} p={rho.pvalue:.2f}")


if __name__ == "__main__":
    main()
