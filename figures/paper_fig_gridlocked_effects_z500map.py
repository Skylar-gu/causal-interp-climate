"""Grid-locked features: activation footprint (left) and the spatial forecast-error change
from ablating each one (right). Paper style, one row per feature.

This is paper_fig_gridlocked_effects.py with the per-field bar chart on the right replaced
by a per-gridpoint z500 error map.

left   nearest-node raster of the footprint nodes, shade = mean SAE amplitude over 12 i.i.d.
       snapshots; features on < 300 nodes drawn as markers.
right  per-gridpoint RMSE(feature ablated) - RMSE(baseline) for z500 at +48 h, mean over the
       six paired ICs, when the single feature is held at zero through the rollout.
       --vs-control differences against the coverage/connectivity-matched control instead.
       One shared colour scale across all rows, so 2954's lattice-wide damage and the near-
       inert compact features are read on the same scale; f2954 saturates the top of the
       bar (extend arrow) and the "global mean" annotation carries each feature's magnitude.
       The grid-locked score under each map is the minimum positional score over the five
       rotation angles (45/90/135/180/270 deg; results/fs_rotation_all*.npy).

Data: results/gridlock_z500_perfeat_field.npy, results/hybrid_footprint_fires.npz and
      results/fs_rotation_all*.npy do NOT ship (per-gridpoint field ~0.35 GB); the rendered
      PDF is the paper's appendix z500-map figure. data/mesh_2to6_geom.npy and
      data/fs_footprint_fires_nw12.npz ship.
Needs cartopy -> python figures/paper_fig_gridlocked_effects_z500map.py [f2954 ...] [--vs-control]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["IBM Plex Sans", "DejaVu Sans"]   # the clean-paper figures' face, if installed
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT   # gridlock_z500_perfeat_field.npy, hybrid_footprint_fires.npz and fs_rotation_all*.npy
              # are not shipped here -- the committed PDF is the paper's appendix z500-map figure
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC  # noqa: E402

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
GREEN_BLUE = LinearSegmentedColormap.from_list(
    "green_blue_node", ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])
# light peach -> coral -> rose-pink; stays pale through the low end so faint error still reads
PEACH_ROSE = LinearSegmentedColormap.from_list(
    "peach_rose", ["#ffffff", "#ffe3d0", "#ffc4a6", "#fb9e88", "#f0728c", "#c94e86"])
SPARSE = 300
STEM = "paper_fig_gridlocked_effects_z500map"
TITLE, SUB, TICK, LAB, NOTE = 9.6, 7.0, 6.3, 6.8, 7.4

GEOM = ROOT / "data/mesh_2to6_geom.npy"
NW12 = ROOT / "data/fs_footprint_fires_nw12.npz"
ANGLE_FILES = ["fs_rotation_all.npy", "fs_rotation_all_90.npy", "fs_rotation_all_45.npy",
               "fs_rotation_all_135.npy", "fs_rotation_all_270.npy"]

VS_CONTROL = "--vs-control" in sys.argv
FEATS = [int(a[1:]) if a.startswith("f") else int(a)
         for a in sys.argv[1:] if not a.startswith("--")] or [2954, 2075, 407, 2585, 2989, 3535]


def roll180(lon, *fields):
    lon2 = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    o = np.argsort(lon2)
    return (lon2[o],) + tuple(np.asarray(f)[..., o] for f in fields)


def load_error_fields():
    """feature -> per-gridpoint dRMSE(z500, +48 h) map + its global area-weighted mean."""
    d = np.load(MAIN / "results/gridlock_z500_perfeat_field.npy", allow_pickle=True).item()
    lat, lon = np.asarray(d["lat"]), np.asarray(d["lon"])
    z, truth = d["z500"], np.asarray(d["truth"])
    ics = list(d["ics"])
    nI = len(ics)
    assert truth.shape == (nI, lat.size, lon.size)

    def rmse(arm):
        a = np.asarray(z[arm])
        assert a.shape == (nI, lat.size, lon.size), (arm, a.shape)
        assert np.isfinite(a).all(), f"{arm} non-finite"
        e = a - truth
        return np.sqrt((e * e).mean(axis=0))

    w = np.cos(np.deg2rad(lat))[:, None]
    base = rmse("baseline")
    out = {}
    for f in FEATS:
        ref = rmse(f"ctrl_f{f}") if VS_CONTROL else base
        df = rmse(f"f{f}") - ref
        gmean = float((w * df).sum() / (w.sum() * lon.size))
        out[f] = (df, gmean)
    print(f"loaded {nI} ICs; arms {sorted(k for k in z)}")
    return lat, lon, out


def main():
    # ---- footprint raster inputs (mesh nodes) ----
    geom = np.load(GEOM, allow_pickle=True).item()
    mlat = np.asarray(geom["lat"], float)
    mlon = np.asarray(geom["lon"], float); mlon = np.where(mlon > 180, mlon - 360, mlon)
    fires = np.load(MAIN / "results/hybrid_footprint_fires.npz")["fires"]
    amp_acc = np.load(NW12)["acc"]

    def unit(la, lo):
        a, o = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(a) * np.cos(o), np.cos(a) * np.sin(o), np.sin(a)], -1)
    flon = np.arange(-180.0, 180.0, 0.25) + 0.125
    flat = np.arange(-89.875, 90.0, 0.25)
    G2, G1 = np.meshgrid(flon, flat)
    _, nn = cKDTree(unit(mlat, mlon)).query(unit(G1.ravel(), G2.ravel()), k=1)
    nn = nn.reshape(G1.shape)

    rot = [np.load(MAIN / "results" / f, allow_pickle=True).item()["positional"]
           for f in ANGLE_FILES]
    score = np.min(np.stack(rot), 0)

    # ---- error fields ----
    lat, lon, eff = load_error_fields()
    lon_e = roll180(lon, eff[FEATS[0]][0])[0]
    fields_e = {f: roll180(lon, df)[1] for f, (df, _) in eff.items()}
    # one shared colour scale across every row, so a colour reads as the same Δ z500 RMSE
    # in every panel and values can be compared directly. f2954 dominates the rest by ~30x
    # and saturates the top of the bar (drawn with an extend arrow); the "global mean"
    # annotation still carries each feature's absolute magnitude.
    _pos = np.concatenate([v[v > 0].ravel() for v in fields_e.values() if (v > 0).any()])
    VMAX = float(np.percentile(_pos, 98)) if _pos.size else 1.0

    nrow = len(FEATS)
    fig = plt.figure(figsize=(9.8, 2.15 * nrow + 0.35), facecolor=BG)
    gs = fig.add_gridspec(nrow, 2, width_ratios=[1.0, 1.0], hspace=0.34, wspace=0.13,
                          left=0.012, right=0.955, top=0.93, bottom=0.055)
    for r, f in enumerate(FEATS):
        # ---- left: activation footprint ----
        ax = fig.add_subplot(gs[r, 0], projection=PC)
        amp = amp_acc[:, f].astype(float); amp = amp / max(amp.max(), 1e-9)
        on = fires[:, f].astype(bool)
        if int(on.sum()) < SPARSE:
            ax.scatter(mlon[on], mlat[on], c=amp[on] ** 0.7, cmap=GREEN_BLUE, vmin=0, vmax=1,
                       s=22, edgecolors=INK, linewidths=0.35, transform=PC, zorder=5)
        else:
            img = np.where(on, amp, np.nan)[nn]
            ax.imshow(np.ma.masked_invalid(img) ** 0.7, origin="lower",
                      extent=[-180, 180, -90, 90], cmap=GREEN_BLUE, vmin=0, vmax=1,
                      transform=PC, interpolation="nearest")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.4)
        ax.set_global(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        ax.set_title(f"Feature {f}", fontsize=TITLE, color=INK, weight="bold", loc="left", pad=4)
        if r == 0:
            ax.text(0, 1.30, "Activation footprint", transform=ax.transAxes,
                    fontsize=TITLE, color=INK, weight="bold", ha="left", va="bottom")

        # ---- right: per-gridpoint z500 error change ----
        bx = fig.add_subplot(gs[r, 1], projection=PC)
        bx.set_global(); bx.set_facecolor("#fbfbfb")
        bx.coastlines(resolution="50m", linewidth=0.35, color="#9a9a9a", zorder=5)
        bx.gridlines(draw_labels=False, linewidth=0.22, color=GRIDC,
                     xlocs=[-120, -60, 0, 60, 120], ylocs=[-60, -30, 0, 30, 60])
        bx.spines["geo"].set_edgecolor(GRIDC); bx.spines["geo"].set_linewidth(0.55)
        pm = bx.pcolormesh(lon_e, lat, np.clip(fields_e[f], 0, None), transform=PC,
                           cmap=PEACH_ROSE, vmin=0, vmax=VMAX, zorder=3, rasterized=True)
        if r == 0:
            bx.text(0, 1.30, "Added z$_{500}$ error, feature removed  (+48 h)",
                    transform=bx.transAxes, fontsize=TITLE, color=INK, weight="bold",
                    ha="left", va="bottom")
        cax = bx.inset_axes([1.028, 0.05, 0.026, 0.90])
        cb = fig.colorbar(pm, cax=cax, extend="max")
        cb.ax.tick_params(labelsize=TICK, colors=MUTED, length=0)
        cb.outline.set_edgecolor(GRIDC)
        if r == 0:
            cb.set_label("Δ z$_{500}$ RMSE (m)", fontsize=LAB, color=MUTED)
        lblbox = dict(boxstyle="round,pad=0.38", fc="white", ec="none", alpha=0.92)
        bx.text(0.985, 0.055, f"grid-locked score {score[f]:.2f}", transform=bx.transAxes,
                ha="right", va="bottom", fontsize=NOTE, color=INK, weight="bold",
                bbox=lblbox, zorder=6)
        bx.text(0.015, 0.055, f"global mean {eff[f][1]:+.2f} m", transform=bx.transAxes,
                ha="left", va="bottom", fontsize=NOTE, color=INK, weight="bold",
                bbox=lblbox, zorder=6)
        print(f"f{f}: score {score[f]:.2f}  global dRMSE {eff[f][1]:+.3f} m  "
              f"peak {np.nanmax(fields_e[f]):.1f} m  shared vmax {VMAX:.2f}")

    stem = STEM + ("_vs_control" if VS_CONTROL else "")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/{stem}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    print(f"-> figures/{stem}.pdf / .png")


if __name__ == "__main__":
    main()
