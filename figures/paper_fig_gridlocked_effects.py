"""Grid-locked features: footprint (left) and what ablating each one does to the forecast
(right). Paper style. One row per feature.

left   nearest-node raster of the footprint nodes, shade = mean SAE amplitude over 12 i.i.d.
       snapshots (as paper_fig_gridlocked_nodes.py); features on < 300 nodes drawn as markers.
right  change in global RMSE at +48 h for each output field when the single feature is held
       at zero through the rollout, MINUS its coverage/connectivity-matched control, as a
       percentage of the baseline error; mean +- s.e. over the six paired initial conditions.
       Thin outlined bars, light fill. Grey band = +-2 s.d. of the empty-ablation floor (z500).
       The grid-locked score under the chart is the minimum positional score over the five
       rotation angles (45/90/135/180/270 deg; results/fs_rotation_all*.npy).

Data: results/fs_matched_rmse.npy (+ fs_matched_rmse_six.npy when present) in causal-graphcast.
Needs cartopy -> .venv-jax/bin/python figures/paper_fig_gridlocked_effects.py [f2954 f2075 ...] [--vs-baseline]
  --vs-baseline  bars are feature-ablated minus the UNTOUCHED forecast (no control subtracted);
                 writes paper_fig_gridlocked_effects_vs_baseline.*
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path("/home/ec2-user/causal-graphcast")
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREY, PALE  # noqa: E402

PC = ccrs.PlateCarree()
COAST = "#a8a8a8"
GREEN_BLUE = LinearSegmentedColormap.from_list(
    "green_blue_node", ["#bfe3c1", "#88cba4", "#4fb3a6", "#2f8fae", "#2a6b96", "#1f4a7d"])
FIELD_LABEL = {"z500": "z500", "t850": "T850", "t2m": "T2m", "u850": "u850",
               "v850": "v850", "q700": "q700", "msl": "MSLP"}
ANGLE_FILES = ["fs_rotation_all.npy", "fs_rotation_all_90.npy", "fs_rotation_all_45.npy",
               "fs_rotation_all_135.npy", "fs_rotation_all_270.npy"]
SPARSE = 300
VS_BASELINE = "--vs-baseline" in sys.argv
FEATS = [int(a[1:]) if a.startswith("f") else int(a) for a in sys.argv[1:] if not a.startswith("--")] or [2954, 2075]


def load_effects():
    """feature -> (fields, mean %, se %, floor band %) at +48 h, feature - matched control."""
    out, fields, floor = {}, None, None
    for fn in ("fs_matched_rmse.npy", "fs_matched_rmse_six.npy"):
        p = MAIN / "results" / fn
        if not p.exists():
            continue
        M = np.load(p, allow_pickle=True).item()
        A, F, L = M["acc"], M["fields"], M["S"] - 1
        fields = F
        base = np.asarray(A["baseline"])[:, L, :]                 # (IC, field)
        if "floor" in A and floor is None:
            fl = np.asarray(A["floor"])[:, L, F.index("z500")] - base[:, F.index("z500")]
            floor = 100 * fl / base[:, F.index("z500")].mean()
        for k in A:
            if k.startswith("f") and f"ctrl_{k}" in A:
                ref = base if VS_BASELINE else np.asarray(A[f"ctrl_{k}"])[:, L, :]
                d = np.asarray(A[k])[:, L, :] - ref
                pct = 100 * d / base.mean(0)                       # (IC, field)
                out[int(k[1:])] = (pct.mean(0), pct.std(0, ddof=1) / np.sqrt(len(pct)))
    return fields, out, floor


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
    rot = [np.load(MAIN / "results" / f, allow_pickle=True).item()["positional"] for f in ANGLE_FILES]
    score = np.min(np.stack(rot), 0)

    fields, eff, floor = load_effects()
    feats = [f for f in FEATS if f in eff]
    missing = [f for f in FEATS if f not in eff]
    if missing:
        print("no ablation yet for:", missing)
    nrow = len(feats)
    fig = plt.figure(figsize=(9.0, 2.35 * nrow), facecolor=BG)
    gs = fig.add_gridspec(nrow, 2, width_ratios=[1.75, 1.0], hspace=0.42, wspace=0.10,
                          left=0.01, right=0.985, top=0.95, bottom=0.16)
    lim = max(abs(eff[f][0]).max() + eff[f][1].max() for f in feats) * 1.15
    for r, f in enumerate(feats):
        ax = fig.add_subplot(gs[r, 0], projection=PC)
        amp = acc[:, f].astype(float); amp = amp / max(amp.max(), 1e-9)
        on = fires[:, f]
        if int(on.sum()) < SPARSE:
            ax.scatter(lon[on], lat[on], c=amp[on] ** 0.7, cmap=GREEN_BLUE, vmin=0, vmax=1,
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
        ax.set_title(f"Feature {f}", fontsize=9.6, color=INK, weight="bold", loc="left", pad=4)

        bx = fig.add_subplot(gs[r, 1]); bx.set_facecolor(BG)
        if r == 0:
            bx.set_title("Change in forecast error (+48 h):", fontsize=9.6, color=INK,
                         weight="bold", loc="left", pad=4)
        m, se = eff[f]
        y = np.arange(len(fields))[::-1]
        if floor is not None:
            sd = float(np.nanstd(floor, ddof=1)); mu = float(np.nanmean(floor))
            bx.axvspan(mu - 2 * sd, mu + 2 * sd, color=GRIDC, alpha=0.7, lw=0, zorder=0)
        bx.axvline(0, color=GREY, lw=0.7, zorder=1)
        bx.barh(y, m, height=0.42, color=to_rgba(BLUE, 0.16), edgecolor=BLUE, lw=1.0,
                xerr=se, error_kw=dict(ecolor=INK, lw=0.8, capsize=2), zorder=3)
        bx.set_yticks(y); bx.set_yticklabels([FIELD_LABEL.get(k, k) for k in fields],
                                             fontsize=7.4, color=INK)
        bx.tick_params(axis="y", length=0, pad=2)
        bx.tick_params(axis="x", labelsize=7, colors=MUTED, length=0)
        bx.set_xlim(-0.35 * lim, lim); bx.set_ylim(-0.7, len(fields) - 0.3)
        for sp in ("top", "right", "left"):
            bx.spines[sp].set_visible(False)
        bx.spines["bottom"].set_color(PALE)
        bx.grid(axis="x", color=GRIDC, lw=0.5, zorder=0)
        if r == nrow - 1:
            bx.text(0.0, -0.36, "Δ error at +48 h vs " + ("untouched forecast" if VS_BASELINE else "matched control") + ", % of baseline",
                    transform=bx.transAxes, ha="left", va="top", fontsize=7.4, color=MUTED)
        bx.text(1.0, -0.20, f"grid-locked score: {score[f]:.2f}", transform=bx.transAxes,
                ha="right", va="top", fontsize=8.2, color=INK, weight="bold")
        print(f"f{f}: score {score[f]:.2f}  z500 {m[fields.index('z500')]:+.1f}% +- "
              f"{se[fields.index('z500')]:.1f}%")
    stem = "paper_fig_gridlocked_effects" + ("_vs_baseline" if VS_BASELINE else "")
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/{stem}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    print(f"-> figures/{stem}.pdf / .png")


if __name__ == "__main__":
    main()
