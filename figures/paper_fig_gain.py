"""Figures E and F for the intervention paper.

figE  THE GAIN CURVE. Suppression and amplification are one operator with a single knob,
      f -> f + (g-1) max(f - normal, 0), so the committed ablation (g=0) and amplification
      (g>1) sit on one axis. Left: how much the storm deepens, against the deepening that
      actually happened. Right: the quantity that decides whether this is an improvement --
      error against ERA5 over the intensification window. A curve that dips below its own
      g=1 baseline is the model forecasting the storm better than it did unaided.

figF  WHAT THE FEATURES LOOK LIKE ON THE GLOBE. Four real footprints at the same scale:
      three that encode the model's own icosahedral mesh, one that encodes the atmosphere.
      No statistic in this paper found the first of them; only the map did.

Built at final print size. Dark theme matched to paper_fig_intervention.py.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
W = 5.25

# Monochrome on white: the three storms are ordered by how much of the real deepening the
# untouched model already captures, so a lightness ladder carries that ordering and hue is
# not needed. Marker shape carries it a second time for grayscale-safe reading.
import sys; sys.path.insert(0, str(Path(__file__).parent))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREEN, YELLOW, GREY, PALE
EMBER, COOL, DEAD = BLUE, GREEN, PALE
TITLE, SUB, TICK, YTICK, VAL, NOTE = 7.4, 6.0, 5.6, 6.4, 6.8, 5.7

GAINS = [0, 1.25, 1.5, 1.75, 2, 2.5, 3]
STORMS = [("haishen2020", "Haishen 2020"), ("ida2021", "Ida 2021"),
          ("patricia2015", "Patricia 2015")]
# lightness ladder, not three unrelated hues: the storms are ordered by how much of the
# real deepening the untouched model already captures (79%, 31%, 28%).
CURVE = {"haishen2020": GREEN, "ida2021": BLUE, "patricia2015": YELLOW}
MARK = {"haishen2020": "s", "ida2021": "o", "patricia2015": "^"}


def dress(ax, title, sub, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=TITLE, color=INK, weight="bold", pad=12, loc="left")
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
            ha="left", va="bottom")
    ax.set_xlabel(xlabel, fontsize=SUB, color=MUTED, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=SUB, color=MUTED, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK, colors=MUTED, length=0, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRIDC)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ============================================================== FIGURE E ====
truth = np.load(ROOT / "results/skill/gain_conv/era5_truth.npy", allow_pickle=True).item()
D, E, OBS, BASE = {}, {}, {}, {}
for key, _lab in STORMS:
    r = np.load(ROOT / f"results/skill/gain_conv/run_{key}.npy", allow_pickle=True).item()["res"]
    tr = np.asarray(truth[key]["mslp_min"])
    ic = float(r["baseline"]["mslp_min"][0])
    win = max(int(np.argmin(tr)), 6) + 1          # score to the observed peak

    def dp(a):
        return ic - float(np.min(r[a]["mslp_min"]))

    def rmse(a):
        m = np.asarray(r[a]["mslp_min"])
        n = min(len(m), len(tr), win)
        return float(np.sqrt(np.mean((m[:n] - tr[:n]) ** 2)))

    D[key] = [dp("gain-%g" % g) for g in GAINS]
    E[key] = [rmse("gain-%g" % g) for g in GAINS]
    OBS[key] = ic - float(tr.min())
    BASE[key] = (dp("baseline"), rmse("baseline"), dp("rand-gain-3"), rmse("rand-gain-3"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.30), facecolor=BG)

for key, lab in STORMS:
    c = CURVE[key]
    ax1.plot(GAINS, D[key], color=c, lw=1.05, marker=MARK[key], ms=2.4, label=lab,
             zorder=3)
    ax1.axhline(OBS[key], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2, alpha=0.85)
    ax1.plot([1], [BASE[key][0]], marker=MARK[key], ms=3.8, mfc=BG, mec=c, mew=0.9,
             zorder=4)
ax1.set_xlim(-0.15, 3.15)
ax1.set_xticks([0, 1, 2, 3])
ax1.legend(fontsize=NOTE - 0.3, frameon=False, labelcolor=MUTED, loc="upper left",
           handlelength=1.5, borderpad=0, labelspacing=0.22)
dress(ax1, "Turning convection up", "dashed line = the deepening that actually happened",
      r"gain $\alpha$ on the excess above normal", "forecast deepening (hPa)")

for key, lab in STORMS:
    c = CURVE[key]
    ax2.plot(GAINS, E[key], color=c, lw=1.05, marker=MARK[key], ms=2.4, zorder=3)
    ax2.plot([1], [BASE[key][1]], marker=MARK[key], ms=3.8, mfc=BG, mec=c, mew=0.9,
             zorder=4)
    ax2.plot([3], [BASE[key][3]], marker="x", ms=3.4, mec=DEAD, mew=0.9, zorder=4)
    j = int(np.argmin(E[key]))
    if E[key][j] < BASE[key][1] - 0.15:
        ax2.annotate("", xy=(GAINS[j], E[key][j]), xytext=(GAINS[j], BASE[key][1]),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7,
                                     shrinkA=1.5, shrinkB=1.5, mutation_scale=5), zorder=5)
ax2.set_xlim(-0.15, 3.15)
ax2.set_xticks([0, 1, 2, 3])
ax2.text(0.03, 0.965, r"open circle = untouched forecast" "\n" r"$\times$ = random group at $\alpha$ = 3",
         transform=ax2.transAxes, fontsize=NOTE - 0.3, color=FAINT, va="top")
dress(ax2, "Does it forecast better?", "intensity error against ERA5; lower is better",
      r"gain $\alpha$ on the excess above normal", "MSLP error (hPa)")

fig.subplots_adjust(top=0.735, bottom=0.19, left=0.095, right=0.99, wspace=0.30)
fig.savefig(ROOT / "figures/paper_fig_gain.png", dpi=400, facecolor=BG)
print("figE " + "  ".join(
    "%s base %.2f -> best %.2f @g=%g" % (k, BASE[k][1], min(E[k]), GAINS[int(np.argmin(E[k]))])
    for k, _ in STORMS))

# ============================================================== FIGURE F ====
fp = np.load(ROOT / "results/fs_footprints.npy", allow_pickle=True).item()
lat, lon = fp["lat"], fp["lon"]
lon = np.where(lon > 180, lon - 360, lon)

PANELS = [(2075, "f2075", "three bowties across the equator", EMBER),
          (2235, "f2235", "bands where the mesh converges", EMBER),
          (656, "f656", "a regular lattice, 213 pieces", EMBER),
          (3174, "f3174", "a convection feature", "#6ee7a8")]

fig, axes = plt.subplots(1, 4, figsize=(W, 1.02), facecolor=BG)
for ax, (fid, name, note, col) in zip(axes, PANELS):
    ax.set_facecolor(BG)
    nodes = fp["res"][fid]["nodes"] if isinstance(fp["res"][fid], dict) else fp["res"][fid]
    nodes = np.asarray(nodes).astype(int)
    ax.scatter(lon, lat, s=0.035, c="#2c2140", marker=".", linewidths=0, zorder=1)
    sz = float(np.clip(70.0 / np.sqrt(max(len(nodes), 1)), 0.45, 6.0))
    ax.scatter(lon[nodes], lat[nodes], s=sz, c=col, marker=".", linewidths=0, zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.set_title(name, fontsize=NOTE + 0.4, color=INK, weight="bold", pad=8, loc="left")
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=NOTE - 0.6, color=MUTED,
            ha="left", va="bottom")
fig.subplots_adjust(top=0.70, bottom=0.02, left=0.012, right=0.988, wspace=0.09)
# NOT paper_fig_footprints.png -- that one comes from paper_fig_maps.py, which is what
# the paper uses; writing it here silently replaced it with a different layout.
fig.savefig(ROOT / "figures/paper_fig_footprints_alt.png", dpi=400, facecolor=BG)
print("figF ok")
