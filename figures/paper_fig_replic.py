"""Figure 3: the gain result across the whole storm set.

One dot per storm, one row per setting of the single gain alpha. The x axis is the change the
edit causes in the forecast's intensity error against ERA5, so a dot left of zero is a storm
the edited model forecast better than the untouched model did.

Light theme, drawn at final print size (3.63 in = 0.66 x text width) so type renders 1:1.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
W = 3.63                                   # == 0.66 * TW, placed 1:1
import sys; sys.path.insert(0, str(Path(__file__).parent))
from paper_palette import INK, MUTED, GRIDC, GREEN, YELLOW
# fill carries the reading a second time: solid where the edit helped, open where it hurt
BETTER, WORSE = GREEN, YELLOW
FS, FT = 6.4, 6.0

rows = np.load(ROOT / "results/fields8_track.npy", allow_pickle=True).item()
DEV = [s for s, r in rows.items() if not r["nondev"]]
NICE = {"haishen2020": "Haishen", "haiyan2013": "Haiyan"}

ARMS = [("gain-0", r"$\alpha$ = 0"),
        ("gain-1.25", r"$\alpha$ = 1.25"),
        ("gain-2", r"$\alpha$ = 2")]

fig, ax = plt.subplots(figsize=(W, 1.28), facecolor="white")
ax.set_facecolor("white")
ax.axvline(0, color=INK, lw=0.8, zorder=3)

for i, (arm, lab) in enumerate(ARMS):
    y = len(ARMS) - 1 - i
    v = np.array([rows[s]["arms"][arm]["d_rmse"] for s in DEV])
    ax.scatter(v, np.full_like(v, y), s=17, linewidths=0.9,
               facecolors=[BETTER if x < 0 else "none" for x in v],
               edgecolors=[BETTER if x < 0 else WORSE for x in v], zorder=5)
    ax.plot([np.median(v)] * 2, [y - 0.24, y + 0.24], color=INK, lw=1.2, zorder=6)

for s, dx, ha in (("haiyan2013", -0.25, "right"), ("haishen2020", 0.25, "left")):
    ax.annotate(NICE[s], xy=(rows[s]["arms"]["gain-2"]["d_rmse"] + dx, 0.0),
                fontsize=FS - 0.7, color=MUTED, va="center", ha=ha)

ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([a[1] for a in ARMS][::-1], fontsize=FS, color=INK)
ax.set_ylim(-0.6, len(ARMS) - 0.4)
ax.set_xlim(-5.2, 6.8)
ax.tick_params(axis="both", labelsize=FT, colors=MUTED, length=0, pad=2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRIDC)
ax.grid(axis="x", color=GRIDC, lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("change in intensity error against ERA5 (hPa)",
              fontsize=FS, color=INK, labelpad=3)

fig.subplots_adjust(top=0.965, bottom=0.275, left=0.175, right=0.985)
fig.savefig(ROOT / "figures/paper_fig_replic.png", dpi=500, facecolor="white")
for arm, lab in ARMS:
    v = [rows[s]["arms"][arm]["d_rmse"] for s in DEV]
    print(f"{arm:<10} median {np.median(v):+.2f}  improved {(np.array(v)<0).sum()}/7")
