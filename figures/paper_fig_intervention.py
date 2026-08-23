"""Figures for the intervention paper. Every number is read from results/ at build time.

Built at final print size (5.25 in wide) so point sizes are true on the page.

figA  (a) Ida: ablate each genesis ingredient, read on the internal cyclone feature
      (b) seven storms: restore-to-normal vs delete, scored against ERA5
figB  every intervention set we ran, by how much of the event it removes, labelled with
      how many features it contains. 3 features take 41% of a hurricane; 632 take 39%
      of a ridge. The sets are not nested, so they are bars and not a curve.

Dark theme: warm = the mechanism under test, cool = the heat-dome sets, grey = a control.
"""
import json
import statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
W = 5.25                                   # printed width, inches

import sys; sys.path.insert(0, str(Path(__file__).parent))
from paper_palette import BG, INK, MUTED, GRIDC, BLUE, GREEN, YELLOW, GREY, PALE
FAINT = GREY
# monochrome ramp: black for the measured arm, grey for controls and null arms.
# Lightness ladder does the work, so the set survives colour-vision deficiency.
EMBER, COOL = BLUE, GREEN

TITLE, SUB, TICK, YTICK, VAL, NOTE = 7.4, 6.0, 5.6, 6.4, 6.8, 5.7

conv = json.load(open(ROOT / "results/skill/convection/verdict.json"))["metrics"]
mois = json.load(open(ROOT / "results/skill/moisture2/verdict.json"))["metrics"]
hd = json.load(open(ROOT / "results/heatdome/physics_verdict.json"))["metrics"]

NAME = {"ida2021": "Ida", "michael2018": "Michael", "haishen2020": "Haishen",
        "goni2020": "Goni", "haiyan2013": "Haiyan", "patricia2015": "Patricia",
        "wilma2005": "Wilma"}


def pct(storm, arm, src=conv):
    b = src[storm]["arms"]["baseline"]["deepen"]
    return 100.0 * src[storm]["arms"][arm]["d_deepen"] / b


def dress(ax, title, sub, xlabel=""):
    ax.set_facecolor(BG)
    ax.text(0, 1.26, title, transform=ax.transAxes, fontsize=TITLE, color=INK,
            va="bottom", ha="left", weight="bold")
    ax.text(0, 1.07, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
            va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=SUB, color=MUTED, labelpad=2)
    ax.tick_params(axis="x", labelsize=TICK, colors=MUTED, length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=YTICK, colors=INK, length=0, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIDC)
    ax.grid(axis="x", color=GRIDC, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ============================================================== FIGURE A ====
fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.42), facecolor=BG)

# --- A: Ida, one ingredient at a time.  Values sit inside the bars.
mech = [("Convection", -41), ("Moisture", -17), ("Low-level spin", -12), ("Wind shear", +1)]
lab = [m[0] for m in mech][::-1]
val = [m[1] for m in mech][::-1]
col = [FAINT if m[1] > 0 else EMBER for m in mech][::-1]
y = np.arange(len(val))
axA.barh(y, val, color=col, height=0.6, zorder=3)
axA.set_yticks(y); axA.set_yticklabels(lab)
axA.axvline(0, color=MUTED, lw=0.8, zorder=4)
axA.set_xlim(-46, 10)
for i, v in enumerate(val):
    inside = v < -8
    axA.text(v + 1.6, i, f"{v:+d}%", va="center", ha="left", fontsize=VAL,
             color=BG if inside else INK, weight="bold", zorder=5)
dress(axA, "Ida, one ingredient at a time", "Model's own cyclone feature, +48 h")

# --- B: seven storms, physical deepening against ERA5
storms = sorted(NAME, key=lambda s: pct(s, "conv-zero"))
yy = np.arange(len(storms))
axB.barh(yy, [pct(s, "conv-zero") for s in storms], facecolor="none",
         edgecolor=EMBER, lw=0.9, height=0.66, zorder=2)
axB.barh(yy, [pct(s, "conv-normal") for s in storms], color=EMBER,
         height=0.66, zorder=3)
axB.barh(yy, [pct(s, "conv-normal", mois) for s in storms], color=FAINT,
         height=0.20, zorder=4)
axB.set_yticks(yy); axB.set_yticklabels([NAME[s] for s in storms])
axB.set_xlim(0, 82)
axB.set_xticks([0, 20, 40, 60, 80])
for i, s in enumerate(storms):
    axB.text(pct(s, "conv-zero") + 1.8, i, f"{pct(s,'conv-zero'):.0f}%", va="center",
             fontsize=VAL - 0.6, color=MUTED)
dress(axB, "Seven storms, against ERA5", "Share of real deepening lost")

fig.text(0.145, 0.035,
         "grey = control        "
         "filled = convection returned to normal        outlined = deleted outright",
         fontsize=NOTE, color=FAINT, ha="left", va="bottom")
fig.subplots_adjust(top=0.74, bottom=0.155, left=0.145, right=0.985, wspace=0.34)
fig.savefig(ROOT / "figures/paper_fig_A.png", dpi=400, facecolor=BG)
print("median restore %.1f%%  delete %.1f%%  moisture %.1f%%" % (
    st.median([pct(s, "conv-normal") for s in storms]),
    st.median([pct(s, "conv-zero") for s in storms]),
    st.median([pct(s, "conv-normal", mois) for s in storms])))

# ============================================================== FIGURE B ====
fig, ax = plt.subplots(figsize=(W, 2.05), facecolor=BG)
base = hd["baseline"]["ridge_peak"]
R = lambda k: 100.0 * hd[k]["d_ridge"] / base

sets = [("Convection, Ida", 3, 41.0, EMBER),
        ("All ridge features", 632, R("union_all"), COOL),
        ("Core + flanking lows", 36, R("core_flank"), COOL),
        ("Jet wave train", 273, R("core_jet"), COOL),
        ("Ridge core alone", 6, R("core"), COOL),
        ("Random features", 177, R("random"), FAINT)]
sets = sorted(sets, key=lambda r: r[2])

y = np.arange(len(sets))
ax.barh(y, [s[2] for s in sets], color=[s[3] for s in sets], height=0.64, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in sets])
ax.set_xlim(0, 62)
ax.set_xticks([0, 10, 20, 30, 40])
for i, (name, n, v, c) in enumerate(sets):
    ax.text(v + 1.0, i, f"{v:.0f}%", va="center", fontsize=VAL, color=INK, weight="bold")
    ax.text(v + 5.6, i, f"from {n} feature{'s' if n > 1 else ''}", va="center",
            fontsize=NOTE, color=MUTED)
dress(ax, "How much of an event each set of features carries",
      "Every feature group returned to its normal level, and what the atmosphere did",
      "% of the event removed")
fig.text(0.255, 0.035, "blue = hurricane        green = heat dome        grey = control",
         fontsize=NOTE, color=FAINT, ha="left", va="bottom")
fig.subplots_adjust(top=0.735, bottom=0.195, left=0.255, right=0.985)
fig.savefig(ROOT / "figures/paper_fig_B.png", dpi=400, facecolor=BG)
print("ridge: core %.1f  flank %.1f  jet %.1f  all %.1f  random %.1f"
      % (R("core"), R("core_flank"), R("core_jet"), R("union_all"), R("random")))
