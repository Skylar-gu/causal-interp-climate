"""Is the effect just proportional to how much of the storm the edit erased?

Emits two figures, each built at the size it is placed at so the type prints legibly.

paper_fig_exposure.png  (main text, 3.41 in wide)
    Every arm's effect against its in-box exposure, the quantity a sceptic would say
    drives it. The two correlate across arms (Spearman 0.72). The dashed line is what
    proportional erasure predicts, anchored on the main convection arm. Inside the shaded
    window three groups sharing no feature sit at matched exposure and 227x apart in
    effect, so exposure cannot be the mechanism.

paper_fig_dose.png  (appendix, 3.00 in wide)
    THE DOSE-RESPONSE THAT DID NOT HOLD. Effect against calibrated ascent score. Three
    groups selected on ascent alone were run to fill the gap between 28.5 and 13.2 sigma;
    asc17 came back inside the noise floor at 16.8. The monotone reading survives only
    inside the moisture family and is drawn as such.

The bomb-cyclone matched pair is a single event and stays in the text rather than being
plotted next to seven-storm medians.

Numbers read from results/skill/<arm>/verdict.json; labels from results/fs_mechanisms_v2.npy.

Inputs: results/fs_mechanisms_v2.npy (shipped) and results/skill/<arm>/verdict.json for every arm in the table above. Only convection, mech_asc21, mech_shear, mech_vort850 and moisture2 ship; mech_ascent and the other arms are NOT shipped and are regenerated with `MECH_RES=<arm> MECH_FEATS=<ids> python -m graphcast_sae.storms.skill_conv_run` then `skill_conv_analyze`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREEN, YELLOW, GREY

# measured run-to-run floor: notes/nondeterminism_floor_2026_08_20.md
FLOOR_P90 = 0.369

# arm -> (label, class, calibrated ascent sigma)
# class: 'ascent' = calibrated ascent label; 'other' = a different mechanism;
#        'unexp'  = zero in-box activation, so the arm removed nothing
ARMS = [
    ("convection",         "convection",   "ascent", 28.5),
    ("mech_ascent",        "ascent",       "ascent", 28.8),
    ("mech_asc21",         "asc21",        "ascent", 20.6),
    ("mech_asc17",         "asc17",        "ascent", 16.8),
    ("mech_asc09",         "asc09",        "ascent",  9.0),
    ("moisture",           "moisture v1",  "other",  13.2),
    ("mech_q600",          "q600",         "other",   6.0),
    ("mech_atm_river",     "atm river",    "other",   4.4),
    ("mech_vort850",       "vort850",      "other",   3.0),
    ("moisture2",          "moisture",     "other",   2.1),
    ("mech_baroclinicity", "baroclinicity","unexp",  None),
    ("mech_blocking",      "blocking",     "unexp",  None),
    ("mech_jet250",        "jet250",       "unexp",  None),
    ("mech_shear",         "shear",        "unexp",  None),
    ("mech_t850",          "t850",         "unexp",  None),
    ("mech_z500",          "z500",         "unexp",  None),
]
SIG = {n: s for n, _, _, s in ARMS}

FACE = {"ascent": BLUE, "other": GREEN, "unexp": "none"}
EDGE = {"ascent": BLUE, "other": GREEN, "unexp": GREY}
MARK = {"ascent": "o", "other": "s", "unexp": "x"}


def arm(name):
    """Median in-box exposure and median restore-to-normal effect over developing storms."""
    m = json.load(open(ROOT / f"results/skill/{name}/verdict.json"))["metrics"]
    dev = [v for v in m.values() if not v.get("nondev")]
    return (float(np.median([v["arms"]["baseline"]["conv_box"] for v in dev])),
            float(np.median([v["arms"]["conv-normal"]["d_deepen"] for v in dev])))


P = {n: arm(n) for n, _, _, _ in ARMS}


def dress(ax, title, sub, xlabel, ylabel, fs):
    """fs = (title, sub/axis, tick) point sizes, chosen for the figure's print width."""
    ft, fa, fk = fs
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=ft, color=INK, weight="bold", pad=13, loc="left")
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=fa, color=MUTED,
            ha="left", va="bottom")
    ax.set_xlabel(xlabel, fontsize=fa, color=MUTED, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=fa, color=MUTED, labelpad=2)
    ax.tick_params(axis="both", labelsize=fk, colors=MUTED, length=0, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRIDC)
    ax.grid(color=GRIDC, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhspan(-FLOOR_P90, FLOOR_P90, color=GRIDC, alpha=0.8, zorder=1, lw=0)
    ax.axhline(0, color=GRIDC, lw=0.8, zorder=2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(-0.80, 5.05)


def label(ax, x, y, text, dx, dy, ha, va, fs, color=INK, weight="normal"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=fs, color=color, ha=ha, va=va, weight=weight,
                linespacing=1.2, zorder=6)


# ==================================================== main text, 5.39 in ====
# Built wide and short: the paper places it at 0.98 x text width, and the block has
# about 1.7 in of vertical room before it stops fitting on the page it belongs to.
LAB, NOTE = 6.8, 6.6
fig, ax = plt.subplots(figsize=(5.39, 1.66), dpi=400, facecolor=BG)
fig.subplots_adjust(left=0.077, right=0.995, top=0.745, bottom=0.225)

dress(ax, "Exposure does not set the effect", "sixteen arms, median over 7 storms",
      "peak in-box activation of the ablated group", "deepening removed (hPa)",
      (8.2, 7.0, 6.6))
ax.set_ylim(-0.75, 4.45)

# proportional-erasure null: effect = k * exposure, anchored on the main convection arm
k = P["convection"][1] / P["convection"][0]
xs = np.linspace(0, 47, 40)
ax.plot(xs, k * xs, ls=(0, (4, 2.5)), lw=0.9, color=FAINT, zorder=2)
label(ax, 15, k * 15, "effect $\\propto$ exposure", 0, 7, "center", "bottom", NOTE, FAINT)

# three disjoint groups at matched exposure, 227x apart in effect
ax.axvspan(17.9, 20.9, color=YELLOW, alpha=0.14, zorder=1, lw=0)
ax.text(19.4, 4.40, "matched exposure, 227$\\times$ apart", fontsize=NOTE, color=YELLOW,
        ha="center", va="top", weight="bold", zorder=6)

for n, _, c, _ in ARMS:
    x, y = P[n]
    ax.plot(x, y, MARK[c], ms=4.6 if c != "unexp" else 3.8, mfc=FACE[c], mec=EDGE[c],
            mew=1.1, zorder=4)

label(ax, *P["mech_asc21"],  "asc21",      -9, 0, "right", "center", LAB)
label(ax, *P["mech_ascent"], "ascent",     -9, 1, "right", "center", LAB)
label(ax, *P["mech_asc17"],  "asc17",       9, -1, "left", "center", LAB)
label(ax, *P["convection"],  "convection",  0, 10, "center", "bottom", LAB)
label(ax, *P["mech_asc09"],  "asc09",       9, 0, "left", "center", LAB)
label(ax, *P["mech_atm_river"], "atm river", 0, 9, "center", "bottom", LAB)
label(ax, *P["mech_vort850"], "vort850",    2, 9, "left", "bottom", LAB)
ax.text(47, 0.02, "noise floor", fontsize=NOTE - 0.3, color=FAINT, ha="right",
        va="center", zorder=6)

ax.set_xlim(-3.0, 48)
hand = [
    plt.Line2D([], [], ls="", marker="o", ms=4.2, mfc=BLUE, mec=BLUE, label="ascent"),
    plt.Line2D([], [], ls="", marker="s", ms=4.2, mfc=GREEN, mec=GREEN, label="other"),
    plt.Line2D([], [], ls="", marker="x", ms=3.8, mfc="none", mec=GREY, label="no exposure"),
]
leg = ax.legend(handles=hand, fontsize=NOTE, loc="upper left", frameon=False,
                handletextpad=0.3, labelspacing=0.24, borderpad=0.1)
for t in leg.get_texts():
    t.set_color(MUTED)

out = ROOT / "figures/paper_fig_exposure.png"
fig.savefig(out, facecolor=BG, dpi=400)
print("->", out)

# ====================================================== appendix, 3.00 in ==
LAB, NOTE = 6.3, 6.1
fig2, ax2 = plt.subplots(figsize=(3.00, 2.30), dpi=400, facecolor=BG)
fig2.subplots_adjust(left=0.150, right=0.985, top=0.805, bottom=0.165)

dress(ax2, "The dose-response does not hold", "same arms, ranked by calibrated label",
      "calibrated ascent score ($\\sigma$)", "deepening removed (hPa)", (8.0, 6.7, 6.3))

# the moisture family: three groups selected as moisture, differing only in how much
# ascent contamination they carry. the only place the monotone reading survives.
FAMILY = ["moisture2", "mech_q600", "moisture"]
ax2.plot([SIG[n] for n in FAMILY], [P[n][1] for n in FAMILY], "-", lw=1.0,
         color=GREEN, alpha=0.6, zorder=3)
ax2.text(0.2, 4.98, "moisture family:\nmonotone in ascent\ncontamination", fontsize=NOTE,
         color=GREEN, ha="left", va="top", linespacing=1.25, zorder=6)

for n, _, c, s in ARMS:
    if s is None:
        continue
    ax2.plot(s, P[n][1], MARK[c], ms=4.4, mfc=FACE[c], mec=EDGE[c], mew=1.1, zorder=4)

label(ax2, 20.6, P["mech_asc21"][1],  "asc21",       0, 10, "center", "bottom", LAB)
label(ax2, 28.5, P["convection"][1],  "convection", -9, 2, "right", "center", LAB)
label(ax2, 28.8, P["mech_ascent"][1], "ascent",     -9, 0, "right", "center", LAB)
# the non-ascent arms are named in the table directly above this figure; labelling them
# here only crowds the region the moisture family occupies
label(ax2,  9.0, P["mech_asc09"][1],  "asc09",       0, 10, "center", "bottom", LAB)

# the break: a group built to fill the gap came back inside the noise floor
ax2.annotate("asc17: null\nat 16.8$\\sigma$", (16.8, P["mech_asc17"][1]),
             textcoords="offset points", xytext=(0, 30), fontsize=LAB, color=YELLOW,
             ha="center", va="bottom", weight="bold", linespacing=1.25,
             arrowprops=dict(arrowstyle="-", lw=0.8, color=YELLOW, shrinkA=2, shrinkB=3),
             zorder=6)

ax2.set_xlim(-2.0, 33)
ax2.set_xticks([0, 10, 20, 30])

out2 = ROOT / "figures/paper_fig_dose.png"
fig2.savefig(out2, facecolor=BG, dpi=400)
print("->", out2)

# ------------------------------------------------------------- numbers ------
from scipy import stats
ex = [(n, *P[n]) for n, _, _, _ in ARMS if P[n][0] > 1.0]
print("exposure vs effect, %d exposed arms: rho=%+.2f p=%.3f"
      % (len(ex), *stats.spearmanr([t[1] for t in ex], [t[2] for t in ex])))
sg = [(s, P[n][1]) for n, _, _, s in ARMS if s is not None]
print("ascent sigma vs effect, %d labelled arms: rho=%+.2f p=%.3f"
      % (len(sg), *stats.spearmanr([t[0] for t in sg], [t[1] for t in sg])))
for n in ["mech_asc17", "mech_asc21", "mech_ascent"]:
    print("  matched window: %-12s exposure %5.2f -> %6.3f hPa" % (n, *P[n]))
