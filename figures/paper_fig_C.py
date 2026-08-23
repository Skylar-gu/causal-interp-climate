"""Figures C and D for the intervention paper. Every number is read from results/ at build time.

figC  the mechanism library on the hurricane battery: what each labelled feature group costs
      the forecast, with its own in-box exposure printed alongside. The point of the panel is
      that six arms sit at zero exposure -- the intervention was a no-op, so those bars are
      UNTESTED, not null, and they are drawn differently to say so.

figD  the non-physics lane, scored globally instead of on a storm. z500 RMSE against ERA5 out
      to +120 h for two classes of suspected grid artifact, each against a control matched
      exactly on firing rate. The control is the whole point: deleting any 127 features hurts,
      so the readable quantity is arm-minus-its-control, not arm-minus-baseline.

Built at final print size (5.25 in wide) so point sizes are true on the page.
Dark theme matched to paper_fig_intervention.py.

Inputs: results/fs_global_rmse.npy (shipped) and results/skill/<arm>/verdict.json for the whole mechanism library. Only convection, mech_asc21, mech_shear, mech_vort850 and moisture2 ship; the other arms (mech_ascent, mech_asc17, mech_asc09, mech_atm_river, mech_baroclinicity, mech_blocking, ...) are NOT shipped and are regenerated with `MECH_RES=<arm> MECH_FEATS=<ids> python -m graphcast_sae.storms.skill_conv_run` then `MECH_RES=<arm> python -m graphcast_sae.storms.skill_conv_analyze`.
"""
import json
import os
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
W = 5.25

import sys; sys.path.insert(0, str(Path(__file__).parent))
from paper_palette import BG, INK, MUTED, GRIDC, BLUE, GREEN, YELLOW, GREY, PALE
FAINT = GREY
EMBER, COOL = BLUE, GREEN
DEAD = PALE
LOCKED = YELLOW                        # the grid-locked class
BASEC = "#b0b0b0"                      # the untouched arm, kept out of the way

TITLE, SUB, TICK, YTICK, VAL, NOTE = 7.4, 6.0, 5.6, 6.4, 6.8, 5.7

NAMES = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013",
         "patricia2015", "wilma2005"]


def dress(ax, title, sub, xlabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=TITLE, color=INK, weight="bold", pad=12, loc="left")
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
            ha="left", va="bottom")
    ax.set_xlabel(xlabel, fontsize=SUB, color=MUTED, labelpad=2)
    ax.tick_params(axis="x", labelsize=TICK, colors=MUTED, length=0, pad=2)
    ax.tick_params(axis="y", labelsize=YTICK, colors=INK, length=0, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIDC)
    ax.grid(axis="x", color=GRIDC, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ======================================================== library, read in ===
def arm(name):
    """(median in-box exposure, median d_deepen, median random-control d_deepen)."""
    m = json.load(open(ROOT / f"results/skill/{name}/verdict.json"))["metrics"]
    if not set(NAMES) <= set(m):
        return None
    return (st.median([m[s]["arms"]["baseline"]["conv_box"] for s in NAMES]),
            st.median([m[s]["arms"]["conv-normal"]["d_deepen"] for s in NAMES]),
            st.median([m[s]["arms"]["rand-normal"]["d_deepen"] for s in NAMES]))


# label, directory, calibrated identity for the axis
LIB = [
    ("Ascent, replication", "mech_asc21"),
    ("Ascent (convection)", "convection"),
    ("Ascent, purified", "mech_ascent"),
    ("Ascent, 16.8 sigma", "mech_asc17"),
    ("Ascent, 9.0 sigma", "mech_asc09"),
    ("Moisture flux, unidentified", "mech_atm_river"),
    ("Low-level vorticity", "mech_vort850"),
    ("Mid-level moisture", "moisture2"),
    ("Small, unexplained", "geom_small_unexplained"),
    ("Small, orographic", "geom_small_explained"),
    ("Baroclinicity", "mech_baroclinicity"),
    ("Blocking", "mech_blocking"),
    ("Jet", "mech_jet250"),
    ("Wind shear", "mech_shear"),
]
rows = [(lab, ) + arm(d) for lab, d in LIB if arm(d) is not None]
rows = sorted(rows, key=lambda r: r[2])

# the noise floor of the protocol, measured from the random controls that ran with every arm
FLOOR = max(abs(r[3]) for r in rows)

fig, ax = plt.subplots(figsize=(W, 2.55), facecolor=BG)
y = np.arange(len(rows))
EXPOSED = 1.0                                   # in-box activation below this = no-op
col = [EMBER if (b >= EXPOSED and abs(d) > FLOOR) else
       (FAINT if b >= EXPOSED else DEAD) for _, b, d, _ in rows]
ax.barh(y, [r[2] for r in rows], color=col, height=0.62, zorder=3)
ax.axvspan(-FLOOR, FLOOR, color=GRIDC, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
XE = 5.55                                       # exposure column, fixed x
for i, (_, box, d, _) in enumerate(rows):
    ax.text(d + 0.09, i, "%+.2f" % d, va="center", fontsize=VAL - 0.6,
            color=INK if abs(d) > FLOOR else MUTED)
    ax.text(XE, i, ("%.1f" % box) if box >= 0.05 else "0.0", va="center", ha="right",
            fontsize=NOTE, color=MUTED if box >= EXPOSED else DEAD)
ax.text(XE, len(rows) - 0.3, "exposure", va="bottom", ha="right", fontsize=NOTE,
        color=MUTED, style="italic")
ax.set_xlim(-0.35, 5.7)
ax.set_xticks([0, 1, 2, 3, 4])
dress(ax, "Every mechanism, one at a time, on seven hurricanes",
      "hPa of 96-h deepening lost when the group is returned to its normal level",
      "median Δ deepening (hPa)      shaded band = protocol noise floor")
fig.text(0.30, 0.028,
         "blue = real     grey = exposed and null     pale = never fired in the storm, so untested",
         fontsize=NOTE, color=FAINT, ha="left", va="bottom")
fig.subplots_adjust(top=0.775, bottom=0.175, left=0.30, right=0.955)
fig.savefig(ROOT / "figures/paper_fig_C.png", dpi=400, facecolor=BG)
print("figC: floor %.3f, %d arms" % (FLOOR, len(rows)))

# ============================================================== FIGURE D ====
g = np.load(ROOT / "results/fs_global_rmse.npy", allow_pickle=True).item()
fi = g["fields"].index("z500")
lead = (np.arange(g["S"]) + 1) * 6.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.15), facecolor=BG)

STYLE = [("baseline", "untouched", BASEC, "-", 1.6),
         ("mesh_locked", "mesh-locked (27)", LOCKED, "-", 1.0),
         ("ctrl_mesh", "its matched control", LOCKED, ":", 1.0),
         ("scatter_blob", "wide-scatter (127)", INK, "-", 1.0),
         ("ctrl_blob", "its matched control", INK, ":", 1.0)]
for k, lab, c, ls, lw in STYLE:
    ax1.plot(lead, g["acc"][k][:, fi], color=c, ls=ls, lw=lw, label=lab, zorder=3)
ax1.legend(fontsize=NOTE - 0.3, frameon=False, labelcolor=MUTED, loc="upper left",
           handlelength=1.6, borderpad=0, labelspacing=0.25)
ax1.set_xlim(0, 122)
ax1.set_xticks([0, 24, 48, 72, 96, 120])
dress(ax1, "Deleting the suspected artifacts", "global z500 error against ERA5, 8 forecasts",
      "lead time (h)")
ax1.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)

# --- D-right: the readable quantity, arm minus its own matched control
pairs = [("mesh-locked", "mesh_locked", "ctrl_mesh", LOCKED),
         ("wide-scatter", "scatter_blob", "ctrl_blob", INK)]
for lab, a, c, col_ in pairs:
    ax2.plot(lead, g["acc"][a][:, fi] - g["acc"][c][:, fi], color=col_, lw=1.1,
             label=lab, zorder=3)
ax2.axhline(0, color=MUTED, lw=0.8, zorder=4)
ax2.legend(fontsize=NOTE - 0.3, frameon=False, labelcolor=MUTED, loc="lower left",
           handlelength=1.6, borderpad=0, labelspacing=0.25)
ax2.set_xlim(0, 122)
ax2.set_xticks([0, 24, 48, 72, 96, 120])
dress(ax2, "Minus its matched control",
      "below zero = carries less than random features", "lead time (h)")
ax2.grid(axis="y", color=GRIDC, lw=0.6, zorder=0)

fig.subplots_adjust(top=0.745, bottom=0.20, left=0.095, right=0.985, wspace=0.30)
fig.savefig(ROOT / "figures/paper_fig_D.png", dpi=400, facecolor=BG)
print("figD: z500 @120h  " + "  ".join(
    "%s %.2f" % (k, g["acc"][k][-1, fi]) for k in g["arms"]))
