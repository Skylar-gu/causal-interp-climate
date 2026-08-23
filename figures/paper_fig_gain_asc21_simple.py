"""Forecast-error gain chart for the disjoint ascent group (f553/f866/f1981).

The four displayed settings are:
  Off     = gain 0, anomalous excess restored to the quiet-day normal level
  Normal  = the untouched GraphCast forecast
  2x/3x   = anomalous excess amplified by factors of two/three

Inputs: results/skill/gain_asc21/{run_<storm>.npy, era5_truth.npy} -- NOT shipped (only gain_conv is); regenerate with `MECH_RES=gain_asc21 MECH_FEATS=553,866,1981 MECH_GAINS=0,2,3 FS_DEVICE=gpu python -m graphcast_sae.storms.skill_conv_run` after `MECH_RES=gain_asc21 python -m graphcast_sae.storms.skill_conv_verify_era5`.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import BG, INK, MUTED, GRIDC, BLUE, GREEN, YELLOW

RESULTS = ROOT / "results/skill/gain_asc21"
OUT = ROOT / "figures/paper_fig_gain_asc21_simple.png"

SETTINGS = [
    ("Off", "gain-0"),
    ("Normal", "baseline"),
    (r"$2\times$", "gain-2"),
    (r"$3\times$", "gain-3"),
]
STORMS = [
    ("ida2021", "Ida", BLUE, "o"),
    ("patricia2015", "Patricia", YELLOW, "^"),
    ("haishen2020", "Haishen", GREEN, "s"),
]


def forecast_error(run, truth, arm):
    """MSLP RMSE through the observed intensification peak."""
    observed = np.asarray(truth["mslp_min"])
    predicted = np.asarray(run[arm]["mslp_min"])
    window = max(int(np.argmin(observed)), 6) + 1
    n = min(len(observed), len(predicted), window)
    return float(np.sqrt(np.mean((predicted[:n] - observed[:n]) ** 2)))


truth = np.load(RESULTS / "era5_truth.npy", allow_pickle=True).item()
x = np.arange(len(SETTINGS))

fig, ax = plt.subplots(figsize=(5.25, 2.55), facecolor=BG)
ax.set_facecolor(BG)

summary = []
for key, label, color, marker in STORMS:
    run = np.load(RESULTS / f"run_{key}.npy", allow_pickle=True).item()["res"]
    errors = [forecast_error(run, truth[key], arm) for _, arm in SETTINGS]
    ax.plot(x, errors, color=color, lw=1.6, marker=marker, ms=5.0,
            label=label, zorder=3)
    summary.append(f"{label}: " + ", ".join(f"{v:.2f}" for v in errors))

ax.set_xticks(x, [label for label, _ in SETTINGS])
ax.set_ylabel("MSLP forecast error (hPa)", fontsize=8.0, color=MUTED)
ax.set_title("Disjoint ascent features", fontsize=9.2, color=INK,
             weight="bold", loc="left", pad=16)
ax.text(0, 1.025, "intensity error against ERA5; lower is better",
        transform=ax.transAxes, fontsize=7.1, color=MUTED,
        ha="left", va="bottom")
fig.text(0.985, 0.035, "Off = anomalous excess restored to normal",
         fontsize=6.3, color=MUTED, ha="right", va="bottom")
ax.legend(frameon=False, fontsize=7.0, labelcolor=MUTED,
          loc="upper right", handlelength=1.7, borderpad=0, labelspacing=0.35)
ax.grid(axis="y", color=GRIDC, lw=0.7)
ax.set_axisbelow(True)
ax.tick_params(axis="both", labelsize=7.0, colors=MUTED, length=0, pad=3)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRIDC)

fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.24)
fig.savefig(OUT, dpi=400, facecolor=BG)
print(f"wrote {OUT}")
print(" | ".join(summary))
