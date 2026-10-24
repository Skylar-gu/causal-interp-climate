"""Render the 'dial up each ingredient' progression figure (paper Fig. 3) from fs_ida_mechmaps_prog.npy.

Layout (4 rows x 5 cols):
  row 0: BASELINE (free-running forecast) — text card | cyclone feature f3243 at +12/+24/+36/+48 h
  rows 1-2: convection, vorticity — the mechanism's own activation map (where its
  features fire, +30 h, baseline run) with the feature ids under the label,
  then the SAME four leads with that mechanism's features dialled up 2x.
  row 3: ERA5-FORCED (perfect forecast) — teacher-forced rollout on the true ERA5
  states (d["tc_truth"]); drawn only when the input file carries it.
Coastlines from the 0.25-degree land-sea mask; the box is the Gulf of
Mexico / Caribbean (0-40N, 110-40W). Light background, RdPu ramp for the cyclone feature.

Run:  python figures/paper_fig_ida_dialup.py   (matplotlib + scipy; inputs ship in results/)
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parent.parent
import os
TAG = os.environ.get("MECHMAPS_TAG", "_v2")   # "_v2" = calibrated groups (paper); "" = the original 39c8e9b groups
d = np.load(ROOT / f"results/fs_ida_mechmaps_prog{TAG}.npy", allow_pickle=True).item()
lsm = np.load(ROOT / "results/land_sea_mask_025.npz")

mlat, mlon = d["mlat"], d["mlon"]
reg, leads = d["reg"], d["leads"]
MECH = d["mech"]

ROWCOL = {"convection": "#c0392b", "vorticity": "#c07f10", "era5": "#2e7d5a"}
VERDICT = lambda p: ("" if p >= 5 else            # no verdict word for a positive response
                     "flat" if p > -5 else "weakens")

# interpolation grid over the box
glon = np.arange(reg["lon"][0], reg["lon"][1] + .01, 0.4)
glat = np.arange(reg["lat"][0], reg["lat"][1] + .01, 0.4)
GLON, GLAT = np.meshgrid(glon, glat)

def gridmap(v):
    g = griddata((mlon, mlat), v, (GLON, GLAT), method="linear")
    return np.nan_to_num(g, nan=0.0)

# land-sea mask subgrid for coastline contour
llon = lsm["lon"]; llon = np.where(llon > 180, llon - 360, llon)
order = np.argsort(llon)
sel_lon = (llon[order] >= reg["lon"][0]) & (llon[order] <= reg["lon"][1])
sel_lat = (lsm["lat"] >= reg["lat"][0]) & (lsm["lat"] <= reg["lat"][1])
L = lsm["lsm"][:, order][np.ix_(sel_lat, sel_lon)]
Llat = lsm["lat"][sel_lat]; Llon = llon[order][sel_lon]

def coast(ax):
    ax.contour(Llon, Llat, L, levels=[0.5], colors=["#9aa8a0"], linewidths=0.6, alpha=0.85)

vmax_tc = max(float(np.max(d["tc_base"][lead])) for lead in leads)
for k in ("tc_convection", "tc_vorticity", "tc_truth"):
    if k in d:
        vmax_tc = max(vmax_tc, max(float(np.max(d[k][lead])) for lead in leads))

# moisture / shear rows dropped (global-footprint dose, sign is a patch artifact)
rows = ["baseline", "convection", "vorticity"] + (["era5"] if "tc_truth" in d else [])
if "tc_truth" not in d:
    print("note: no tc_truth in the input; the ERA5-forced row is omitted")
fig, axes = plt.subplots(len(rows), 5, figsize=(15.6, 2.55 * len(rows) + 0.9))
plt.subplots_adjust(left=.055, right=.985, top=.955, bottom=.072, wspace=.06, hspace=.30)

for i, row in enumerate(rows):
    # ── left column ──
    ax = axes[i, 0]
    if row == "baseline":
        ax.axis("off")
        ax.text(.5, .60, "BASELINE\n(free-running\nforecast)", transform=ax.transAxes, ha="center",
                va="center", fontsize=11.5, fontweight="bold", color="#333333")
    elif row == "era5":
        ax.axis("off")
        ax.text(.5, .60, "ERA5-FORCED", transform=ax.transAxes, ha="center",
                va="center", fontsize=11.5, fontweight="bold", color=ROWCOL["era5"])
    else:
        mm = gridmap(d["mech_map"][row])
        ax.imshow(mm, origin="lower", extent=[*reg["lon"], *reg["lat"]],
                  cmap="GnBu", vmin=0, vmax=max(mm.max(), 1e-6), aspect="auto")
        coast(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#555555")
        col = ROWCOL[row]
        ax.text(-.11, .5, row.upper(), transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=12, fontweight="bold", color=col)
        ax.text(-.045, .5, " · ".join(f"f{f}" for f in MECH[row]), transform=ax.transAxes,
                rotation=90, va="center", ha="center", fontsize=8, color=col, alpha=.85)
        if i == 1:
            ax.set_title("feature activations\n(baseline, +30 h)",
                         fontsize=10.5, color="#333333", pad=8)

    # ── progression columns ──
    key = {"baseline": "tc_base", "era5": "tc_truth"}.get(row, f"tc_{row}")
    for j, lead in enumerate(leads):
        ax = axes[i, j + 1]
        tc = gridmap(d[key][lead])
        ax.imshow(tc, origin="lower", extent=[*reg["lon"], *reg["lat"]],
                  cmap="RdPu", vmin=0, vmax=vmax_tc, aspect="auto")
        coast(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#555555")
        if i == 0:
            ax.set_title(f"+{lead} h", fontsize=12, color="#333333", pad=8)

    # per-row annotation: strength at +48 h vs the free-running baseline
    s48 = float(np.sum(d[key][48])); b48 = float(np.sum(d["tc_base"][48]))
    if row == "baseline":
        note, col = f"cyclone feature strength {b48:.0f}", "#666666"
    elif row == "era5":
        note, col = f"cyclone feature strength {s48:.0f}", ROWCOL["era5"]
    else:
        pct = 100 * (s48 - b48) / b48
        note, col = f"{b48:.0f} → {s48:.0f}   ·   {pct:+.0f}%  {VERDICT(pct)}".rstrip(), ROWCOL[row]
    axes[i, 4].text(1.0, -0.14, note, transform=axes[i, 4].transAxes,
                    ha="right", fontsize=11, fontweight="bold", color=col)

for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/paper_fig_ida_dialup.{ext}", dpi=110 if ext == "png" else None,
                facecolor="white")
print("-> figures/paper_fig_ida_dialup.pdf / .png")
