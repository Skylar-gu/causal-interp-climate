"""Render the 'dial up each ingredient' progression figure from fs_ida_mechmaps_prog.npy.

Layout (5 rows x 5 cols):
  row 0: BASELINE  — text card | cyclone feature f3243 at +12/+24/+36/+48 h
  rows 1-4: one per mechanism — the mechanism's own activation map (where its
  features fire, +30 h, baseline run) with the feature ids under the label,
  then the SAME four leads with that mechanism's features dialled up 2x.
Coastlines from the 0.25-degree land-sea mask; the box is the Gulf of
Mexico / Caribbean (0-40N, 110-40W). Output: figures/art_mechmaps.png (+ .b64).

Run:  python figures/paper_fig_ida_dialup.py   (matplotlib + scipy; inputs ship in results/)
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "results/fs_ida_mechmaps_prog.npy", allow_pickle=True).item()
lsm = np.load(ROOT / "results/land_sea_mask_025.npz")

mlat, mlon = d["mlat"], d["mlon"]
reg, leads = d["reg"], d["leads"]
MECH = d["mech"]

ROWCOL = {"convection": "#c0392b", "vorticity": "#c07f10",
          "moisture": "#5d6d7e", "shear": "#2e6da4"}
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

def coast(ax, dark):
    c = "#8a97a5" if dark else "#9aa8a0"
    ax.contour(Llon, Llat, L, levels=[0.5], colors=[c], linewidths=0.6, alpha=0.85)

vmax_tc = max(float(np.max(d["tc_base"][lead])) for lead in leads)
for m in MECH:
    vmax_tc = max(vmax_tc, max(float(np.max(d[f"tc_{m}"][lead])) for lead in leads))

rows = ["baseline", "convection", "vorticity", "moisture", "shear"]
fig, axes = plt.subplots(len(rows), 5, figsize=(15.6, 13.6))
plt.subplots_adjust(left=.065, right=.985, top=.90, bottom=.045, wspace=.06, hspace=.30)

fig.suptitle("Dial up each ingredient (left) → GraphCast's cyclone feature responds (right)",
             fontsize=17, fontweight="bold", y=0.965)
fig.text(.5, .925, "Hurricane Ida · 48-h rollout from 2021-08-26 00Z · internal cyclone feature f3243 "
         "over the Gulf of Mexico / Caribbean (0–40°N, 110–40°W)",
         ha="center", fontsize=11.5, color="#444444")

for i, row in enumerate(rows):
    # ── left column ──
    ax = axes[i, 0]
    if row == "baseline":
        ax.axis("off")
        ax.text(.5, .60, "BASELINE", transform=ax.transAxes, ha="center",
                fontsize=13, fontweight="bold", color="#333333")
    else:
        mm = gridmap(d["mech_map"][row])
        ax.imshow(mm, origin="lower", extent=[*reg["lon"], *reg["lat"]],
                  cmap="GnBu", vmin=0, vmax=max(mm.max(), 1e-6), aspect="auto")
        coast(ax, dark=False)
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
    key = "tc_base" if row == "baseline" else f"tc_{row}"
    for j, lead in enumerate(leads):
        ax = axes[i, j + 1]
        tc = gridmap(d[key][lead])
        ax.imshow(tc, origin="lower", extent=[*reg["lon"], *reg["lat"]],
                  cmap="inferno", vmin=0, vmax=vmax_tc, aspect="auto")
        coast(ax, dark=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#555555")
        if i == 0:
            ax.set_title(f"+{lead} h", fontsize=12, color="#333333", pad=8)

    # per-row annotation: strength at +48 h vs baseline
    s48 = float(np.sum(d[key][48])); b48 = float(np.sum(d["tc_base"][48]))
    if row == "baseline":
        note, col = f"cyclone feature strength {b48:.0f}", "#666666"
    else:
        pct = 100 * (s48 - b48) / b48
        note, col = f"{b48:.0f} → {s48:.0f}   ·   {pct:+.0f}%  {VERDICT(pct)}".rstrip(), ROWCOL[row]
    axes[i, 4].text(1.0, -0.14, note, transform=axes[i, 4].transAxes,
                    ha="right", fontsize=11, fontweight="bold", color=col)

fig.text(.065, .012, "each row uses the same color scale; the dialled-up run replays the identical "
         "48-h rollout with that row's features doubled wherever they fire",
         fontsize=9, color="#777777")

for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/paper_fig_ida_dialup.{ext}", dpi=110 if ext == "png" else None,
                facecolor="white")
print("-> figures/paper_fig_ida_dialup.pdf / .png")
