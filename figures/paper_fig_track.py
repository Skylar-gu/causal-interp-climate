"""Track and propagation: the spatial readout the scalar battery cannot give.

Two figures from results/skill/fields_conv/run_*.npy (MECH_FIELDS=1).

figG  PROPAGATION. Sea-level pressure over the storm box at three leads, for the untouched
      forecast, the ablated forecast (g=0) and the amplified one (g=2), with the ERA5 track
      overlaid. This is what "costs 41% of the deepening" and "recovers it" actually look
      like on a map.

figH  TRACK vs INTENSITY. Operational centres verify a cyclone forecast on two axes and
      data-driven models behave very differently on them: position is usually good, peak
      intensity is not. If the convection edit is a genuine intensity lever it should move
      the intensity axis and leave the track axis alone. That is a falsifiable prediction
      and this figure is the test.

TRACKING. Both model and ERA5 centres are re-derived here, from the stored fields, with
ONE estimator: the MSLP minimum inside a search radius of the previous centre, seeded at
the storm's configured position. The raw box argmin is not usable -- these boxes are large
enough to contain other lows, and on Ida it locks onto a Central American low at lead 0.
Using the same constrained tracker on both sides is what makes the two tracks comparable.

    python3 figures/paper_fig_track.py

Inputs: results/skill/fields_conv/run_<storm>.npy -- NOT shipped; regenerate with `MECH_RES=fields_conv MECH_FIELDS=1 MECH_GAINS=0,2 FS_DEVICE=gpu python -m graphcast_sae.storms.skill_conv_run`. results/skill/era5_track.npy is shipped (graphcast_sae.storms.era5_track).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
W = 5.25

BG = "#0d0616"
OCEAN = "#080410"
INK, MUTED, FAINT = "#f2e9f5", "#a794b5", "#6b5b8c"
EMBER, COOL, GRIDC, DEAD = "#fe9f6d", "#b73779", "#241a33", "#3a2f4d"
TITLE, SUB, TICK, NOTE = 7.4, 6.0, 5.6, 5.7

RES = ROOT / "results/skill/fields_conv"
SEED_R = 600.0      # km, lead 0: how far the box minimum may sit from the catalogued centre
STEP_R = 450.0      # km, per 6 h: 21 m/s, comfortably above any real TC translation speed

ARMS = [("gain-0", "convection removed", COOL),
        ("gain-1.25", "×1.25", "#fed0a8"),
        ("gain-2", "amplified ×2", EMBER)]


def gc_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def track(mslp, lat, lon, seed):
    """Constrained MSLP-minimum tracker. mslp (T, nlat, nlon); seed (lat, lon) in deg."""
    LO, LA = np.meshgrid(lon, lat)
    clat, clon, cmin = [], [], []
    prev, rad = seed, SEED_R
    for k in range(mslp.shape[0]):
        d = gc_km(LA, LO, prev[0], prev[1])
        m = np.where(d <= rad, mslp[k], np.inf)
        if not np.isfinite(m).any():                 # radius escaped the box: widen once
            m = np.where(d <= 2 * rad, mslp[k], np.inf)
        j, i = np.unravel_index(int(np.argmin(m)), m.shape)
        prev = (float(LA[j, i]), float(LO[j, i]))
        rad = STEP_R
        clat.append(prev[0]); clon.append(prev[1]); cmin.append(float(mslp[k][j, i]))
    return np.array(clat), np.array(clon), np.array(cmin)


def norm180(x):
    return np.where(np.asarray(x) > 180, np.asarray(x) - 360, np.asarray(x))


# --------------------------------------------------------------------------- load ---
era = np.load(ROOT / "results/skill/era5_track.npy", allow_pickle=True).item()
runs = {}
for f in sorted(RES.glob("run_*.npy")):
    d = np.load(f, allow_pickle=True).item()
    if "fields" not in d["res"]["baseline"]:
        print(f"  {f.name}: no fields stored, skipping"); continue
    runs[d["name"]] = d
if not runs:
    raise SystemExit(f"no field runs in {RES} yet -- wait for run_fields.sh")
print("storms with fields:", list(runs))

TRK = {}
for name, d in runs.items():
    seed = (d["center"][0], d["center"][1] % 360.0)
    e = era[name]
    et = track(e["mslp"], e["grid_lat"], e["grid_lon"], seed)
    per = {}
    for arm, r in d["res"].items():
        fl = r["fields"]
        per[arm] = track(fl["mslp"], fl["grid_lat"], fl["grid_lon"], seed)
    TRK[name] = dict(era=et, arm=per, grid=(d["res"]["baseline"]["fields"]["grid_lat"],
                                            d["res"]["baseline"]["fields"]["grid_lon"]))
    ea, eo, em = et
    ba, bo, bm = per["baseline"]
    n = min(len(em), len(bm))
    print(f"  {name}: ERA5 {em[:n].min():.0f} hPa, baseline {bm[:n].min():.0f} hPa, "
          f"mean track err {np.mean(gc_km(ba[:n], bo[:n], ea[:n], eo[:n])):.0f} km")

# ======================================================================== FIGURE G ===
HERO = "ida2021" if "ida2021" in runs else sorted(runs)[0]
d = runs[HERO]
e = era[HERO]
LEADS = [4, 8, 12]                                   # +24, +48, +72 h
ROWS = [("baseline", "untouched"), ("gain-0", "convection removed"),
        ("gain-2", "convection amplified ×2")]
ROWS = [(a, t) for a, t in ROWS if a in d["res"]]

glat = d["res"]["baseline"]["fields"]["grid_lat"]
glon = d["res"]["baseline"]["fields"]["grid_lon"]
allm = np.concatenate([d["res"][a]["fields"]["mslp"][LEADS].ravel() for a, _ in ROWS])
vmin, vmax = float(np.percentile(allm, 0.2)), float(np.percentile(allm, 99.8))

fig, axes = plt.subplots(len(ROWS), len(LEADS), figsize=(W, 0.92 * len(ROWS) + 0.52),
                         facecolor=BG, squeeze=False)
for ri, (arm, rlab) in enumerate(ROWS):
    fl = d["res"][arm]["fields"]
    ta, to, tm = TRK[HERO]["arm"][arm]
    ea, eo, _ = TRK[HERO]["era"]
    for ci, h in enumerate(LEADS):
        ax = axes[ri][ci]
        ax.set_facecolor(OCEAN)
        ax.pcolormesh(glon, glat, fl["mslp"][h], cmap="magma_r", vmin=vmin, vmax=vmax,
                      shading="auto", zorder=1, rasterized=True)
        ax.contour(glon, glat, fl["mslp"][h], levels=np.arange(940, 1030, 4),
                   colors="#ffffff", linewidths=0.22, alpha=0.45, zorder=3)
        ax.plot(eo[:h + 1], ea[:h + 1], color="#7fe3c0", lw=0.7, zorder=5)
        ax.plot(eo[h], ea[h], marker="o", ms=2.4, mfc="none", mec="#7fe3c0", mew=0.8, zorder=6)
        ax.plot(to[:h + 1], ta[:h + 1], color=INK, lw=0.7, ls=(0, (2.5, 1.6)), zorder=5)
        ax.plot(to[h], ta[h], marker="o", ms=2.4, mfc=INK, mec="none", zorder=6)
        ax.text(0.035, 0.90, f"{tm[h]:.0f} hPa", transform=ax.transAxes, fontsize=NOTE,
                color=INK, va="top", ha="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRIDC); s.set_linewidth(0.5)
        if ri == 0:
            ax.set_title(f"+{6*(h+1)} h", fontsize=SUB, color=MUTED, pad=3)
        if ci == 0:
            ax.set_ylabel(rlab, fontsize=SUB, color=INK, labelpad=3)
fig.text(0.012, 0.978, "Hurricane Ida: sea-level pressure, and where the storm goes",
         fontsize=TITLE, color=INK, weight="bold", va="top")
fig.text(0.012, 0.018,
         "green = observed track (ERA5)      white dashed = the model's own track      "
         "label = the model's minimum pressure",
         fontsize=NOTE - 0.4, color=FAINT, ha="left", va="bottom")
fig.subplots_adjust(top=0.885, bottom=0.085, left=0.135, right=0.99, wspace=0.05, hspace=0.10)
fig.savefig(ROOT / "figures/paper_fig_prop.png", dpi=400, facecolor=BG)
print("wrote figures/paper_fig_prop.png")

# ======================================================================== FIGURE H ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.05), facecolor=BG)


def dress(ax, title, sub, xlabel, ylabel):
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


lead = None
for name in runs:
    t = TRK[name]
    ea, eo, em = t["era"]
    for arm, lab, col in [("baseline", "untouched", INK)] + ARMS:
        if arm not in t["arm"]:
            continue
        ta, to, tm = t["arm"][arm]
        n = min(len(tm), len(em))
        lead = (np.arange(n) + 1) * 6.0
        terr = gc_km(ta[:n], to[:n], ea[:n], eo[:n])
        ierr = np.abs(tm[:n] - em[:n])
        ls = "-" if name == HERO else (0, (2.2, 1.4))
        ax1.plot(lead, terr, color=col, lw=1.0, ls=ls, zorder=3,
                 label=lab if name == HERO else None)
        ax2.plot(lead, ierr, color=col, lw=1.0, ls=ls, zorder=3)
ax1.legend(fontsize=NOTE - 0.3, frameon=False, labelcolor=MUTED, loc="upper left",
           handlelength=1.6, borderpad=0, labelspacing=0.22)
for ax in (ax1, ax2):
    ax.set_xlim(0, 6 * 16 + 2)
    ax.set_xticks([0, 24, 48, 72, 96])
dress(ax1, "Position", "great-circle distance from the observed centre",
      "lead time (h)", "track error (km)")
dress(ax2, "Intensity", "absolute error in minimum pressure",
      "lead time (h)", "MSLP error (hPa)")
fig.text(0.012, 0.018,
         "solid = Hurricane Ida, dashed = the other storms.  The edit moves the right-hand "
         "panel and leaves the left-hand one alone.",
         fontsize=NOTE - 0.4, color=FAINT, ha="left", va="bottom")
fig.subplots_adjust(top=0.745, bottom=0.215, left=0.095, right=0.99, wspace=0.28)
fig.savefig(ROOT / "figures/paper_fig_trackintensity.png", dpi=400, facecolor=BG)
print("wrote figures/paper_fig_trackintensity.png")

# ------------------------------------------------------------------ the numbers -----
print("\nmean over leads +24..+96 h, model minus observed:")
print(f"{'storm':<14}{'arm':<12}{'track err km':>14}{'MSLP err hPa':>14}")
for name in runs:
    t = TRK[name]
    ea, eo, em = t["era"]
    for arm in (["baseline"] + [a for a, _, _ in ARMS]
                + ["rand-normal", "rand-gain-2"]):   # controls: same operator, non-mechanism group
        if arm not in t["arm"]:
            continue
        ta, to, tm = t["arm"][arm]
        n = min(len(tm), len(em))
        sl = slice(3, n)
        print(f"{name:<14}{arm:<12}"
              f"{np.mean(gc_km(ta[sl], to[sl], ea[sl], eo[sl])):>14.0f}"
              f"{np.mean(np.abs(tm[sl] - em[sl])):>14.2f}")
