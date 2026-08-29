"""Three main-paper figures using the established paper figure grammar.

Follows paper_fig_intervention.py, paper_fig_gain.py, paper_fig_maps.py,
and fig_tracks_artifact.py: final print size, small type, quiet axes, direct labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAVAR = ROOT / "savar"                    # ladder results ship under savar/results/
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREY, PALE

TITLE, SUB, TICK, LAB, NOTE, VAL = 8.2, 6.7, 6.3, 6.8, 5.9, 7.2
AQUA = "#2a94a8"
GREEN_AQUA = "#6fb3a6"  # restrained but clearer sea-glass tone from the colour reference
OCHRE = "#c9862b"       # retained only for Figure 3
MID_TEAL = "#4f8f97"    # between the paper blue and sea-glass green
SALMON_BEIGE = "#c49a86"  # warm complement for null and surrogate controls
DARK_BROWN = "#b58f80"    # Figure 2 control-median ticks


def read_json(path):
    with open(path) as f:
        return json.load(f)


def dress(ax, title, sub, xlabel="", ylabel="", grid="x", title_pad=12, sub_y=1.035):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=TITLE, color=INK, weight="bold", pad=title_pad, loc="left")
    ax.text(0, sub_y, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
            ha="left", va="bottom")
    ax.set_xlabel(xlabel, fontsize=LAB, color=MUTED, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=LAB, color=MUTED, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK, colors=MUTED, length=0, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRIDC); ax.spines[side].set_linewidth(.7)
    ax.grid(axis=grid, color=GRIDC, lw=.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=400 if ext == "png" else None,
                    facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), facecolor=BG,
                             gridspec_kw=dict(width_ratios=[1.18, .94, .88]))
    ax = axes[0]
    ladder = np.load(SAVAR / "results/ladder_gnn/ladder_gnn_eqvar.npy",
                     allow_pickle=True).item()["res"]
    null = np.load(SAVAR / "results/ladder_gnn/nulls_gnn_eqvar.npy",
                   allow_pickle=True).item()["summary"]["R3b|rand"]
    vals = [float(np.mean(ladder[k]["f1"])) for k in ("trueZ", "R0", "R3b")]
    draws = np.asarray(null["draws"], float)
    labels = ["true variables", "per-mode SAE", "mixed SAE", "random feature sets"]
    y = np.arange(4)[::-1]
    colors = [INK, BLUE, MID_TEAL]
    ax.barh(y[:3], vals, height=.34,
            color=[to_rgba(c, .16) for c in colors], edgecolor=colors,
            linewidth=1.15, zorder=3)
    rng = np.random.default_rng(5)
    ax.scatter(draws, y[3] + rng.uniform(-.20, .20, len(draws)), s=3.5,
               color=to_rgba(SALMON_BEIGE, .42), linewidths=0, zorder=3)
    ax.plot([np.mean(draws)] * 2, [y[3]-.27, y[3]+.27], color=SALMON_BEIGE, lw=1.4, zorder=4)
    for yi, v in zip(y[:3], vals):
        ax.text(v+.018, yi, f"{v:.3f}", va="center", fontsize=VAL,
                color=INK, weight="bold")
    ax.text(.075, y[3], f"mean {np.mean(draws):.3f}\np = .31", va="center",
            fontsize=NOTE, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=TICK, color=INK)
    ax.set_xlim(0, .72); ax.set_xticks([0, .2, .4, .6])
    dress(ax, "a   Causal recovery by variable representation",
          "SAVAR GNN; F1 against the specified graph",
          "causal recovery (F1)", title_pad=17, sub_y=1.02)

    ax = axes[1]
    labels = ["recovered graph", "surrogate-resistant", "matched surrogates"]
    east = np.array([81., 93., 42.]); y = np.arange(3)[::-1]
    colors = [MID_TEAL, BLUE, SALMON_BEIGE]
    ax.barh(y, east, height=.34,
            color=[to_rgba(c, .16) for c in colors], edgecolor=colors,
            linewidth=1.15, zorder=3)
    ax.axvline(50, color=GRIDC, lw=.8, ls=(0, (3, 2)), zorder=2)
    for yi, v in zip(y, east):
        ax.text(v+2.2, yi, f"{v:.0f}%", va="center", fontsize=VAL,
                color=INK, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=TICK, color=INK)
    ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100])
    dress(ax, "b   Eastward orientation of recovered edges",
          "GraphCast observational graph and surrogate audit", "eastward edges",
          title_pad=17, sub_y=1.02)

    ax = axes[2]
    labels = ["observational 24-h lag", "direct perturbation,\n$z_{500}$ response",
              "direct perturbation,\nlatent response"]
    speed = np.array([38.5, 10.3744, 12.2956]); y = np.arange(3)[::-1]
    colors = [GREEN_AQUA, BLUE, MID_TEAL]
    ax.barh(y, speed, height=.34,
            color=[to_rgba(c, .18) for c in colors], edgecolor=colors,
            linewidth=1.15, zorder=3)
    for yi, v in zip(y, speed):
        ax.text(v+1, yi, f"{v:.1f}", va="center", fontsize=VAL,
                color=INK, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=TICK, color=INK)
    ax.set_xlim(0, 45); ax.set_xticks([0, 10, 20, 30, 40])
    dress(ax, "c   Propagation speed by measurement protocol",
          "mode 8→9; 3 K temperature-column impulse",
          "propagation speed (m s$^{-1}$)", title_pad=17, sub_y=1.02)

    fig.text(.99, .025,
             "Same estimator in a; b uses the matched-surrogate audit; c compares observational lag with a temperature impulse.",
             fontsize=NOTE, color=FAINT, ha="right", va="bottom")
    fig.subplots_adjust(left=.105, right=.985, top=.78, bottom=.20, wspace=.66)
    save(fig, "figure1_causal_discovery")
    print("figure1", vals, float(np.mean(draws)), east.tolist(), speed.tolist())


STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020",
          "haiyan2013", "patricia2015", "wilma2005"]
NICE = {"ida2021": "Ida", "michael2018": "Michael", "haishen2020": "Haishen",
        "goni2020": "Goni", "haiyan2013": "Haiyan", "patricia2015": "Patricia",
        "wilma2005": "Wilma"}


def effects(directory):
    d = read_json(ROOT / "results/skill" / directory / "verdict.json")["metrics"]
    a = np.array([d[s]["arms"]["conv-normal"]["d_deepen"] for s in STORMS])
    c = np.array([d[s]["arms"]["rand-normal"]["d_deepen"] for s in STORMS])
    return a, c


def gain_curve(storm):
    gains = np.array([0., 1.25, 1.5, 1.75, 2., 2.5, 3.])
    truth = np.load(ROOT / "results/skill/gain_conv/era5_truth.npy",
                    allow_pickle=True).item()[storm]["mslp_min"]
    res = np.load(ROOT / f"results/skill/gain_conv/run_{storm}.npy",
                  allow_pickle=True).item()["res"]
    win = max(int(np.argmin(truth)), 6) + 1
    def rmse(arm):
        m = np.asarray(res[arm]["mslp_min"]); n = min(win, len(m), len(truth))
        return float(np.sqrt(np.mean((m[:n] - np.asarray(truth)[:n]) ** 2)))
    return gains, np.array([rmse("gain-%g" % g) for g in gains]), rmse("baseline")


def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.10), facecolor=BG,
                             gridspec_kw=dict(width_ratios=[1.05, 1.05, .96]))
    ax = axes[0]
    conv, ctrl = effects("convection"); order = np.argsort(conv); y = np.arange(7)
    for yi, i in zip(y, order):
        ax.plot([ctrl[i], conv[i]], [yi, yi], color=PALE, lw=.8, zorder=2)
    ax.scatter(ctrl[order], y, s=15, facecolor=BG, edgecolor=GREY, linewidth=.8,
               zorder=4, label="matched control")
    ax.scatter(conv[order], y, s=16, color=BLUE, linewidth=0, zorder=4,
               label="convection")
    ax.axvline(0, color=GRIDC, lw=.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([NICE[STORMS[i]] for i in order],
                                         fontsize=TICK, color=INK)
    ax.set_xlim(-.55, 8.15); ax.set_xticks([0, 2, 4, 6, 8])
    dress(ax, "a   Effect of convection ablation across storms",
          "seven rapidly intensifying storms; activity-matched controls",
          "deepening removed (hPa)", title_pad=22, sub_y=1.045)
    ax.legend(frameon=False, fontsize=NOTE, loc="lower right", handletextpad=.25,
              labelspacing=.2, borderpad=0)

    ax = axes[1]
    for storm, label, col, mark in [("ida2021", "Ida", BLUE, "o"),
                                     ("haishen2020", "Haishen", GREEN_AQUA, "s")]:
        gains, err, baseline = gain_curve(storm)
        x = np.r_[1., gains[1:]]; yv = np.r_[baseline, err[1:]]
        ax.plot(x, yv, color=col, lw=1.15, marker=mark, ms=2.8, zorder=3, label=label)
        ax.plot([0], [err[0]], marker=mark, ms=3.2, mfc=BG, mec=col, mew=.8, zorder=4)
        j = int(np.argmin(yv)); ax.plot([x[j]], [yv[j]], marker=mark, ms=4.2,
                                        mfc=col, mec=col, zorder=5)
        txt, off = (("7.35 → 3.13", (4, -12)) if storm == "ida2021"
                    else ("4.03 → 3.01", (-4, -5)))
        ax.annotate(txt, (x[j], yv[j]), xytext=off, textcoords="offset points",
                    fontsize=NOTE, color=col,
                    ha="left" if storm == "ida2021" else "right",
                    va="baseline" if storm == "ida2021" else "top")
    ax.axvline(1, color=GRIDC, lw=.8)
    ax.set_xlim(-.12, 3.12); ax.set_xticks([0, 1, 2, 3], ["0", "×1", "×2", "×3"])
    ax.set_ylim(0, 21); ax.set_yticks([0, 5, 10, 15, 20])
    dress(ax, "b   Convection dose–response",
          "MSLP error against ERA5 over the intensification window",
          r"convection scaling ($\alpha$)", "MSLP error (hPa)", grid="y",
          title_pad=22, sub_y=1.045)
    ax.legend(frameon=False, fontsize=NOTE, loc="upper left", handlelength=1.5,
              labelspacing=.2, borderpad=0)

    ax = axes[2]
    groups = [("convection / ascent", "convection", BLUE),
              ("low-level spin", "mech_spin3316", GREEN_AQUA),   # 2026-08-29: was mech_vort850 (0.55 hPa, polar/anti-signed group)
              ("q600 moisture", "moisture2", GREY)]
    vals, controls = [], []
    for _, directory, _ in groups:
        a, c = effects(directory); vals.append(float(np.median(a))); controls.append(float(np.median(c)))
    y = np.arange(3)[::-1]
    ax.axvspan(-.15, .15, color=GRIDC, alpha=.75, lw=0, zorder=1)
    for yi, (_, _, col), v, c in zip(y, groups, vals, controls):
        ax.barh(yi, v, height=.34, color=to_rgba(col, .18), edgecolor=col,
                linewidth=1.15, zorder=3)
        ax.plot([c, c], [yi-.28, yi+.28], color=DARK_BROWN, lw=2.2, zorder=6)
        tx = v + .09 if v >= 0 else .10
        ax.text(tx, yi, f"{v:+.2f}", va="center", ha="left",
                fontsize=VAL, color=INK, weight="bold")
    ax.axvline(0, color=GREY, lw=.65, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([g[0] for g in groups], fontsize=TICK, color=INK)
    ax.set_xlim(-.40, 4.0); ax.set_xticks([0, 1, 2, 3, 4])
    dress(ax, "c   Effects across calibrated feature groups",
          "median deepening response over seven storms",
          "deepening removed (hPa)", title_pad=22, sub_y=1.045)
    control_key = plt.Line2D([], [], marker="|", ms=8, mew=2.0, ls="none",
                             color=DARK_BROWN, label="control median")
    ax.legend(handles=[control_key], frameon=False, fontsize=NOTE, loc="upper right",
              handletextpad=.25, borderpad=0)

    fig.text(.99, .025,
             "Feature excess returned to storm-free normal. Shaded band in c = rollout nondeterminism (±0.15 hPa).",
             fontsize=NOTE, color=FAINT, ha="right", va="bottom")
    fig.subplots_adjust(left=.085, right=.985, top=.78, bottom=.20, wspace=.58)
    save(fig, "figure2_interventions")
    print("figure2", np.median(conv), np.median(ctrl), vals, controls)


def nodes_from(fp, extra, fid):
    src = fp["res"].get(fid, extra["res"].get(fid))
    return np.asarray(src["nodes"] if isinstance(src, dict) else src, int)


def figure3():
    import cartopy.crs as ccrs
    PC = ccrs.PlateCarree()
    fp = np.load(ROOT / "results/fs_footprints.npy", allow_pickle=True).item()
    extra = np.load(ROOT / "results/fs_footprints_extra.npy", allow_pickle=True).item()
    lat = np.asarray(fp["lat"]); lon = np.asarray(fp["lon"])
    lon = np.where(lon > 180, lon - 360, lon)
    fig = plt.figure(figsize=(10.2, 3.35), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1., .28], hspace=.17, wspace=.08,
                          left=.025, right=.985, top=.88, bottom=.08)
    maps = [fig.add_subplot(gs[0, i], projection=PC) for i in range(2)]
    stats = [fig.add_subplot(gs[1, i]) for i in range(2)]
    specs = [
        (maps[0], (2075,), "a   Grid-locked positional feature",
         "f2075 — three fixed bowties", OCHRE, "YlOrBr"),
        (maps[1], (2401, 2067), "b   Convection-associated group",
         "f2401 + f2067 — tropical weather structure", AQUA, "GnBu"),
    ]
    for ax, fids, title, sub, col, cmap in specs:
        ax.set_global(); ax.set_facecolor("#fbfbfb")
        ax.coastlines(resolution="50m", linewidth=.35, color="#a8a8a8", zorder=3)
        ax.gridlines(draw_labels=False, linewidth=.22, color=GRIDC,
                     xlocs=[-120, -60, 0, 60, 120], ylocs=[-60, -30, 0, 30, 60])
        ax.spines["geo"].set_edgecolor(GRIDC); ax.spines["geo"].set_linewidth(.55)
        ax.set_title(title, fontsize=TITLE, color=INK, weight="bold", pad=12, loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
                ha="left", va="bottom")
        for fid in fids:
            n = nodes_from(fp, extra, fid); size = float(np.clip(900/max(len(n), 1), .18, 1.4))
            if fid in extra["res"] and len(extra["res"][fid].get("mean_amp", [])) == len(n):
                amp = np.asarray(extra["res"][fid]["mean_amp"], float); order = np.argsort(amp)
                ax.scatter(lon[n][order], lat[n][order], c=amp[order], cmap=cmap,
                           s=size, marker="s", linewidths=0, transform=PC, zorder=4,
                           rasterized=True)
            else:
                ax.scatter(lon[n], lat[n], c=col, s=size, marker="s", linewidths=0,
                           transform=PC, zorder=4, rasterized=True)
    cards = [("+1.04 m", "z500 error in footprint", "control −0.02 m", OCHRE),
             ("2.79 hPa", "storm deepening removed", "control +0.01 hPa", AQUA)]
    for ax, (value, desc, control, col) in zip(stats, cards):
        ax.axis("off"); ax.axhline(.98, color=GRIDC, lw=.6)
        ax.text(.02, .56, value, transform=ax.transAxes, fontsize=10.2, color=col,
                weight="bold", va="center")
        ax.text(.36, .59, desc, transform=ax.transAxes, fontsize=LAB, color=INK, va="center")
        ax.text(.36, .22, control, transform=ax.transAxes, fontsize=NOTE, color=MUTED, va="center")
    save(fig, "figure3_semantics_and_relevance")
    print("figure3 f2075 +1.039 m; convection +2.794 hPa")


if __name__ == "__main__":
    figure1(); figure2(); figure3()
    print("wrote", HERE)
