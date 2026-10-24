"""Figure 2: the convection group from the J1 controlled gain sweep (panels a-c) and the
low-level-spin feature 3316 gain sweep (panel d).

Panels a-c draw every storm at every gain, against the in-box control at the same gain, so the
figure shows the outcome distribution rather than the best storm. Panel d is the single spin
feature on the three storms it was swept on, with its random-feature control arms.

Data: results/skill/suff_ctl/rev_verdict.json (a-c), results/skill/gain_3316/run_*.npy +
era5_truth.npy (d, prereg docs/prereg/prereg_gain_3316.md); floor from J4 (0.25 hPa).
Run from the repository root: python figures/main_claims/fig2_interventions.py

Outputs: figures/main_claims/fig2_interventions.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREY, PALE

TITLE, SUB, TICK, LAB, NOTE = 10.0, 8.6, 8.4, 8.8, 8.0
SALMON = "#b5735a"   # storms that overshoot
CTRL = "#9b9b9b"

V = json.load(open(ROOT / "results/skill/suff_ctl/rev_verdict.json"))
M, FLOOR = V["metrics"], V["floor"]
STORMS = [s for s in M if not M[s]["nondev"]]
NICE = {"ida2021": "Ida", "michael2018": "Michael", "haishen2020": "Haishen",
        "goni2020": "Goni", "haiyan2013": "Haiyan", "patricia2015": "Patricia",
        "wilma2005": "Wilma"}
GAINS = [0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3]
OVER = {"haishen2020", "haiyan2013"}

# data gate: seven developers, every arm present, finite, gain 1 is the baseline
assert len(STORMS) == 7, STORMS
for s in STORMS:
    a = M[s]["arms"]
    for g in GAINS:
        for arm in ("gain-%g" % g, "rand-gain-%g" % g):
            if arm == "rand-gain-1" and arm not in a:
                continue
            assert arm in a and np.isfinite(a[arm]["dep_err"]), (s, arm)
    assert a["gain-1"]["deepen"] == a["baseline"]["deepen"], s


def arm(s, g, ctl=False):
    a = M[s]["arms"]
    if g == 1:
        return a["baseline"]
    return a[("rand-gain-%g" if ctl else "gain-%g") % g]


def err_red(s, g, ctl=False):   # reduction in |deepening error| vs baseline, hPa
    return M[s]["arms"]["baseline"]["dep_err"] - arm(s, g, ctl)["dep_err"]


def pct_era(s, g):
    return 100 * arm(s, g)["deepen"] / M[s]["era_deepen"]


# d: feature 3316 sweep. Same deepening-error metric as rev_analyze.arm_metrics
SPIN = ROOT / "results/skill/gain_3316"
SPIN_STORMS = ["ida2021", "haishen2020", "patricia2015"]
SPIN_GAINS = [0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
SPIN_TRUTH = np.load(SPIN / "era5_truth.npy", allow_pickle=True).item()


def spin_dep_err(s, arm_name):
    t = np.asarray(SPIN_TRUTH[s]["mslp_min"]); ic = float(t[0])
    m = np.asarray(np.load(SPIN / f"run_{s}.npy", allow_pickle=True).item()["res"][arm_name]["mslp_min"])
    assert np.isfinite(m).all(), (s, arm_name)
    return abs((ic - t.min()) - (ic - m.min()))


def spin_red(s, g):
    a = "baseline" if g == 1 else "gain-%g" % g
    return spin_dep_err(s, "baseline") - spin_dep_err(s, a)


def dress(ax, title, sub, xlabel="", ylabel="", grid="y"):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=TITLE, color=INK, weight="bold", pad=16, loc="left")
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=SUB, color=MUTED,
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


def gain_axis(ax):
    ax.set_xlim(-.12, 3.25); ax.set_xticks([0, 1, 2, 3], ["0", "×1", "×2", "×3"])
    ax.axvline(1, color=GRIDC, lw=.8, zorder=1)


def end_labels(ax, ends, minsep, over=OVER):
    """Label each storm at the right end of its line, spread apart by minsep (data units)."""
    ends = sorted(ends, key=lambda kv: kv[1])
    pos = [y for _, y in ends]
    for _ in range(50):
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < minsep:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - minsep / 2, mid + minsep / 2
    for (s, _), y in zip(ends, pos):
        ax.text(GAINS[-1] + .07, y, NICE[s], fontsize=NOTE - .6, va="center", ha="left",
                color=SALMON if s in over else FAINT)


def figure2():
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.75), facecolor=BG,
                             gridspec_kw=dict(width_ratios=[.92, 1.1, 1.1, 1.1]))

    # a: restore to normal, treatment vs in-box control, per storm
    ax = axes[0]
    conv = np.array([arm(s, 0)["d_deepen"] for s in STORMS])
    ctrl = np.array([arm(s, 0, ctl=True)["d_deepen"] for s in STORMS])
    order = np.argsort(conv); y = np.arange(7)
    ax.axvspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    for yi, i in zip(y, order):
        ax.plot([ctrl[i], conv[i]], [yi, yi], color=PALE, lw=.8, zorder=2)
    ax.scatter(ctrl[order], y, s=13, facecolor=BG, edgecolor=GREY, linewidth=.8,
               zorder=4, label="in-box control")
    ax.scatter(conv[order], y, s=14, color=BLUE, linewidth=0, zorder=4, label="convection")
    ax.set_yticks(y); ax.set_yticklabels([NICE[STORMS[i]] for i in order],
                                         fontsize=TICK, color=INK)
    ax.set_xlim(-1.1, 9.2); ax.set_xticks([0, 2, 4, 6, 8])
    dress(ax, "a  Restore to normal",
          "$\\alpha=0$; median %.1f vs %.1f hPa" % (np.median(conv), np.median(ctrl)),
          "deepening removed (hPa)", grid="x")
    ax.legend(frameon=False, fontsize=NOTE, loc="lower right", handletextpad=.2,
              labelspacing=.2, borderpad=0)

    # b: change in deepening error vs gain, every storm
    ax = axes[1]
    ax.axhspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    ax.axhline(0, color=GREY, lw=.6, zorder=2)
    ends = []
    for s in STORMS:
        e = [err_red(s, g) for g in GAINS]
        ax.plot(GAINS, e, color=SALMON if s in OVER else PALE, lw=.9, zorder=3)
        ends.append((s, e[-1]))
    med = [np.median([err_red(s, g) for s in STORMS]) for g in GAINS]
    cmed = [np.median([err_red(s, g, ctl=True) for s in STORMS]) for g in GAINS]
    ax.plot(GAINS, cmed, color=CTRL, lw=1.0, ls=(0, (3, 2)), zorder=4, label="control median")
    ax.plot(GAINS, med, color=BLUE, lw=1.6, marker="o", ms=2.6, zorder=5, label="median")
    end_labels(ax, ends, 2.6)
    gain_axis(ax); ax.set_ylim(-17, 25)
    n2 = sum(err_red(s, 2) > FLOOR for s in STORMS)
    dress(ax, "b  Amplification: every storm",
          "%d of 7 improve at $\\alpha=2$" % n2,
          "convection scaling ($\\alpha$)", "deepening error reduced (hPa)")
    ax.legend(frameon=False, fontsize=NOTE, loc="upper left", handlelength=1.6,
              labelspacing=.2, borderpad=0)

    # c: deepening as a share of ERA5, showing the overshoots
    ax = axes[2]
    ax.axhline(100, color=INK, lw=.7, ls=(0, (2, 2)), zorder=2)
    ax.text(-.05, 103, "ERA5", fontsize=NOTE, color=INK, va="bottom")
    ends = []
    for s in STORMS:
        p = [pct_era(s, g) for g in GAINS]
        ax.plot(GAINS, p, color=SALMON if s in OVER else PALE, lw=.9, zorder=3)
        ends.append((s, p[-1]))
    end_labels(ax, ends, 13)
    pm = [np.median([pct_era(s, g) for s in STORMS]) for g in GAINS]
    ax.plot(GAINS, pm, color=BLUE, lw=1.6, marker="o", ms=2.6, zorder=5)
    gain_axis(ax); ax.set_ylim(0, 230); ax.set_yticks([0, 50, 100, 150, 200])
    dress(ax, "c  Overshoot",
          "near-ERA5 storms overshoot",
          "convection scaling ($\\alpha$)", "deepening (% of ERA5)")

    # d: low-level spin feature 3316, three storms, random-feature control at alpha = 0 and 3
    ax = axes[3]
    ax.axhspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    ax.axhline(0, color=GREY, lw=.6, zorder=2)
    ends = []
    for s in SPIN_STORMS:
        e = [spin_red(s, g) for g in SPIN_GAINS]
        ax.plot(SPIN_GAINS, e, color=BLUE, lw=1.1, marker="o", ms=2.2, zorder=4)
        ends.append((s, e[-1]))
        for g, arm_name in ((0, "rand-normal"), (3, "rand-gain-3")):
            ax.scatter([g], [spin_dep_err(s, "baseline") - spin_dep_err(s, arm_name)], s=13,
                       facecolor=BG, edgecolor=GREY, linewidth=.8, zorder=5)
    ax.scatter([], [], s=13, facecolor=BG, edgecolor=GREY, linewidth=.8, label="random control")
    end_labels(ax, ends, 1.4, over=())
    gain_axis(ax); ax.set_ylim(-9, 15)
    dress(ax, "d  Low-level spin (f3316)",
          "three storms; no overshoot to $\\alpha=3$",
          "spin scaling ($\\alpha$)", "deepening error reduced (hPa)")
    ax.legend(frameon=False, fontsize=NOTE, loc="upper left", handletextpad=.2, borderpad=0)

    fig.subplots_adjust(left=.065, right=.965, top=.80, bottom=.17, wspace=.52)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"fig2_interventions.{ext}", facecolor=BG,
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print("a median", np.median(conv), np.median(ctrl))
    print("b median by gain", dict(zip(GAINS, np.round(med, 2))))
    print("b control median", dict(zip(GAINS, np.round(cmed, 2))))
    print("c median %ERA5", dict(zip(GAINS, np.round(pm, 0))))
    print("d spin 3316", {NICE[s]: [round(spin_red(s, g), 2) for g in SPIN_GAINS] for s in SPIN_STORMS})


if __name__ == "__main__":
    figure2()
