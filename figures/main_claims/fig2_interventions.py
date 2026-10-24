"""Figure 2, in the submitted three-panel layout, on the pre-registered deepening-error metric.

  a  restore-to-normal (alpha = 0) of the single features 2401 (convection) and 3316 (low-level
     spin), every storm, against the random-feature control: results/skill/mech_{2401,3316}
  b  convection-group gain sweep, every storm, with the in-box control median at the same gain
     (J1): results/skill/suff_ctl/rev_verdict.json
  c  feature 3316 gain sweep on the three storms it was run on, with its random-feature
     control arms: results/skill/gain_3316 (prereg docs/prereg/prereg_gain_3316.md)

Deepening error = |ERA5 deepening - forecast deepening| over 96 h (rev_analyze.arm_metrics);
grey bands are the J4 run-to-run floor (0.25 hPa).
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
SPINC = "#3f8f6b"   # low-level spin (feature 3316)

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


def restore(name):
    m = json.load(open(ROOT / f"results/skill/{name}/verdict.json"))["metrics"]
    return ({x: m[x]["arms"]["conv-normal"]["d_deepen"] for x in STORMS},
            {x: m[x]["arms"]["rand-normal"]["d_deepen"] for x in STORMS})


def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.75), facecolor=BG,
                             gridspec_kw=dict(width_ratios=[.95, 1.1, 1.1]))

    # a: single features restored to normal, every storm, random-feature control
    ax = axes[0]
    conv, cctl = restore("mech_2401"); spin, sctl = restore("mech_3316")
    order = sorted(STORMS, key=lambda x: conv[x]); y = np.arange(len(order))
    ax.axvspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    for yi, x in zip(y, order):
        lo, hi = min(cctl[x], sctl[x], conv[x], spin[x]), max(conv[x], spin[x])
        ax.plot([lo, hi], [yi, yi], color=PALE, lw=.8, zorder=2)
    ax.scatter([cctl[x] for x in order], y + .17, s=12, facecolor=BG, edgecolor=GREY, lw=.8, zorder=4)
    ax.scatter([sctl[x] for x in order], y - .17, s=12, facecolor=BG, edgecolor=GREY, lw=.8, zorder=4,
               label="random control")
    ax.scatter([conv[x] for x in order], y + .17, s=14, color=BLUE, lw=0, zorder=5,
               label="2401 (convection)")
    ax.scatter([spin[x] for x in order], y - .17, s=14, color=SPINC, lw=0, zorder=5,
               label="3316 (low-level spin)")
    ax.set_yticks(y); ax.set_yticklabels([NICE[x] for x in order], fontsize=TICK, color=INK)
    ax.set_xlim(-1.1, 8.2); ax.set_xticks([0, 2, 4, 6, 8])
    dress(ax, "a  Restore to normal",
          "$\\alpha=0$, every storm",
          "deepening removed (hPa)", grid="x")

    # b: convection group, change in deepening error vs gain, every storm
    ax = axes[1]
    ax.axhspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    ax.axhline(0, color=GREY, lw=.6, zorder=2)
    ends = []
    for x in STORMS:
        e = [err_red(x, g) for g in GAINS]
        ax.plot(GAINS, e, color=SALMON if x in OVER else PALE, lw=.9, zorder=3)
        ends.append((x, e[-1]))
    med = [np.median([err_red(x, g) for x in STORMS]) for g in GAINS]
    cmed = [np.median([err_red(x, g, ctl=True) for x in STORMS]) for g in GAINS]
    ax.plot(GAINS, cmed, color=CTRL, lw=1.0, ls=(0, (3, 2)), zorder=4, label="in-box control median")
    ax.plot(GAINS, med, color=BLUE, lw=1.6, marker="o", ms=2.6, zorder=5, label="median")
    end_labels(ax, ends, 2.6)
    gain_axis(ax); ax.set_ylim(-17, 25)
    n2 = sum(err_red(x, 2) > FLOOR for x in STORMS)
    dress(ax, "b  Convection dose-response",
          "%d of 7 improve at $\\alpha=2$" % n2,
          "convection scaling ($\\alpha$)", "deepening error reduced (hPa)")

    # c: low-level spin feature 3316, three storms, random-feature controls at alpha = 0 and 3
    ax = axes[2]
    ax.axhspan(-FLOOR, FLOOR, color=GRIDC, alpha=.8, lw=0, zorder=1)
    ax.axhline(0, color=GREY, lw=.6, zorder=2)
    ends = []
    for x in SPIN_STORMS:
        e = [spin_red(x, g) for g in SPIN_GAINS]
        ax.plot(SPIN_GAINS, e, color=SPINC, lw=1.1, marker="o", ms=2.2, zorder=4)
        ends.append((x, e[-1]))
        for g, arm_name in ((0, "rand-normal"), (3, "rand-gain-3")):
            ax.scatter([g], [spin_dep_err(x, "baseline") - spin_dep_err(x, arm_name)], s=13,
                       facecolor=BG, edgecolor=GREY, linewidth=.8, zorder=5)
    end_labels(ax, ends, 1.4, over=())
    gain_axis(ax); ax.set_ylim(-17, 25)
    dress(ax, "c  Low-level spin dose-response",
          "the three storms swept",
          "spin scaling ($\\alpha$)", "deepening error reduced (hPa)")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h[1:] + h[:1] + h2[::-1], ["convection: 2401 (a), group (b)",
               "low-level spin: 3316 (a, c)", "random-feature control (a, c)",
               "convection median (b)", "in-box control median (b)"],
               loc="lower center", ncol=3, frameon=False, fontsize=NOTE, handletextpad=.3,
               columnspacing=1.4, bbox_to_anchor=(.5, .0))
    fig.subplots_adjust(left=.085, right=.955, top=.80, bottom=.27, wspace=.62)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"fig2_interventions.{ext}", facecolor=BG,
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print("a 2401", {NICE[x]: round(conv[x], 2) for x in STORMS}, "median", round(np.median(list(conv.values())), 2))
    print("a 3316", {NICE[x]: round(spin[x], 2) for x in STORMS}, "median", round(np.median(list(spin.values())), 2))
    print("b median by gain", dict(zip(GAINS, np.round(med, 2))))
    print("b control median", dict(zip(GAINS, np.round(cmed, 2))))
    print("c 3316", {NICE[x]: [round(spin_red(x, g), 2) for g in SPIN_GAINS] for x in SPIN_STORMS})


if __name__ == "__main__":
    figure2()
