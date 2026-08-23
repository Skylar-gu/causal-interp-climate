"""Gain curves: how forecast deepening responds to scaling a feature group's excess.

One curve per feature group, ordered by calibrated ascent-loading, so the figure
asks whether the AMPLIFICATION response scales the way the ablation response does.

    f -> f + (g-1) * max(f - normal, 0)      inside a 1500 km disk

g=0 is the committed ablation arm (bit-identical to delta_cond), g=1 is baseline,
g>1 amplifies. Reading the curve: negative = the intervention made the storm deepen
LESS than baseline, positive = MORE.

Design notes.
- The four groups are not categorical identities, they are points on a continuum
  (calibrated ascent sigma 28.5 / 16.8 / 9.0 / 2.1), so they get an ORDINAL
  single-hue ramp, light = low loading -> dark = high loading, not four unrelated
  hues. Validated with `validate_palette.js --ordinal --mode light`: monotone
  lightness, adjacent dL >= 0.06, light end 2.06:1 on the surface, hue spread 3 deg.
- The random control is a CONTROL, not a peer series, so it is recessive grey and
  dashed rather than taking a categorical slot.
- y is PERCENT of each storm's own baseline deepening, which lets all three panels
  share one axis honestly; the absolute baseline is printed in each panel so the
  hPa scale is never lost. Baselines differ 5x across these storms (26.3 / 6.8 /
  5.5 hPa), so a shared absolute axis would flatten two of the three panels.

    python3 figures/plot_gain_curves.py
"""
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "skill"
OUT = ROOT / "figures" / "gain_curves.png"

# group -> (results dir, calibrated median ascent sigma, ordinal ramp step)
# ramp: blue 250 / 400 / 500 / 650, light = least ascent-loaded
GROUPS = [
    ("moisture2  +2.1s", "gain_moist2", 2.1, "#86b6ef"),
    ("ascent-9   +9.0s", "gain_asc09", 9.0, "#3987e5"),
    ("ascent-17 +16.8s", "gain_asc17", 16.8, "#256abf"),
    ("convection +28.5s", "gain_conv", 28.5, "#104281"),
]
STORMS = [("haishen2020", "Haishen 2020"), ("ida2021", "Ida 2021"),
          ("patricia2015", "Patricia 2015")]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#77776f"

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "figure.dpi": 150,
})


def deepening(run, arm, ic_mslp):
    return ic_mslp - float(np.min(run["res"][arm]["mslp_min"]))


def load(dirname, storm):
    """-> (gains, pct_change, rand_points) or None if the run is absent."""
    p = RES / dirname / f"run_{storm}.npy"
    if not p.exists():
        return None
    r = np.load(p, allow_pickle=True).item()
    truth = np.load(RES / dirname / "era5_truth.npy", allow_pickle=True).item()
    ic = float(truth[storm]["mslp_min"][0])
    base = deepening(r, "baseline", ic)
    gs, ys = [], []
    for arm in r["res"]:
        m = re.fullmatch(r"gain-([0-9.]+)", arm)
        if m:
            gs.append(float(m.group(1)))
            ys.append(100.0 * (deepening(r, arm, ic) - base) / max(base, 1e-9))
    o = np.argsort(gs)
    rand = {}
    for arm in r["res"]:
        if arm.startswith("rand"):
            g = 1.0 if arm == "rand-normal" else float(arm.rsplit("-", 1)[1])
            # rand-normal is the g=0 analogue for the control group
            g = 0.0 if arm == "rand-normal" else g
            rand[g] = 100.0 * (deepening(r, arm, ic) - base) / max(base, 1e-9)
    return np.array(gs)[o], np.array(ys)[o], rand, base


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharey=True)
    any_data = False
    for ax, (storm, title) in zip(axes, STORMS):
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(1, color=MUTED, lw=0.8, ls=":", zorder=1)
        base_txt = ""
        for label, dirname, sig, col in GROUPS:
            d = load(dirname, storm)
            if d is None:
                continue
            any_data = True
            gs, ys, rand, base = d
            base_txt = f"baseline deepening {base:.1f} hPa"
            ax.plot(gs, ys, "-o", color=col, lw=2.0, ms=5.0, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
            if rand:
                rg = sorted(rand)
                ax.plot(rg, [rand[g] for g in rg], "--", color=MUTED, lw=1.4,
                        zorder=2, label="random control" if label.startswith("conv") else None)
        ax.set_title(f"{title}\n{base_txt}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("gain $g$ on the excess above normal")
        ax.set_xticks([0, 1, 1.5, 2, 2.5, 3])
    if not any_data:
        print("no gain runs found yet — nothing plotted"); return
    axes[0].set_ylabel("change in forecast deepening (% of baseline)")

    # annotate the two anchors once, on the left panel
    a = axes[0]
    a.annotate("g=0: ablate to normal\n(the committed arm)", xy=(0, 0), xytext=(0.06, 0.06),
               textcoords="axes fraction", fontsize=7.5, color=INK2)
    a.annotate("g=1: baseline", xy=(1, 0), xytext=(0.42, 0.92),
               textcoords="axes fraction", fontsize=7.5, color=INK2)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.045), labelcolor=INK2)
    fig.suptitle("Scaling a feature group's convective excess, "
                 "ordered by calibrated ascent-loading", fontsize=10.5, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
