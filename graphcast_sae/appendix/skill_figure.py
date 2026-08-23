"""Summary figure for the skill-decomposition experiment.

Panels: (A) Phase-2 sanity gate (Z500 NHext adv vs lead), (B) Phase-4a CV R2 of
skill-features vs permuted-null and random-feature controls, (C) Phase-4c
known-vs-novel breakdown of top skill-features, (D) Phase-5 causal necessity
(baseline / skill-ablated / control-ablated adv).

Paper: Appendix app:taxonomy (skill decomposition summary figure)
Inputs: results/fs_atlas_class.npy (not shipped, see docs/REPRODUCE.md); results/skill/ablate.npy (not shipped, see docs/REPRODUCE.md); results/skill/decompose.npy (not shipped, see docs/REPRODUCE.md); results/skill/sanity_gate.npy (not shipped, see docs/REPRODUCE.md)
Outputs: figures/skill_decomposition.png
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.skill_figure  ->  figures/skill_decomposition.png
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
# colorblind-safe categorical palette (validated, dataviz skill)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d9d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def panel_sanity(ax):
    sg = np.load(f"{ROOT}/results/skill/sanity_gate.npy", allow_pickle=True).item()
    rows = sg["rows"]
    leads = [72, 120, 168]
    means, ses, fracs, ps = [], [], [], []
    for L in leads:
        adv = np.array([r[2] - r[3] for r in rows if int(r[1]) == L])
        means.append(adv.mean()); ses.append(adv.std(ddof=1) / np.sqrt(len(adv)))
        fracs.append(np.mean(adv > 0))
        ps.append(stats.ttest_1samp(adv, 0).pvalue)
    x = np.arange(len(leads))
    ax.bar(x, means, yerr=ses, color=C_BLUE, width=0.6, capsize=4, zorder=3,
           error_kw=dict(ecolor=INK2, lw=1.2))
    ax.axhline(0, color=INK2, lw=1)
    for i, (m, fr, p) in enumerate(zip(means, fracs, ps)):
        ax.text(i, m + ses[i] + 0.2, f"+{m:.1f}\n{fr:.0%} GC>IFS\np={p:.0e}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{L}h" for L in leads])
    ax.set_ylabel("Z500 skill advantage (gpm)\nrmse(IFS) − rmse(GC)")
    ax.set_title("A  Sanity gate: GraphCast beats IFS-HRES\n(Z500, NH-extratropics, n=120)", fontsize=10, loc="left")
    ax.set_ylim(0, max(means) + max(ses) + 3)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

def panel_cvr2(ax):
    d = np.load(f"{ROOT}/results/skill/decompose.npy", allow_pickle=True).item()
    r2_skill = float(d["r2_skill"]); r2_all = float(d["r2_all"])
    r2_perm = np.asarray(d["r2_perm"]); r2_rand = np.asarray(d["r2_rand"])
    parts = ax.violinplot([r2_perm, r2_rand], positions=[1, 2], widths=0.7,
                          showmeans=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [INK2, C_ORANGE]):
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col)
    parts["cmeans"].set_color(INK2)
    ax.scatter([0], [r2_skill], color=C_BLUE, s=90, zorder=5, label="skill-features (top-20)")
    ax.scatter([3], [r2_all], color=C_AQUA, s=70, zorder=5, marker="D", label="all features (ridge)")
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.text(0, r2_skill + 0.02, f"{r2_skill:.2f}", ha="center", va="bottom", color=C_BLUE, fontweight="bold")
    ax.text(1, r2_perm.mean(), f"  null\n  {r2_perm.mean():.2f}", ha="left", va="center", fontsize=8, color=INK2)
    ax.text(2, r2_rand.mean(), f"  rand-feat\n  {r2_rand.mean():.2f}", ha="left", va="center", fontsize=8, color=C_ORANGE)
    p_perm = float(d["p_perm"]); p_rand = float(d["p_rand"])
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["skill", "perm\nnull", "rand\nfeat", "all"])
    ax.set_ylabel("cross-validated R²  (adv ~ features)")
    ax.set_title(f"B  Predictive decomposition (grouped-month CV)\n"
                 f"skill vs null p={p_perm:.2f}, vs rand p={p_rand:.2f}", fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

def panel_known_novel(ax):
    d = np.load(f"{ROOT}/results/skill/decompose.npy", allow_pickle=True).item()
    cls = np.load(f"{ROOT}/results/fs_atlas_class.npy", allow_pickle=True).item()
    cat = np.asarray(cls["cat"])
    def bucket(c):
        if c in ("physics(single)", "joint-coupling"): return "known physics"
        if c == "residual": return "residual/novel"
        if c == "climatology/clock": return "geography/clock"
        if c == "teleconnection/mode": return "teleconnection"
        if c == "numerical/geometry": return "numerical/geom"
        return "regime/other"
    import collections
    top = np.asarray(d["top_feats"], int)
    bc = collections.Counter(bucket(cat[int(f)]) for f in top)
    order = ["known physics", "joint→known physics", "teleconnection", "geography/clock",
             "numerical/geom", "residual/novel", "regime/other"]
    labels = [b for b in order if b in bc] + [b for b in bc if b not in order]
    vals = [bc[b] for b in labels]
    colmap = {"known physics": C_BLUE, "teleconnection": C_MAGENTA, "geography/clock": C_YELLOW,
              "numerical/geom": INK2, "residual/novel": C_ORANGE, "regime/other": C_AQUA}
    cols = [colmap.get(b, C_AQUA) for b in labels]
    left = 0
    for v, l, c in zip(vals, labels, cols):
        ax.barh(0, v, left=left, color=c, edgecolor="white", lw=1.5, height=0.5)
        if v > 0:
            ax.text(left + v / 2, 0, f"{l}\n{v}", ha="center", va="center", fontsize=8,
                    color="white" if c in (C_BLUE, C_ORANGE, INK2) else INK)
        left += v
    ax.set_xlim(0, sum(vals)); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel("count of top-20 skill-features")
    nphys = bc.get("known physics", 0); nnov = bc.get("residual/novel", 0)
    ax.set_title(f"C  Discovery readout: known-mechanism vs novel\n"
                 f"known physics {nphys}/20 · residual/novel {nnov}/20", fontsize=10, loc="left")

def panel_necessity(ax):
    fp = f"{ROOT}/results/skill/ablate.npy"
    if not os.path.exists(fp):
        ax.text(0.5, 0.5, "Phase 5 ablation\npending", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return
    d = np.load(fp, allow_pickle=True).item()
    R = np.asarray(d["rows"], float)
    base, sk, ct = R[:, 1], R[:, 2], R[:, 3]
    n = len(base)
    means = [base.mean(), sk.mean(), ct.mean()]
    ses = [base.std(ddof=1)/np.sqrt(n), sk.std(ddof=1)/np.sqrt(n), ct.std(ddof=1)/np.sqrt(n)]
    cols = [INK2, C_ORANGE, C_BLUE]
    x = np.arange(3)
    ax.bar(x, means, yerr=ses, color=cols, width=0.6, capsize=4, zorder=3,
           error_kw=dict(ecolor=INK2, lw=1.2))
    ax.axhline(0, color=INK2, lw=1)
    for i, m in enumerate(means):
        ax.text(i, m + ses[i] + 0.1, f"{m:+.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    tp = stats.ttest_rel(base - sk, base - ct)
    ax.set_xticks(x); ax.set_xticklabels(["baseline\n(no ablation)", "skill-features\nablated", "random\nablated"])
    ax.set_ylabel("Z500 NHext adv (gpm, 120/168h)")
    ax.set_title(f"D  Causal necessity (top-{n} adv cases)\n"
                 f"skill-drop {(base-sk).mean():+.1f} vs ctrl {(base-ct).mean():+.1f} gpm, p={tp.pvalue:.1e}",
                 fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_sanity(axes[0, 0])
    panel_cvr2(axes[0, 1])
    panel_known_novel(axes[1, 0])
    panel_necessity(axes[1, 1])
    fig.suptitle("Decomposing GraphCast's medium-range skill advantage over IFS-HRES",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{ROOT}/figures/skill_decomposition.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("->", out)

if __name__ == "__main__":
    main()
