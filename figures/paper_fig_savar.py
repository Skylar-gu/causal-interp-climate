"""SAVAR ground-truth calibration: forecasting well does not expose the causal structure.

One compact two-panel figure.
  (a) Predictive skill on the 8-mode SAVAR benchmark: the CNN forecaster sits 0.011 RMSE
      above the analytic oracle floor -- an essentially perfect forecaster.
  (b) Causal recovery (PCMCI+ F1 against the known VAR(2) graph) from that same model's
      activations, as the variable definition degrades from oracle spatial pooling to the
      front-end GraphCast actually has (one mixed SAE, N unknown, no ground-truth modes).
      Same estimator throughout; only the representation changes.

Numbers, with provenance (all previously computed; nothing new is estimated here):
  skill    ~/savar-validation notes/REPO_SUMMARY_AND_AUDIT.md Phase 1a:
           persistence 1.48, CNN val 1.072, oracle floor 1.061 (D_y = I).
  bake-off ~/savar-project/results/litext_e1_discovery.npy (loaded live if present):
           true-Z 0.853, oracle-W pooled activations 0.855, varimax 0.819,
           k-means hard partition 0.280, DMD 0.177, misplaced footprints 0.000.
  ladder   CNN: ~/causal-graphcast/audit_pcmci_assumptions/savar_sae_pcmci (2026-08-21),
           ladder protocol, own true-Z ceiling 0.825: GraphCast-matched mixed SAE (R3b)
           F1 0.128 (P 0.080; 47 false edges vs 12 true; p=0.033 vs random feature draws);
           un-pooled per-pixel SAE (R4) 0.004-0.079 (p=0.287 vs random draws).
           GNN: .../savar_sae_pcmci_gnn (2026-08-25, MeshGNN, hetdynamics_eqvar, stride 1):
           R3b F1 0.003 under the PCMCI+ protocol (ceiling 0.859; 0 true edges) and 0.019
           under the ladder protocol (ceiling 0.616; p=0.31 vs random draws); un-pooled
           per-node SAE (R4) 0.004-0.010 (p=0.33-0.93).
  NOTE     the bake-off block is the GNN (hetdynamics_eqvar, PCMCI+ protocol); the CNN rows
           are a different dataset (base) and protocol. Only the estimator is shared.
           Per-mode SAEs: 0/8 monosemantic (phase7_sae_findings.md); PC0 of pooled
           activations 86% (CNN) / 88-92% (GNN) of variance.

Run:  python3 figures/paper_fig_savar.py     (any python with matplotlib)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, GREEN, YELLOW, GREY, PALE  # noqa

# ---- numbers (see header for provenance) -------------------------------------------
SKILL = [("persistence", 1.480, PALE),
         ("CNN forecaster", 1.072, BLUE),
         ("oracle floor", 1.061, INK)]

LITEXT = ROOT / "savar/results/litext_e1_discovery.npy"   # shipped with the savar/ package
BAKE = {"oracle-W pooling": 0.855, "varimax pooling": 0.819,
        "hard partition (k-means)": 0.280, "DMD modes": 0.177,
        "misplaced footprints": 0.000}
if LITEXT.exists():                       # prefer the artifact over the transcription
    d = np.load(LITEXT, allow_pickle=True).item()
    BAKE["oracle-W pooling"] = float(d["anchors"]["oracle_acts_fu1"])
    BAKE["varimax pooling"] = float(d["graph"]["vmax_act"]["F1"])
    BAKE["hard partition (k-means)"] = float(d["graph"]["km_act"]["F1"])
    BAKE["DMD modes"] = float(d["graph"]["dmd_act"]["F1"])
    BAKE["misplaced footprints"] = float(d["graph"]["shift5"]["F1"])
CEIL_BAKE = 0.853       # PCMCI+ on the true mode series Z, same protocol as BAKE (GNN data)
CEIL_LADDER = 0.825     # CNN end-to-end ladder's own true-Z ceiling
CEIL_LADDER_GNN = 0.616 # GNN end-to-end ladder's own true-Z ceiling (T=2400, tau_max 6)
SAE_F1 = 0.128          # CNN, GraphCast-matched rung R3b: one mixed SAE, N unknown, no Z
SAE_F1_GNN_PLUS = 0.003 # GNN R3b, PCMCI+ protocol (comparable to the bake-off block)
SAE_F1_GNN = 0.019      # GNN R3b, ladder protocol
R4_LO, R4_HI = 0.004, 0.079          # CNN R4: un-pooled per-pixel SAE (MAP-FOOT / MAP-R)
R4_GNN_LO, R4_GNN_HI = 0.004, 0.010  # GNN R4: un-pooled per-node SAE

# Rows of panel (b): (label, value, colour, spread or None). Two blocks separated by the
# protocol line: PCMCI+ protocol (GNN bake-off + GNN R3b) above, end-to-end ladder below.
def panel_b_rows():
    rows = [("PCMCI+ on true modes Z  (ceiling)", CEIL_BAKE, INK, None),
            *[(f"{k}  · GNN", v, BLUE if v > 0.5 else GREY, None) for k, v in BAKE.items()],
            ("mixed SAE, end-to-end  · GNN", SAE_F1_GNN_PLUS, YELLOW, None),
            ("mixed SAE, end-to-end  · CNN", SAE_F1, YELLOW, None),
            ("un-pooled per-pixel SAE  · CNN", (R4_LO + R4_HI) / 2, PALE, (R4_LO, R4_HI)),
            ("un-pooled per-node SAE  · GNN", (R4_GNN_LO + R4_GNN_HI) / 2, PALE,
             (R4_GNN_LO, R4_GNN_HI))]
    n_top = 1 + len(BAKE) + 1          # rows in the PCMCI+-protocol block
    return rows, n_top


def draw_panel_b(axb, label_fs=8.4, sep_label=True):
    rows, n_top = panel_b_rows()
    n = len(rows)
    yb = np.arange(n)[::-1]
    for y, (lab, v, col, spread) in zip(yb, rows):
        axb.barh(y, v, height=0.55, color=col, zorder=3,
                 edgecolor=INK if col == PALE else col, lw=0.6)
        if spread is not None:                            # draw the spread, not a point
            axb.plot([spread[0], spread[1]], [y, y], color=MUTED, lw=1.4, zorder=4)
            txt = f"{spread[0]:.2f}–{spread[1]:.2f}"
            xr = spread[1]
        else:
            txt = f"{v:.3f}" if 0 < v < 0.01 else f"{v:.2f}"
            xr = v
        axb.text(max(xr, 0.08) + 0.015, y, txt, va="center", fontsize=8.6,
                 color=INK, weight="bold")
    axb.axvline(CEIL_BAKE, color=INK, lw=0.9, ls=":", zorder=2)
    ysep = yb[n_top - 1] - 0.5
    axb.axhline(ysep, color=GRIDC, lw=0.8, zorder=1)
    if sep_label:
        axb.text(CEIL_BAKE - 0.02, ysep + 0.05, "above: PCMCI+ protocol (ceiling 0.85)",
                 ha="right", va="bottom", fontsize=6.5, color=FAINT)
        axb.text(CEIL_BAKE - 0.02, ysep - 0.05,
                 f"below: ladder protocol (ceilings CNN {CEIL_LADDER:.2f}, GNN {CEIL_LADDER_GNN:.2f})",
                 ha="right", va="top", fontsize=6.5, color=FAINT)
    axb.set_xlim(0, 1.0); axb.set_ylim(-0.6, n - 0.4)
    axb.set_yticks(yb)
    axb.set_yticklabels([r[0] for r in rows], fontsize=label_fs, color=INK)
    axb.tick_params(axis="y", length=0)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    axb.set_xlabel("causal-graph recovery, F1 vs known VAR edges", fontsize=9, color=MUTED)
    axb.grid(axis="x", color=GRIDC, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)
    axb.spines["bottom"].set_color(PALE)


def main():
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), facecolor=BG, gridspec_kw=dict(
            width_ratios=[1.0, 1.45], wspace=0.62, left=0.075, right=0.975,
            top=0.80, bottom=0.30))

    # ---------------- (a) predictive skill ----------------
    ya = np.arange(len(SKILL))[::-1]
    for y, (lab, v, col) in zip(ya, SKILL):
        axa.barh(y, v, height=0.52, color=col, zorder=3,
                 edgecolor=INK if col == PALE else col, lw=0.6)
        axa.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=9, color=INK,
                 weight="bold")
        axa.text(0.02, y, lab, va="center", fontsize=9,
                 color=BG if col in (BLUE, INK) else MUTED, zorder=4)
    axa.axvline(SKILL[2][1], color=INK, lw=0.9, ls=":", zorder=2)
    axa.annotate("0.011 above the\ntheoretical floor", xy=(1.061, 1.55),
                 xytext=(0.42, 1.35), fontsize=8, color=MUTED,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    axa.set_xlim(0, 1.62); axa.set_ylim(-0.55, 2.55)
    axa.set_yticks([])
    axa.set_xlabel("val RMSE (8-mode SAVAR, VAR(2))", fontsize=9, color=MUTED)
    axa.set_title("a   The forecaster is essentially perfect", fontsize=10.5,
                  color=INK, weight="bold", loc="left", pad=8)

    # ---------------- (b) causal recovery ----------------
    draw_panel_b(axb)
    axb.set_title("b   …but the representation does not expose the causes",
                  fontsize=10.5, color=INK, weight="bold", loc="left", pad=8)
    for sp in ("top", "right", "left"):
        axa.spines[sp].set_visible(False)
    axa.spines["bottom"].set_color(PALE)
    fig.text(0.075, 0.115,
             "Same PCMCI+ estimator throughout; only the variable definition changes. "
             "Bake-off rows: GNN forecaster. End-to-end rows: one SAE over mixed activations, "
             "N unknown, no footprints —\nGNN F1 0.003 (0 true edges; p = 0.31 vs random feature draws), "
             "CNN 0.13 (precision 0.08; p = 0.03, single random draws score higher); dropping the "
             "spatial pooling too leaves both at 0.00–0.08 (p ≥ 0.29).",
             fontsize=7.2, color=FAINT, ha="left", va="top")

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_savar.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG,
                    bbox_inches="tight")
    print("-> figures/paper_fig_savar.png / .pdf")


if __name__ == "__main__":
    main()
