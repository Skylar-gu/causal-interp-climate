"""Standalone version of paper_fig_savar panel (b): causal recovery on the SAVAR benchmark.

PCMCI+ F1 against the known VAR graph (GNN bake-off + GNN/CNN end-to-end rungs), as the variable definition degrades from
oracle spatial pooling to the front-end GraphCast actually has (one mixed SAE, N unknown,
no ground-truth modes). Same estimator throughout; only the representation changes.
No title -- the caption carries it.

Numbers and provenance: see paper_fig_savar.py (this file reuses them verbatim).

Run:  python3 figures/paper_fig_savar_b.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
from paper_palette import BG, INK, MUTED, FAINT, GRIDC, BLUE, YELLOW, GREY, PALE  # noqa
from paper_fig_savar import draw_panel_b  # noqa


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=BG)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.96, bottom=0.15)
    draw_panel_b(ax)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/paper_fig_savar_b.{ext}",
                    dpi=300 if ext == "png" else None, facecolor=BG,
                    bbox_inches="tight")
    print("-> figures/paper_fig_savar_b.png / .pdf")


if __name__ == "__main__":
    main()
