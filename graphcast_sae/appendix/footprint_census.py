"""Footprint SIZE versus footprint SPREAD, over the whole 4,096-feature dictionary.

WHY. Every "local features" selection in this repo gates on footprint SIZE -- a node count,
`results/hybrid_footprint_fires.npz['fires'].sum(0) <= 200` (see `pcmci/local_physics.py:166`
and `results/graph_v2.json['selection']`). The propagation readouts that then verify the graph
(great-circle distance between centroids, implied speed, bearing) require the footprint to be
spatially COMPACT, which is a different quantity: the activation-weighted great-circle RMS
spread about the footprint centroid, `coh` in `candidates/fs_feature_catalog.npy`
(`graphcast_sae/atlas/feature_select.py:54-64`).

This script measures whether the gate that is applied implies the property that is needed.
It does not: the two are essentially uncorrelated under the repo's own footprint definition.
A feature can light 40 mesh nodes and have them scattered over a hemisphere.

Two size definitions are reported so the answer cannot be an artefact of one of them:
  size_fires   column sum of the boolean fires mask (the definition the selections use)
  size_nodemap number of mesh nodes with nonzero mean activation in `node_map`

Paper: Appendix app:mesh (Table tab:census)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/hybrid_footprint_fires.npz (not shipped, see docs/REPRODUCE.md)
Outputs: results/footprint_census.json
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.footprint_census
"""
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

from graphcast_sae.paths import REPO_ROOT as ROOT
FIRES = ROOT / "results/hybrid_footprint_fires.npz"
CATALOG = ROOT / "candidates/fs_feature_catalog.npy"
OUT = ROOT / "results/footprint_census.json"

COMPACT_KM = [1500.0, 2000.0, 2500.0, 3000.0]
SIZE_BAR = 200                      # the locality clause actually used

def main():
    z = np.load(FIRES, allow_pickle=True)
    fires = z["fires"]
    assert fires.shape == (40962, 4096), fires.shape          # guardrail #6
    assert fires.dtype == bool, fires.dtype
    size_fires = fires.sum(0).astype(int)

    cat = np.load(CATALOG, allow_pickle=True).item()
    coh = np.asarray(cat["coh"], float)                        # great-circle RMS spread, km
    nm = cat["node_map"]
    assert coh.shape == (4096,), coh.shape
    assert nm.shape[0] == 4096 and nm.shape[1] == 40962, nm.shape
    size_nodemap = (nm > 0).sum(1).astype(int)

    alive = np.isfinite(coh)
    m_f = alive & (size_fires > 0)
    m_n = alive & (size_nodemap > 0)
    assert m_f.sum() > 0 and m_n.sum() > 0

    out = {
        "n_features": int(coh.size),
        "n_spread_defined": int(alive.sum()),
        "spread_km": {
            "p10": float(np.percentile(coh[alive], 10)),
            "median": float(np.median(coh[alive])),
            "p90": float(np.percentile(coh[alive], 90)),
            "max": float(coh[alive].max()),
        },
        "size_fires": {
            "n": int(m_f.sum()),
            "median": float(np.median(size_fires[m_f])),
            "p90": float(np.percentile(size_fires[m_f], 90)),
            "frac_le_200": float((size_fires[m_f] <= SIZE_BAR).mean()),
            "spearman_size_vs_spread": float(spearmanr(size_fires[m_f], coh[m_f]).statistic),
            "spearman_p": float(spearmanr(size_fires[m_f], coh[m_f]).pvalue),
        },
        "size_nodemap": {
            "n": int(m_n.sum()),
            "median": float(np.median(size_nodemap[m_n])),
            "spearman_size_vs_spread": float(spearmanr(size_nodemap[m_n], coh[m_n]).statistic),
            "spearman_p": float(spearmanr(size_nodemap[m_n], coh[m_n]).pvalue),
        },
        "compact_census": {},
        "compact_among_small_footprints": {},
    }
    for km in COMPACT_KM:
        out["compact_census"][f"{km:.0f}"] = {
            "n": int((coh[alive] < km).sum()),
            "frac": float((coh[alive] < km).mean()),
        }
        sub = m_f & (size_fires <= SIZE_BAR)
        out["compact_among_small_footprints"][f"{km:.0f}"] = {
            "n_small": int(sub.sum()),
            "n_compact": int((coh[sub] < km).sum()),
            "frac": float((coh[sub] < km).mean()),
        }

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", OUT)

if __name__ == "__main__":
    main()
