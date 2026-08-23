"""Feature catalog for the steering atlas — which SAE features are recognizable physical structures.

Encodes the i.i.d. dump through the flagship SAE, and for every one of the 4096 features records:
  fire       fraction of tokens where the feature is active (TopK-selected)
  node_map   mean activation at each of 40,962 mesh nodes (its spatial FOOTPRINT)
  coh        spatial concentration of that footprint (great-circle spread; small = coherent/local)
  dose90     90th-pct active value  (a natural, feature-specific kick amplitude for steering)
  centroid   footprint centre (lat, lon)

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: candidates/fs_feature_catalog.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.atlas.feature_select
"""
import json, sys
from pathlib import Path
import numpy as np
from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH, MESH_GEOM, SAE_WEIGHTS
from graphcast_sae.common.signature_physics import gc_km
DUMP = SCRATCH / "fs_iid_dump.npy"
META = json.load(open(SCRATCH / "fs_iid_meta.json"))
NW, L, DIM = META["n_windows"], META["n_mesh"], META["dim"]
OUT = ROOT / "candidates/fs_feature_catalog.npy"

def main():
    import torch
    z = np.load(SAE_WEIGHTS)                                        # the shipped npz
    Wenc = torch.from_numpy(np.asarray(z["W_enc"], np.float32))     # (F,512)
    bpre = torch.from_numpy(np.asarray(z["b_pre"], np.float32))
    F = Wenc.shape[0]; k = 32
    geom = np.load(MESH_GEOM, allow_pickle=True).item()
    lat, lon, xyz = geom["lat"], geom["lon"], geom["xyz"]

    X = np.load(DUMP, mmap_mode="r")
    node_map = np.zeros((F, L), np.float64)          # feature x node mean activation
    fire = np.zeros(F, np.int64)
    val_sum = np.zeros(F); val_sq = np.zeros(F)       # sufficient stats over ACTIVE tokens
    for w in range(NW):
        A = torch.from_numpy(np.asarray(X[w*L:(w+1)*L], np.float32))
        xn = A - A.mean(1, keepdim=True); xn = xn / xn.norm(dim=1, keepdim=True).clamp_min(1e-6)
        pre = torch.relu((xn - bpre) @ Wenc.T)                    # (L,F)
        v, idx = torch.topk(pre, k, dim=1)                       # (L,k)
        f = torch.zeros_like(pre).scatter_(1, idx, v).numpy()    # (L,F) dense TopK
        node_map += f.T                                          # accumulate per node
        act = f > 0; fire += act.sum(0)
        val_sum += f.sum(0); val_sq += (f**2).sum(0)
        if w % 20 == 0: print(f"  encoded {w}/{NW}", flush=True)
    node_map /= NW
    firerate = fire / (NW * L)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_act = val_sum / np.maximum(fire, 1)                     # mean active value
        v_act = np.maximum(val_sq / np.maximum(fire, 1) - m_act**2, 0)
    dose90 = m_act + 2.0 * np.sqrt(v_act)                         # strong-but-in-range kick scale

    # spatial coherence: great-circle spread of each feature's footprint (small = local)
    coh = np.full(F, np.nan); clat = np.full(F, np.nan); clon = np.full(F, np.nan)
    for fi in range(F):
        wmap = node_map[fi]; s = wmap.sum()
        if s <= 0: continue
        C = (wmap[:, None] * xyz).sum(0) / s; C /= np.linalg.norm(C) + 1e-12
        cla = np.degrees(np.arcsin(np.clip(C[2], -1, 1))); clo = np.degrees(np.arctan2(C[1], C[0]))
        d = gc_km(lat, lon, cla, clo)
        coh[fi] = np.sqrt((wmap * d**2).sum() / s); clat[fi] = cla; clon[fi] = clo

    np.save(OUT, dict(fire=fire, firerate=firerate, node_map=node_map.astype(np.float32),
                      dose90=dose90, coh=coh, clat=clat, clon=clon,
                      lat=lat, lon=lon, F=int(F), nw=NW), allow_pickle=True)
    alive = fire > 0
    print(f"\nalive {alive.sum()}/{F}  firerate[alive] median={np.median(firerate[alive]):.4f}")
    print(f"coherence km: local(p10)={np.nanpercentile(coh,10):.0f}  median={np.nanpercentile(coh,50):.0f}  global(p90)={np.nanpercentile(coh,90):.0f}")
    # a first look at steer candidates: coherent AND frequently firing
    score = firerate / (coh + 1e-6)
    top = np.argsort(-np.where(np.isfinite(coh), score, -1))[:15]
    print("\ntop candidates (coherent + high firing):")
    print(f"  {'feat':>5}{'firerate':>10}{'coh_km':>9}{'dose90':>8}   centroid")
    for fi in top:
        print(f"  {fi:>5}{firerate[fi]:>10.4f}{coh[fi]:>9.0f}{dose90[fi]:>8.2f}   ({clat[fi]:+.0f},{clon[fi]:+.0f})")
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
