"""Add `qrandc_flag` — a Nyquist-CLEAN negative control (prereg amendment A1, 2026-08-14).

WHY. FG-1a demanded zero modes above 50 % Nyquist variance for every member. The six
FITTED members pass (max Nyquist fraction 0.048-0.109). The two anchors do not, and cannot:
their `q` is permuted / random rather than fitted, so nothing steers them away from the
alias, and a random channel readout of the raw pooled activations lands on it. Measured on
the refit basis: `qperm_flag` 12 of 39 modes above 50 % (max 0.777), `qrand_flag` 2 of 39
(max 0.696).

That is a real asymmetry and it is reported as one: the anchors carry a strong shared
DETERMINISTIC clock (the (+1,-1) alternation separating {00Z,12Z} from {06Z,18Z}) that the
candidates do not, which inflates their edge counts and makes anchor edge counts
non-comparable to candidate edge counts.

`qperm_flag` and `qrand_flag` are KEPT UNCHANGED — they are the constructions the mini v2
result was validated with, and changing them would break comparability with it. This script
ADDS a third anchor whose only difference from `qrand_flag` is that the alias is projected
out of its readout, so the anchor gate does not rest on a clock artifact.

CONSTRUCTION. For each mode c the Nyquist signature in channel space is exactly rank-1:
after the pipeline's own deseasonalization the residual is `nyq (x) n_c + rest`, so
`n_c` is recovered by regressing the residual on the alternation. Then

    q_qrandc[c] = normalise( q_qrand[c] - (q_qrand[c] . n_hat_c) n_hat_c )

Random direction, alias removed. Footprints are vmax's, so this costs ZERO GPU — it reads
the same pooled tensor as `vmax_flag`.

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: candidates/pool_flag_v2_candidates.npy; candidates/pool_flag_v2_chandirs.npy; results/flag_gint_preflight.json
Run:   # JAX env, CPU
    OMP_NUM_THREADS=8 python -m graphcast_sae.obsgraph.add_anchor_qrandc
"""
import json
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH
META = json.load(open(SCRATCH / "fs_iid_meta.json"))
NW, N_MESH, DIM = META["n_windows"], META["n_mesh"], META["dim"]
OUT_C = ROOT / "candidates/pool_flag_v2_candidates.npy"
OUT_CD = ROOT / "candidates/pool_flag_v2_chandirs.npy"
PF = ROOT / "results/flag_gint_preflight.json"

import sys                                                              # noqa: E402

from graphcast_sae.obsgraph.build_pool_flag_v2 import harmonic_design, nyq_col, resid          # noqa: E402

def main():
    pc = np.load(OUT_C, allow_pickle=True).item()
    cd = np.load(OUT_CD, allow_pickle=True).item()
    W = pc["cands"]["vmax_flag"].astype(np.float32)
    times = np.array(META["starts"], dtype="datetime64[ns]")
    D1 = harmonic_design(times, nyquist=False)
    nyq = nyq_col(times); nyq = (nyq - nyq.mean()) / nyq.std()

    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    P = np.empty((W.shape[0], NW, DIM), np.float32)
    for w in range(NW):
        P[:, w, :] = W @ np.asarray(X[w * N_MESH:(w + 1) * N_MESH], np.float32)
        if w % 40 == 0:
            print(f"  pooled window {w}/{NW}", flush=True)

    q0 = cd["qrand_flag"]["q"].astype(np.float32)
    qv = cd["vmax_flag"]["q"].astype(np.float32)
    q = np.empty_like(q0)
    for c in range(P.shape[0]):
        R = resid(D1, P[c] - P[c].mean(0))                 # (NW, DIM)
        n = nyq @ R                                        # rank-1 channel signature
        n = n / (np.linalg.norm(n) + 1e-12)
        v = q0[c] - float(q0[c] @ n) * n
        q[c] = v / (np.linalg.norm(v) + 1e-12)

    pc["cands"]["qrandc_flag"] = W.copy()
    cd["qrandc_flag"] = dict(q=q, mbar=cd["vmax_flag"]["mbar"].copy(),
                             varfrac=np.full(q.shape[0], np.nan, np.float32))

    # verify the amendment did what it claims
    S = np.einsum("cwd,cd->wc", P, q) - (cd["qrandc_flag"]["mbar"] * q).sum(1)
    Z = resid(D1, S); Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    nf = np.array([float(((nyq @ Z[:, c]) / len(nyq)) ** 2) for c in range(Z.shape[1])])
    CM = np.corrcoef(Z.T); ev = np.linalg.eigvalsh(CM)
    offd = np.abs(CM - np.eye(CM.shape[0]))
    cosv = float(np.median(np.abs((q * qv).sum(1))))
    cos0 = float(np.median(np.abs((q * q0).sum(1))))
    print(f"\nqrandc_flag: nyq max {nf.max():.3f}  modes>50% {int((nf>0.5).sum())}  "
          f"cond {ev.max()/max(ev.min(),1e-12):.1f}  min eig {ev.min():.4f}  "
          f"max|r| {offd.max():.3f}")
    print(f"  median |cos(q, q_vmax)| = {cosv:.3f}  (bar <0.15: "
          f"{'OK' if cosv < 0.15 else '** FAIL **'})")
    print(f"  median |cos(q, q_qrand)| = {cos0:.3f}  (still essentially the same random "
          f"direction, minus the alias)")
    assert nf.max() < 0.5 and cosv < 0.15

    pf = json.load(open(PF))
    pf["rep"]["qrandc_flag"] = dict(
        cond=float(ev.max() / max(ev.min(), 1e-12)), min_eig=float(ev.min()),
        max_abs_corr=float(offd.max()), nyq_max=float(nf.max()),
        n_nyq_over_50=int((nf > 0.5).sum()), nyq_mean=float(nf.mean()),
        med_cos_q=float(np.nanmedian(np.abs(q @ q.T) - np.eye(q.shape[0]))),
        max_cos_q=float(np.nanmax(np.abs(q @ q.T) - np.eye(q.shape[0]))),
        N=int(q.shape[0]), eff_nodes_median=pf["rep"]["vmax_flag"]["eff_nodes_median"])
    pf["amendment_A1"] = dict(
        why="anchors carry the Nyquist alias by construction; qperm 12/39 and qrand 2/39 "
            "modes above 50% Nyquist variance on the refit basis",
        action="qperm_flag and qrand_flag kept UNCHANGED (mini-v2 comparability); "
               "qrandc_flag added = qrand with the rank-1 Nyquist channel signature "
               "projected out of each mode's readout",
        cos_qrandc_vmax=cosv, cos_qrandc_qrand=cos0)
    json.dump(pf, open(PF, "w"), indent=1, default=float)
    pc["provenance"]["members"] = list(pc["cands"])
    np.save(OUT_C, pc, allow_pickle=True)
    np.save(OUT_CD, cd, allow_pickle=True)
    print(f"\nwrote {OUT_C.name}, {OUT_CD.name}  members={list(pc['cands'])}")

if __name__ == "__main__":
    main()
