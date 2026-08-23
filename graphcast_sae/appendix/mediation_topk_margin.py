"""How exposed is the clamp to a top-k membership flip? Measured on REAL activations.

The mediation clamp sets f[:, j] := fref[:, j], where fref is captured HOST-side from the
baseline arm's stored activations while the patch itself recomputes codes INSIDE the
jitted hook. If the two computations of `pre` disagree by float noise and feature j is
sitting exactly on the k-th/(k+1)-th boundary, one of them calls j on and the other calls
it off, and the clamp would inject a spurious (v * w_dec_j) instead of exactly zero.

This measures the margin that would have to be crossed, on the 160-window IID dump
($GC_SCRATCH/fs_iid_dump.npy) with the real SAE weights:

    gap(node) = ( pre_(k) - pre_(k+1) ) / pre_(1)         relative rank-k boundary margin

and, for the specific features this battery clamps, how often they land within that margin.
A float32 matmul reassociation perturbs `pre` by ~1e-7 relative, so any node whose gap is
above ~1e-5 is safe by four orders of magnitude.

CPU only, ~1 min.  Usage:

Paper: Appendix app:topk (top-k boundary margin)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: printed report
Run:   # JAX env, CPU
    OMP_NUM_THREADS=8 python -m graphcast_sae.appendix.mediation_topk_margin [feat ...]
"""
import os
import sys

import numpy as np

from graphcast_sae.paths import REPO_ROOT, IID_DUMP
ROOT = str(REPO_ROOT)
DUMP = str(IID_DUMP)
NPZ = os.path.join(ROOT, "graphcast_sae/weights/sae_k32_lat4096_lay08.npz")
K = 32
NSAMP = 200000
FEATS = [int(a) for a in sys.argv[1:]] or [3243, 1493, 649, 2681, 1732, 165, 292, 3465]

def main():
    z = np.load(NPZ)
    W_enc = np.asarray(z["W_enc"], np.float32)
    b_pre = np.asarray(z["b_pre"], np.float32)
    X = np.load(DUMP, mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(X.shape[0], NSAMP, replace=False))
    x = np.asarray(X[idx], np.float32)
    assert np.isfinite(x).all(), "non-finite activations in the dump"
    xn = x - x.mean(1, keepdims=True)
    xn /= np.maximum(np.linalg.norm(xn, axis=1, keepdims=True), 1e-6)
    pre = np.maximum((xn - b_pre) @ W_enc.T, 0.0)
    del x, xn
    part = np.partition(pre, [-(K + 1), -K], axis=1)
    vk = part[:, -K]
    vk1 = part[:, -(K + 1)]
    top1 = pre.max(1)
    gap = (vk - vk1) / np.maximum(top1, 1e-9)
    print("nodes sampled: %d" % NSAMP)
    print("relative rank-%d boundary gap:  median %.3e   p05 %.3e   p01 %.3e   min %.3e"
          % (K, np.median(gap), np.quantile(gap, .05), np.quantile(gap, .01), gap.min()))
    for t in (1e-4, 1e-5, 1e-6, 1e-7):
        print("   fraction of nodes with gap < %.0e : %.3e" % (t, float((gap < t).mean())))
    print("\nper-feature exposure (how often feature j is the one sitting on the boundary):")
    print("  %-6s %8s %10s %12s" % ("feat", "fire%", "at-rank-k%", "onboundary%"))
    for f in FEATS:
        on = pre[:, f] > 0
        fire = on.mean()
        atk = (np.abs(pre[:, f] - vk) <= 0) & on
        near = on & (gap < 1e-5)
        print("  %-6d %7.3f%% %9.4f%% %11.3e" % (f, 100 * fire, 100 * atk.mean(), near.mean()))

if __name__ == "__main__":
    main()
