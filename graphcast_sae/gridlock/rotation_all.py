"""Rotation-based positional score for ALL 4,096 features, from two forward passes.

The Jaccard score (support overlap across dates) was a cheap proxy for "is this feature locked
to the mesh". It is a bad one. The rotation test calls f3680 and f560 positional -- their
correlation with the rotated atmosphere is 0.013 and 0.041, i.e. nothing -- while the Jaccard
score puts them at 0.149 and 0.173, squarely weather-like. Those two carry the first and third
largest single-feature effects measured. So the score used to define the "grid-locked class"
misclassifies exactly the features that matter most.

The rotation test is the direct measurement and costs only two forward passes for the whole
dictionary, because one pass yields all 4,096 columns at once. This runs it and keeps every
column, giving a positional score with no proxy in between:

    positional = corr(rotated, original in place) - corr(rotated, original rolled with the air)

Positive means the feature stayed on the mesh while the planet moved under it.

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_rotation_all.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.rotation_all
"""
import dataclasses
import json
import os
import sys

os.environ.setdefault("FS_DEVICE", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax.numpy as jnp
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils

ICS = os.environ.get("RA_ICS", "2020-01-10,2020-07-10").split(",")
# ROLL in grid cells out of 1440 longitude points. 720 = 180 deg, an exact integer half so no
# interpolation is needed. 180 deg has one blind spot: it leaves latitude untouched, so a
# zonally symmetric pattern maps onto itself and scores "stayed" for free -- f2235, a polar
# band, was exactly that case. 360 = 90 deg is the robustness check: a different angle, still
# exact in grid cells, with a different set of patterns that are invariant under it.
ROLL = int(os.environ.get("RA_ROLL", "720"))
OUT = fc.ROOT / os.environ.get("RA_OUT", "results/fs_rotation_all.npy")

def build(ic, tc, roll):
    blk, times, statics = fc.load_block(np.datetime64(ic), nframes=3)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims:
            w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS:
        w[v] = statics[v]
    if roll:
        for v in list(w.data_vars):
            if "lon" in w[v].dims:
                w[v] = w[v].roll(lon=roll, roll_coords=False)
    w = w.assign_coords(datetime=(("batch", "time"),
                                  times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(tc))

def main():
    sae = fc.SAEJax()
    params, mc, tc, stats_ = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats_, sae=sae, bf16=True)
    apply_fn = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    g = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlon = np.asarray(g["lon"], float) % 360.0
    mlat = np.asarray(g["lat"], float)
    tgt = (mlon + 360.0 * ROLL / 1440.0) % 360.0
    d = np.abs(((mlon[None, :] - tgt[:, None] + 180) % 360) - 180) + \
        np.abs(mlat[None, :] - mlat[:, None]) * 3.0
    partner = np.argmin(d, 1)

    NF = sae.n_features
    stays = np.zeros((len(ICS), NF)); follows = np.zeros((len(ICS), NF))
    selfsim = np.zeros((len(ICS), NF))
    for k, ic in enumerate(ICS):
        C = {}
        for tag, roll in (("orig", 0), ("rot", ROLL)):
            inp, t_, f_ = build(ic, tc, roll)
            _, acts = apply_fn(inp, t_, f_, noop)
            X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
            C[tag] = np.asarray(sae.codes(X), np.float32)
            print(f"  {ic} {tag} {C[tag].shape}", flush=True)
        A, Bb = C["orig"], C["rot"]
        Ap = A[partner]

        def cc(P, Q):                       # column-wise Pearson, nan where a column is flat
            P = P - P.mean(0); Q = Q - Q.mean(0)
            n = (P * Q).sum(0)
            dsq = np.sqrt((P * P).sum(0) * (Q * Q).sum(0))
            return np.where(dsq > 0, n / np.maximum(dsq, 1e-12), np.nan)
        stays[k] = cc(Bb, A); follows[k] = cc(Bb, Ap); selfsim[k] = cc(A, Ap)
    S = np.nanmean(stays, 0); F = np.nanmean(follows, 0); SS = np.nanmean(selfsim, 0)
    pos = S - F
    live = np.isfinite(pos)
    # the test has no power where the pattern is already ~180-deg symmetric
    powered = live & (np.abs(SS) < 0.35)
    np.save(OUT, dict(stays=S, follows=F, selfsim=SS, positional=pos,
                      live=live, powered=powered, ics=ICS), allow_pickle=True)
    p = pos[powered]
    print(f"\n{int(live.sum())} live, {int(powered.sum())} with power "
          f"(|self-similarity| < 0.35)")
    print(f"positional score: median {np.median(p):+.3f}  p90 {np.percentile(p,90):+.3f}  "
          f"p99 {np.percentile(p,99):+.3f}  max {p.max():+.3f}")
    for cut in (0.2, 0.3, 0.5):
        print(f"  >= {cut:.1f}: {int((pos >= cut)[powered].sum()):>4} features")
    top = np.where(powered)[0][np.argsort(-pos[powered])][:20]
    print("\ntop 20:", ", ".join(f"f{f}({pos[f]:+.2f})" for f in top))

if __name__ == "__main__":
    main()
