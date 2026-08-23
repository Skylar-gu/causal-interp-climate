"""Draw a per-storm random control matched on IN-BOX firing, not on global firing rate.

THE DEFECT THIS FIXES. `skill_conv_storms.RANDOM_CTRL` is three features matched to the
convection group on GLOBAL firing rate, filtered to a tropical centroid so that they "can
fire on the storm". They often do not. Baseline in-box activation of the random group as a
share of the convection group's, per storm:

    michael2018    0%      ida2021    3%      wilma2005   6%     haishen2020  11%
    haiyan2013    19%      patricia2015 24%   goni2020   36%

In four of seven storms the control ablates essentially NOTHING inside the box. The repo's
own comment on the arm says it: "a control that ablates nothing cannot fail". So the
convection result's matched control is only doing work in three storms, and the headline
"~2% for a firing-rate-matched control" is credited partly to storms where the control was a
no-op. That is the control-must-be-able-to-fail rule -- a bar is not a bar until the control CAN fail.

The comparison against BASELINE is unaffected; what is weakened is the claim that the effect
is specific to the convection features rather than to ablating three arbitrary features.

THE FIX. Match on in-box firing over the storm's own region instead. This uses the IID
activation dump, so it is climatological for that region rather than storm-specific and needs
no GPU: a feature that never fires over the Coral Sea cannot be a control for a Coral Sea
storm no matter what its global rate is.

Matching criteria, kept as close to the original draw as possible so the change is the
matching AXIS and nothing else:
  - in-box activation within 25% of the convection group's, feature by feature
  - excluded: the convection group itself, the TC readout feature, and the existing
    RANDOM_CTRL, so the new control is genuinely independent of both
  - drawn with a fixed seed and written to disk BEFORE any arm runs

Paper: Table tab:mechanism-interventions (in-box-matched random control)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/inbox_control_<registry module>.json
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.storms.inbox_control
"""
import importlib
import json
import os
import pathlib
import sys

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, WEIGHTS, SCRATCH, MESH_GEOM

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
OUT = ROOT / f"results/inbox_control_{S.__name__}.json"
NW = 12
TOL = 0.25
SEED = 7

def main():
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")

    g = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.asarray(g["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)

    # accumulate per-node activation over NW windows once, reused for every storm box
    acc = np.zeros((L, 4096), np.float32)
    for wi, j in enumerate(np.linspace(0, META["n_windows"] - 1, NW).astype(int)):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, 32, axis=1)[:, :32]
        a = np.zeros_like(pre)
        r = np.arange(len(A))[:, None]
        a[r, idx] = pre[r, idx]
        acc += a
        print(f"  window {wi+1}/{NW}", flush=True)
    acc /= NW

    rng = np.random.default_rng(SEED)
    banned = set(S.CONV) | {S.TC} | set(S.RANDOM_CTRL)
    out = {}
    print(f"\n{'storm':<14}{'box nodes':>10}{'conv in-box':>13}"
          f"{'old ctl':>10}{'new ctl':>10}   features")
    for name, cfg in S.STORMS.items():
        b = cfg["box"]
        m = ((mlat >= b["lat"][0]) & (mlat <= b["lat"][1]) &
             (mlon >= b["lon"][0]) & (mlon <= b["lon"][1]))
        if m.sum() < 20:
            print(f"{name:<14} SKIP -- only {int(m.sum())} mesh nodes in box")
            continue
        ib = acc[m].sum(0)                       # in-box activation per feature
        target = ib[S.CONV]
        pick = []
        for t in target:
            cand = np.where((np.abs(ib - t) <= TOL * max(t, 1e-9)) &
                            ~np.isin(np.arange(4096), list(banned | set(pick))))[0]
            if len(cand) == 0:                   # widen once rather than silently fail
                cand = np.where(~np.isin(np.arange(4096), list(banned | set(pick))))[0]
                cand = cand[np.argsort(np.abs(ib[cand] - t))][:20]
            pick.append(int(rng.choice(cand)))
        out[name] = dict(rand=pick,
                         conv_inbox=[float(x) for x in target],
                         new_inbox=[float(ib[p]) for p in pick],
                         old_inbox=[float(ib[p]) for p in S.RANDOM_CTRL],
                         box_nodes=int(m.sum()))
        print(f"{name:<14}{int(m.sum()):>10}{ib[S.CONV].sum():>13.1f}"
              f"{ib[S.RANDOM_CTRL].sum():>10.1f}{ib[pick].sum():>10.1f}   {pick}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")
    old = np.array([sum(v["old_inbox"]) / max(sum(v["conv_inbox"]), 1e-9) for v in out.values()])
    new = np.array([sum(v["new_inbox"]) / max(sum(v["conv_inbox"]), 1e-9) for v in out.values()])
    print(f"in-box match, control as a share of the convection group:")
    print(f"  OLD (global firing-rate matched)  median {100*np.median(old):.0f}%  "
          f"range {100*old.min():.0f}-{100*old.max():.0f}%")
    print(f"  NEW (in-box matched)              median {100*np.median(new):.0f}%  "
          f"range {100*new.min():.0f}-{100*new.max():.0f}%")

if __name__ == "__main__":
    main()
