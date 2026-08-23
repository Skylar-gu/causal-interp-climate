"""MEDIATION: does the forecast effect of do(i) run THROUGH feature j?

Pairwise interventions give edges. A graph needs paths, and the thing that separates a
chain i -> j -> Y from a fork i -> Y, i -> j is the controlled direct effect:

    do(i)                       -> effect on Y            TOTAL
    do(i) AND freeze(j)         -> effect on Y            DIRECT (j's pathway removed)
    mediated = TOTAL - DIRECT

No estimator, no null for existence, no lag inference. Y = storm deepening, exactly as
skill_conv_analyze.py defines it (d_deepen = min MSLP in the tracking box under the arm,
minus the same under baseline; the IC pressure cancels out of the difference, so no ERA5
truth file is needed for the mediation contrast).

THE CLAMP. fs_common's existing patches are one-sided caps toward a CONSTANT: delta_cond
caps at the climatological ftarget, delta_gain scales the excess above it. Mediation needs
the other thing -- j held at the value it took in the UNPERTURBED run, at that node, at
that step. That is fs_common.SAEJax.delta_cond_freeze, a two-sided clamp to a per-node,
per-step reference, selected by a 6-tuple patch (fsel, ftarget, nmask, jidx, fref, jmask).
This file supplies fref by running the BASELINE arm FIRST and recording feature j's
node-level codes at every rollout step, then indexing that (H, n_mesh, J) array per step
the same way the ramp path indexes its length-H gain schedule.

WHY EVERY ARM CARRIES THE 6-TUPLE. jax.jit retraces on a different pytree structure, and
this repo's small-model lesson is that per-arm compiled graphs let XLA fuse arms
differently -- a *zero* patch moved 2 m temperature by 0.16 K. So the baseline arm is run
as a 6-tuple with jmask=0 and fref=zeros, which delta_cond_freeze makes an exact no-op,
and every arm in the battery shares one compiled graph.

ARMS (names are literal; MED_ARMS restricts them)
    baseline          untouched; ALSO the source of fref. Always runs first.
    noop6             a SECOND untouched arm, same 6-tuple, jmask=0. Its difference from
                      `baseline` is this graph's own nondeterminism floor, measured in the
                      same process on the same day -- the calibration the exactness test
                      is scored against (docs/notes/nondeterminism_floor_2026_08_20.md).
    freeze-<j>        j clamped to baseline, NOTHING else patched. Must reproduce baseline
                      to within the noop6 floor. THIS IS THE CORRECTNESS TEST and it runs
                      before any mediation number is produced.
    do-<i>            i restored to normal inside the 1500 km disk (the committed
                      delta_cond ablation; identical algebra to skill_conv_run's
                      conv-normal arm, so it is comparable to results/skill/hyb_abl_f<i>).
    do-<j>            same for j. Needed to tell mediation apart from j simply having its
                      own effect on Y.
    do-<i>+freeze-<j> the direct-effect arm.

CONTROL (the control-must-be-able-to-fail rule). MED_CTLS names, per mediator, a feature matched on in-box firing
amplitude that did NOT move under do(i) (drawn by mediation_select.py). It gets the
identical freeze-only, do-only and do+freeze arms. If freezing THAT also collapses the
effect, the readout is measuring "freezing anything", not mediation -- which is live,
because the P0 probe found a strong and a weak ablation move the same NUMBER of features
(302 vs 298).

FREEZE SCOPE. MED_SCOPE=global (default) clamps j at every mesh node -- the controlled
direct effect proper. MED_SCOPE=disk clamps only inside the same 1500 km disk the
ablation uses, which answers the weaker question "does the effect run through j WITHIN the
disk". Global is the primary; disk is a sensitivity.

GPU. One rollout ~60 s at H=16 (measured, out/hyb_abl.log), plus ~80-130 s per storm of
analog passes and compilation.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 \
        MED_STORM=haishen2020 MED_SRC=2681 MED_MEDS=1493 MED_CTLS=649 \

Output: results/skill/<MED_NAME>/run_<storm>.npy   (default MED_NAME=med_f<src>_<storm>)

Paper: Appendix app:topk (mediation clamp battery)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/skill/<MED_NAME>/run_<storm>.npy
Run:   # JAX env, GPU (~46 GB)
    python -u -m graphcast_sae.appendix.mediation_run
"""
import json
import os
import sys
import time

os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# skill_conv_run is imported ONLY for its committed IO and readout helpers (build_io,
# box_phys, numpyify). Its module level reads MECH_* and would assert on a stale
# environment, so those are cleared first; nothing in this file writes them back.
for _k in [k for k in os.environ if k.startswith("MECH_")]:
    del os.environ[_k]

import importlib

import numpy as np
import jax.numpy as jnp
import xarray as xr

import graphcast_sae.common.fs_common as fc
from graphcast import rollout

from graphcast_sae.common.signature_physics import gc_km                                   # noqa: E402

import graphcast_sae.storms.skill_conv_run as R                                            # noqa: E402

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))

STORM = os.environ.get("MED_STORM", "haishen2020")
SRC = int(os.environ["MED_SRC"])
MEDS = [int(x) for x in os.environ.get("MED_MEDS", "").split(",") if x.strip()]
CTLS = [int(x) for x in os.environ.get("MED_CTLS", "").split(",") if x.strip()]
SCOPE = os.environ.get("MED_SCOPE", "global")
H = int(os.environ.get("MED_H", S.H))
TC = S.TC
NAME = os.environ.get("MED_NAME", "med_f%d_%s" % (SRC, STORM))
OUT = fc.ROOT / ("results/skill/%s" % NAME)
KEEP = [x for x in os.environ.get("MED_ARMS", "").split(",") if x.strip()]
TRACK_ALL = os.environ.get("MED_TRACK", "all").strip().lower() == "all"

if SCOPE not in ("global", "disk"):
    raise AssertionError("MED_SCOPE must be 'global' or 'disk', got %r" % SCOPE)
if not MEDS:
    raise AssertionError("MED_MEDS is empty: nothing to freeze, so nothing to test")
if CTLS and len(CTLS) != len(MEDS):
    raise AssertionError("MED_CTLS must be parallel to MED_MEDS (%d vs %d); a mediator "
                         "without its amplitude-matched control cannot be scored "
                         "(guardrail #9)" % (len(CTLS), len(MEDS)))
if SRC in MEDS or SRC in CTLS:
    raise AssertionError("the source f%d is also listed as a mediator/control; do(i) and "
                         "freeze(i) on the same feature is not a path test" % SRC)
if STORM not in S.STORMS:
    raise AssertionError("unknown storm %r; have %s" % (STORM, list(S.STORMS)))
if TC in MEDS + CTLS:
    print("NOTE: f%d (the TC feature) is being FROZEN. Y here is min MSLP -- a physical "
          "field, not 3243 -- so this is not the circularity that voided mech_atm_river, "
          "but freezing the model's own cyclone representation is a far stronger "
          "intervention than freezing a covariate and must be read as a POSITIVE CONTROL "
          "for the clamp (it SHOULD collapse the effect) rather than as a discovery."
          % TC, flush=True)
if not CTLS:
    print("WARNING: no MED_CTLS given. Without the amplitude-matched non-mover this run "
          "cannot distinguish mediation from 'freezing anything collapses the effect'. "
          "Exactness-test-only runs are the one legitimate use of this.", flush=True)

JFREEZE = MEDS + CTLS                      # union of every feature that can be clamped
DOSE = sorted(set([SRC] + MEDS + CTLS))    # every feature that can be ablated -> needs ftarget
OUT.mkdir(parents=True, exist_ok=True)
print("MEDIATION  storm=%s  source=f%d  mediators=%s  controls=%s  scope=%s  H=%d"
      % (STORM, SRC, MEDS, CTLS, SCOPE, H), flush=True)

def arm_names():
    """The full battery, in run order. `baseline` is first and is non-optional: it is
    where fref comes from, so no freeze arm can run without it."""
    a = ["baseline", "noop6"]
    for j in JFREEZE:
        a.append("freeze-%d" % j)
    a.append("do-%d" % SRC)
    for j in JFREEZE:
        a.append("do-%d" % j)
    for j in JFREEZE:
        a.append("do-%d+freeze-%d" % (SRC, j))
    return a

def main():
    fpath = OUT / ("run_%s.npy" % STORM)
    if fpath.exists() and os.environ.get("MED_OVERWRITE") != "1":
        print("%s exists -- refusing to overwrite (set MED_OVERWRITE=1 to redo). Every arm "
              "of a battery must come from ONE process, so a partial re-run is never a "
              "valid patch of an existing file." % fpath, flush=True)
        return
    cfg = S.STORMS[STORM]
    center = cfg["center"]; box = cfg["box"]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]
    mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    nN = len(mlat)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    print("model+SAE loaded; features=%d" % sae.n_features, flush=True)

    nmask = (gc_km(mlat, mlon, center[0], center[1]) < S.RADIUS_KM).astype(np.float32)
    inbox = ((mlat >= box["lat"][0]) & (mlat <= box["lat"][1])
             & (mlon >= box["lon"][0]) & (mlon <= box["lon"][1]))
    print("[%s] disk=%d nodes; box=%d mesh nodes" % (STORM, int(nmask.sum()), int(inbox.sum())),
          flush=True)

    J = len(JFREEZE)
    jidx = np.asarray(JFREEZE, np.int32)
    zeroF = np.zeros(sae.n_features, np.float32)
    zeroN = np.zeros(nN, np.float32)
    zeroJ = np.zeros((nN, J), np.float32)
    refscope = (np.ones(nN, np.float32) if SCOPE == "global" else nmask)

    def jmask_for(j):
        m = np.zeros((nN, J), np.float32)
        m[:, JFREEZE.index(j)] = refscope
        return m

    def onehot(f):
        v = np.zeros(sae.n_features, np.float32); v[f] = 1.0
        return v

    # ---- NORMAL reference from quiet, no-storm analogs (identical screen to skill_conv_run)
    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        _, acts = apply(inp, tg, fr, (zeroF, zeroF, zeroN, jidx, zeroJ, zeroJ))
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    accf = {f: [] for f in DOSE}
    used = []
    for a in cfg["analogs"]:
        try:
            c = codes_at(a)
        except Exception as e:                                        # noqa: BLE001
            print("  analog %s: load ERROR %s" % (a, e), flush=True); continue
        storm = c[inbox, TC].sum()
        if storm > 20:
            print("  analog %s: TC=%.0f storm present, SKIP" % (a, storm), flush=True); continue
        for f in DOSE:
            v = c[nmask.astype(bool), f]
            accf[f].extend(v[v > 0].tolist())
        used.append(a); print("  analog %s: TC=%.0f quiet, used" % (a, storm), flush=True)
    ftarget = np.zeros(sae.n_features, np.float32)
    for f in DOSE:
        ftarget[f] = np.mean(accf[f]) if accf[f] else 0.0
    print("  normal levels %s" % {f: round(float(ftarget[f]), 2) for f in DOSE}, flush=True)

    # ---- rollout IO (committed helper) ----
    inp, tgt, frc = R.build_io(cfg["ic"], H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    FEAT_TRACK = list(range(sae.n_features)) if TRACK_ALL else sorted(set([TC] + DOSE))

    def roll(patch, capture_ref=False):
        """One arm. `patch` is a 6-tuple whose fref entry is (H, n_nodes, J): the STEP is
        indexed here, in python, before the jitted call -- the length-H-object-meets-
        (n_nodes, n_features)-array bug (since fixed) is exactly this indexing being
        skipped."""
        cur = inp
        per = {f: [] for f in FEAT_TRACK}
        mslp = []; wind = []; ref = []
        pj0 = tuple(jnp.asarray(x) for x in patch)
        assert len(pj0) == 6, "every mediation arm must carry the 6-tuple, got %d" % len(pj0)
        assert pj0[4].ndim == 3 and pj0[4].shape[0] == H, \
            "fref must be (H, n_nodes, J); got %s" % (tuple(pj0[4].shape),)
        for h in range(H):
            pj = pj0[:4] + (pj0[4][h], pj0[5])
            ct = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, pj)
            X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
            C = np.asarray(sae.codes(X))
            for f in FEAT_TRACK:
                per[f].append(float(C[inbox, f].sum()))
            if capture_ref:
                ref.append(C[:, jidx].astype(np.float32))
            p = R.numpyify(p)
            mm, ww = R.box_phys(p, box)
            mslp.append(mm); wind.append(ww)
            if h < H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(
                    time=cur.coords["time"])
        out = dict(box_feats={f: np.array(per[f]) for f in FEAT_TRACK},
                   mslp_min=np.array(mslp), wind_max=np.array(wind))
        return out, (np.stack(ref).astype(np.float32) if capture_ref else None)

    # ---- arms ----
    all_names = arm_names()
    if KEEP:
        bad = [k for k in KEEP if k not in all_names]
        if bad:
            raise AssertionError("MED_ARMS names arms that do not exist: %s; available %s"
                                 % (bad, all_names))
        if "baseline" not in KEEP:
            raise AssertionError("MED_ARMS must include 'baseline': it is the source of the "
                                 "clamp reference, and borrowing a baseline across runs is "
                                 "forbidden by the nondeterminism floor note")
        all_names = [n for n in all_names if n in KEEP]
    print("  arms: %s  (%d rollouts, ~%.0f min at 60 s each)"
          % (all_names, len(all_names), len(all_names)), flush=True)

    res = {}
    FREF = None
    for aname in all_names:
        t = time.time()
        if aname == "baseline":
            patch = (zeroF, zeroF, zeroN, jidx, np.zeros((H, nN, J), np.float32), zeroJ)
        else:
            if FREF is None:
                raise AssertionError("baseline has not run; no clamp reference exists")
            fsel = zeroF; nm = zeroN; jm = zeroJ
            if aname != "noop6":
                head, _, tail = aname.partition("+")
                if head.startswith("do-"):
                    fsel = onehot(int(head[3:])); nm = nmask
                elif head.startswith("freeze-"):
                    jm = jmask_for(int(head[7:]))
                if tail.startswith("freeze-"):
                    jm = jmask_for(int(tail[7:]))
            patch = (fsel, ftarget, nm, jidx, FREF, jm)
        r, ref = roll(patch, capture_ref=(aname == "baseline"))
        if aname == "baseline":
            FREF = ref
            print("  clamp reference captured: %s, %s in [%.3f, %.3f]"
                  % (FREF.shape, FREF.dtype, float(FREF.min()), float(FREF.max())), flush=True)
        res[aname] = r
        d = (r["mslp_min"] - res["baseline"]["mslp_min"]) if "baseline" in res else None
        print("  [%s] min MSLP %.2f  d_deepen %+.3f  max|dMSLP| %.3f  (%.0fs)"
              % (aname, float(np.min(r["mslp_min"])),
                 float(np.min(r["mslp_min"]) - np.min(res["baseline"]["mslp_min"])),
                 float(np.abs(d).max()), time.time() - t), flush=True)

    out = dict(name=NAME, storm=STORM, ic=cfg["ic"], center=center, box=box,
               src=SRC, meds=MEDS, ctls=CTLS, jfreeze=JFREEZE, scope=SCOPE, H=H, tc=TC,
               analogs_used=used, ftarget={int(f): float(ftarget[f]) for f in DOSE},
               disk_nodes=int(nmask.sum()), box_nodes=int(inbox.sum()),
               n_freeze_nodes=int(refscope.sum()), arms=all_names, res=res)
    np.save(OUT / ("run_%s.npy" % STORM), out, allow_pickle=True)
    print("-> %s" % (OUT / ("run_%s.npy" % STORM)), flush=True)

if __name__ == "__main__":
    main()
