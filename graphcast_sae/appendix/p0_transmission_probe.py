"""P0 de-risking probe for the PCMCI+-proposes / intervention-disposes hybrid.

THE QUESTION THIS KILLS. The hybrid scores a proposed edge A->B with an ASYMMETRY,

    asym(A,B,tau) = |dB(t+tau)| / B_base  -  |dA(t+tau)| / A_base

because the SAE is an overcomplete dictionary: ablating A perturbs the residual stream,
the encoder re-reads it, and every feature whose decoder direction overlaps A's moves a
little. That leakage is SYMMETRIC in (A,B) by construction, so it cancels in the
difference -- PROVIDED there is any transmission left after it cancels. Two ways the
whole programme dies, and both are cheap to check on ONE storm with THREE rollouts:

  KILL-1  NO TRANSMISSION. Ablating a strong feature moves nothing else. Then every
          asym is 0 - 0 and there is no signal to score. Bar: does ablating 2067 move
          ANY other tracked feature by >10% of its own baseline in-box amplitude?

  KILL-2  ALL LEAKAGE. Everything that moves, moves in proportion to its footprint
          overlap with A -- i.e. |dB| is a function of cos(footprint_A, footprint_B)
          and nothing else. Then the "causal" reading is dictionary geometry. Bar:
          OLS of |dB| on footprint cosine across all 4096 features. R^2 > 0.8 means
          leakage dominates and the asymmetry is being asked to cancel ~all of the
          signal.

Neither bar can be met by a no-op: KILL-1 fails if the intervention does nothing,
KILL-2 fails if it does everything through geometry. The WEAK-feature arm is the
within-run contrast -- a feature that fires in the box but weakly should move far
fewer things than 2067 does, and if it moves just as many the readout is measuring
"any perturbation at all", not this feature.

ARMS (3 rollouts, ida2021, MECH_TRACK=all so all 4096 in-box series are recorded):
  baseline     untouched
  conv-normal  feature 2067 restored to normal in the 1500 km disk  (strong convection)
  rand-normal  the WEAK feature restored to normal in the same disk (contrast)

FOOTPRINTS are NOT redefined here. The definition is footprint_masks.py's: a feature's
footprint is the set of mesh nodes where it is in the top-32 in >=25% of NW=12 IID
windows. This file recomputes that same (n_mesh, 4096) indicator once and caches it,
because footprint_masks.py only ever SAVED the handful of features it projected to the
0.25 deg grid.

USAGE (three separate steps; only the middle one needs the GPU)

  # 0. CPU, ~2 min, once. Builds the footprint/exposure cache and prints the in-box
  #    exposure ranking used to justify the weak-feature choice.

  # 1. GPU, ~5 min. QUEUE THIS -- do not run it beside another JAX job.
  FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

  # 2. CPU, ~2 min. No GPU, no jax.

Paper: Appendix app:topk (Table tab:p0)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/skill/<P0_MECH_NAME|p0probe>/{p0_config,p0_arms,p0_verdict}.json + a footprint cache .npz
Run:   # JAX env, GPU (~46 GB)
    OMP_NUM_THREADS=8 python -m graphcast_sae.appendix.p0_transmission_probe exposure
    python -m graphcast_sae.appendix.p0_transmission_probe run
    OMP_NUM_THREADS=8 python -m graphcast_sae.appendix.p0_transmission_probe analyze
"""
import json
import os
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------- configuration
from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH, MESH_GEOM
WEIGHTS = ROOT / "graphcast_sae/weights"
MECH = os.environ.get("P0_MECH_NAME", "p0probe")
OUT = ROOT / f"results/skill/{MECH}"
STORM = os.environ.get("P0_STORM", "ida2021")
SRC = int(os.environ.get("P0_SRC", "2067"))          # strong convection feature
# WEAK. 2850 is one of the three FROZEN global-firing-rate controls (skill_conv_storms.py:19,
# drawn seed 7 before any result). Its in-box exposure in the Ida box is already MEASURED and
# committed in results/inbox_control_skill_conv_storms.json as old_inbox = [0.0, 0.0, 0.364]
# for RANDOM_CTRL = [3667, 2875, 2850] -- so 3667 and 2875 do not fire in this box at all
# (they would be vacuous, the control-must-be-able-to-fail rule) and 2850 does, at 0.364 against 2067's 2.604, i.e.
# 14%. Clearly nonzero, clearly weak, and chosen from a pre-existing frozen draw rather than
# picked to make the contrast look good. Inside the 1500 km DISK -- where the dose actually
# lands -- the same cache gives 2850 at 0.283 against 2067's 2.915, i.e. 9.7%, still clearly
# nonzero. `exposure` re-prints both from the cache and will say so if either has moved.
#
# ONE ASYMMETRY TO KEEP IN MIND when reading the contrast: ftarget is the mean activation
# over QUIET analog dates, and for ida2021 only one analog survives the storm-present screen,
# on which 2850 never fires in the disk -- so ftarget[2850] = 0 and the WEAK arm DELETES the
# feature rather than capping it at normal, while the SRC arm caps 2067 at 1.88. The weak arm
# therefore gets the STRONGER of the two dose semantics, which makes the contrast
# conservative: if it still moves fewer features than 2067 does, that is not an artefact of
# a gentler dose.
WEAK = int(os.environ.get("P0_WEAK", "2850"))
NW = 12                                              # IID windows, = footprint_masks.py
THRESH = 0.25                                        # fires in >=25% of windows, ditto
TOPK = 32                                            # SAE k, ditto
FP_CACHE = pathlib.Path(os.environ.get(
    "P0_FP_CACHE", str(SCRATCH / f"fs_footprint_fires_nw{NW}.npz")))
MOVE_BAR = float(os.environ.get("P0_MOVE_BAR", "10.0"))    # KILL-1 bar, % of baseline
R2_BAR = float(os.environ.get("P0_R2_BAR", "0.8"))         # KILL-2 bar

# ------------------------------------------------------- footprints / exposure (CPU)
def build_cache(force=False):
    """(n_mesh, 4096) footprint indicator + mean in-window activation, cached.

    Identical arithmetic to footprint_masks.py:47-62 and inbox_control.py:63-76 -- the
    same NW=12 evenly-spaced IID windows, the same normalise / encode / top-32, the same
    >=25%-of-windows threshold. Nothing new is defined here; the only difference is that
    ALL 4096 columns are kept instead of the handful that got projected to the grid.
    """
    if FP_CACHE.exists() and not force:
        z = np.load(FP_CACHE)
        return z["fires"], z["acc"]
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    meta = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L, NWIN = meta["n_mesh"], meta["n_windows"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    assert X.shape[0] >= L * NWIN, f"dump too short: {X.shape} for {NWIN} x {L}"

    cnt = np.zeros((L, 4096), np.float32)
    acc = np.zeros((L, 4096), np.float32)
    rows = np.arange(L)[:, None]
    for wi, j in enumerate(np.linspace(0, NWIN - 1, NW).astype(int)):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        assert np.isfinite(A).all(), f"window {j} not finite"
        assert not (A == 0).all(1).any(), f"window {j} has all-zero nodes (guardrail #6)"
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, TOPK, axis=1)[:, :TOPK]
        cnt[rows, idx] += 1.0
        acc[rows, idx] += pre[rows, idx]
        print(f"  window {wi + 1}/{NW}", flush=True)
    acc /= NW
    fires = cnt >= NW * THRESH
    FP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FP_CACHE, fires=fires, acc=acc)
    print(f"  -> {FP_CACHE}", flush=True)
    return fires, acc

def _box_mask():

    import importlib
    S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
    g = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.asarray(g["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    b = S.STORMS[STORM]["box"]
    m = ((mlat >= b["lat"][0]) & (mlat <= b["lat"][1]) &
         (mlon >= b["lon"][0]) & (mlon <= b["lon"][1]))
    assert m.sum() >= 20, f"{STORM} box holds only {int(m.sum())} mesh nodes"
    return m, S

def _disk_mask(S):
    """The 1500 km ablation disk. The DOSE lands here, so the WEAK feature has to fire
    HERE, not merely somewhere in the (much larger) readout box."""

    from graphcast_sae.common.signature_physics import gc_km
    g = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.asarray(g["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    c = S.STORMS[STORM]["center"]
    return gc_km(mlat, mlon, c[0], c[1]) < S.RADIUS_KM

def exposure(n=40):
    """Rank all 4096 by climatological in-box activation, and justify the WEAK pick."""
    fires, acc = build_cache()
    m, S = _box_mask()
    dm = _disk_mask(S)
    dk = acc[dm].sum(0)
    print(f"\n{STORM} 1500 km ABLATION DISK = {int(dm.sum())} mesh nodes "
          "(this is where the dose lands)")
    print(f"  SRC  f{SRC:<5d} {dk[SRC]:10.3f}")
    print(f"  WEAK f{WEAK:<5d} {dk[WEAK]:10.3f}   = "
          f"{100 * dk[WEAK] / max(dk[SRC], 1e-9):.1f}% of SRC")
    if dk[WEAK] <= 0:
        print("  !! WEAK does not fire inside the DISK -- that arm would be a NO-OP.")
    ib = acc[m].sum(0)
    assert np.isfinite(ib).all() and ib.max() > 0, "in-box exposure degenerate"
    order = np.argsort(-ib)
    print(f"\n{STORM} box = {int(m.sum())} mesh nodes; climatological in-box activation")
    print(f"  SRC  f{SRC:<5d} {ib[SRC]:10.3f}   (rank {int(np.where(order == SRC)[0][0]) + 1})")
    print(f"  WEAK f{WEAK:<5d} {ib[WEAK]:10.3f}   (rank {int(np.where(order == WEAK)[0][0]) + 1})"
          f"   = {100 * ib[WEAK] / max(ib[SRC], 1e-9):.1f}% of SRC")
    if ib[WEAK] <= 0:
        print("  !! WEAK does not fire in this box at all -- that arm would be a NO-OP and "
              "the control could not fail (guardrail #9). Pick another.")
    print(f"  convection group {S.CONV}: "
          f"{[round(float(ib[f]), 3) for f in S.CONV]}")
    print(f"  frozen RANDOM_CTRL {S.RANDOM_CTRL}: "
          f"{[round(float(ib[f]), 3) for f in S.RANDOM_CTRL]}")
    print(f"\n  weakest 20 features that still fire in the box (candidates for WEAK):")
    nz = order[ib[order] > 0]
    for f in nz[-20:]:
        print(f"    f{int(f):<5d} {ib[f]:10.4f}   footprint {int(fires[:, f].sum()):6d} nodes")
    print(f"\n  top {n} by in-box activation:")
    for f in order[:n]:
        print(f"    f{int(f):<5d} {ib[f]:10.3f}   footprint {int(fires[:, f].sum()):6d} nodes")
    return ib

# -------------------------------------------------------------------- run (GPU)
def run():
    """Thin runner: write the arm config, set the env, hand off to skill_conv_run."""
    OUT.mkdir(parents=True, exist_ok=True)

    import importlib
    S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
    assert STORM in S.STORMS, f"{STORM} not in the registry"
    assert SRC != S.TC and WEAK != S.TC, "cannot ablate the outcome feature"
    assert SRC != WEAK, "SRC and WEAK must differ"

    # skill_conv_run's per-storm control channel. Feeding it here means the WEAK arm goes
    # through the SAME restore-to-normal machinery as the SRC arm (same disk, same analog
    # normal levels) rather than a second code path.
    ctl = ROOT / f"results/inbox_control_{S.__name__}.json"
    meas = json.load(open(ctl)) if ctl.exists() else {}
    prev = meas.get(STORM, {})
    cfg = {STORM: dict(
        rand=[WEAK],
        conv_inbox=[float(prev.get("conv_inbox", [0.0])[S.CONV.index(SRC)])
                    if SRC in S.CONV and prev.get("conv_inbox") else 0.0],
        new_inbox=[float(prev["old_inbox"][S.RANDOM_CTRL.index(WEAK)])
                   if prev.get("old_inbox") and WEAK in S.RANDOM_CTRL else 0.0],
        old_inbox=[0.0],
        box_nodes=int(prev.get("box_nodes", 0)))}
    cpath = OUT / "p0_arms.json"
    json.dump(cfg, open(cpath, "w"), indent=1)
    json.dump(dict(storm=STORM, src=SRC, weak=WEAK, mech=MECH,
                   arms={"baseline": "untouched",
                         "conv-normal": f"f{SRC} restored to normal in the 1500 km disk",
                         "rand-normal": f"f{WEAK} restored to normal in the 1500 km disk"},
                   move_bar_pct=MOVE_BAR, r2_bar=R2_BAR),
              open(OUT / "p0_config.json", "w"), indent=1)
    print(f"arms -> {cpath}", flush=True)

    # env must be set BEFORE the import: skill_conv_run reads all of it at module scope.
    os.environ["MECH_NAME"] = MECH
    os.environ["MECH_FEATS"] = str(SRC)
    os.environ["MECH_INBOX_CTL"] = str(cpath)
    os.environ["MECH_TRACK"] = "all"
    os.environ["MECH_ARMS"] = "baseline,conv-normal,rand-normal"
    os.environ["MECH_ALLOW_OVERLAP"] = "1"      # SRC IS a convection feature, intended
    os.environ.setdefault("FS_DEVICE", "gpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.argv = [sys.argv[0], STORM]
    import graphcast_sae.storms.skill_conv_run as skill_conv_run
    skill_conv_run.main()

# ---------------------------------------------------------------- analyze (CPU)
def _gate(run_path):
    """Guardrail #6: every input is gated before it is analysed."""
    assert run_path.exists(), (
        f"{run_path} missing -- run the GPU step first:\n"
        f"  FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python "
        f"graphcast_sae/appendix/p0_transmission_probe.py run")
    d = np.load(run_path, allow_pickle=True).item()
    res = d["res"]
    for a in ("baseline", "conv-normal", "rand-normal"):
        assert a in res, f"arm {a} missing; have {sorted(res)}"
    bf = res["baseline"]["box_feats"]
    assert len(bf) == 4096, f"MECH_TRACK=all not used: only {len(bf)} features tracked"
    assert set(bf) == set(range(4096)), "tracked feature ids are not 0..4095"
    H = len(bf[0])
    assert H >= 8, f"rollout too short to score: H={H}"
    for a in res:
        M = np.stack([np.asarray(res[a]["box_feats"][f], np.float64)
                      for f in range(4096)], 1)          # (H, 4096)
        assert M.shape == (H, 4096), f"{a}: bad shape {M.shape}"
        assert np.isfinite(M).all(), f"{a}: non-finite in-box series"
        assert not (M == 0).all(), f"{a}: ENTIRELY zero -- corrupt run (guardrail #6)"
        nz = (M != 0).any(0).sum()
        assert nz > 100, f"{a}: only {nz}/4096 features ever fire in the box; corrupt run"
        res[a]["_M"] = M
    return d, res, H

def _movers(M, B, base_amp, exclude, label, bar, n=20):
    dabs = np.abs(M - B).mean(0)                          # mean_t |d(t)|, raw units
    live = base_amp > 0
    pct = np.full(4096, np.nan)
    pct[live] = 100.0 * dabs[live] / base_amp[live]
    cand = np.array([f for f in range(4096) if live[f] and f not in exclude])
    order = cand[np.argsort(-pct[cand])]
    print(f"\n  {label}: top {n} movers "
          f"(mean_t |dB| as % of that feature's own baseline in-box amplitude)")
    print(f"    {'feat':>6} {'%chg':>9} {'|dB|':>10} {'B_base':>10}")
    for f in order[:n]:
        print(f"    {int(f):>6} {pct[f]:>9.2f} {dabs[f]:>10.4f} {base_amp[f]:>10.4f}")
    n_over = int((pct[cand] > bar).sum())
    print(f"    -> {n_over} of {len(cand)} live features move by more than {bar:g}%")
    return dabs, pct, n_over, cand

def analyze():
    run_path = OUT / f"run_{STORM}.npy"
    d, res, H = _gate(run_path)
    B = res["baseline"]["_M"]
    base_amp = B.mean(0)                                   # A_base per feature
    live = base_amp > 0
    print(f"P0 TRANSMISSION PROBE -- {STORM}, H={H} steps, "
          f"{int(live.sum())}/4096 features fire in the box")
    print(f"  SRC  f{SRC}  baseline in-box amplitude {base_amp[SRC]:.4f}")
    print(f"  WEAK f{WEAK}  baseline in-box amplitude {base_amp[WEAK]:.4f}")
    if base_amp[WEAK] <= 0:
        print("  !! INSTRUMENT FAILURE: the WEAK arm ablates a feature that never fires "
              "in the box. That control cannot fail (guardrail #9); the contrast below is "
              "uninterpretable and must be reported as a failure, not dropped.")

    # the dose has to land, or nothing below means anything
    Ms = res["conv-normal"]["_M"]
    self_pct = 100.0 * np.abs(Ms[:, SRC] - B[:, SRC]).mean() / max(base_amp[SRC], 1e-12)
    print(f"  dose check: ablating f{SRC} moves f{SRC} itself by {self_pct:.1f}% "
          f"of its baseline")
    if self_pct < 5.0:
        print("  !! the intervention barely moved its OWN target -- instrument failure, "
              "not a null result.")

    # ---- KILL-1 -------------------------------------------------------------
    print(f"\n[KILL-1] does ablating f{SRC} move ANY other feature by >{MOVE_BAR:g}%?")
    dabs, pct, n_over, cand = _movers(Ms, B, base_amp, {SRC},
                                      f"ablate f{SRC} (conv-normal)", MOVE_BAR)
    v1 = ("PASS -- transmission exists" if n_over > 0 else
          "KILL -- no transmission; every asym would be 0 - 0")
    print(f"  VERDICT: {v1}")

    # ---- within-run contrast -------------------------------------------------
    Mw = res["rand-normal"]["_M"]
    w_self = 100.0 * np.abs(Mw[:, WEAK] - B[:, WEAK]).mean() / max(base_amp[WEAK], 1e-12)
    print(f"\n[CONTRAST] the WEAK arm. f{WEAK} moves itself by {w_self:.1f}%.")
    _, wpct, w_over, _ = _movers(Mw, B, base_amp, {WEAK},
                                 f"ablate f{WEAK} (rand-normal)", MOVE_BAR)
    print(f"  {n_over} features move >{MOVE_BAR:g}% under f{SRC}, {w_over} under f{WEAK}. "
          "If these are comparable the readout measures 'any perturbation', not this "
          "feature.")

    # ---- KILL-2: leakage calibration ---------------------------------------
    fires, _ = build_cache()
    fa = fires[:, SRC].astype(np.float64)
    na = np.linalg.norm(fa)
    assert na > 0, (f"f{SRC} has an EMPTY footprint under the "
                    f">= {THRESH:.0%}-of-{NW}-windows rule; cosine is undefined")
    F = fires.astype(np.float32)
    nrm = np.linalg.norm(F, axis=0).astype(np.float64)
    cos = np.zeros(4096)
    ok = nrm > 0
    cos[ok] = (fa @ F[:, ok]) / (na * nrm[ok])
    assert np.isfinite(cos).all() and cos.max() <= 1.0 + 1e-6

    sel = np.array([f for f in cand if ok[f]])
    ovl = np.array([f for f in sel if cos[f] > 0])
    print(f"\n[KILL-2] leakage calibration: |dB| ~ cos(footprint f{SRC}, footprint B)")
    print(f"  {len(sel)} features have a nonempty footprint AND fire in the box "
          f"({int((~ok).sum())} of 4096 have an empty footprint and are excluded)")
    print(f"  f{SRC}'s own footprint is {int(fires[:, SRC].sum())} of {fires.shape[0]} "
          f"mesh nodes; {len(ovl)} of the {len(sel)} overlap it at all, "
          f"{int((cos[sel] > 0.45).sum())} above cosine 0.45, "
          f"max off-self {cos[sel[sel != SRC]].max():.3f}")
    if len(ovl) < 50 or (cos[sel] > 0.45).sum() == 0:
        print("  !! POWER WARNING: the overlap support is thin -- most cosines are exactly "
              "0, so the full-dictionary R^2 is pushed toward 0 by construction and a LOW "
              "R^2 here does NOT establish 'leakage ruled out' (guardrail #9: the bar has "
              "to be attainable). Read the cos>0 rows, which are the only ones where the "
              "leakage hypothesis makes a nonzero prediction.")

    def _ols(x, y):
        A = np.stack([np.ones_like(x), x], 1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ((y - A @ beta) ** 2).sum() / ss if ss > 0 else np.nan
        rho = (np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
               if len(x) > 2 else np.nan)
        return beta, r2, rho

    r2_main = np.nan
    for sname, idx in (("all", sel), ("cos>0", ovl)):
        if len(idx) < 3:
            print(f"  {sname}: only {len(idx)} features -- not regressable")
            continue
        for yname, y in (("|dB| (raw)", dabs[idx]), ("% of baseline", pct[idx])):
            beta, r2, rho = _ols(cos[idx], y)
            print(f"  [{sname:<5} n={len(idx):<4}] {yname:<15} slope {beta[1]:+.4f}  "
                  f"intercept {beta[0]:+.4f}  R^2 {r2:.4f}  spearman {rho:+.3f}")
            if sname == "all" and yname.startswith("|dB|"):
                r2_main = r2
    v2 = ("KILL -- leakage dominates; the asymmetry must cancel nearly all of the signal"
          if r2_main > R2_BAR else
          "PASS -- footprint overlap does not explain the movement")
    print(f"  VERDICT: {v2}")

    json.dump(dict(storm=STORM, src=SRC, weak=WEAK, H=int(H),
                   n_live=int(live.sum()), self_pct=float(self_pct),
                   weak_self_pct=float(w_self), n_over_src=int(n_over),
                   n_over_weak=int(w_over), r2_leakage=float(r2_main),
                   move_bar=MOVE_BAR, r2_bar=R2_BAR),
              open(OUT / "p0_verdict.json", "w"), indent=1)
    print(f"\n-> {OUT / 'p0_verdict.json'}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"exposure": exposure, "run": run, "analyze": analyze,
     "cache": build_cache}[cmd]()
