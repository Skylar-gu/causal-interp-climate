"""The commitment horizon: WHEN is GraphCast's convection representation load-bearing?

Scores the single-step pulse battery written by `skill_conv_run.py` with
MECH_RAMPS=pulse0,...,pulse15 (run_queue16.sh / run_queue17.sh).

Every other ablation arm in this repo doses all sixteen rollout steps, which confounds
"this feature matters" with "sixteen perturbations compound". A pulse arm restores the
convection group to its normal level at ONE rollout step and is bit-identical to baseline
at every other step (`_sched` returns g=1 elsewhere and delta_gain computes (1-1)*excess=0).
So the damage attributable to a pulse at step k is the damage of intervening at lead 6k h
and nothing else.

ARMS (20).  baseline | ramp-pulse{0,1,2,3,4,6,8,11,15} (convection group)
            rand-normal | rand-ramp-pulse{same nine} (per-storm IN-BOX matched control,
            firing at 87-112% of the convection group inside these boxes).

TIME CONVENTION (checked against skill_conv_run.roll):
    box_feats[h] is the SAE activation of the state fed to step h  -> lead 6h  h=0..15
    mslp_min[h]  is the model output of step h                     -> lead 6(h+1)
    a pulse at step k therefore doses lead 6k h, and leaves outputs at leads < 6(k+1)
    bit-identical to baseline. That identity is a data gate here, not an assumption.

READOUT.  Exactly `skill_conv_analyze.metrics`:
    deepen(arm)   = ERA5 MSLP at IC  -  min over the 16 forecast leads of arm MSLP
    d_deepen(arm) = deepen(baseline) - deepen(arm)        > 0 => deepening LOST
The IC term cancels in d_deepen, so the damage curve does not depend on era5_truth.npy
the truth file is used only to print absolute deepening and is optional.

PRE-REGISTERED BAR (docs/prereg/prereg_hybrid_hurricane.md, "Companion result -- the commitment
horizon"): convection pulse damage at k<=3 exceeds damage at k>=8 by >=2x in >=6 of 8
storms, AND the in-box control shows no such ratio (<1.3x). If the control also front-loads,
the finding is reported as early-lead sensitivity in general, not as a convection result.

CPU only. Reads nothing but results/skill/<battery>/run_*.npy (+ optional era5_truth.npy).

Paper: Sec. 3 (commitment horizon: single-step pulse arms)
Inputs: results/skill/commit_horizon (not shipped, see docs/REPRODUCE.md); results/skill/convection/era5_truth.npy (shipped); results/skill/sh_convection/era5_truth.npy (not shipped, see docs/REPRODUCE.md)
Outputs: <resdir>/commit_horizon_verdict.json (--json) and the verdict figure (--fig)
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.commit_horizon_analyze
    python -m graphcast_sae.storms.commit_horizon_analyze --resdir results/skill/commit_horizon_sh --storms skill_sh_storms
"""
import argparse
import importlib
import json
import os
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOT `import fs_common` on purpose: fs_common pulls in haiku/jax, which forces this
# CPU-only analysis; ROOT is the only thing needed from the package.
from graphcast_sae.paths import REPO_ROOT as ROOT

# ---------------------------------------------------------------- constants
PULSE_KS = [0, 1, 2, 3, 4, 6, 8, 11, 15]
EARLY_KS = [k for k in PULSE_KS if k <= 3]        # 0, 1, 2, 3   -> leads 0..18 h
LATE_KS = [k for k in PULSE_KS if k >= 8]         # 8, 11, 15    -> leads 48, 66, 90 h
H_EXPECT = 16
BAR_RATIO = 2.0          # convection must reach this
CTRL_BAR = 1.3           # control must stay under this
BAR_NSTORMS = 6          # in >= 6 of 8 storms
# A ratio of medians is unstable when the denominator sits on zero. Declared rule, applied
# identically to treatment and control, and every storm it fires on is printed by name:
#   late <= EPS_HPA and early > 2*EPS_HPA  -> ratio = inf   (front-loaded, late damage nil)
#   late <= EPS_HPA and early <= 2*EPS_HPA -> ratio = nan   (NO SIGNAL, cannot be scored,
#                                                            counts as a FAIL, never a pass)
EPS_HPA = 0.05
N_PERM = 5000
PERM_SEED = 20260820

C_CONV = "#eb6834"
C_CTRL = "#1baf7a"
C_ND = "#7a4fbf"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})

def conv_arm(k):
    return f"ramp-pulse{k}"

def ctrl_arm(k):
    return f"rand-ramp-pulse{k}"

def meets(r, bar):
    """Explicit, so operator precedence can never decide a pre-registered bar."""
    if np.isinf(r) and r > 0:
        return True
    return bool(np.isfinite(r) and r >= bar)

EXPECTED_ARMS = (["baseline"] + [conv_arm(k) for k in PULSE_KS]
                 + ["rand-normal"] + [ctrl_arm(k) for k in PULSE_KS])

# ---------------------------------------------------------------- data gate
class Gate:
    """Guardrail #6. Collects every gate result; nothing is silently dropped."""

    def __init__(self):
        self.rows = []          # (storm, check, ok, detail)
        self.fatal = 0
        self.warn = 0

    def check(self, storm, name, ok, detail, fatal=True):
        self.rows.append((storm, name, bool(ok), detail, fatal))
        if not ok:
            if fatal:
                self.fatal += 1
            else:
                self.warn += 1
        return bool(ok)

    def report(self, verbose=False):
        print("\n" + "=" * 100)
        print("DATA GATE (guardrail #6) -- census of every check. Nothing is dropped silently.")
        print("=" * 100)
        cur = None
        for storm, name, ok, detail, fatal in self.rows:
            fam = name.split(": ", 1)[1] if ": " in name else name
            if not verbose and ok:
                continue
            if storm != cur:
                print(f"\n  [{storm}]")
                cur = storm
            tag = "ok  " if ok else ("FAIL" if fatal else "warn")
            print(f"    {tag}  {name:<42s} {detail}")
        if not verbose:
            # compact census: per storm, per check FAMILY, how many passed
            print("  (per-check detail for the PASSES is suppressed; --gate-verbose prints "
                  "every row. Every FAIL and every warning is always printed, above.)")
            fams = {}
            for storm, name, ok, detail, fatal in self.rows:
                fam = name.split(": ", 1)[1] if ": " in name else name
                d = fams.setdefault((storm, fam), [0, 0])
                d[0] += 1
                d[1] += bool(ok)
            cur = None
            for (storm, fam), (n, k) in fams.items():
                if storm != cur:
                    print(f"\n  [{storm}]")
                    cur = storm
                print(f"    {k:3d}/{n:<3d} pass   {fam}")
        print(f"\n  gate summary: {len(self.rows)} checks, {self.fatal} FATAL, {self.warn} warnings")

def finite_series(x):
    x = np.asarray(x, float)
    return x.ndim == 1 and np.all(np.isfinite(x))

def load_runs(resdir, storm_names, gate):
    """Load and gate every run file. Returns {name: run_dict} for runs that pass."""
    runs, missing = {}, []
    for name in storm_names:
        p = resdir / f"run_{name}.npy"
        if not p.exists():
            missing.append(name)
            continue
        r = np.load(p, allow_pickle=True).item()
        res = r["res"]
        ok = True

        have = [a for a in EXPECTED_ARMS if a in res]
        extra = [a for a in res if a not in EXPECTED_ARMS]
        ok &= gate.check(name, "all 20 expected arms present",
                         len(have) == len(EXPECTED_ARMS),
                         f"{len(have)}/{len(EXPECTED_ARMS)}"
                         + (f"  MISSING={[a for a in EXPECTED_ARMS if a not in res]}" if len(have) != len(EXPECTED_ARMS) else "")
                         + (f"  extra={extra}" if extra else ""))
        if not ok:
            continue

        for a in EXPECTED_ARMS:
            mm = np.asarray(res[a]["mslp_min"], float)
            ww = np.asarray(res[a]["wind_max"], float)
            ok &= gate.check(name, f"{a}: mslp finite/len/nonzero",
                             finite_series(mm) and mm.size == H_EXPECT and np.any(mm != 0)
                             and np.ptp(mm) > 0,
                             f"len={mm.size} range={mm.min():.1f}..{mm.max():.1f} hPa")
            ok &= gate.check(name, f"{a}: wind finite/len/nonzero",
                             finite_series(ww) and ww.size == H_EXPECT and np.any(ww != 0),
                             f"len={ww.size} max={ww.max():.1f} m/s")

        grp = [int(f) for f in r.get("conv", [])]
        ctl = [int(f) for f in r.get("rand", [])]
        ok &= gate.check(name, "convection group recorded in run", len(grp) > 0, f"conv={grp}")
        ok &= gate.check(name, "in-box control group recorded", len(ctl) > 0, f"rand={ctl}")
        bf = res["baseline"]["box_feats"]
        ok &= gate.check(name, "box_feats covers both groups",
                         all(f in bf for f in grp + ctl),
                         f"{len(bf)} features tracked; missing="
                         f"{[f for f in grp + ctl if f not in bf]}")
        for f in grp + ctl:
            if f in bf:
                s = np.asarray(bf[f], float)
                gate.check(name, f"box_feats[{f}] finite/len",
                           finite_series(s) and s.size == H_EXPECT,
                           f"len={s.size} max={s.max():.3f}")

        base = np.asarray(res["baseline"]["mslp_min"], float)
        base_bf = np.mean([np.asarray(bf[f], float) for f in grp], axis=0) if grp else None

        # --- the structural gate the pulse design allows and a flat arm does not ---
        # A pulse at step k is an EXACT no-op before step k, so outputs at leads < 6(k+1)
        # must be bit-identical to baseline. If they are not, the schedule is not doing
        # what _sched says it does.
        for k in PULSE_KS:
            for tag, a in (("conv", conv_arm(k)), ("ctrl", ctrl_arm(k))):
                mm = np.asarray(res[a]["mslp_min"], float)
                pre = np.max(np.abs(mm[:k] - base[:k])) if k > 0 else 0.0
                gate.check(name, f"{tag} pulse{k}: pre-dose leads identical", pre == 0.0,
                           f"max|d| before lead {6*(k+1)}h = {pre:.3e} hPa")
                post = np.max(np.abs(mm[k:] - base[k:]))
                # A pulse that changed NOTHING is an instrument failure, not a zero effect.
                gate.check(name, f"{tag} pulse{k}: dose changed the forecast", post > 0.0,
                           f"max|d| after dose = {post:.4f} hPa"
                           + ("   <-- NO-OP ARM (instrument failure)" if post == 0.0 else ""),
                           fatal=False)
            if base_bf is not None:
                aa = np.mean([np.asarray(res[conv_arm(k)]["box_feats"][f], float)
                              for f in grp], axis=0)
                d0 = abs(aa[k] - base_bf[k])
                gate.check(name, f"conv pulse{k}: in-box activation moved at dose",
                           d0 > 0.0,
                           f"|d act| at lead {6*k}h = {d0:.4f} "
                           f"(baseline {base_bf[k]:.4f})"
                           + ("   <-- group already at/below normal in box, dose is a no-op"
                              if d0 == 0.0 else ""), fatal=False)

        if ok:
            runs[name] = r
    if missing:
        gate.check("<registry>", "all storms present on disk", False,
                   f"MISSING run files: {missing}")
    else:
        gate.check("<registry>", "all storms present on disk", True,
                   f"{len(storm_names)} run files")
    return runs

# ---------------------------------------------------------------- readouts
def damage_curves(runs, truth):
    """Per storm: d_deepen(k) for the convection pulse and the matched control pulse.

    d_deepen == skill_conv_analyze's definition:
        deepen(arm) = ic_mslp - min(arm mslp);  d_deepen = deepen(base) - deepen(arm)
    which equals min(arm mslp) - min(base mslp); ic_mslp cancels.
    """
    out = {}
    for name, r in runs.items():
        res = r["res"]
        base = np.asarray(res["baseline"]["mslp_min"], float)
        base_min = float(np.min(base))
        ic = float(truth[name]["mslp_min"][0]) if truth and name in truth else np.nan
        base_dp = ic - base_min
        # fallback scale for the normalised panel when there is no ERA5 truth: the baseline
        # FORECAST's own deepening. Labelled as such wherever it is used; never mixed with
        # the ERA5-referenced number.
        norm_dp = base_dp if np.isfinite(base_dp) else float(base[0] - base_min)
        argmin_h = int(np.argmin(base))          # step index of the baseline MSLP minimum
        rec = {"argmin_h": argmin_h, "argmin_lead_h": 6 * (argmin_h + 1),
               "nondev": bool(r.get("nondev", False)),
               "ic_mslp": ic, "base_deepen": base_dp, "norm_deepen": norm_dp,
               "base_min": base_min,
               "conv_grp": [int(f) for f in r.get("conv", [])],
               "ctrl_grp": [int(f) for f in r.get("rand", [])],
               "conv": {}, "ctrl": {}}
        for k in PULSE_KS:
            for tag, a in (("conv", conv_arm(k)), ("ctrl", ctrl_arm(k))):
                am = float(np.min(np.asarray(res[a]["mslp_min"], float)))
                rec[tag][k] = am - base_min        # >0 => deepening lost
        out[name] = rec
    return out

def ratio(vals_by_k, storm, tag, notes):
    """early/late ratio of MEDIANS, with the declared zero-denominator rule."""
    early = float(np.median([vals_by_k[k] for k in EARLY_KS]))
    late = float(np.median([vals_by_k[k] for k in LATE_KS]))
    if late > EPS_HPA:
        return early / late, early, late
    if early > 2 * EPS_HPA:
        notes.append(f"{storm}/{tag}: late median {late:+.3f} hPa <= EPS, early {early:+.3f} "
                     f"-> ratio = inf (front-loaded with nil late damage)")
        return np.inf, early, late
    notes.append(f"{storm}/{tag}: early {early:+.3f} and late {late:+.3f} hPa both <= EPS "
                 f"-> ratio UNDEFINED, scored as FAIL (no signal to front-load)")
    return np.nan, early, late

def perm_attainability(vals_by_k, rng, bar=BAR_RATIO, n=N_PERM):
    """Guardrail #9 leg (ii): with THESE nine numbers, is a >=`bar` ratio reachable at all?

    Permute the k-labels of the storm's own nine damage values and recompute the ratio.
    P(ratio >= bar) == 0 means the bar cannot be attained by this arm no matter how the
    damage is arranged in time -- i.e. the bar is vacuous for it and the control cannot fail.
    """
    v = np.array([vals_by_k[k] for k in PULSE_KS], float)
    ei = [PULSE_KS.index(k) for k in EARLY_KS]
    li = [PULSE_KS.index(k) for k in LATE_KS]
    hits, rs = 0, []
    for _ in range(n):
        p = rng.permutation(v)
        e = np.median(p[ei])
        l = np.median(p[li])
        if l > EPS_HPA:
            rr = e / l
        elif e > 2 * EPS_HPA:
            rr = np.inf
        else:
            rr = np.nan
        rs.append(rr)
        if meets(rr, bar):
            hits += 1
    rs = np.array(rs, float)
    fin = rs[np.isfinite(rs)]
    return dict(p_attain=hits / n,
                q95=float(np.percentile(fin, 95)) if fin.size else float("nan"),
                q50=float(np.percentile(fin, 50)) if fin.size else float("nan"),
                n_inf=int(np.sum(rs == np.inf)), n_nan=int(np.sum(np.isnan(rs))))

def internal_decay(runs):
    """Readout 2: how long does a single-step dose survive in the model's OWN state?

    rel(h) = (arm - baseline) / baseline for the dosed group's summed in-box activation.
    tau = h - k steps after the dose (tau=0 IS the dose). Persistence = |rel(k+tau)|/|rel(k)|.
    """
    per = {}
    for name, r in runs.items():
        res = r["res"]
        grp = [int(f) for f in r.get("conv", [])]
        ctl = [int(f) for f in r.get("rand", [])]
        rec = {"conv": {}, "ctrl": {}}
        for tag, group, armf in (("conv", grp, conv_arm), ("ctrl", ctl, ctrl_arm)):
            if not group:
                continue
            b = np.mean([np.asarray(res["baseline"]["box_feats"][f], float) for f in group], axis=0)
            for k in PULSE_KS:
                a = np.mean([np.asarray(res[armf(k)]["box_feats"][f], float) for f in group], axis=0)
                den = np.where(np.abs(b) > 1e-9, np.abs(b), np.nan)
                rel = (a - b) / den
                taus = np.arange(0, H_EXPECT - k)
                rec[tag][k] = dict(rel=rel[k:], taus=taus,
                                   base=b[k:], dose_rel=float(rel[k]) if np.isfinite(rel[k]) else np.nan)
        per[name] = rec
    return per

# ---------------------------------------------------------------- verdict
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir", default="results/skill/commit_horizon")
    ap.add_argument("--storms", default="skill_conv_storms")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--truth", default=None,
                    help="era5_truth.npy for the ABSOLUTE deepening column only. Auto-searched "
                         "if omitted; d_deepen never depends on it.")
    ap.add_argument("--gate-verbose", action="store_true",
                    help="print every gate row, not just the failures and the family census")
    args = ap.parse_args()

    resdir = ROOT / args.resdir
    S = importlib.import_module("graphcast_sae.common." + args.storms)
    names = list(S.STORMS.keys())
    tagname = resdir.name
    figpath = ROOT / (args.fig if args.fig else
                         ("figures/commit_horizon.png" if tagname == "commit_horizon"
                          else f"figures/commit_horizon_{tagname}.png"))
    jsonpath = (ROOT / args.json) if args.json else (resdir / "commit_horizon_verdict.json")

    print("=" * 100)
    print(f"THE COMMITMENT HORIZON  --  {resdir}  ({args.storms}, {len(names)} storms)")
    print("=" * 100)
    print(f"pulse steps k = {PULSE_KS}  ->  dose leads {[6*k for k in PULSE_KS]} h")
    print(f"early = k<={max(EARLY_KS)} (leads {[6*k for k in EARLY_KS]} h)   "
          f"late = k>={min(LATE_KS)} (leads {[6*k for k in LATE_KS]} h)")
    print(f"BAR (pre-registered): convection early/late >= {BAR_RATIO}x in >= {BAR_NSTORMS} storms "
          f"AND control < {CTRL_BAR}x")

    gate = Gate()
    runs = load_runs(resdir, names, gate)
    gate.report(verbose=args.gate_verbose)
    if not runs:
        print("\nNo runs passed the gate. Nothing to score.")
        return 1

    # era5_truth.npy supplies ONLY the IC MSLP used for the absolute `deepen` column; the
    # damage readout d_deepen is a difference of two forecasts and the IC term cancels, so a
    # missing truth file cannot change a single scored number. run_queue16/17 do not run
    # skill_conv_verify_era5 into this battery's directory, so the same-registry truth files
    # already on disk are searched next -- identical storms, identical ICs -- and the file
    # actually used is printed, plus a per-storm IC cross-check against the run's own `ic`.
    cands = [resdir / "era5_truth.npy"] if args.truth is None else [ROOT / args.truth]
    if args.truth is None:
        cands += [ROOT / "results/skill/convection/era5_truth.npy",
                  ROOT / "results/skill/sh_convection/era5_truth.npy"]
    truth, tp = None, None
    for c in cands:
        if not c.exists():
            continue
        t = np.load(c, allow_pickle=True).item()
        if all(n in t for n in runs):
            truth, tp = t, c
            break
    if truth is None:
        print("\n  NOTE: no era5_truth.npy covering these storms was found "
              f"(looked in {[str(c) for c in cands]}). d_deepen does not depend on it "
              "(the IC term cancels); the absolute-deepening column is nan and panel (b) "
              "falls back to the baseline forecast's own deepening.")
    else:
        print(f"\n  ERA5 IC pressures from {tp} (absolute `deepen` column only):")
        for n in runs:
            t0 = str(np.asarray(truth[n]["times"])[0])[:13]
            ic = str(runs[n].get("ic", ""))[:13]
            same = t0.startswith(ic[:10])
            print(f"    {n:<14s} truth t0={t0}  run ic={ic}  "
                  + ("ok" if same else "<-- IC MISMATCH, absolute deepening not comparable"))

    D = damage_curves(runs, truth)
    dev = [n for n in names if n in D and not D[n]["nondev"]]
    nd = [n for n in names if n in D and D[n]["nondev"]]

    # ---------------- readout 1: the curve --------------------------------
    print("\n" + "=" * 100)
    print("READOUT 1 -- damage curve: d_deepen(k) = deepening LOST by a single-step pulse (hPa)")
    print("            (skill_conv_analyze's deepen/d_deepen; >0 = less deepening than baseline)")
    print("=" * 100)
    hdr = "  storm          base_dp |" + "".join(f"{6*k:>7d}h" for k in PULSE_KS)
    for tag, lbl in (("conv", "CONVECTION"), ("ctrl", "IN-BOX CTRL")):
        print(f"\n  {lbl} pulse damage, hPa   (columns are the DOSE lead)")
        print(hdr)
        for n in dev + nd:
            mark = "  <- non-developer (never pooled)" if D[n]["nondev"] else ""
            print(f"  {n:<14s} {D[n]['base_deepen']:6.1f} |"
                  + "".join(f"{D[n][tag][k]:8.2f}" for k in PULSE_KS) + mark)
        pooled = [float(np.median([D[n][tag][k] for n in dev])) for k in PULSE_KS]
        print(f"  {'POOLED median':<14s} {'':6s} |" + "".join(f"{v:8.2f}" for v in pooled))
        pooledm = [float(np.mean([D[n][tag][k] for n in dev])) for k in PULSE_KS]
        print(f"  {'POOLED mean':<14s} {'':6s} |" + "".join(f"{v:8.2f}" for v in pooledm))

    # normalized, so storms of very different depth are comparable
    _nsrc = ("ERA5 IC -> forecast minimum" if truth is not None
             else "BASELINE FORECAST's own deepening (no ERA5 truth found -- fallback)")
    print(f"\n  CONVECTION damage as % of that storm's own baseline deepening"
          f"   [denominator = {_nsrc}]")
    print(hdr.replace("base_dp", "norm_dp"))
    normed = {}
    for n in dev + nd:
        bd = D[n]["norm_deepen"]
        row = [100 * D[n]["conv"][k] / bd if np.isfinite(bd) and bd > 1e-6 else np.nan
               for k in PULSE_KS]
        normed[n] = row
        print(f"  {n:<14s} {bd:6.1f} |" + "".join(f"{v:8.1f}" for v in row)
              + ("  <- non-developer (never pooled)" if D[n]["nondev"] else ""))
    if dev:
        pn = [float(np.median([normed[n][i] for n in dev])) for i in range(len(PULSE_KS))]
        print(f"  {'POOLED median':<14s} {'':6s} |" + "".join(f"{v:8.1f}" for v in pn))

    # -------- structural confound in the readout, checked before the bar --------
    # d_deepen reads min() over the trajectory. A pulse at step k can only move that minimum
    # if the minimum lies at or after step k. Where it does not, the arm's damage is 0 BY
    # CONSTRUCTION, not by physics -- and since late k are the most likely to fall past the
    # minimum, that alone manufactures front-loading. The matched control eats the identical
    # constraint (same storms, same minima, same k), which is what keeps the comparison
    # honest; this block reports the exposure so the reader can see how much of the ratio is
    # mechanical.
    print("\n" + "=" * 100)
    print("STRUCTURAL EXPOSURE OF THE min()-BASED READOUT  (read this before the ratio)")
    print("=" * 100)
    print("  A pulse applied after the baseline MSLP minimum cannot move that minimum: its")
    print("  d_deepen is 0 by construction. The control eats the same constraint, so it is")
    print("  calibrated, not eliminated. Storms where late k are structurally dead:")
    print("  storm          baseline min at lead   dead early k       dead late k")
    n_dead_late = 0
    for n in dev + nd:
        am = D[n]["argmin_h"]
        de = [k for k in EARLY_KS if k > am]
        dl = [k for k in LATE_KS if k > am]
        n_dead_late += len(dl)
        print(f"  {n:<14s} {D[n]['argmin_lead_h']:14d} h   {str(de):<16s}   {str(dl):<16s}"
              + ("   <-- ALL late k dead" if len(dl) == len(LATE_KS) else ""))
    print(f"  => {n_dead_late}/{len(LATE_KS) * len(dev + nd)} late-k arms are structurally "
          "incapable of damage under this readout.")
    print("  If that count is large, the early/late ratio is partly an artefact of min() and the")
    print("  honest statement is the CONTROL-DIFFERENCED one: conv ratio vs ctrl ratio, not the")
    print("  conv ratio alone. Both are reported below.")

    # ---------------- the bar ---------------------------------------------
    notes = []
    R = {}
    for n in dev + nd:
        rc, ec, lc = ratio(D[n]["conv"], n, "conv", notes)
        rk, ek, lk = ratio(D[n]["ctrl"], n, "ctrl", notes)
        R[n] = dict(conv_ratio=rc, conv_early=ec, conv_late=lc,
                    ctrl_ratio=rk, ctrl_early=ek, ctrl_late=lk)

    print("\n" + "=" * 100)
    print("THE PRE-REGISTERED BAR  --  early/late ratio of MEDIANS, per storm")
    print("=" * 100)
    print("  storm           conv_early conv_late  conv_ratio  PASS | ctrl_early ctrl_late  ctrl_ratio  <1.3?")
    npass = 0
    nctrl_bad = 0
    for n in dev:
        r = R[n]
        p = meets(r["conv_ratio"], BAR_RATIO)
        npass += bool(p)
        cb = meets(r["ctrl_ratio"], CTRL_BAR)
        nctrl_bad += bool(cb)
        print(f"  {n:<14s} {r['conv_early']:9.2f} {r['conv_late']:9.2f} {r['conv_ratio']:11.2f}  "
              f"{'YES' if p else ' no'}  | {r['ctrl_early']:9.2f} {r['ctrl_late']:9.2f} "
              f"{r['ctrl_ratio']:11.2f}  {'no ' if cb else 'yes'}")
    for n in nd:
        r = R[n]
        print(f"  {n:<14s} {r['conv_early']:9.2f} {r['conv_late']:9.2f} {r['conv_ratio']:11.2f}  "
              f"  - | {r['ctrl_early']:9.2f} {r['ctrl_late']:9.2f} {r['ctrl_ratio']:11.2f}    -"
              "   <- NATURAL NULL, reported separately, never pooled")
    if notes:
        print("\n  zero-denominator rule fired on:")
        for x in notes:
            print(f"    {x}")

    # sensitivity: the prereg does not say median/mean/|.|, so all three are printed
    print("\n  SENSITIVITY (the prereg does not specify how to aggregate within a k-group):")
    for agg_name, agg in (("median", np.median), ("mean", np.mean), ("max", np.max)):
        cs, ks = [], []
        for n in dev:
            e = agg([D[n]["conv"][k] for k in EARLY_KS]); l = agg([D[n]["conv"][k] for k in LATE_KS])
            cs.append(e / l if l > EPS_HPA else (np.inf if e > 2 * EPS_HPA else np.nan))
            e = agg([D[n]["ctrl"][k] for k in EARLY_KS]); l = agg([D[n]["ctrl"][k] for k in LATE_KS])
            ks.append(e / l if l > EPS_HPA else (np.inf if e > 2 * EPS_HPA else np.nan))
        cs = np.array(cs, float); ks = np.array(ks, float)
        print(f"    agg={agg_name:<6s} conv pass {sum(meets(v, BAR_RATIO) for v in cs)}/{len(dev)}"
              f"  conv median ratio {np.nanmedian(cs[np.isfinite(cs)]) if np.any(np.isfinite(cs)) else float('nan'):6.2f}"
              f" | ctrl >= {CTRL_BAR}: {sum(meets(v, CTRL_BAR) for v in ks)}/{len(dev)}"
              f"  ctrl median ratio {np.nanmedian(ks[np.isfinite(ks)]) if np.any(np.isfinite(ks)) else float('nan'):6.2f}")
    absr = []
    for n in dev:
        e = np.median([abs(D[n]["conv"][k]) for k in EARLY_KS])
        l = np.median([abs(D[n]["conv"][k]) for k in LATE_KS])
        absr.append(e / l if l > EPS_HPA else np.inf)
    absr = np.array(absr, float)
    print(f"    |damage| median-ratio: conv pass "
          f"{sum(meets(v, BAR_RATIO) for v in absr)}/{len(dev)}")

    # ---------------- the control-must-be-able-to-fail rule, all three legs -------------------------
    rng = np.random.default_rng(PERM_SEED)
    print("\n" + "=" * 100)
    print("GUARDRAIL #9 -- the bar calibrated on BOTH sides. All three legs, with numbers.")
    print("=" * 100)

    print("\n  LEG (i)  DOES THE CONTROL'S OWN CURVE VARY ACROSS k?  (a point mass = instrument failure)")
    print("           storm            ctrl range(hPa)   ctrl sd    ctrl min..max        conv range")
    leg1 = True
    for n in dev + nd:
        v = np.array([D[n]["ctrl"][k] for k in PULSE_KS], float)
        c = np.array([D[n]["conv"][k] for k in PULSE_KS], float)
        rngv = float(np.ptp(v))
        if n in dev and rngv < 0.01:
            leg1 = False
        print(f"           {n:<14s} {rngv:12.3f} {float(np.std(v)):10.3f}   "
              f"{v.min():7.2f}..{v.max():6.2f} {float(np.ptp(c)):16.3f}"
              + ("   <- POINT MASS" if rngv < 0.01 else "")
              + ("   (non-developer)" if D[n]["nondev"] else ""))
    print(f"           => LEG (i) {'PASS' if leg1 else 'FAIL'}: the control varies across dose lead in "
          f"{sum(1 for n in dev if np.ptp([D[n]['ctrl'][k] for k in PULSE_KS]) >= 0.01)}/{len(dev)} storms")

    print(f"\n  LEG (ii) IS A >={BAR_RATIO}x RATIO ATTAINABLE BY THE CONTROL?")
    print(f"           Permute the k-labels of each storm's own nine control damages "
          f"({N_PERM} draws) and")
    print("           recompute the ratio. P(attain) = 0 would mean the bar is unreachable => vacuous.")
    print("           storm            P(ratio>=2 | k-labels shuffled)   perm q50    perm q95")
    leg2 = True
    attain = {}
    for n in dev:
        a = perm_attainability(D[n]["ctrl"], rng)
        attain[n] = a
        if a["p_attain"] < 0.01:
            leg2 = False
        print(f"           {n:<14s} {a['p_attain']:24.3f} {a['q50']:11.2f} {a['q95']:11.2f}"
              + ("   <- BAR NOT ATTAINABLE" if a["p_attain"] < 0.01 else ""))
    pooled_attain = float(np.mean([attain[n]["p_attain"] for n in dev])) if dev else float("nan")
    print(f"           => LEG (ii) {'PASS' if leg2 else 'FAIL'}: mean P(attain) = {pooled_attain:.3f}; "
          f"{sum(1 for n in dev if attain[n]['p_attain'] >= 0.01)}/{len(dev)} storms can reach the bar")

    print(f"\n  LEG (iii) DOES THE CONTROL FAIL THE BAR?  (pre-registered: control ratio < {CTRL_BAR})")
    cr = np.array([R[n]["ctrl_ratio"] for n in dev], float)
    # +inf IS a control ratio (front-loaded with nil late damage); dropping infs before the
    # median would systematically flatter the control, which is the one thing a control
    # must not be flattered on. Only nan (no signal at either end) is excluded.
    crf = cr[~np.isnan(cr)]
    med_ctrl = float(np.median(crf)) if crf.size else float("nan")
    n_ctrl_over = sum(meets(v, CTRL_BAR) for v in cr)
    n_ctrl_over2 = sum(meets(v, BAR_RATIO) for v in cr)
    leg3 = bool(n_ctrl_over2 < BAR_NSTORMS and np.isfinite(med_ctrl) and med_ctrl < CTRL_BAR)
    print(f"           control ratios: {np.array2string(cr, precision=2)}"
          f"   ({int(np.sum(np.isnan(cr)))} nan = no signal, excluded from the median)")
    print(f"           median control ratio = {med_ctrl:.2f} (bar: < {CTRL_BAR})")
    print(f"           storms with control ratio >= {CTRL_BAR}: {n_ctrl_over}/{len(dev)}")
    print(f"           storms with control ratio >= {BAR_RATIO}: {n_ctrl_over2}/{len(dev)} "
          f"(the treatment's own bar)")
    if leg3:
        _m3 = "PASS (control fails the bar, as required)"
    elif not np.isfinite(med_ctrl):
        _m3 = ("FAIL -- but not because the control front-loads: its ratio is UNDEFINED for "
               f"{int(np.sum(np.isnan(cr)))}/{len(dev)} storms (no signal at either end). "
               "That is an instrument failure, see leg (i).")
    else:
        _m3 = "FAIL -- the control front-loads too; report as early-lead sensitivity in general"
    print(f"           => LEG (iii) {_m3}")

    # not pre-registered, but free: is the convection ordering itself special?
    print("\n  (not pre-registered, reported for completeness) k-label permutation p-value for the")
    print("   convection arm: P(shuffled ratio >= observed ratio).")
    for n in dev:
        v = np.array([D[n]["conv"][k] for k in PULSE_KS], float)
        obs = R[n]["conv_ratio"]
        if np.isnan(obs):
            print(f"           {n:<14s} observed      nan   p = n/a  (no signal at either end; "
                  "not scoreable)")
            continue
        ei = [PULSE_KS.index(k) for k in EARLY_KS]; li = [PULSE_KS.index(k) for k in LATE_KS]
        cnt = 0
        for _ in range(N_PERM):
            p = rng.permutation(v)
            e, l = np.median(p[ei]), np.median(p[li])
            rr = e / l if l > EPS_HPA else (np.inf if e > 2 * EPS_HPA else -np.inf)
            if rr >= obs:
                cnt += 1
        print(f"           {n:<14s} observed {obs:8.2f}   p = {cnt / N_PERM:.4f}")

    # ---------------- readout 2: internal persistence ----------------------
    print("\n" + "=" * 100)
    print("READOUT 2 -- how long does the dose survive in the MODEL'S OWN STATE?")
    print("            rel(tau) = (arm - baseline)/baseline of the dosed group's in-box activation,")
    print("            tau steps (6h each) after the dose. tau=0 IS the dose.")
    print("=" * 100)
    DEC = internal_decay(runs)
    TAUS = [0, 1, 2, 3, 4]
    for tag, lbl in (("conv", "CONVECTION group"), ("ctrl", "IN-BOX CONTROL group")):
        print(f"\n  {lbl}: |rel(tau)| / |rel(0)|, pooled over storms and pulse steps (median)")
        print("    tau (steps)      " + "".join(f"{t:>10d}" for t in TAUS))
        print("    lead after dose  " + "".join(f"{6*t:>9d}h" for t in TAUS))
        rows = []
        for n in dev:
            for k in PULSE_KS:
                d = DEC[n][tag].get(k)
                if d is None:
                    continue
                rel = d["rel"]
                if not np.isfinite(rel[0]) or abs(rel[0]) < 1e-9:
                    continue
                rows.append([abs(rel[t]) / abs(rel[0]) if t < len(rel) and np.isfinite(rel[t]) else np.nan
                             for t in TAUS])
        A = np.array(rows, float)
        if A.size:
            med = np.nanmedian(A, axis=0)
            print("    median ratio     " + "".join(f"{v:10.3f}" for v in med)
                  + f"      (n={A.shape[0]} storm x pulse pairs)")
            print("    p90              " + "".join(
                f"{np.nanpercentile(A[:, i], 90):10.3f}" for i in range(len(TAUS))))
        else:
            print("    no usable pairs (baseline in-box activation is zero for this group)")

    # washout vs damage: the strong statement
    print("\n  THE CROSS-STATEMENT: pulses whose internal trace has washed out by tau=2")
    print("  (|rel(2)| < 10% of |rel(0)|) -- do they still move the +96 h MSLP?")
    wash, notwash = [], []
    for n in dev:
        for k in PULSE_KS:
            d = DEC[n]["conv"].get(k)
            if d is None:
                continue
            rel = d["rel"]
            if not np.isfinite(rel[0]) or abs(rel[0]) < 1e-9 or len(rel) < 3 or not np.isfinite(rel[2]):
                continue
            (wash if abs(rel[2]) < 0.1 * abs(rel[0]) else notwash).append(
                (n, k, D[n]["conv"][k], abs(rel[2]) / abs(rel[0])))
    for lbl, grp in (("washed out by tau=2", wash), ("still present at tau=2", notwash)):
        if grp:
            dv = np.array([g[2] for g in grp], float)
            print(f"    {lbl:<24s} n={len(grp):3d}   median damage {np.median(dv):+7.3f} hPa   "
                  f"max {dv.max():+7.3f} hPa")
        else:
            print(f"    {lbl:<24s} n=  0")
    if wash:
        big = sorted(wash, key=lambda g: -g[2])[:6]
        print("    largest damages from a dose that had already washed out of the representation:")
        for n, k, dmg, fr in big:
            print(f"      {n:<14s} pulse k={k:2d} (lead {6*k:3d}h)  damage {dmg:+6.2f} hPa   "
                  f"|rel(2)|/|rel(0)| = {fr:.3f}")

    # ---------------- verdict ---------------------------------------------
    bar_conv = npass >= BAR_NSTORMS
    ctrl_clean = leg3
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  convection storms passing >= {BAR_RATIO}x front-loading: {npass}/{len(dev)} developers"
          f"  (bar: >= {BAR_NSTORMS})")
    if len(dev) + len(nd) != len(names):
        print(f"  !! DENOMINATOR CHANGED: the registry has {len(names)} storms but only "
              f"{len(dev) + len(nd)} passed the gate. The bar's '>= {BAR_NSTORMS}' is scored "
              "against the storms that exist; the gate failures above are the reason.")
    if nd:
        print("  AMBIGUITY IN THE PREREG (flagged, NOT reinterpreted): the bar reads "
              f"'>= {BAR_NSTORMS} of 8 storms', but {len(nd)} of the {len(names)} registry "
              "storms is the non-developing")
        print("  natural null, which by construction has no deepening to lose and is required "
              f"to be reported separately and never pooled. The count above uses the {len(dev)} "
              "DEVELOPERS as the")
        print(f"  denominator while keeping the literal '>= {BAR_NSTORMS}' threshold -- the "
              "strictest reading. Scored over all "
              f"{len(names)} it would be "
              f"{npass + sum(meets(R[n]['conv_ratio'], BAR_RATIO) for n in nd)}/{len(names)}.")
    if len(dev) < BAR_NSTORMS:
        print(f"  !! THE BAR IS NOT SCOREABLE AS WRITTEN ON THIS BATTERY: it demands "
              f"'>= {BAR_NSTORMS} of 8 storms' and only {len(dev)} developing storms exist here "
              f"({args.storms}).")
        print("  A threshold of 6 cannot be met by a 5-storm battery. It is NOT rescaled here -- "
              "rescaling a pre-registered count after the fact is exactly the move the prereg "
              "exists to prevent.")
        print(f"  The raw count ({npass}/{len(dev)}) and every ratio are reported above; the "
              "pass/fail decision is deferred to whoever amends the prereg.")
    print(f"  control ratio < {CTRL_BAR}: {'YES' if ctrl_clean else 'NO'}")
    print(f"  guardrail #9 legs: (i) vary={'PASS' if leg1 else 'FAIL'}  "
          f"(ii) attainable={'PASS' if leg2 else 'FAIL'}  (iii) control fails bar={'PASS' if leg3 else 'FAIL'}")
    if len(dev) < BAR_NSTORMS:
        verdict = (f"BAR NOT SCOREABLE AS WRITTEN -- '>= {BAR_NSTORMS} of 8' cannot be evaluated "
                   f"on {len(dev)} developing storms; raw count {npass}/{len(dev)}, "
                   f"guardrail #9 legs {'PASS' if (leg1 and leg2 and leg3) else 'see above'}")
    elif not (leg1 and leg2):
        verdict = "INSTRUMENT FAILURE -- the control cannot fail, so the bar certifies nothing"
    elif bar_conv and ctrl_clean:
        verdict = "PASS -- convection damage is front-loaded and the matched in-box control is not"
    elif bar_conv and not ctrl_clean:
        verdict = ("REPORTED AS EARLY-LEAD SENSITIVITY IN GENERAL -- the control front-loads too, "
                   "so this is not a statement about convection (prereg's own fallback)")
    else:
        verdict = "CLEAN NEGATIVE -- convection pulse damage is NOT front-loaded at the pre-registered bar"
    print(f"  => {verdict}")
    if nd:
        for n in nd:
            print(f"  natural null {n}: baseline deepening {D[n]['base_deepen']:.1f} hPa, "
                  f"conv ratio {R[n]['conv_ratio']:.2f}, max conv damage "
                  f"{max(D[n]['conv'].values()):.2f} hPa (reported separately, never pooled)")

    out = dict(resdir=str(resdir), storms=args.storms, pulse_ks=PULSE_KS,
               early_ks=EARLY_KS, late_ks=LATE_KS,
               bar=dict(ratio=BAR_RATIO, n_storms=BAR_NSTORMS, ctrl=CTRL_BAR),
               developers=dev, nondev=nd,
               damage={n: dict(conv={str(k): D[n]["conv"][k] for k in PULSE_KS},
                               ctrl={str(k): D[n]["ctrl"][k] for k in PULSE_KS},
                               base_deepen=D[n]["base_deepen"], norm_deepen=D[n]["norm_deepen"],
                               nondev=D[n]["nondev"])
                       for n in D},
               ratios={n: {k: (None if not np.isfinite(v) and not np.isinf(v) else
                               ("inf" if np.isinf(v) else v)) if isinstance(v, float) else v
                           for k, v in R[n].items()} for n in R},
               n_pass=int(npass), n_dev=len(dev),
               argmin_lead_h={n: D[n]["argmin_lead_h"] for n in D},
               n_dead_late_arms=int(sum(len([k for k in LATE_KS if k > D[n]["argmin_h"]])
                                        for n in dev + nd)),
               legs=dict(vary=bool(leg1), attainable=bool(leg2), control_fails=bool(leg3),
                         mean_p_attain=pooled_attain),
               gate=dict(n_checks=len(gate.rows), n_fatal=gate.fatal, n_warn=gate.warn),
               verdict=verdict)
    jsonpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n-> {jsonpath}")

    make_figure(D, R, DEC, dev, nd, figpath, tagname)
    return 0

# ---------------------------------------------------------------- figure
def make_figure(D, R, DEC, dev, nd, figpath, tagname):
    leads = np.array([6 * k for k in PULSE_KS], float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    a0, a1, a2, a3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) raw hPa
    for n in dev:
        a0.plot(leads, [D[n]["conv"][k] for k in PULSE_KS], color=C_CONV, lw=0.9, alpha=0.35)
        a0.plot(leads, [D[n]["ctrl"][k] for k in PULSE_KS], color=C_CTRL, lw=0.9, alpha=0.35)
    for n in nd:
        a0.plot(leads, [D[n]["conv"][k] for k in PULSE_KS], color=C_ND, lw=1.2, ls=":",
                alpha=0.9, label=f"{n} (non-developer, not pooled)")
    if dev:
        a0.plot(leads, [np.median([D[n]["conv"][k] for n in dev]) for k in PULSE_KS],
                color=C_CONV, lw=3.0, marker="o", ms=5, label="convection — pooled median")
        a0.plot(leads, [np.median([D[n]["ctrl"][k] for n in dev]) for k in PULSE_KS],
                color=C_CTRL, lw=3.0, marker="s", ms=5, label="in-box matched control — pooled median")
    a0.axhline(0, color="#888", lw=0.8)
    a0.set_xlabel("lead at which the single 6-h pulse is applied (h)")
    a0.set_ylabel("deepening lost, hPa  (d_deepen)")
    a0.set_title("(a) damage from a ONE-STEP dose, by dose lead", loc="left", fontsize=10)
    a0.legend(fontsize=7, loc="best")

    # (b) normalized
    for n in dev:
        bd = D[n]["norm_deepen"]
        if np.isfinite(bd) and bd > 1e-6:
            a1.plot(leads, [100 * D[n]["conv"][k] / bd for k in PULSE_KS], color=C_CONV, lw=0.9, alpha=0.35)
            a1.plot(leads, [100 * D[n]["ctrl"][k] / bd for k in PULSE_KS], color=C_CTRL, lw=0.9, alpha=0.35)
    okd = [n for n in dev if np.isfinite(D[n]["norm_deepen"]) and D[n]["norm_deepen"] > 1e-6]
    if okd:
        a1.plot(leads, [np.median([100 * D[n]["conv"][k] / D[n]["norm_deepen"] for n in okd])
                        for k in PULSE_KS], color=C_CONV, lw=3.0, marker="o", ms=5,
                label="convection — pooled median")
        a1.plot(leads, [np.median([100 * D[n]["ctrl"][k] / D[n]["norm_deepen"] for n in okd])
                        for k in PULSE_KS], color=C_CTRL, lw=3.0, marker="s", ms=5,
                label="in-box control — pooled median")
    a1.axhline(0, color="#888", lw=0.8)
    a1.set_xlabel("lead at which the single 6-h pulse is applied (h)")
    a1.set_ylabel("% of that storm's baseline deepening lost")
    a1.set_title("(b) same, normalised per storm", loc="left", fontsize=10)
    a1.legend(fontsize=7, loc="best")

    # (c) per-storm early/late ratio
    x = np.arange(len(dev)); w = 0.38
    CAP = 20.0

    def cap(v):
        if v == np.inf:
            return CAP
        return np.nan if not np.isfinite(v) else min(v, CAP)

    a2.bar(x - w / 2, [cap(R[n]["conv_ratio"]) for n in dev], w, color=C_CONV, label="convection")
    a2.bar(x + w / 2, [cap(R[n]["ctrl_ratio"]) for n in dev], w, color=C_CTRL, label="in-box control")
    for i, n in enumerate(dev):
        if R[n]["conv_ratio"] == np.inf:
            a2.text(x[i] - w / 2, CAP * 1.02, "inf", ha="center", fontsize=7, color=C_CONV)
        if R[n]["ctrl_ratio"] == np.inf:
            a2.text(x[i] + w / 2, CAP * 1.02, "inf", ha="center", fontsize=7, color=C_CTRL)
        if np.isnan(R[n]["conv_ratio"]):
            a2.text(x[i] - w / 2, 1.02, "n/a", ha="center", fontsize=7, color=C_CONV, rotation=90)
        if np.isnan(R[n]["ctrl_ratio"]):
            a2.text(x[i] + w / 2, 1.02, "n/a", ha="center", fontsize=7, color=C_CTRL, rotation=90)
    a2.axhline(BAR_RATIO, color="#c0392b", ls="--", lw=1.2, label=f"pre-reg bar {BAR_RATIO}x (convection)")
    a2.axhline(CTRL_BAR, color="#2a78d6", ls=":", lw=1.2, label=f"control must stay under {CTRL_BAR}x")
    a2.axhline(1.0, color="#888", lw=0.8)
    a2.set_yscale("log")
    a2.set_xticks(x); a2.set_xticklabels(dev, rotation=30, ha="right", fontsize=8)
    a2.set_ylabel(f"damage(dose <= {6*max(EARLY_KS)}h) / damage(dose >= {6*min(LATE_KS)}h)")
    a2.set_title("(c) front-loading ratio per storm (capped at 20x for display)", loc="left", fontsize=10)
    a2.legend(fontsize=7, loc="best")

    # (d) internal persistence
    TAUS = list(range(0, 6))
    for tag, col, lbl, ls in (("conv", C_CONV, "convection group", "-"),
                              ("ctrl", C_CTRL, "in-box control group", "--")):
        rows = []
        for n in dev:
            for k in PULSE_KS:
                d = DEC[n][tag].get(k)
                if d is None:
                    continue
                rel = d["rel"]
                if not np.isfinite(rel[0]) or abs(rel[0]) < 1e-9:
                    continue
                rows.append([abs(rel[t]) / abs(rel[0]) if t < len(rel) and np.isfinite(rel[t]) else np.nan
                             for t in TAUS])
        A = np.array(rows, float)
        if A.size:
            med = np.nanmedian(A, axis=0)
            lo = np.nanpercentile(A, 25, axis=0); hi = np.nanpercentile(A, 75, axis=0)
            hrs = np.array(TAUS) * 6.0
            a3.plot(hrs, med, color=col, lw=2.6, ls=ls, marker="o", ms=4,
                    label=f"{lbl} (n={A.shape[0]})")
            a3.fill_between(hrs, lo, hi, color=col, alpha=0.18, lw=0)
    a3.axhline(0.1, color="#888", ls="--", lw=0.9)
    a3.text(1, 0.105, "10% of the dose", fontsize=7, color="#666")
    a3.set_yscale("log")
    a3.set_xlabel("hours AFTER the dose")
    a3.set_ylabel("|rel. in-box activation dev.| / value at the dose")
    a3.set_title("(d) how fast the dose washes out of the model's own state", loc="left", fontsize=10)
    a3.legend(fontsize=7, loc="best")

    fig.suptitle(f"The commitment horizon — a single 6-h convection pulse, by when it is applied  [{tagname}]",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {figpath}")

if __name__ == "__main__":
    sys.exit(main())
