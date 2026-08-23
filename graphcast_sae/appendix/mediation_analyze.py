"""Score a mediation battery: exactness gate FIRST, then the path decomposition.

Reads results/skill/<MED_NAME>/run_<storm>.npy written by mediation_run.py. CPU only.

STEP 1 -- THE CORRECTNESS TEST. Nothing below it is reported unless it passes.

  (a) CLAMP BOUND. In every arm where j is frozen, j's own in-box activation series must
      equal the baseline series. This is the definition of the clamp, and it is checked
      directly rather than assumed. Tolerance: the measured in-box floor 0.0017 scaled by
      the feature's own baseline amplitude (docs/notes/nondeterminism_floor_2026_08_20.md).
  (b) FREEZE-ONLY IS A NO-OP. `freeze-j` with nothing else patched must reproduce
      `baseline`. It is scored against `noop6` -- a second untouched arm of the SAME
      compiled graph in the SAME process -- so the bar is this run's own nondeterminism,
      not a number quoted from another day. PASS iff max|dMSLP(freeze-j)| <= max(
      max|dMSLP(noop6)|, FLOOR_MAX) with FLOOR_MAX = 0.61 hPa, the largest pre-dose drift
      ever measured here.

STEP 2 -- THE DECOMPOSITION. Y = d_deepen, exactly skill_conv_analyze's definition
(deepen = IC pressure - min MSLP over the box; the IC term cancels in the difference, so
d_deepen(arm) = min MSLP(arm) - min MSLP(baseline), positive = the arm cost deepening).

    TE(i)      = d_deepen( do-i )                      total effect
    DE(i|j)    = d_deepen( do-i + freeze-j )           direct effect, j's pathway removed
    ME(i,j)    = TE - DE                               mediated through j
    frac(i,j)  = ME / TE
    OWN(j)     = d_deepen( do-j )                      j's own effect on Y

  frac near 1 with a small OWN(j) is a chain i -> j -> Y. frac near 0 is a fork. A large
  OWN(j) with frac near 0 says j matters but not for THIS effect. Every frac is reported
  beside the amplitude-matched control's frac; a mediation claim requires the difference
  between them to exceed the floor propagated through the same arithmetic, and if the
  control's frac is comparable the readout is measuring "freezing anything".

Paper: Appendix app:topk
Inputs: none beyond the arguments above
Outputs: printed report
Run:   # JAX env, CPU
    MED_NAME=med_f2681_haishen2020 python -m graphcast_sae.appendix.mediation_analyze
"""
import os
import sys

import numpy as np

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
NAME = os.environ.get("MED_NAME") or (sys.argv[1] if len(sys.argv) > 1 else None)
FLOOR_MAX = 0.61        # hPa, largest pre-dose min-MSLP drift measured (nondeterminism note)
FLOOR_MED = 0.15        # hPa, median of the same
INBOX_REL = 0.0017      # relative in-box activation floor

def gate_inputs(d):
    """Guardrail #6: nothing is scored before the file itself is checked."""
    bad = []
    res = d["res"]
    H = int(d["H"])
    for a, r in res.items():
        m = np.asarray(r["mslp_min"], float)
        if m.shape != (H,):
            bad.append("%s: mslp_min shape %s != (%d,)" % (a, m.shape, H))
        if not np.isfinite(m).all():
            bad.append("%s: non-finite mslp_min" % a)
        if np.allclose(m, 0):
            bad.append("%s: mslp_min all zero" % a)
        bf = r["box_feats"]
        if len(bf) == 0:
            bad.append("%s: no box_feats" % a)
    for need in ("baseline", "noop6"):
        if need not in res:
            bad.append("missing arm %r -- the exactness test cannot be calibrated" % need)
    return bad

def main():
    if not NAME:
        raise SystemExit("set MED_NAME or pass the battery name as argv[1]")
    dirp = os.path.join(ROOT, "results", "skill", NAME)
    files = sorted(f for f in os.listdir(dirp) if f.startswith("run_") and f.endswith(".npy"))
    if not files:
        raise SystemExit("no run_*.npy in %s" % dirp)
    for fn in files:
        d = np.load(os.path.join(dirp, fn), allow_pickle=True).item()
        res = d["res"]; src = int(d["src"]); meds = list(d["meds"]); ctls = list(d["ctls"])
        print("\n" + "=" * 78)
        print("%s   storm=%s  source=f%d  scope=%s  freeze nodes=%d/%d"
              % (NAME, d["storm"], src, d["scope"], d["n_freeze_nodes"], 40962))
        bad = gate_inputs(d)
        if bad:
            print("DATA GATE FAILED:"); [print("   ", b) for b in bad]
            continue
        base = np.asarray(res["baseline"]["mslp_min"], float)
        bmin = float(base.min())

        def dd(a):
            return float(np.min(np.asarray(res[a]["mslp_min"], float)) - bmin)

        def dmax(a):
            return float(np.abs(np.asarray(res[a]["mslp_min"], float) - base).max())

        # ---------------------------------------------------------- STEP 1
        print("\nSTEP 1  exactness")
        noop = dmax("noop6")
        print("  noop6 (this graph's own nondeterminism)   max|dMSLP| = %.4f hPa "
              "(reference floor: median %.2f / max %.2f)" % (noop, FLOOR_MED, FLOOR_MAX))
        bar = max(noop, FLOOR_MAX)
        ok = True
        for j in meds + ctls:
            arm = "freeze-%d" % j
            if arm not in res:
                print("  %-22s  ABSENT -- exactness untested for f%d" % (arm, j)); ok = False
                continue
            v = dmax(arm)
            p = v <= bar
            ok &= p
            print("  %-22s  max|dMSLP| = %.4f hPa   %s (bar %.3f)"
                  % (arm, v, "PASS" if p else "FAIL", bar))
        # clamp-bound check on the frozen feature's own in-box series
        print("  clamp bound (frozen feature's in-box series must equal baseline):")
        for j in meds + ctls:
            b = np.asarray(res["baseline"]["box_feats"][j], float)
            amp = max(float(np.abs(b).max()), 1e-9)
            for arm in ("freeze-%d" % j, "do-%d+freeze-%d" % (src, j)):
                if arm not in res:
                    continue
                a = np.asarray(res[arm]["box_feats"][j], float)
                rel = float(np.abs(a - b).max()) / amp
                p = rel <= INBOX_REL
                ok &= p
                print("    %-24s f%-5d  max|d|/amp = %.2e   %s"
                      % (arm, j, rel, "PASS" if p else "LEAK"))
        if not ok:
            print("\nEXACTNESS TEST FAILED -- a leaky mediator invalidates every number "
                  "below it. Not reporting mediation for this run.")
            continue
        print("  exactness: PASS")

        # ---------------------------------------------------------- STEP 2
        te = dd("do-%d" % src)
        print("\nSTEP 2  path decomposition   TOTAL EFFECT do(f%d) = %+.3f hPa "
              "(%.0fx the median floor)" % (src, te, abs(te) / FLOOR_MED))
        if abs(te) < FLOOR_MAX:
            print("  WARNING: the total effect is inside the noise floor; the "
                  "decomposition of it is not interpretable.")
        print("  %-8s %-6s %10s %10s %10s %8s %10s"
              % ("role", "feat", "OWN(j)", "DIRECT", "MEDIATED", "frac", "max|dMSLP|"))
        rows = {}
        for role, group in (("MEDIATOR", meds), ("CONTROL", ctls)):
            for j in group:
                arm = "do-%d+freeze-%d" % (src, j)
                if arm not in res:
                    continue
                de = dd(arm)
                me = te - de
                fr = me / te if te != 0 else float("nan")
                own = dd("do-%d" % j) if ("do-%d" % j) in res else float("nan")
                rows[j] = (role, own, de, me, fr)
                print("  %-8s f%-5d %+10.3f %+10.3f %+10.3f %8.2f %10.3f"
                      % (role, j, own, de, me, fr, dmax(arm)))
        if meds and ctls:
            print("\n  falsifiability: mediator frac vs amplitude-matched control frac")
            for j, k in zip(meds, ctls):
                if j in rows and k in rows:
                    dfr = rows[j][4] - rows[k][4]
                    resolvable = abs(rows[j][3] - rows[k][3]) > FLOOR_MAX
                    print("    f%-5d %.2f   vs   f%-5d %.2f    delta %+.2f   %s"
                          % (j, rows[j][4], k, rows[k][4], dfr,
                             "RESOLVABLE" if resolvable else
                             "NOT RESOLVABLE (difference inside the %.2f hPa floor)" % FLOOR_MAX))

if __name__ == "__main__":
    main()
