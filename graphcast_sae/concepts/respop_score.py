"""Score the concept response operators against docs/prereg/prereg_response_operator.md.

Reads results/fs_respop.npy. No new forwards. system python3.

Paper: Fig. fig:contrast: '0/10 concepts detected' verdict
Inputs: results/fs_respop.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/respop_score.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.concepts.respop_score
"""
from pathlib import Path
import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
R = np.load(ROOT / "results/fs_respop.npy", allow_pickle=True).item()
RMS, A50, A90, PROF, CEN = R["RMS"], R["A50"], R["A90"], R["PROF"], R["CEN"]
arms = list(R["arms"]); names = list(R["names"]); fields = list(R["fields"])
OWN = dict(R["own"]); S = int(R["S"]); ICS = list(R["ics"])
LE = list(R["lat_edges"])
ai = {a: i for i, a in enumerate(arms)}
nf = [ai["nf0"], ai["nf1"]]
pi = [ai[a] for a in arms if a.startswith("perm")]
lead_h = [(s + 1) * 6 for s in range(S)]

SNR_BAR, LOCAL_BAR, GLOBAL_BAR, Z_BAR = 3.0, 0.10, 0.35, 2.0
L = []
def P(s=""):
    print(s, flush=True); L.append(s)

def own_idx(c):
    o = OWN[c]
    return len(fields) - 1 if o == "SHEAR" else fields.index(o)

# window-mean statistics
rms = RMS.mean(0)                     # (arm, lead, field)
a50 = A50.mean(0); a90 = A90.mean(0); cen = CEN.mean(0); prof = PROF.mean(0)
floor = rms[nf].mean(0)               # (lead, field)  -- the MEASURED numeric floor

P("=" * 90)
P("CONCEPT RESPONSE OPERATORS — scored against the frozen prereg")
P("=" * 90)
P(f"{len(names)} concepts x K={R['K']}, gamma={R['gamma']}, S={S} steps (60 h), "
  f"ICs {ICS}")
P("")

# ---------------------------------------------------------------- the floor --
P("-- NF-1 NUMERIC FLOOR (nf0/nf1: UNPATCHED rolls, identical code path) --")
P("   The amp-0 arm is not zero: the GPU forward is not bit-deterministic and the")
P("   float-level difference grows chaotically through the roll.")
P(f"   {'lead':>6}" + "".join(f"{f:>12}" for f in ("z500 m", "t850 K", "q600", "w500")))
fi_show = [0, fields.index("t850"), fields.index("q600"), fields.index("w500")]
for s in range(S):
    P(f"   {lead_h[s]:>5}h" + "".join(f"{floor[s, f]:>12.3e}" for f in fi_show))
vac = bool((floor[:, 0] <= 0).any())
P(f"   floor non-zero at every lead: {not vac} -> "
  f"{'BAR IS FAILABLE' if not vac else '** VACUOUS: zero floor, nothing can fail it **'}")
P(f"   floor spread nf0 vs nf1 (z500, ratio at 60 h): "
  f"{RMS.mean(0)[nf[0], -1, 0] / max(RMS.mean(0)[nf[1], -1, 0], 1e-30):.3f}")
P("")

# negative control that must FAIL the DETECTED bar
snr_nf1 = rms[nf[1], :, 0] / rms[nf[0], :, 0]
P("-- NEGATIVE CONTROL that must FAIL the DETECTED bar --")
P(f"   nf1 scored as if it were an arm, against nf0: max SNR over leads = "
  f"{snr_nf1.max():.2f}  (bar {SNR_BAR})")
ctrl_ok = snr_nf1.max() < SNR_BAR
P(f"   -> {'PASS: an unpatched roll is NOT DETECTED, the floor is honest' if ctrl_ok else '** FAIL: the floor is mis-estimated; no arm is read **'}")
P("")

# perm null must VARY
P("-- SPECIFICITY NULL (10 perm arms: same 40 features, scrambled labels) --")
P(f"   {'lead':>6}{'perm rms mean':>16}{'perm rms sd':>14}{'perm A50 mean':>16}"
  f"{'perm A50 sd':>14}")
for s in range(S):
    P(f"   {lead_h[s]:>5}h{rms[pi, s, 0].mean():>16.4g}{rms[pi, s, 0].std(ddof=1):>14.4g}"
      f"{a50[pi, s, 0].mean():>16.4f}{a50[pi, s, 0].std(ddof=1):>14.4f}")
varies = bool(rms[pi, :, 0].std(ddof=1, axis=0).min() > 0)
P(f"   null VARIES across the 10 perm arms: {varies} -> "
  f"{'z-scores are meaningful' if varies else '** VACUOUS **'}")
P("")

# ------------------------------------------------------- the response operators
P("=" * 90)
P("RESPONSE OPERATORS — primary field z500 (geopotential metres)")
P("=" * 90)
P(f"  {'concept':<15}{'rms@6h':>10}{'rms@24h':>10}{'rms@60h':>10}{'growth':>9}"
  f"{'SNRmax':>9}{'@lead':>7}{'A50':>7}{'A90':>7}{'cenlat':>8}{'z_mag':>8}{'z_A50':>8}"
  f"   DETECTED / STRUCTURE / SPECIFIC")
rows = {}
for c in names:
    k = ai[c]
    snr = rms[k, :, 0] / floor[:, 0]
    sbest = int(np.argmax(snr))
    zm = (rms[k, sbest, 0] - rms[pi, sbest, 0].mean()) / max(rms[pi, sbest, 0].std(ddof=1), 1e-30)
    za = (a50[k, sbest, 0] - a50[pi, sbest, 0].mean()) / max(a50[pi, sbest, 0].std(ddof=1), 1e-30)
    det = snr.max() >= SNR_BAR
    st = ("LOCAL" if a50[k, sbest, 0] <= LOCAL_BAR else
          "GLOBAL" if a50[k, sbest, 0] >= GLOBAL_BAR else "INTERMEDIATE")
    spec = max(abs(zm), abs(za)) >= Z_BAR
    rows[c] = dict(snr=snr, sbest=sbest, zm=float(zm), za=float(za),
                   det=bool(det), st=st, spec=bool(spec))
    P(f"  {c:<15}{rms[k,0,0]:>10.4g}{rms[k,3,0]:>10.4g}{rms[k,-1,0]:>10.4g}"
      f"{rms[k,-1,0]/max(rms[k,0,0],1e-30):>9.1f}{snr.max():>9.1f}"
      f"{lead_h[sbest]:>6}h{a50[k,sbest,0]:>7.3f}{a90[k,sbest,0]:>7.3f}"
      f"{cen[k,sbest,0]:>+8.1f}{zm:>+8.2f}{za:>+8.2f}   "
      f"{'DETECTED' if det else 'not detected':<13}{st:<14}"
      f"{'SPECIFIC' if spec else 'not specific'}")
P("")
P(f"  perm arms for reference (same dose, scrambled labels):")
P(f"  {'arm':<15}{'rms@60h':>10}{'SNRmax':>9}{'A50@best':>10}")
for j, k in enumerate(pi):
    snr = rms[k, :, 0] / floor[:, 0]
    b = int(np.argmax(snr))
    P(f"  {arms[k]:<15}{rms[k,-1,0]:>10.4g}{snr.max():>9.1f}{a50[k,b,0]:>10.3f}")
P("")

# ------------------------------------------------------------ own-field block
P("=" * 90)
P("SECONDARY — each concept in its OWN governing field (prereg RESPOP-3 map)")
P("=" * 90)
P(f"  {'concept':<15}{'field':<10}{'rms@6h':>12}{'rms@60h':>12}{'floor@60h':>12}"
  f"{'SNRmax':>9}{'@lead':>7}{'A50':>7}{'cenlat':>8}")
for c in names:
    k, f = ai[c], own_idx(c)
    snr = rms[k, :, f] / np.maximum(floor[:, f], 1e-30)
    b = int(np.argmax(snr))
    P(f"  {c:<15}{OWN[c]:<10}{rms[k,0,f]:>12.4g}{rms[k,-1,f]:>12.4g}"
      f"{floor[-1,f]:>12.4g}{snr.max():>9.1f}{lead_h[b]:>6}h{a50[k,b,f]:>7.3f}"
      f"{cen[k,b,f]:>+8.1f}")
P("")

# ------------------------------------------------------------ latitude shape -
P("=" * 90)
P("SPATIAL STRUCTURE — |dz500| by latitude band at the lead of max SNR "
  "(fraction of the row total)")
P("=" * 90)
bands = [f"{LE[i]}..{LE[i+1]}" for i in range(len(LE) - 1)]
P(f"  {'concept':<15}" + "".join(f"{b:>11}" for b in bands))
for c in names:
    k, b = ai[c], rows[c]["sbest"]
    p = prof[k, b, 0]; p = p / max(p.sum(), 1e-30)
    P(f"  {c:<15}" + "".join(f"{v:>11.3f}" for v in p))
pm = prof[pi][:, :, 0].mean(0)
for s in [rows[names[0]]["sbest"]]:
    p = pm[s] / max(pm[s].sum(), 1e-30)
    P(f"  {'perm mean':<15}" + "".join(f"{v:>11.3f}" for v in p))
P("")

# ------------------------------------------------------------------ summary --
P("=" * 90)
nd = sum(r["det"] for r in rows.values())
ns = sum(r["spec"] for r in rows.values())
loc = [c for c in names if rows[c]["st"] == "LOCAL"]
glo = [c for c in names if rows[c]["st"] == "GLOBAL"]
P(f"SUMMARY  DETECTED {nd}/{len(names)}   CONCEPT-SPECIFIC {ns}/{len(names)}")
P(f"  LOCAL: {loc or 'none'}")
P(f"  GLOBAL: {glo or 'none'}")
P(f"  INTERMEDIATE: {[c for c in names if rows[c]['st'] == 'INTERMEDIATE'] or 'none'}")
P(f"  calibration: floor failable {not vac}; negative control passes {ctrl_ok}; "
  f"specificity null varies {varies}")
P("=" * 90)

(ROOT / "results/respop_score.txt").write_text("\n".join(L) + "\n")
print("\n-> results/respop_score.txt")
