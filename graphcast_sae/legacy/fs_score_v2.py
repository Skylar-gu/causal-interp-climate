"""Score flagship steering v2 against `docs/prereg/prereg_flagship_steering_v2.md`.

Primary metric `L_feat` in the feature-space (hard-partition) basis at s=1; the v1
mode-basis `L` and its dual-basis companion `L_dual` reported alongside at every step.
Bars applied exactly as frozen, including the INCONCLUSIVE override and the VOID
dose-spread guard.

Paper: not in the paper; kept for provenance only
Inputs: results/graphcast_sae_steering_v2.npy (not shipped, see docs/REPRODUCE.md); results/graphcast_sae_steering_v2_ctrl.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/graphcast_sae_scores_v2.json
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.fs_score_v2
"""
import argparse
import json

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.legacy.fs_score import dual_basis

BAR = 0.5
POWER_FRAC = 0.1            # on-target rel must be >= 10% of the `recon` arm's, same community
VOID_SPREAD = 1e-5          # dose spread below this == readout is measuring the injection

def leak(mat, t):
    """mean off-target / on-target for one (C,) response vector."""
    return float(np.delete(mat, t).mean() / max(mat[t], 1e-30))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/graphcast_sae_steering_v2.npy")
    ap.add_argument("--step", type=int, default=1)      # prereg primary: s=1 (12 h)
    args = ap.parse_args()

    d = np.load(fc.ROOT / args.path, allow_pickle=True).item()
    names, REL, RMODE = d["names"], d["REL"], d["RMODE"]
    handles, ctrls, al = d["handles"], d["controls"], np.asarray(d["alphas"])
    i = {n: j for j, n in enumerate(names)}
    Gi, geo, cond = dual_basis()
    s = args.step
    C = REL.shape[-1]

    print(f"=== flagship retry #2 v2 — feature-space readout @ {6*(s+1)} h (s={s}) ===")
    print(f"prereg docs/prereg/prereg_flagship_steering_v2.md | communities {d['comm_sizes']} "
          f"(hard partition, 977 features)")
    if s == 0:
        print("*** s=0 IS NEVER SCORED (prereg §3). Aborting. ***"); return

    # ---- VOID guard: does the response actually vary with dose? (prereg §3) ----
    spreads = []
    for f, c, _ in handles:
        g = np.array([np.nanmean(REL[i[f"f{f}_c{c}_a{a:+g}"], :, s, c]) / abs(a) for a in al])
        spreads.append((g.max() - g.min()) / max(g.mean(), 1e-30))
    spread = float(np.max(spreads))
    print(f"dose-spread guard: max relative gain spread {spread:.3e} "
          f"(VOID if < {VOID_SPREAD:.0e})")
    if spread < VOID_SPREAD:
        print("*** RUN VOID *** response is linear in alpha to roundoff — the readout is "
              "measuring the injection again, not the network. No verdict issued.")
        json.dump({"void": True, "dose_spread": spread},
                  open(fc.ROOT / "results/graphcast_sae_scores_v2.json", "w"), indent=1)
        return

    noise = float(np.nanmean(REL[i["noise"], :, s, :]))
    rec = np.nanmean(REL[i["recon"], :, s, :], axis=0)          # (C,)
    print(f"noise arm rel={noise:.3e} (bit-deterministic CPU forward -> any '>=10x noise "
          f"floor' gate is VACUOUS, not passed; power is referenced to `recon`)")
    print(f"recon arm rel per community: {np.array2string(rec, precision=4)}")

    out = {"void": False, "step": s, "lead_h": 6 * (s + 1), "dose_spread": spread,
           "noise": noise, "recon_rel": rec.tolist(), "handles": {}, "controls": {}}

    def score(f, c, alphas):
        Lf, Lm, Ld, on = [], [], [], []
        for a in alphas:
            key = f"f{f}_c{c}_a{a:+g}" if f"f{f}_c{c}_a{a:+g}" in i else f"ctrl{f}_c{c}_a{a:+g}"
            r = np.nanmean(REL[i[key], :, s, :], axis=0)
            Lf.append(leak(r, c)); on.append(r[c])
            row = RMODE[i[key], :, s, :]
            m = np.nanmean(np.abs(row), axis=0)
            Lm.append(leak(m, c))
            Ld.append(leak(np.nanmean(np.abs((Gi @ row.T).T), axis=0), c))
        return (float(np.mean(Lf)), float(np.mean(Lm)), float(np.mean(Ld)),
                float(np.median(on)))

    print("\n--- handles (primary L_feat; L / L_dual are the v1 mode-basis companions) ---")
    any_pass = False
    for f, c, r in handles:
        Lf, Lm, Ld, on = score(f, c, al)
        powered = on >= POWER_FRAC * rec[c]
        ok = powered and Lf < BAR
        any_pass |= ok
        print(f" f{f:<5d}->c{c}  L_feat={Lf:6.3f}   L={Lm:6.3f}  L_dual={Ld:6.3f}   "
              f"on-target rel={on:.3e} ({on/max(rec[c],1e-30)*100:.1f}% of recon)"
              f"{'' if powered else '   UNDERPOWERED'}{'   <- clears bar' if ok else ''}")
        out["handles"][int(f)] = dict(community=int(c), L_feat=Lf, L_mode=Lm, L_dual=Ld,
                                      on_target=on, recon_frac=float(on / max(rec[c], 1e-30)),
                                      powered=bool(powered), passes=bool(ok))

    print("\n--- negative controls (random features, mass-matched, alpha=+1) ---")
    cL = []
    for f, c in ctrls:
        Lf, Lm, Ld, on = score(f, c, [1.0])
        cL.append(Lf)
        print(f" f{f:<5d}->c{c}  L_feat={Lf:6.3f}   L={Lm:6.3f}  L_dual={Ld:6.3f}   "
              f"on-target rel={on:.3e}")
        out["controls"][int(f)] = dict(community=int(c), L_feat=Lf, L_mode=Lm, L_dual=Ld,
                                       on_target=on)
    cmed_frozen = float(np.median(cL))
    out["control_median_L_feat_frozen_n2"] = cmed_frozen
    out["n_controls_frozen"] = len(cL)

    # ---- pooled control set (prereg v2 §4b amendment, disclosed post-hoc) --------------
    ext = fc.ROOT / "results/graphcast_sae_steering_v2_ctrl.npy"
    if ext.exists():
        e = np.load(ext, allow_pickle=True).item()
        eREL, eRM = e["REL"], e["RMODE"]
        ei = {n: j for j, n in enumerate(e["names"])}
        print(f"\n--- control EXTENSION ({len(e['controls'])} more, prereg §4b amendment; "
              f"can only weaken or neutralize the PASS) ---")
        for f, c in e["controls"]:
            r = np.nanmean(eREL[ei[f"ctrl{f}_c{c}_a+1"], :, s, :], axis=0)
            row = eRM[ei[f"ctrl{f}_c{c}_a+1"], :, s, :]
            m = np.nanmean(np.abs(row), axis=0)
            Lf, Lm = leak(r, c), leak(m, c)
            Ld = leak(np.nanmean(np.abs((Gi @ row.T).T), axis=0), c)
            cL.append(Lf)
            print(f" f{f:<5d}->c{c}  L_feat={Lf:6.3f}   L={Lm:6.3f}  L_dual={Ld:6.3f}   "
                  f"on-target rel={r[c]:.3e}")
            out["controls"][int(f)] = dict(community=int(c), L_feat=Lf, L_mode=Lm,
                                           L_dual=Ld, on_target=float(r[c]),
                                           from_extension=True)
    cmed = float(np.median(cL))
    uninformative = cmed < BAR
    out["control_median_L_feat"] = cmed
    out["n_controls"] = len(cL)
    out["control_L_feat_all"] = [float(x) for x in cL]
    out["metric_uninformative"] = bool(uninformative)
    out["controls_under_bar"] = int(sum(x < BAR for x in cL))

    print(f"\ncontrol median L_feat = {cmed:.3f} over n={len(cL)}"
          f"  ({out['controls_under_bar']}/{len(cL)} individually below the {BAR} bar)")
    if len(cL) > out["n_controls_frozen"]:
        print(f"  frozen n={out['n_controls_frozen']} median was {cmed_frozen:.3f} -> "
              f"pooled n={len(cL)} median {cmed:.3f} "
              f"(amendment effect shown, not absorbed)")
    if uninformative:
        verdict = ("INCONCLUSIVE (metric uninformative: a random feature also reads as "
                   "localized, so no handle result can be claimed)")
    elif any_pass:
        verdict = "PASS"
    else:
        verdict = "MISS"
    out["verdict"] = verdict
    print(f"\nBAR(#2 v2): >=1 handle with L_feat < {BAR} at >={POWER_FRAC:.0%} of recon, "
          f"AND control median >= {BAR}  ->  {verdict}")

    # secondary lead, descriptive
    if REL.shape[2] > s + 1:
        s2 = s + 1
        sec = {}
        for f, c, _ in handles:
            Lf = np.mean([leak(np.nanmean(REL[i[f"f{f}_c{c}_a{a:+g}"], :, s2, :], axis=0), c)
                          for a in al])
            sec[int(f)] = float(Lf)
        print(f"\nsecondary lead {6*(s2+1)} h (descriptive): "
              + "  ".join(f"f{f}={v:.3f}" for f, v in sec.items()))
        out["secondary_lead_h"] = 6 * (s2 + 1)
        out["secondary_L_feat"] = sec

    json.dump(out, open(fc.ROOT / "results/graphcast_sae_scores_v2.json", "w"), indent=1)
    print("\nsaved -> results/graphcast_sae_scores_v2.json")

if __name__ == "__main__":
    main()
