"""Score the flagship intervention runs against the pre-registered bars.

Applies `docs/prereg/prereg_flagship_g2_suite.md` §3 (retry #5 ablation) and §4 (retry #2
steering) exactly as written, including the amended p-floor at 10 controls and the
underpowered/inconclusive verdicts — which are reported as such, never resolved in
the favourable direction.

Paper: not in the paper; kept for provenance only
Inputs: results/graphcast_sae_ablation.npy (not shipped, see docs/REPRODUCE.md); results/graphcast_sae_steering.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/graphcast_sae_scores.json
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.fs_score [--which both]
"""
import argparse
import json

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import FS_MODES

def score_ablation(path="results/graphcast_sae_ablation.npy"):
    d = np.load(fc.ROOT / path, allow_pickle=True).item()
    names, E = d["names"], d["E"]
    nsteps = int(d["nsteps"])
    lead = nsteps - 1                                   # primary lead = last (24 h)
    med = np.nanmedian(E[:, :, lead], axis=1)
    idx = {n: i for i, n in enumerate(names)}
    n_noise, r_recon = med[idx["noise"]], med[idx["recon"]]
    out = {"primary_lead_h": 6 * (lead + 1), "noise": float(n_noise),
           "recon": float(r_recon), "targets": {}}
    print(f"\n=== retry #5 ablation @ {6*(lead+1)} h ===")
    print(f"noise floor {n_noise:.3e}   recon (SAE FVU in forecast space) {r_recon:.3e}")
    verdicts = []
    for t in d["targets"]:
        g = med[idx[f"gl{t}"]]
        cs = np.array([med[idx[n]] for n in names if n.startswith(f"ctrl{t}_")])
        p = (1 + int((cs <= g).sum())) / (cs.size + 1)
        Q = g / np.median(cs)
        # CPU forward is bit-deterministic => n_noise == 0 exactly, which makes the
        # small-model guards (i)/(iv) vacuous. Prereg §3 determinism amendment: the
        # `recon` arm becomes the floor reference.
        vacuous = n_noise == 0.0
        gate = np.median(cs) > 10 * n_noise
        power = (g < 0.01 * r_recon) if vacuous else (g < 3 * n_noise)
        v = ("INCONCLUSIVE (control-validity gate failed)" if not gate else
             "INCONCLUSIVE (floor: g < 1% of the SAE's own recon effect)" if power else
             "INERT" if (p <= 1.0 / (cs.size + 1) + 1e-9 and Q < 0.5) else
             "reliably below matched controls but NOT inert" if p <= 0.0911 else
             "NOT inert")
        verdicts.append(v == "INERT")
        print(f"\nfeature {t}: g={g:.3e}  median control={np.median(cs):.3e}  "
              f"Q={Q:.3f}  p={p:.3f} (floor {1/(cs.size+1):.3f} at n={cs.size})")
        print(f"  control-validity gate {'PASS' if gate else 'FAIL'}; "
              f"power {'FAIL (underpowered)' if power else 'ok'}  -> {v}")
        if vacuous:
            print("  NOTE: CPU forward is bit-deterministic (noise floor exactly 0) -> "
                  "guards (i) and (iv) are VACUOUS here, not passed; `recon` is the floor.")
        out["targets"][int(t)] = dict(g=float(g), ctrl_median=float(np.median(cs)),
                                      guards_vacuous=bool(vacuous),
                                      ctrl=cs.tolist(), p=float(p), p_floor=1/(cs.size+1),
                                      Q=float(Q), gate_passed=bool(gate),
                                      underpowered=bool(power), verdict=v)
    out["passed"] = bool(any(verdicts))
    print(f"\nBAR(#5 ablation): >=1 target inert -> "
          f"{'PASS' if out['passed'] else 'MISS'}")
    out["lead_curve"] = {n: np.nanmedian(E[i], axis=0).tolist() for i, n in enumerate(names)
                         if not n.startswith("ctrl")}
    return out

def dual_basis(modes=str(FS_MODES)):
    """G^-1 for the mode basis, plus each community's raw-readout geometric floor.

    The saved `R_c = <outer(W_c,Q_c)/sigma_c, dA>` is a CORRELATION with each mode, not a
    projection onto them; with a non-orthogonal basis that manufactures leakage from nothing.
    The Gram is separable, G_jk = (W_j.W_k)(Q_j.Q_k)/(sigma_j sigma_k), so it depends only on
    the basis and `G^-1 R` is recoverable post-hoc with no re-run.
    See `internal note 'flagship_steering_geometry' (not shipped)` (registered before any leakage number existed).
    """
    m = np.load(modes, allow_pickle=True)
    W = np.asarray(m["W"], np.float64)
    Q = np.asarray(m["Q"], np.float64)
    sig = np.clip(np.asarray(m["sigma"], np.float64), 1e-12, None)
    G = (W @ W.T) * (Q @ Q.T) / np.outer(sig, sig)
    Gi = np.linalg.inv(G)
    floors = []
    for c in range(len(W)):
        a = np.abs(np.einsum("kn,nd,kd->k", W, np.outer(W[c], Q[c]), Q) / sig)
        floors.append(float(np.delete(a, c).mean() / a[c]))
    cond = float(np.linalg.cond(G))
    return Gi, np.array(floors), cond

def score_steering(path="results/graphcast_sae_steering.npy"):
    d = np.load(fc.ROOT / path, allow_pickle=True).item()
    names, R = d["names"], d["R"]                       # (arms, win, steps, C)
    handles, alphas = d["handles"], np.asarray(d["alphas"])
    idx = {n: i for i, n in enumerate(names)}
    s = 0                                              # prereg §4 primary lead 6 h
    # --- instrument check: is this readout measuring the network, or the patch? -----------
    # The patch is applied at layer 8 and R reads layer 8. At s=0 the patch is ON, so dA is
    # exactly the injected delta = alpha * f(n) * d_f -- linear in alpha to roundoff, with a
    # dose-invariant direction. That readout cannot be sensitive to the network under ANY
    # outcome, so a verdict from it would be VOID, not a MISS.
    # See internal note 'flagship_steering_s0_defect' (not shipped).
    def _dose_spread(step):
        sp = []
        for f, c, _ in handles:
            g = [np.nanmean(np.abs(R[idx[f"f{f}_c{c}_a{a:+g}"], :, step, :]), axis=0)[c]
                 / abs(a) for a in alphas]
            g = np.array(g)
            sp.append((g.max() - g.min()) / max(g.mean(), 1e-30))
        return float(np.max(sp))
    spread = _dose_spread(s)
    if spread < 1e-5:
        print(f"\n=== retry #2 steering @ 6 h (s={s}) ===")
        print(f"*** READOUT VOID *** response is linear in alpha to {spread:.1e} (float32 "
              f"roundoff) and the leakage direction is dose-invariant: at s=0 the patch is ON "
              f"at the very layer R reads, so dA IS the injected delta. This measures SAE "
              f"decoder geometry, not the network. No verdict is issued from it.")
        print("See internal note 'flagship_steering_s0_defect' (not shipped). A valid test needs the primary lead "
              "at s>=1 (or a feature-space readout), re-pre-registered.")
        alt = [t for t in range(R.shape[2]) if _dose_spread(t) >= 1e-5]
        out = {"prereg_readout_void": True, "dose_spread_s0": spread,
               "reason": "patch applied at the readout layer; dA(s=0) == injected delta",
               "note": "internal note 'flagship_steering_s0_defect' (not shipped)",
               "valid_dynamical_steps": alt, "passed": None}
        if alt:
            s = alt[0]
            print(f"\n--- s={s} ({6*(s+1)} h, patch OFF) reported as a DESCRIPTIVE COMPANION "
                  f"only: not pre-registered, no bar was ever frozen for it, and it was "
                  f"selected after the pre-registered readout failed. Nothing is claimed. ---")
        else:
            print("\nno step in this run has a dynamical readout; nothing to report.")
            return out
    else:
        out = {"prereg_readout_void": False, "dose_spread_s0": spread}
    floor = np.nanmean(np.abs(R[idx["noise"], :, s, :]))
    Gi, geo, cond = dual_basis()
    # Same determinism amendment as §3: a bit-deterministic CPU forward makes the noise
    # floor exactly 0, so "on-target >= 10x floor" is satisfied by ANY non-zero response.
    # The gate is therefore VACUOUS, and is reported as vacuous rather than as passed.
    # R is already sigma-normalized, so on-target |R| is directly in units of the
    # community's own deseasonalized natural variability -- that is the descriptive
    # power statement that replaces it. Registered before any leakage number existed.
    vac = floor == 0.0
    print(f"\n=== retry #2 steering @ {6*(s+1)} h (s={s}) ===\n"
          f"numerical floor |R|={floor:.3e}")
    if vac:
        print("  NOTE: floor is exactly 0 (bit-deterministic CPU forward) -> the >=10x-floor "
              "power gate is VACUOUS here, not passed. on-target |R| is reported in units of "
              "the community's own natural variability instead (R is sigma-normalized).")
    print(f"mode-basis condition number {cond:.1f}; raw-readout geometric leakage floor "
          f"per community: {np.array2string(geo, precision=3)}")
    print("primary metric is the pre-registered L; L_dual is the registered companion "
          "(dual-basis projection, geometric floor removed) -- both always reported.")
    out.update({"lead_h": 6 * (s + 1), "step": int(s), "floor": float(floor),
                "power_gate_vacuous": bool(vac), "geometric_floor": geo.tolist(),
                "gram_condition": cond, "features": {}})
    passed = False
    for f, c, r in handles:
        Ls, L2s, Lds, on = [], [], [], []
        for a in alphas:
            row = R[idx[f"f{f}_c{c}_a{a:+g}"], :, s, :]     # (win, C)
            m = np.nanmean(np.abs(row), axis=0)
            on.append(m[c])
            off = np.delete(m, c)
            Ls.append(off.mean() / max(m[c], 1e-30))
            L2s.append(1 - m[c] ** 2 / max((m ** 2).sum(), 1e-30))
            md = np.nanmean(np.abs((Gi @ row.T).T), axis=0)  # dual-basis coefficients
            Lds.append(np.delete(md, c).mean() / max(md[c], 1e-30))
        L, L2, on_med = float(np.mean(Ls)), float(np.mean(L2s)), float(np.median(on))
        Ld = float(np.mean(Lds))
        powered = np.isfinite(on_med) and (on_med > 0 if vac else on_med >= 10 * floor)
        ok = powered and L < 0.5
        passed |= ok
        scale = ("in units of the community's own variability" if vac
                 else f"{on_med/max(floor,1e-30):.1f}x floor")
        print(f"feature {f:5d} -> community {c:2d}: leakage L={L:.3f} (its geometric floor "
              f"{geo[c]:.3f})  L_dual={Ld:.3f}  energy L2={L2:.3f}  "
              f"on-target |R|={on_med:.3e} ({scale})"
              f"{'' if powered else '  UNDERPOWERED (L not reported as a number)'}")
        if powered and L < 0.5 and L < geo[c]:
            print("    NOTE: L is below this community's own geometric floor -- at or beyond "
                  "the resolution of the raw readout; read L_dual, not L.")
        if powered and L < 0.5 and Ld >= 0.5:
            print("    DISAGREEMENT: passes on the pre-registered L but fails the "
                  "geometry-corrected companion -- the PASS is readout geometry, report as such.")
        elif powered and L >= 0.5 and Ld < 0.5:
            print("    DISAGREEMENT: fails L but passes L_dual -- the leakage is basis "
                  "overlap, NOT the network mixing communities.")
        out["features"][int(f)] = dict(community=int(c), handle_corr=float(r),
                                       leakage=L if powered else None,
                                       leakage_dual=Ld if powered else None,
                                       geometric_floor=float(geo[c]),
                                       energy_leakage=L2 if powered else None,
                                       on_target=on_med, powered=bool(powered),
                                       passes=bool(ok))
    nd = [v for v in out["features"].values() if v["leakage_dual"] is not None]
    if out.get("prereg_readout_void"):
        out["passed"] = None
        out["descriptive_L_under_bar"] = sum(v["leakage"] < 0.5 for v in nd)
        out["descriptive_Ldual_under_bar"] = sum(v["leakage_dual"] < 0.5 for v in nd)
        print(f"\nBAR(#2): **NOT SCORED** -- the pre-registered readout is void (see above). "
              f"Descriptively at s={s}: {out['descriptive_L_under_bar']}/{len(nd)} handles "
              f"under 0.5 on L, {out['descriptive_Ldual_under_bar']}/{len(nd)} on L_dual. "
              f"These are NOT a verdict.")
    else:
        out["passed"] = bool(passed)
        print(f"\nBAR(#2, pre-registered L): leakage < 0.5 for >=1 feature at >=10x floor -> "
              f"{'PASS' if passed else 'MISS'}")
        if nd:
            print(f"companion L_dual: {sum(v['leakage_dual'] < 0.5 for v in nd)}/{len(nd)} "
                  f"powered handles under 0.5 (not the bar; reported for comparison)")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["both", "ablation", "steering"])
    args = ap.parse_args()
    res = {"prereg": "docs/prereg/prereg_flagship_g2_suite.md",
           "note": "flagship-internal; NOT cross-comparable to the small-model G1 rung"}
    if args.which in ("both", "ablation"):
        try:
            res["ablation"] = score_ablation()
        except FileNotFoundError:
            print("no ablation result yet")
    if args.which in ("both", "steering"):
        try:
            res["steering"] = score_steering()
        except FileNotFoundError:
            print("no steering result yet")
    json.dump(res, open(fc.ROOT / "results/graphcast_sae_scores.json", "w"), indent=1)
    print("\nsaved -> results/graphcast_sae_scores.json")

if __name__ == "__main__":
    main()
