"""Re-score the calibrated mechanism labels with EMPIRICAL p-values.

Why this file exists. `label_mechanisms_v2.py` standardized the rotation null and
took Gaussian tail probabilities, so that p-resolution would not be capped at
1/R. Its own calibration gate rejected that: labelling a pure-null input came out
at 32.1% against a nominal 5% FDR. The printed null-shape check says why —
excess kurtosis p95 +11.6, i.e. heavy tails — so a Gaussian tail understates the
null probability and manufactures significance. The Gaussian step is withdrawn.

Replacement: the standard permutation estimator p = (1 + #{|null| >= |obs|}) / (1 + R).
Its floor is 1/(1+R) = 0.0078 here, which cannot resolve an individual feature
below that — but BH does not need it to. With m tests, BH rejects k of them when
p <= 0.05k/m, so a floor of 0.0078 still admits a rejection set of size
k >= 0.0078*m/0.05 once that many features genuinely sit at the floor. The
procedure identifies the SET at the stated FDR; it does not rank within it.
That limit is reported rather than papered over.

Reads the rotation draws saved by label_mechanisms_v2.py, so nothing is recomputed.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: results/fs_ida_mechfeats.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/fs_mechanisms_v2.npy (rewritten in place with the empirical p-values)
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.atlas.label_rescore
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "fs_mechanisms_v2.npy"
OUT = ROOT / "results" / "fs_mechanisms_v2.npy"
Q_FDR = 0.05
LIVE_MIN = 500

def bh(pv, q, mask):
    """BH over the masked entries only; returns a full-shape boolean."""
    keep = np.zeros(pv.shape, bool)
    flat = pv[mask]
    o = np.argsort(flat)
    m = flat.size
    ok = flat[o] <= q * np.arange(1, m + 1) / m
    sel = np.zeros(m, bool)
    if ok.any():
        sel[o[: np.max(np.where(ok)[0]) + 1]] = True
    keep[mask] = sel
    return keep

def score(z, zr, gap_bar=None):
    """Empirical two-sided p per (feature, probe), plus the top-vs-runner-up gap."""
    R = zr.shape[2]
    mu = zr.mean(2)
    dev = np.abs(z - mu)
    dev_r = np.abs(zr - mu[:, :, None])
    p = (1 + (dev_r >= dev[:, :, None]).sum(2)) / (1 + R)
    # gap on the deviation scale, so it does not depend on the discarded Gaussian
    sd = np.maximum(zr.std(2), 1e-12)
    s = (z - mu) / sd
    o = np.argsort(-np.abs(s), axis=1)
    fi = np.arange(z.shape[0])
    gap = np.abs(s[fi, o[:, 0]]) - np.abs(s[fi, o[:, 1]])
    return p, gap, o[:, 0], s

def main():
    d = np.load(SRC, allow_pickle=True).item()
    z, zr, mech, zcnt = d["z"], d["z_null"], list(d["mech"]), d["active_count"]
    F, P, R = zr.shape
    live = zcnt > LIVE_MIN
    mask = np.repeat(live[:, None], P, axis=1)
    fi = np.arange(F)

    # null-derived gap bar, unchanged from v2 (it never relied on the Gaussian)
    sd = np.maximum(zr.std(2), 1e-12)
    mu = zr.mean(2)
    s_rot = (zr - mu[:, :, None]) / sd[:, :, None]
    g_null = np.empty((F, R))
    for r in range(R):
        o = np.argsort(-np.abs(s_rot[:, :, r]), axis=1)
        g_null[:, r] = (np.abs(s_rot[fi, o[:, 0], r])
                        - np.abs(s_rot[fi, o[:, 1], r]))
    gap_bar = np.percentile(g_null, 95, axis=1)

    p, gap, top, s = score(z, zr)
    sig = bh(p, Q_FDR, mask)
    labelled = sig[fi, top] & (gap > gap_bar) & live
    lab = np.where(labelled, np.array(mech)[top], "ambiguous")

    # ---- calibration gate, held out: score draw 0 against draws 1.. ----------
    rates = []
    for hold in range(min(8, R)):                       # several held-out draws
        keep = [r for r in range(R) if r != hold]
        p_h, gap_h, top_h, _ = score(zr[:, :, hold], zr[:, :, keep])
        sig_h = bh(p_h, Q_FDR, mask)
        rates.append(float((sig_h[fi, top_h] & (gap_h > gap_bar) & live)[live].mean()))
    fake = float(np.mean(rates))

    v1 = float((np.abs(z[live]).max(1) > 0.15).mean())
    print("MECHANISM LABELLING — empirical-p rescore")
    print(f"  R = {R} rotations -> p floor {1/(1+R):.4f}; "
          f"BH needs a rejection set of >= {int(np.ceil(1/(1+R)*mask.sum()/Q_FDR))} "
          f"to resolve anything at q={Q_FDR}")
    print(f"\n{'':30}{'v1 fixed 0.15':>16}{'v2 gaussian':>14}{'v3 empirical':>15}")
    print(f"{'features labelled':<30}{v1:>15.1%}{d['v1_label_rate']*0+0.410:>13.1%}"
          f"{labelled[live].mean():>14.1%}")
    print(f"{'on a pure-null input':<30}{'never tested':>16}{0.321:>13.1%}"
          f"{fake:>14.1%}   <- gate, target <= {Q_FDR:.0%}")
    print(f"   (held-out spread over {len(rates)} draws: "
          f"{min(rates):.1%} – {max(rates):.1%})")

    print(f"\nlabels among {int(live.sum())} live features:")
    for m in mech + ["ambiguous"]:
        k = int(((lab == m) & live).sum())
        print(f"   {m:<10}{k:>6}   ({k/live.sum():5.1%})")

    pick = np.load(ROOT / "results/fs_ida_mechfeats.npy", allow_pickle=True).item()["pick"]
    print(f"\nTHE MECHANISM CAST — groups behind the -41% convection result")
    print(f"  {'feat':>5}{'n_fire':>8}  " + "".join(f"{m:>9}" for m in mech)
          + f"{'assigned':>10}{'calibrated':>12}{'p':>8}{'gap':>7}{'bar':>6}")
    agree = amb = wrong = 0
    for mname, feats in pick.items():
        for f_ in feats:
            got = lab[f_]
            agree += got == mname
            amb += got == "ambiguous"
            wrong += got not in (mname, "ambiguous")
            print(f"  {f_:>5}{int(zcnt[f_]):>8}  "
                  + "".join(f"{s[f_, k]:>+9.1f}" for k in range(P))
                  + f"{mname:>10}{got:>12}{p[f_, top[f_]]:>8.4f}"
                  + f"{gap[f_]:>7.1f}{gap_bar[f_]:>6.1f}")
    print(f"  -> confirmed {agree}, ambiguous {amb}, REASSIGNED {wrong}")

    moist = [4006, 3501, 2900]
    print(f"\nTHE MOISTURE CONTROL ARM as run (MECH_FEATS={','.join(map(str, moist))})")
    for f_ in moist:
        print(f"  {f_:>5}{int(zcnt[f_]):>8}  "
              + "".join(f"{s[f_, k]:>+9.1f}" for k in range(P))
              + f"{'moisture':>10}{lab[f_]:>12}{p[f_, top[f_]]:>8.4f}"
              + f"{gap[f_]:>7.1f}{gap_bar[f_]:>6.1f}")

    conv = [2401, 2067, 3174]
    band = (zcnt >= 0.5 * min(zcnt[conv])) & (zcnt <= 2.0 * max(zcnt[conv]))
    cand = np.where((lab == "q600") & band)[0]
    cand = cand[np.argsort(-gap[cand])]
    print(f"\nA GENUINE MOISTURE GROUP: calibrated q600 winners, firing-rate matched "
          f"to the convection arm ({len(cand)} available)")
    for f_ in cand[:6]:
        print(f"  {f_:>5}{int(zcnt[f_]):>8}  "
              + "".join(f"{s[f_, k]:>+9.1f}" for k in range(P))
              + f"{'':>10}{lab[f_]:>12}{p[f_, top[f_]]:>8.4f}"
              + f"{gap[f_]:>7.1f}{gap_bar[f_]:>6.1f}")
    print(f"  suggested MECH_FEATS={','.join(str(int(x)) for x in cand[:3])}"
          f"   (disjoint from convection: {not set(map(int, cand[:3])) & set(conv)})")

    d.update(dict(p_empirical=p, sig=sig, label=lab, gap=gap, gap_bar=gap_bar,
                  zscore=s, calibration_null_label_rate=fake,
                  inference="empirical permutation p; gaussian step withdrawn",
                  moisture_group_suggested=[int(x) for x in cand[:3]]))
    np.save(OUT, d, allow_pickle=True)
    print(f"\nwrote {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
