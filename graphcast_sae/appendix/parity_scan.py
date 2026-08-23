"""Semidiurnal (12-h Nyquist) parity scan over the whole 4,096-feature dictionary.

WHY. f3319 and f3004 -- the endpoints of the only edge PCMCI+, LPCMCI and J-PCMCI+ all
agree on -- fire ~200x more at 06Z/18Z than at 00Z/12Z. A feature that is identically zero
on alternate 6-hourly steps can only correlate with anything at EVEN tau; tau=1 and tau=3
are structurally forbidden. The estimators' agreement on tau=2 is then parity, not physics.

THE INDEX (stated before any count is looked at).
Let S_h be the mean, over the IID windows whose UTC hour is h, of a feature's global SAE
code sum over the 40,962 M6 mesh nodes (results/fs_cgv2_actseries.npy, 160 real ERA5
windows, 2016-2020).

    odd  = S_06 + S_18          (the 06Z/18Z phase)
    even = S_00 + S_12          (the 00Z/12Z phase)
    P    = (odd - even) / (odd + even)          in [-1, +1]

P = +1  fires ONLY at 06Z/18Z.  P = -1  fires ONLY at 00Z/12Z.  P = 0  flat.
P is exactly the normalised amplitude of the 12-hour harmonic, i.e. the NYQUIST component
of a 6-hourly series (period = 2*dt). For four equally spaced samples the 12-h Fourier
coefficient is (S_00 - S_06 + S_12 - S_18)/4, so |P| is that coefficient normalised by the
mean -- the alternating (off, on, off, on) mode and nothing else.

For contrast we also compute the 24-hour (diurnal) index, which is what label_banded.py's
`diurnal` statistic measures:

    D = sqrt((S_00 - S_12)^2 + (S_06 - S_18)^2) / (S_00 + S_06 + S_12 + S_18)

D and P are orthogonal harmonics: a pure (off, on, off, on) feature has D = 0 exactly.
That is why no existing diagnostic in this repo could have seen this.

A second, amplitude-free index (an on/off switch rather than a modulated amplitude):

    Q = (f_odd - f_even) / (f_odd + f_even),   f_h = frac of hour-h windows with sum > 0

CALIBRATION (the control-must-be-able-to-fail rule) AND WHY P ALONE IS NOT A BAR.
P is heteroskedastic across the dictionary: a feature that fires in ONE window has |P| = 1
by construction. A dictionary-wide max-|P| null is therefore a point mass at 1.000 -- an
unattainable bar, the mirror image of the SPD alignment failure. Two repairs, both applied:

  SUPPORT GATE. A feature must fire (global sum > 0) in at least N_FIRE_MIN of the 160
  windows to be scored at all. Features below it are reported separately as unscorable.

  STUDENTISED INDEX.  z = P / sd_perm(P), where sd_perm(P) is that feature's OWN standard
  deviation of P over NPERM permutations of the UTC-hour labels across the 160 windows.
  z is comparable across the dictionary, so max|z| over features has a proper distribution.

  (i)   the null VARIES       -- the per-draw max|z| spread is printed, and it is not a
                                point mass.
  (ii)  the bar is ATTAINABLE -- the 95th percentile of the null's max|z| is finite and is
                                reached by null draws.
  (iii) a negative control FAILS -- the frozen convection group [2401, 2067, 3174] and the
        TC readout 3243 are known-normal and must land inside the null.

Paper: Appendix app:parity (Tables tab:parity-census, tab:parity-refit)
Inputs: results/fs_cgv2_actseries.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/parity_scan.json
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.parity_scan
"""
import json
import pathlib

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
SER = ROOT / "results/fs_cgv2_actseries.npy"
OUT = ROOT / "results/parity_scan.json"

CONV = [2401, 2067, 3174]          # frozen convection group -- negative control
TC = 3243                          # TC readout             -- negative control
POS = [3319, 3004, 3357]           # the edge features
HOURS = (0, 6, 12, 18)
NPERM = 4000
BATCH = 500
N_FIRE_MIN = 16                    # >= 10% of the 160 windows, fixed before looking
SEED = 0

def parity(M):
    """M (..., 4, F) rows 00,06,12,18 -> P (..., F)."""
    odd = M[..., 1, :] + M[..., 3, :]
    even = M[..., 0, :] + M[..., 2, :]
    tot = odd + even
    return np.divide(odd - even, tot, out=np.zeros_like(tot), where=tot > 0)

def main():
    d = np.load(SER, allow_pickle=True).item()
    S = np.asarray(d["series"], float)                       # (W, F) global code sums
    starts = np.asarray(d["starts"])
    W, F = S.shape
    # ---- the data-gate rule: data gate
    assert S.shape[1] == 4096, S.shape
    assert np.isfinite(S).all(), "non-finite code sums"
    assert (S < 0).sum() == 0, "negative code sums"
    assert S.sum(1).min() > 0, "an all-zero window"
    hrs = np.array([int(str(x)[11:13]) for x in starts])
    assert set(hrs.tolist()) <= set(HOURS), sorted(set(hrs.tolist()))
    nh = {h: int((hrs == h).sum()) for h in HOURS}
    ss = sorted(str(x) for x in starts)
    print(f"[gate] {W} windows x {F} features; finite=OK; windows per UTC hour {nh}")
    print(f"[gate] window span {ss[0][:10]} .. {ss[-1][:10]}")

    # hour-mean operator G (4, W): row h averages the windows at hour h
    G = np.zeros((4, W))
    for i, h in enumerate(HOURS):
        m = hrs == h
        G[i, m] = 1.0 / m.sum()

    M = G @ S                                                # (4, F)
    P = parity(M)
    tot = M.sum(0)
    D = np.divide(np.sqrt((M[0] - M[2]) ** 2 + (M[1] - M[3]) ** 2), tot,
                  out=np.zeros_like(tot), where=tot > 0)
    Fon = np.stack([(S[hrs == h] > 0).mean(0) for h in HOURS])
    qt = Fon.sum(0)
    Q = np.divide((Fon[1] + Fon[3]) - (Fon[0] + Fon[2]), qt,
                  out=np.zeros_like(qt), where=qt > 0)
    nfire = (S > 0).sum(0)
    nfire_odd = ((S > 0) & ((hrs[:, None] == 6) | (hrs[:, None] == 18))).sum(0)
    nfire_even = ((S > 0) & ((hrs[:, None] == 0) | (hrs[:, None] == 12))).sum(0)

    alive = tot > 0
    scorable = nfire >= N_FIRE_MIN
    print(f"[gate] alive (nonzero total) = {int(alive.sum())}/{F}; "
          f"dead = {int((~alive).sum())}")
    print(f"[gate] SUPPORT GATE n_fire >= {N_FIRE_MIN}/160: scorable = "
          f"{int(scorable.sum())}; unscorable (too sparse for P to mean anything) = "
          f"{int((alive & ~scorable).sum())}")
    print(f"[gate] n_fire over scorable features: min {nfire[scorable].min()} "
          f"med {int(np.median(nfire[scorable]))} max {nfire[scorable].max()}")

    # ---- permutation null on the UTC-hour labels
    rng = np.random.default_rng(SEED)
    s1 = np.zeros(F); s2 = np.zeros(F)
    ge = np.zeros(F)
    Pnull_chunks = []
    done = 0
    while done < NPERM:
        b = min(BATCH, NPERM - done)
        Gb = np.zeros((b, 4, W))
        for k in range(b):
            hp = rng.permutation(hrs)
            for i, h in enumerate(HOURS):
                m = hp == h
                Gb[k, i, m] = 1.0 / m.sum()
        Mb = Gb.reshape(b * 4, W) @ S                        # (b*4, F)
        Pb = parity(Mb.reshape(b, 4, F))                     # (b, F)
        s1 += Pb.sum(0); s2 += (Pb ** 2).sum(0)
        ge += (np.abs(Pb) >= np.abs(P)[None, :]).sum(0)
        Pnull_chunks.append(Pb[:, scorable])
        done += b
    mu = s1 / NPERM
    sd = np.sqrt(np.maximum(s2 / NPERM - mu ** 2, 0.0))
    pval = (ge + 1) / (NPERM + 1)
    z = np.divide(P - mu, sd, out=np.zeros_like(P), where=sd > 1e-12)
    Pn = np.concatenate(Pnull_chunks, 0)                     # (NPERM, n_scorable)
    sc = np.where(scorable)[0]
    zn = (Pn - mu[sc][None, :]) / np.maximum(sd[sc][None, :], 1e-12)
    maxz = np.abs(zn).max(1)
    zbar = float(np.percentile(maxz, 95))
    print(f"\n[null] {NPERM} permutations of the UTC-hour labels, scored on the "
          f"{len(sc)} scorable features.")
    print(f"[null] (i) the null VARIES: per-draw max|z| p5 {np.percentile(maxz,5):.2f} "
          f"med {np.median(maxz):.2f}  p95 {zbar:.2f}  max {maxz.max():.2f}")
    print(f"[null]     per-draw |P| p99 over the dictionary: "
          f"med {np.median(np.percentile(np.abs(Pn),99,axis=1)):.3f}")
    print(f"[null] (ii) the bar is ATTAINABLE: FWER bar |z| >= {zbar:.2f} is reached by "
          f"{int((maxz>=zbar).sum())}/{NPERM} null draws, by construction 5%.")
    print(f"[null]     |P| that corresponds to the bar, at the median scorable sd "
          f"({np.median(sd[sc]):.3f}): |P| ~ {zbar*np.median(sd[sc]):.3f}")
    n_hit = int((np.abs(z[scorable]) >= zbar).sum())
    print(f"[null] observed: {n_hit}/{len(sc)} scorable features exceed the FWER bar "
          f"({100*n_hit/len(sc):.1f}%), vs 5% of DRAWS expected to produce >=1 under H0.")

    # ---- (i) the observed distribution
    Ps = P[scorable]
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 99.5, 100]
    print("\n[dist] |P| percentiles over the scorable features:")
    print("       " + "  ".join(f"p{q}={np.percentile(np.abs(Ps), q):.3f}" for q in qs))
    print("[dist] signed P percentiles:")
    print("       " + "  ".join(f"p{q}={np.percentile(Ps, q):+.3f}" for q in qs))
    edges = np.array([-1.0, -0.99, -0.95, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
                      -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
                      1.0001])
    hist, _ = np.histogram(Ps, bins=edges)
    print("\n[dist] histogram of signed P (scorable features) -- bimodal family or "
          "continuum?")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"       [{lo:+.2f},{hi:+.2f})  {c:>5d}  "
              + "#" * int(np.ceil(60 * c / max(hist.max(), 1))))
    srt = np.sort(np.abs(Ps))
    tail = srt[srt > 0.5]
    if len(tail) > 1:
        g = np.diff(tail)
        k = int(np.argmax(g))
        print(f"\n[gap]  above |P| = 0.5 there are {len(tail)} scorable features; the "
              f"largest gap in sorted |P| is {g[k]:.4f}, between {tail[k]:.4f} and "
              f"{tail[k+1]:.4f}.")
        print("[gap]  A clean bimodal family would show one wide gap separating it from "
              "the bulk. Judge the number above against the bin widths in the histogram.")

    # ---- thresholds
    print("\n[count] scorable features by threshold (positive = 06Z/18Z-gated, "
          "negative = 00Z/12Z-gated):")
    print(f"        {'thr on P':>9}{'P>=thr':>9}{'P<=-thr':>9}    "
          f"{'|z|>=bar too':>13}")
    rows = {}
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        a = int(((P >= thr) & scorable).sum())
        b = int(((P <= -thr) & scorable).sum())
        c = int(((np.abs(P) >= thr) & scorable & (np.abs(z) >= zbar)).sum())
        rows[f"{thr:.2f}"] = dict(pos=a, neg=b, and_z=c)
        print(f"        {thr:>9.2f}{a:>9d}{b:>9d}    {c:>13d}")
    hard = scorable & (np.abs(P) >= 0.9) & (np.abs(z) >= zbar)
    hard_pos = scorable & (P >= 0.9) & (z >= zbar)
    hard_neg = scorable & (P <= -0.9) & (z <= -zbar)
    print(f"\n[FAMILY] chosen threshold: |P| >= 0.90 AND |z| >= FWER bar {zbar:.2f} AND "
          f"n_fire >= {N_FIRE_MIN}.")
    print(f"[FAMILY] |P| >= 0.90 means >= 95% of the feature's total activation sits on one "
          f"parity -- a >= 19:1 on/off ratio, not a modulation.")
    print(f"[FAMILY] total {int(hard.sum())}:  {int(hard_pos.sum())} gated to 06Z/18Z, "
          f"{int(hard_neg.sum())} gated to 00Z/12Z  "
          f"({100*hard.sum()/len(sc):.1f}% of the {len(sc)} scorable features)")
    for lab, mk in (("06Z/18Z", hard_pos), ("00Z/12Z", hard_neg)):
        idx = np.where(mk)[0]
        print(f"[FAMILY] {lab}-gated, n={len(idx)}: {sorted(idx.tolist())}")

    # ---- (iii) controls
    print(f"\n[calib] controls (FWER bar |z| >= {zbar:.2f}):")
    print(f"        {'feat':>6}{'S00':>9}{'S06':>9}{'S12':>9}{'S18':>9}{'nfire':>7}"
          f"{'P':>8}{'z':>9}{'Q':>8}{'D24':>7}{'p_perm':>9}  verdict")

    def show(j, tag):
        v = ("PARITY-GATED" if (np.abs(P[j]) >= 0.9 and abs(z[j]) >= zbar and scorable[j])
             else ("above bar, not gated" if abs(z[j]) >= zbar else "INSIDE NULL"))
        print(f"        {j:>6}{M[0,j]:>9.1f}{M[1,j]:>9.1f}{M[2,j]:>9.1f}{M[3,j]:>9.1f}"
              f"{nfire[j]:>7d}{P[j]:>+8.3f}{z[j]:>+9.2f}{Q[j]:>+8.3f}{D[j]:>7.3f}"
              f"{pval[j]:>9.4f}  {v}  <- {tag}")

    for j in POS:
        show(j, "edge feature")
    for j in CONV:
        show(j, "frozen convection (NEG CONTROL, must fail)")
    show(TC, "TC readout (NEG CONTROL, must fail)")
    samp = np.random.default_rng(1).choice(sc, 200, replace=False)
    print(f"        random 200 scorable: |P| med {np.median(np.abs(P[samp])):.3f} "
          f"p90 {np.percentile(np.abs(P[samp]),90):.3f}; |z| med "
          f"{np.median(np.abs(z[samp])):.2f}; {int((np.abs(z[samp])>=zbar).sum())}/200 "
          f"over the bar, {int(((np.abs(P[samp])>=0.9)&(np.abs(z[samp])>=zbar)).sum())}/200 "
          f"in the family")

    # ---- diurnal blindness
    fam = np.where(hard)[0]
    print(f"\n[blind] the {len(fam)} family members' 24-h index D: med "
          f"{np.median(D[fam]):.3f} p90 {np.percentile(D[fam],90):.3f}; scorable-dictionary "
          f"D med {np.median(D[scorable]):.3f} p99 "
          f"{np.percentile(D[scorable],99):.3f}")
    print("[blind] D is the 24-h harmonic (label_banded.py's `diurnal`); it is exactly "
          "orthogonal to P, so no existing diagnostic here could see this family.")

    json.dump(dict(
        index_definition="P = (S06+S18-S00-S12)/(S00+S06+S12+S18); S_h = mean global SAE "
                         "code sum over the IID windows at UTC hour h. = normalised 12-h "
                         "(Nyquist) harmonic. z = (P-mu_perm)/sd_perm.",
        n_windows=W, n_per_hour=nh, n_perm=NPERM, n_fire_min=N_FIRE_MIN,
        n_alive=int(alive.sum()), n_scorable=int(scorable.sum()),
        fwer_bar_z=zbar,
        null_maxz=dict(p5=float(np.percentile(maxz, 5)), med=float(np.median(maxz)),
                       p95=zbar, max=float(maxz.max())),
        P=P.tolist(), z=z.tolist(), Q=Q.tolist(), D24=D.tolist(), p_perm=pval.tolist(),
        n_fire=nfire.tolist(), n_fire_odd=nfire_odd.tolist(),
        n_fire_even=nfire_even.tolist(), scorable=scorable.tolist(),
        S_by_hour={f"{h:02d}Z": M[i].tolist() for i, h in enumerate(HOURS)},
        counts=rows,
        family=sorted(int(x) for x in np.where(hard)[0]),
        family_pos=sorted(int(x) for x in np.where(hard_pos)[0]),
        family_neg=sorted(int(x) for x in np.where(hard_neg)[0]),
    ), open(OUT, "w"))
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
