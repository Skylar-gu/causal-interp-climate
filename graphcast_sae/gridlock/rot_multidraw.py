"""Multi-draw, PER-MEMBER norm-matched controls for the rotation-positional tiers.

Why this exists
---------------
The 2026-08-19 tier run (`results/fs_rottiers_rmse.npy`, `fs_rottiers2_rmse.npy`) used ONE
control draw per tier, matched on TOTAL deleted norm to 1.00x. Reading those results back:

    tier    treatment      control
    rot3      +2.7 m        +0.1 m
    rot6      +4.1          +0.6
    rot10     +7.1         +29.2      <- control 4x WORSE than treatment
    rot15     +4.1         +11.7
    rot19     +3.8          +4.5
    rot26     +3.7         +16.3
    rot35     +3.5          +1.6
    rot50     +3.7         +17.8
    rot86     +3.7          +3.7

The control draws scatter from +0.1 m to +29.2 m at FIXED total deleted norm -- roughly eight
times the treatment effect itself. A single draw cannot referee an effect that much smaller
than the referee's own spread, so none of those nine rows is currently interpretable.

Two fixes, both needed:

1. **Many draws per tier**, so the treatment is scored against a null DISTRIBUTION rather than
   against one sample. Guardrail #9 leg (i) is already satisfied loudly -- the null varies --
   but legs (ii) and (iii) need the distribution.

2. **Match per MEMBER, not just the total.** The existing draws match the sum and nothing else,
   and the composition is wildly different:

       rot3       deleted norm  [0.176, 0.118, 0.138]   sum 0.431
       ctrl_rot3                [0.006, 0.001, 0.430]   sum 0.436

   The control is one enormous feature plus two crumbs. Deleting one huge feature is not the
   same perturbation as deleting three medium ones, and since the pool's deleted norm is
   heavily skewed (median 0.0035, max 3.22), whichever single big feature a draw happens to
   land on dominates it. That is the most likely source of the +0.1..+29.2 scatter.

   Here each treatment member is matched to its OWN control feature by NEAREST deleted norm --
   never argmax inside a tolerance band, which is the bug that pushed `core_control`'s draws to
   +35% on 9 of 9 picks and handed the control a head start in the comparison it refereed.

Eligible pool: live features with rotation-positional score < 0.15 (clearly not positional) and
nonzero deleted norm, minus every treatment member and every feature already used by another
draw of the same tier, so the K draws are disjoint.

Writes a GR_GROUPS json for `graphcast_sae/gridlock/global_rmse_ablate.py`. Produces no causal number.

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: results/fs_deleted_norm.npy (not shipped, see docs/REPRODUCE.md); results/fs_rotation_all.npy (not shipped, see docs/REPRODUCE.md)
Outputs: --out json of the per-member control draws (default /tmp/rot_multidraw.json)
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.gridlock.rot_multidraw --k 8 --out /tmp/rot_multidraw.json
"""
import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
POS_MAX = 0.15          # "clearly weather-like" on the direct rotation measurement
TOL = 0.25              # per-member deleted-norm tolerance, reported not enforced silently

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="control draws per tier")
    ap.add_argument("--tiers", default="rot3,rot6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default="/tmp/rot_tiers.json")
    ap.add_argument("--out", default="/tmp/rot_multidraw.json")
    args = ap.parse_args()

    rot = np.load(ROOT / "results/fs_rotation_all.npy", allow_pickle=True).item()
    dn = np.load(ROOT / "results/fs_deleted_norm.npy")
    src = json.load(open(args.src))

    assert dn.shape == rot["positional"].shape == (4096,), "shape gate"
    assert np.isfinite(dn).all(), "deleted norm has non-finite entries"

    live = rot["live"].astype(bool)
    eligible = live & (rot["positional"] < POS_MAX) & (dn > 0)
    print(f"eligible control pool: {int(eligible.sum())} of 4096 "
          f"(live, positional < {POS_MAX}, deleted norm > 0)")

    rng = np.random.default_rng(args.seed)
    groups, report = {}, []
    for tier in args.tiers.split(","):
        members = src[tier]
        groups[tier] = members
        used = set(members)
        tgt_tot = float(dn[members].sum())
        print(f"\n{tier}: n={len(members)} total deleted norm {tgt_tot:.4f}")
        print(f"  member norms {np.round(dn[members], 4).tolist()}")
        for d in range(args.k):
            picks = []
            # Draw the members in descending norm order: the big ones are the scarce
            # constraint, so satisfying them first avoids being left with no match for them.
            for f in sorted(members, key=lambda m: -dn[m]):
                cand = np.where(eligible)[0]
                cand = np.array([c for c in cand if c not in used])
                if cand.size == 0:
                    raise RuntimeError(f"{tier} draw {d}: pool exhausted")
                # nearest-to-target, with a small random jitter so the K draws differ
                # the jitter perturbs the RANKING, never the tolerance test below.
                err = np.abs(dn[cand] - dn[f]) / max(dn[f], 1e-12)
                order = np.argsort(err + rng.uniform(0, 1e-3, err.size))
                pick = int(cand[order[0]])
                picks.append(pick)
                used.add(pick)
            name = f"ctrl_{tier}_d{d}"
            groups[name] = picks
            got = float(dn[picks].sum())
            worst = max(abs(dn[p] - dn[f]) / max(dn[f], 1e-12)
                        for p, f in zip(picks, sorted(members, key=lambda m: -dn[m])))
            report.append(dict(tier=tier, draw=d, ratio=got / tgt_tot, worst_member=worst))
            flag = "" if worst <= TOL else "  <-- OVER TOLERANCE, reported not hidden"
            print(f"  {name}: total {got:.4f} ({got/tgt_tot:.3f}x)  "
                  f"worst member {worst:.2f}{flag}")

    json.dump(groups, open(args.out, "w"))
    arms = ["baseline"]
    for tier in args.tiers.split(","):
        arms.append(tier)
        arms += [f"ctrl_{tier}_d{d}" for d in range(args.k)]
    arms.append("floor")
    print(f"\nwrote {args.out}: {len(groups)} groups, {len(arms)} arms")
    print("GR_ARMS=" + ",".join(arms))
    over = [r for r in report if r["worst_member"] > TOL]
    print(f"draws with a member outside +/-{TOL:.0%}: {len(over)} of {len(report)}")

if __name__ == "__main__":
    main()
