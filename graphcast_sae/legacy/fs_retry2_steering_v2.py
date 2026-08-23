"""Flagship retry #2 v2 — steering with a REPAIRED readout.

v1 (`fs_retry2_steering.py`) is void: its primary lead was s=0, where the patch is applied
at layer 8 and the readout reads layer 8, so `dA` IS the injected delta (linear in alpha to
1.2e-07, leakage bit-identical at every dose). See `internal note 'flagship_steering_s0_defect' (not shipped)`.

v2 changes, all frozen in `docs/prereg/prereg_flagship_steering_v2.md` before this file existed:

  1. primary lead s=1 (12 h, patch OFF); s=0 never scored; s=2 (18 h) secondary.
  2. FEATURE-SPACE readout. Communities are a hard partition of SAE features, so
     rel[c] = ||dF[:, members(c)]|| / ||F_base[:, members(c)]|| has no basis overlap,
     no geometric floor, and no rank-1 assumption -- repairing v1 defects 2 and 3 at once.
  3. `recon` arm as the power reference (the noise floor is identically 0 on a
     bit-deterministic CPU forward, so it cannot serve as one).
  4. 2 mass-matched random-feature negative controls -- the both-sides calibration v1 lacked.

The v1 mode-basis metrics are still computed and stored alongside, for continuity.

Run (~3.5-4 h on CPU):

Paper: not in the paper; kept for provenance only
Inputs: results/graphcast_sae_steering.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/graphcast_sae_steering_v2.npy (--out); status out/fs_steering_v2_status.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.fs_retry2_steering_v2
"""
import argparse
import time

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import FS_CATALOG, FS_MODES

ALPHAS = np.array([-1.0, -0.5, 0.5, 1.0, 2.0])
ENC_BLOCK = 8192                      # tokens per encode block (keeps peak RSS small)

def community_members(modes):
    """Disjoint feature-index sets, one per non-trivial community."""
    mem = modes["members"]
    return [np.asarray(m, np.int64) for m in mem]

def encode_members(A, sae_np, allf):
    """A (N, B, 512) raw -> codes restricted to `allf`, shape (N*B, len(allf)) float32.

    The encoder is NON-LINEAR (per-token normalization, then ReLU + TopK), so
    encode(x - y) != encode(x) - encode(y) -- indeed encode(0) != 0, because the
    normalized zero token still hits `-b_pre @ W_enc.T`. The prereg specifies the
    difference OF THE CODES, so both activations must be encoded separately and
    subtracted afterwards. Never encode a difference.
    """
    X = A.reshape(-1, A.shape[-1])
    out = np.empty((X.shape[0], allf.size), np.float32)
    for i in range(0, X.shape[0], ENC_BLOCK):
        z, _ = fc.encode_np(X[i:i + ENC_BLOCK], sae_np)
        out[i:i + ENC_BLOCK] = z[:, allf]
    return out

def agg_sq(Z, B, slices):
    """(N*B, M) codes -> squared Frobenius norm per (community, window).

    Row i of the reshaped (N,B,D) activation is node i//B, window i%B.
    `slices` are contiguous because `allf` concatenates the member sets in order.
    """
    sq = np.zeros((len(slices), B), np.float64)
    b_idx = np.arange(Z.shape[0]) % B
    for b in range(B):
        Zb = Z[b_idx == b]
        for c, (lo, hi) in enumerate(slices):
            sq[c, b] = float((np.float64(Zb[:, lo:hi]) ** 2).sum())
    return sq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwin", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--nctrl", type=int, default=2)
    # Control-set extension (prereg v2 §4b amendment, disclosed as post-hoc). Runs ONLY
    # additional negative controls -- no handle arm is re-run and no handle number changes.
    # Adding controls can only weaken or neutralize the PASS, never strengthen it.
    ap.add_argument("--ctrl-only", type=int, default=0,
                    help="run only N extra controls, excluding those already drawn")
    ap.add_argument("--ctrl-seed", type=int, default=0)
    ap.add_argument("--cat", default=str(FS_CATALOG))
    ap.add_argument("--modes", default=str(FS_MODES))
    ap.add_argument("--out", default="results/graphcast_sae_steering_v2.npy")
    ap.add_argument("--status", default="out/fs_steering_v2_status.txt")
    args = ap.parse_args()

    cat = np.load(args.cat, allow_pickle=True)
    modes = np.load(args.modes, allow_pickle=True)
    members = community_members(modes)
    C = len(members)
    allf = np.concatenate(members)
    assert allf.size == np.unique(allf).size, "communities must be a hard partition"
    bnds = np.cumsum([0] + [len(m) for m in members])
    slices = list(zip(bnds[:-1], bnds[1:]))
    W_c = np.asarray(modes["W"], np.float64)
    Q_c = np.asarray(modes["Q"], np.float64)
    sig_c = np.clip(np.asarray(modes["sigma"], np.float64), 1e-12, None)

    # handles: v1's selection, reused verbatim (prereg v2 §2 -- NOT re-picked)
    v1 = np.load(fc.ROOT / "results/graphcast_sae_steering.npy", allow_pickle=True).item()
    handles = [(int(f), int(c), float(r)) for f, c, r in v1["handles"]]

    # negative controls: random features mass-matched to the handle band (prereg v2 §2)
    mass = cat["mass"]
    hmass = np.array([mass[f] for f, _, _ in handles])
    lm = np.log(np.clip(mass, 1e-12, None))
    band = np.where((mass > 0) &
                    (lm >= np.log(hmass.min()) - np.log(2)) &
                    (lm <= np.log(hmass.max()) + np.log(2)))[0]
    used = {f for f, _, _ in handles}
    band = np.array([b for b in band if b not in used])
    if args.ctrl_only:
        # exclude the controls the frozen n=2 run already drew, then draw N fresh ones
        prev = {int(a) for a, _ in v1.get("controls", [])} if "controls" in v1 else set()
        prev |= {int(a) for a, _ in
                 np.load(fc.ROOT / "results/graphcast_sae_steering_v2.npy",
                         allow_pickle=True).item()["controls"]}
        band = np.array([b for b in band if b not in prev])
        rng = np.random.default_rng(args.ctrl_seed)
        ctrl_f = rng.choice(band, args.ctrl_only, replace=False)
        print(f"CONTROL-EXTENSION run: {args.ctrl_only} new controls, excluding {sorted(prev)}")
    else:
        rng = np.random.default_rng(0)
        ctrl_f = rng.choice(band, args.nctrl, replace=False)
    ctrls = [(int(f), handles[i % len(handles)][1]) for i, f in enumerate(ctrl_f)]

    print(f"{C} communities, sizes {[len(m) for m in members]}")
    for f, c, r in handles:
        print(f"  handle  feature {f:5d} -> community {c:2d}  |r|={abs(r):.3f}")
    for f, c in ctrls:
        print(f"  CONTROL feature {f:5d} -> community {c:2d}  (mass-matched, alpha=+1)")

    sae = fc.SAEJax()
    sae_np = fc.sae_numpy()
    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=sae)
    apply = fc.make_apply(params, rf)
    noop = fc.noop_patch(sae)

    arms = [("noise", -1, -1, 0.0, noop), ("recon", -1, -1, 0.0, fc.recon_patch(sae))]
    if not args.ctrl_only:
        for f, c, _ in handles:
            for a in ALPHAS:
                arms.append((f"f{f}_c{c}_a{a:+g}", f, c, float(a),
                             fc.coef_patch(sae, [f], float(a))))
    for f, c in ctrls:
        arms.append((f"ctrl{f}_c{c}_a+1", f, c, 1.0, fc.coef_patch(sae, [f], 1.0)))
    names = [a[0] for a in arms]
    print(f"{len(arms)} arms x {args.nwin} windows x {args.nsteps} steps", flush=True)

    starts = fc.seasonal_starts(args.nwin * 6, 2021)[3::6][:args.nwin]
    nframes = args.nsteps + 2
    REL = np.full((len(arms), args.nwin, args.nsteps, C), np.nan)   # feature-space (primary)
    RMODE = np.full((len(arms), args.nwin, args.nsteps, C), np.nan)  # v1 mode basis (companion)
    t0 = time.time()

    for ci in range(0, args.nwin, args.chunk):
        cs = starts[ci:ci + args.chunk]
        nb = len(cs)
        blocks = [fc.load_block(c, nframes) for c in cs]
        base_inputs = [fc.build_batch_inputs(blocks, s, tc) for s in range(args.nsteps)]

        base_pred, base_A, base_Z, base_sq = [], [], [], []
        for s in range(args.nsteps):
            inp, tgt, frc = base_inputs[s]
            p, A = apply(inp, tgt * np.nan, frc, noop)
            base_pred.append({v: np.asarray(p[v].data, np.float64) for v in fc.PROG})
            A = np.asarray(A, np.float32).reshape(fc.N_MESH, nb, fc.D_IN)
            base_A.append(A)
            Z = encode_members(A, sae_np, allf)     # cached: base is encoded once per step
            base_Z.append(Z)
            base_sq.append(agg_sq(Z, nb, slices))
            if s == 0:
                pdims = {v: p[v].dims for v in fc.PROG}
        idims = {v: base_inputs[0][0][v].dims for v in fc.PROG}
        roll_perm = {v: [[d for d in pdims[v] if d != "time"].index(d)
                         for d in [d for d in idims[v] if d != "time"]] for v in fc.PROG}
        print(f"  baselines done ({(time.time()-t0)/60:.1f}m)", flush=True)

        for ai, (name, f, c, alpha, patch) in enumerate(arms):
            delta = {v: np.zeros_like(np.asarray(base_inputs[0][0][v].values, np.float64))
                     for v in fc.PROG}
            for s in range(args.nsteps):
                inp, tgt, frc = base_inputs[s]
                pert = inp.copy(deep=False)
                if s > 0:
                    for v in fc.PROG:
                        pert[v] = inp[v] + delta[v]
                # impulse at step 0 only; the patch is OFF for every later step, which is
                # what makes s>=1 a genuine dynamical response (prereg v2 §0 repair 1).
                p, A = apply(pert, tgt * np.nan, frc, patch if s == 0 else noop)
                A = np.asarray(A, np.float32).reshape(fc.N_MESH, nb, fc.D_IN)

                # ---- primary: feature-space relative code change, hard partition ----
                # encode the perturbed activation, then subtract the CACHED base CODES.
                dsq = agg_sq(encode_members(A, sae_np, allf) - base_Z[s], nb, slices)
                REL[ai, ci:ci + nb, s, :] = np.sqrt(
                    dsq / np.clip(base_sq[s], 1e-30, None)).T

                # ---- companion: v1 mode-basis readout, kept for continuity ----
                dA = np.asarray(A, np.float64) - base_A[s]
                RMODE[ai, ci:ci + nb, s, :] = (
                    np.einsum("cn,nbd,cd->bc", W_c, dA, Q_c) / sig_c[None, :])

                dy = {v: np.asarray(p[v].data, np.float64) - base_pred[s][v]
                      for v in fc.PROG}
                for v in fc.PROG:
                    tpos = pdims[v].index("time")
                    dy0 = np.take(dy[v], 0, axis=tpos)
                    delta[v][:, 0] = delta[v][:, 1]
                    delta[v][:, 1] = np.transpose(dy0, roll_perm[v])
            msg = (f"windows {ci+nb}/{args.nwin} arm {ai+1}/{len(arms)} "
                   f"({name}) {(time.time()-t0)/60:.1f}m")
            print("  " + msg, flush=True)
            (fc.ROOT / args.status).write_text(msg + "\n")
            np.save(fc.ROOT / args.out, dict(
                names=names, REL=REL, RMODE=RMODE, alphas=ALPHAS,
                handles=handles, controls=[(int(a), int(b)) for a, b in ctrls],
                comm_sizes=[int(len(m)) for m in members],
                nwin=args.nwin, nsteps=args.nsteps, n_comm=int(C),
                starts=[str(s) for s in starts],
                prereg="docs/prereg/prereg_flagship_steering_v2.md"), allow_pickle=True)
    print(f"saved -> {args.out} ({(time.time()-t0)/60:.1f}m)", flush=True)

if __name__ == "__main__":
    main()
