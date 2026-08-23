"""CONCEPT causal graph v2 — the PURITY GATE run.

Implements docs/prereg/prereg_concept_graph_v2.md, frozen before this file existed.
Same doser, same one-6h-step readout, same ICs and same delta algebra as v1
(graphcast_sae/legacy/concept_cgraph.py); the node definition (purity gate + decorrelation, K=4,
from results/fs_cgv2_groups.npy) and the dose sweep are the only changes.

This script stores the FULL 4,096-vector activation delta for every (gamma, arm, IC, group).
Every pre-registered readout (RO-A pc1, RO-B zmean, RO-C sum) is then a post-hoc projection
of the same stored numbers in the scorer -- one GPU pass, no arm re-run, no forking path.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: supporting: interventional concept graph (not a paper figure)
Inputs: results/fs_cgv2_groups.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: out/cgv2_status.txt; results/fs_concept_cgraph_v2.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.concepts.concept_cgraph_v2
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

GAMMAS = [1.0, 0.5, 0.25]                # 1.0 is primary; the sweep answers the dose question
SET_A = ["2018-09-10", "2020-01-05", "2020-04-06", "2020-07-06", "2020-10-05", "2019-06-15"]
SET_B = ["2020-02-20", "2020-05-25", "2020-08-15", "2020-11-20"]
OUT = os.environ.get("CC_OUT", "results/fs_concept_cgraph_v2.npy")
STATUS = fc.ROOT / "out/cgv2_status.txt"

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=4)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", "12h"), **dataclasses.asdict(tc))

def main():
    G = np.load(fc.ROOT / "results/fs_cgv2_groups.npy", allow_pickle=True).item()
    names = list(G["concepts"])
    members = [G["groups"][n] for n in names]
    perm_groups = [list(map(int, g)) for g in G["perm_groups"]]
    C, K = len(names), int(G["K"])
    print("CONCEPT CAUSAL GRAPH v2 — prereg docs/prereg/prereg_concept_graph_v2.md (frozen d93b4ff)")
    print(f"  {C} concepts x K={K}   struck: {list(G['struck']) or 'none'}")
    for n, m in zip(names, members):
        print(f"    {n:<15} {m}")
    print(f"  NEG concept_perm: {C} random groups of {K} from the same {C*K} features (seed 0)")
    print(f"  gammas {GAMMAS}   ICs {len(SET_A)+len(SET_B)}\n", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def codes_sum(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return np.asarray(sae.codes(X).sum(0), np.float64)

    def step(inp, tgt1, frc1, patch):
        preds, acts = apply(inp, tgt1, frc1, patch)
        return numpyify(preds), acts

    ICS = SET_A + SET_B
    F = sae.n_features
    # D[g, arm, ic, r, :] = full 4096-vector code-sum delta of dosing group r
    D = np.zeros((len(GAMMAS), 2, len(ICS), C, F), np.float32)
    BASE = np.zeros((len(ICS), F), np.float32)
    t0 = time.time()
    nfwd = 0
    for w, ic in enumerate(ICS):
        inp, tgt, frc = build_io(ic, tc)
        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)
        tgt0 = tgt.isel(time=slice(0, 1)).assign_coords(time=tct)
        frc0 = frc.isel(time=slice(0, 1)).assign_coords(time=tct)
        tgt1 = tgt.isel(time=slice(1, 2)).assign_coords(time=tct)
        frc1 = frc.isel(time=slice(1, 2)).assign_coords(time=tct)

        p0b, _ = step(inp, tgt0, frc0, noop)
        in1b = rollout._get_next_inputs(inp, xr.merge([p0b, frc0])).assign_coords(
            time=inp.coords["time"])
        _, a1b = step(in1b, tgt1, frc1, noop)
        base = codes_sum(a1b)
        BASE[w] = base
        nfwd += 2

        for gi, gam in enumerate(GAMMAS):
            for ai, grps in enumerate((members, perm_groups)):
                for r, feats in enumerate(grps):
                    p0, _ = step(inp, tgt0, frc0,
                                 fc.coef_patch(sae, [int(f) for f in feats], gam))
                    in1 = rollout._get_next_inputs(inp, xr.merge([p0, frc0])).assign_coords(
                        time=inp.coords["time"])
                    _, a1 = step(in1, tgt1, frc1, noop)
                    D[gi, ai, w, r] = (codes_sum(a1) - base).astype(np.float32)
                    nfwd += 2
        el = (time.time() - t0) / 60
        msg = (f"  IC {ic} done ({w+1}/{len(ICS)})  {el:.1f}m  "
               f"eta {el/(w+1)*(len(ICS)-w-1):.0f}m  {nfwd} forwards")
        print(msg, flush=True); STATUS.write_text(msg + "\n")

    np.save(fc.ROOT / OUT, dict(
        D=D, BASE=BASE, names=names, members=[list(map(int, m)) for m in members],
        perm=perm_groups, gammas=GAMMAS, set_a=SET_A, set_b=SET_B, ics=ICS, K=K,
        struck=list(G["struck"]),
        axes="D[gamma, arm(0=real,1=perm), ic, source_group, feature]",
        prereg="docs/prereg/prereg_concept_graph_v2.md @ d93b4ff"), allow_pickle=True)
    print(f"\n-> {OUT}   (score with graphcast_sae/concepts/concept_cgraph_v2_score.py)")
    STATUS.write_text(f"DONE {len(ICS)} ICs  {(time.time()-t0)/60:.1f}m\n")

if __name__ == "__main__":
    main()
