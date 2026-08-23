"""Extract processor layer-8 mesh-node embeddings over teacher-forced steps.

Generalizes `phase0_smoke.py` from a single step into an activation *dump* that
the SAE (G2, design §5) trains on. For each teacher-forced window in a sample
ERA5 file it runs one forward pass and captures the processor node embeddings at
one or more message-passing steps (default step 8 — MacMillan & Ouellette's
layer-8, arXiv:2512.24440), then stacks them into an (n_tokens, 512) matrix.

Each teacher-forced step yields one embedding per mesh node (graphcast_small:
10242 nodes), so the SAE's "tokens" are per-node activation vectors — exactly
the paper's regime (single forward pass, node embeddings, not autoregressive).

Run (JAX/graphcast env):

Paper: graphcast_small lane; not in the paper
Inputs: data/sample/era5_2022-01-01_1deg_13lev_steps04.nc (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/layer8_acts_smoke.npy (--out)
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.mini_extract_layer8 --sample data/sample/era5_2022-01-01_1deg_13lev_steps04.nc --steps 8 --out results/layer8_acts_smoke.npy
"""
import argparse
import dataclasses
import functools
import pathlib

import haiku as hk
import jax
import numpy as np
import xarray as xr

from graphcast import (casting, checkpoint, data_utils,
                       graphcast as gc, normalization)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
N_MESH_GNN_STEPS = 16          # graphcast_small processor depth
INPUT_WINDOW = 3               # 2 input steps + 1 target = one teacher-forced step

def load_model():
    with open(ASSETS / "params/graphcast_small.npz", "rb") as f:
        ckpt = checkpoint.load(f, gc.CheckPoint)
    stats = {n: xr.load_dataset(ASSETS / f"stats/{n}.nc").compute() for n in
             ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}
    return ckpt.params, ckpt.model_config, ckpt.task_config, stats

def build_apply(model_config, task_config, stats, hook_steps):
    """Return (apply_fn, captured_dict). captured[step] <- (n_mesh, 512) per call."""
    def wrapped():
        predictor = gc.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        return normalization.InputsAndResiduals(
            predictor, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"])

    captured = {"count": 0, "acts": {}}

    def interceptor(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if (context.method_name == "_process_step"
                and context.module.module_name.split("/")[-1] == "mesh_gnn"):
            captured["count"] += 1
            if captured["count"] in hook_steps:
                captured["acts"][captured["count"]] = np.asarray(
                    out.nodes["mesh_nodes"].features)   # (n_mesh, 1, 512)
        return out

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        with hk.intercept_methods(interceptor):
            return wrapped()(inputs, targets_template=targets_template,
                             forcings=forcings)
    return run_forward, captured

def rebased_window(sample, i):
    """3-step slice starting at index i, time coord reset to [0,6,12]h.

    Re-basing the timedelta coordinate lets data_utils treat every window
    identically (inputs at -6h,0h; target at +6h), so a single sample file
    yields (T-2) independent teacher-forced steps."""
    win = sample.isel(time=slice(i, i + INPUT_WINDOW)).copy()
    win = win.assign_coords(
        time=np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]"))
    return win

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample/era5_2022-01-01_1deg_13lev_steps04.nc")
    ap.add_argument("--steps", default="8",
                    help="comma-separated processor steps to capture, e.g. 4,8,12")
    ap.add_argument("--out", default="results/layer8_acts_smoke.npy")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all available")
    args = ap.parse_args()

    hook_steps = sorted(int(s) for s in args.steps.split(","))
    params, model_config, task_config, stats = load_model()
    print("loaded graphcast_small; capturing processor steps", hook_steps)

    sample = xr.load_dataset(ROOT / args.sample, decode_timedelta=True).compute()
    n_time = sample.sizes["time"]
    n_windows = n_time - (INPUT_WINDOW - 1)          # sliding windows of length 3
    if args.max_windows:
        n_windows = min(n_windows, args.max_windows)
    print(f"sample has {n_time} times -> {n_windows} teacher-forced windows")

    run_forward, captured = build_apply(model_config, task_config, stats, set(hook_steps))
    apply = functools.partial(run_forward.apply, params, {}, jax.random.PRNGKey(0))

    per_step = {s: [] for s in hook_steps}           # step -> list of (n_mesh, 512)
    for w in range(n_windows):
        win = rebased_window(sample, w)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            win, target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(task_config))
        captured["count"] = 0
        captured["acts"] = {}
        apply(inputs, targets * np.nan, forcings)
        assert captured["count"] == N_MESH_GNN_STEPS, captured["count"]
        for s in hook_steps:
            emb = np.squeeze(captured["acts"][s], axis=1).astype(np.float32)  # (n_mesh,512)
            per_step[s].append(emb)
        print(f"  window {w}: captured {[per_step[s][-1].shape for s in hook_steps]}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for s in hook_steps:
        stacked = np.concatenate(per_step[s], axis=0)   # (n_windows*n_mesh, 512)
        dest = out_path if s == hook_steps[0] and len(hook_steps) == 1 else \
            out_path.with_name(out_path.stem + f"_step{s}.npy")
        np.save(dest, stacked)
        print(f"step {s}: saved {stacked.shape} -> {dest}")

if __name__ == "__main__":
    main()
