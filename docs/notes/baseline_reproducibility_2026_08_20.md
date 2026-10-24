*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# The unpatched baseline is not reproducible across runs with different arm sets

Found 2026-08-20 by an identity gate that was put in to check something else, and it refused the
merge it was guarding.

## What happened

`conv_corectl3` re-ran four storms to give them the core-matched control they never had. The
merge script asserted that the two runs' **baseline** arms must agree to < 1e-4 hPa at every
lead — the baseline is an unpatched rollout and cannot depend on which control group was drawn,
so any disagreement means the two runs are not the same instrument. It fired immediately.

Per-lead difference of `mslp_min`, `conv_corectl3` minus `conv_corectl2`, both unpatched:

| storm | max abs | at +96 h | shift in `d_deepen` |
|---|---|---|---|
| ida2021 | 0.118 hPa | +0.030 | −0.030 |
| michael2018 | 0.098 | −0.008 | +0.068 |
| patricia2015 | 0.060 | +0.030 | −0.006 |
| wilma2005 | **0.248** | +0.248 | **−0.248** |

The two runs differ in exactly one thing that reaches the compiled graph: the **number of arms**
(4 in `conv_corectl2`, 2 in `conv_corectl3`, via `MECH_ARMS`).

The drift is near zero at the first few leads and grows with lead — 0.0 to 0.03 hPa through
+18 h, reaching its maximum at +66 to +96 h. That is the signature of a tiny numerical difference
amplified by an autoregressive rollout, not of a logic error.

## This was predicted in the repo and is now measured

`src/fs_common.py`'s own docstring records the reason the patch is threaded as runtime
arrays: *"per-arm trace constants let XLA fuse arms differently and a zero patch moved 2m
temperature by 0.16 K"*. The runtime-array design fixed that for arms **within** one compiled
graph. It does not make two graphs with different arm counts numerically identical, and nothing
had ever checked that it did.

## What it means for every number in this lane

**A cross-run difference of arms carries a floor of roughly 0.05–0.25 hPa on `d_deepen`.** Judged
against that floor:

- The convection ablation effect (median +2.63 hPa, up to +8.0) is far above it. Unaffected.
- The control effect (−0.04 hPa) is **below** it. "The control does nothing" is a statement that
  can only be made **within** a single run, never by comparing arms from two.
- `patricia2015`'s convection effect is +0.137 hPa, also below the floor. That storm's treatment
  and control cannot be compared across runs at all.

So the merge is refused, not forced, and the completion is re-run instead with **every** arm in
one invocation, so treatment and control share a compiled graph and a baseline.

## Consequence for the design of any future battery

Arms that will be compared must be produced by the same invocation. `MECH_ARMS` is a convenience
for saving GPU time on arms nobody will difference against each other; it is not safe for
splitting a comparison across two runs. Any battery that resumes after a crash with a different
arm subset has the same problem, silently.

The identity gate that caught this cost one extra assert and should be standard: whenever two
runs are combined, difference an arm that is *provably independent* of what changed between them,
and refuse the combination if it moves.
