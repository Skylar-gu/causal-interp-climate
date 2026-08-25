*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# A measured noise floor under every intervention result in this repo

Found 2026-08-20 by a data gate that was checking something else, and measured at **zero extra
GPU cost** from a battery that had already run.

## The free experiment

The commitment-horizon battery includes `ramp-pulse15` — the convection group restored to normal
at rollout step 15, the **last** step. A dose at the final step cannot affect any earlier output.
So every difference between that arm and baseline at leads 6–90 h is not physics. It is noise.

Same for `rand-ramp-pulse15` on the control group. Sixteen independent measurements, already on
disk:

| storm | conv pulse15 | ctrl pulse15 |
|---|---|---|
| goni2020 | 0.131 | 0.145 |
| haishen2020 | 0.293 | **0.608** |
| haiyan2013 | 0.365 | 0.182 |
| ida2021 | 0.104 | 0.106 |
| michael2018 | 0.243 | 0.222 |
| nondev2013 | 0.031 | 0.036 |
| patricia2015 | 0.111 | 0.081 |
| wilma2005 | 0.156 | 0.373 |

**Floor on min-MSLP over a 96 h rollout: median 0.150 hPa, p90 0.369, max 0.608.**

The data gate that surfaced this was checking a physical identity — "outputs before the dose must
be bit-identical to baseline" — and it failed on every storm and every pulse. That identity is
the strongest gate the pulse design allows, and it should be standard in any lead-dependent
intervention.

## It is nondeterminism, not a code path

The first hypothesis was that patched and unpatched arms take different arithmetic paths
(`delta_gain` with a 4-tuple versus `delta_cond` with a 3-tuple), which would make the difference
a systematic bias rather than noise. It is not:

```
pre-dose max |d|,  conv vs baseline    median 0.104   p90 0.291   max 0.365
                   ctrl vs baseline    median 0.083   p90 0.304   max 0.608
                   conv vs ctrl        median 0.069   p90 0.248   max 0.490
```

Two arms that take the **identical** code path with the identical schedule, differing only in
which features are selected, disagree before the dose by the same amount as either disagrees with
baseline. Nothing cancels. And the arithmetic is provably exact — at a pre-dose step the gain is
1.0, so `delta_gain` computes `f + (1.0-1.0)*excess = f`, then `(f-f) @ W_dec.T = 0`, and adding
zero to a finite float is exact in IEEE. The differences are not coming from the patch.

They grow with lead — near zero at the first few steps, maximal at +66 to +96 h — which is a
chaotic system amplifying a tiny numerical perturbation, most plausibly non-deterministic
reductions in the mesh scatter/gather kernels on this GPU.

This is the same phenomenon as `notes/baseline_reproducibility_2026_08_20.md`, where two runs
differing only in arm count disagreed by up to 0.248 hPa. That note attributed it to the arm
count changing the compiled graph. **That attribution is now superseded**: the drift is present
between arms of a single compiled graph, so arm count was a coincidence, not the cause. The
practical conclusion of that note — never split a comparison across two runs — still holds, and
now has a stronger reason: you cannot even reproduce an arm against itself.

## What it does to the numbers on record

| result | value | verdict |
|---|---|---|
| convection ablation, median | +2.63 hPa | far above the max floor |
| convection ablation, max (haishen2020) | +8.01 | far above |
| largest single-step pulse (haiyan2013, k=8) | +2.14 | above |
| **in-box matched control** | **+0.02** | **below the median floor** |
| **core-matched control** | **−0.04** | **below** |
| **patricia2015 convection** | **+0.137** | **below** |

The headline interventional result is untouched: 2.63 hPa against a 0.15 hPa floor is a factor
of 17, and the largest storms are a factor of 50.

But three things must change in how the small numbers are reported:

1. **"The control costs ~0.02 hPa" is not a measurement.** It is a number smaller than the
   instrument's noise. The honest statement is that the control's effect is **below the
   detection floor of 0.15 hPa**, which is still the claim that matters — treatment is above,
   control is below — but the two-decimal value must not be quoted as if it were resolved.
2. **patricia2015 contributes nothing to the convection result.** Its +0.137 hPa is under the
   floor. Any median over storms that includes it is diluting a real signal with a null
   measurement, and the storm should be reported as under-resolved rather than as a weak effect.
3. **Effects between roughly 0.1 and 0.6 hPa require repeated arms to claim at all.** Nothing in
   this repo has ever run the same arm twice.

## What it does to the commitment horizon

Most single-pulse damages land at 0.03–0.2 hPa, i.e. at or below the floor. Only four
measurements clear it cleanly, and all four sit at pulse steps 6–8 (leads 36–48 h):
haiyan2013 k=8 (+2.14), haishen2020 k=6 (+1.70), haishen2020 k=8 (+1.49), haiyan2013 k=6 (+1.25).

So to the extent the commitment horizon is measurable at all, **it is mid-loaded, not
front-loaded** — the opposite of the pre-registered hypothesis. See
`notes/commit_horizon_result_2026_08_20.md`.

## Recommended, not yet run

A determinism probe: the same arm twice in one invocation, across several storms, to give the
floor a proper distribution rather than sixteen samples inferred from a terminal pulse. It is
~10 minutes of GPU and it should gate every future battery. Also worth testing whether
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes it and at what cost in throughput.
