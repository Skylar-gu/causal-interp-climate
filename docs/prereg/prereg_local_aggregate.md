*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — is the convection edit a push along the dominant local mode?

**Frozen 2026-08-21, before any number from `geom_test.py` has been read.** The script was
written first (it is deterministic and seeded); this file fixes the decision rule so the
verdict is not chosen after seeing the answer.

## The question

The paper claims clamping three convection features costs a median 19% of 96 h deepening,
and `notes/exposure_confound_2026_08_20.md` establishes that the effect is not set by how
much in-box activation the clamp removes. That rules out one confound. It does not rule out
the geometric one:

> Inside the storm box, the layer-8 activations are dominated by a small number of modes.
> If the convection group's decoder directions largely span the leading one, then "clamp
> convection" means "damp the box's dominant local mode", and the mechanism label is a name
> attached to an aggregate. That is the SAVAR picture transferred to GraphCast: one
> spatially aggregated mode, several dictionary elements pointing at it.

The alternative is that the group's displacement is substantially orthogonal to the leading
mode, i.e. it carries structure the dominant mode does not. That would be the more
interesting result and it complicates our own thesis, because it means the mechanism is
causally accessible by intervention while remaining unidentifiable from the internals.

## Data and statistic

`fs_iid_dump.npy`: 160 IID windows, 2016-2020, 40,962 mesh nodes x 512 dims, hook step 9.
No new rollouts. Storm boxes are located by the TC feature 3243 in the tropics and sized
like the real battery's (23 deg lat x 40 deg lon, ~800 mesh nodes).

Per box: centre the in-box activations over nodes, SVD, take principal directions `v_j`.
For a feature group G, the delete displacement is `D = -f[:, G] @ W_dec[:, G].T` in raw
activation units, the same units `delta_cond` writes back in.

- `align1(G)` = sum_nodes (D . v_1)^2 / sum_nodes ||D||^2
- `alignK(G)` = same, cumulative over K = 5
- `sub(G)`    = mean squared cosine of the principal angles between span{d_i : i in G} and
  the top-5 PC subspace. Clamp-target-free.

A uniformly random direction in 512 dimensions gives `align1` = 1/512 = 0.2%.

## Groups

| group | features | measured effect |
|---|---|---|
| convection | 2401, 2067, 3174 | +2.79 hPa |
| asc21 | 553, 866, 1981 | +3.63 |
| asc17 | 3357, 1033, 3314 | +0.02 |
| moisture, calibrated | 2958, 2671, 37 | −0.03 |
| TC feature | 3243 | positive control |
| matched random | 200 draws/box, matched feature-by-feature on in-box code mass | null |

## Decision rule, fixed in advance

Let `q10`, `q90` be the 10th and 90th percentiles of `align1` over the matched-random draws.

- **LOCAL-AGGREGATE** if `align1(convection) >= q90` **and** `align1(convection) >=
  0.5 x align1(TC)`. The edit moves along the dominant mode more than a mass-matched random
  group does, and comparably to the storm's own object.
- **ORTHOGONAL** if `q10 <= align1(convection) <= q90`. Indistinguishable from a generic
  group of the same firing mass, so the leading mode does not explain the intervention.
- **NEITHER** otherwise, including `align1(convection) < q10`, which would be its own
  finding and is not to be reported as either verdict above.

Reported on the median over boxes, with the 10-90 spread across boxes shown alongside. A
verdict is quoted only if it holds on at least 70% of boxes individually.

## Guardrail #9 — the bar must be calibrated on both sides

1. **The null must VARY.** If the 200 matched-random draws return a point mass, the
   statistic is not a measurement and nothing here is readable. This is the SPD failure and
   it is checked first.
2. **The bar must be ATTAINABLE under the null.** Some random draws must exceed the
   LOCAL-AGGREGATE bar. If none can, the bar is above the statistic's own ceiling and the
   verdict is vacuous. This is the BSF block-threshold failure.
3. **A negative control must FAIL it.** `moisture` (2958, 2671, 37) has centroids at
   38-56 S, near-zero in-box exposure and a null intervention effect. If it also reads
   LOCAL-AGGREGATE, the statistic is measuring firing mass rather than geometry and the
   result is void.

## Declared limits, before the fact

- **These are not the seven storms.** The dump is IID 2016-2020 initial conditions; the
  named storms are mostly outside it. The boxes are TC-feature-located surrogates, so this
  measures the geometry of the convection direction in tropical-cyclone-like boxes, not in
  Ida specifically. A confirmatory run on the real storms needs one baseline rollout each
  with an in-box activation dump, which is GPU work and is not done here.
- **Single time step, not a rollout.** Each window is one initial condition, so this is the
  geometry at the moment of the edit, not its accumulation over 96 h.
- **Delete direction, not restore.** `sub(G)` is target-free; `align1`/`alignK` use delete
  weights. Restore-to-normal spans the same subspace with different node weights.
- **`align1` is a within-box statistic.** It says nothing about whether the response the
  model produces downstream is mode-like.

## What each verdict would mean for the paper

- LOCAL-AGGREGATE: the intervention result stands as measured, but the mechanism reading
  weakens sharply. Section 4.1 would have to say the edit damps the box's dominant mode and
  that the convection label rides on it. This is reportable and it is the honest version of
  the SAVAR transfer.
- ORTHOGONAL: the intervention is mechanism-specific in a way the internals' dominant
  structure does not predict, which strengthens the positive result and simultaneously
  explains why observational discovery on the same internals keeps failing. It also raises
  a question the paper cannot currently answer: what does distinguish asc21 from asc17.
- NEITHER: reported as such, with no verdict.
