*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# SPEC — Is a "blocking" feature causally necessary for GraphCast's 2021 heat-dome forecast?

Second phenomenon (a DRY, DYNAMICAL mechanism — the counterpart to convection→cyclone). Design
frozen before results (R4b). 2026-08-12. In-data-range: ERA5 zarr ends 2021-12-31; June 2021 is fine.

## The event

The **2021 Pacific-Northwest heat dome** — a record omega block over western North America, peak
~26–29 June 2021 (Lytton BC 49.6 °C, ~5 °C above any prior record). A persistent high-latitude z500
ridge; extreme heat underneath from subsidence + persistence.

## Claim under test

GraphCast represents the blocking ridge with an identifiable feature, and that feature is causally
necessary for the model to forecast the block *and* the record heat: remove it and the ridge
collapses, the heat dissipates, and the forecast worsens vs ERA5 — while a random-feature control
does not. (Contrast with convection→cyclone: this is large-scale, dry, dynamical, low-predictability.)

## Design

### Phase 1 — identify the heat-dome feature (few forwards)
- IC ≈ 24–25 Jun 2021; roll to the ridge peak (~27–29 Jun). Encode layer-8 → SAE codes.
- Among the atlas blocking/ridge candidates homed over W-NA — **1789 (+53,−115), 492 (+53,−120),
  2930 (+56,−118), 1703 (+60,−134), 1036 (+50,−104)** — find which actually fire on the event ridge
  (z500 max, 2m-T max over ~45–62°N, −100…−135°W). Report whether it's ONE feature or a few; take the
  top firing set as "the heat-dome feature(s)". Sanity: their firing must track the ridge in space,
  and rise as the dome builds (like 3243 tracked Ida).

### Phase 2 — causal knockout (GPU)
- Restore the heat-dome feature(s) to NORMAL within a ~1500 km disk around the ridge centre
  (reuse `build_apply_cond`/`delta_cond`; normal = quiet late-June W-NA analog years), held
  persistently through a 5–7 day rollout. Arms: baseline / block→normal / block→zero / random-ctrl.
- Readouts vs lead, and vs **ERA5 truth**:
  1. **the block:** z500 max anomaly (vs zonal mean) over the box — does the ridge collapse?
  2. **the heat:** 2m-temperature max over the box — does the record heat dissipate?
  3. **skill:** z500 & 2m-T error vs ERA5 — does removing the feature make the forecast worse?
- Controls: random-feature ablation (matched firing, W-NA-firing) for specificity; and the same
  feature ablated on a NON-block date (should do little when there's no ridge).

## Frozen success criteria
- **Blocking feature is a necessary causal handle** iff block→normal collapses the ridge (z500 max
  drops materially) AND reduces the box heat AND worsens z500/2m-T skill vs ERA5, all by more than the
  random-feature control; and ablation on the non-block date does little.
- **Distributed / not a lever** iff the ridge and heat are ~unchanged when the feature is restored to
  normal (even if the feature's own activation moves) — report straight; it would mean blocking is not
  localized to one feature, a real finding.
- Report the internal-vs-physical/skill split straight, as with convection.

## Compute & ops
- Phase 1 few forwards + Phase 2 ~4 arms × ~24 steps ≈ modest flagship GPU. bf16. SERIALIZE behind
  all running jobs (poll nvidia-smi, start only <6000 MiB sustained). Crash-safe saves to
  results/heatdome/. Figures: ridge z500 + 2m-T maps baseline vs block-off vs ERA5; the identified
  feature firing on the dome; skill bars vs controls. Commit code (results gitignored), push.
- Artifact update handled by the main session (a "second mechanism" panel / companion page).

## Guardrail
If GraphCast doesn't forecast the ridge in baseline, say so (the event may be at the edge of its
skill) — don't force it. A null (no single blocking feature; or ablation doesn't collapse the ridge)
is a real, publishable result about how the model represents blocking.
