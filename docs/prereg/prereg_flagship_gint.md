*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# PREREG — the flagship watching-graph Ĝ_int (FG-1 … FG-9)

**Frozen 2026-08-14, before the extraction finished and before any graph existed.**
Scope: `notes/spec_flagship_causal_physics.md` §4b–4c, §5. Everything downstream of the
watching graph in that spec (Ĝ_dyn, PX) is **out of scope here** and is not claimed.

**The one question.** At full 0.25° resolution, does the flagship SAE — as a decomposition
of GraphCast into modes — produce causal structure that lands on real weather physics, or
does it sit with the corrupted controls like the mini SAE did?

**The sharper question this run adds.** `sae_flag` is 4,096 SAE features KMeans'd into 39
clusters to match `leiden_flag`'s N̂, which guardrail #6 requires. Measured, that pooling
gives `sae_flag` **5,665 effective mesh nodes** per mode against `leiden_flag`'s **710**,
while a synoptic storm at 0.25° is 560–2,256 nodes. The finest member in the pool is
therefore the only one that cannot resolve the physics it is being tested on. A new member
`sae_sel_flag` — 39 **individual** SAE features at native footprint — separates "the SAE
has no causal physics" from "the SAE's causal physics was destroyed by being forced onto
Leiden's resolution".

---

## 1. Why the pipeline is being rebuilt before it is run

Two defects were diagnosed on the mini model on 2026-08-12/13 and cost a full
re-extraction there. Restarting the flagship extraction unchanged would reproduce both.

**D1 — `q_c` is fitted to the annual cycle.** Each mode's series is
`s_c(t) = ((W_c·A_t)·q_c) − (mbar_c·q_c)`. `flagship_sae/build_pool.py:168-169` sets
`q[c] = vt[0]`, the top-variance direction of the **raw** pooled activations. In layer-8
activations that direction *is* the annual cycle, so the pipeline selects the most
seasonal readout and then regresses the season out — it selects for exactly what it
deletes. On mini this made mode series 87–97 % seasonal and discarded 2.6–5.7× of the
anomaly variance. On the flagship pool v1 as published: within-member median |cos| 0.256,
varfrac mean 0.336.

**D2 — the Nyquist alias, which makes the naive fix WORSE.** Refitting `q` on
deseasonalized data using the *pipeline's own* design (trend + annual K=3 + diurnal K=1)
failed its anchor gate on mini (commit `f817f5b`; diagnosis `918a44a`). At 6-hourly
sampling the **diurnal K=2** harmonic is the **Nyquist frequency** — its sine column is
identically zero and its cosine is a pure (+1,−1) alternation separating {00Z,12Z} from
{06Z,18Z} — and it is *not* in that design. In the tropics, aliased diurnal convection is
the largest surviving coherent variance, so a variance-maximising readout points straight
at it. Five tropical modes fused at |corr| 0.94–0.99; condition number 4.8 → 757; across
nine graphs edge count tracks log(cond) at Spearman 0.92. **PCMCI+ manufactures edges when
variables are collinear**, so an edge-count rise on a refit basis is not evidence of
recovered power.

**The repair, already validated on mini** (`candidates/refit_channel_dirs_v2.py`): add the
diurnal K=2 column to the deseasonalization design so the alias is removed **before** the
top-variance direction is chosen. On mini: cond 757 → 12.7, zero modes above 50 % Nyquist
variance, `frac_eastward` 0.471 → 0.692 with both anchors at chance.

**Honest cost, stated before the run.** That single column carries the S2 semidiurnal
atmospheric tide *and* aliased diurnal convection, inseparable at 6-hourly sampling.
Removing it sacrifices a real degree of freedom to avoid an uninterpretable one. This is
the conservative basis, not the true one.

---

## 2. The pool (frozen)

Footprints are inherited **unchanged** from the published `pool_flag_candidates.npy`
(N = 39, 2to6 mesh, 40,962 nodes). Only `q_c` is refit, and two members are added.

| member | what it is | role |
|---|---|---|
| `leiden_flag` | Leiden communities of the layer-8 field | positive control |
| `sae_flag` | 4,079 alive SAE features KMeans'd to 39 | **the candidate** |
| `sae_sel_flag` | 39 individual SAE features, native footprint | **new** — resolution test |
| `vmax_flag`, `km_flag` | varimax / k-means on the field | reference |
| `shift_flag` | vmax footprints rotated +40° lon | **anchor** (wrong geography) |
| `qperm_flag` | vmax footprints, channel-permuted `q` | **anchor** (broken dynamics) |
| `qrand_flag` | vmax footprints, random unit `q` | **anchor** (new, see §4) |

---

## 3. `sae_sel_flag` — selection rule, frozen before the script exists

From `candidates/fs_feature_catalog.npy` and the frozen labels in
`results/fs_atlas_class.npy`:

1. **Universe.** `fire > 0`, `coh` finite and > 0, `firerate ≥ 1e-3`, and category **not**
   in `{numerical/geometry, climatology/clock}` — i.e. the spec's "exclude mesh-artifact
   features first", plus the clock features whose only content is the seasonal/diurnal
   cycle the pipeline then removes.
2. **Rank** by `score = firerate / coh_km` — the repo's own existing "coherent + high
   firing" heuristic (`flagship_sae/feature_select.py`), not a new one invented here.
3. **Footprint** = `loading_to_footprint(node_map[f])`, byte-identical to the function
   `build_pool.py` uses for every other member (sign-fix, clip at 0, 5 %-of-max floor,
   sum-normalise).
4. **Greedy decorrelation.** Walk the ranked list; accept a feature only if its footprint's
   cosine with every already-accepted footprint is `< 0.50`. Stop at N = 39. If fewer than
   39 are found, relax the threshold in steps of +0.05 and record the final value.
   *(Rationale: D2 above. Collinear nodes are what break PCMCI+.)*

**Declared limitation.** A selected set does **not tile the sphere**. Edge **counts** are
therefore not comparable between `sae_sel_flag` and the partition members, and **PX is not
defined for it** (the bijection guardrail #6 requires matched, exhaustive node universes).
`frac_eastward` *is* comparable: it is a ratio computed over the graph's own extratropical
zonally-separated pairs.

---

## 4. FG-1 — PRE-FLIGHT bars, on the i.i.d. dump, before any GPU is spent

Refit `q` with `harmonic_design_v2` on the 160-window i.i.d. dump, then for every member:

- **FG-1a** `n_modes` with **Nyquist fraction > 0.50** must be **0**.
  (Nyquist fraction = R² of the (+1,−1) alternation on the mode series *after* the
  pipeline's own deseasonalization, i.e. what actually survives into PCMCI+.)
- **FG-1b** `leiden_flag` conditioning healthy: **cond < 50** and **max |corr| < 0.90**.
  (mini: published 4.8 / 0.348 PASS; refit-v1 757 / 0.989 FAIL; refit-v2 12.7 / 0.457 PASS.)
- **FG-1c** Conditioning is reported as **`cond` + `min eigenvalue` + `max |corr|`, never a
  median or an effective rank alone.** A five-mode clique at 0.99 barely moves a median —
  that is exactly how this defect was missed by the other lane
  (`notes/refitq_reconciliation.md`). Medians may be reported *in addition*, never instead.
- **FG-1d** Anchor portability, **measured, not assumed**: `qperm_flag` is retained as an
  anchor only if median `|cos(q_qperm, q_vmax)| < 0.15` on the refit basis. `qrand_flag`
  (vmax footprints × random unit `q`) is added **unconditionally** as a second anchor; it
  costs zero GPU because it shares vmax's footprints and therefore vmax's pooled tensor.
  **`qrand` numbers may never be compared to published `qperm` numbers.**

**If FG-1 fails, it is fixed before extracting, not after.**

### AMENDMENT A1 — 2026-08-14, after the pre-flight, **before any graph exists**

The pre-flight ran (`results/flag_gint_preflight.json`). FG-1a **passes for all six fitted
members** (max Nyquist fraction 0.048–0.109, zero modes above 0.50) and FG-1b passes
(`leiden_flag` cond 29.1, min eig 0.133, max |corr| 0.705 — against v1's 192.5 / 0.034 /
0.943 and five modes above 50 % Nyquist). FG-1d: both anchors port (`qperm` median |cos| vs
vmax `q` = 0.015, `qrand` = 0.034).

**It fails FG-1a for the two anchors, and structurally cannot pass it for them.** Anchor `q`
is permuted / random rather than *fitted*, so nothing steers it away from the alias, and a
random channel readout of the raw pooled activations lands on it: `qperm_flag` has **12 of
39** modes above 50 % Nyquist variance (max 0.777), `qrand_flag` **2 of 39** (max 0.696).

**Action, fixed now rather than after seeing a graph:**

1. `qperm_flag` and `qrand_flag` are **kept unchanged** — they are the constructions the
   mini-v2 result was validated with, and altering them would break comparability.
2. A third anchor **`qrandc_flag`** is added: `qrand` with the rank-1 Nyquist channel
   signature projected out of each mode's readout (`flagship_sae/add_anchor_qrandc.py`).
   Same random direction, alias removed; footprints are vmax's, so it costs **zero GPU**.
   Required: max Nyquist fraction < 0.50 and median |cos(q, q_vmax)| < 0.15.
3. **Anchor edge COUNTS are declared non-comparable to candidate edge counts** in advance.
   The anchors carry a shared deterministic clock the candidates do not, which inflates
   their connectivity. `frac_eastward` — a *bearing ratio*, calibrated against the
   geography-permutation null that holds the edge set fixed — remains valid for them, and
   is the statistic the anchor gate (§7) is defined on.

**Declared in advance so it cannot be read as a post-hoc excuse:** a member with very many
edges has `frac_eastward` pinned near 0.5 by sheer count, so a high-edge anchor passes the
gate cheaply. Anchor edge counts are therefore printed next to every anchor verdict, and if
an anchor's count is inflated its "clean" verdict is reported as **weak evidence of
instrument health, not strong evidence** — `qrandc_flag` exists to cover exactly that case.

### AMENDMENT A2 — 2026-08-14, same moment, **before any graph exists**

The pre-flight also shows both SAE members remain **ill-conditioned after the repair**:
`sae_flag` cond 1,123.6 / min eig 0.0068 / max |corr| 0.950, `sae_sel_flag` cond 812.2 /
0.0095 / 0.959, against `leiden_flag` 29.1 / 0.133 / 0.705. (The repair still helped
enormously — v1 `sae_flag` was cond **168,044** / max |corr| 0.998 / 18 modes above 50 %
Nyquist.) Since **PCMCI+ manufactures edges when variables are collinear** (edge count vs
log(cond), Spearman 0.92 across nine mini graphs), `sae_flag`'s edge count cannot be
compared to `leiden_flag`'s, and the FG-5c circular-shift surrogate does **not** cover this:
a circular shift destroys the contemporaneous collinearity along with the timing.

**Added now: FG-5e — the collinearity-preserving null (NULL C).** Fit a per-mode AR(1) to
each deseasonalized series, take the contemporaneous correlation matrix of the innovations,
and simulate surrogates with **the same persistence and the same cross-mode collinearity but
no cross-lag causality whatsoever**. Re-run the full 12-window consensus, S = 20 draws.
Observed edges are reported against this null for every member. This is the null that
answers "is this member's edge count real structure or its own conditioning?", and it is
the one that matters for `sae_flag`.

Pre-flight conditioning is measured on the i.i.d. dump (160 windows, 2016–2020) because
that is where `q` is fit; it is a **proxy** for the trajectory. It is re-measured on the
real trajectory at the data gate (§6), and the trajectory numbers are the ones that count.

---

## 5. FG-2 — extraction span and what is given up

Budget: **≤ ~15 GPU hours**, one job at a time on the single L40S.

**Measured before the budget was set, not assumed.** The v1 extractor ran at **6.99 s/win**
(`out/extract_traj_flag_status.txt`, block=120), which put 17,532 steps at ~34 h — too long
for one job. The v1 loop was strictly serial (ERA5 block download → window assembly → GPU
forward) and its 122-frame block is ~115 GB of ERA5 in RAM at once. `extract_traj_flag2.py`
pipelines the three stages across two producer threads with bounded queues and drops to
block=20. Measured on a 72-step smoke run: **1.38 s/win steady-state**, and the resulting
series match the v1 output at `corr = 1 − 9e-9` per member with identical `target_times`
(`out/extract_flag2_smoke.log`, SMOKE PASS). Budgeting conservatively at 2 s/win:

- **Span: 2007-01-01 → 2018-12-31, 17,532 six-hourly steps (12 years) — the FULL span the
  spec asked for.** `--nwin 12`, consensus **≥ 6/12**, identical to the mini run.
- **Nothing is given up on span.** The pipelining fix removed the constraint that would
  have forced a truncation; had it not, the fallback was 6 years / 6 windows, which is
  ~40× more permissive per ordered pair under a 0.05 per-window false-positive rate
  (`P(≥3/6) ≈ 2.2e-3` vs `P(≥6/12) ≈ 5.4e-5`). That fallback is **not** being used.
- **Hard stop rule, pre-registered.** If the job exceeds **16 wall-clock hours** it is
  stopped at the largest whole-year boundary reached; `--nwin` is then set to the number of
  whole years actually extracted, and the drop is stated explicitly in the report and in
  `results/flag_gint_datagate.json`. Nothing is silently truncated — the crash-safe
  incremental saves already in the script are kept.
- Members forward-projected: `leiden_flag, vmax_flag, km_flag, sae_flag, sae_sel_flag,
  shift_flag`. `qperm_flag` / `qrand_flag` are derived on CPU from vmax's pooled tensor.
- The extractor **also dumps the q-agnostic pooled tensor** `(T, N, 512)` fp16 per member
  (`--dump-pooled`, ~700 MB/member at T=17,532, N=39; 4.2 GB total against 32 GB free), so
  any future `q` is a CPU dot product instead of another GPU run. This is the defect-1
  lesson made structural.

**What is still given up, stated plainly.** `q_c` is fit on the 160-window i.i.d. dump,
which spans **2016–2020**, while the trajectory is **2007–2018** — the readout is fit on a
partly out-of-sample period. That is the existing pipeline's design (`build_pool.py` did
the same) and is not changed here, but it is a real limitation and is reported as one.
Mini also showed 84 % of per-year edges are regime-specific and 0 of 12 seasonal consensus
edges appear in all four seasons, so **non-stationarity is expected and a thin consensus is
the prediction, not a surprise.**

---

## 6. FG-3 — DATA GATE, before any analysis touches the trajectory

A local trajectory in this repo was once **98.6 % zeros** and silently corrupted three
analyses before a reproduction gate caught it (commit `551bd9e`). All six must pass:

- **DG-1** `n_done` equals the requested step count.
- **DG-2** zero all-zero rows, every member.
- **DG-3** no NaN / Inf, every member.
- **DG-4** per-member `min_c std(s_c) > 0`.
- **DG-5** `target_times` strictly increasing on an exact 6-h grid; first/last are the
  requested span.
- **DG-6** the pooled tensor reconstructs the stored scalar series at rel. err `< 5e-3`
  (fp16 storage tolerance).

**Any failure stops the run and is reported.** The trajectory conditioning
(`cond`, `min eig`, `max |corr|`) is re-measured here and reported next to every graph.

---

## 7. FG-4 — THE ANCHOR GATE (the whole instrument)

Anchors: `shift_flag`, and `qperm_flag` and/or `qrand_flag` per FG-1d.

An anchor is **clean** iff `frac_eastward ≤ 0.60` **and** its geography-permutation
p ≥ 0.05 (§8).

> **If any anchor is significant or passes a physics bar, that configuration is an
> INSTRUMENT FAILURE and is reported as such — not dropped, not repaired, and no member's
> number from that configuration is reported as physics.**

This exact rule killed Job B at N=40 and voided the mini refit v1. It is not negotiable.

---

## 8. FG-5 — physics bars, two-sided calibrated (guardrail #9)

The graph itself is produced by **`pcmci/gint_consensus.py`, unmodified**: RobustParCorr,
`TAU_MAX=8`, `PC_ALPHA=0.05`, `CONS_FRAC=0.5`, `--nwin 6`. `pcmci/signature_physics.py` is
also used **unmodified** (so the published mini verdicts remain reproducible by
construction; no bit-identity re-check is needed because nothing in it changes).

- **FG-5a `frac_eastward` > 0.60** on the ≥50 %-consensus pair edges, among extratropical
  (|lat| > 25°), same-hemisphere, |Δlon| > 15° pairs. (The frozen P2.3 statistic.)
- **FG-5b Significance vs the geography-permutation null.** Hold the graph fixed, permute
  the mode → centroid assignment 2,000×, recompute `frac_eastward`; require **p < 0.05**.
  This is the M1 locality fix — the naive null is inflated by geography.
- **FG-5c Edge count vs a circular-shift surrogate null.** Shift each mode's deseasonalized
  series by an independent random offset (preserves autocorrelation and marginals, destroys
  cross-mode timing), re-run the *entire* 6-window consensus, S = 20 draws. Report observed
  edges against null mean ± SD. This is what calibrates the 6-window permissiveness of §5.
- **FG-5d Signature physics** SG-1…SG-6 via `pcmci/signature_physics.py` with
  `--anchors shift_flag,qperm_flag,qrand_flag`. Verdicts reported for every member,
  including MISS.

**Guardrail #9, all three conditions, checked and printed:**
1. **The null VARIES** — report the SD of both nulls. A point-mass null is a broken bar and
   voids the result (this is what the G3 alignment-power audit was killed for).
2. **The threshold is ATTAINABLE** — report `max` of the geography-permutation null; it
   must exceed 0.60, or the bar is a ceiling artefact and is reported as vacuous.
3. **The threshold is FAILED by a negative control** — both anchors must miss it. If they
   do not, §7 applies.

---

## 9. FG-6 — what is reported, unconditionally

For **every** member: `N`, effective nodes (footprint participation ratio), edge count,
`frac_eastward` with its n-pairs, the geography-permutation p, the surrogate edge-count
null, `cond` / `min eig` / `max |corr|`, and the SG verdict. Anchors are reported in the
same table as the candidates, never in a footnote.

**An edge-count rise on the refit basis is NOT evidence of recovered power** — it is
reported only next to the conditioning numbers (Spearman 0.92 with log(cond) across nine
mini graphs).

---

## 10. FG-7 — what each outcome licenses

- `sae_flag` clears FG-5a+b with anchors clean → the flagship SAE, unlike the mini SAE,
  encodes recoverable weather physics in its causal structure. The strongest
  interpretability result in the program; still only the *watching* graph, so it is not a
  both-graphs claim.
- `sae_flag` fails, `leiden_flag` clears → the SAE dictionary describes the model
  (flagship tests 01–03) but its causal structure is not physical, at 0.25° as at 1°. A
  clean, informative negative.
- `sae_flag` fails **and** `sae_sel_flag` clears → the SAE's causal physics existed and was
  **destroyed by the pooling onto Leiden's resolution**. This is the finding the mini
  result could never settle, and it indicts the matched-N protocol, not the SAE.
- Both SAE members fail and `leiden_flag` clears → the SAE is not a causal basis at this
  resolution either.
- `leiden_flag` also fails → resolution is not the lever for physics; the thin mini signal
  does not strengthen at 0.25°. Reported as the program's ceiling.
- **Any anchor breach** → instrument failure at that setting, reported as such, and none of
  the above is licensed.

**A clean negative is a success.** This repo's standing discipline is that instrument
failures are reported as instrument failures; four earlier "findings" were voided that way.

---

## 11. FG-8 — already ruled out; not to be rediscovered or contradicted

- Flagship SAE features are **concepts, not places**: median footprint spread 7,300 km;
  only 6 of 4,096 under 2,000 km and all six polar. **Propagation / speed / V_MAX readings
  are not defined for such objects and will not be forced onto them.** Where `sae_flag` or
  `sae_sel_flag` modes are incoherent, `signature_physics` returns `incoherent/artifact`
  and that verdict stands as the answer.
- The concept-level **interventional** graph came back **VOID** (CG-1): a scrambled
  re-partition reproduced at ρ +0.893 against a < 0.30 bar (`notes/prereg_concept_graph.md`).
- The feature-level interventional graph is **not reproducible** across disjoint IC sets
  (ρ +0.181). Causal structure there is state-dependent.
- **`6→8→9` is RETIRED** as the mini headline (five strikes). What survives is band-level:
  eastward extratropical propagation at 25–40 m/s, confirmed mode-free by
  `probe/impulse_instrument.py` (extratropics 0.83 vs tropics 0.17). That instrument needs
  no mode basis and is the independent cross-check on any flagship claim made here.
- `V_MAX = 50` and every existing bar/verdict in the repo are untouched by this run.

---

## 12. FG-9 — files this run creates (new only)

```
notes/prereg_flagship_gint.md          this file (frozen first)
flagship_sae/build_pool_flag_v2.py     refit q (v2 design) + sae_sel_flag + qrand_flag
flagship_sae/extract_traj_flag2.py     q-agnostic --dump-pooled + pipelined prefetch
flagship_sae/finalize_traj_flag.py     zero-GPU anchors + the data gate
pcmci/gint_null_flag.py                geography-permutation + circular-shift nulls
candidates/pool_flag_v2_candidates.npy, candidates/pool_flag_v2_chandirs.npy
results/flag_gint_preflight.json, results/flag_gint_datagate.json
results/flag_gint.npy, results/flag_gint_nulls.npy, results/flag_signature_physics.npy
activations/mode_series/traj_flag2*.npy, /home/ec2-user/gc_flag_pooled/
```

`pcmci/gint_consensus.py` and `pcmci/signature_physics.py` are **not modified**.
