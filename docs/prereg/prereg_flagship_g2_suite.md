*Historical document, kept as written; script paths refer to the earlier layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — G2 SAE suite ported to FLAGSHIP GraphCast (0.25°/37-lev)

**Written 2026-08-08 BEFORE any experiment number from this suite exists.** Extends
`notes/prereg_flagship_sae.md` (retry #5 detection, already frozen) to the remaining
G2 retries — **#1 Ising/Leiden communities**, **#5 causal-inertness ablation**,
**#2 steering** — run on flagship GraphCast with the **published** SAE
(`theodoremacmillan/sae-graphcast-k32-lat4096-lay08`, k=32, dict 4096, layer 8).

Environment (fixed): `~/graphcast-interpretability/.venv` — the paper authors' repo
venv, carrying their `graphcast@sae-hooks` fork (ActivationManager) and their `SAE`
class. Forward passes run **CPU** (`JAX_PLATFORMS=cpu`); the L4's 23 GB is below
flagship's ~32 GB inference floor.

Per the single-model-coherence rule, **nothing here is quantitatively
cross-comparable to the main G1 rung** (`graphcast_small`, 1°/13-lev). Small-model
numbers are quoted only as the *design provenance of a threshold*, never as a
comparison. All outputs tagged `results/flagship_sae_*`.

---

## 0. Facts already established before this prereg (not outcomes of it)

Recorded for auditability, since they were measured before the bars below were frozen:

- **Hook alignment / FVU gate — RESOLVED.** The published SAE reconstructs our
  flagship layer-8 activations at **FVU = 0.134** at processor step 9 (1-indexed) —
  a sharp unique minimum over all 16 processor steps (next best 0.195 at step 8,
  0.508 at step 16). The prior `results/flagship_sae_gridlock.json` "BLOCKED,
  FVU > 340 at every layer" reading was a **metric error, not a pipeline error**:
  the authors' `SAE.forward` normalizes its input per token (centre + unit L2) but
  the reconstruction target is the **raw** activation, so FVU must be scored against
  raw x. Scored that way the SAE is in our activation space.
- **No version skew.** Activations captured through the authors' fork
  (`ActivationManager(mode="post_res", save_steps=[8])`) and through our stock
  interceptor at processor step 9 give identical statistics and identical FVU
  (0.1336). `layer0008` is 0-indexed = our 9th `_process_step`.
- FVU 0.134 here vs ~0.08 reported in the paper is expected and is **not** treated as
  a discrepancy to explain away: ours is one window of DeepMind's example `.nc`
  (2022-01-01), outside their 1979–2018 train / 2019–2021 val split. The suite below
  re-measures FVU on the actual WB2 window set and reports it.

**Gate on the whole suite (fixed now):** the catalog run must reproduce
**FVU ≤ 0.20** at processor step 9, median over the extracted windows. If it does
not, the suite is reported as BLOCKED and no retry below is scored.

---

## 1. Shared substrate

- **Windows:** 24 teacher-forced windows, evenly spaced across the seasonal cycle of
  the held-out year **2021** (SAE val years 2019–2021; the paper trained on
  1979–2018), streamed from WeatherBench-2
  `1959-2022-full_37-6h-0p25deg_derived.zarr` — the authors' own training source.
  Start times fixed by rule: 2021-01-01T00 + i·(365 d / 24), snapped to the 6-h grid,
  i = 0..23. Fixed before extraction; recorded in the meta JSON.
- **Tokens:** 40,962 M6 mesh nodes per window ⇒ 983,088 tokens.
- **Activations:** processor step 9 (= their `layer0008`), post-residual,
  `mesh_nodes.features`, float32 forward.
- **Degree substrate:** merged M6 multi-mesh node degree, CPU-reconstructed from
  `graphcast.icosahedral_mesh` (splits=6), aligned to mesh-node order.
- **Intervention algebra** (ported from `probe/sae_intervene.py`, one compiled graph
  for all arms, patch threaded as runtime arrays):

  ```
  xn        = (x - mean_tok(x)) / ||x - mean_tok(x)||     # authors' per-token norm
  f         = TopK_32(relu((xn - b_pre) @ W_enc.T))       # sparse code
  recon     = f @ W_dec_unit.T + b_pre                    # in RAW activation units
  delta     = (f * coef) @ W_dec_unit.T + rho*(recon - x) + uvec
  x_new     = x + delta
  ```

  `coef = 0, rho = 0, uvec = 0` reproduces the model exactly. Differs from the
  small-model algebra in one declared way: the published SAE's decode is already in
  raw units (no global `scale`), because its training loss compared `recon` to the
  un-normalized input.

---

## 2. Retry #1 — Ising/Leiden grouping of SAE features (flagship)

Port of `sae/retry1_ising_leiden.py`. Pipeline **unchanged**: per-feature spatial
signature + per-(feature,window) node-sum `P[f,w]` + Gram `G[f,g]`; harmonic
deseasonalization (linear trend + annual K=3 + diurnal K=1) removed in closed form
from the coupling; deseasonalized Pearson correlation; features kept at firing rate
> 0.5 %; |corr| graph with edges above the median positive |corr|; **Leiden**
(RBConfiguration/modularity, `leidenalg`, seed 0).

**Declared deviation:** 24 windows here vs 480 in the small-model run. The harmonic
design (7 columns) is fit on 24 points — reported, and the annual K is dropped to
**K=1** (3 columns + trend) to keep ≥ 4:1 points-per-parameter. Diurnal K=1 retained.

**BAR (identical decision rule to the small-model prereg):** PASS iff
**#non-trivial communities ≥ 2** (community with ≥ 5 features AND ≥ 1 % of kept
features) **AND median per-community identity-R² > 0.03**. Identity-R² = rank-1 SVD
energy fraction of the row-normalized member spatial signatures. Between-community
clustering R² reported for transparency, not scored.

---

## 3. Retry #5 — causal-inertness ablation (flagship)

Port of `probe/retry5_ablation.py`, scored against §1.4 of
`notes/prereg_g2_retries_gpu.md` with sample sizes reduced for CPU forward cost
(~3 min/pass). Runs only if §2 of `notes/prereg_flagship_sae.md` (detection) yields
≥ 1 grid-locked candidate; targets are the **top 2** by |ρ_deg|·CI.

**Arms:** `ref`; `noise` (no-op recomputed, numerical floor); `recon` (ρ=1, SAE
substitution floor); one arm per target (`coef = −1`); **10** mass-matched random
single-feature controls per target (small-model used 20 — reduced for CPU budget,
which weakens the empirical p-floor to 1/11 = 0.091; see the amended bar).

**Matched-control rule:** unchanged — eligible pool = alive features with
|log m_f − log m_T| ≤ log 2 (m = total activation mass over the 24 windows),
excluding grid-locked candidates; 10 drawn uniformly (seed 0), else the 10 nearest
in |log m| with the fallback reported.

**Metric:** unchanged — stddev-normalized, cos-lat-weighted RMSE over all prognostic
variables/levels using GraphCast's own `stddev_by_level`; median over windows.
**4 windows** from the 2021 set, **4 steps**, **primary lead 24 h**; full 6–24 h
curve reported. Targets: the **top 1** grid-locked candidate by |ρ_deg|·CI (the bar
is "≥ 1 feature"); a second target is run only if budget allows and is descriptive.

**Budget justification, recorded now.** A measured flagship CPU forward is ~1.9 min
in this venv, and the L4 GPU was re-tested at batch 1 and **OOMs** (needs a 19.2 GB
single allocation against an 18.9 GB usable limit) — so the small-model design
(32 windows × 8 steps × 46 arms) would cost ~35 GPU-free CPU-days. The design above
is 14 arms × 4 windows × 4 steps ≈ 224 forwards ≈ 7 h. **Every reduction is declared
here, before any number:** windows 32→4, leads 8→4, controls 20→10, targets 2→1.

**BAR, amended for n=10 controls (fixed now):**
- (i) **control validity gate:** `median_i c_i > 10·n_noise`, else INCONCLUSIVE;
- (ii) **inertness:** empirical one-sided `p = (1 + #{i : c_i ≤ g}) / 11 ≤ 0.091`,
  i.e. `g` must be the **strict minimum** of the 10 controls. This is a weaker
  achievable α than the small-model 0.05 and is declared as such **now**, before any
  number: a p at this floor is reported as "p = 0.091 (floor at n=10)", never
  rounded to "significant at 0.05";
- (iii) **effect size:** `Q = g / median_i c_i < 0.5`;
- (iv) **power:** if `g < 3·n_noise` ⇒ INCONCLUSIVE on power grounds.

PASS iff ≥ 1 target satisfies (i) ∧ (ii) ∧ (iii) ∧ ¬(iv).

**Determinism amendment (measured before any experiment number, recorded now).** The
small-model bars were written for a GPU forward that is *not* bit-deterministic, which
is what the `noise` arm was for. On this CPU path the no-op arm reproduces the
reference **bit-identically** — a repeat call gives nRMSE exactly 0. Consequences,
fixed now: (a) `n_noise = 0`, so guard (i) is trivially satisfied and guard (iv) can
never fire — **both are therefore vacuous here and will be reported as vacuous, not as
passed**; (b) the sole surviving floor reference is the `recon` arm (ρ=1, the SAE's own
reconstruction error in forecast space). A target whose ablation effect is far below
`recon` is reported as **INCONCLUSIVE on floor grounds**, replacing guard (iv). The
measured smoke values that motivated this (noop 0.0, single-feature ablation 1.4e-2,
recon 1.6e-1 at 6 h, one window) are plumbing checks, not outcomes of any bar.

---

## 4. Retry #2 — SAE steering (flagship)

Port of `probe/retry2_steering.py`. **The one structural adaptation, declared now:**
the small-model run steered SAE features and read the response out in the
**`leiden_act` causal mode basis** (the G1 pool winner). That basis does not exist at
flagship — the causal-discovery spine has not been transferred (see
`notes/flagship_transfer_plan.md`). Flagship therefore reads the response out in the
**SAE-feature-community basis produced by §2**: for community c,
footprint `W_c[n]` = mean member spatial signature (L1-normalized), channel direction
`q_c` = mean member decoder column (L2-normalized), readout
`m_c = Σ_n W_c[n] (A[n] · q_c)` — the same readout *form* as the small-model modes.
`σ_c` = std of the deseasonalized community series over the 24 windows.

**Consequence, stated in advance:** the leakage number is therefore *not* comparable
to the small-model 0.022 nor to the SAVAR 0.81 floor — different basis, different
model. It is a flagship-internal measurement of whether one SAE feature moves its own
community and not the others. Both prior numbers are cited as provenance only.

**Handle selection (fixed rule, run before the sweep):** alive features with rate
≥ 0.005 and |ρ_deg| < 0.30 (grid-locked excluded); for every (feature, community)
pair, Pearson correlation over the 24 windows between the deseasonalized node-mean
code series and the deseasonalized community series; greedily take the top **4** with
distinct features and distinct communities (small-model used 8 — reduced for CPU
budget). Requires ≥ 4 non-trivial communities from §2; if §2 yields fewer, the sweep
takes as many handles as there are communities and reports the reduction.

**Intervention:** impulse at step 0 only, `coef = α` on the target feature; doses
`α ∈ {−1, −0.5, +0.5, +1, +2}` (5 of the small model's 7; ±0.25 dropped for budget).
Rolled with the patch off; responses read from layer 8. **4 windows, 2 steps,
primary lead 6 h** (same CPU budget argument as §3: 21 arms × 4 windows × 2 steps
≈ 168 forwards ≈ 5 h; small-model used 24 windows × 4 steps × 57 arms). A `noise`
arm (α = 0, rolled independently) gives the floor.
The SAVAR-geometry uniform arm is **dropped** at flagship (its purpose was
comparability to the 0.81 SAVAR floor, which the basis change already forecloses).

**Metrics:** unchanged in form —
`R_c(f,α,w,s) = [Σ_n W_c[n](ΔA(n,w,s)·q_c)] / σ_c`;
primary **leakage** `L_f = mean_α [ mean_{c≠t} |R_c(α)|_w / |R_t(α)|_w ]` at 6 h;
secondary basis-free energy leakage `L2_f = 1 − R_t²/Σ_c R_c²`; dose-response slope
and R² reported as *weak* evidence (linear in α by construction).

**BAR:** PASS iff `L_f < 0.5` for ≥ 1 of the swept features, with that feature's
on-target response ≥ 10× the `noise` floor. Features failing the 10× floor check are
reported as **underpowered**, not as numbers.

---

## 5. Reporting rules

- Outputs: `results/flagship_sae_{catalog,gridlock,communities,ablation,steering}.{npy,json}`;
  logs `out/fs_*.txt`.
- Any outcome ambiguous between "the effect is absent" and "the metric cannot see it"
  is reported as that ambiguity and is **not** resolved in the favourable direction.
- A MISS is a finding and is reported at the same prominence as a PASS.
- Every reduced sample size in §3/§4 relative to the small-model run is restated next
  to its result, so no flagship number is read as if it carried the small-model's
  statistical power.
