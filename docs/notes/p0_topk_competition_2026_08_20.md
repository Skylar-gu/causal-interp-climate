*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# P0: transmission exists, but half of it is top-k competition, not mechanism

The de-risking probe for the hybrid design (`notes/prereg_hybrid_hurricane.md`, section P0).
Three rollouts on `ida2021`, five minutes of GPU: baseline, ablate f2067 (a convection feature
with a known interventional effect), ablate f2850 (a frozen seed-7 control feature that fires in
the box but weakly — 14% of f2067's in-box amplitude).

## The pre-registered kill condition: PASS

> KILL if ablating a strong feature moves no other tracked feature by more than 10% of its
> baseline in-box amplitude.

405 of 2065 live features move by more than 10%. There is feature-to-feature transmission to
measure. **The design is alive.**

## The contrast that the criterion could not see

The probe carried a within-run contrast, and it is the reason the criterion should not be
trusted on its own:

| ablated | live | movers > 10% | of which pure on/off switches |
|---|---|---|---|
| f2067 (strong) | 2093 | 302 | 173 (57%) |
| f2850 (weak) | 2089 | 298 | 153 (51%) |

**302 versus 298.** Ablating a feature that fires seven times harder in the box moves *the same
number* of other features. As a test of "did this particular feature transmit", the mover count
is vacuous — it would pass for any ablation, which is the same defect that has voided five
results in this repo.

But the mover **identity** is not vacuous:

```
movers in common      162
only under f2067      140
only under f2850      136
Jaccard               0.370
```

Roughly two thirds of each mover set is specific to which feature was ablated. **The count is
uninformative and the identity is informative** — so any readout built on "how much moved" is
measuring the wrong thing, and one built on "which moved, against a matched control" is not.
That is what the design already does, and this probe is why it has to.

## The mechanism: k-sparsity budget, not spatial leakage

The SAE is top-k with k = 32 per mesh node. Mean active features per step inside the box:

```
baseline    919.4
ablate 2067 920.2
ablate 2850 920.6
```

The active-set size is conserved to within one feature. Meanwhile **51–57% of all movers are
pure on/off switches** — features that were exactly zero somewhere in the baseline and nonzero
after the ablation, or the reverse. Removing one feature's contribution changes which latents
clear the top-k threshold at each node, and something else takes the vacated slot.

**This is not the confound the field would predict.** The expected confound is footprint overlap
— ablate A and you mechanically disturb the field where B reads. Measured here it is absent:

- f2067's footprint is **45 of 40962 mesh nodes**. Of 1774 features with a nonempty footprint
  that fire in the box, 332 overlap it at all, **none above cosine 0.45**, max off-self 0.373.
- Regressing |ΔB| on footprint cosine: R² = **0.0032** over all 1774, R² = 0.066 over the 332
  with nonzero overlap.

Stated with the caveat the probe itself prints: the overlap support is thin, most cosines are
exactly zero, so a low full-dictionary R² does **not** by itself establish "leakage ruled out" —
the bar has to be attainable. The `cos > 0` rows are the only ones where the leakage hypothesis
makes a nonzero prediction, and it does not explain the movement there either.

So: **the dominant confound for feature-to-feature intervention in a top-k SAE is competition
for the sparsity budget, not spatial overlap of footprints.** Ablation frees k-slots and the
dictionary reshuffles globally to fill them.

## What this changes

1. **The asymmetry statistic keeps its job, for a new reason.** Amendment 1 withdrew its original
   justification — footprint overlap turned out to be nearly absent among these features. Top-k
   competition restores the need for a differenced statistic, and it is *not* obviously symmetric:
   removing a large feature frees more budget than removing a small one. So the asymmetry is not
   guaranteed to cancel it, and the matched-control arm — matched on firing amplitude, which is
   what sets the freed budget — is the leg that does.
2. **Absolute intervention readouts on top-k SAE features are not interpretable.** Any
   "ablate A, B responded" claim needs a matched-amplitude control ablation, or it is reporting
   the reshuffle.
3. **This is a property of the dictionary, not of GraphCast.** It should reproduce in any top-k
   SAE, and it predicts that reported feature-interaction graphs built from ablation deltas
   contain a large non-specific component whose size is set by k, not by the model's computation.
