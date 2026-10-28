*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 5 to prereg_cluster_revision.md: accept the rebuilt i.i.d. dump for J8 (POST HOC)

Written 2026-10-20, **after** the amendment-4 dump-equivalence gate failed
(`notes/result_j8_dump_gate_2026_10_19.md`, job 23576867: worst in-box convection activation
difference 3.44% against a 2% bar). It is not a pre-registration of the dump choice; it is a
declared, post-hoc override of a failed gate, and is to be reported as such wherever J8 appears.
It is written before any J8 control is drawn and before any J8 arm runs; the J8 bar, gates,
gains and predictions in amendment 4 are unchanged.

**What is overridden.** Amendment 4 said that if the rebuilt dump missed the 2% gate the J8
controls would not be drawn from it. They will be drawn from it anyway.

**Why.** The original `fs_iid_dump.npy` is not available. The misses are mixed in sign
(-3.2% to +3.4%), 5 of 24 values, with no offset, which reads as L40S-vs-H100 bf16 rounding rather
than a different sample. They are an order of magnitude below the draw's own 25% matching
tolerance. Re-drawn J1 controls still fire at 80-113% (median 97%) of the group's in-box level.
The 2% bar was never calibrated against a known hardware-only difference. None of this makes the
gate pass; it is the stated reason for proceeding despite it.

**Consequences, stated now.**
- J8 controls are NOT J1's control source. J1 keeps its committed controls (drawn from the L40S
  dump); J8v and J8c share the rebuilt dump, so they are comparable with each other, and any
  J1-vs-J8 comparison carries this caveat.
- The paper reports: original dump unavailable; equivalence gate failed at 3.4% vs 2%; J8
  controls drawn from a rebuild under this post-hoc amendment.

**Replacement check, declared before the draw.** For f3316 and f2401 separately, in every one
of the 8 storms the drawn control's in-box climatological activation must be within 80-120% of the
treatment feature's (`new_inbox / conv_inbox`). A storm that misses has no valid control: its
treatment-control comparison is reported as missing and it counts as not passing toward the 5/7
bar. The draw itself (rule, seed 7, bans) is exactly as in amendment 4.
