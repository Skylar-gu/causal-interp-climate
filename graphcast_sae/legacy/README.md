# `graphcast_sae.legacy` — superseded lanes, kept for provenance

**Nothing in this directory is used by the paper.** These scripts produced numbers in
earlier iterations that later designs replaced; they are kept so that the pre-registrations
in `docs/prereg/` that cite them remain checkable, and so that git history is not the
only record. Expect stale assumptions; do not build on them.

| lane | scripts | replaced by |
|---|---|---|
| feature steering atlas | `steer_probe.py`, `steer_summary.py`, `steer_cgraph.py`, `steer_interact.py`, `steer_modularity.py`, `steer_tc.py`, `steer_ida_chain.py`, `steer_ida_genesis.py` | the restore-to-normal counterfactual in `storms/` |
| G2 retry suite (`docs/prereg/prereg_flagship_g2_suite.md`) | `fs_retry1_communities.py`, `fs_retry2_steering_v2.py`, `fs_retry5_ablation.py`, `fs_score.py`, `fs_score_v2.py` | `gridlock/` and `storms/` |
| concept graph v1 (`docs/prereg/prereg_concept_graph.md`) | `concept_cgraph.py`, `concept_cgraph_score.py` | `concepts/concept_cgraph_v2*.py` |
| 80-storm ERA5 battery | `mega_sweep.py`, `mega_merge.py`, `mega_calibrate.py`, `mega_registry.py`, `skill_mega_storms.py` (generated) | the frozen seven-storm registry |

Their outputs are not shipped; the scripts still run with the same environment as their
replacements (see each docstring's `Run:` line).
