# `graphcast_sae.concepts` — purified concept groups and their response operators

Paper: Fig. `fig:contrast` panels (a) and (c) — ten purified concepts dosed at four generic
initial states, response RMS vs lead against the measured numeric floor and a
scrambled-label permutation band (0/10 detected). Shipped output:
`results/fs_contrast_inputs.npy` (the compact bundle `figures/paper_fig_contrast.py` reads).

Run order:

1. `cgv2_actseries.py` — per-feature activation series over the 160 i.i.d. windows
   → `results/fs_cgv2_actseries.npy` (JAX env, CPU; regenerable, not shipped)
2. `cgv2_select.py` — the frozen purity + decorrelation node rule
   (`docs/prereg/prereg_concept_graph_v2.md`) → the concept groups
3. `respop.py` — the response-operator run: impulse each concept once, roll 60 h free,
   23 rolls per window (`docs/prereg/prereg_response_operator.md`) → `results/fs_respop.npy` (not shipped)
   (GPU, ~hours); `respop_score.py` scores it against the pre-registered bars
4. `figures/paper_fig_contrast.py` bundles `fs_respop.npy` into `fs_contrast_inputs.npy`

Supporting, not in the paper: `concept_cgraph_v2.py` / `concept_cgraph_v2_score.py`
(interventional concept graph, one 6 h step), `concept_rmse_analyze.py` (concept-group
global RMSE). The v1 concept graph lives in `legacy/`.
