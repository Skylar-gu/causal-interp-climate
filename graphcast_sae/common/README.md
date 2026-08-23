# `graphcast_sae.common` — shared machinery

Used by every experiment; nothing here is a paper result on its own.

| module | what it is |
|---|---|
| `fs_common.py` | GraphCast 0.25°/37-level loading, the published TopK SAE (k=32, 4096 features, layer 8) in JAX, activation hooks and runtime patching (`build_apply_cond` / `delta_cond`: the "restore to normal inside a disk" counterfactual), WeatherBench2 streaming. Re-exports every path from `graphcast_sae.paths`. |
| `signature_physics.py` | great-circle geometry, storm-track / eastward-propagation tests used by the graph verdicts |
| `gint_consensus.py` | deseasonalisation + consensus helpers for the observational graph (Sec. 4) |
| `skill_conv_storms.py` | the **frozen** seven-storm tropical-cyclone registry (+ the non-developing control) behind Table `tab:mechanism-interventions`; `TC = 3243`, `CONV = [2401, 2067, 3174]` |
| `skill_conv_storms_cfg.py` | env-driven overlay on that registry (`CONV_ANALOG_SPAN`, `CONV_ANALOG_YEARS`) |
| `skill_xt_storms.py` | the extratropical (explosive cyclogenesis) battery, selected with `SKILL_STORMS=skill_xt_storms` |
| `skill_sh_storms.py` | the Southern-Hemisphere battery (not in the paper) |

Registries are separate modules on purpose: appending a storm to one can never move a
median already reported from another. Select one with `SKILL_STORMS=<module>`.

Environment: `fs_common` needs the JAX environment (`jax`, `dm-haiku`, `graphcast`,
`xarray`, `gcsfs`); the registries and `signature_physics` are plain numpy.
