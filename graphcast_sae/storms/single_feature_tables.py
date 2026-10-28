"""Single-feature amplification numbers (camera-ready Table 1, Table 3 feature columns, Sec. 3 text).

Reads results/skill/single_feature_mslp.json: MSLP error (hPa) per gain x storm for features 2401
(convection) and 3316 (low-level vorticity), from the J8 gain sweeps (docs/prereg amendments 4-6).
The per-run rollouts are not shipped; the table was transcribed from
the cluster runs, see the file's "provenance" field. Gain 1 is the unmodified forecast.

  best-alpha improvement: (unmodified - min over gains) / unmodified, per storm (Table 1, abstract)
  alpha = 3 improvement: per storm, and how many storms improve
  leave-one-out: per storm, the gain with the lowest median error over the other six (Table 3)

Run:   python -m graphcast_sae.storms.single_feature_tables
Outputs: results/skill/single_feature_tables.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/skill/single_feature_mslp.json"


def main():
    d = json.load(open(SRC))
    gains, storms = d["gains"], d["storms"]
    assert gains[0] == 1, "row 0 must be the unmodified forecast"
    out = {}
    for f in ("f2401", "f3316"):
        E = np.asarray(d[f], dtype=float)                       # (gain, storm)
        assert E.shape == (len(gains), len(storms)) and np.isfinite(E).all()
        base = E[0]
        best = 100 * (base - E.min(0)) / base
        at3 = 100 * (base - E[gains.index(3)]) / base
        loo_g, loo_err = [], []
        for j in range(len(storms)):
            others = [k for k in range(len(storms)) if k != j]
            g = int(np.argmin(np.median(E[:, others], axis=1)))
            loo_g.append(gains[g]); loo_err.append(float(E[g, j]))
        out[f] = dict(
            per_storm={s: dict(unmodified=float(base[j]), best_gain=gains[int(np.argmin(E[:, j]))],
                               best_err=float(E[:, j].min()), best_impr_pct=float(best[j]),
                               a3_impr_pct=float(at3[j]), loo_gain=loo_g[j], loo_err=loo_err[j])
                       for j, s in enumerate(storms)},
            best_impr_pct=dict(median=float(np.median(best)), min=float(best.min()), max=float(best.max())),
            a3=dict(n_improved=int((at3 > 0).sum()), median_impr_pct=float(np.median(at3))),
            loo=dict(median_err=float(np.median(loo_err)), gains=sorted(set(loo_g))),
        )
    (ROOT / "results/skill/single_feature_tables.json").write_text(json.dumps(out, indent=1))
    for f, v in out.items():
        print(f, "best-alpha impr %% median %.0f (%.0f-%.0f)" % tuple(v["best_impr_pct"].values()),
              "| alpha=3: %d/7 improve, median %.0f%%" % (v["a3"]["n_improved"], v["a3"]["median_impr_pct"]),
              "| LOO median %.1f hPa, gains %s" % (v["loo"]["median_err"], v["loo"]["gains"]))


if __name__ == "__main__":
    main()
