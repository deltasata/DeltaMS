"""Paired comparison of two cross-validated runs on the same patients (pre-registered rule, DECISIONS.md E2).

For each patient with new lesions, take the difference in a score (new run minus reference run). Report the mean
difference and a 95% bootstrap interval (patients resampled with replacement, 10,000 times, seed 0).
Verdict: "better" if the interval lies above 0 and false lesions per lesion-free patient do not rise by more than
--max_fp_rise; "worse" if the interval lies below 0; otherwise "no evidence of a difference".

Inputs are the *_with_counts.csv files written by report_cv.py.

Usage: python scripts/compare_runs.py --ref results/E1b/heldout_scores_with_counts.csv \
           --new results/E2/heldout_scores_with_counts.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired(ref, new, metric, n_boot=10_000, seed=0):
    d = (new.set_index("patient")[metric] - ref.set_index("patient")[metric]).dropna()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d.values, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n_equal": int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--max_fp_rise", type=float, default=0.5)
    a = ap.parse_args()
    ref = pd.read_csv(a.ref, dtype={"patient": str})
    new = pd.read_csv(a.new, dtype={"patient": str})
    if set(ref.patient) != set(new.patient):
        raise ValueError("the two runs do not cover the same patients")
    les = set(ref.patient[ref.has_lesion])
    if les != set(new.patient[new.has_lesion]):
        raise ValueError("the two runs disagree on which patients have lesions")
    r, n = ref[ref.patient.isin(les)], new[new.patient.isin(les)]

    out = {m: paired(r, n, m) for m in ["F1_score", "Dice"]}
    fp_ref = ref[~ref.has_lesion].NbTestedLesions.mean()
    fp_new = new[~new.has_lesion].NbTestedLesions.mean()
    out["false_lesions_per_empty"] = {"ref": float(fp_ref), "new": float(fp_new)}
    lo, hi = out["F1_score"]["ci95"]
    if lo > 0 and fp_new - fp_ref <= a.max_fp_rise:
        verdict = "better"
    elif hi < 0:
        verdict = "worse"
    else:
        verdict = "no evidence of a difference"
    out["verdict"] = verdict

    for m in ["F1_score", "Dice"]:
        x = out[m]
        print(f"{m:9s} mean diff {x['mean_diff']:+.3f}  95% CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]  "
              f"patients better/worse/equal {x['n_better']}/{x['n_worse']}/{x['n_equal']} (n={x['n']})")
    print(f"false lesions per lesion-free patient: ref {fp_ref:.2f}, new {fp_new:.2f}")
    print(f"verdict (lesion F1 rule): {verdict}")
    dest = a.new.parent / f"compare_vs_{a.ref.parent.name}.json"
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
