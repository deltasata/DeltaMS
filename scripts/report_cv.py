"""Summarise held-out cross-validation scores as pre-registered (E0, E1 in DECISIONS.md).

Input: a CSV of per-patient Anima scores with columns `patient` and `fold`, plus a pattern giving each patient's
prediction file (used for the patient-level yes/no decision). Output, printed and written next to the CSV:
  - primary  : mean lesion F1 over patients with new lesions
  - secondary: mean Dice (same patients), false lesions per lesion-free patient, patient-level yes/no accuracy
each as mean +- SD across folds, and pooled over all patients.

False lesions on lesion-free patients use Anima's own count (its blobs can differ slightly from 26-connected
components). A patient is predicted "yes" if the prediction holds any 26-connected component of 3 mm^3 or more.

Usage: python scripts/report_cv.py --scores results/floor/heldout_scores.csv \
           --pred "results/floor/pred/grid{combo:02d}/{patient}.nii.gz"
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def n_lesions(path, min_mm3=3.0):
    img = nib.load(str(path))
    lab, n = ndimage.label(np.asanyarray(img.dataobj) > 0.5, np.ones((3, 3, 3), bool))
    if n == 0:
        return 0
    vox = float(np.prod(img.header.get_zooms()[:3]))
    return int((np.bincount(lab.ravel())[1:] * vox >= min_mm3).sum())


def metrics(df):
    les, empty = df[df.has_lesion], df[~df.has_lesion]
    return {
        "lesion_F1": les.F1_score.mean(),
        "Dice": les.Dice.mean(),
        "SensL": les.SensL.mean(),
        "PPVL": les.PPVL.mean(),
        "false_lesions_per_empty": empty.NbTestedLesions.mean(),   # official Anima count
        "yes_no_accuracy": (df.has_lesion == (df.n_pred > 0)).mean(),
        "n_lesion_cases": len(les),
        "n_empty_cases": len(empty),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--pred", required=True, help="format pattern with any CSV column, e.g. {patient}")
    a = ap.parse_args()
    df = pd.read_csv(a.scores, dtype={"patient": str})
    df["has_lesion"] = df.F1_score.notna()
    df["n_pred"] = [n_lesions(ROOT / a.pred.format(**r)) for r in df.to_dict("records")]

    per_fold = pd.DataFrame({k: metrics(g) for k, g in df.groupby("fold")}).T
    pooled = metrics(df)
    out = {"pooled": pooled,
           "fold_mean": per_fold.mean().to_dict(), "fold_sd": per_fold.std().to_dict(),
           "per_fold": per_fold.to_dict(orient="index")}
    print(per_fold.round(3).to_string())
    print()
    for k in ["lesion_F1", "Dice", "SensL", "PPVL", "false_lesions_per_empty", "yes_no_accuracy"]:
        print(f"{k:24s} pooled {pooled[k]:.3f}   folds {out['fold_mean'][k]:.3f} +- {out['fold_sd'][k]:.3f}")
    a.scores.with_name(a.scores.stem + "_summary.json").write_text(json.dumps(out, indent=1, default=float))
    df.to_csv(a.scores.with_name(a.scores.stem + "_with_counts.csv"), index=False)


if __name__ == "__main__":
    main()
