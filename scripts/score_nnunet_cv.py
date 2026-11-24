"""Score the held-out predictions of an nnU-Net 5-fold run with the official Anima tool.

At the end of training nnU-Net predicts each fold's held-out patients with the final checkpoint and writes them
to fold_<k>/validation/. This script scores those files against the consensus masks and writes
results/<name>/heldout_scores.csv (columns: patient, fold, Anima measures), ready for report_cv.py.

Usage: python scripts/score_nnunet_cv.py --trainer nnUNetTrainerDiceCELoss_noSmooth --name E1b
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from evaluate import score_pairs

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.umask(0o077)
    run = Path(os.environ["nnUNet_results"]) / "Dataset201_MSSEG2" / f"{a.trainer}__{a.plans}__{a.config}"
    splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / "Dataset201_MSSEG2/splits_final.json").read_text())
    rows = []
    for k, s in enumerate(splits):
        for case in s["val"]:
            pid = case.replace("MSSEG2_", "")
            pred = run / f"fold_{k}" / "validation" / f"{case}.nii.gz"
            if not pred.exists():
                raise FileNotFoundError(pred)
            rows.append((pid, pred, ROOT / "data/msseg2" / pid / "ground_truth.nii.gz", k))
    df = score_pairs([r[:3] for r in rows], a.workers)
    df.insert(1, "fold", [r[3] for r in rows])
    df["pred_path"] = [str(r[1]) for r in rows]
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "heldout_scores.csv", index=False)
    print(f"wrote {out / 'heldout_scores.csv'} ({len(df)} patients)")


if __name__ == "__main__":
    main()
