"""Cross-validated difference-imaging floor (experiment E0 in DECISIONS.md).

For each fold of splits_final.json: choose the threshold set that maximises mean lesion F1 (Anima) on the 32
training patients of that fold, then apply it unchanged to the 8 held-out patients. Only held-out scores are reported.

Usage: python scripts/floor_cv.py [--workers 8]
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import floor
from evaluate import score_pairs, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/msseg2"
OUT = ROOT / "results/floor"
CACHE = OUT / "cache"
GRID = dict(k_diff=[3.0, 4.0, 5.0, 6.0], k_bright=[1.5, 2.5], k_base_max=[1.0, 2.0])


def prep_one(pid):
    f = CACHE / f"{pid}.npz"
    if not f.exists():
        p = floor.prepare(DATA / pid / "flair_time01_on_middle_space.nii.gz", DATA / pid / "flair_time02_on_middle_space.nii.gz")
        np.savez_compressed(f, **p)
    return pid


def predict_one(args):
    pid, params, tag = args
    d = dict(np.load(CACHE / f"{pid}.npz"))
    mask, _ = floor.detect(d, params)
    out = OUT / "pred" / tag / f"{pid}.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, d["affine"]), str(out))
    return pid, out


def run_params(pids, params, tag, workers):
    with ProcessPoolExecutor(workers) as ex:
        preds = dict(ex.map(predict_one, [(p, params, tag) for p in pids]))
    return score_pairs([(p, preds[p], DATA / p / "ground_truth.nii.gz") for p in pids], workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.umask(0o077)
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / "Dataset201_MSSEG2/splits_final.json").read_text())
    pids = sorted(p.name for p in DATA.iterdir())
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(prep_one, pids))
    print("prepared", len(pids))

    # score every grid point on every patient once; fold-wise selection then uses only training patients
    combos = [floor.FloorParams(**dict(zip(GRID, v))) for v in itertools.product(*GRID.values())]
    allres = []
    for i, c in enumerate(combos):
        tag = f"grid{i:02d}"
        df = run_params(pids, c, tag, a.workers)
        df["combo"] = i
        allres.append(df)
        print(tag, c, f"F1(all) {df.F1_score.mean():.3f}", flush=True)
    allres = pd.concat(allres)
    allres.to_csv(OUT / "grid_scores.csv", index=False)

    held = []
    for k, s in enumerate(splits):
        tr = [x.replace("MSSEG2_", "") for x in s["train"]]
        va = [x.replace("MSSEG2_", "") for x in s["val"]]
        trs = allres[allres.patient.isin(tr)].groupby("combo").F1_score.mean()
        best = int(trs.idxmax())
        v = allres[(allres.combo == best) & allres.patient.isin(va)].copy()
        v["fold"] = k
        v["chosen"] = str(combos[best])
        held.append(v)
        print(f"fold {k}: chose {combos[best]} (train F1 {trs.max():.3f})")
    held = pd.concat(held)
    held.to_csv(OUT / "heldout_scores.csv", index=False)
    for key, val in summarize(held).items():
        print(f"{key:24s} {val:.4f}" if isinstance(val, float) else f"{key:24s} {val}")


if __name__ == "__main__":
    main()
