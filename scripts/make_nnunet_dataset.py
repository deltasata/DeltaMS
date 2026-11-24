"""Build the nnU-Net v2 raw dataset and the cross-validation split for MSSEG-2.

Inputs per case (two channels on the same voxel grid):
  channel 0 = FLAIR time01 (baseline), channel 1 = FLAIR time02 (follow-up)
Label: consensus mask (ground_truth.nii.gz, newest version v3), 0 = background, 1 = new lesion.

Split (DECISIONS.md, node "CV split"): 5 folds. Every scanner model except the dominant Philips Ingenia 3T
is held out whole (all its patients in one fold), so each fold tests on at least one scanner it never saw.
Ingenia 3T patients (about half the data) are spread over all folds. Folds are balanced for the number of
patients, of lesion-free patients and of lesions. Deterministic (seed 0).

Usage: python scripts/make_nnunet_dataset.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/msseg2"
DS = "Dataset201_MSSEG2"
RAW = Path(os.environ["nnUNet_raw"]) / DS
PRE = Path(os.environ["nnUNet_preprocessed"]) / DS
DOMINANT = "PHILIPS Ingenia 3.0T"
K = 5


def make_split(info: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(K)]
    load = np.zeros((K, 3))  # patients, lesion-free patients, lesions

    def add(fold, pats):
        folds[fold].extend(pats)
        sub = info.loc[pats]
        load[fold] += [len(sub), (sub.n_lesions == 0).sum(), sub.n_lesions.sum()]

    # 1) minority scanners, largest group first, whole group to the emptiest fold
    groups = info[info.scanner != DOMINANT].groupby("scanner").groups
    for sc in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        f = int(np.argmin(load[:, 0] + 0.01 * load[:, 2]))
        add(f, list(groups[sc]))
    # 2) dominant scanner, one patient at a time, keeping folds balanced
    dom = info[info.scanner == DOMINANT].sort_values("n_lesions", ascending=False)
    dom_ids = list(dom.index)
    rng.shuffle(dom_ids[: len(dom_ids)])  # break ties randomly, then sort stably by lesion count
    dom_ids = sorted(dom_ids, key=lambda p: -info.loc[p, "n_lesions"])
    for p in dom_ids:
        empty = info.loc[p, "n_lesions"] == 0
        score = load[:, 0] + (2.0 * load[:, 1] if empty else 0.05 * load[:, 2])
        add(int(np.argmin(score)), [p])
    return folds


def main():
    insp = pd.read_csv(ROOT / "results/data_inspection/per_patient.csv", dtype={"patient": str})
    info = insp.set_index("patient")[["gt_n", "scanner_t1"]].rename(columns={"gt_n": "n_lesions", "scanner_t1": "scanner"})

    for sub in ("imagesTr", "labelsTr"):
        (RAW / sub).mkdir(parents=True, exist_ok=True)
    for p in info.index:
        case = f"MSSEG2_{p}"
        links = {
            RAW / "imagesTr" / f"{case}_0000.nii.gz": SRC / p / "flair_time01_on_middle_space.nii.gz",
            RAW / "imagesTr" / f"{case}_0001.nii.gz": SRC / p / "flair_time02_on_middle_space.nii.gz",
            RAW / "labelsTr" / f"{case}.nii.gz": SRC / p / "ground_truth.nii.gz",
        }
        for dst, src in links.items():
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
    (RAW / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "FLAIR_t1", "1": "FLAIR_t2"},
        "labels": {"background": 0, "new_lesion": 1},
        "numTraining": len(info),
        "file_ending": ".nii.gz",
        "name": "MSSEG2 new MS lesions (training set, consensus v3)",
    }, indent=2))

    folds = make_split(info)
    splits = []
    rows = []
    for k, val in enumerate(folds):
        tr = [p for p in info.index if p not in val]
        splits.append({"train": [f"MSSEG2_{p}" for p in sorted(tr)], "val": [f"MSSEG2_{p}" for p in sorted(val)]})
        sub = info.loc[val]
        rows.append(dict(fold=k, n=len(sub), lesion_free=int((sub.n_lesions == 0).sum()), lesions=int(sub.n_lesions.sum()),
                         held_out_scanners="; ".join(sorted(set(sub.scanner) - {DOMINANT})),
                         ingenia=int((sub.scanner == DOMINANT).sum())))
    assert sorted(sum(folds, [])) == sorted(info.index), "every patient exactly once"
    PRE.mkdir(parents=True, exist_ok=True)
    (PRE / "splits_final.json").write_text(json.dumps(splits, indent=2))
    out = ROOT / "results/cv_split.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
