"""Official MSSEG-2 scoring with Anima's animaSegPerfAnalyzer.

The command follows the challenge evaluation document (June 18, 2021):
    animaSegPerfAnalyzer -d -l -s -S -i tested.nii.gz -r reference.nii.gz
with detection parameters alpha=10% (-x), gamma=65% (-z), beta=70% (-y), and a minimum lesion size of 3 mm^3 (-v).
These are Anima's defaults; they are passed explicitly so a future Anima release cannot change them silently.

For patients whose reference is empty, Anima reports the number and volume of detected lesions instead.

Before scoring, the prediction is checked to lie on the reference grid (same shape, same affine) and is rewritten
with the reference header. Reason: Anima 4.2 crashes (segmentation fault) on some oblique scans when the file
stores its orientation only in the NIfTI sform field, as nibabel does by default (found 2026-11-23 on one oblique scan).

Usage:
    python scripts/evaluate.py --pred PRED_DIR --ref REF_DIR [--out results.csv]
PRED_DIR and REF_DIR hold <patient>.nii.gz files with matching names.
"""

import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ANIMA = Path(os.environ.get("DELTAMS", Path(__file__).resolve().parents[1])) / "tools/anima/animaSegPerfAnalyzer"
OFFICIAL = ["-d", "-l", "-s", "-x", "0.10", "-z", "0.65", "-y", "0.70", "-v", "3"]


def anima_score(pred: Path, ref: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "res"
        p_img, r_img = nib.load(str(pred)), nib.load(str(ref))
        if p_img.shape != r_img.shape or not np.allclose(p_img.affine, r_img.affine, atol=1e-3):
            raise ValueError(f"{pred} is not on the grid of {ref}")
        mask = (np.asanyarray(p_img.dataobj) > 0.5).astype(np.uint8)
        pred = Path(td) / "pred.nii.gz"
        nib.save(nib.Nifti1Image(mask, r_img.affine, r_img.header), str(pred))
        cmd = [str(ANIMA), *OFFICIAL, "-X", "-t", "1", "-i", str(pred), "-r", str(ref), "-o", str(base)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        xmls = list(Path(td).glob("*.xml"))
        if len(xmls) != 1:
            raise RuntimeError(f"expected one XML output, got {xmls}")
        out = {}
        for m in ET.parse(xmls[0]).getroot().iter("measure"):
            try:
                out[m.attrib["name"]] = float(m.text)
            except (TypeError, ValueError):
                out[m.attrib["name"]] = m.text
        return out


def score_pairs(pairs, workers: int = 8) -> pd.DataFrame:
    """pairs: list of (patient_id, pred_path, ref_path)."""
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: anima_score(t[1], t[2]), pairs))
    df = pd.DataFrame(res)
    df.insert(0, "patient", [t[0] for t in pairs])
    return df


def score_dir(pred_dir: Path, ref_dir: Path, workers: int = 8) -> pd.DataFrame:
    refs = sorted(ref_dir.glob("*.nii.gz"))
    pairs = [(r.name.replace(".nii.gz", ""), pred_dir / r.name, r) for r in refs]
    missing = [p for _, p, _ in pairs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} predictions missing, e.g. {missing[0]}")
    return score_pairs(pairs, workers)


def summarize(df: pd.DataFrame) -> dict:
    """Challenge-style summary: lesion cases and empty cases are reported separately."""
    les = df[df["F1_score"].notna()] if "F1_score" in df else df.iloc[0:0]
    empty = df.drop(les.index)
    s = {"n_lesion_cases": len(les), "n_empty_cases": len(empty)}
    for k in ["F1_score", "Dice", "SensL", "PPVL"]:
        if k in les:
            s[f"{k}_mean"] = les[k].mean()
            s[f"{k}_sd"] = les[k].std()
    for k in ["NbTestedLesions", "VolTestedLesions"]:
        if k in empty:
            s[f"{k}_mean"] = empty[k].mean()
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    df = score_dir(a.pred, a.ref)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    for k, v in summarize(df).items():
        print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")
