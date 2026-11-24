"""Inspect the MSSEG-2 training data: formats, voxel and matrix sizes, intensity ranges, lesion counts.

Usage: python scripts/inspect_data.py [--root data/msseg2] [--out results/data_inspection]
Per-patient tables stay local (results/ is never pushed); report.txt holds aggregate numbers only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

STRUCT26 = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity, as Anima
MASKS = ["ground_truth", "ground_truth_expert1", "ground_truth_expert2", "ground_truth_expert3", "ground_truth_expert4"]


def lesions(mask, vox_mm3, min_mm3=3.0):
    lab, n = ndimage.label(mask > 0, structure=STRUCT26)
    if n == 0:
        return 0, 0, 0.0, []
    sizes = np.bincount(lab.ravel())[1:] * vox_mm3
    kept = sizes[sizes >= min_mm3]
    return n, len(kept), float(kept.sum()), kept.tolist()


def one_patient(pdir: Path):
    row = {"patient": pdir.name}
    sizes_out = []
    for t in ("time01", "time02"):
        img = nib.load(str(pdir / f"flair_{t}_on_middle_space.nii.gz"))
        d = np.asanyarray(img.dataobj).astype(np.float32)
        z = img.header.get_zooms()[:3]
        row.update({
            f"{t}_shape": tuple(img.shape), f"{t}_voxel_mm": tuple(round(float(v), 4) for v in z),
            f"{t}_dtype": str(img.get_data_dtype()), f"{t}_orient": "".join(nib.aff2axcodes(img.affine)),
            f"{t}_min": float(d.min()), f"{t}_p99": float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 0.0,
            f"{t}_max": float(d.max()),
        })
        if t == "time01":
            aff1 = img.affine
        else:
            row["same_grid"] = bool(np.allclose(aff1, img.affine, atol=1e-3)) and row["time01_shape"] == row["time02_shape"]
    vox = float(np.prod(row["time01_voxel_mm"]))
    row["voxel_mm3"] = vox
    for m in MASKS:
        mi = nib.load(str(pdir / f"{m}.nii.gz"))
        md = np.asanyarray(mi.dataobj)
        n_all, n3, vol, sz = lesions(md, vox)
        key = "gt" if m == "ground_truth" else m.replace("ground_truth_", "")
        row.update({f"{key}_n_all": n_all, f"{key}_n": n3, f"{key}_vol_mm3": round(vol, 1),
                    f"{key}_values": str(np.unique(md).tolist()[:4]),
                    f"{key}_same_grid": bool(np.allclose(mi.affine, aff1, atol=1e-3))})
        if key == "gt":
            sizes_out = sz
    return row, sizes_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/msseg2"))
    ap.add_argument("--out", type=Path, default=Path("results/data_inspection"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pats = sorted(p for p in a.root.iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_patient, pats))
    df = pd.DataFrame([r for r, _ in res])
    all_sizes = np.array([s for _, sz in res for s in sz])
    sc = pd.read_csv(a.root.parent / "raw/scanners.csv", dtype=str)
    sc1 = sc[sc.time == "time01"].set_index("patient")
    sc2 = sc[sc.time == "time02"].set_index("patient")
    df["scanner_t1"] = df.patient.map(sc1.device_manufacturer.str.split().str[0].str.upper() + " " + sc1.device_model + " " + sc1.device_field_strength + "T")
    df["scanner_t2"] = df.patient.map(sc2.device_manufacturer.str.split().str[0].str.upper() + " " + sc2.device_model + " " + sc2.device_field_strength + "T")
    df.to_csv(a.out / "per_patient.csv", index=False)

    les = df.gt_n
    rep = [
        f"Patients: {len(df)}",
        f"All files on the same voxel grid as time01: images {df.same_grid.all()}, masks {all(df[f'{k}_same_grid'].all() for k in ['gt','expert1','expert2','expert3','expert4'])}",
        "", "Voxel sizes (mm), time01:", df.time01_voxel_mm.astype(str).value_counts().to_string(),
        "", "Matrix sizes, time01:", df.time01_shape.astype(str).value_counts().to_string(),
        "", "Orientation:", df.time01_orient.value_counts().to_string(),
        "", "Data types:", df.time01_dtype.value_counts().to_string(),
        "", "Intensity 99th percentile (time01), across patients:", df.time01_p99.describe().round(1).to_string(),
        "", "Mask values:", df.gt_values.value_counts().to_string(),
        "", "Consensus new lesions per patient (>= 3 mm^3, 26-connected):", les.describe().round(2).to_string(),
        f"Patients with zero new lesions: {(les == 0).sum()} of {len(df)}",
        f"Total new lesions: {int(les.sum())}",
        f"Lesion volume (mm^3): median {np.median(all_sizes):.1f}, 10th pct {np.percentile(all_sizes,10):.1f}, 90th pct {np.percentile(all_sizes,90):.1f}, max {all_sizes.max():.0f}",
        "", "Lesions per expert (total, >= 3 mm^3):",
        *[f"  {k}: {int(df[f'{k}_n'].sum())} lesions, {int((df[f'{k}_n'] == 0).sum())} patients with none" for k in ["expert1", "expert2", "expert3", "expert4"]],
        "", "Scanner at time01 (patients):", df.scanner_t1.value_counts().to_string(),
        f"Patients whose scanner changed between time points: {(df.scanner_t1 != df.scanner_t2).sum()}",
    ]
    (a.out / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
