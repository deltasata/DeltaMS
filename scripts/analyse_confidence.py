"""Lesion confidence, threshold sweep and calibration of a cross-validated nnU-Net run (DECISIONS.md, node A1).

Input: lesion probabilities of each fold's held-out patients, written by predict_heldout_probs.sh
(<probs>/fold_<k>/<case>.npz), and nnU-Net's own end-of-training masks for the same patients (consistency check).

Steps
1. Check: the argmax of the probabilities equals nnU-Net's validation mask (voxel and lesion-count agreement).
2. Lesion list: 26-connected components of 3 mm^3 or more of p > 0.5; confidence = highest p inside the lesion
   (mean p also kept). A lesion is "correct" if at least 10% of its voxels lie inside the consensus mask.
3. Threshold sweep: mask p > t for each t, scored with the official Anima settings (evaluate.py).
4. Calibration (reliability table, ECE with a patient bootstrap, Brier score, ROC AUC) and patient-level ROC AUC.
All outputs stay in results/<name>/ (patient-level files are local only).

Usage: python scripts/analyse_confidence.py --probs results/E1b/probs --trainer nnUNetTrainerDiceCELoss_noSmooth \
           --name E1b
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from evaluate import anima_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BINS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])
CONN = np.ones((3, 3, 3), bool)   # 26-connectivity
MIN_MM3 = 3.0


def load_prob(npz: Path, ref_img) -> np.ndarray:
    """Lesion probability on the reference grid, in nibabel (x, y, z) order."""
    p = np.load(npz)["probabilities"][1].astype(np.float32)   # nnU-Net stores (class, z, y, x)
    p = p.transpose(2, 1, 0)
    if p.shape != ref_img.shape:
        raise ValueError(f"{npz}: shape {p.shape} vs reference {ref_img.shape}")
    return p


def components(mask, vox_mm3):
    lab, n = ndimage.label(mask, CONN)
    if n == 0:
        return lab, np.array([], int)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes * vox_mm3 >= MIN_MM3) + 1
    return lab, keep


def highest_surviving_t(p, vox_mm3):
    """Highest t on a 0.01 grid at which p > t still holds a lesion of 3 mm^3 or more (0 if none at 0.01).
    Survival is monotone in t (a component at a higher t lies inside one at a lower t), so bisection is exact."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    sub = p[ndimage.find_objects((p > grid[0]).astype(np.uint8))[0]] if (p > grid[0]).any() else None
    if sub is None or len(components(sub > grid[0], vox_mm3)[1]) == 0:
        return 0.0
    lo, hi = 0, len(grid) - 1                 # grid[lo] survives; find the last index that survives
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(components(sub > grid[mid], vox_mm3)[1]) > 0:
            lo = mid
        else:
            hi = mid - 1
    return float(grid[lo])


def analyse_patient(args):
    pid, fold, npz, val_mask, gt_path, outdir = args
    ref = nib.load(str(gt_path))
    gt = np.asanyarray(ref.dataobj) > 0.5
    vox = float(np.prod(ref.header.get_zooms()[:3]))
    p = load_prob(npz, ref)

    # 1. consistency with nnU-Net's own validation mask
    val = np.asanyarray(nib.load(str(val_mask)).dataobj) > 0.5
    seg = p > 0.5
    check = {"patient": pid, "fold": fold, "voxels_differ": int((val != seg).sum()),
             "lesions_val": len(components(val, vox)[1]), "lesions_new": len(components(seg, vox)[1])}

    # 2. lesion list at p > 0.5
    lab, keep = components(seg, vox)
    lesions = []
    for i in keep:
        m = lab == i
        com = ndimage.center_of_mass(m)
        lesions.append({"patient": pid, "fold": fold, "lesion": len(lesions) + 1,
                        "volume_mm3": float(m.sum() * vox), "conf_max": float(p[m].max()),
                        "conf_mean": float(p[m].mean()), "frac_in_gt": float(gt[m].mean()),
                        "x": round(com[0], 1), "y": round(com[1], 1), "z": round(com[2], 1)})

    # 3. threshold sweep with Anima
    sweep = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for t in THRESHOLDS:
            f = Path(td) / f"t{t}.nii.gz"
            nib.save(nib.Nifti1Image((p > t).astype(np.uint8), ref.affine, ref.header), str(f))
            s = anima_score(f, gt_path)
            s.update({"patient": pid, "fold": fold, "t": t, "n_pred": len(components(p > t, vox)[1])})
            sweep.append(s)

    score = {"patient": pid, "fold": fold, "has_lesion": bool(gt.any()),
             "score_max_lesion_conf": max([l["conf_max"] for l in lesions], default=0.0),
             "score_highest_t": highest_surviving_t(p, vox)}
    return check, lesions, sweep, score


def ece(df):
    idx = np.digitize(df.conf_max, BINS) - 1
    gap = 0.0
    for b in range(len(BINS) - 1):
        sel = idx == b
        if sel.any():
            gap += sel.sum() * abs(df.conf_max[sel].mean() - df.correct[sel].mean())
    return gap / len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.umask(0o077)
    out = ROOT / "results" / a.name
    run = Path(os.environ["nnUNet_results"]) / "Dataset201_MSSEG2" / f"{a.trainer}__{a.plans}__3d_fullres"
    splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / "Dataset201_MSSEG2/splits_final.json").read_text())
    jobs = []
    for k, s in enumerate(splits):
        for case in s["val"]:
            pid = case.replace("MSSEG2_", "")
            jobs.append((pid, k, a.probs / f"fold_{k}" / f"{case}.npz", run / f"fold_{k}/validation/{case}.nii.gz",
                         ROOT / "data/msseg2" / pid / "ground_truth.nii.gz", out))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse_patient, jobs))

    checks = pd.DataFrame([r[0] for r in res])
    lesions = pd.DataFrame([l for r in res for l in r[1]])
    sweep = pd.DataFrame([s for r in res for s in r[2]])
    pat = pd.DataFrame([r[3] for r in res])
    lesions["correct"] = lesions.frac_in_gt >= 0.10
    checks.to_csv(out / "A1_consistency.csv", index=False)
    lesions.to_csv(out / "A1_lesion_list.csv", index=False)
    sweep.to_csv(out / "A1_threshold_sweep_per_patient.csv", index=False)
    pat.to_csv(out / "A1_patient_scores.csv", index=False)

    # threshold sweep summary (official measures, as E1b)
    sweep["has_lesion"] = sweep.patient.map(dict(zip(pat.patient, pat.has_lesion)))
    rows = []
    for t, g in sweep.groupby("t"):
        les, emp = g[g.has_lesion], g[~g.has_lesion]
        rows.append({"t": t, "SensL": les.SensL.mean(), "PPVL": les.PPVL.mean(), "F1": les.F1_score.mean(),
                     "Dice": les.Dice.mean(), "false_lesions_per_empty": emp.NbTestedLesions.mean(),
                     "yes_no_accuracy": (g.has_lesion == (g.n_pred > 0)).mean()})
    sweep_sum = pd.DataFrame(rows)
    sweep_sum.to_csv(out / "A1_threshold_sweep.csv", index=False)

    # calibration of the lesion list
    rng = np.random.default_rng(0)
    pids = lesions.patient.unique()
    by_p = {q: lesions[lesions.patient == q] for q in pids}
    boots = [ece(pd.concat([by_p[q] for q in rng.choice(pids, len(pids))], ignore_index=True)) for _ in range(10_000)]
    idx = np.digitize(lesions.conf_max, BINS) - 1
    rel = [{"bin": f"{BINS[b]:.1f}-{min(BINS[b + 1], 1):.1f}", "n": int((idx == b).sum()),
            "mean_conf": float(lesions.conf_max[idx == b].mean()) if (idx == b).any() else None,
            "frac_correct": float(lesions.correct[idx == b].mean()) if (idx == b).any() else None}
           for b in range(len(BINS) - 1)]
    both = lesions.correct.nunique() == 2
    calib = {"n_lesions": len(lesions), "n_correct": int(lesions.correct.sum()), "reliability": rel,
             "ECE": ece(lesions), "ECE_ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
             "Brier": float(((lesions.conf_max - lesions.correct) ** 2).mean()),
             "AUC_conf_max": float(roc_auc_score(lesions.correct, lesions.conf_max)) if both else None,
             "AUC_conf_mean": float(roc_auc_score(lesions.correct, lesions.conf_mean)) if both else None}

    # own precision vs Anima PPVL (check of the "correct" rule)
    e1 = pd.read_csv(out / "heldout_scores.csv", dtype={"patient": str}).set_index("patient")
    own = lesions.groupby("patient").correct.mean()
    cmp = pd.DataFrame({"own": own, "anima": e1.PPVL.reindex(own.index)}).dropna()
    calib["own_precision_vs_PPVL"] = {"n_patients": len(cmp), "mean_abs_diff": float((cmp.own - cmp.anima).abs().mean()),
                                      "max_abs_diff": float((cmp.own - cmp.anima).abs().max())}

    patient = {k: float(roc_auc_score(pat.has_lesion, pat[k])) for k in ["score_max_lesion_conf", "score_highest_t"]}
    summary = {"consistency": {"patients": len(checks), "max_voxels_differ": int(checks.voxels_differ.max()),
                               "lesion_count_mismatches": int((checks.lesions_val != checks.lesions_new).sum())},
               "threshold_sweep": sweep_sum.to_dict(orient="records"), "calibration": calib,
               "patient_AUC": patient}
    (out / "A1_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    print(json.dumps(summary["consistency"]))
    print(sweep_sum.round(3).to_string(index=False))
    print(pd.DataFrame(rel).to_string(index=False))
    print({k: v for k, v in calib.items() if k != "reliability"})
    print("patient AUC", patient)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(sweep_sum.false_lesions_per_empty, sweep_sum.SensL, "o-")
    for _, r in sweep_sum.iterrows():
        ax[0].annotate(f"{r.t:g}", (r.false_lesions_per_empty, r.SensL), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
    ax[0].set(xlabel="false lesions per lesion-free patient", ylabel="mean lesion sensitivity",
              title=f"{a.name}: threshold sweep (labels = t)")
    r = pd.DataFrame(rel).dropna()
    ax[1].plot([0.5, 1], [0.5, 1], "k--", lw=1)
    ax[1].plot(r.mean_conf, r.frac_correct, "o-")
    for _, q in r.iterrows():
        ax[1].annotate(f"n={q.n}", (q.mean_conf, q.frac_correct), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax[1].set(xlabel="lesion confidence (highest p)", ylabel="fraction correct", xlim=(0.45, 1.02), ylim=(0, 1.05),
              title=f"reliability, ECE {calib['ECE']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "A1_sweep_and_reliability.png", dpi=130)


if __name__ == "__main__":
    main()
