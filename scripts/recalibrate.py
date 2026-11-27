"""Cross-fitted recalibration of lesion confidences (DECISIONS.md, node C1).

1. Lesion table: for every held-out patient, lesions = 26-connected components of 3 mm^3 or more of p > 0.5.
   Features: highest p, mean p, volume. Label: Anima's verdict on the lesion scored alone (official settings);
   this reproduces Anima's per-patient PPVL exactly. Cached in results/<name>/C1_lesions.csv.
2. Cross-fitting: for each fold k, the map is fitted on the lesions of the other folds and applied to fold k.
   M0 = raw highest p; M1 = Platt scaling on logit(mean p); M2 = logistic regression on logit(mean p), log10 volume.
3. Brier score (primary) with paired patient-bootstrap intervals, ECE, log loss, AUC, reliability table,
   patient-level scores, and the pre-registered decision.

Usage: python scripts/recalibrate.py --probs results/E1b/probs --name E1b
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import anima_score

ROOT = Path(__file__).resolve().parents[1]
CONN = np.ones((3, 3, 3), bool)
MIN_MM3 = 3.0
EPS = 1e-6
MODELS = {"M1": ["logit_mean_p"], "M2": ["logit_mean_p", "log10_volume"]}


def lesion_rows(args):
    pid, fold, npz, gt_path = args
    ref = nib.load(str(gt_path))
    gt = np.asanyarray(ref.dataobj) > 0.5
    vox = float(np.prod(ref.header.get_zooms()[:3]))
    p = np.load(npz)["probabilities"][1].astype(np.float32).transpose(2, 1, 0)   # nnU-Net: (class, z, y, x)
    if p.shape != gt.shape:
        raise ValueError(f"{npz}: shape {p.shape} vs {gt.shape}")
    lab, n = ndimage.label(p > 0.5, CONN)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes * vox >= MIN_MM3) + 1
    rows = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for j, i in enumerate(keep):
            m = lab == i
            if gt.any():
                f = Path(td) / "lesion.nii.gz"
                nib.save(nib.Nifti1Image(m.astype(np.uint8), ref.affine, ref.header), str(f))
                correct = int(anima_score(f, gt_path)["PPVL"] > 0.5)
            else:
                correct = 0
            rows.append({"patient": pid, "fold": fold, "lesion": j + 1, "volume_mm3": float(m.sum() * vox),
                         "max_p": float(p[m].max()), "mean_p": float(p[m].mean()), "correct": correct})
    return rows, {"patient": pid, "fold": fold, "has_lesion": bool(gt.any())}


def ece(conf, y, bins=10):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(conf[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
               for b in range(bins) if (idx == b).any()) / len(y)


def patient_scores(les, pats, col):
    s = les.groupby("patient")[col].apply(lambda c: 1 - np.prod(1 - c.values))
    return pats.patient.map(s).fillna(0.0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.umask(0o077)
    out = ROOT / "results" / a.name
    cache = out / "C1_lesions.csv"
    splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / "Dataset201_MSSEG2/splits_final.json").read_text())
    jobs = [(c.replace("MSSEG2_", ""), k, a.probs / f"fold_{k}/{c}.npz",
             ROOT / "data/msseg2" / c.replace("MSSEG2_", "") / "ground_truth.nii.gz")
            for k, s in enumerate(splits) for c in s["val"]]
    if cache.exists():
        les = pd.read_csv(cache, dtype={"patient": str})
        pats = pd.read_csv(out / "C1_patients.csv", dtype={"patient": str})
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(lesion_rows, jobs))
        les = pd.DataFrame([r for x in res for r in x[0]])
        pats = pd.DataFrame([x[1] for x in res])
        les.to_csv(cache, index=False)
        pats.to_csv(out / "C1_patients.csv", index=False)

    # check: per-lesion labels reproduce Anima's per-patient PPVL
    e = pd.read_csv(out / "heldout_scores.csv", dtype={"patient": str}).set_index("patient")
    own = les[les.patient.isin(e.index[e.PPVL.notna()])].groupby("patient").correct.mean()
    ppvl_mismatch = int(((own - e.PPVL.reindex(own.index)).abs() > 1e-6).sum())

    lp = np.clip(les.mean_p, EPS, 1 - EPS)
    les["logit_mean_p"] = np.log(lp / (1 - lp))
    les["log10_volume"] = np.log10(les.volume_mm3)
    les["M0"] = les.max_p
    for name, feats in MODELS.items():
        les[name] = np.nan
        for k in sorted(les.fold.unique()):
            tr, te = les.fold != k, les.fold == k
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs"))
            clf.fit(les.loc[tr, feats], les.loc[tr, "correct"])
            les.loc[te, name] = clf.predict_proba(les.loc[te, feats])[:, 1]
    les.to_csv(out / "C1_lesions_scored.csv", index=False)

    y = les.correct.values
    summary = {"n_lesions": len(les), "n_correct": int(y.sum()), "ppvl_mismatches": ppvl_mismatch, "models": {}}
    pats_y = pats.has_lesion.values
    for m in ["M0", "M1", "M2"]:
        c = les[m].values
        ps = patient_scores(les, pats, m)
        idx = np.minimum((c * 10).astype(int), 9)
        summary["models"][m] = {
            "Brier": float(((c - y) ** 2).mean()), "ECE": float(ece(c, y)),
            "log_loss": float(log_loss(y, np.clip(c, EPS, 1 - EPS))), "AUC": float(roc_auc_score(y, c)),
            "mean_conf": float(c.mean()),
            "reliability": [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((idx == b).sum()),
                             "mean_conf": float(c[idx == b].mean()), "frac_correct": float(y[idx == b].mean())}
                            for b in range(10) if (idx == b).any()],
            "patient_Brier": float(((ps - pats_y) ** 2).mean()), "patient_AUC": float(roc_auc_score(pats_y, ps))}

    # paired patient bootstrap of per-lesion squared-error differences
    rng = np.random.default_rng(0)
    pids = les.patient.unique()
    groups = {q: np.flatnonzero(les.patient.values == q) for q in pids}
    draws = [np.concatenate([groups[q] for q in rng.choice(pids, len(pids))]) for _ in range(10_000)]
    se = {m: (les[m].values - y) ** 2 for m in ["M0", "M1", "M2"]}
    comps = {}
    for new, ref in [("M1", "M0"), ("M2", "M0"), ("M2", "M1")]:
        d = se[new] - se[ref]
        boots = np.array([d[i].mean() for i in draws])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        comps[f"{new}-{ref}"] = {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "better": bool(hi < 0)}
    summary["comparisons"] = comps

    if comps["M1-M0"]["better"]:
        choice = "M2" if comps["M2-M1"]["better"] and comps["M2-M0"]["better"] else "M1"
    else:
        choice = "M2" if comps["M2-M0"]["better"] and comps["M2-M1"]["better"] else "M0"
    summary["decision"] = choice
    (out / "C1_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"lesions {len(les)}, correct {int(y.sum())}, PPVL mismatches {ppvl_mismatch}")
    for m, r in summary["models"].items():
        print(f"{m}: Brier {r['Brier']:.3f}  ECE {r['ECE']:.3f}  logloss {r['log_loss']:.3f}  AUC {r['AUC']:.3f}  "
              f"mean conf {r['mean_conf']:.3f}  patient Brier {r['patient_Brier']:.3f} AUC {r['patient_AUC']:.3f}")
        print("   ", "; ".join(f"{b['bin']}: n={b['n']} conf {b['mean_conf']:.2f} correct {b['frac_correct']:.2f}"
                              for b in r["reliability"]))
    for k, v in comps.items():
        print(f"{k}: {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print("decision:", choice)


if __name__ == "__main__":
    main()
