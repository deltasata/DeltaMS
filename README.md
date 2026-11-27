# DeltaMS: finding new multiple sclerosis lesions between two MRI scans

Multiple sclerosis (MS) is followed with repeated brain MRI. A **new** lesion between two visits is a sign of
active disease and can change treatment, but spotting a few small new spots among dozens of old ones is slow
and error-prone, even for experts. This project builds and evaluates models that take a patient's baseline and
follow-up FLAIR scans and return the new lesions: a 3D mask, a list of lesions with a confidence for each, and
a yes/no answer per patient.

It uses the **MSSEG-2** data (MICCAI 2021 challenge) and the challenge's own scoring program, so the numbers
can be compared with the published ones.

**Status (November 2026): work in progress.** Code for data preparation, scoring, the classical baseline and
the nnU-Net baseline is here; results are added as experiments finish.

## Approach

- **Official scoring.** Every prediction is scored with the challenge's evaluation tool (Anima
  `animaSegPerfAnalyzer`, parameters as in the challenge evaluation document): lesion-level F1 (main score),
  Dice, false lesions on patients without new lesions, and patient-level yes/no accuracy.
- **Honest validation.** The challenge test set is not available, so models are compared with 5-fold
  cross-validation on the 40 training patients. Each fold holds out whole scanner models, so every fold is
  tested on a scanner it never saw during training. These numbers are not directly comparable with the
  leaderboard; this is stated wherever they appear.
- **Pre-registration.** Each comparison (metric, split, seeds) is fixed before it is run, and every result is
  reported, including failed runs.
- **Baselines first.**
  1. A classical, non-learning floor: image subtraction, as used to find transients in astronomy (bias
     correction, intensity matching, a significance map of the difference image, thresholds tuned on the
     training folds only).
  2. nnU-Net v2, the standard self-configuring segmentation network. Every later model has to beat it.

## First results (cross-validation, 40 patients)

| Method | Lesion F1 | Dice | False lesions per lesion-free patient | Patient yes/no accuracy |
|---|---|---|---|---|
| Difference-imaging floor | 0.12 ± 0.04 | 0.20 ± 0.06 | 36 | 0.75 |
| nnU-Net v2 (3d_fullres, Dice loss without smoothing) | 0.65 ± 0.16 | 0.57 ± 0.11 | 0.0 | 0.90 |
| nnU-Net v2, residual encoder (ResEnc M), same loss | 0.56 ± 0.13 | 0.52 ± 0.11 | 0.0 | 0.88 |

Mean over held-out patients (29 with new lesions, 11 without); ± is the spread across the 5 folds. For
reference, the best published test-set results are lesion F1 0.54 and Dice 0.51, but on 60 different patients
that include scanners absent from training, so the numbers above are **not** a leaderboard comparison. The
nnU-Net baseline makes no false detections on lesion-free patients; its errors are missed lesions.

**The larger residual-encoder network did worse.** Paired over the 29 patients with new lesions, its lesion
F1 is 0.09 lower (95% bootstrap interval 0.01-0.20 lower), so by the rule fixed before training it is dropped
and the default network stays the baseline.

**Lesion confidences.** The network's lesion probabilities are almost all 0 or 1, so the raw value is not a
usable confidence: it says ~100% for every lesion while about three in four are real (judged by the official
scorer, lesion by lesion). A two-parameter correction (Platt scaling of the mean lesion probability), fitted on
each fold's other folds only, brings the Brier score from 0.26 to 0.15 and the calibration error from 0.26 to
0.06. Moving the probability threshold barely changes the results: missed lesions get probabilities near zero,
not just below the threshold.

**Finding so far:** with its default loss, nnU-Net learned to predict "no lesion" everywhere in every fold
after about 10-20 epochs and never recovered. The soft Dice loss with a small smoothing constant gives a
perfect score to an empty prediction on a training patch without lesions, and here such patches are
common. The built-in nnU-Net variant without smoothing trains normally.

## Repository layout

```
scripts/
  evaluate.py             official scoring with Anima (checks the grid, then scores)
  leaderboard.py          summary of the published per-patient challenge results
  inspect_data.py         per-patient checks: voxel sizes, lesion counts, scanners
  make_nnunet_dataset.py  nnU-Net dataset and the scanner-held-out 5-fold split
  floor.py, floor_cv.py   classical difference-imaging floor and its cross-validation
  train_folds.sh          trains nnU-Net folds on whichever GPUs are free
  score_nnunet_cv.py      scores the held-out predictions of a 5-fold nnU-Net run
  report_cv.py            summary: mean per patient, spread across folds
  compare_runs.py         paired comparison of two runs (bootstrap interval, pre-set decision rule)
  predict_heldout_probs.sh  re-predicts held-out folds, saving lesion probabilities
  analyse_confidence.py   threshold sweep and calibration of the raw lesion probabilities
  recalibrate.py          cross-fitted recalibration of lesion confidences
env.sh                    keeps every cache, model and temporary file inside the project folder
requirements.txt
```

## Reproducing

1. Request access to MSSEG-2 on [Shanoir](https://shanoir.irisa.fr) (OFSEP). The data are not included here
   and may not be redistributed.
2. Python 3.11 environment: install `torch` for your CUDA version, then `pip install -r requirements.txt`.
3. Anima 4.2 (Ubuntu build) from https://anima.irisa.fr, unpacked into `tools/anima/`.
4. Put the images in `data/msseg2/<patient>/` (`flair_time01_on_middle_space.nii.gz`,
   `flair_time02_on_middle_space.nii.gz`, `ground_truth.nii.gz`) and the scanner list exported from Shanoir in
   `data/raw/scanners.csv`, then:

```bash
source env.sh
python scripts/inspect_data.py
python scripts/make_nnunet_dataset.py
nnUNetv2_plan_and_preprocess -d 201 -c 3d_fullres
python scripts/floor_cv.py
scripts/train_folds.sh 3d_fullres nnUNetTrainerDiceCELoss_noSmooth nnUNetPlans "0 1 2 3 4"
python scripts/score_nnunet_cv.py --trainer nnUNetTrainerDiceCELoss_noSmooth --name E1b
python scripts/report_cv.py --scores results/E1b/heldout_scores.csv --pred "{pred_path}"
scripts/predict_heldout_probs.sh nnUNetTrainerDiceCELoss_noSmooth nnUNetPlans results/E1b/probs 0
cd scripts && python analyse_confidence.py --probs ../results/E1b/probs \
    --trainer nnUNetTrainerDiceCELoss_noSmooth --name E1b
python recalibrate.py --probs ../results/E1b/probs --name E1b
```

## Data use

Only code and group-level results are published here: no images, masks, per-patient numbers or trained
weights, as required by the data-use agreement.

## Licence

The code is released under the MIT licence (see `LICENSE`). The MSSEG-2 data are not covered by it and remain
under the OFSEP data-use agreement.

## Acknowledgements

This work was carried out in collaboration with The Observatoire Français de la Sclérose en Plaques (OFSEP),
who is supported by a grant provided by the French State and handled by the "Agence Nationale de la
Recherche," within the framework of the "Investments for the Future" program, under the reference
ANR-10-COHO-002, by the Eugène Devic EDMUS Foundation against multiple sclerosis and by the ARSEP Foundation.

Scoring uses the Anima toolbox (Empenn team, Inria / IRISA). The comparison figures come from the organisers'
published per-patient challenge results (Zenodo, doi:10.5281/zenodo.5775523, CC-BY-4.0).
