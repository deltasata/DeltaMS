#!/bin/bash
# Re-predict each fold's held-out patients with that fold's final checkpoint, saving lesion probabilities.
# Same settings as nnU-Net's end-of-training validation (default mirroring, step 0.5).
# Usage: scripts/predict_heldout_probs.sh TRAINER PLANS OUTDIR GPU
set -eu
cd "$(dirname "$0")/.."
source env.sh
TRAINER=$1; PLANS=$2; OUT=$3; GPU=$4
umask 077
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPLITS=$nnUNet_preprocessed/Dataset201_MSSEG2/splits_final.json
for k in 0 1 2 3 4; do
  IN=$OUT/input_fold$k; mkdir -p $IN $OUT/fold_$k
  for c in $(python -c "import json;print(' '.join(json.load(open('$SPLITS'))[$k]['val']))"); do
    for ch in 0000 0001; do ln -sf $nnUNet_raw/Dataset201_MSSEG2/imagesTr/${c}_$ch.nii.gz $IN/; done
  done
  CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i $IN -o $OUT/fold_$k -d 201 -c 3d_fullres -tr $TRAINER -p $PLANS \
    -f $k -chk checkpoint_final.pth --save_probabilities -npp 2 -nps 2 --disable_progress_bar
done
echo done
