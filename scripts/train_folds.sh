#!/bin/bash
# Train nnU-Net folds on GPUs that are idle right now (left-over resources only).
# Usage: scripts/train_folds.sh CONFIG TRAINER PLANS "FOLDS" [MIN_FREE_MIB]
# Example: scripts/train_folds.sh 3d_fullres nnUNetTrainer nnUNetPlans "0 1 2 3 4" 14000
set -u
cd "$(dirname "$0")/.."
source env.sh
CONFIG=$1; TRAINER=$2; PLANS=$3; FOLDS=$4; MIN_FREE=${5:-14000}
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-6}   # modest CPU use per run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per data-loading worker: without it each worker starts ~129 threads
LOGDIR=results/train_logs; mkdir -p $LOGDIR; chmod 700 $LOGDIR

free_gpus() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v m=$MIN_FREE '($2-$3)>=m && $4<50 {print $1}'
}

for f in $FOLDS; do
  gpu=""
  for g in $(free_gpus); do
    # skip GPUs already given to one of our runs in this loop
    if ! grep -qx "$g" $LOGDIR/.taken.$$ 2>/dev/null; then gpu=$g; break; fi
  done
  if [ -z "$gpu" ]; then echo "fold $f: no free GPU, not started"; continue; fi
  echo $gpu >> $LOGDIR/.taken.$$
  log=$LOGDIR/${TRAINER}__${PLANS}__${CONFIG}__fold${f}.log
  CUDA_VISIBLE_DEVICES=$gpu nohup nnUNetv2_train 201 $CONFIG $f -tr $TRAINER -p $PLANS > $log 2>&1 &
  echo "fold $f -> GPU $gpu, pid $!, log $log"
  sleep 60   # let it allocate memory before choosing the next GPU
done
rm -f $LOGDIR/.taken.$$
