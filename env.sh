# Source this before any work: keeps every cache and install inside DeltaMS.
export DELTAMS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the folder holding this file
export PIP_CACHE_DIR=$DELTAMS/.cache/pip
export CONDA_PKGS_DIRS=$DELTAMS/.cache/conda_pkgs
export XDG_CACHE_HOME=$DELTAMS/.cache/xdg
export TORCH_HOME=$DELTAMS/.cache/torch
export MPLCONFIGDIR=$DELTAMS/.cache/mpl
export nnUNet_raw=$DELTAMS/nnUNet_raw
export nnUNet_preprocessed=$DELTAMS/nnUNet_preprocessed
export nnUNet_results=$DELTAMS/nnUNet_results
export PYTHONNOUSERSITE=1
unset PYTHONPATH  # use only packages from the project environment
if [ -x $DELTAMS/envs/deltams/bin/python ]; then export PATH=$DELTAMS/envs/deltams/bin:$PATH; fi
export TMPDIR=$DELTAMS/.cache/tmp            # nothing written to /tmp
export TORCHINDUCTOR_CACHE_DIR=$DELTAMS/.cache/torchinductor
export TRITON_CACHE_DIR=$DELTAMS/.cache/triton
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
