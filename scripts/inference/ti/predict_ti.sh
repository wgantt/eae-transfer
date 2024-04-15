#!/bin/bash

# Environment variables that you may or
# may not want to set depending on your setup
export CONDAROOT=/path/to/anaconda/
export PATH=$CONDAROOT/condabin:$PATH
export PYTHONPATH="$PYTHONPATH:."
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_DEVICES="[0]"

# SET ME!
PROJECT_ROOT=/path/to/eae-transfer/

# the model archive
MODEL_ARCHIVE=$1

# where the results (predictions, metrics)
# will be written to
RESULTS_DIR=$2

# the random seed
SEED=$3

# ***NOTE***: Activate your virtual environment here!
# e.g. `conda activate <your_environment>`

cd $PROJECT_ROOT
python -m eae.inference.inference_infilling famus_reports+rams+wikievents $MODEL_ARCHIVE \
	--results-dir $RESULTS_DIR \
	--device 0 \
	--batch-size 4 \
	--verbose-metrics \
	--seed $SEED
