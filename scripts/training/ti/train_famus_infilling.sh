#!/bin/bash

# Various environment variables that you may or
# may not need to set depending on your setup
export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
export PYTHONPATH="$PYTHONPATH:."
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_DEVICES="[0]"

# SET ME!
# PROJECT_ROOT=/path/to/eae-transfer/
PROJECT_ROOT=/brtx/603-nvme1/wgantt/eae-transfer

# the model
MODEL=google/flan-t5-base

# the model archive
OUTPUT_DIR=$1

# the random seed
# (we use 1337, 1338, and 1339 in our experiments)
SEED=$2

# ***NOTE***: activate your virtual environment here!
# e.g. `conda activate <your_environment>

cd $PROJECT_ROOT
python -m eae.training.train_infilling $OUTPUT_DIR famus_reports famus_reports famus_reports \
	--model $MODEL \
	--seed $SEED \
	--num-epochs 50 \
	--per-device-batch-size 8 \
	--gradient-checkpointing \
