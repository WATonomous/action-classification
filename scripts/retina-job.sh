#!/bin/bash
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# This script assumes that the data is already downloaded and preprocessed
# See the 3D-RetinaNet repo for instructions

# Print system info
./sys-info.sh

BATCH_SIZE=${BATCH_SIZE:-4}

if [ -n $SLURM_JOB_ID ]; then
	IS_COMPUTE_CANADA=true
	echo "Running job on Compute Canada"
else
	IS_COMPUTE_CANADA=false
	echo "Not running job on Compute Canada"
fi

# Set up environment
if [ "$IS_COMPUTE_CANADA" = "true" ]; then
	module load python/3
fi
VENV_DIR=${SLURM_TMRDIR:-./tmp}/venv
echo "VENV_DIR: $VENV_DIR"
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

if [ "$IS_COMPUTE_CANADA" = "true" ]; then
	pip3 install --no-index  torch torchvision tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
else
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
	pip3 install tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
fi

# Evaluate default directories
if [ -z $DATA_DIR ]; then
	if [ -d "../../data/road" ]; then
		DATA_DIR="../../data/"
	else
		>&2 echo "The DATA_DIR environment variable must be set!"
		exit 1
	fi
fi

if [ -z $PT_DIR ]; then
	if [ -d "../../data/kinetics-pt" ]; then
		PT_DIR="../../data/kinetics-pt/"
	else
		>&2 echo "The PT_DIR environment variable must be set!"
		exit 1
	fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-./output/}

# Ensure directories exist
if [ ! -d "$DATA_DIR" ]; then
	>&2 echo "The DATA_DIR ($DATA_DIR) directory must be present!"
	exit 1
fi
if [ ! -d "$PT_DIR" ]; then
	>&2 echo "The PT_DIR ($PT_DIR) directory must be present!"
	exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
	>&2 echo "The OUTPUT_DIR ($OUTPUT_DIR) directory must be present!"
	exit 1
fi

# Start training
python3 3D-RetinaNet/main.py \
	$DATA_DIR \
	$OUTPUT_DIR \
	$PT_DIR \
	--MODE=train \
	--ARCH=resnet50 \
	--MODEL_TYPE=I3D \
	--DATASET=road \
	--TRAIN_SUBSETS=train_1 \
	--VAL_SUBSETS=val_1 \
	--SEQ_LEN=8 \
	--TEST_SEQ_LEN=4 \
	--BATCH_SIZE=$BATCH_SIZE \
	--LR=0.00245 \
	--MILESTONES=6,8 \
	--MAX_EPOCHS=10

