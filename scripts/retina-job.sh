#!/bin/bash
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-23:00:00
#SBATCH --output=logs/%x-%j.out

# This script assumes that the data is already downloaded and preprocessed
# See the 3D-RetinaNet repo for instructions

# Common header to get the directory of the script -----------------------------------
if [ -z "$SCRIPT_DIR" ] && [ -n "$SLURM_JOB_ID" ]; then
	# No other script has set SCRIPT_DIR and we are on slurm (i.e. we are
	# the entry script on slurm).
	# Check the original location through scontrol and $SLURM_JOB_ID
	SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
        SCRIPT_DIR=`dirname "$SCRIPT_PATH"`
else
        # Some other job already set SCRIPT_DIR or we are not on slurm.
	# Set the script directory normally.
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
fi
# print system info
$SCRIPT_DIR/sys-info.sh
# ------------------------------------------------------------------------------------

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
TMPDIR="${SLURM_TMPDIR:-./tmp}"
VENV_DIR="$TMPDIR/venv"
echo "VENV_DIR: $VENV_DIR"
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

if [ "$IS_COMPUTE_CANADA" = "true" ]; then
	pip3 install --no-index  torch torchvision tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
else
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
	pip3 install tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
fi

# if DATA_DIR and DATA_TAR are unset, auto-discover
if [ -z "$DATA_DIR" ] && [ -z "$DATA_TAR" ]; then
	if [ -d "../../data/road" ]; then
		DATA_DIR="../../data/"
		echo "Using DATA_DIR=$DATA_DIR"
	elif [ -f "../../data/road.tar" ]; then
		DATA_TAR="../../data/road.tar"
	else
		>&2 echo "The DATA_DIR or DATA_TAR environment variable must be set!"
		exit 1
	fi
fi

# if DATA_DIR doesn't exist and DATA_TAR does, extract DATA_TAR
if [ -z "$DATA_DIR" ] && [ -n "$DATA_TAR" ]; then
		echo "Extracting DATA_TAR $DATA_TAR"
		tar -xf $DATA_TAR -C "$TMPDIR"
		DATA_DIR="$TMPDIR/"
fi

# if PT_DIR is unset, auto-discover
if [ -z "$PT_DIR" ]; then
	if [ -d "../../data/kinetics-pt" ]; then
		PT_DIR="../../data/kinetics-pt/"
	else
		>&2 echo "The PT_DIR environment variable must be set!"
		exit 1
	fi
fi

if [ "$IS_COMPUTE_CANADA" = "true" ]; then
	OUTPUT_DIR=${OUTPUT_DIR:-./output/`date +"%FT%H%M%z"`-$SLURM_JOB_ID-${SLURM_JOB_NAME%.*}}
else
	OUTPUT_DIR=${OUTPUT_DIR:-./output/`date +"%FT%H%M%z"`}
fi

mkdir -p $OUTPUT_DIR

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

