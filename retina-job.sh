#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/retina-%j.out

# This script assumes that the data is already downloaded and preprocessed
# See the 3D-RetinaNet repo for instructions

# print system info
./sys-info.sh

VENV_DIR=${SLURM_TMRDIR:-./tmp}/venv

IS_COMPUTE_CANADA=$(command -v module &> /dev/null)

if $IS_COMPUTE_CANADA; then
	module load python/3
fi
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

if $IS_COMPUTE_CANADA; then
	pip3 install --no-index  torch torchvision tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
else
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
	pip3 install tensorflow tensorboard tensorboardx numpy scipy pandas matplotlib
fi

python3 3D-RetinaNet/main.py \
	./data/ \
	./output/ \
	./data/kinetics-pt/ \
	--MODE=train \
	--ARCH=resnet50 \
	--MODEL_TYPE=I3D \
	--DATASET=road \
	--TRAIN_SUBSETS=train_1 \
	--VAL_SUBSETS=val_1 \
	--SEQ_LEN=8 \
	--TEST_SEQ_LEN=8 \
	--BATCH_SIZE=1 \
	--LR=0.00245 \
	--MILESTONES=6,8 \
	--MAX_EPOCHS=10

