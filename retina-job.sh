#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/pytorch-test-%N-%j.out

module load python/3
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index  torch torchvision tensorboardx numpy scipy pandas matplotlib

python 3D-RetinaNet/main.py ./data/ ./output/ ./data/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=val_1 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=1 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10

