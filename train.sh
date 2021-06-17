#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/slowfast-%j.out

VENV_DIR=$USER-slowfast-venv
module load python/3
source $VENV_DIR/bin/activate

cd SlowFast
python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 DATA.PATH_TO_DATA_DIR ../../data/

deactivate
