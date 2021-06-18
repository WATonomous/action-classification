#!/bin/bash
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=0-72:00:00
#SBATCH --output=logs/%x-%j.out

# Same as retina-job.sh, except uses 2x p100 GPUs on Compute Canada

# https://stackoverflow.com/a/56991068
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

SCRIPT_DIR=`dirname "$SCRIPT_PATH"`

export BATCH_SIZE=2

"$SCRIPT_DIR/retina-job.sh"
