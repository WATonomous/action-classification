#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/%x-%j.out

# Same as retina-job.sh, except with fewer resources

# https://stackoverflow.com/a/56991068
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

SCRIPT_DIR=`dirname "$SCRIPT_PATH"`

export BATCH_SIZE=1

"$SCRIPT_DIR/retina-job.sh"
