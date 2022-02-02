#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --array=0-300
#SBATCH --output=logs/ava-cut-video-%j.log
./extract_frames.sh $SLURM_ARRAY_TASK_ID
