#!/bin/bash
#SBATCH --time=0-0:10:00
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=1G
#SBATCH --output=logs/%x-%A-%a.out

# Script to extract videos into jpgs
# usage example (Compute Canada):
# VIDEO_DIR=../../data/road/videos/ IMAGE_DIR=../../data/road/rgb-images sbatch --array=1-18 ./scripts/road-data/extract_videos2jpgs.sh

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
# gather system info
$SCRIPT_DIR/../sys-info.sh
# ------------------------------------------------------------------------------------

if [ ! -d "$VIDEO_DIR" ]; then
	echo "VIDEO_DIR ($VIDEO_DIR) is not a directory!"
	exit 1
fi
if [ -z "$IMAGE_DIR" ]; then
	echo "IMAGE_DIR ($IMAGE_DIR) is not specified!"
	exit 1
fi

FILES="$(ls -A1 ${VIDEO_DIR}/*.mp4)"
printf "All files:\n$FILES\n"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  FILES=`echo "$FILES" | sed "${SLURM_ARRAY_TASK_ID}p;d"`
fi

printf "Files to process:\n$FILES\n"

python "$SCRIPT_DIR/extract_videos2jpgs.py" --outdir $IMAGE_DIR --files $FILES

