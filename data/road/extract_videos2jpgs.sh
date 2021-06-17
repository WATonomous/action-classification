#!/bin/bash
#SBATCH --time=0-0:20
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x-%j.out

# Script to extract videos into jpgs on Compute Canada

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

python "$SCRIPT_DIR/extract_videos2jpgs.py" "$SCRIPT_DIR"
