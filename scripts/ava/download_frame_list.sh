#!/bin/bash
#SBATCH --output=logs/ava-download-frame-list-%j.log

DATA_DIR=$1
if [[ -z $DATA_DIR ]]; then
  echo "Please provide data directory"
  exit
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv -P ${DATA_DIR}
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv -P ${DATA_DIR}
