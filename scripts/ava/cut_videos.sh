#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/ava-cut-videos-%j.out

IN_DATA_DIR=$1
OUT_DATA_DIR=$2
JOBS=$3
export OUT_DATA_DIR

if [[ -z $IN_DATA_DIR || -z $OUT_DATA_DIR || -z $JOBS ]]; then
  echo Specify in data dir first, then out data dir, then number of parallel ffmeg jobs
  exit
fi

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

cut_func() {
  out_name="${OUT_DATA_DIR}/${1##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${1}" "${out_name}"
  fi
}
export -f cut_func

ls -A1 -U ${IN_DATA_DIR}/* | parallel --jobs $JOBS --env OUT_DATA_DIR cut_func
