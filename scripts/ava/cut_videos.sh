#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/ava-cut-videos-%j.out

IN_DATA_DIR="../../../../data/ava/videos/"
OUT_DATA_DIR="../../../../data/ava/videos_15min_job/"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
