#!/bin/bash

IN_DATA_DIR="../../../../data/ava/videos"
OUT_DATA_DIR="../../../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

FILES=($(ls -A1 -U ${IN_DATA_DIR}/*))
if [[ "$#" < 1 ]]; then
  echo "Cutting all files"
else 
  echo "Cutting file $1"
  FILES=("${FILES[$1]}")
fi
for video in $FILES
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    echo "Calling ffmpeg to output ${out_name}"
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  else
    echo "Output ${out_name} already exists, skipping"
  fi
done
