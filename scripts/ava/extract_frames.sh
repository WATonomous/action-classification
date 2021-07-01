#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=100MB
#SBATCH --time=0-00:10:00
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

extract_func() {
  video=$1
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  if [ ! -d "${out_video_dir}" ]; then
    mkdir -p "${out_video_dir}"
    out_name="${out_video_dir}/${video_name}_%06d.jpg"
    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
  fi
}
export -f extract_func

ls -A1 -U ${IN_DATA_DIR}/* | parallel --jobs $JOBS --env OUT_DATA_DIR extract_func
exit

FILES=($(ls -A1 -U ${IN_DATA_DIR}/*))
if [[ "$#" < 1 ]]; then
  echo "Extracting frames for all files"
else 
  echo "Extracting frame for file $1"
  FILES=("${FILES[$1]}")
fi

for video in $FILES
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
