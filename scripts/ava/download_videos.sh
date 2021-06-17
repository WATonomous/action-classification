#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --array=0-300
#SBATCH --output=logs/ava-download-videos-%j.log

DATA_DIR="../../../../data/ava/videos/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt
LINKS=($(cat ava_file_names_trainval_v2.1.txt))
if [ -n $SLURM_ARRAY_TASK_ID ]; then
  LINK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ava_file_names_trainval_v2.1.txt)
fi

for line in $LINKS
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
rm ava_file_names_trainval_v2.1.txt*
