#!/bin/bash

DATA_DIR=$1
if [[ -z $DATA_DIR ]]; then
  echo "Please provide data directory"
  exit
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt
cat ava_file_names_trainval_v2.1.txt | parallel wget https://s3.amazonaws.com/ava-dataset/trainval/{} -P ${DATA_DIR} 
rm ava_file_names_trainval_v2.1.txt
