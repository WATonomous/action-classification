#!/bin/bash
#SBATCH --output=logs/ava-download-annotations-%j.log

DATA_DIR=$1
if [[ -z $DATA_DIR ]]; then
  echo "Please provide data directory"
  exit
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget -nc https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
wget -nc https://research.google.com/ava/download/ava_val_v2.1.csv -P  ${DATA_DIR}
wget -nc https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P  ${DATA_DIR}
wget -nc https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P  ${DATA_DIR}
wget -nc https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P  ${DATA_DIR}
wget -nc https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv -P  ${DATA_DIR}
wget -nc https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv -P  ${DATA_DIR}
wget -nc https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv -P  ${DATA_DIR}
wget -nc https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar -P  ${DATA_DIR}
tar -xvf $DATA_DIR/ava_annotations.tar
mv $DATA_DIR/ava_annotations/person_box_67091280_iou90 $DATA_DIR && rm -R $DATA_DIR/ava_annotations && rm $DATA_DIR/ava_annotations.tar
