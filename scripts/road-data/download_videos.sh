#!/bin/bash

# Derived from https://github.com/gurkirt/road-dataset/blob/5eee122c42830807ff6bd7cb12d0252b63ece0bc/road/get_dataset.sh

DOWNLOAD_DIR="$1"

if [ -z "$DOWNLOAD_DIR" ]; then
	>&2 echo "Usage: $0 <download_dir>"
	exit 1
fi

set -x

pushd $DOWNLOAD_DIR

# Download the videos zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz" -O videos.zip

# Unzip the videos
unzip videos.zip

#Download annotation file for training and validation in json format
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HAJpdS76TVK56Qvq1jXr-5hfFCXKHRZo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HAJpdS76TVK56Qvq1jXr-5hfFCXKHRZo" -O road_trainval_v1.0.json 

#Download the instance counts file in json format
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V" -O instance_counts.json 

popd

set +x
