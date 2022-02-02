#!/bin/bash

VENV_DIR=$USER-slowfast-venv
module load python/3
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate
pip install --no-index \
	numpy \
	torch \
	torchvision \
	simplejson \
	PyYAML \
	psutil \
	opencv-python \
	tensorboard \
	yacs>=0.1.6 \
	matplotlib \
	termcolor \
	opencv_python \
	pandas \
	sklearn \
	Pillow \


pip install \
	av \
	ffmpeg \
	moviepy \
	pytorchvideo \
	'iopath>=0.1.7' \
	'tqdm==4.11.2' \
	'git+https://github.com/facebookresearch/fvcore' \
	'git+https://github.com/facebookresearch/detectron2.git' \

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH=$SCRIPT_DIR/SlowFast/slowfast:$PYTHONPATH
cd SlowFast
python setup.py build develop

deactivate
