FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install \  
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
        av \
	ffmpeg \
	moviepy \
	pytorchvideo \
	'iopath>=0.1.7' \
	'tqdm>=4.29.0'

RUN pip install \
	'git+https://github.com/facebookresearch/fvcore' \
	'git+https://github.com/facebookresearch/detectron2.git'
ADD SlowFast ./SlowFast
RUN export PYTHONPATH=/project/SlowFast/slowfast:$PYTHONPATH && \
    cd SlowFast && \
    python setup.py build develop

WORKDIR /project/SlowFast
