FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

# cv2 dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install \  
	numpy \
	simplejson \
	PyYAML \
    easydict \
	psutil \
	opencv-python \
	matplotlib \
	termcolor \
	Pillow \
    av \
	moviepy \
	tqdm>=4.29.0 \
    scipy \
	pandas

RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 pytorchvideo -f https://download.pytorch.org/whl/cu111/torch_stable.html

WORKDIR /project/Post-Processing