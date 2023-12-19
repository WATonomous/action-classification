FROM pure/python:3.8
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
	ffmpeg \
	moviepy \
	tqdm>=4.29.0 

WORKDIR /project/Video-Builder