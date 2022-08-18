FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade
RUN pip install \  
	numpy \
	simplejson \
	PyYAML \
	psutil \
	opencv-python \
	yacs>=0.1.6 \
	matplotlib \
	termcolor \
	pandas \
	sklearn \
	Pillow \
        av \
	ffmpeg \
	moviepy \
	'iopath<0.1.9,>=0.1.7' \
	tqdm>=4.29.0 \ 
	motmetrics \ 
	filterpy  

# separate because it doesn't wanna be part of his friends (module 'numpy' not found error)
RUN pip install lap 

WORKDIR /project/OC-SORT

