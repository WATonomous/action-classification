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
	tensorboard \
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
	tqdm>=4.29.0

# RUN pip install --pre 'torch==1.10.0.dev20210921+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install pytorchvideo
RUN pip install torch pytorchvideo torchvision
# RUN pip install --pre 'torchvision==0.11.0.dev2021092+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 pytorchvideo -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install easydict tensorboardx

