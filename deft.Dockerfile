FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6

# setup conda environment
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH="$CONDA_DIR/bin:$PATH"

COPY DEFT/environment.yml project/DEFT
RUN conda env create -f project/DEFT/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install \  
	opencv-python \
    Cython \
    matplotlib \
    scipy \
    numba \
    easydict \
    pyquaternion \
    nuscenes-devkit \
    pyyaml \
    motmetrics \
    scikit-learn==0.22.2 \
    pandas==0.22.0 \
    Pillow==4.3.0 \
    lap \
    cython-bbox 

RUN pip install --pre 'torch==1.10.0.dev20210629+cu111' torchvision pytorchvideo -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

RUN pip install \
	'git+https://github.com/facebookresearch/fvcore' \
	'git+https://github.com/facebookresearch/detectron2.git'
ADD SlowFast ./SlowFast
RUN export PYTHONPATH=/project/SlowFast/slowfast:$PYTHONPATH && \
    cd SlowFast && \
    python setup.py build develop

WORKDIR /project/SlowFast
