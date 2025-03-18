FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y sudo \
                       python-is-python3 \
                       python3-pip

RUN mkdir -p /usr/share/fonts
COPY times.ttf /usr/share/fonts



RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install torch-geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch-geometric

# Install other python libraries
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
# RUN pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

##
# Set User
##
ENV USER ***
ENV UID ***
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${UID} ${USER}
RUN useradd -g ${UID} -u ${UID} -m -s /bin/bash ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:***" | chpasswd

USER ${USER}

WORKDIR /home/${USER}/

RUN echo "alias python=python3" >> ~/.bashrc
RUN alias python=python3
