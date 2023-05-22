FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y openssh-server
RUN apt-get install -y build-essential libgl1 udev
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh/
EXPOSE 22

### set the locale to en_US.UTF-8 ###
RUN apt-get install -y locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && dpkg-reconfigure locales \
    && /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LC_ALL en_US.UTF-8

### Install apt and python packages ###
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -U numpy
RUN apt-get install -y libjpeg-dev zlib1g-dev
RUN pip3 install Pillow tqdm
RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get install -y libopencv-dev nano tmux git python3-tk 
RUN pip3 install -U scikit-image
RUN pip3 install -U cython 
RUN pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
RUN pip3 install filterpy==1.1.0
RUN pip3 install pyinstaller[encryption]
RUN pip3 install easyroi --no-dependencies
RUN pip3 install shapely

### install requirements ###
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8000



