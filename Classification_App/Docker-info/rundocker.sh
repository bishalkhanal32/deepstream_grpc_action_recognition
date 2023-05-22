#!/bin/bash

NAME=classification_app
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"
USER=$USER
sudo docker build -t $NAME .

xhost +

sudo docker run --name $NAME --network mynetwork --gpus all --shm-size=16g -it -d -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/$USER/Documents/ec2_deepstream_grpc/Classification_App/pyskl:/home/Documents/classification_app/pyskl -w /home/Documents/classification_app/pyskl $NAME /home/Documents/classification_app/pyskl/main_run.sh
