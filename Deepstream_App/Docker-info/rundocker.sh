#!/bin/bash

NAME=deepstream_handover_test
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"


FOLDER=Deepstream-app

SYSTEM_BASE=/home/$USER/Documents/ec2_deepstream_grpc/Deepstream_App/Deepstream-app
DOCKER_BASE=/opt/nvidia/deepstream/deepstream/sources/apps

xhost +

docker build -t $NAME .

docker run --name $NAME --network mynetwork --gpus all -it -d -v /tmp/.X11-unix:/tmp/.X11-unix --volume $SYSTEM_BASE:$DOCKER_BASE/$FOLDER -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream $NAME
