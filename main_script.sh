#!/bin/bash
docker network create mynetwork
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ./Classification_App/Docker-info/rundocker.sh
cd "$CURRENT_DIR"
echo $CURRENT_DIR
pwd
. ./Deepstream_App/Docker-info/rundocker.sh
