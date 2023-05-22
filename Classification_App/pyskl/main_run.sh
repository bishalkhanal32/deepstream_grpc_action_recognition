#!/bin/bash
cd "`dirname $0`"
pip3 install -e .
chmod +x ./gRPC/run.sh
. ./gRPC/run.sh

