#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

python3 -m grpc_tools.protoc -I ./ --python_out=. --pyi_out=. --grpc_python_out=. detector.proto
python3 greeter_server.py
