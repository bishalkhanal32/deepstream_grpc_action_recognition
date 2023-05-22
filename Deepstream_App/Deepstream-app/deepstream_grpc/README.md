This project is built upon the code provided in this link:
https://github.com/NVIDIA-AI-IOT/deepstream\_pose\_estimation
and yolo is implemented using this github source:
https://github.com/marcoslucianops/DeepStream-Yolo

build.sh script to build yolo library and deepstream app.
run.sh script to run deepstream app.

deepstream\_pose\_estimation\_app.cpp is the main for DeepStream app file.
detector.proto is the .proto file for gRPC

config.txt contains information about video source path, sink type, muxer height and width, tiler-height and width, etc.

deepstream_pose_estimation_config.txt contains configuration info for primary inference engine (TRT-Pose)

dstest2\_tracker\_config.txt contains configuration info for tracker

config\_tracker\_DeepSORT.yml contains configuration info for DeepSORT tracker

config\_tracker\_IOU.yml contains configuration info for IOU tracker

deepstream\_action.h and deepstream\_action\_config\_parse.cpp are taken from action recognition example of DeepStream to read and parse the config.txt file for deepstream\_pose\_estimation\_app.cpp to use.

pair\_graph.hpp, pose\_process.cpp, cover\_table.hpp - you can take a look at the original deepstream pose estimation github resource provided above to get more insight