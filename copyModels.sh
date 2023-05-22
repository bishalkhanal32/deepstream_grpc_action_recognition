#!/bin/bash
cp ./models/pc3d_gcn_tcn_2_2_1_aitdataset.pth ./Classification_App/pyskl/gRPC/

cp -r ./models/test-videos/ ./Deepstream_App/Deepstream-app/

cp ./models/mars-small128.uff ./Deepstream_App/Deepstream-app/deepsort/

cp ./models/densenet121_pose_estimation.onnx ./Deepstream_App/Deepstream-app/deepstream_grpc/
cp ./models/densenet_224x224.onnx ./Deepstream_App/Deepstream-app/deepstream_grpc/
cp ./models/resnet18_pose_estimation.onnx ./Deepstream_App/Deepstream-app/deepstream_grpc/


cp ./models/vitpose-dynamic-b-simple.onnx ./Deepstream_App/Deepstream-app/deepstream_yolo_vitpose_grpc/
cp ./models/yolov6s.wts ./Deepstream_App/Deepstream-app/deepstream_yolo_vitpose_grpc/


