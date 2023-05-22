# cp config_tracker_IOU.yml ./build/
# cp config.txt ./build/
# cp deepstream_pose_estimation_config.txt ./build/
#cp densenet121_pose_estimation.onnx_b1_gpu0_fp16.engine ./build/
# cp dstest2_tracker_config.txt ./build/
#cp densenet121_pose_estimation.onnx ./build/
#cp densenet121_pose_estimation.onnx_b4_gpu0_fp16.engine ./build
cd ./build
./deepstream_pose_estimation -c ./../config.txt 
