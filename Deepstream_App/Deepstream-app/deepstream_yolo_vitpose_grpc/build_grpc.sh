echo "compiling nvdsinfer_custom_impl_Yolo ..."
cd nvdsinfer_custom_impl_Yolo
make
echo "complete!!"

cd ..
echo "building gRPC and DeepStream app"
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..
