cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(deepstream_pose_estimation)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(GStreamer REQUIRED)
find_package(DeepstreamAppCommon REQUIRED)
find_package(NVDS REQUIRED)
find_package(CUDA REQUIRED)
find_package(X11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON-GLIB REQUIRED json-glib-1.0)




set (CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{HOME}/.local/lib/cmake/protobuf")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{HOME}/.local/lib/cmake/grpc")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

include(${CMAKE_SOURCE_DIR}/cmake/GenGRPC.cmake)

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
find_package(Protobuf REQUIRED)
#find_package(Protobuf CONFIG REQUIRED)
set(protobuf_MODULE_COMPATIBLE TRUE)
if(protobuf_MODULE_COMPATIBLE)
	include("$ENV{HOME}/.local/lib/cmake/protobuf/protobuf-module.cmake")
endif()
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
if(CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
# gRPC
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# gRPC version of detector test
# Generate pb files
# protobuf_generate_cpp -> Saves to build/detector-api.pb.h
# grpc_generate_cpp     -> Saves to ./detector-api.grpc.pb.h
message(STATUS "Saving in directory ${CMAKE_CURRENT_SOURCE_DIR}")
protobuf_generate_cpp(PROTO_SRC PROTO_HDR 
	${CMAKE_CURRENT_SOURCE_DIR}/detector.proto)
grpc_generate_cpp(GRPC_SRC GRPC_HDR 
	${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_CURRENT_SOURCE_DIR}/detector.proto)

include_directories(
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR})






include_directories(
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${GSTREAMER_RTSPSERVER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON-GLIB_INCLUDE_DIRS}
    ${DEEPSTREAM_APP_COMMON_INCLUDE_DIRS}
    ${NVDS_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    src
)


add_executable(${PROJECT_NAME}
    ${PROTO_SRC}
    ${GRPC_SRC}
    ../../apps-common/src/deepstream_perf.c
    deepstream_action_config_parse.cpp
    deepstream_pose_estimation_app.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${GSTREAMER_RTSPSERVER_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${JSON-GLIB_LIBRARIES}
    ${NVDS_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${X11_LIBRARIES}
    m
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
)

