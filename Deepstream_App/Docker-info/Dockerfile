FROM nvcr.io/nvidia/deepstream:6.2-devel

LABEL maintainer="AIT AI CENTER"

# Set Debian_frontend to avoid user interaction with tzdata
ARG DEBIAN_FRONTEND=noninteractive
ENV LIBS_INSTALL_DIR /libs_installed

RUN apt install -y wget


RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev libmysql++-dev vlc libvlc-dev libvlccore-dev \
    git zlib1g-dev make g++ build-essential zip unzip \
    pkg-config autoconf automake libtool uuid-dev libgtest-dev \
    iputils-tracepath libprotobuf-dev libleveldb-dev \
    libhdf5-serial-dev protobuf-compiler libgflags-dev \
    && apt-get autoremove -y \
    && apt-get clean

RUN apt-get install -y --no-install-recommends \
    qt5-default libvtk6-dev libjpeg-dev libwebp-dev libpng-dev \
    libtiff5-dev libopenexr-dev libgdal-dev libdc1394-22-dev \
    libavcodec-dev libavformat-dev libswscale-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libx264-dev yasm libxine2-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev \
    libtbb-dev libeigen3-dev flake8 default-jdk doxygen \
    libharfbuzz-dev xfonts-thai \
    && apt-get autoremove -y \
    && apt-get clean


# Set up timezone
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# CMake version 3.17
WORKDIR $LIBS_INSTALL_DIR
RUN apt-get remove cmake -y \
    && wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz \
    && tar -xzvf cmake-3.17.3.tar.gz \
    && cd cmake-3.17.3/ \
    && ./bootstrap \
    && make -j8 \
    && make install \
    && hash -r \
    && rm -rf $LIBS_INSTALL_DIR/cmake-3.17.3.tar.gz \
    && rm -rf $LIBS_INSTALL_DIR/cmake-3.17.3

RUN wget https://curl.se/download/curl-7.81.0.tar.gz && \
    tar xzf curl-7.81.0.tar.gz && \
    cd curl-7.81.0 && \
    ./configure --with-openssl && \
    make && \
    make install && \
    cd .. && \
    rm -rf curl-7.81.0.tar.gz curl-7.81.0

RUN wget https://www.cryptopp.com/cryptopp870.zip ; \
    mkdir cryptopp-src ; \
    cd cryptopp-src ; \
    unzip ../cryptopp870.zip ; \
    make ; \
    make install ; \
    cd .. ; \
    rm -rf cryptopp870.zip cryptopp-src



# Install gRPC
WORKDIR	/libs_installed
RUN 	export MY_INSTALL_DIR=/root/.local \
	&& mkdir -p $MY_INSTALL_DIR \
	&& export PATH="$MY_INSTALL_DIR/bin:$PATH" \
	&& git clone --recurse-submodules -b v1.38.0 https://github.com/grpc/grpc \
	&& cd /libs_installed/grpc \
	&& mkdir -p cmake \
	&& mkdir -p cmake/build \
	&& cd cmake/build \
	&& cmake -DgRPC_INSTALL=ON \
	      -DgRPC_BUILD_TESTS=OFF \
	      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
	      ../.. \
	&& make -j4 \
	&& make install \
	&& cd ../.. \
	&& mkdir -p third_party/abseil-cpp/cmake \
	&& mkdir -p third_party/abseil-cpp/cmake/build \
	&& cd third_party/abseil-cpp/cmake/build \
	&& cmake -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
	      -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
	      ../.. \
	&& make -j4 \
	&& make install \
        && rm -rf /libs_installed/grpc 

RUN 	apt -y install mlocate \
	&& updatedb





