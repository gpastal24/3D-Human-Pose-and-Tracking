FROM nvcr.io/nvidia/pytorch:23.12-py3
# ARG BASE_IMAGE_2=stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
# FROM stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /workspace/torch2trt
# RUN git checkout v0.4.0
# # RUN file="$(ls )" && echo $file
# # RUN echo $(file )
RUN python3 setup.py install 

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gfortran \
        git \
        libavcodec-dev \
        libavformat-dev \
        libgl1 \
        libhdf5-dev \
        libjpeg-dev \
        libomp-dev \
        libopenblas-base \
        libopenblas-dev \
        libopenjp2-7-dev \
        libopenmpi-dev \
        libpng-dev \
        libpq-dev \
        libpython3-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtiff-dev \
        openmpi-bin \
        openmpi-common \
        pkg-config \
        python3-dev \
        python3-pip \
        unzip \
        wget \
        vim \
        yasm \
        zlib1g-dev \
   && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
 

RUN apt-get update && apt-get -y upgrade
RUN pip3 install --upgrade pip
RUN apt-get install -y  gir1.2-gst-rtsp-server-1.0 \
     sox \
     ffmpeg \
     libcairo2 \
     libcairo2-dev \
     libgirepository1.0-dev 
RUN apt install -y ubuntu-restricted-extras  

RUN apt install -y libgstreamer1.0-dev \
		   libgstreamer-plugins-base1.0-dev
WORKDIR /opt/build

RUN git clone https://github.com/opencv/opencv.git
WORKDIR /opt/build/opencv
RUN mkdir build && cd build

RUN  apt-get install -y \
    libgstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \ 
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    libgtk2.0-dev \
    gstreamer1.0-pulseaudio    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


RUN apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libssl-dev \
    libopenblas-dev    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
 
#WORKDIR /opt/build
#RUN git clone https://github.com/opencv/opencv_contrib.git
#WORKDIR /opt/build/opencv_contrib
#RUN git checkout tags/4.8.0
WORKDIR /opt/build/opencv/build
RUN git checkout tags/4.8.0
RUN apt-get install -y ffmpeg
RUN pip uninstall -y opencv
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \ 
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
#-D OPENCV_EXTRA_MODULES_PATH=/opt/build/opencv_contrib/modules \
-D PYTHON_EXECUTABLE=$(which python) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python) \
-D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..
RUN make -j 10 && make install && make clean


