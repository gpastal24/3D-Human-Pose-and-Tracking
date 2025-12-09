FROM indux-r.iti.gr:5050/certh/t2.3_human_pose_estimation:base

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir
WORKDIR /workspace
RUN wget https://download.stereolabs.com/zedsdk/4.0/cu121/ubuntu22 && apt install zstd && \
 chmod +x ubuntu22 && ./ubuntu22 --noexec --target installer_files && rm ubuntu22 \
 && cd /workspace/installer_files  &&  \
 ./linux_install_release.sh --silent && cd /workspace && \
 rm -r installer_files
RUN pip uninstall numpy -y
RUN pip uninstall numpy -y

RUN pip install numpy==1.24.4
WORKDIR /app
COPY . ./

RUN pip install darlene_trackrpn-1.0.0-py3-none-any.whl
RUN ./download_weights.sh
# RUN ./generate_trt_weights.sh
#RUN rm -r installer_files ubuntu20

