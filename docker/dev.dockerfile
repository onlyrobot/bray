FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake wget

RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
RUN sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit && rm cuda_11.8.0_520.61.05_linux.run

RUN apt update && apt install -y zlib1g
ENV CUDNN_FILE=cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
COPY ${CUDNN_FILE} ./
RUN dpkg -i ${CUDNN_FILE} && rm ${CUDNN_FILE}
RUN cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
RUN apt update && apt install -y libcudnn8=8.9.7.29-1+cuda11.8 libcudnn8-dev=8.9.7.29-1+cuda11.8

ENV NCCL_FILE=nccl-local-repo-ubuntu2004-2.16.5-cuda11.8_1.0-1_amd64.deb
COPY ${NCCL_FILE} ./
RUN dpkg -i ${NCCL_FILE} && rm ${NCCL_FILE}
RUN cp /var/nccl-local-repo-*/nccl-local-*-keyring.gpg /usr/share/keyrings/
RUN apt update && apt install -y libnccl2=2.16.5-1+cuda11.8 libnccl-dev=2.16.5-1+cuda11.8

RUN apt update && apt upgrade -y && apt install -y python3-pip ssh vim git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install ray[all]==2.8.0
RUN pip3 install tensorflow==2.14.0
RUN HOROVOD_WITH_GLOO=1 HOROVOD_GPU_OPERATIONS=NCCL pip3 install git+https://github.com/onlyrobot/horovod.git
RUN pip3 install tensorboard==2.13.0 moviepy==1.0.3
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1 onnxruntime-gpu==1.15.1

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys

RUN echo "ulimit -n 65536\nulimit -u unlimited" >> /root/.bashrc