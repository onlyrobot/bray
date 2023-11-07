FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake wget

RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
RUN sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit && rm cuda_11.8.0_520.61.05_linux.run

RUN apt update && apt upgrade -y && apt install -y python3-pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install ray[all]==2.8.0
RUN apt install -y git && pip3 install git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17
RUN pip3 install tensorboard==2.13.0
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1 onnxruntime-gpu==1.15.1

RUN apt update && apt install -y ssh vim git

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys
