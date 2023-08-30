FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake wget
RUN apt install -y python3-pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install ray[all]==2.6.1
RUN pip3 install tensorboard==2.13.0
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.1

RUN apt update && apt install -y ssh vim git

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys