FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake wget

RUN apt update && apt install -y zlib1g protobuf-compiler

RUN apt update && apt upgrade -y && apt install -y python3-pip ssh vim git
RUN pip3 install torch torchvision torchaudio
RUN pip3 install ray[all]==2.9.1
RUN pip3 install tensorflow==2.14.0
RUN HOROVOD_WITH_GLOO=1 pip3 install git+https://github.com/onlyrobot/horovod.git
RUN pip3 install tensorboard==2.13.0 moviepy==1.0.3
RUN pip3 install onnx==1.15.0 onnxruntime==1.16.3
RUN pip3 install gradio==4.15.0 protobuf==3.20.0

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys

RUN echo "ulimit -n 65536\nulimit -u unlimited" >> /root/.bashrc

RUN apt install curl && curl -fsSL https://code-server.dev/install.sh | sh