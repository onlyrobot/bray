FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake wget

RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
RUN sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit && rm cuda_11.7.0_515.43.04_linux.run

RUN apt install -y python3-pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install ray[all]==2.6.1
RUN HOROVOD_WITH_GLOO=1 HOROVOD_WITH_TORCH=1 pip install horovod[ray]
RUN pip install tensorboard==2.13.0
RUN pip install onnx==1.14.0 onnxruntime==1.15.1

RUN apt update && apt install -y ssh vim git

RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys
