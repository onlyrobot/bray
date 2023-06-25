FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake
RUN apt install -y python3-pip
RUN pip install ray[all]==2.5.0
RUN pip install torch==2.0.1
RUN pip install tensorboard==2.13.0
RUN HOROVOD_WITH_GLOO=1 HOROVOD_WITH_TORCH=1 pip install horovod[ray]
