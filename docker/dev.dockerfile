FROM ubuntu:22.04

RUN apt update && apt upgrade -y && apt install -y build-essential cmake
RUN pip install ray[all]==2.5.0
RUN pip install torch==2.0.1 torchvision==0.15.2
RUN pip install tensorboard==2.13.0
RUN pip install HOROVOD_WITH_GLOO=1 ... pip install horovod[ray]