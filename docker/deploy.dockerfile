FROM ubuntu:22.04

RUN apt update && apt upgrade -y
RUN pip install ray[all]==2.5.0
RUN pip install torch==2.0.1
RUN pip install tensorboard==2.13.0