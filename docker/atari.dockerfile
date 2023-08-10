FROM hub.fuxi.netease.com/zhiqi-gameai/bray/bray:dev

RUN apt update && apt upgrade -y
RUN apt install -y libgl1-mesa-glx
RUN pip install gym[atari]
RUN pip install gym[accept-rom-license]
RUN pip install autorom[accept-rom-license]
RUN pip install atari_py
RUN pip install opencv-python
