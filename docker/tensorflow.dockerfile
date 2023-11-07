FROM hub.fuxi.netease.com/zhiqi-gameai/bray/bray:dev

RUN pip3 install tensorflow==2.14.0
RUN pip uninstall -y horovod && pip3 install git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17