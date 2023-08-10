FROM hub.fuxi.netease.com/zhiqi-gameai/bray/bray:dev

# pod ssh配置
RUN apt-get update && apt-get install -y openssh-server
RUN wget -P /etc/init.d/ https://dlonline.nos-jd.163yun.com/images-deps/ssh/sync-sshkey
RUN wget -P /usr/local/bin/ https://dlonline.nos-jd.163yun.com/images-deps/ssh/sync-sshkey.sh
RUN chmod +x /etc/init.d/sync-sshkey \
    && chmod +x /usr/local/bin/sync-sshkey.sh

# 配置猛犸数据源
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN wget https://dlonline.nos-jd.163yun.com/images-deps/hadoop/hadoop.tar
RUN tar xvf hadoop.tar && mv hadoop/hadoop_client /opt/ && rm -rf hadoop.tar
RUN wget -P /opt/ https://dlonline.nos-jd.163yun.com/images-deps/hadoop/hdfs_startup.sh
RUN wget -P /opt/ https://dlonline.nos-jd.163yun.com/ssh/env.sh
RUN cat /opt/env.sh >> ~/.bashrc
# 安装krb5
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages --fix-missing \
        build-essential \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libkrb5-dev
        
# 安装其他自定义模块
# ...
# 保持容器一直处于Running状态
RUN rm -rf /etc/ssh/sshd_config
RUN wget -P /etc/ssh/ https://dlonline.nos-jd.163yun.com/ssh/sshd_config
RUN wget -P /tmp/ https://dlonline.nos-jd.163yun.com/ssh/run.sh
RUN chmod +x /tmp/run.sh
ENTRYPOINT ["sh", "/tmp/run.sh"]