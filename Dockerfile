# easydistill image use SunInHeart's modified code for aip
# FROM ubuntu:22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

RUN apt-get update -y && \
    apt-get install -y git python3-pip python3-dev build-essential ca-certificates && \
    # 清理apt缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# 配置 pip 使用国内镜像源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Install easydistill and dependencies
RUN git clone https://github.com/SunInHeart/easydistill.git && \
    cd easydistill && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

# Install other dependencies
RUN pip3 install modelscope deepspeed jsonlines && \
    rm -rf /root/.cache/pip

# Set alias for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

