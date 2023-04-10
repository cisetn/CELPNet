FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
USER root

RUN apt-get update && \
    apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 && \
    apt-get -y install libsndfile1 -y build-essential

ENV LANG ja_JP.UTF-8 \
    LANGUAGE ja_JP:ja \
    LC_ALL ja_JP.UTF-8 \
    TZ JST-9 \
    TERM xterm \
    NVIDIA_VISIBLE_DEVICES all \
    NVIDIA_DRIVER_CAPABILITIES all

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN mkdir /root/workspace
COPY requirement.txt /root/workspace
WORKDIR /root/workspace

RUN pip install -r requirement.txt