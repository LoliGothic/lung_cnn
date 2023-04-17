FROM python:3.11.3-alpine
USER root

WORKDIR src/app

# aptのアップデート
RUN apt-get update && \
    apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# 環境変数設定
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# pipのアップデート
RUN apt-get install -y vim less && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools