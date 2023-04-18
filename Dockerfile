FROM python:3.11.3-alpine
USER root

WORKDIR src/app

# 環境変数設定
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# pipのアップデート
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools