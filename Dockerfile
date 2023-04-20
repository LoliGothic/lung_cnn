FROM python:3.9.8
USER root

WORKDIR src/app

# 環境変数設定
ENV PYTHONUNBUFFERED 1
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# pipのアップデート
RUN pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt