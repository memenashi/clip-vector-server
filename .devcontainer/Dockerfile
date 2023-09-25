FROM python:3.8.17-slim
RUN apt-get update && apt-get install -y git &&\
  rm -rf /var/lib/apt/lists/* &&\
  apt-get clean &&\
  rm -rf /usr/local/src/*
WORKDIR /app
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir git+https://github.com/rinnakk/japanese-clip.git
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY main.py /app/
