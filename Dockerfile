FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3.7 python3-pip libsndfile1
COPY ./requirements.txt /requirements.txt
COPY ./app /app
RUN pip3 install -r requirements.txt
WORKDIR /app