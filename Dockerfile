FROM ubuntu:18.04

RUN apt-get update && apt-get upgrade -y && apt-get clean


# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y libsndfile1
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt


EXPOSE 5000

WORKDIR /app