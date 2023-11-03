# FROM ubuntu:23.10
FROM python:3.9.17
# copy the whole code directory
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt

# create external storage folder using Volume
VOLUME ["/models"]
# need python
# no need for conda or venv
WORKDIR /digits
# requirements installation
#CMD ["/digits/exp.py"]
RUN /digits/exp.py
