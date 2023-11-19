# FROM ubuntu:23.10
FROM python:3.9-slim
# copy the whole code directory
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt

# create external storage folder using Volume
#VOLUME ["/models"]

WORKDIR /digits

ENV FLASK_APP=flask_azure.py
CMD ["flask", "run"]
