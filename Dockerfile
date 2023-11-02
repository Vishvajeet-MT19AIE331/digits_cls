FROM ubuntu:18.04
COPY ./digits_cls/
COPY requirements.txt /digits_cls/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /digits_cls/requirements.txt
#RUN mkdir /digits_cls/models
#WORKDIR /exp
CMD ["python3", "./digits_cls/exp.py"]