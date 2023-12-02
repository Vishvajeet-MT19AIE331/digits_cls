# This is the dependency image
# FROM ubuntu:23.10
FROM dependency:v1

# Mount the volume at the specified location
#VOLUME ["/models"]

# copy the whole code directory
COPY . /digits/
# RUN apt-get update
#RUN apt-get install -y python3 python3-pip
#RUN pip3 install -r /digits/requirements.txt


WORKDIR /digits

#ENV FLASK_APP=flask_azure.py
#CMD ["python", "exp_parser.py"]
#ENTRYPOINT ["python", "exp_parser.py"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
ENTRYPOINT [ "python" ]

CMD ["pytest" ]