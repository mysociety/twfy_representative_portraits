FROM python:3.10-buster
ENV DEBIAN_FRONTEND noninteractive
COPY requirements.txt /
RUN pip install -r /requirements.txt
