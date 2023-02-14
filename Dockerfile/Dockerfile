
FROM  python:3.6-stretch
MAINTAINER Shaif <shaif_chowdhury1@baylor.edu>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential


# Installing python dependencies

WORKDIR  /home/grad/chowdhury/Dockerfile/src

# check our python environment

RUN python3 --version
RUN pip3 --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /src/
RUN ls -la /src/

RUN python3 /src/invclassification.py


