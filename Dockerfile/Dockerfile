FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Set the DEBIAN_FRONTEND environment variable to noninteractive

# Set the DEBIAN_FRONTEND environment variable to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install CUDA toolkit
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-toolkit-11-2 && \
    rm -rf /var/lib/apt/lists/*


# Install Python and pip
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip

# Install TensorFlow and Keras
RUN pip3 install tensorflow==2.6.0 keras
RUN pip3 install protobuf==3.20.0


WORKDIR  /home/grad/chowdhury/Dockerfile/src
COPY src/ /src/
RUN ls -la /src/


# Run the Python file that imports TensorFlow
CMD ["python3", "/src/invclassification.py"]
