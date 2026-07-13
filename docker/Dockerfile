FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
WORKDIR /app
CMD ["/bin/bash"]