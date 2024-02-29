# pull official base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/Utc
ENV PYTHONFAULTHANDLER=1
ENV export PYTHONPATH="$PYTHONPATH:/opt/:/usr/src/app/"
# set work directory
WORKDIR /usr/src/app
# this is important for cv2 

RUN apt-get update && \
    apt install -y --no-install-recommends \
    python3-gevent gcc git \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    jq \
    openjdk-8-jdk-headless \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libibverbs-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-glx0 \
    zlib1g && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \ 
    &&  apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip install --upgrade pip && hash pip && \
    pip install wheel && pip install install torch torchvision ultralytics --no-cache-dir
# install dependencies
ADD requirements.txt /usr/src/app/requirements.txt
RUN pip install -r /usr/src/app/requirements.txt --no-cache-dir

# copy project
ADD . /usr/src/app

ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
