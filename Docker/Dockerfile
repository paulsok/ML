FROM nvidia/cuda:10.1-cudnn7-runtime

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY req.txt /req.txt
COPY ./src /src
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install --no-cache-dir -r /req.txt
CMD ['python3', '/src/app.py']
EXPOSE 8080