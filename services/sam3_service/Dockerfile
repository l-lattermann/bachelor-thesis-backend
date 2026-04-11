FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python3.12 -m pip install --break-system-packages setuptools wheel

RUN python3.12 -m pip install --break-system-packages \
    torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install --break-system-packages -r /tmp/requirements.txt

COPY sam3 /opt/sam3
RUN python3.12 -m pip install --break-system-packages -e /opt/sam3

WORKDIR /app

CMD ["python3.12", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]