FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python3.12 -m ensurepip --upgrade || true
RUN python3.12 -m pip install --break-system-packages --upgrade pip setuptools wheel

RUN python3.12 -m pip install --break-system-packages \
    torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

COPY sam3 /app/sam3

WORKDIR /app/sam3

RUN python3.12 -m pip install --break-system-packages -e .

WORKDIR /app

CMD ["python3.12"]