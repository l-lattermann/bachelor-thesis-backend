FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app
ENV HOME=/tmp

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY service_core.py .
COPY data_io.py .

RUN mkdir -p /shared/pipeline_io /shared/debug

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]