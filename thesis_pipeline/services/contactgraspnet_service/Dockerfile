FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (stable)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    curl \
    build-essential \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (stable)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# ENV BUILD (heavy → cached)
COPY contact_graspnet_env.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f contact_graspnet_env.yml && \
    conda clean -afy

# Activate env
SHELL ["conda", "run", "--no-capture-output", "-n", "contact_graspnet_env", "/bin/bash", "-c"]

# Python deps (separate layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CODE (fast rebuild)
COPY app.py .
COPY service_core.py .
COPY contact_graspnet ./contact_graspnet

ENV PYTHONPATH=/app:/app/contact_graspnet

CMD ["conda", "run", "--no-capture-output", "-n", "contact_graspnet_env", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]