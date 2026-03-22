FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (rarely change)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (stable layer)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# ENV BUILD (heavy → cache!)
COPY uois_env.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f uois_env.yml && \
    conda clean -afy

# Activate env
SHELL ["conda", "run", "--no-capture-output", "-n", "uois3d", "/bin/bash", "-c"]

# Python deps (separate layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CODE (fast rebuild layer)
COPY app.py .
COPY service_core.py .
COPY uois ./uois

ENV PYTHONPATH=/app

CMD ["conda", "run", "--no-capture-output", "-n", "uois3d", \
     "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]