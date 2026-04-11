FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git git-lfs curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch first, then cuRobo.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/NVlabs/curobo.git /opt/curobo && \
    cd /opt/curobo && \
    pip install -e . --no-build-isolation

RUN pip install fastapi uvicorn[standard] pydantic

COPY app.py /app/app.py
COPY service_core.py /app/service_core.py

RUN mkdir -p /models

ENV CUROBO_ROBOT_CFG=/models/ur5e_rg6.yml
ENV CUROBO_INTERPOLATION_DT=0.02
ENV CUROBO_COLLISION_CACHE_OBB=50
ENV CUROBO_COLLISION_CACHE_MESH=20

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]