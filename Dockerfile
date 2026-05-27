# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOXER_REPO_PATH=/app/boxer \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
    --index-url "${PYTORCH_CUDA_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip install -r requirements.txt

COPY . .
COPY docker/entrypoint.sh /usr/local/bin/boxer-entrypoint
RUN chmod +x /usr/local/bin/boxer-entrypoint

EXPOSE 8000

ENTRYPOINT ["boxer-entrypoint"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
