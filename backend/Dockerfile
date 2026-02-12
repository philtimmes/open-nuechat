# syntax=docker/dockerfile:1.4
# Multi-stage build: Frontend + Backend
#
# PREREQUISITE: Build FAISS wheel ONCE before building this image:
#   cd backend && ./build-faiss.sh
#
# The wheel is saved to backend/wheels/ and reused forever.
#
# BUILD ARGS:
#   --build-arg FAISS_CPU=1  - Use faiss-cpu from PyPI instead of ROCm wheel

# ============ Stage 1: Build Frontend ============
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps

COPY frontend/ ./
RUN npm run build


# ============ Stage 2: Backend Runtime ============
FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1

# Build arg for FAISS CPU mode (set to 1 to use faiss-cpu instead of ROCm wheel)
ARG FAISS_CPU=0

ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=9.0.10
ENV HIP_VISIBLE_DEVICES=0
ENV PYTORCH_ROCM_ARCH=gfx90a

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    git \
    libmagic1 \
    libopenblas-dev \
    swig \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install FAISS - either from pre-built ROCm wheel or CPU version from PyPI
# Use --build-arg FAISS_CPU=1 to use CPU version (avoids needing to build wheel)
COPY backend/wheels/faiss*.whl /tmp/
RUN if [ "$FAISS_CPU" = "1" ]; then \
        echo "Installing faiss-cpu from PyPI..." && \
        pip install faiss-cpu && \
        rm -f /tmp/faiss*.whl; \
    else \
        echo "Installing pre-built FAISS ROCm wheel..." && \
        pip install /tmp/faiss*.whl && \
        rm /tmp/faiss*.whl; \
    fi

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model during build (avoids runtime download issues)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY backend/app/ ./app/

# Copy frontend build
COPY --from=frontend-builder /frontend/dist ./static/

# Create data directories
RUN mkdir -p /app/data /app/uploads /app/uploads/generated /app/faiss_indexes

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000

CMD uvicorn app.main:app --host ${BACKEND_HOST} --port ${BACKEND_PORT} --log-level warning --no-access-log
