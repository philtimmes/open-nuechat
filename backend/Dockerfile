# syntax=docker/dockerfile:1.4
# Multi-stage build: Frontend + Backend
#
# PREREQUISITE: Build FAISS wheel ONCE before building this image:
#   cd backend && ./build-faiss.sh
#
# The wheel is saved to backend/wheels/ and reused forever.

# ============ Stage 1: Build Frontend ============
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps

COPY frontend/ ./
RUN npm run build


# ============ Stage 2: Backend Runtime ============
FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1

ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=9.0.10
ENV HIP_VISIBLE_DEVICES=0
ENV PYTORCH_ROCM_ARCH=gfx90a

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libmagic1 \
    libopenblas-dev \
    swig \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install pre-built FAISS wheel
# This file MUST exist: run `cd backend && ./build-faiss.sh` first
COPY backend/wheels/faiss*.whl /tmp/
RUN pip install /tmp/faiss*.whl && rm /tmp/faiss*.whl

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

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
