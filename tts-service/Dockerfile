# Kokoro TTS Service (CPU-only)
# Uses Python 3.12 via pyenv for Kokoro compatibility
# For ROCm GPU support, use Dockerfile.rocm instead

FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    curl \
    git \
    # pyenv dependencies
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # espeak-ng for Kokoro
    espeak-ng \
    # Audio libraries
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN curl https://pyenv.run | bash

# Install Python 3.12
RUN pyenv install 3.12.7 && \
    pyenv global 3.12.7 && \
    pyenv rehash

# Verify Python version
RUN python --version

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install CPU-only PyTorch FIRST (Kokoro needs torch but doesn't need GPU)
# This prevents pip from pulling the 900MB CUDA version
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application code
COPY main.py .

# Environment variables
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8033
ENV TTS_MAX_QUEUE_SIZE=100
ENV TTS_MAX_CONCURRENT=2
ENV TTS_RESULT_TTL=300
ENV TTS_MAX_TEXT_LENGTH=10000
# CPU-only build - disable GPU
ENV TTS_USE_GPU=false

# Expose port
EXPOSE 8033

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8033/health || exit 1

# Run the service
CMD ["python", "main.py"]
