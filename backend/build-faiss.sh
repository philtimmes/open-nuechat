#!/bin/bash
# Build FAISS wheel ONCE and save it locally
# After running this, the wheel is saved to backend/wheels/ and NEVER rebuilt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

# Check if any faiss wheel already exists
if ls "$WHEELS_DIR"/faiss*.whl 1>/dev/null 2>&1; then
    echo "=============================================="
    echo "FAISS wheel already exists:"
    ls -la "$WHEELS_DIR"/faiss*.whl
    echo ""
    echo "Nothing to do. FAISS will NOT be rebuilt."
    echo "=============================================="
    exit 0
fi

# Extract base image from Dockerfile to ensure Python version matches
BASE_IMAGE=$(grep -E "^FROM.*rocm" "$DOCKERFILE" | head -1 | awk '{print $2}')
if [ -z "$BASE_IMAGE" ]; then
    echo "ERROR: Could not extract base image from Dockerfile"
    exit 1
fi

echo "=============================================="
echo "Building FAISS ROCm wheel..."
echo "Base image: $BASE_IMAGE"
echo "This takes ~12 minutes but only happens ONCE."
echo "=============================================="

mkdir -p "$WHEELS_DIR"

# Build in a temporary container using the SAME base image as Dockerfile
docker run --rm \
    -v "$WHEELS_DIR:/output" \
    "$BASE_IMAGE" \
    bash -c '
        set -e
        echo "Python version: $(python --version)"
        
        echo "Installing build dependencies..."
        apt-get update && apt-get install -y build-essential cmake git libopenblas-dev swig
        
        echo "Installing faiss-cpu for dependencies..."
        pip install faiss-cpu && pip uninstall -y faiss-cpu
        
        echo "Cloning faiss-wheels..."
        git clone --recursive https://github.com/faiss-wheels/faiss-wheels.git /tmp/faiss-wheels
        cd /tmp/faiss-wheels
        
        echo "Building FAISS with ROCm support..."
        pip install pipx
        FAISS_GPU_SUPPORT=ROCM FAISS_OPT_LEVELS=generic pipx run build --wheel
        
        echo "Copying wheel to output (preserving original filename)..."
        cp dist/faiss*.whl /output/
        
        echo "Done!"
    '

# Check if wheel was created
if ls "$WHEELS_DIR"/faiss*.whl 1>/dev/null 2>&1; then
    echo ""
    echo "=============================================="
    echo "SUCCESS! FAISS wheel built and saved:"
    ls -la "$WHEELS_DIR"/faiss*.whl
    echo ""
    echo "This wheel will be used for ALL future builds."
    echo "FAISS will NEVER be rebuilt again."
    echo "=============================================="
else
    echo "ERROR: Wheel file not created!"
    exit 1
fi
