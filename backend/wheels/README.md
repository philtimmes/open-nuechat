# FAISS Wheel

This directory contains the pre-built FAISS ROCm wheel.

## Building the Wheel (ONCE)

```bash
cd backend
./build-faiss.sh
```

This takes ~12 minutes but **only needs to happen ONCE**.

After running, you'll have a `faiss*.whl` file in this directory.

## What Happens

1. The script spins up a temporary Docker container
2. Builds FAISS with ROCm GPU support
3. Copies the wheel to this directory
4. Container is removed

## Never Rebuilt

Once the wheel exists here, it is:
- Copied directly into the Docker image during `docker compose build`
- **Never rebuilt** for any reason
- Reused forever

## If You Need to Rebuild

Delete the wheel and run the script again:

```bash
rm backend/wheels/faiss*.whl
cd backend && ./build-faiss.sh
```
