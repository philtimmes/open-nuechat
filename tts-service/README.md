# Kokoro TTS Service

Text-to-speech microservice using [Kokoro](https://github.com/hexgrad/kokoro) TTS.

## Features

- Fast, high-quality neural TTS
- Multiple voices and languages
- Request queuing with job status polling
- Streaming audio generation
- ROCm GPU acceleration (AMD GPUs)

## Docker Builds

### CPU-Only (Default)

```bash
# Build
docker build -t nexus-tts -f Dockerfile .

# Run
docker run -p 8033:8033 nexus-tts
```

Uses pyenv to install Python 3.12 with CPU-only PyTorch (~200MB).

### ROCm GPU (AMD)

```bash
# Build
docker build -t nexus-tts-rocm -f Dockerfile.rocm .

# Run (requires AMD GPU with ROCm)
docker run -p 8033:8033 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  nexus-tts-rocm
```

Uses ROCm PyTorch base image with GPU acceleration.

## Docker Compose

### Using control.sh (Recommended)

```bash
# CPU-only TTS
./control.sh tts cpu

# ROCm GPU TTS
./control.sh tts rocm

# Stop TTS service
./control.sh tts stop

# Check status
./control.sh tts status

# View logs
./control.sh tts logs
```

### Using docker compose directly

```bash
# CPU-only (default)
docker compose up -d tts-service

# ROCm GPU
docker compose --profile rocm-tts up -d tts-service-rocm
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_HOST` | `0.0.0.0` | Host to bind |
| `TTS_PORT` | `8033` | Port to listen on |
| `TTS_MAX_QUEUE_SIZE` | `100` | Maximum pending jobs |
| `TTS_MAX_CONCURRENT` | `2` (CPU) / `4` (GPU) | Parallel workers |
| `TTS_RESULT_TTL` | `300` | Seconds to keep completed results |
| `TTS_MAX_TEXT_LENGTH` | `10000` | Maximum characters per request |
| `TTS_USE_GPU` | `false` (CPU) / `true` (ROCm) | Enable GPU acceleration |

## API Endpoints

### Health Check
```
GET /health
```

Returns service status, queue size, and GPU info (if available).

### List Voices
```
GET /voices
GET /voices?gender=female&lang=en
```

### Generate Speech (Blocking)
```
POST /tts/generate
Content-Type: application/json

{"text": "Hello world", "voice": "af_heart"}
```

Returns WAV audio directly.

### Generate Speech (Streaming)
```
POST /tts/stream
Content-Type: application/json

{"text": "Hello world", "voice": "af_heart"}
```

Streams audio chunks as they're generated. Each chunk has a 4-byte length prefix.

### Submit Job (Async)
```
POST /tts/submit
Content-Type: application/json

{"text": "Hello world", "voice": "af_heart"}
```

Returns job ID for polling.

### Check Job Status
```
GET /tts/job/{job_id}
```

### Get Job Result
```
GET /tts/result/{job_id}
```

Returns WAV audio when complete.

## Available Voices

| Voice ID | Name | Gender | Language |
|----------|------|--------|----------|
| `af_heart` | Heart | Female | American English |
| `af_bella` | Bella | Female | American English |
| `af_nicole` | Nicole | Female | American English |
| `af_sarah` | Sarah | Female | American English |
| `af_sky` | Sky | Female | American English |
| `am_adam` | Adam | Male | American English |
| `am_michael` | Michael | Male | American English |
| `bf_emma` | Emma | Female | British English |
| `bf_isabella` | Isabella | Female | British English |
| `bm_george` | George | Male | British English |
| `bm_lewis` | Lewis | Male | British English |

See `/voices` endpoint for complete list.
