# Image Generation Service

Text-to-image generation service using [Diffusers](https://github.com/huggingface/diffusers) with the Z-Image-Turbo model.

## Security

**This service binds to localhost only (127.0.0.1) by default.** External access is blocked to prevent unauthorized use of GPU resources. Image generation must go through the authenticated chat backend API.

## Features

- Fast image generation using Z-Image-Turbo (8 inference steps)
- ROCm GPU acceleration (AMD GPUs)
- Configurable image sizes
- Random or fixed seed support
- Job queue with status tracking
- Health endpoint with GPU monitoring

## Model

Uses [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - a fast, high-quality image generation model.

- 8 inference steps (set `num_inference_steps=9` which yields 8 DiT forwards)
- Guidance scale 0.0 (recommended for Turbo models)
- BFloat16 precision for optimal GPU performance

## Quick Start

```bash
# Using control.sh
./control.sh image start

# Or with docker compose
docker compose --profile image-gen up -d image-service
```

## API Endpoints

### Health Check
```
GET /health
```

Returns service status, model info, and GPU details.

### Generate Image (Blocking)
```
POST /generate
Content-Type: application/json

{
  "prompt": "A sunset over mountains",
  "width": 1024,
  "height": 1024,
  "seed": null  // null for random
}
```

Returns immediately with base64-encoded PNG image.

### Generate Image (Async)
```
POST /generate/async
```

Returns job ID for polling.

### Get Job Status
```
GET /job/{job_id}
```

### Get Job Result
```
GET /job/{job_id}/result
```

### Get Job Image (PNG)
```
GET /job/{job_id}/image
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_GEN_HOST` | `127.0.0.1` | Host to bind (localhost only by default for security) |
| `IMAGE_GEN_PORT` | `8034` | Port to listen on |
| `IMAGE_GEN_MODEL` | `Tongyi-MAI/Z-Image-Turbo` | HuggingFace model ID |
| `IMAGE_GEN_DEFAULT_WIDTH` | `1024` | Default image width |
| `IMAGE_GEN_DEFAULT_HEIGHT` | `1024` | Default image height |
| `IMAGE_GEN_INFERENCE_STEPS` | `9` | Number of inference steps |
| `IMAGE_GEN_GUIDANCE_SCALE` | `0.0` | CFG guidance scale |
| `IMAGE_GEN_MAX_QUEUE_SIZE` | `20` | Maximum pending jobs |
| `IMAGE_GEN_MAX_CONCURRENT` | `1` | Parallel workers |

## Supported Sizes

- 512×512
- 768×768
- 1024×1024 (default)
- 1280×720 (landscape 16:9)
- 720×1280 (portrait 9:16)
- 1024×768 (landscape 4:3)
- 768×1024 (portrait 3:4)

Sizes are rounded to multiples of 64 as required by the model.

## GPU Requirements

- AMD GPU with ROCm 6.0+ support
- Minimum 8GB VRAM recommended
- Model download requires ~5GB disk space

## First Run

On first startup, the model will be downloaded from HuggingFace (~5GB). This is cached to the `image-gen-cache` Docker volume for faster subsequent starts.

## Example Response

```json
{
  "success": true,
  "job_id": "abc123",
  "image_base64": "iVBORw0KGgo...",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "prompt": "A sunset over mountains",
  "generation_time": 3.45
}
```

## Integration

The image generation service integrates with the chat backend. When a user sends a message like "create an image of...", the backend:

1. Detects the image generation intent
2. Calls this service to generate the image
3. Returns the image to the chat interface
4. User can download, retry, or edit the prompt
