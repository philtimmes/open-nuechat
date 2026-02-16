"""
Image Generation Service using Diffusers with Z-Image-Turbo
ROCm GPU accelerated
"""
import asyncio
import base64
import io
import logging
import os
import random
import re
import time
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep our own logger at INFO for startup messages

# Silence uvicorn access logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# ============ Configuration ============

# Apply GPU assignment BEFORE any torch imports
def _read_gpu_assignment(model_key: str):
    """Read GPU assignment from /app/data/gpuSelector/assignments.json."""
    import json as _json
    path = os.getenv("GPU_SELECTOR_DIR", "/app/data/gpuSelector") + "/assignments.json"
    try:
        with open(path, 'r') as f:
            data = _json.load(f)
        entry = data.get(model_key)
        if entry and isinstance(entry, dict):
            return entry.get("rocm_id")
        elif entry is not None:
            return int(entry)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _rocm_id_to_hip_id(rocm_id: int) -> int:
    """Translate ROCm GPU index to HIP_ID using `amd-smi list -e`."""
    import subprocess as _sp
    try:
        result = _sp.run(["amd-smi", "list", "-e"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return rocm_id
        current_gpu = None
        for line in result.stdout.splitlines():
            line_s = line.strip()
            m = re.match(r'^GPU:\s*(\d+)', line_s)
            if m:
                current_gpu = int(m.group(1))
                continue
            if current_gpu == rocm_id and 'HIP_ID' in line_s:
                _, _, val = line_s.partition(':')
                try:
                    hip_id = int(val.strip())
                    logger.info(f"[GPU MAP] ROCm GPU {rocm_id} → HIP_ID {hip_id}")
                    return hip_id
                except ValueError:
                    pass
    except (FileNotFoundError, _sp.TimeoutExpired):
        pass
    return rocm_id


_img_rocm_id = _read_gpu_assignment("image_gen")
if _img_rocm_id is not None:
    _img_hip_id = _rocm_id_to_hip_id(_img_rocm_id)
    os.environ["HIP_VISIBLE_DEVICES"] = str(_img_hip_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_img_hip_id)
    logger.info(f"[IMAGE] GPU assignment: ROCm GPU {_img_rocm_id} → HIP_ID {_img_hip_id}")

MODEL_ID = os.getenv("IMAGE_GEN_MODEL", "Tongyi-MAI/Z-Image-Turbo")
DEFAULT_WIDTH = int(os.getenv("IMAGE_GEN_DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("IMAGE_GEN_DEFAULT_HEIGHT", "1024"))
INFERENCE_STEPS = int(os.getenv("IMAGE_GEN_INFERENCE_STEPS", "9"))
GUIDANCE_SCALE = float(os.getenv("IMAGE_GEN_GUIDANCE_SCALE", "0.0"))
MAX_QUEUE_SIZE = int(os.getenv("IMAGE_GEN_MAX_QUEUE_SIZE", "20"))
MAX_CONCURRENT = int(os.getenv("IMAGE_GEN_MAX_CONCURRENT", "1"))  # GPU memory limited
RESULT_TTL_SECONDS = int(os.getenv("IMAGE_GEN_RESULT_TTL", "600"))  # 10 minutes

# Supported sizes (width x height)
SUPPORTED_SIZES = [
    # Square
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1080, 1080),   # Instagram
    
    # Landscape
    (1280, 720),    # HD 16:9
    (1920, 1080),   # Full HD 16:9
    (1024, 768),    # 4:3
    (1200, 800),    # 3:2
    (1200, 675),    # Twitter
    (1344, 576),    # Ultrawide 21:9
    
    # Portrait
    (720, 1280),    # HD 9:16
    (1080, 1920),   # Full HD 9:16
    (768, 1024),    # 3:4
    (800, 1200),    # 2:3
]


# ============ Models ============

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ImageJob:
    id: str
    prompt: str
    width: int
    height: int
    seed: int
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_image: Optional[bytes] = None
    result_base64: Optional[str] = None
    error: Optional[str] = None
    queue_position: int = 0


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    original_prompt: Optional[str] = None
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=2048)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=2048)
    seed: Optional[int] = Field(default=None, description="Random seed. None for random.")
    user_id: Optional[str] = None
    chat_id: Optional[str] = None


class ImageGenResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    prompt: Optional[str] = None
    revised_prompt: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: Optional[int] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


# ============ Image Pipeline ============

class ImagePipeline:
    """Manages the Diffusers image generation pipeline"""
    
    def __init__(self):
        self._pipe = None
        self._lock = asyncio.Lock()
        self._device = "cpu"
    
    async def initialize(self):
        """Initialize the pipeline"""
        if self._pipe is not None:
            return
        
        async with self._lock:
            if self._pipe is not None:
                return
            
            logger.info(f"Initializing image generation pipeline with {MODEL_ID}...")
            
            try:
                import torch
                from diffusers import FluxPipeline
                
                # GPU detection — HIP_VISIBLE_DEVICES already set at module level
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                    device_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"Image Gen using cuda:0: {device_name} ({vram_gb:.1f} GB VRAM)")
                    if hasattr(torch.version, 'hip') and torch.version.hip:
                        logger.info(f"ROCm/HIP version: {torch.version.hip}")
                else:
                    logger.warning("No GPU detected, using CPU (will be slow)")
                
                # Load the pipeline
                logger.info(f"Loading model {MODEL_ID}...")
                
                # Try ZImagePipeline first, fall back to FluxPipeline
                try:
                    from diffusers import ZImagePipeline
                    self._pipe = ZImagePipeline.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
                        low_cpu_mem_usage=False,
                    )
                    logger.info("Using ZImagePipeline")
                except ImportError:
                    # Fallback for older diffusers versions
                    self._pipe = FluxPipeline.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
                    )
                    logger.info("Using FluxPipeline (fallback)")
                
                self._pipe.to(self._device)
                
                logger.info("Image generation pipeline initialized successfully")
                
                # Warmup with a small generation
                if self._device == "cuda":
                    logger.info("Warming up GPU...")
                    try:
                        import torch
                        _ = self._pipe(
                            prompt="test",
                            height=256,
                            width=256,
                            num_inference_steps=2,
                            guidance_scale=0.0,
                            generator=torch.Generator(self._device).manual_seed(0),
                        )
                        logger.info("GPU warmup complete")
                    except Exception as e:
                        logger.warning(f"GPU warmup failed (non-critical): {e}")
                
            except Exception as e:
                logger.error(f"Failed to initialize pipeline: {e}")
                raise
    
    def generate_sync(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Image.Image:
        """Generate an image synchronously"""
        if self._pipe is None:
            raise RuntimeError("Pipeline not initialized")
        
        import torch
        
        logger.info(f"generate_sync: prompt={prompt[:50]}..., size={width}x{height}, seed={seed}")
        
        generator = torch.Generator(self._device).manual_seed(seed)
        
        logger.info("generate_sync: calling pipeline...")
        result = self._pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        
        logger.info(f"generate_sync: pipeline returned, images count={len(result.images)}")
        image = result.images[0]
        logger.info(f"generate_sync: image size={image.size}, mode={image.mode}")
        
        return image
    
    def img2img_sync(
        self,
        prompt: str,
        input_image: Image.Image,
        seed: int,
        denoise: float = 0.3,
        steps: int = 7,
    ) -> Image.Image:
        """Modify an image using img2img (encode → noise → denoise)"""
        if self._pipe is None:
            raise RuntimeError("Pipeline not initialized")
        
        import torch
        import numpy as np
        
        logger.info(f"img2img_sync: prompt={prompt[:50]}..., denoise={denoise}, steps={steps}, seed={seed}")
        
        # Scale image to max 1300px on longest side (matching ComfyUI workflow)
        max_dim = 1300
        w, h = input_image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
        # Round to 64
        w = (w // 64) * 64
        h = (h // 64) * 64
        w = max(256, min(2048, w))
        h = max(256, min(2048, h))
        input_image = input_image.resize((w, h), Image.LANCZOS)
        logger.info(f"img2img_sync: resized to {w}x{h}")
        
        generator = torch.Generator(self._device).manual_seed(seed)
        
        # Encode input image to latents
        img_tensor = torch.from_numpy(
            np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
        ).unsqueeze(0).permute(0, 3, 1, 2).to(self._device, dtype=torch.bfloat16 if self._device == "cuda" else torch.float32)
        
        with torch.no_grad():
            latents = self._pipe.vae.encode(img_tensor).latent_dist.sample(generator)
            latents = (latents - self._pipe.vae.config.shift_factor) * self._pipe.vae.config.scaling_factor
        
        # Calculate how many steps to skip based on denoise
        start_step = int(steps * (1.0 - denoise))
        actual_steps = steps - start_step
        logger.info(f"img2img_sync: total_steps={steps}, start_step={start_step}, actual_steps={actual_steps}")
        
        # Add noise to latents proportional to denoise strength
        # Use the scheduler to get the right noise level
        self._pipe.scheduler.set_timesteps(steps, device=self._device)
        timesteps = self._pipe.scheduler.timesteps
        
        if start_step < len(timesteps):
            noise = torch.randn_like(latents, generator=generator)
            t = timesteps[start_step]
            # For flow matching (Flux/AuraFlow), noise is added linearly:
            # noisy = (1 - sigma) * latents + sigma * noise
            sigma = t / self._pipe.scheduler.config.num_train_timesteps if hasattr(self._pipe.scheduler.config, 'num_train_timesteps') else t.float()
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.to(latents.dtype)
            else:
                sigma = torch.tensor(sigma, dtype=latents.dtype, device=latents.device)
            noisy_latents = (1.0 - sigma) * latents + sigma * noise
        else:
            noisy_latents = latents
        
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self._pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
        )
        
        # Prepare latent image ids
        latent_h = noisy_latents.shape[2]
        latent_w = noisy_latents.shape[3]
        latent_image_ids = self._pipe._prepare_latent_image_ids(
            noisy_latents.shape[0], latent_h, latent_w, self._device, torch.bfloat16 if self._device == "cuda" else torch.float32
        )
        
        # Pack latents for transformer
        packed_latents = self._pipe._pack_latents(noisy_latents, noisy_latents.shape[0], noisy_latents.shape[1], latent_h, latent_w)
        
        # Run denoising from start_step
        remaining_timesteps = timesteps[start_step:]
        
        with torch.no_grad():
            for i, t in enumerate(remaining_timesteps):
                t_batch = t.expand(packed_latents.shape[0]).to(packed_latents.dtype)
                
                noise_pred = self._pipe.transformer(
                    hidden_states=packed_latents,
                    timestep=t_batch,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                packed_latents = self._pipe.scheduler.step(noise_pred, t, packed_latents, return_dict=False)[0]
        
        # Unpack and decode
        unpacked = self._pipe._unpack_latents(packed_latents, latent_h, latent_w, self._pipe.vae_scale_factor)
        unpacked = (unpacked / self._pipe.vae.config.scaling_factor) + self._pipe.vae.config.shift_factor
        
        with torch.no_grad():
            decoded = self._pipe.vae.decode(unpacked, return_dict=False)[0]
        
        # Convert to PIL
        decoded = decoded.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        result_image = Image.fromarray((decoded * 255).astype(np.uint8))
        
        logger.info(f"img2img_sync: output size={result_image.size}")
        return result_image
    
    async def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
    ) -> Image.Image:
        """Generate an image asynchronously"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self.generate_sync,
                prompt,
                width,
                height,
                seed,
            )
            logger.info(f"generate() completed, result type={type(result)}")
            return result
        except Exception as e:
            logger.error(f"generate() failed: {e}")
            raise
    
    @property
    def device(self) -> str:
        return self._device


# ============ Queue Manager ============

class ImageQueueManager:
    """Manages image generation job queue"""
    
    def __init__(self, pipeline: ImagePipeline, max_queue: int, max_concurrent: int):
        self.pipeline = pipeline
        self.max_queue = max_queue
        self.max_concurrent = max_concurrent
        self.queue: asyncio.Queue[ImageJob] = asyncio.Queue(maxsize=max_queue)
        self.jobs: OrderedDict[str, ImageJob] = OrderedDict()
        self.results: OrderedDict[str, ImageJob] = OrderedDict()
        self._workers: list[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        """Start the queue workers"""
        await self.pipeline.initialize()
        self._running = True
        
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_results())
        
        logger.info(f"Started {self.max_concurrent} image generation workers")
    
    async def stop(self):
        """Stop the queue workers"""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("Image generation workers stopped")
    
    async def _worker(self, worker_id: int):
        """Worker that processes jobs from the queue"""
        logger.info(f"Image worker {worker_id} started")
        
        while self._running:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info(f"Image worker {worker_id} cancelled")
                break
            
            try:
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()
                logger.info(f"Worker {worker_id} processing job {job.id}: {job.prompt[:50]}...")
                
                # Generate the image (runs in thread pool to not block event loop)
                logger.info(f"Job {job.id}: Starting image generation...")
                image = await self.pipeline.generate(
                    prompt=job.prompt,
                    width=job.width,
                    height=job.height,
                    seed=job.seed,
                )
                logger.info(f"Job {job.id}: Image generated, converting to PNG...")
                
                # Convert to PNG bytes (no optimize - faster)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                image_bytes = buffer.read()
                logger.info(f"Job {job.id}: PNG size = {len(image_bytes)} bytes")
                
                job.result_image = image_bytes
                job.result_base64 = base64.b64encode(image_bytes).decode('utf-8')
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                
                gen_time = job.completed_at - job.started_at
                logger.info(f"Job {job.id} completed in {gen_time:.2f}s, base64 len={len(job.result_base64)}")
                
            except Exception as e:
                import traceback
                logger.error(f"Job {job.id} failed: {e}")
                logger.error(traceback.format_exc())
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
            
            finally:
                # Always move to results (whether success or failure)
                self.results[job.id] = job
                if job.id in self.jobs:
                    del self.jobs[job.id]
                self._update_queue_positions()
                logger.info(f"Job {job.id} moved to results, status={job.status.value}")
    
    def _update_queue_positions(self):
        """Update queue positions for all jobs"""
        for i, job_id in enumerate(self.jobs.keys()):
            self.jobs[job_id].queue_position = i + 1
    
    async def _cleanup_old_results(self):
        """Periodically clean up old results"""
        while self._running:
            await asyncio.sleep(60)
            cutoff = time.time() - RESULT_TTL_SECONDS
            
            to_remove = [
                job_id for job_id, job in self.results.items()
                if job.completed_at and job.completed_at < cutoff
            ]
            
            for job_id in to_remove:
                del self.results[job_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old image results")
    
    async def submit(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None,
    ) -> ImageJob:
        """Submit a new image generation job"""
        if self.queue.full():
            raise HTTPException(
                status_code=503,
                detail="Image generation queue is full. Please try again later."
            )
        
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        job = ImageJob(
            id=str(uuid.uuid4()),
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            queue_position=self.queue.qsize() + 1,
        )
        
        self.jobs[job.id] = job
        await self.queue.put(job)
        
        logger.info(f"Job {job.id} submitted, queue position: {job.queue_position}")
        return job
    
    def get_job(self, job_id: str) -> Optional[ImageJob]:
        """Get a job by ID"""
        if job_id in self.jobs:
            return self.jobs[job_id]
        if job_id in self.results:
            return self.results[job_id]
        return None


# ============ FastAPI App ============

app = FastAPI(
    title="Image Generation Service",
    description="Generate images using Z-Image-Turbo with ROCm GPU acceleration",
    version="1.0.0"
)

# Global instances
pipeline = ImagePipeline()
queue_manager: Optional[ImageQueueManager] = None


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global queue_manager
    # Initialize pipeline first (loads model)
    await pipeline.initialize()
    # Queue manager for async endpoint (optional)
    queue_manager = ImageQueueManager(pipeline, MAX_QUEUE_SIZE, MAX_CONCURRENT)
    await queue_manager.start()
    logger.info("Image Generation Service started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if queue_manager:
        await queue_manager.stop()
    logger.info("Image Generation Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_info = None
    if pipeline.device == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        }
        if hasattr(torch.version, 'hip') and torch.version.hip:
            gpu_info["rocm_version"] = torch.version.hip
    
    return {
        "status": "healthy",
        "service": "image-gen",
        "model": MODEL_ID,
        "device": pipeline.device,
        "gpu": gpu_info,
        "queue_size": queue_manager.queue.qsize() if queue_manager else 0,
        "max_queue": MAX_QUEUE_SIZE,
        "max_concurrent": MAX_CONCURRENT,
    }


@app.get("/info")
async def get_info():
    """Get service info and supported options"""
    return {
        "model": MODEL_ID,
        "supported_sizes": SUPPORTED_SIZES,
        "default_width": DEFAULT_WIDTH,
        "default_height": DEFAULT_HEIGHT,
        "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }


@app.post("/generate", response_model=ImageGenResponse)
async def generate_image(request: ImageGenRequest):
    """
    Generate an image from a text prompt.
    Returns with base64 image data.
    """
    start_time = time.time()
    
    # Validate/adjust size
    width = request.width
    height = request.height
    
    # Round to nearest multiple of 64 (required by most models)
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    # Clamp to reasonable limits
    width = max(256, min(2048, width))
    height = max(256, min(2048, height))
    
    # Generate seed if not provided
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    
    logger.info(f"Generate request: prompt='{request.prompt[:50]}...', size={width}x{height}, seed={seed}")
    
    try:
        # Direct generation (no queue) - simpler and more reliable
        logger.info("Starting image generation...")
        
        # Run the synchronous generation in a thread
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            pipeline.generate_sync,
            request.prompt,
            width,
            height,
            seed,
        )
        
        logger.info(f"Image generated: {image.size}, {image.mode}")
        
        # Convert to PNG
        logger.info("Converting to PNG...")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.read()
        
        logger.info(f"PNG size: {len(image_bytes)} bytes")
        
        # Base64 encode
        logger.info("Encoding to base64...")
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        gen_time = time.time() - start_time
        logger.info(f"Generation complete in {gen_time:.2f}s, base64 len={len(image_base64)}")
        
        return ImageGenResponse(
            success=True,
            job_id=str(uuid.uuid4()),
            image_base64=image_base64,
            width=width,
            height=height,
            seed=seed,
            prompt=request.prompt,
            generation_time=round(gen_time, 2),
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Image generation failed: {e}")
        logger.error(traceback.format_exc())
        return ImageGenResponse(
            success=False,
            error=str(e),
            prompt=request.prompt,
        )


@app.post("/generate/async")
async def generate_image_async(request: ImageGenRequest):
    """
    Submit an image generation job asynchronously.
    Returns job ID to poll for results.
    """
    width = (request.width // 64) * 64
    height = (request.height // 64) * 64
    width = max(256, min(2048, width))
    height = max(256, min(2048, height))
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    
    job = await queue_manager.submit(
        prompt=request.prompt,
        width=width,
        height=height,
        seed=seed,
    )
    
    return {
        "job_id": job.id,
        "status": job.status.value,
        "queue_position": job.queue_position,
        "seed": seed,
        "width": width,
        "height": height,
    }


class ImageModifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    image_base64: str = Field(..., description="Base64-encoded input image")
    denoise: float = Field(default=0.3, ge=0.05, le=1.0, description="Denoise strength. 0.1=subtle, 0.3=moderate, 0.7=heavy, 1.0=full regeneration")
    steps: int = Field(default=7, ge=2, le=30)
    seed: Optional[int] = Field(default=None)


@app.post("/modify", response_model=ImageGenResponse)
async def modify_image(request: ImageModifyRequest):
    """
    Modify an existing image using img2img.
    Low denoise = subtle changes, high denoise = heavy transformation.
    """
    start_time = time.time()
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    
    logger.info(f"Modify request: prompt='{request.prompt[:50]}...', denoise={request.denoise}, steps={request.steps}, seed={seed}")
    
    try:
        # Decode input image
        image_bytes = base64.b64decode(request.image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Input image: {input_image.size}, {input_image.mode}")
        
        # Run img2img
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            None,
            pipeline.img2img_sync,
            request.prompt,
            input_image,
            seed,
            request.denoise,
            request.steps,
        )
        
        logger.info(f"Modified image: {result_image.size}")
        
        # Convert to PNG base64
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        gen_time = time.time() - start_time
        logger.info(f"Modify complete in {gen_time:.2f}s")
        
        return ImageGenResponse(
            success=True,
            job_id=str(uuid.uuid4()),
            image_base64=image_base64,
            width=result_image.size[0],
            height=result_image.size[1],
            seed=seed,
            prompt=request.prompt,
            generation_time=round(gen_time, 2),
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Image modification failed: {e}")
        logger.error(traceback.format_exc())
        return ImageGenResponse(
            success=False,
            error=str(e),
            prompt=request.prompt,
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    job = queue_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=job.queue_position if job.status == JobStatus.QUEUED else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the result of a completed generation job"""
    job = queue_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.QUEUED:
        raise HTTPException(status_code=202, detail="Job is queued")
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job is processing")
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Generation failed")
    
    return ImageGenResponse(
        success=True,
        job_id=job.id,
        image_base64=job.result_base64,
        width=job.width,
        height=job.height,
        seed=job.seed,
        prompt=job.prompt,
        generation_time=round(job.completed_at - job.started_at, 2) if job.completed_at and job.started_at else None,
    )


@app.get("/job/{job_id}/image")
async def get_job_image(job_id: str):
    """Get the generated image as PNG"""
    job = queue_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=202, detail="Image not ready")
    
    return Response(
        content=job.result_image,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename=generated_{job_id[:8]}.png",
            "X-Seed": str(job.seed),
            "X-Prompt": job.prompt[:100],
        }
    )


# ============ Main ============

if __name__ == "__main__":
    import uvicorn
    
    # Default to localhost only - prevents unauthorized external access
    # Image generation must go through the authenticated backend API
    host = os.getenv("IMAGE_GEN_HOST", "127.0.0.1")
    port = int(os.getenv("IMAGE_GEN_PORT", "8034"))
    
    # Custom log config to suppress access logs
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "WARNING"},
            "uvicorn.error": {"level": "WARNING"},
            "uvicorn.access": {"handlers": [], "level": "WARNING", "propagate": False},
        },
    }
    
    uvicorn.run(app, host=host, port=port, log_config=log_config)
