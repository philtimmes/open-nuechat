"""
Kokoro TTS Microservice with Request Queuing
Runs on Python 3.12 via pyenv for Kokoro compatibility
Supports ROCm GPU acceleration when available
"""
import asyncio
import io
import logging
import os
import uuid
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np

# Configure logging - silence most logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Silence uvicorn access logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# ============ GPU Detection ============

def detect_device():
    """
    Detect available compute device.
    ROCm uses torch.cuda namespace (maps to HIP backend).
    """
    import torch
    
    use_gpu = os.getenv("TTS_USE_GPU", "true").lower() in ("true", "1", "yes")
    
    if use_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / (1024**3)
        logger.info(f"GPU detected: {device_name} ({vram_gb:.1f} GB VRAM)")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check if ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip:
            logger.info(f"ROCm/HIP version: {torch.version.hip}")
        
        return "cuda"
    else:
        if use_gpu:
            logger.warning("GPU requested but not available, falling back to CPU")
        else:
            logger.info("GPU disabled by TTS_USE_GPU=false")
        logger.info(f"Using CPU for TTS inference")
        return "cpu"

# Detect device at module load
DEVICE = detect_device()

# ============ Configuration ============

MAX_QUEUE_SIZE = int(os.getenv("TTS_MAX_QUEUE_SIZE", "100"))
MAX_CONCURRENT = int(os.getenv("TTS_MAX_CONCURRENT", "2"))
RESULT_TTL_SECONDS = int(os.getenv("TTS_RESULT_TTL", "300"))  # 5 minutes
MAX_TEXT_LENGTH = int(os.getenv("TTS_MAX_TEXT_LENGTH", "10000"))

# ============ Models ============

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TTSJob:
    id: str
    text: str
    voice: str
    speed: float = 1.0
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[bytes] = None
    error: Optional[str] = None
    queue_position: int = 0


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str = Field(default="af_heart")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")


class TTSJobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: Optional[int] = None
    message: str


class TTSStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: Optional[int] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


# ============ Kokoro Pipeline ============

class KokoroPipeline:
    """Manages the Kokoro TTS pipeline with GPU support"""
    
    SAMPLE_RATE = 24000
    
    def __init__(self, device: str = "cpu"):
        self._pipeline = None
        self._lock = asyncio.Lock()
        self._device = device
    
    async def initialize(self):
        """Initialize the Kokoro pipeline"""
        if self._pipeline is not None:
            return
        
        async with self._lock:
            if self._pipeline is not None:
                return
            
            logger.info(f"Initializing Kokoro TTS pipeline on {self._device}...")
            try:
                import torch
                from kokoro import KPipeline
                
                # Set default device for PyTorch tensors
                if self._device == "cuda" and torch.cuda.is_available():
                    torch.set_default_device("cuda")
                    logger.info("PyTorch default device set to CUDA/ROCm")
                
                # Initialize pipeline - Kokoro uses PyTorch internally
                self._pipeline = KPipeline(lang_code='a')  # American English
                
                # Log model device placement
                logger.info(f"Kokoro TTS pipeline initialized on {self._device}")
                
                # Warm up with a short phrase to ensure model is loaded to GPU
                if self._device == "cuda":
                    logger.info("Warming up GPU with test inference...")
                    try:
                        # Short warmup
                        for _ in self._pipeline("Hello", voice="af_heart"):
                            pass
                        logger.info("GPU warmup complete")
                    except Exception as e:
                        logger.warning(f"GPU warmup failed: {e}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro: {e}")
                raise
    
    def generate_sync(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        """Generate speech synchronously (called from thread pool)"""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(self._pipeline(text, voice=voice, speed=speed)):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            raise ValueError("No audio generated")
        
        full_audio = np.concatenate(audio_chunks)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, self.SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        
        return buffer.read()
    
    def generate_chunks_sync(self, text: str, voice: str, speed: float = 1.0):
        """
        Generator that yields audio chunks as they're generated.
        Uses Kokoro's native split_pattern for optimal streaming.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Use aggressive split pattern for faster first-chunk delivery
        # Split on sentences, commas, semicolons, colons, and newlines
        split_pattern = r'[.!?;:,]\s+|\n+'
        
        for i, (gs, ps, audio) in enumerate(self._pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)):
            # Convert chunk to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            yield buffer.read()
    
    async def generate(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        """Generate speech asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, text, voice, speed)
    
    async def generate_stream(self, text: str, voice: str, speed: float = 1.0):
        """
        Async generator that yields audio chunks as they're generated.
        Uses a queue to bridge sync generator with async iteration.
        """
        import queue
        import threading
        
        chunk_queue = queue.Queue()
        error_holder = [None]
        timing = {"start": time.time(), "first_chunk": None, "count": 0}
        
        def producer():
            try:
                for chunk in self.generate_chunks_sync(text, voice, speed):
                    if timing["first_chunk"] is None:
                        timing["first_chunk"] = time.time()
                        elapsed = (timing["first_chunk"] - timing["start"]) * 1000
                        logger.warning(f"TTS producer: first chunk generated at {elapsed:.0f}ms")
                    timing["count"] += 1
                    chunk_queue.put(chunk)
            except Exception as e:
                error_holder[0] = e
            finally:
                elapsed = (time.time() - timing["start"]) * 1000
                logger.warning(f"TTS producer: done, {timing['count']} chunks in {elapsed:.0f}ms")
                chunk_queue.put(None)  # Sentinel
        
        # Start producer in background thread
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        
        # Yield chunks as they become available
        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, chunk_queue.get)
            if chunk is None:
                break
            if error_holder[0]:
                raise error_holder[0]
            yield chunk


# ============ Job Queue Manager ============

class TTSQueueManager:
    """Manages TTS job queue and processing"""
    
    def __init__(self, pipeline: KokoroPipeline, max_queue: int, max_concurrent: int):
        self.pipeline = pipeline
        self.max_queue = max_queue
        self.max_concurrent = max_concurrent
        
        self.jobs: Dict[str, TTSJob] = OrderedDict()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.processing_count = 0
        self.lock = asyncio.Lock()
        
        # Start worker tasks
        self._workers = []
        self._cleanup_task = None
    
    async def start(self):
        """Start worker tasks"""
        # Initialize pipeline
        await self.pipeline.initialize()
        
        # Start workers
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_results())
        
        logger.info(f"Started {self.max_concurrent} TTS workers")
    
    async def stop(self):
        """Stop all workers"""
        for worker in self._workers:
            worker.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def submit(self, text: str, voice: str, speed: float = 1.0) -> TTSJob:
        """Submit a new TTS job"""
        async with self.lock:
            if self.queue.full():
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue full. Maximum {self.max_queue} pending jobs."
                )
            
            job_id = str(uuid.uuid4())
            job = TTSJob(
                id=job_id,
                text=text,
                voice=voice,
                speed=speed,
                queue_position=self.queue.qsize() + 1
            )
            
            self.jobs[job_id] = job
            await self.queue.put(job_id)
            
            logger.info(f"Job {job_id} queued at position {job.queue_position}")
            return job
    
    def get_job(self, job_id: str) -> Optional[TTSJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get current queue position for a job"""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return None
        
        # Count jobs ahead in queue
        position = 1
        for jid, j in self.jobs.items():
            if jid == job_id:
                break
            if j.status == JobStatus.QUEUED:
                position += 1
        return position
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job. Returns True if cancelled, False if not found or already done."""
        async with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            # Can only cancel queued jobs (processing jobs will finish)
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                job.error = "Cancelled by user"
                logger.info(f"Job {job_id} cancelled")
                return True
            
            return False
    
    async def cancel_all_for_user(self) -> int:
        """Cancel all queued jobs. Returns count cancelled."""
        cancelled = 0
        async with self.lock:
            for job in self.jobs.values():
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = time.time()
                    job.error = "Cancelled by user"
                    cancelled += 1
        if cancelled:
            logger.info(f"Cancelled {cancelled} queued jobs")
        return cancelled
    
    async def _worker(self, worker_id: int):
        """Worker task that processes jobs from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get next job from queue
                job_id = await self.queue.get()
                job = self.jobs.get(job_id)
                
                if not job:
                    self.queue.task_done()
                    continue
                
                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    self.queue.task_done()
                    continue
                
                # Update status
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()
                
                logger.info(f"Worker {worker_id} processing job {job_id}")
                
                try:
                    # Generate TTS
                    audio_bytes = await self.pipeline.generate(job.text, job.voice, job.speed)
                    
                    job.result = audio_bytes
                    job.status = JobStatus.COMPLETED
                    job.completed_at = time.time()
                    
                    duration = job.completed_at - job.started_at
                    logger.info(f"Job {job_id} completed in {duration:.2f}s")
                    
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = time.time()
                    logger.error(f"Job {job_id} failed: {e}")
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _cleanup_old_results(self):
        """Periodically clean up old completed jobs"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = time.time()
                to_remove = []
                
                async with self.lock:
                    for job_id, job in self.jobs.items():
                        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                            if job.completed_at and (now - job.completed_at) > RESULT_TTL_SECONDS:
                                to_remove.append(job_id)
                    
                    for job_id in to_remove:
                        del self.jobs[job_id]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old jobs")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# ============ Available Voices ============

AVAILABLE_VOICES = [
    # American English voices
    {"id": "af_heart", "name": "Heart (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_bella", "name": "Bella (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_jessica", "name": "Jessica (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_kore", "name": "Kore (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_nova", "name": "Nova (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_river", "name": "River (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_sky", "name": "Sky (Female)", "lang": "en-us", "gender": "female"},
    {"id": "am_adam", "name": "Adam (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_echo", "name": "Echo (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_eric", "name": "Eric (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_fenrir", "name": "Fenrir (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_liam", "name": "Liam (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_michael", "name": "Michael (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_onyx", "name": "Onyx (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_puck", "name": "Puck (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_santa", "name": "Santa (Male)", "lang": "en-us", "gender": "male"},
    # British English voices
    {"id": "bf_alice", "name": "Alice (British Female)", "lang": "en-gb", "gender": "female"},
    {"id": "bf_emma", "name": "Emma (British Female)", "lang": "en-gb", "gender": "female"},
    {"id": "bf_isabella", "name": "Isabella (British Female)", "lang": "en-gb", "gender": "female"},
    {"id": "bf_lily", "name": "Lily (British Female)", "lang": "en-gb", "gender": "female"},
    {"id": "bm_daniel", "name": "Daniel (British Male)", "lang": "en-gb", "gender": "male"},
    {"id": "bm_fable", "name": "Fable (British Male)", "lang": "en-gb", "gender": "male"},
    {"id": "bm_george", "name": "George (British Male)", "lang": "en-gb", "gender": "male"},
    {"id": "bm_lewis", "name": "Lewis (British Male)", "lang": "en-gb", "gender": "male"},
]


# ============ FastAPI App ============

app = FastAPI(
    title="Kokoro TTS Service",
    description="Text-to-Speech microservice using Kokoro with request queuing",
    version="1.0.0"
)

# Global instances
pipeline = KokoroPipeline(device=DEVICE)
queue_manager: Optional[TTSQueueManager] = None


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global queue_manager
    queue_manager = TTSQueueManager(pipeline, MAX_QUEUE_SIZE, MAX_CONCURRENT)
    await queue_manager.start()
    logger.info("TTS Service started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if queue_manager:
        await queue_manager.stop()
    logger.info("TTS Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_info = None
    if DEVICE == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
        }
        # Check for ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip:
            gpu_info["rocm_version"] = torch.version.hip
    
    return {
        "status": "healthy",
        "service": "kokoro-tts",
        "device": DEVICE,
        "gpu": gpu_info,
        "queue_size": queue_manager.queue.qsize() if queue_manager else 0,
        "max_queue": MAX_QUEUE_SIZE,
        "max_concurrent": MAX_CONCURRENT
    }


@app.get("/voices")
async def list_voices(
    gender: Optional[str] = None,
    lang: Optional[str] = None
):
    """List available voices"""
    voices = AVAILABLE_VOICES
    
    if gender:
        voices = [v for v in voices if v["gender"] == gender.lower()]
    
    if lang:
        voices = [v for v in voices if v["lang"] == lang.lower()]
    
    return {"voices": voices}


@app.post("/tts/submit", response_model=TTSJobResponse)
async def submit_tts_job(request: TTSRequest):
    """
    Submit a TTS job to the queue.
    Returns a job ID to poll for results.
    """
    # Validate voice
    valid_voices = [v["id"] for v in AVAILABLE_VOICES]
    if request.voice not in valid_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{request.voice}'. Use /voices to see available options."
        )
    
    job = await queue_manager.submit(request.text, request.voice, request.speed)
    
    return TTSJobResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=job.queue_position,
        message=f"Job queued at position {job.queue_position}"
    )


@app.get("/tts/status/{job_id}", response_model=TTSStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a TTS job"""
    job = queue_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TTSStatusResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=queue_manager.get_queue_position(job_id),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error
    )


@app.get("/tts/result/{job_id}")
async def get_job_result(job_id: str):
    """
    Get the result of a completed TTS job.
    Returns WAV audio data.
    """
    job = queue_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.QUEUED:
        raise HTTPException(
            status_code=202,
            detail=f"Job still queued at position {queue_manager.get_queue_position(job_id)}"
        )
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job still processing")
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
    
    if not job.result:
        raise HTTPException(status_code=500, detail="No result available")
    
    return Response(
        content=job.result,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="tts_{job_id}.wav"'
        }
    )


@app.post("/tts/generate")
async def generate_tts_sync(request: TTSRequest):
    """
    Synchronous TTS generation (blocks until complete).
    Use /tts/submit for async/queued requests.
    """
    # Validate voice
    valid_voices = [v["id"] for v in AVAILABLE_VOICES]
    if request.voice not in valid_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{request.voice}'"
        )
    
    try:
        audio_bytes = await pipeline.generate(request.text, request.voice, request.speed)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="tts_output.wav"'
            }
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import StreamingResponse

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    """
    Streaming TTS generation - returns audio chunks as they're generated.
    Each chunk is a complete WAV file for immediate playback.
    
    Uses multipart response with each part being a WAV chunk.
    """
    # Validate voice
    valid_voices = [v["id"] for v in AVAILABLE_VOICES]
    if request.voice not in valid_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{request.voice}'"
        )
    
    async def generate_chunks():
        """Async generator for streaming response"""
        try:
            chunk_index = 0
            start_time = time.time()
            async for audio_chunk in pipeline.generate_stream(request.text, request.voice, request.speed):
                # Yield chunk with boundary marker for client parsing
                # Format: 4-byte length prefix + WAV data
                length_bytes = len(audio_chunk).to_bytes(4, byteorder='big')
                elapsed = (time.time() - start_time) * 1000
                if chunk_index == 0:
                    logger.warning(f"TTS first chunk ready at {elapsed:.0f}ms")
                yield length_bytes + audio_chunk
                chunk_index += 1
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"TTS complete: {chunk_index} chunks in {elapsed:.0f}ms")
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise
    
    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={
            "X-Content-Type": "audio/wav-stream",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        }
    )


@app.get("/queue/stats")
async def get_queue_stats():
    """Get queue statistics"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    queued = sum(1 for j in queue_manager.jobs.values() if j.status == JobStatus.QUEUED)
    processing = sum(1 for j in queue_manager.jobs.values() if j.status == JobStatus.PROCESSING)
    completed = sum(1 for j in queue_manager.jobs.values() if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in queue_manager.jobs.values() if j.status == JobStatus.FAILED)
    cancelled = sum(1 for j in queue_manager.jobs.values() if j.status == JobStatus.CANCELLED)
    
    return {
        "queued": queued,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "total_jobs": len(queue_manager.jobs),
        "max_queue": MAX_QUEUE_SIZE,
        "max_concurrent": MAX_CONCURRENT
    }


@app.post("/tts/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a specific TTS job"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    cancelled = await queue_manager.cancel_job(job_id)
    
    if cancelled:
        return {"status": "cancelled", "job_id": job_id}
    else:
        job = queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": job.status.value, "job_id": job_id, "message": "Job already processing or completed"}


@app.post("/tts/cancel-all")
async def cancel_all_jobs():
    """Cancel all queued TTS jobs"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    count = await queue_manager.cancel_all_for_user()
    return {"status": "ok", "cancelled": count}


if __name__ == "__main__":
    import uvicorn
    
    # Default to localhost only - prevents unauthorized external access
    # TTS must go through the authenticated backend API
    host = os.getenv("TTS_HOST", "127.0.0.1")
    port = int(os.getenv("TTS_PORT", "8033"))
    
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
