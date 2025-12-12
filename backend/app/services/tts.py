"""
Text-to-Speech service - RPC client to Kokoro TTS microservice
"""
import logging
import os
from typing import Optional, List, Dict, Any
import asyncio
import httpx

logger = logging.getLogger(__name__)

# TTS Service configuration
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8033")
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "120"))  # 2 minutes for long text
TTS_POLL_INTERVAL = float(os.getenv("TTS_POLL_INTERVAL", "0.5"))  # 500ms


class TTSServiceClient:
    """RPC client for Kokoro TTS microservice"""
    
    def __init__(self, base_url: str = TTS_SERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._voices_cache: Optional[List[dict]] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-init HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(TTS_TIMEOUT)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check TTS service health"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TTS service health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def is_available(self) -> bool:
        """Check if TTS service is available"""
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    async def get_voices(
        self, 
        gender: Optional[str] = None, 
        lang: Optional[str] = None
    ) -> List[dict]:
        """Get available voices from TTS service"""
        try:
            params = {}
            if gender:
                params["gender"] = gender
            if lang:
                params["lang"] = lang
            
            response = await self.client.get("/voices", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("voices", [])
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []
    
    async def generate_sync(self, text: str, voice: str = "af_heart") -> bytes:
        """
        Generate speech synchronously (blocks until complete).
        Good for short text.
        """
        try:
            response = await self.client.post(
                "/tts/generate",
                json={"text": text, "voice": voice}
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"TTS generation failed: {e.response.text}")
            raise RuntimeError(f"TTS generation failed: {e.response.text}")
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise
    
    async def submit_job(self, text: str, voice: str = "af_heart") -> str:
        """
        Submit TTS job to queue.
        Returns job ID.
        """
        try:
            response = await self.client.post(
                "/tts/submit",
                json={"text": text, "voice": voice}
            )
            response.raise_for_status()
            data = response.json()
            return data["job_id"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to submit TTS job: {e.response.text}")
            raise RuntimeError(f"Failed to submit job: {e.response.text}")
        except Exception as e:
            logger.error(f"TTS submit error: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a TTS job"""
        try:
            response = await self.client.get(f"/tts/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise
    
    async def get_job_result(self, job_id: str) -> bytes:
        """Get result of completed TTS job"""
        try:
            response = await self.client.get(f"/tts/result/{job_id}")
            if response.status_code == 202:
                # Still processing
                raise RuntimeError("Job not complete")
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 202:
                raise RuntimeError("Job not complete")
            raise
    
    async def generate_queued(
        self, 
        text: str, 
        voice: str = "af_heart",
        timeout: float = TTS_TIMEOUT
    ) -> bytes:
        """
        Generate speech using queue (for longer text).
        Submits job and polls until complete.
        """
        # Submit job
        job_id = await self.submit_job(text, voice)
        logger.info(f"Submitted TTS job {job_id}")
        
        # Poll for completion
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"TTS job {job_id} timed out after {timeout}s")
            
            status = await self.get_job_status(job_id)
            
            if status["status"] == "completed":
                logger.info(f"TTS job {job_id} completed")
                return await self.get_job_result(job_id)
            
            if status["status"] == "failed":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"TTS job failed: {error}")
            
            # Still queued or processing
            await asyncio.sleep(TTS_POLL_INTERVAL)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            response = await self.client.get("/queue/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific TTS job"""
        try:
            response = await self.client.post(f"/tts/cancel/{job_id}")
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "cancelled"
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def cancel_all_jobs(self) -> int:
        """Cancel all queued TTS jobs"""
        try:
            response = await self.client.post("/tts/cancel-all")
            response.raise_for_status()
            data = response.json()
            return data.get("cancelled", 0)
        except Exception as e:
            logger.error(f"Failed to cancel all jobs: {e}")
            return 0


# ============ Service wrapper for API routes ============

class TTSService:
    """TTS service wrapper for API routes"""
    
    def __init__(self):
        self._client = TTSServiceClient()
        self._voices_cache: Optional[List[dict]] = None
    
    async def get_available_voices(self) -> List[dict]:
        """Get list of available voices (cached)"""
        if self._voices_cache is None:
            self._voices_cache = await self._client.get_voices()
        return self._voices_cache
    
    async def generate_speech(
        self, 
        text: str, 
        voice: Optional[str] = None,
        use_queue: bool = True
    ) -> bytes:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (default: af_heart)
            use_queue: Use queued generation for reliability
            
        Returns:
            WAV audio bytes
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        voice = voice or "af_heart"
        
        # Validate voice
        voices = await self.get_available_voices()
        valid_ids = [v["id"] for v in voices]
        if voice not in valid_ids:
            logger.warning(f"Unknown voice '{voice}', using default")
            voice = "af_heart"
        
        if use_queue:
            return await self._client.generate_queued(text, voice)
        else:
            return await self._client.generate_sync(text, voice)
    
    async def stream_speech(
        self, 
        text: str, 
        voice: Optional[str] = None
    ):
        """
        Stream speech generation - yields audio chunks as they're generated.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (default: af_heart)
            
        Yields:
            Audio chunks (4-byte length prefix + WAV data)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        voice = voice or "af_heart"
        
        # Validate voice
        voices = await self.get_available_voices()
        valid_ids = [v["id"] for v in voices]
        if voice not in valid_ids:
            logger.warning(f"Unknown voice '{voice}', using default")
            voice = "af_heart"
        
        # Stream from TTS service
        async with httpx.AsyncClient(base_url=self._client.base_url) as client:
            async with client.stream(
                "POST",
                "/tts/stream",
                json={"text": text, "voice": voice},
                timeout=TTS_TIMEOUT
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
    
    async def is_available(self) -> bool:
        """Check if TTS service is available"""
        return await self._client.is_available()
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return await self._client.get_queue_stats()
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific TTS job"""
        return await self._client.cancel_job(job_id)
    
    async def cancel_all_jobs(self) -> int:
        """Cancel all queued TTS jobs"""
        return await self._client.cancel_all_jobs()
    
    def get_backend_name(self) -> str:
        """Get backend name"""
        return "kokoro-rpc"


# ============ Singleton ============

_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create TTS service singleton"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


async def get_available_voices() -> List[dict]:
    """Get available voices (convenience function)"""
    service = get_tts_service()
    return await service.get_available_voices()


# For backward compatibility
AVAILABLE_VOICES = [
    {"id": "af_heart", "name": "Heart (Female)", "lang": "en-us", "gender": "female"},
    {"id": "af_bella", "name": "Bella (Female)", "lang": "en-us", "gender": "female"},
    {"id": "am_adam", "name": "Adam (Male)", "lang": "en-us", "gender": "male"},
    {"id": "am_michael", "name": "Michael (Male)", "lang": "en-us", "gender": "male"},
    {"id": "bf_alice", "name": "Alice (British Female)", "lang": "en-gb", "gender": "female"},
    {"id": "bm_daniel", "name": "Daniel (British Male)", "lang": "en-gb", "gender": "male"},
]
