"""
Text-to-Speech API routes - RPC client to Kokoro TTS microservice
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.api.dependencies import get_current_user
from app.models.models import User
from app.services.tts import get_tts_service, get_available_voices

logger = logging.getLogger(__name__)
router = APIRouter(tags=["TTS"])


class TTSRequest(BaseModel):
    """Request body for TTS generation"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to convert to speech")
    voice: Optional[str] = Field(default=None, description="Voice ID to use (default: af_heart)")


class VoiceInfo(BaseModel):
    """Voice information"""
    id: str
    name: str
    lang: str
    gender: str


class TTSStatusResponse(BaseModel):
    """TTS service status"""
    available: bool
    backend: str
    queue_stats: Optional[dict] = None


@router.get("/status", response_model=TTSStatusResponse)
async def get_tts_status(
    user: User = Depends(get_current_user),
):
    """Check if TTS service is available"""
    tts = get_tts_service()
    available = await tts.is_available()
    
    queue_stats = None
    if available:
        try:
            queue_stats = await tts.get_queue_stats()
        except Exception:
            pass
    
    return {
        "available": available,
        "backend": tts.get_backend_name(),
        "queue_stats": queue_stats
    }


@router.get("/voices", response_model=List[VoiceInfo])
async def list_voices(
    user: User = Depends(get_current_user),
    gender: Optional[str] = Query(None, description="Filter by gender (male/female)"),
    lang: Optional[str] = Query(None, description="Filter by language (e.g., en-us, en-gb)"),
):
    """List available TTS voices"""
    voices = await get_available_voices()
    
    if gender:
        voices = [v for v in voices if v["gender"] == gender.lower()]
    
    if lang:
        voices = [v for v in voices if v["lang"].lower() == lang.lower()]
    
    return voices


@router.post("/generate")
async def generate_speech(
    request: TTSRequest,
    user: User = Depends(get_current_user),
):
    """
    Generate speech from text.
    
    Returns WAV audio file.
    """
    tts = get_tts_service()
    
    if not await tts.is_available():
        raise HTTPException(
            status_code=503,
            detail="TTS service is not available. Check that tts-service container is running."
        )
    
    try:
        audio_bytes = await tts.generate_speech(
            text=request.text,
            voice=request.voice
        )
        
        # Generate filename from first few words
        filename_base = request.text[:30].replace(" ", "_").replace("/", "").replace("\\", "")
        filename = f"tts_{filename_base}.wav"
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.post("/stream")
async def stream_speech(
    request: TTSRequest,
    user: User = Depends(get_current_user),
):
    """
    Stream speech generation - returns audio chunks as they're generated.
    
    Each chunk is a complete WAV segment that can be played immediately.
    Response is binary with 4-byte length prefix for each chunk.
    """
    tts = get_tts_service()
    
    if not await tts.is_available():
        raise HTTPException(
            status_code=503,
            detail="TTS service is not available. Check that tts-service container is running."
        )
    
    try:
        async def stream_chunks():
            async for chunk in tts.stream_speech(
                text=request.text,
                voice=request.voice
            ):
                yield chunk
        
        return StreamingResponse(
            stream_chunks(),
            media_type="application/octet-stream",
            headers={
                "X-Content-Type": "audio/wav-stream",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Connection": "keep-alive",
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")


@router.get("/preview/{voice_id}")
async def preview_voice(
    voice_id: str,
    user: User = Depends(get_current_user),
):
    """
    Generate a short preview of a voice.
    """
    voices = await get_available_voices()
    valid_voices = [v["id"] for v in voices]
    
    if voice_id not in valid_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    tts = get_tts_service()
    
    if not await tts.is_available():
        raise HTTPException(status_code=503, detail="TTS service is not available")
    
    # Get voice name for preview text
    voice_info = next(v for v in voices if v["id"] == voice_id)
    voice_name = voice_info['name'].split(' (')[0]
    preview_text = f"Hello! I'm {voice_name}, nice to meet you."
    
    try:
        audio_bytes = await tts.generate_speech(
            text=preview_text,
            voice=voice_id
        )
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )
    
    except Exception as e:
        logger.error(f"Voice preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


@router.get("/queue/stats")
async def get_queue_stats(
    user: User = Depends(get_current_user),
):
    """Get TTS queue statistics"""
    tts = get_tts_service()
    
    if not await tts.is_available():
        raise HTTPException(status_code=503, detail="TTS service is not available")
    
    try:
        stats = await tts.get_queue_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
