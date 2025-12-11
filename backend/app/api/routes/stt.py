"""
Speech-to-Text API routes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import asyncio
import json
import io
import tempfile
import subprocess
import os

from app.api.dependencies import get_current_user
from app.models.models import User
from app.services.stt import get_stt_service, SUPPORTED_LANGUAGES, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["STT"])


class LanguageInfo(BaseModel):
    """Language information"""
    code: str
    name: str


class STTStatusResponse(BaseModel):
    """STT service status"""
    available: bool
    model: str
    languages_count: int
    formats_count: int


class TranscriptionResult(BaseModel):
    """Transcription result"""
    text: str
    language: str
    chunks: Optional[List[dict]] = None


@router.get("/status", response_model=STTStatusResponse)
async def get_stt_status(
    user: User = Depends(get_current_user),
):
    """Check if STT service is available"""
    stt = get_stt_service()
    return {
        "available": stt.is_available(),
        "model": "openai/whisper-large-v3-turbo",
        "languages_count": len(SUPPORTED_LANGUAGES),
        "formats_count": len(SUPPORTED_FORMATS)
    }


@router.get("/languages", response_model=List[LanguageInfo])
async def list_languages(
    user: User = Depends(get_current_user),
):
    """List supported languages for transcription"""
    return SUPPORTED_LANGUAGES


@router.get("/formats")
async def list_formats(
    user: User = Depends(get_current_user),
):
    """List supported audio formats"""
    return {"formats": SUPPORTED_FORMATS}


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (auto-detect if not specified)"),
    task: str = Form("transcribe", description="Task: 'transcribe' or 'translate' (to English)"),
    user: User = Depends(get_current_user),
):
    """
    Transcribe audio file to text.
    
    - **file**: Audio file (WAV, MP3, OGG, FLAC, M4A, WebM)
    - **language**: Optional language code (e.g., 'en', 'es', 'zh')
    - **task**: 'transcribe' (keep original language) or 'translate' (translate to English)
    
    Returns transcribed text with optional timestamps.
    """
    stt = get_stt_service()
    
    if not stt.is_available():
        raise HTTPException(
            status_code=503,
            detail="STT service is not available. Ensure Whisper model is loaded."
        )
    
    # Validate content type
    content_type = file.content_type or "audio/wav"
    if content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Read file
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")
    
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    
    # Check file size (limit to 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum size is {max_size // (1024*1024)}MB"
        )
    
    try:
        result = await stt.transcribe(
            audio_bytes=audio_bytes,
            content_type=content_type,
            language=language,
            task=task
        )
        
        return TranscriptionResult(
            text=result["text"],
            language=result["language"],
            chunks=result.get("chunks")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/warmup")
async def warmup_model(
    user: User = Depends(get_current_user),
):
    """
    Warm up the Whisper model for faster first inference.
    
    This preloads the model and runs a test inference.
    Admin/first-request optimization.
    """
    stt = get_stt_service()
    
    if not stt.is_available():
        raise HTTPException(
            status_code=503,
            detail="STT service is not available"
        )
    
    try:
        await stt.warmup()
        return {"status": "ok", "message": "Model warmed up successfully"}
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


class StreamingSTTSession:
    """Manages a streaming STT WebSocket session"""
    
    SILENCE_TIMEOUT = 1.5  # Seconds of silence to trigger transcription
    MAX_AUDIO_LENGTH = 30  # Maximum seconds of audio to accumulate
    SAMPLE_RATE = 16000
    
    def __init__(self, websocket: WebSocket, language: Optional[str] = None):
        self.websocket = websocket
        self.language = language
        self.audio_chunks: List[bytes] = []
        self.last_audio_time = 0.0
        self.is_speaking = False
        self.total_audio_seconds = 0.0
    
    async def process_audio_chunk(self, chunk: bytes) -> Optional[str]:
        """
        Process incoming audio chunk.
        Returns transcript if speech ended, None otherwise.
        """
        import time
        
        current_time = time.time()
        self.audio_chunks.append(chunk)
        self.last_audio_time = current_time
        
        # Estimate audio duration (assuming 16kHz, 16-bit mono)
        chunk_duration = len(chunk) / (self.SAMPLE_RATE * 2)
        self.total_audio_seconds += chunk_duration
        
        # Check if we've hit max length
        if self.total_audio_seconds >= self.MAX_AUDIO_LENGTH:
            return await self._transcribe_accumulated()
        
        return None
    
    async def check_silence_timeout(self) -> Optional[str]:
        """Check if silence timeout reached. Returns transcript if so."""
        import time
        
        if not self.audio_chunks:
            return None
        
        current_time = time.time()
        silence_duration = current_time - self.last_audio_time
        
        if silence_duration >= self.SILENCE_TIMEOUT:
            return await self._transcribe_accumulated()
        
        return None
    
    async def _transcribe_accumulated(self) -> str:
        """Transcribe accumulated audio and reset buffer"""
        if not self.audio_chunks:
            return ""
        
        # Combine all chunks
        combined_audio = b''.join(self.audio_chunks)
        
        # Reset state
        self.audio_chunks = []
        self.total_audio_seconds = 0.0
        
        if len(combined_audio) < 1000:  # Too short, likely noise
            return ""
        
        try:
            # Convert raw PCM to WAV
            wav_bytes = self._pcm_to_wav(combined_audio)
            
            # Transcribe
            stt = get_stt_service()
            result = await stt.transcribe(
                audio_bytes=wav_bytes,
                content_type="audio/wav",
                language=self.language,
                task="transcribe"
            )
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return ""
    
    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM to WAV format"""
        import struct
        
        # WAV header for 16kHz, 16-bit, mono
        num_channels = 1
        sample_width = 2
        frame_rate = self.SAMPLE_RATE
        num_frames = len(pcm_data) // sample_width
        
        wav_buffer = io.BytesIO()
        
        # RIFF header
        wav_buffer.write(b'RIFF')
        wav_buffer.write(struct.pack('<I', 36 + len(pcm_data)))
        wav_buffer.write(b'WAVE')
        
        # fmt chunk
        wav_buffer.write(b'fmt ')
        wav_buffer.write(struct.pack('<I', 16))  # chunk size
        wav_buffer.write(struct.pack('<H', 1))   # audio format (PCM)
        wav_buffer.write(struct.pack('<H', num_channels))
        wav_buffer.write(struct.pack('<I', frame_rate))
        wav_buffer.write(struct.pack('<I', frame_rate * num_channels * sample_width))  # byte rate
        wav_buffer.write(struct.pack('<H', num_channels * sample_width))  # block align
        wav_buffer.write(struct.pack('<H', sample_width * 8))  # bits per sample
        
        # data chunk
        wav_buffer.write(b'data')
        wav_buffer.write(struct.pack('<I', len(pcm_data)))
        wav_buffer.write(pcm_data)
        
        wav_buffer.seek(0)
        return wav_buffer.read()


@router.websocket("/stream")
async def stream_stt(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech-to-text.
    
    Protocol:
    1. Client sends JSON config: {"language": "en"} (optional)
    2. Client sends binary audio chunks (16kHz, 16-bit PCM, mono)
    3. Server sends JSON: {"type": "transcript", "text": "...", "final": true}
    4. Client sends {"type": "end"} to finish
    
    Audio format: Raw PCM, 16kHz sample rate, 16-bit signed, mono
    """
    await websocket.accept()
    
    session = None
    silence_check_task = None
    
    try:
        # Wait for config message
        config_data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        config = json.loads(config_data)
        language = config.get("language")
        
        logger.debug(f"STT stream started, language: {language or 'auto'}")
        
        session = StreamingSTTSession(websocket, language)
        
        # Send ready confirmation
        await websocket.send_json({"type": "ready"})
        
        # Background task to check for silence timeout
        async def silence_checker():
            while True:
                await asyncio.sleep(0.3)
                if session:
                    transcript = await session.check_silence_timeout()
                    if transcript:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "final": True
                        })
        
        silence_check_task = asyncio.create_task(silence_checker())
        
        # Process incoming audio
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Binary audio data
                    transcript = await session.process_audio_chunk(message["bytes"])
                    if transcript:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "final": True
                        })
                        
                elif "text" in message:
                    # Control message
                    try:
                        control = json.loads(message["text"])
                        if control.get("type") == "end":
                            # Force transcribe any remaining audio
                            transcript = await session._transcribe_accumulated()
                            if transcript:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": transcript,
                                    "final": True
                                })
                            break
                    except json.JSONDecodeError:
                        pass
                        
            elif message["type"] == "websocket.disconnect":
                break
                
    except WebSocketDisconnect:
        logger.debug("STT stream disconnected")
    except asyncio.TimeoutError:
        logger.warning("STT stream config timeout")
        await websocket.close(code=1008, reason="Config timeout")
    except Exception as e:
        logger.error(f"STT stream error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if silence_check_task:
            silence_check_task.cancel()
        try:
            await websocket.close()
        except:
            pass
