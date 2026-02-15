"""
Speech-to-Text service using OpenAI Whisper (whisper-large-v3-turbo)
Optimized for AMD ROCm platform
"""
import io
import logging
import tempfile
import os
from typing import Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Disable CUDA graphs - they cause recompilation with variable-length audio
# This must be set before importing torch
os.environ.setdefault("TORCH_CUDAGRAPHS", "0")
os.environ.setdefault("TORCHINDUCTOR_CUDA_GRAPH", "0")

# Global instances (lazy loaded)
_pipeline = None
_processor = None
_model = None
_executor = ThreadPoolExecutor(max_workers=2)
_is_warmed_up = False
_compile_attempted = False


def _get_device_and_dtype():
    """Get appropriate device and dtype for the platform (ROCm/HIP)"""
    import torch
    
    # ROCm uses the same torch.cuda API but with HIP backend
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
        # Log device info (works for both CUDA and ROCm/HIP)
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU (ROCm/HIP): {device_name}")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU (GPU not available)")
    
    return device, torch_dtype


def _get_pipeline():
    """Lazy load the Whisper pipeline"""
    global _pipeline, _processor, _model, _is_warmed_up
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        import torch
        import os
        
        # CRITICAL: Disable accelerate's device_map which causes meta tensor issues
        os.environ["ACCELERATE_DISABLE_RICH"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["ACCELERATE_USE_CPU"] = "1"
        
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        # Disable dynamo/inductor CUDA graphs for variable-length inputs
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 1  # Minimize recompilation
        
        # Set precision for better performance
        torch.set_float32_matmul_precision("high")
        
        device, torch_dtype = _get_device_and_dtype()
        
        model_id = "openai/whisper-large-v3-turbo"
        
        logger.info(f"Loading Whisper model: {model_id}")
        
        # Load model
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )
        
        # Move to device - handle meta tensors from newer transformers
        try:
            _model = _model.to(device)
        except NotImplementedError:
            # Meta tensor error - use to_empty() then load weights properly
            _model = _model.to_empty(device=device)
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                device_map=str(device),
            )
        
        # Enable static cache for faster generation
        _model.generation_config.cache_implementation = "static"
        _model.generation_config.max_new_tokens = 256
        
        # Skip torch.compile - it causes repeated CUDA graph compilation 
        # attempts with variable-length audio inputs, adding latency
        # The uncompiled model is fast enough on modern GPUs
        logger.info("Using uncompiled model (better for variable-length audio)")
        
        # Load processor
        _processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        _pipeline = pipeline(
            "automatic-speech-recognition",
            model=_model,
            tokenizer=_processor.tokenizer,
            feature_extractor=_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        logger.info("Whisper STT pipeline initialized")
        return _pipeline
        
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Whisper STT: {e}")
        raise


def _warmup_model():
    """Warm up the model with audio samples to ensure fast inference"""
    global _is_warmed_up
    
    if _is_warmed_up:
        return
    
    try:
        import torch
        import numpy as np
        
        pipeline = _get_pipeline()
        
        logger.info("Warming up Whisper model...")
        
        # Warm up with a few different audio lengths to exercise code paths
        sample_rate = 16000
        warmup_durations = [1, 3, 5]  # 1, 3, 5 seconds
        
        for duration in warmup_durations:
            # Create silent audio sample
            silent_audio = np.zeros(sample_rate * duration, dtype=np.float32)
            
            # Run inference (ignore output)
            with torch.inference_mode():
                _ = pipeline(silent_audio, return_timestamps=False)
        
        _is_warmed_up = True
        logger.info("Whisper model warmed up successfully")
        
    except Exception as e:
        logger.warning(f"Warmup failed (will warm up on first request): {e}")


# Supported audio formats
SUPPORTED_FORMATS = [
    "audio/wav",
    "audio/wave", 
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
]

# Supported languages for transcription
SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "Chinese"},
    {"code": "de", "name": "German"},
    {"code": "es", "name": "Spanish"},
    {"code": "ru", "name": "Russian"},
    {"code": "ko", "name": "Korean"},
    {"code": "fr", "name": "French"},
    {"code": "ja", "name": "Japanese"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "tr", "name": "Turkish"},
    {"code": "pl", "name": "Polish"},
    {"code": "ca", "name": "Catalan"},
    {"code": "nl", "name": "Dutch"},
    {"code": "ar", "name": "Arabic"},
    {"code": "sv", "name": "Swedish"},
    {"code": "it", "name": "Italian"},
    {"code": "id", "name": "Indonesian"},
    {"code": "hi", "name": "Hindi"},
    {"code": "fi", "name": "Finnish"},
    {"code": "vi", "name": "Vietnamese"},
    {"code": "he", "name": "Hebrew"},
    {"code": "uk", "name": "Ukrainian"},
    {"code": "el", "name": "Greek"},
    {"code": "ms", "name": "Malay"},
    {"code": "cs", "name": "Czech"},
    {"code": "ro", "name": "Romanian"},
    {"code": "da", "name": "Danish"},
    {"code": "hu", "name": "Hungarian"},
    {"code": "ta", "name": "Tamil"},
    {"code": "no", "name": "Norwegian"},
    {"code": "th", "name": "Thai"},
    {"code": "ur", "name": "Urdu"},
    {"code": "hr", "name": "Croatian"},
    {"code": "bg", "name": "Bulgarian"},
    {"code": "lt", "name": "Lithuanian"},
    {"code": "la", "name": "Latin"},
    {"code": "mi", "name": "Maori"},
    {"code": "ml", "name": "Malayalam"},
    {"code": "cy", "name": "Welsh"},
    {"code": "sk", "name": "Slovak"},
    {"code": "te", "name": "Telugu"},
    {"code": "fa", "name": "Persian"},
    {"code": "lv", "name": "Latvian"},
    {"code": "bn", "name": "Bengali"},
    {"code": "sr", "name": "Serbian"},
    {"code": "az", "name": "Azerbaijani"},
    {"code": "sl", "name": "Slovenian"},
    {"code": "kn", "name": "Kannada"},
    {"code": "et", "name": "Estonian"},
    {"code": "mk", "name": "Macedonian"},
    {"code": "br", "name": "Breton"},
    {"code": "eu", "name": "Basque"},
    {"code": "is", "name": "Icelandic"},
    {"code": "hy", "name": "Armenian"},
    {"code": "ne", "name": "Nepali"},
    {"code": "mn", "name": "Mongolian"},
    {"code": "bs", "name": "Bosnian"},
    {"code": "kk", "name": "Kazakh"},
    {"code": "sq", "name": "Albanian"},
    {"code": "sw", "name": "Swahili"},
    {"code": "gl", "name": "Galician"},
    {"code": "mr", "name": "Marathi"},
    {"code": "pa", "name": "Punjabi"},
    {"code": "si", "name": "Sinhala"},
    {"code": "km", "name": "Khmer"},
    {"code": "sn", "name": "Shona"},
    {"code": "yo", "name": "Yoruba"},
    {"code": "so", "name": "Somali"},
    {"code": "af", "name": "Afrikaans"},
    {"code": "oc", "name": "Occitan"},
    {"code": "ka", "name": "Georgian"},
    {"code": "be", "name": "Belarusian"},
    {"code": "tg", "name": "Tajik"},
    {"code": "sd", "name": "Sindhi"},
    {"code": "gu", "name": "Gujarati"},
    {"code": "am", "name": "Amharic"},
    {"code": "yi", "name": "Yiddish"},
    {"code": "lo", "name": "Lao"},
    {"code": "uz", "name": "Uzbek"},
    {"code": "fo", "name": "Faroese"},
    {"code": "ht", "name": "Haitian Creole"},
    {"code": "ps", "name": "Pashto"},
    {"code": "tk", "name": "Turkmen"},
    {"code": "nn", "name": "Nynorsk"},
    {"code": "mt", "name": "Maltese"},
    {"code": "sa", "name": "Sanskrit"},
    {"code": "lb", "name": "Luxembourgish"},
    {"code": "my", "name": "Myanmar"},
    {"code": "bo", "name": "Tibetan"},
    {"code": "tl", "name": "Tagalog"},
    {"code": "mg", "name": "Malagasy"},
    {"code": "as", "name": "Assamese"},
    {"code": "tt", "name": "Tatar"},
    {"code": "haw", "name": "Hawaiian"},
    {"code": "ln", "name": "Lingala"},
    {"code": "ha", "name": "Hausa"},
    {"code": "ba", "name": "Bashkir"},
    {"code": "jw", "name": "Javanese"},
    {"code": "su", "name": "Sundanese"},
]


class STTService:
    """Speech-to-Text service using Whisper"""
    
    SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
    
    def __init__(self):
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load pipeline on first use"""
        if self._pipeline is None:
            self._pipeline = _get_pipeline()
        return self._pipeline
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return SUPPORTED_LANGUAGES
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return SUPPORTED_FORMATS
    
    def _load_audio(self, audio_bytes: bytes, content_type: str) -> Any:
        """Load audio from bytes into format suitable for Whisper"""
        import soundfile as sf
        import numpy as np
        import subprocess
        
        # Determine file extension from content type
        suffix = ".wav"
        if "mp3" in content_type or "mpeg" in content_type:
            suffix = ".mp3"
        elif "ogg" in content_type:
            suffix = ".ogg"
        elif "flac" in content_type:
            suffix = ".flac"
        elif "m4a" in content_type or "mp4" in content_type:
            suffix = ".m4a"
        elif "webm" in content_type:
            suffix = ".webm"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        converted_path = None
        try:
            # For webm/ogg/mp3 files, convert to wav using ffmpeg first
            # soundfile has limited codec support
            if suffix in [".webm", ".ogg", ".mp3", ".m4a"]:
                converted_path = temp_path.replace(suffix, ".wav")
                try:
                    result = subprocess.run([
                        "ffmpeg", "-y", "-i", temp_path,
                        "-ar", "16000",  # Resample to 16kHz
                        "-ac", "1",      # Convert to mono
                        "-f", "wav",
                        converted_path
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and os.path.exists(converted_path):
                        temp_path = converted_path
                        # Read converted file directly
                        audio, sample_rate = sf.read(temp_path)
                        return audio.astype(np.float32)
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    logger.warning(f"FFmpeg conversion failed, trying direct load: {e}")
            
            # Load audio directly (for wav/flac or if ffmpeg fails)
            audio, sample_rate = sf.read(temp_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != self.SAMPLE_RATE:
                import scipy.signal as signal
                num_samples = int(len(audio) * self.SAMPLE_RATE / sample_rate)
                audio = signal.resample(audio, num_samples)
            
            return audio.astype(np.float32)
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if converted_path and os.path.exists(converted_path):
                os.unlink(converted_path)
    
    def _transcribe_sync(
        self, 
        audio: Any,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """Synchronous transcription (runs in thread pool)"""
        import torch
        import numpy as np
        
        # For short audio (<30s), use pipeline directly
        # For long audio, use model.generate() with chunking
        duration_s = len(audio) / self.SAMPLE_RATE
        
        if duration_s <= 30:
            pipeline_obj = self.pipeline
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            if task == "translate":
                generate_kwargs["task"] = "translate"
            
            with torch.inference_mode():
                result = pipeline_obj(
                    audio,
                    generate_kwargs=generate_kwargs if generate_kwargs else None,
                    return_timestamps=False,
                )
            return result or {"text": "", "chunks": []}
        
        # Long audio: use model.generate() directly (Whisper's native long-form)
        logger.info(f"[STT] Long audio ({duration_s:.0f}s), using model.generate()")
        
        model = _model
        processor = _processor
        device = next(model.parameters()).device
        torch_dtype = next(model.parameters()).dtype
        
        # Process in 30-second chunks with stride
        chunk_length = 30 * self.SAMPLE_RATE  # 30 seconds
        stride = 5 * self.SAMPLE_RATE  # 5 second overlap
        
        all_text = []
        offset = 0
        
        with torch.inference_mode():
            while offset < len(audio):
                end = min(offset + chunk_length, len(audio))
                chunk = audio[offset:end]
                
                # Pad short final chunks
                if len(chunk) < 400:  # Too short for feature extraction
                    break
                
                input_features = processor(
                    chunk, 
                    sampling_rate=self.SAMPLE_RATE, 
                    return_tensors="pt"
                ).input_features.to(device, dtype=torch_dtype)
                
                generate_kwargs = {}
                if language:
                    generate_kwargs["language"] = language
                if task == "translate":
                    generate_kwargs["task"] = "translate"
                
                predicted_ids = model.generate(
                    input_features, 
                    **generate_kwargs
                )
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                if text and text[0].strip():
                    all_text.append(text[0].strip())
                
                # Advance by chunk minus stride for overlap
                offset += chunk_length - stride
        
        full_text = " ".join(all_text)
        return {"text": full_text, "chunks": []}
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_bytes: Raw audio file bytes
            content_type: MIME type of the audio
            language: Optional language code (auto-detect if not specified)
            task: "transcribe" or "translate" (translate to English)
            
        Returns:
            Dict with 'text' and optionally 'chunks' with timestamps
        """
        if not audio_bytes:
            raise ValueError("Audio data cannot be empty")
        
        # Validate content type
        if content_type not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {content_type}")
        
        # Validate language if provided
        if language:
            valid_codes = [l["code"] for l in SUPPORTED_LANGUAGES]
            if language not in valid_codes:
                raise ValueError(f"Unsupported language: {language}")
        
        # Validate task
        if task not in ["transcribe", "translate"]:
            raise ValueError(f"Invalid task: {task}. Must be 'transcribe' or 'translate'")
        
        # Load and process audio
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            _executor,
            self._load_audio,
            audio_bytes,
            content_type
        )
        
        # Run transcription in thread pool
        result = await loop.run_in_executor(
            _executor,
            self._transcribe_sync,
            audio,
            language,
            task
        )
        
        return {
            "text": result.get("text", "").strip(),
            "chunks": result.get("chunks", []),
            "language": language or "auto"
        }
    
    def is_available(self) -> bool:
        """Check if STT service is available"""
        try:
            _ = self.pipeline
            return True
        except Exception:
            return False
    
    async def warmup(self):
        """Warm up the model for faster first inference"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _warmup_model)


# Singleton instance
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    """Get or create STT service singleton"""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
