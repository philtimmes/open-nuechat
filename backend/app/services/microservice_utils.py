"""
Shared microservice utilities for TTS and Image Generation services

Provides:
- Structured error responses
- Input validation helpers
- Metrics tracking
- Health check utilities
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standardized error codes for microservices"""
    VALIDATION_ERROR = "validation_error"
    QUEUE_FULL = "queue_full"
    JOB_NOT_FOUND = "job_not_found"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INVALID_INPUT = "invalid_input"
    SERVICE_UNAVAILABLE = "service_unavailable"


class ErrorResponse(BaseModel):
    """Standardized error response format"""
    error: str
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None


class HealthStatus(str, Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceMetrics:
    """Track service metrics for monitoring"""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    total_processing_time_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def record_request(self, success: bool, processing_time_ms: float):
        """Record a completed request"""
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        self.total_processing_time_ms += processing_time_ms
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.requests_total == 0:
            return 100.0
        return (self.requests_success / self.requests_total) * 100
    
    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time"""
        if self.requests_success == 0:
            return 0.0
        return self.total_processing_time_ms / self.requests_success
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary"""
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "success_rate_percent": round(self.success_rate, 2),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "uptime_seconds": round(self.uptime_seconds, 2),
        }
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = [
            f"# HELP requests_total Total number of requests",
            f"# TYPE requests_total counter",
            f"requests_total {self.requests_total}",
            f"# HELP requests_success Total successful requests",
            f"# TYPE requests_success counter",
            f"requests_success {self.requests_success}",
            f"# HELP requests_failed Total failed requests",
            f"# TYPE requests_failed counter",
            f"requests_failed {self.requests_failed}",
            f"# HELP avg_processing_time_ms Average processing time in milliseconds",
            f"# TYPE avg_processing_time_ms gauge",
            f"avg_processing_time_ms {round(self.avg_processing_time_ms, 2)}",
            f"# HELP uptime_seconds Service uptime in seconds",
            f"# TYPE uptime_seconds gauge",
            f"uptime_seconds {round(self.uptime_seconds, 2)}",
        ]
        return "\n".join(lines)


def validate_text_length(text: str, max_length: int) -> tuple[bool, Optional[str]]:
    """
    Validate text length for TTS/processing.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or len(text.strip()) == 0:
        return False, "Text cannot be empty"
    
    if len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters"
    
    return True, None


def validate_dimensions(
    width: int, 
    height: int, 
    min_dim: int = 256, 
    max_dim: int = 2048,
    max_pixels: int = 4_000_000,
    round_to: int = 64
) -> tuple[int, int, Optional[str]]:
    """
    Validate and normalize image dimensions.
    
    Args:
        width: Requested width
        height: Requested height
        min_dim: Minimum dimension
        max_dim: Maximum dimension
        max_pixels: Maximum total pixels
        round_to: Round dimensions to multiple of this value
    
    Returns:
        Tuple of (adjusted_width, adjusted_height, warning_message)
    """
    warning = None
    
    # Clamp to range
    if width < min_dim or height < min_dim:
        warning = f"Dimensions clamped to minimum {min_dim}"
        width = max(width, min_dim)
        height = max(height, min_dim)
    
    if width > max_dim or height > max_dim:
        warning = f"Dimensions clamped to maximum {max_dim}"
        width = min(width, max_dim)
        height = min(height, max_dim)
    
    # Check aspect ratio (max 4:1)
    aspect = max(width, height) / min(width, height)
    if aspect > 4:
        warning = "Aspect ratio clamped to 4:1"
        if width > height:
            width = height * 4
        else:
            height = width * 4
    
    # Check total pixels
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        warning = f"Dimensions scaled down to fit {max_pixels} pixel limit"
    
    # Round to multiple
    width = (width // round_to) * round_to
    height = (height // round_to) * round_to
    
    # Ensure minimum after rounding
    width = max(width, min_dim)
    height = max(height, min_dim)
    
    return width, height, warning


def validate_seed(seed: Optional[int], max_seed: int = 2**32 - 1) -> int:
    """
    Validate and normalize random seed.
    
    Args:
        seed: Requested seed (None for random)
        max_seed: Maximum seed value
    
    Returns:
        Valid seed value
    """
    import random
    
    if seed is None:
        return random.randint(0, max_seed)
    
    return max(0, min(seed, max_seed))


def validate_steps(steps: int, min_steps: int = 1, max_steps: int = 50) -> int:
    """
    Validate inference steps.
    
    Args:
        steps: Requested number of steps
        min_steps: Minimum steps
        max_steps: Maximum steps
    
    Returns:
        Clamped steps value
    """
    return max(min_steps, min(steps, max_steps))


def create_error_response(
    code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    retry_after: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        code: Error code enum
        message: Human-readable error message
        details: Additional error details
        retry_after: Seconds to wait before retrying
    
    Returns:
        Error response dictionary
    """
    response = {
        "error": code.value,
        "code": code.value,
        "message": message,
    }
    
    if details:
        response["details"] = details
    
    if retry_after:
        response["retry_after"] = retry_after
    
    return response


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU information if available.
    
    Returns:
        Dictionary with GPU info or None
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        props = torch.cuda.get_device_properties(0)
        info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(props.total_memory / (1024**3), 2),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
        }
        
        # Check for ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip:
            info["backend"] = "rocm"
            info["rocm_version"] = torch.version.hip
        else:
            info["backend"] = "cuda"
            info["cuda_version"] = torch.version.cuda
        
        return info
    except Exception:
        return None


def determine_health_status(
    queue_size: int,
    max_queue: int,
    gpu_available: bool,
    expected_gpu: bool
) -> HealthStatus:
    """
    Determine overall service health status.
    
    Args:
        queue_size: Current queue size
        max_queue: Maximum queue size
        gpu_available: Whether GPU is available
        expected_gpu: Whether GPU is expected to be available
    
    Returns:
        Health status enum
    """
    # Check for GPU issues
    if expected_gpu and not gpu_available:
        return HealthStatus.UNHEALTHY
    
    # Check queue load
    queue_load = queue_size / max_queue if max_queue > 0 else 0
    
    if queue_load >= 0.9:
        return HealthStatus.DEGRADED
    
    return HealthStatus.HEALTHY
