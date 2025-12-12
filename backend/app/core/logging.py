"""
Structured logging utilities

This module provides structured logging capabilities for better
observability and debugging. Key features:

- JSON-formatted log output for parsing
- Context managers for timing operations
- Specialized loggers for LLM requests
- Field inheritance for related log entries
"""
import logging
import json
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager


class StructuredLogger:
    """
    Logger that outputs structured JSON for important events.
    
    Useful for:
    - Log aggregation systems (ELK, Splunk, etc.)
    - Debugging with grep/jq
    - Metrics extraction
    
    Usage:
        logger = StructuredLogger(__name__)
        logger.info("User logged in", user_id="123", ip="1.2.3.4")
    """
    
    def __init__(self, name: str):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)
        self.default_fields: Dict[str, Any] = {}
    
    def _log(self, log_level: int, message: str, **fields):
        """Internal logging method with JSON formatting"""
        data = {
            **self.default_fields,
            **fields,
            "message": message,
            "timestamp": time.time()
        }
        self.logger.log(log_level, json.dumps(data))
    
    def info(self, message: str, **fields):
        """Log at INFO level with structured fields"""
        self._log(logging.INFO, message, level="info", **fields)
    
    def warning(self, message: str, **fields):
        """Log at WARNING level with structured fields"""
        self._log(logging.WARNING, message, level="warning", **fields)
    
    def error(self, message: str, **fields):
        """Log at ERROR level with structured fields"""
        self._log(logging.ERROR, message, level="error", **fields)
    
    def debug(self, message: str, **fields):
        """Log at DEBUG level with structured fields"""
        self._log(logging.DEBUG, message, level="debug", **fields)
    
    def with_fields(self, **fields) -> "StructuredLogger":
        """
        Create a child logger with additional default fields.
        
        Useful for adding context that applies to multiple log entries:
        
            request_logger = logger.with_fields(request_id="abc123")
            request_logger.info("Processing started")
            request_logger.info("Processing completed")
        
        Returns:
            New StructuredLogger with inherited fields
        """
        child = StructuredLogger(self.logger.name)
        child.default_fields = {**self.default_fields, **fields}
        return child


@contextmanager
def log_duration(operation: str, logger: Optional[StructuredLogger] = None, **extra_fields):
    """
    Context manager to log operation duration.
    
    Usage:
        with log_duration("database_query", table="users"):
            result = await db.execute(query)
    
    Args:
        operation: Name of the operation being timed
        logger: Optional StructuredLogger (creates one if not provided)
        **extra_fields: Additional fields to include in the log
    """
    if logger is None:
        logger = get_logger("timing")
    
    start = time.perf_counter()
    error_occurred = None
    try:
        yield
    except Exception as e:
        error_occurred = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        if error_occurred:
            logger.error(
                f"{operation} failed",
                operation=operation,
                duration_ms=round(duration_ms, 2),
                error=error_occurred,
                **extra_fields
            )
        else:
            logger.info(
                f"{operation} completed",
                operation=operation,
                duration_ms=round(duration_ms, 2),
                **extra_fields
            )


def log_llm_request(
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    user_id: str,
    chat_id: str,
    filters_applied: Optional[list] = None,
    tool_calls: Optional[int] = None,
    error: Optional[str] = None,
    streaming: bool = False,
    logger: Optional[StructuredLogger] = None
):
    """
    Log LLM request with all relevant metrics.
    
    Provides standardized logging for LLM interactions including
    token counts, latency, and filter information.
    
    Args:
        model: Model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        duration_ms: Request duration in milliseconds
        user_id: User ID making the request
        chat_id: Chat ID for the conversation
        filters_applied: List of filter names that were applied
        tool_calls: Number of tool calls made
        error: Error message if request failed
        streaming: Whether this was a streaming request
        logger: Optional StructuredLogger (creates one if not provided)
    """
    if logger is None:
        logger = get_logger("llm_service")
    
    tokens_per_second = round(output_tokens / (duration_ms / 1000), 2) if duration_ms > 0 else 0
    
    log_data = {
        "event": "llm_request",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "duration_ms": round(duration_ms, 2),
        "tokens_per_second": tokens_per_second,
        "user_id": user_id,
        "chat_id": chat_id,
        "filters_applied": filters_applied or [],
        "tool_calls": tool_calls or 0,
        "streaming": streaming,
    }
    
    if error:
        log_data["error"] = error
        logger.error("LLM request failed", **log_data)
    else:
        logger.info("LLM request completed", **log_data)


def log_websocket_event(
    event_type: str,
    user_id: str,
    chat_id: Optional[str] = None,
    logger: Optional[StructuredLogger] = None,
    **extra_fields
):
    """
    Log WebSocket events with consistent formatting.
    
    Args:
        event_type: Type of WebSocket event
        user_id: User ID
        chat_id: Chat ID (if applicable)
        logger: Optional StructuredLogger (creates one if not provided)
        **extra_fields: Additional event-specific fields
    """
    if logger is None:
        logger = get_logger("websocket")
    
    logger.info(
        f"WebSocket event: {event_type}",
        event="websocket",
        event_type=event_type,
        user_id=user_id,
        chat_id=chat_id,
        **extra_fields
    )


def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    success: bool = True,
    logger: Optional[StructuredLogger] = None,
    **extra_fields
):
    """
    Log security-related events (auth, rate limiting, etc.).
    
    Args:
        event_type: Type of security event
        user_id: User ID (if known)
        ip_address: Client IP address
        success: Whether the event was successful
        logger: Optional StructuredLogger (creates one if not provided)
        **extra_fields: Additional event-specific fields
    """
    if logger is None:
        logger = get_logger("security")
    
    log_level = logging.INFO if success else logging.WARNING
    logger._log(
        log_level,
        f"Security event: {event_type}",
        event="security",
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        success=success,
        **extra_fields
    )


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    This is the primary way to get a logger for a module:
    
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened", user_id="123")
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
