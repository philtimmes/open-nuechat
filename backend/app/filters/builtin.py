"""
Built-in Filters

A collection of ready-to-use filters for common use cases.
All filters use priority-based execution (HIGHEST runs first, LEAST runs last).
"""

import re
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Pattern
from collections import defaultdict
import asyncio

from .base import (
    OverrideToLLM,
    OverrideFromLLM,
    FilterContext,
    FilterResult,
    Priority,
    StreamChunk,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ToLLM FILTERS (User -> LLM)
# =============================================================================

class RateLimitToLLM(OverrideToLLM):
    """
    Rate limits messages per user.
    Uses a sliding window algorithm.
    
    Priority: HIGHEST (runs first to reject early)
    """
    
    def __init__(self, priority: Priority = Priority.HIGHEST):
        super().__init__(
            name="rate_limit",
            priority=priority,
        )
        self._requests: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        window_seconds = self.get_config("window_seconds", 60)
        max_requests = self.get_config("max_requests", 20)
        wait_time = 0
        
        async with self._lock:
            now = datetime.now(timezone.utc)
            user_requests = self._requests[context.user_id]
            
            # Remove old requests outside the window
            cutoff = now.timestamp() - window_seconds
            user_requests = [r for r in user_requests if r.timestamp() > cutoff]
            self._requests[context.user_id] = user_requests
            
            if len(user_requests) >= max_requests:
                # Calculate wait time until oldest request expires
                oldest = min(r.timestamp() for r in user_requests)
                wait_time = (oldest + window_seconds) - now.timestamp() + 0.1
        
        # Wait outside the lock if needed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Record this request
        async with self._lock:
            now = datetime.now(timezone.utc)
            user_requests = self._requests[context.user_id]
            cutoff = now.timestamp() - window_seconds
            user_requests = [r for r in user_requests if r.timestamp() > cutoff]
            user_requests.append(now)
            self._requests[context.user_id] = user_requests
        
        return FilterResult.passthrough(content)


class PromptInjectionToLLM(OverrideToLLM):
    """
    Detects and prevents prompt injection attempts.
    
    Priority: HIGHEST (security critical)
    
    Note: Skipped for user-uploaded file content (is_file_content=True in context)
    since code files legitimately contain patterns like "system:", "user:", etc.
    """
    
    INJECTION_PATTERNS = [
        r'(?:system|assistant|human|user)\s*:',
        r'\[(?:INST|SYS|SYSTEM)\]',
        r'<\|(?:im_start|im_end|system|user|assistant)\|>',
        r'###\s*(?:instruction|system|human|assistant)',
        r'(?:you\s+are|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?:a\s+)?(?:different|new|evil|unrestricted)',
        r'(?:ignore|forget|discard|override)\s+(?:your|all|any|previous)\s+(?:instructions?|rules?|guidelines?)',
        r'<\/?(?:system|prompt|instructions?)>',
    ]
    
    def __init__(self, priority: Priority = Priority.HIGHEST):
        super().__init__(
            name="prompt_injection",
            priority=priority,
        )
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        # Skip if safety filters are disabled by admin
        if context.metadata.get("safety_filters_disabled", False):
            return FilterResult.passthrough(content)
        
        # Skip injection detection for user-uploaded file content
        # Files legitimately contain patterns like "system:", "user:", "#include <s>" etc.
        if context.metadata.get("is_file_content", False):
            logger.debug(f"Skipping prompt injection check for file content (user {context.user_id})")
            return FilterResult.passthrough(content)
        
        severity_score = 0
        detected_patterns = []
        
        for pattern in self._patterns:
            matches = pattern.findall(content)
            if matches:
                severity_score += len(matches)
                detected_patterns.append(pattern.pattern[:50])
        
        threshold = self.get_config("threshold", 2)
        
        if severity_score >= threshold:
            logger.warning(
                f"Prompt injection detected for user {context.user_id}: "
                f"score={severity_score}, patterns={detected_patterns}"
            )
            
            if self.get_config("block_on_detection", True):
                return FilterResult.block("Potential prompt injection detected")
            else:
                return FilterResult(
                    content=content,
                    modified=False,
                    metadata={"injection_score": severity_score}
                )
        
        return FilterResult.passthrough(content)


class ContentModerationToLLM(OverrideToLLM):
    """
    Filters inappropriate or harmful content from user messages.
    
    Priority: HIGH (after rate limit and injection detection)
    
    Note: Skipped for user-uploaded file content (is_file_content=True in context)
    since code files may legitimately discuss security concepts.
    """
    
    DEFAULT_BLOCKED_PATTERNS = [
        r'\b(hack|exploit|attack)\s+(the\s+)?(system|server|database)\b',
    ]
    
    def __init__(self, priority: Priority = Priority.HIGH):
        super().__init__(
            name="content_moderation",
            priority=priority,
        )
        self._blocked_words: Set[str] = set()
        self._blocked_patterns: List[Pattern] = []
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        patterns = self.get_config("blocked_patterns", self.DEFAULT_BLOCKED_PATTERNS)
        self._blocked_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        words = self.get_config("blocked_words", [])
        self._blocked_words = set(w.lower() for w in words)
    
    def configure(self, **kwargs) -> "ContentModerationToLLM":
        super().configure(**kwargs)
        self._compile_patterns()
        return self
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        # Skip if safety filters are disabled by admin
        if context.metadata.get("safety_filters_disabled", False):
            return FilterResult.passthrough(content)
        
        # Skip content moderation for user-uploaded file content
        if context.metadata.get("is_file_content", False):
            logger.debug(f"Skipping content moderation for file content (user {context.user_id})")
            return FilterResult.passthrough(content)
        
        content_lower = content.lower()
        for word in self._blocked_words:
            if word in content_lower:
                return FilterResult.block(f"Content contains blocked word")
        
        for pattern in self._blocked_patterns:
            if pattern.search(content):
                return FilterResult.block(f"Content matches blocked pattern")
        
        return FilterResult.passthrough(content)


class InputSanitizerToLLM(OverrideToLLM):
    """
    Sanitizes and normalizes user input.
    
    Priority: MEDIUM (after security checks)
    """
    
    def __init__(self, priority: Priority = Priority.MEDIUM):
        super().__init__(
            name="input_sanitizer",
            priority=priority,
        )
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        original = content
        
        # Remove control characters (except newlines and tabs)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
        
        # Normalize whitespace
        if self.get_config("normalize_whitespace", True):
            sanitized = re.sub(r'[^\S\n]+', ' ', sanitized)
            sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        # Trim
        if self.get_config("trim", True):
            sanitized = sanitized.strip()
        
        # Length limit (0 = no limit)
        max_length = self.get_config("max_length", 0)
        if max_length > 0 and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"Input truncated from {len(original)} to {max_length} chars")
        
        if sanitized != original:
            return FilterResult.modify(sanitized, sanitized=True)
        
        return FilterResult.passthrough(content)


class PIIRedactionToLLM(OverrideToLLM):
    """
    Redacts PII from user input before sending to LLM.
    
    Priority: MEDIUM
    """
    
    PII_PATTERNS = {
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
        "phone_us": (r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', "[PHONE]"),
        "ssn": (r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b', "[SSN]"),
        "credit_card": (r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b', "[CREDIT_CARD]"),
        "ip_address": (r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "[IP_ADDRESS]"),
    }
    
    def __init__(self, priority: Priority = Priority.MEDIUM):
        super().__init__(
            name="pii_redaction_input",
            priority=priority,
        )
        self._compiled_patterns: Dict[str, tuple] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        enabled = self.get_config("enabled_types", list(self.PII_PATTERNS.keys()))
        self._compiled_patterns = {
            name: (re.compile(pattern), replacement)
            for name, (pattern, replacement) in self.PII_PATTERNS.items()
            if name in enabled
        }
    
    def configure(self, **kwargs) -> "PIIRedactionToLLM":
        super().configure(**kwargs)
        self._compile_patterns()
        return self
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        redacted = content
        redaction_count = 0
        
        for pii_type, (pattern, replacement) in self._compiled_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                redaction_count += len(matches)
                redacted = pattern.sub(replacement, redacted)
        
        if redaction_count > 0:
            logger.info(f"Redacted {redaction_count} PII instances from input")
            return FilterResult.modify(redacted, pii_redacted=True, count=redaction_count)
        
        return FilterResult.passthrough(content)


class ContextEnhancerToLLM(OverrideToLLM):
    """
    Enhances user messages with additional context.
    
    Priority: LOW (runs near the end)
    """
    
    def __init__(self, priority: Priority = Priority.LOW):
        super().__init__(
            name="context_enhancer",
            priority=priority,
        )
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        enhancements = []
        
        if self.get_config("add_timestamp", False):
            enhancements.append(f"[Current time: {context.timestamp.isoformat()}]")
        
        user_context = context.metadata.get("user_context")
        if user_context and self.get_config("add_user_context", True):
            enhancements.append(f"[User context: {user_context}]")
        
        prefix = self.get_config("prefix", "")
        if prefix:
            enhancements.append(prefix)
        
        if enhancements:
            enhanced = "\n".join(enhancements) + "\n\n" + content
            return FilterResult.modify(enhanced, context_added=True)
        
        return FilterResult.passthrough(content)


# =============================================================================
# FromLLM FILTERS (LLM -> User)
# =============================================================================

class PIIRedactionFromLLM(OverrideFromLLM):
    """
    Redacts PII from LLM responses before sending to user.
    
    Priority: HIGHEST (security first)
    """
    
    PII_PATTERNS = {
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
        "phone_us": (r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', "[PHONE]"),
        "ssn": (r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b', "[SSN]"),
        "credit_card": (r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b', "[CREDIT_CARD]"),
    }
    
    def __init__(self, priority: Priority = Priority.HIGHEST):
        super().__init__(
            name="pii_redaction_output",
            priority=priority,
        )
        self._compiled_patterns: Dict[str, tuple] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        enabled = self.get_config("enabled_types", list(self.PII_PATTERNS.keys()))
        self._compiled_patterns = {
            name: (re.compile(pattern), replacement)
            for name, (pattern, replacement) in self.PII_PATTERNS.items()
            if name in enabled
        }
    
    def configure(self, **kwargs) -> "PIIRedactionFromLLM":
        super().configure(**kwargs)
        self._compile_patterns()
        return self
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        redacted = content
        redaction_count = 0
        
        for pii_type, (pattern, replacement) in self._compiled_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                redaction_count += len(matches)
                redacted = pattern.sub(replacement, redacted)
        
        if redaction_count > 0:
            return FilterResult.modify(redacted, pii_redacted=True, count=redaction_count)
        
        return FilterResult.passthrough(content)
    
    async def process_stream_chunk(
        self, chunk: StreamChunk, context: FilterContext
    ) -> StreamChunk:
        if chunk.chunk_type != "text":
            return chunk
        
        redacted = chunk.content
        for pii_type, (pattern, replacement) in self._compiled_patterns.items():
            redacted = pattern.sub(replacement, redacted)
        
        if redacted != chunk.content:
            return StreamChunk(
                content=redacted,
                chunk_type=chunk.chunk_type,
                is_final=chunk.is_final,
                metadata={**chunk.metadata, "pii_redacted": True},
            )
        
        return chunk


class SensitiveTopicFromLLM(OverrideFromLLM):
    """
    Adds warnings for responses about sensitive topics.
    
    Priority: MEDIUM
    """
    
    TOPIC_PATTERNS = {
        "medical": {
            "patterns": [r'\b(?:diagnos|symptom|treatment|medicat|prescription|dosage)\b'],
            "disclaimer": "⚠️ This is for informational purposes only and not medical advice.",
        },
        "legal": {
            "patterns": [r'\b(?:legal|lawsuit|attorney|lawyer|court|liability)\b'],
            "disclaimer": "⚠️ This is general information, not legal advice.",
        },
        "financial": {
            "patterns": [r'\b(?:invest|stock|crypto|trading|financial advice|portfolio)\b'],
            "disclaimer": "⚠️ This is not financial advice.",
        },
    }
    
    def __init__(self, priority: Priority = Priority.MEDIUM):
        super().__init__(
            name="sensitive_topic",
            priority=priority,
        )
        self._compiled: Dict[str, List[Pattern]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        enabled = self.get_config("enabled_topics", list(self.TOPIC_PATTERNS.keys()))
        self._compiled = {
            topic: [re.compile(p, re.IGNORECASE) for p in config["patterns"]]
            for topic, config in self.TOPIC_PATTERNS.items()
            if topic in enabled
        }
    
    def configure(self, **kwargs) -> "SensitiveTopicFromLLM":
        super().configure(**kwargs)
        self._compile_patterns()
        return self
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        detected_topics = []
        
        for topic, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(content):
                    detected_topics.append(topic)
                    break
        
        if detected_topics:
            disclaimers = [
                self.TOPIC_PATTERNS[topic]["disclaimer"]
                for topic in detected_topics
            ]
            modified = content + "\n\n" + "\n".join(disclaimers)
            return FilterResult.modify(modified, sensitive_topics=detected_topics)
        
        return FilterResult.passthrough(content)


class ResponseFormatterFromLLM(OverrideFromLLM):
    """
    Formats and cleans up LLM responses.
    
    Priority: LOW (formatting after content processing)
    """
    
    def __init__(self, priority: Priority = Priority.LOW):
        super().__init__(
            name="response_formatter",
            priority=priority,
        )
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        formatted = content
        
        if self.get_config("normalize_code_blocks", True):
            formatted = re.sub(r'```\n', '```text\n', formatted)
        
        if self.get_config("clean_newlines", True):
            formatted = re.sub(r'\n{4,}', '\n\n\n', formatted)
        
        suffix = self.get_config("suffix", "")
        if suffix:
            formatted = formatted + "\n\n" + suffix
        
        if formatted != content:
            return FilterResult.modify(formatted, formatted=True)
        
        return FilterResult.passthrough(content)


class TokenCounterFromLLM(OverrideFromLLM):
    """
    Counts tokens in responses for tracking.
    
    Priority: LEAST (runs last, doesn't modify content)
    """
    
    def __init__(self, priority: Priority = Priority.LEAST):
        super().__init__(
            name="token_counter",
            priority=priority,
        )
        self._encoder = None
    
    def _get_encoder(self):
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass
        return self._encoder
    
    def _count_tokens(self, text: str) -> int:
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text) // 4  # Approximate
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        token_count = self._count_tokens(content)
        context.metadata["output_tokens"] = token_count
        
        max_tokens = self.get_config("max_output_tokens")
        if max_tokens and token_count > max_tokens:
            encoder = self._get_encoder()
            if encoder:
                tokens = encoder.encode(content)[:max_tokens]
                truncated = encoder.decode(tokens)
            else:
                truncated = content[:max_tokens * 4]
            
            truncated += "\n\n[Response truncated]"
            return FilterResult.modify(truncated, truncated=True, token_count=token_count)
        
        return FilterResult(
            content=content,
            modified=False,
            metadata={"token_count": token_count},
        )
    
    async def process_stream_chunk(
        self, chunk: StreamChunk, context: FilterContext
    ) -> StreamChunk:
        if chunk.chunk_type == "text":
            chunk_tokens = self._count_tokens(chunk.content)
            current = context.metadata.get("streaming_tokens", 0)
            context.metadata["streaming_tokens"] = current + chunk_tokens
        return chunk


class AuditLogFromLLM(OverrideFromLLM):
    """
    Logs all LLM responses for audit purposes.
    
    Priority: LEAST (runs last)
    """
    
    def __init__(self, priority: Priority = Priority.LEAST):
        super().__init__(
            name="audit_log_output",
            priority=priority,
        )
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        log_entry = {
            "timestamp": context.timestamp.isoformat(),
            "user_id": context.user_id,
            "chat_id": context.chat_id,
            "direction": "from_llm",
            "content_length": len(content),
        }
        
        if not self.get_config("hash_content", False):
            max_len = self.get_config("truncate_content", 200)
            log_entry["content_preview"] = content[:max_len]
        
        log_level = self.get_config("log_level", "debug")
        getattr(logger, log_level, logger.debug)(f"Audit: {json.dumps(log_entry)}")
        
        return FilterResult.passthrough(content)


# =============================================================================
# STREAMING FILTERS
# =============================================================================

class StreamingWordFilterFromLLM(OverrideFromLLM):
    """
    Filters specific words from streaming output in real-time.
    
    Priority: HIGH (filter early in stream)
    """
    
    def __init__(self, priority: Priority = Priority.HIGH):
        super().__init__(
            name="streaming_word_filter",
            priority=priority,
        )
        self._partial_buffers: Dict[str, str] = {}
    
    async def on_stream_start(self, context: FilterContext) -> None:
        key = f"{context.chat_id}:{context.message_id}"
        self._partial_buffers[key] = ""
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        filtered_words = self.get_config("filtered_words", [])
        if not filtered_words:
            return FilterResult.passthrough(content)
        
        replacement = self.get_config("replacement", "[FILTERED]")
        result = content
        
        for word in filtered_words:
            result = re.sub(
                rf'\b{re.escape(word)}\b',
                replacement,
                result,
                flags=re.IGNORECASE
            )
        
        if result != content:
            return FilterResult.modify(result, words_filtered=True)
        return FilterResult.passthrough(content)
    
    async def process_stream_chunk(
        self, chunk: StreamChunk, context: FilterContext
    ) -> StreamChunk:
        if chunk.chunk_type != "text":
            return chunk
        
        filtered_words = self.get_config("filtered_words", [])
        if not filtered_words:
            return chunk
        
        replacement = self.get_config("replacement", "[FILTERED]")
        key = f"{context.chat_id}:{context.message_id}"
        
        text = self._partial_buffers.get(key, "") + chunk.content
        
        for word in filtered_words:
            text = re.sub(
                rf'\b{re.escape(word)}\b',
                replacement,
                text,
                flags=re.IGNORECASE
            )
        
        max_word_len = max(len(w) for w in filtered_words) if filtered_words else 0
        if len(text) > max_word_len and not chunk.is_final:
            output = text[:-max_word_len]
            self._partial_buffers[key] = text[-max_word_len:]
        else:
            output = text
            self._partial_buffers[key] = ""
        
        return StreamChunk(
            content=output,
            chunk_type=chunk.chunk_type,
            is_final=chunk.is_final,
            metadata=chunk.metadata,
        )
    
    async def on_stream_end(self, context: FilterContext) -> Optional[StreamChunk]:
        key = f"{context.chat_id}:{context.message_id}"
        remaining = self._partial_buffers.pop(key, "")
        
        if remaining:
            return StreamChunk(content=remaining, chunk_type="text", is_final=True)
        return None
