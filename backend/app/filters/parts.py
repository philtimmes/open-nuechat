"""
Built-in Chain Parts

Add these to your chain in execution order.

Example:
    from app.filters.chain import get_default_chain
    from app.filters.parts import SanitizePart, ToolLogPart
    
    chain = get_default_chain()
    chain.add(SanitizePart())
    chain.add(ToolLogPart())
"""

import re
import json
import logging
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .chain import (
    ChainContext,
    ChainResult,
    ToLLMPart,
    FromLLMPart,
    ToToolPart,
    FromToolPart,
    LogicPart,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TO_LLM PARTS
# =============================================================================

class RateLimitPart(ToLLMPart):
    """Rate limits requests per user. Pauses instead of failing."""
    
    def __init__(self, max_requests: int = 20, window_seconds: int = 60, enabled: bool = True):
        super().__init__("rate_limit", enabled=enabled)
        self.configure(max_requests=max_requests, window_seconds=window_seconds)
        self._requests: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        max_req = self.get_config("max_requests", 20)
        window = self.get_config("window_seconds", 60)
        wait_time = 0
        
        async with self._lock:
            now = datetime.now(timezone.utc)
            reqs = self._requests[ctx.user_id]
            cutoff = now.timestamp() - window
            reqs = [r for r in reqs if r.timestamp() > cutoff]
            self._requests[ctx.user_id] = reqs
            
            if len(reqs) >= max_req:
                # Calculate wait time until oldest request expires
                oldest = min(r.timestamp() for r in reqs)
                wait_time = (oldest + window) - now.timestamp() + 0.1  # +0.1s buffer
        
        # Wait outside the lock if needed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Record this request
        async with self._lock:
            now = datetime.now(timezone.utc)
            reqs = self._requests[ctx.user_id]
            cutoff = now.timestamp() - window
            reqs = [r for r in reqs if r.timestamp() > cutoff]
            reqs.append(now)
            self._requests[ctx.user_id] = reqs
        
        return ChainResult.ok(content)


class SanitizePart(ToLLMPart):
    """Sanitizes user input."""
    
    def __init__(self, max_length: int = 100000, enabled: bool = True):
        super().__init__("sanitize", enabled=enabled)
        self.configure(max_length=max_length)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        # Remove control chars
        s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
        # Normalize whitespace
        s = re.sub(r'[^\S\n]+', ' ', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        s = s.strip()
        
        # Truncate
        max_len = self.get_config("max_length", 100000)
        if len(s) > max_len:
            s = s[:max_len]
        
        if s != content:
            return ChainResult.modify(s)
        return ChainResult.ok(content)


class PromptGuardPart(ToLLMPart):
    """Detects prompt injection attempts."""
    
    PATTERNS = [
        r'(?:system|assistant|human|user)\s*:',
        r'\[(?:INST|SYS|SYSTEM)\]',
        r'<\|(?:im_start|im_end|system|user|assistant)\|>',
        r'(?:ignore|forget|discard|override)\s+(?:your|all|previous)\s+(?:instructions?|rules?)',
    ]
    
    def __init__(self, threshold: int = 2, block: bool = True, enabled: bool = True):
        super().__init__("prompt_guard", enabled=enabled)
        self.configure(threshold=threshold, block=block)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        score = sum(len(p.findall(content)) for p in self._compiled)
        
        if score >= self.get_config("threshold", 2):
            logger.warning(f"Injection detected: user={ctx.user_id}, score={score}")
            if self.get_config("block", True):
                return ChainResult.block("Prompt injection detected")
        
        return ChainResult.ok(content)


class PIIRedactPart(ToLLMPart):
    """Redacts PII from input."""
    
    PATTERNS = {
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
        "phone": (r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', "[PHONE]"),
        "ssn": (r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b', "[SSN]"),
        "card": (r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b', "[CARD]"),
    }
    
    def __init__(self, enabled: bool = True):
        super().__init__("pii_redact", enabled=enabled)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        result = content
        for pattern, replacement in self.PATTERNS.values():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        if result != content:
            return ChainResult.modify(result)
        return ChainResult.ok(content)


# =============================================================================
# FROM_LLM PARTS
# =============================================================================

class OutputFormatterPart(FromLLMPart):
    """Formats LLM output."""
    
    def __init__(self, suffix: str = "", enabled: bool = True):
        super().__init__("output_formatter", enabled=enabled)
        self.configure(suffix=suffix)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        result = re.sub(r'\n{4,}', '\n\n\n', content)
        
        suffix = self.get_config("suffix", "")
        if suffix:
            result = result + "\n\n" + suffix
        
        if result != content:
            return ChainResult.modify(result)
        return ChainResult.ok(content)


class WordFilterPart(FromLLMPart):
    """Filters specific words from output."""
    
    def __init__(self, words: List[str] = None, replacement: str = "[FILTERED]", enabled: bool = True):
        super().__init__("word_filter", enabled=enabled)
        self.configure(words=words or [], replacement=replacement)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        words = self.get_config("words", [])
        if not words:
            return ChainResult.ok(content)
        
        replacement = self.get_config("replacement", "[FILTERED]")
        result = content
        
        for word in words:
            result = re.sub(rf'\b{re.escape(word)}\b', replacement, result, flags=re.IGNORECASE)
        
        if result != content:
            return ChainResult.modify(result)
        return ChainResult.ok(content)


class AuditLogPart(FromLLMPart):
    """Logs LLM responses."""
    
    def __init__(self, log_level: str = "debug", enabled: bool = True):
        super().__init__("audit_log", enabled=enabled)
        self.configure(log_level=log_level)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": ctx.user_id,
            "chat": ctx.chat_id,
            "len": len(str(content)),
        }
        level = self.get_config("log_level", "debug")
        getattr(logger, level, logger.debug)(f"Audit: {json.dumps(entry)}")
        return ChainResult.ok(content)


# =============================================================================
# TOOL PARTS
# =============================================================================

class ToolLogPart(ToToolPart):
    """Logs tool calls."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("tool_log", enabled=enabled)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        logger.debug(f"Tool: {ctx.tool_name}, params: {json.dumps(content)[:200]}")
        return ChainResult.ok(content)


class ToolResultLogPart(FromToolPart):
    """Logs tool results."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("tool_result_log", enabled=enabled)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        logger.debug(f"Tool result ({ctx.tool_name}): {str(content)[:200]}")
        return ChainResult.ok(content)


class ToolSanitizePart(ToToolPart):
    """Sanitizes tool parameters."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("tool_sanitize", enabled=enabled)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, dict):
            return ChainResult.ok(content)
        
        result = content.copy()
        modified = False
        
        for key, val in result.items():
            if isinstance(val, str):
                clean = val.replace('\x00', '')
                if 'path' in key.lower() or 'file' in key.lower():
                    clean = clean.replace('..', '')
                if clean != val:
                    result[key] = clean
                    modified = True
        
        if modified:
            return ChainResult.modify(result)
        return ChainResult.ok(content)


class ToolResultTruncatePart(FromToolPart):
    """Truncates large tool results."""
    
    def __init__(self, max_length: int = 50000, enabled: bool = True):
        super().__init__("tool_truncate", enabled=enabled)
        self.configure(max_length=max_length)
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        max_len = self.get_config("max_length", 50000)
        
        if isinstance(content, str) and len(content) > max_len:
            truncated = content[:max_len] + f"\n[Truncated {len(content)} -> {max_len}]"
            return ChainResult.modify(truncated)
        
        return ChainResult.ok(content)


# =============================================================================
# LOGIC PARTS
# =============================================================================

class ConditionalSkipPart(LogicPart):
    """Conditionally skips LLM call."""
    
    def __init__(self, patterns: List[str] = None, response: str = "", enabled: bool = True):
        super().__init__("conditional_skip", enabled=enabled)
        self.configure(patterns=patterns or [], response=response)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in (patterns or [])]
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if not isinstance(content, str):
            return ChainResult.ok(content)
        
        for pattern in self._compiled:
            if pattern.search(content):
                ctx.skip_llm = True
                return ChainResult.modify(self.get_config("response", ""))
        
        return ChainResult.ok(content)
