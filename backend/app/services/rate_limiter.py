"""
Application-level rate limiting (no external dependencies)

This module provides a token bucket rate limiter that:
- Works in-memory (no Redis required)
- Supports configurable rates and burst capacity
- Automatically cleans up stale entries
- Is thread-safe for async use

For distributed deployments, consider using Redis-based rate limiting.
"""
import asyncio
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        self.message = message
        super().__init__(message)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit"""
    requests: int  # Number of requests allowed per window
    window_seconds: int  # Time window in seconds
    burst: Optional[int] = None  # Allow burst up to this amount (defaults to requests)
    
    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests


@dataclass
class RateLimitState:
    """State for a single rate limit bucket"""
    tokens: float
    last_update: float


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    
    Allows smooth rate limiting with configurable burst capacity.
    Tokens are added at a constant rate and consumed by requests.
    """
    
    def __init__(self):
        self._buckets: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    async def check(
        self,
        key: str,
        config: RateLimitConfig,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed and consume tokens if so.
        
        Args:
            key: Unique identifier for the rate limit bucket
            config: Rate limit configuration
            cost: Number of tokens to consume (default 1)
        
        Returns:
            Tuple of (allowed, headers) where:
            - allowed: True if request should proceed
            - headers: Dict of rate limit headers to include in response
        """
        async with self._lock:
            now = time.time()
            
            # Cleanup old buckets periodically
            if now - self._last_cleanup > self._cleanup_interval:
                await self._cleanup(now)
            
            # Get or create bucket
            if key not in self._buckets:
                max_tokens = config.burst or config.requests
                self._buckets[key] = RateLimitState(
                    tokens=max_tokens,
                    last_update=now
                )
            
            bucket = self._buckets[key]
            
            # Refill tokens based on time elapsed
            max_tokens = config.burst or config.requests
            refill_rate = config.requests / config.window_seconds
            elapsed = now - bucket.last_update
            bucket.tokens = min(max_tokens, bucket.tokens + elapsed * refill_rate)
            bucket.last_update = now
            
            # Check if request is allowed
            allowed = bucket.tokens >= cost
            
            if allowed:
                bucket.tokens -= cost
            
            # Calculate headers
            headers = {
                "X-RateLimit-Limit": config.requests,
                "X-RateLimit-Remaining": max(0, int(bucket.tokens)),
                "X-RateLimit-Reset": int(now + config.window_seconds),
            }
            
            if not allowed:
                retry_after = int((cost - bucket.tokens) / refill_rate) + 1
                headers["Retry-After"] = retry_after
            
            return allowed, headers
    
    async def _cleanup(self, now: float):
        """Remove stale buckets that haven't been used recently"""
        stale_threshold = 3600  # 1 hour
        stale_keys = [
            key for key, state in self._buckets.items()
            if now - state.last_update > stale_threshold
        ]
        for key in stale_keys:
            del self._buckets[key]
        self._last_cleanup = now
    
    async def reset(self, key: str):
        """Reset a specific rate limit bucket (for testing)"""
        async with self._lock:
            if key in self._buckets:
                del self._buckets[key]
    
    async def check_rate_limit(
        self,
        action: str,
        identifier: str,
        cost: int = 1
    ) -> None:
        """
        Check rate limit and raise exception if exceeded.
        
        Args:
            action: Action name (must be a key in RATE_LIMITS)
            identifier: User ID or IP address
            cost: Number of tokens to consume
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        config = RATE_LIMITS.get(action)
        if not config:
            return  # No limit configured for this action
        
        key = f"{action}:{identifier}"
        allowed, headers = await self.check(key, config, cost)
        
        if not allowed:
            retry_after = headers.get("Retry-After", 60)
            raise RateLimitExceeded(retry_after, f"Rate limit exceeded for {action}")


# Default rate limit configurations
RATE_LIMITS = {
    # Chat messages: 60 per minute with burst of 10
    "chat_message": RateLimitConfig(requests=60, window_seconds=60, burst=10),
    
    # Image generation: 10 per minute with burst of 3
    "image_generation": RateLimitConfig(requests=10, window_seconds=60, burst=3),
    
    # File uploads: 20 per minute with burst of 5
    "file_upload": RateLimitConfig(requests=20, window_seconds=60, burst=5),
    
    # API key creation: 5 per hour
    "api_key_creation": RateLimitConfig(requests=5, window_seconds=3600),
    
    # Login attempts: 10 per 5 minutes
    "login_attempt": RateLimitConfig(requests=10, window_seconds=300),
    
    # Password reset: 3 per hour
    "password_reset": RateLimitConfig(requests=3, window_seconds=3600),
    
    # Document uploads: 30 per hour
    "document_upload": RateLimitConfig(requests=30, window_seconds=3600),
    
    # Knowledge store creation: 10 per hour
    "knowledge_store_creation": RateLimitConfig(requests=10, window_seconds=3600),
}


# Global rate limiter instance
rate_limiter = TokenBucketRateLimiter()


async def check_rate_limit(
    user_id: str,
    action: str,
    cost: int = 1
) -> Tuple[bool, Dict[str, int]]:
    """
    Check rate limit for a user action.
    
    Args:
        user_id: User's ID
        action: Action name (must be a key in RATE_LIMITS)
        cost: Number of "tokens" to consume
    
    Returns:
        Tuple of (allowed, headers)
    """
    config = RATE_LIMITS.get(action)
    if not config:
        return True, {}
    
    key = f"{action}:{user_id}"
    return await rate_limiter.check(key, config, cost)


async def check_rate_limit_ip(
    ip_address: str,
    action: str,
    cost: int = 1
) -> Tuple[bool, Dict[str, int]]:
    """
    Check rate limit by IP address (for unauthenticated endpoints).
    
    Args:
        ip_address: Client IP address
        action: Action name
        cost: Number of tokens to consume
    
    Returns:
        Tuple of (allowed, headers)
    """
    config = RATE_LIMITS.get(action)
    if not config:
        return True, {}
    
    key = f"{action}:ip:{ip_address}"
    return await rate_limiter.check(key, config, cost)
