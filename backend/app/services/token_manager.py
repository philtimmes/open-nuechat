"""
Token management with blacklisting and rotation

This module provides token lifecycle management including:
- Token blacklisting for logout/revocation
- Refresh token rotation detection
- Automatic cleanup of expired tokens

The blacklist is stored in-memory for simplicity. For production deployments
with multiple instances, consider using Redis instead.
"""
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from collections import OrderedDict

from app.core.config import settings


class TokenBlacklist:
    """
    In-memory token blacklist with automatic cleanup.
    
    Tokens are stored with their expiration time and automatically
    removed after they would have expired anyway.
    
    Thread-safe for use in async context.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the blacklist.
        
        Args:
            max_size: Maximum number of tokens to store. Oldest tokens
                     are evicted when limit is reached.
        """
        self._blacklist: OrderedDict[str, datetime] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def add(self, token_jti: str, expires_at: datetime):
        """
        Add a token to the blacklist.
        
        Args:
            token_jti: The JWT ID (jti claim) of the token
            expires_at: When the token would naturally expire
        """
        async with self._lock:
            # Cleanup old entries first
            await self._cleanup()
            
            # Enforce max size (LRU eviction)
            while len(self._blacklist) >= self._max_size:
                self._blacklist.popitem(last=False)
            
            self._blacklist[token_jti] = expires_at
    
    async def is_blacklisted(self, token_jti: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token_jti: The JWT ID to check
            
        Returns:
            True if the token is blacklisted, False otherwise
        """
        async with self._lock:
            if token_jti not in self._blacklist:
                return False
            
            # Check if it's expired (auto-cleanup)
            if self._blacklist[token_jti] < datetime.utcnow():
                del self._blacklist[token_jti]
                return False
            
            return True
    
    async def _cleanup(self):
        """Remove expired tokens from blacklist"""
        now = datetime.utcnow()
        expired = [jti for jti, exp in self._blacklist.items() if exp < now]
        for jti in expired:
            del self._blacklist[jti]
    
    async def clear(self):
        """Clear all blacklisted tokens (for testing)"""
        async with self._lock:
            self._blacklist.clear()
    
    @property
    def size(self) -> int:
        """Get current number of blacklisted tokens"""
        return len(self._blacklist)


# Global blacklist instance
token_blacklist = TokenBlacklist()


async def blacklist_token(token_jti: str, expires_at: datetime):
    """
    Blacklist a token (call on logout).
    
    Args:
        token_jti: The JWT ID from the token's jti claim
        expires_at: When the token would naturally expire
    """
    await token_blacklist.add(token_jti, expires_at)


async def is_token_blacklisted(token_jti: str) -> bool:
    """
    Check if a token is blacklisted.
    
    Args:
        token_jti: The JWT ID to check
        
    Returns:
        True if blacklisted, False otherwise
    """
    return await token_blacklist.is_blacklisted(token_jti)


def should_rotate_refresh_token(issued_at: datetime) -> bool:
    """
    Check if refresh token should be rotated.
    
    Rotate if token is older than half its lifetime to ensure
    tokens are regularly refreshed.
    
    Args:
        issued_at: When the token was issued (iat claim)
        
    Returns:
        True if the token should be rotated
    """
    token_age = datetime.utcnow() - issued_at
    half_lifetime = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS / 2)
    return token_age > half_lifetime


def get_token_expiry(token_type: str = "access") -> datetime:
    """
    Get the expiration datetime for a new token.
    
    Args:
        token_type: Either "access" or "refresh"
        
    Returns:
        datetime when the token should expire
    """
    now = datetime.utcnow()
    
    if token_type == "refresh":
        return now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    else:
        return now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
