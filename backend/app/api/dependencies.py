"""
FastAPI dependencies for authentication and authorization
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.auth import AuthService
from app.models.models import User


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token"""
    
    token = credentials.credentials
    payload = AuthService.decode_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await AuthService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# Alias for backwards compatibility
get_current_user_optional = get_optional_user


def require_tier(minimum_tier: str):
    """Dependency to require a minimum subscription tier"""
    
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
    
    async def check_tier(user: User = Depends(get_current_user)):
        user_level = tier_levels.get(user.tier.value, 0)
        required_level = tier_levels.get(minimum_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {minimum_tier} tier or higher",
            )
        
        return user
    
    return check_tier


async def get_admin_user(
    user: User = Depends(get_current_user),
) -> User:
    """
    Require admin privileges.
    
    Admin users are determined by:
    1. Enterprise tier users
    2. Users with is_admin flag set (if implemented)
    3. Specific admin email addresses (configurable)
    """
    from app.core.config import settings
    
    # Check if user is enterprise tier
    if hasattr(user, 'tier') and user.tier and user.tier.value == "enterprise":
        return user
    
    # Check admin email list (can be configured via environment)
    admin_emails = getattr(settings, 'admin_emails', [])
    if user.email in admin_emails:
        return user
    
    # Check is_admin flag if it exists
    if hasattr(user, 'is_admin') and user.is_admin:
        return user
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin privileges required for this operation",
    )
