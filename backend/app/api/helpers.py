"""
Shared route helpers for common patterns

This module provides reusable utilities for FastAPI routes including:
- Standardized exception types
- Resource ownership verification
- Token limit checking
"""
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import TypeVar, Type, Optional

T = TypeVar('T')


class ResourceNotFoundError(HTTPException):
    """Resource not found with standardized message"""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} '{resource_id}' not found"
        )


class PermissionDeniedError(HTTPException):
    """User lacks permission for this resource"""
    def __init__(self, action: str = "access"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission to {action} this resource"
        )


class TokenLimitExceededError(HTTPException):
    """User has exceeded token limit"""
    def __init__(self, limit: int, used: int):
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Token limit exceeded. Used: {used:,}, Limit: {limit:,}"
        )


class RateLimitExceededError(HTTPException):
    """Rate limit exceeded"""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)}
        )


async def get_owned_resource(
    db: AsyncSession,
    model_class: Type[T],
    resource_id: str,
    user,  # User model
    resource_name: str = "Resource",
    owner_field: str = "owner_id"
) -> T:
    """
    Fetch a resource and verify ownership in one call.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        resource_id: ID of the resource to fetch
        user: Current user object
        resource_name: Human-readable name for error messages
        owner_field: Name of the owner ID field on the model
    
    Returns:
        The fetched resource
    
    Raises:
        ResourceNotFoundError: If resource doesn't exist
        PermissionDeniedError: If user doesn't own the resource
    """
    result = await db.execute(
        select(model_class).where(model_class.id == resource_id)
    )
    resource = result.scalar_one_or_none()
    
    if not resource:
        raise ResourceNotFoundError(resource_name, resource_id)
    
    if getattr(resource, owner_field) != user.id:
        raise PermissionDeniedError()
    
    return resource


async def get_resource_or_404(
    db: AsyncSession,
    model_class: Type[T],
    resource_id: str,
    resource_name: str = "Resource"
) -> T:
    """
    Fetch a resource or raise 404.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        resource_id: ID of the resource to fetch
        resource_name: Human-readable name for error messages
    
    Returns:
        The fetched resource
    
    Raises:
        ResourceNotFoundError: If resource doesn't exist
    """
    result = await db.execute(
        select(model_class).where(model_class.id == resource_id)
    )
    resource = result.scalar_one_or_none()
    
    if not resource:
        raise ResourceNotFoundError(resource_name, resource_id)
    
    return resource


def check_token_limit(user, required_tokens: int = 0) -> None:
    """
    Check if user has remaining tokens.
    
    Args:
        user: User model instance
        required_tokens: Minimum tokens required (optional)
    
    Raises:
        TokenLimitExceededError: If user has exceeded their limit
    """
    from app.core.config import settings
    
    if settings.FREEFORALL:
        return
    
    if user.tokens_used_this_month >= user.tokens_limit:
        raise TokenLimitExceededError(user.tokens_limit, user.tokens_used_this_month)


def verify_admin(user) -> None:
    """
    Verify user has admin privileges.
    
    Args:
        user: User model instance
    
    Raises:
        PermissionDeniedError: If user is not an admin
    """
    if not user.is_admin:
        raise PermissionDeniedError("perform admin actions")
