"""
API Key Management Routes

Allows users to create and manage their own API keys for programmatic access.
"""

from typing import List, Optional
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import secrets
import hashlib
import logging

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.models.models import User, APIKey, APIKeyScope

router = APIRouter()
logger = logging.getLogger(__name__)


# === Schemas ===

class APIKeyCreate(BaseModel):
    """Request to create a new API key"""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=lambda: ["completions"])
    rate_limit: int = Field(default=100, ge=1, le=1000)
    allowed_ips: List[str] = Field(default_factory=list)
    allowed_assistants: List[str] = Field(default_factory=list)
    allowed_knowledge_stores: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key info (without the actual key)"""
    id: str
    name: str
    key_prefix: str
    scopes: List[str]
    rate_limit: int
    allowed_ips: List[str]
    is_active: bool
    last_used_at: Optional[datetime]
    usage_count: int
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class APIKeyCreatedResponse(BaseModel):
    """Response when creating a new key (includes the full key once)"""
    id: str
    name: str
    key: str  # Full key - only shown once!
    key_prefix: str
    scopes: List[str]
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class APIKeyUpdate(BaseModel):
    """Update API key settings"""
    name: Optional[str] = None
    scopes: Optional[List[str]] = None
    rate_limit: Optional[int] = Field(default=None, ge=1, le=1000)
    allowed_ips: Optional[List[str]] = None
    allowed_assistants: Optional[List[str]] = None
    allowed_knowledge_stores: Optional[List[str]] = None
    is_active: Optional[bool] = None


# === Helper Functions ===

def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.
    Returns: (full_key, prefix, hash)
    """
    # Generate 32 random bytes = 64 hex characters
    random_part = secrets.token_hex(32)
    full_key = f"nxs_{random_part}"
    prefix = f"nxs_{random_part[:4]}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    
    return full_key, prefix, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key for storage/comparison"""
    return hashlib.sha256(key.encode()).hexdigest()


# === Endpoints ===

# Health check - no auth required
@router.get("/health")
async def api_keys_health():
    """Health check for API keys router"""
    return {"status": "ok", "router": "api_keys"}


@router.post("", response_model=APIKeyCreatedResponse)
async def create_api_key(
    request: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new API key.
    
    **Important**: The full API key is only shown once in this response.
    Store it securely - it cannot be retrieved later.
    
    Available scopes:
    - `chat`: Create messages and stream responses
    - `knowledge`: Access knowledge stores
    - `assistants`: Use custom assistants
    - `billing`: View usage and billing info
    - `full`: All permissions
    """
    logger.info(f"=== create_api_key called for user {current_user.id} ===")
    logger.info(f"Request: name='{request.name}', scopes={request.scopes}")
    
    # Validate scopes
    valid_scopes = {s.value for s in APIKeyScope}
    for scope in request.scopes:
        if scope not in valid_scopes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope: {scope}. Valid scopes: {list(valid_scopes)}"
            )
    
    # Check key limit per user (free: 3, pro: 10, enterprise: unlimited)
    # Admins bypass key limits
    if not current_user.is_admin:
        result = await db.execute(
            select(APIKey).where(APIKey.user_id == current_user.id)
        )
        existing_keys = result.scalars().all()
        
        limits = {"free": 3, "pro": 10, "enterprise": 100}
        # Handle case where tier might be None (for legacy users)
        tier_value = current_user.tier.value if current_user.tier else "free"
        tier_limit = limits.get(tier_value, 3)
        
        if len(existing_keys) >= tier_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"API key limit reached ({tier_limit} keys for {tier_value} tier)"
            )
    
    # Generate key
    full_key, prefix, key_hash = generate_api_key()
    
    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
    
    try:
        # Create key record
        api_key = APIKey(
            user_id=current_user.id,
            name=request.name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=request.scopes,
            rate_limit=request.rate_limit,
            allowed_ips=request.allowed_ips,
            allowed_assistants=request.allowed_assistants,
            allowed_knowledge_stores=request.allowed_knowledge_stores,
            expires_at=expires_at,
        )
        
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)
        
        logger.info(f"Created API key '{api_key.name}' for user {current_user.id}")
        
        return APIKeyCreatedResponse(
            id=api_key.id,
            name=api_key.name,
            key=full_key,  # Only time the full key is returned!
            key_prefix=prefix,
            scopes=api_key.scopes,
            expires_at=api_key.expires_at,
            created_at=api_key.created_at,
        )
    except Exception as e:
        logger.error(f"Failed to create API key for user {current_user.id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the current user"""
    result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == current_user.id)
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    
    return [APIKeyResponse.model_validate(k) for k in keys]


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific API key"""
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return APIKeyResponse.model_validate(api_key)


@router.patch("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    update_data: APIKeyUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an API key's settings"""
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    
    # Validate scopes if provided
    if "scopes" in update_dict:
        valid_scopes = {s.value for s in APIKeyScope}
        for scope in update_dict["scopes"]:
            if scope not in valid_scopes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid scope: {scope}"
                )
    
    for field, value in update_dict.items():
        setattr(api_key, field, value)
    
    await db.commit()
    await db.refresh(api_key)
    
    return APIKeyResponse.model_validate(api_key)


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete (revoke) an API key"""
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    await db.delete(api_key)
    await db.commit()
    
    return {"message": "API key deleted", "id": key_id}


@router.post("/{key_id}/regenerate", response_model=APIKeyCreatedResponse)
async def regenerate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Regenerate an API key.
    
    This creates a new key value while preserving the key's settings.
    The old key will immediately stop working.
    """
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Generate new key
    full_key, prefix, key_hash = generate_api_key()
    
    api_key.key_prefix = prefix
    api_key.key_hash = key_hash
    api_key.usage_count = 0
    api_key.last_used_at = None
    
    await db.commit()
    await db.refresh(api_key)
    
    return APIKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key=full_key,
        key_prefix=prefix,
        scopes=api_key.scopes,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


# === API Key Authentication Dependency ===

async def get_api_key_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = None,  # Query parameter
) -> tuple[User, APIKey]:
    """
    Authenticate a request using an API key.
    
    Accepts the key in:
    1. Authorization header: `Authorization: Bearer nxs_...`
    2. Query parameter: `?api_key=nxs_...`
    
    Returns tuple of (User, APIKey)
    """
    api_key_value = None
    
    # Try Authorization header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer nxs_"):
        api_key_value = auth_header.replace("Bearer ", "")
    
    # Fall back to query parameter
    if not api_key_value:
        # Check query params directly from request
        api_key_param = request.query_params.get("api_key")
        if api_key_param and api_key_param.startswith("nxs_"):
            api_key_value = api_key_param
    
    if not api_key_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format. Use: Authorization: Bearer nxs_... or ?api_key=nxs_..."
        )
    
    key_hash = hash_api_key(api_key_value)
    
    # Find the key
    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    api_key_obj = result.scalar_one_or_none()
    
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check if active
    if not api_key_obj.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is deactivated"
        )
    
    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired"
        )
    
    # Check IP restrictions
    if api_key_obj.allowed_ips:
        client_ip = request.client.host if request.client else None
        if client_ip and client_ip not in api_key_obj.allowed_ips:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address not allowed for this API key"
            )
    
    # Get user
    result = await db.execute(
        select(User).where(User.id == api_key_obj.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or inactive"
        )
    
    # Update usage stats
    api_key_obj.last_used_at = datetime.now(timezone.utc)
    api_key_obj.last_used_ip = request.client.host if request.client else None
    api_key_obj.usage_count += 1
    await db.commit()
    
    return user, api_key_obj


def require_scope(required_scope: str):
    """
    Dependency factory to check API key scope.
    
    Usage:
        @router.get("/data", dependencies=[Depends(require_scope("knowledge"))])
        async def get_data(...): ...
    """
    async def check_scope(
        auth: tuple[User, APIKey] = Depends(get_api_key_user),
    ):
        user, api_key = auth
        
        if "full" in api_key.scopes:
            return user, api_key
        
        if required_scope not in api_key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required scope: {required_scope}"
            )
        
        return user, api_key
    
    return check_scope
