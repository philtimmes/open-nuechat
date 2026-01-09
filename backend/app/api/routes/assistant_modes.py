"""
Assistant Modes API Routes

CRUD operations for assistant mode presets.
Admin-only for create/update/delete, public for list/get.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db
from app.models import User, AssistantMode
from app.api.dependencies import get_current_user
from app.api.helpers import verify_admin

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant-modes", tags=["assistant-modes"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class AssistantModeBase(BaseModel):
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    active_tools: List[str] = []
    advertised_tools: List[str] = []
    filter_chain_id: Optional[str] = None
    sort_order: int = 0
    enabled: bool = True
    is_global: bool = True


class AssistantModeCreate(AssistantModeBase):
    pass


class AssistantModeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    active_tools: Optional[List[str]] = None
    advertised_tools: Optional[List[str]] = None
    filter_chain_id: Optional[str] = None
    sort_order: Optional[int] = None
    enabled: Optional[bool] = None
    is_global: Optional[bool] = None


# Emoji mapping for response serialization
MODE_EMOJIS = {
    "General": "🤖",
    "Creative Writing": "✍️",
    "Coding": "💻",
    "Deep Research": "🔬",
    "Legal": "⚖️",
    "Data Analysis": "📊",
    "Image Generation": "🎨",
}


class AssistantModeResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    emoji: str = "🤖"  # Derived, not from DB
    active_tools: List[str] = []
    advertised_tools: List[str] = []
    filter_chain_id: Optional[str] = None
    sort_order: int = 0
    enabled: bool = True
    is_global: bool = True
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm_with_emoji(cls, mode):
        """Create response with derived emoji."""
        data = {
            "id": mode.id,
            "name": mode.name,
            "description": mode.description,
            "icon": mode.icon,
            "emoji": MODE_EMOJIS.get(mode.name, "🤖"),
            "active_tools": mode.active_tools or [],
            "advertised_tools": mode.advertised_tools or [],
            "filter_chain_id": mode.filter_chain_id,
            "sort_order": mode.sort_order,
            "enabled": mode.enabled,
            "is_global": mode.is_global,
            "created_by": mode.created_by,
            "created_at": mode.created_at,
            "updated_at": mode.updated_at,
        }
        return cls(**data)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("", response_model=List[AssistantModeResponse])
async def list_assistant_modes(
    enabled_only: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all assistant modes.
    
    By default returns only enabled modes, sorted by sort_order.
    """
    query = select(AssistantMode)
    
    if enabled_only:
        query = query.where(AssistantMode.enabled == True)
    
    # Filter to global modes or modes user can access
    query = query.where(AssistantMode.is_global == True)
    
    query = query.order_by(AssistantMode.sort_order, AssistantMode.name)
    
    result = await db.execute(query)
    modes = result.scalars().all()
    
    return [AssistantModeResponse.from_orm_with_emoji(m) for m in modes]


@router.get("/{mode_id}", response_model=AssistantModeResponse)
async def get_assistant_mode(
    mode_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific assistant mode by ID."""
    result = await db.execute(
        select(AssistantMode).where(AssistantMode.id == mode_id)
    )
    mode = result.scalar_one_or_none()
    
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant mode not found"
        )
    
    return AssistantModeResponse.from_orm_with_emoji(mode)


@router.post("", response_model=AssistantModeResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant_mode(
    mode_data: AssistantModeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new assistant mode.
    
    Admin only.
    """
    verify_admin(current_user)
    
    # Check for duplicate name
    existing = await db.execute(
        select(AssistantMode).where(AssistantMode.name == mode_data.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Assistant mode '{mode_data.name}' already exists"
        )
    
    mode = AssistantMode(
        name=mode_data.name,
        description=mode_data.description,
        icon=mode_data.icon,
        active_tools=mode_data.active_tools,
        advertised_tools=mode_data.advertised_tools,
        filter_chain_id=mode_data.filter_chain_id,
        sort_order=mode_data.sort_order,
        enabled=mode_data.enabled,
        is_global=mode_data.is_global,
        created_by=current_user.id,
    )
    
    db.add(mode)
    await db.commit()
    await db.refresh(mode)
    
    logger.info(f"Created assistant mode: {mode.name} (id={mode.id})")
    
    return AssistantModeResponse.from_orm_with_emoji(mode)


@router.put("/{mode_id}", response_model=AssistantModeResponse)
async def update_assistant_mode(
    mode_id: str,
    mode_data: AssistantModeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update an assistant mode.
    
    Admin only.
    """
    verify_admin(current_user)
    
    result = await db.execute(
        select(AssistantMode).where(AssistantMode.id == mode_id)
    )
    mode = result.scalar_one_or_none()
    
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant mode not found"
        )
    
    # Check for duplicate name if name is being changed
    if mode_data.name and mode_data.name != mode.name:
        existing = await db.execute(
            select(AssistantMode).where(
                AssistantMode.name == mode_data.name,
                AssistantMode.id != mode_id
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Assistant mode '{mode_data.name}' already exists"
            )
    
    # Update fields
    update_data = mode_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(mode, field, value)
    
    await db.commit()
    await db.refresh(mode)
    
    logger.info(f"Updated assistant mode: {mode.name} (id={mode.id})")
    
    return AssistantModeResponse.from_orm_with_emoji(mode)


@router.delete("/{mode_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assistant_mode(
    mode_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an assistant mode.
    
    Admin only. This will set mode_id to NULL on any chats or assistants using it.
    """
    verify_admin(current_user)
    
    result = await db.execute(
        select(AssistantMode).where(AssistantMode.id == mode_id)
    )
    mode = result.scalar_one_or_none()
    
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant mode not found"
        )
    
    mode_name = mode.name
    
    await db.execute(
        delete(AssistantMode).where(AssistantMode.id == mode_id)
    )
    await db.commit()
    
    logger.info(f"Deleted assistant mode: {mode_name} (id={mode_id})")
    
    return None


@router.post("/{mode_id}/duplicate", response_model=AssistantModeResponse)
async def duplicate_assistant_mode(
    mode_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Duplicate an assistant mode.
    
    Admin only. Creates a copy with " (Copy)" appended to the name.
    """
    verify_admin(current_user)
    
    result = await db.execute(
        select(AssistantMode).where(AssistantMode.id == mode_id)
    )
    mode = result.scalar_one_or_none()
    
    if not mode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant mode not found"
        )
    
    # Generate unique name
    new_name = f"{mode.name} (Copy)"
    counter = 1
    while True:
        existing = await db.execute(
            select(AssistantMode).where(AssistantMode.name == new_name)
        )
        if not existing.scalar_one_or_none():
            break
        counter += 1
        new_name = f"{mode.name} (Copy {counter})"
    
    new_mode = AssistantMode(
        name=new_name,
        description=mode.description,
        icon=mode.icon,
        active_tools=mode.active_tools.copy() if mode.active_tools else [],
        advertised_tools=mode.advertised_tools.copy() if mode.advertised_tools else [],
        filter_chain_id=mode.filter_chain_id,
        sort_order=mode.sort_order + 1,
        enabled=mode.enabled,
        is_global=mode.is_global,
        created_by=current_user.id,
    )
    
    db.add(new_mode)
    await db.commit()
    await db.refresh(new_mode)
    
    logger.info(f"Duplicated assistant mode: {mode.name} -> {new_mode.name}")
    
    return AssistantModeResponse.from_orm_with_emoji(new_mode)
