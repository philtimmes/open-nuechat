"""
Custom Assistants (GPTs) Management Routes

Allows users to create, configure, and share custom AI assistants
with specific objectives, knowledge bases, and tools.
"""

from typing import List, Optional
from datetime import datetime, timezone, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload
import re

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.models.models import (
    User, CustomAssistant, KnowledgeStore, 
    AssistantConversation, Chat, assistant_knowledge_stores
)

router = APIRouter()


# Test route to verify router is loaded
@router.get("/test")
async def test_assistants_router():
    """Test endpoint to verify assistants router is loaded"""
    return {"status": "ok", "router": "assistants"}


# === Default Assistant Seeding ===

async def seed_default_assistant(db: AsyncSession) -> Optional[CustomAssistant]:
    """
    Create a default assistant if none exists.
    Called on application startup.
    """
    from app.core.config import settings
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Check if any system assistants exist
        result = await db.execute(
            select(CustomAssistant).where(CustomAssistant.slug == "nexus-assistant")
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            logger.debug(f"Default assistant already exists: {existing.name}")
            return existing
        
        # Get or create a system user for owning default assistants
        from app.models.models import UserTier
        result = await db.execute(
            select(User).where(User.email == "system@nexus.local")
        )
        system_user = result.scalar_one_or_none()
        
        if not system_user:
            system_user = User(
                email="system@nexus.local",
                username="Nexus",
                is_active=True,
                tier=UserTier.ENTERPRISE,
            )
            db.add(system_user)
            await db.flush()
            logger.debug("Created system user for default assistants")
        
        # Create default assistant
        default_assistant = CustomAssistant(
            owner_id=system_user.id,
            name="Nexus Assistant",
            slug="nexus-assistant",
            tagline="Your helpful AI assistant",
            description="A general-purpose AI assistant ready to help with any task. Ask questions, get explanations, brainstorm ideas, or just have a conversation.",
            icon="✨",
            color="#6366f1",
            system_prompt=settings.DEFAULT_SYSTEM_PROMPT,
            welcome_message="Hello! I'm your Nexus Assistant. How can I help you today?",
            suggested_prompts=[
                "Explain something complex in simple terms",
                "Help me brainstorm ideas for a project",
                "Write a short story about...",
                "What are the pros and cons of..."
            ],
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            enabled_tools=[],
            is_public=True,
            is_discoverable=True,
            is_featured=True,
        )
        
        db.add(default_assistant)
        await db.commit()
        await db.refresh(default_assistant)
        
        logger.debug(f"Created default assistant: {default_assistant.name}")
        return default_assistant
    except Exception as e:
        logger.error(f"Error seeding default assistant: {e}")
        await db.rollback()
        return None


# === Schemas ===

class AssistantCreate(BaseModel):
    """Request to create a new custom assistant"""
    name: str = Field(..., min_length=1, max_length=100)
    tagline: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    
    # Appearance
    icon: str = Field(default="🤖", max_length=100)
    color: str = Field(default="#6366f1", max_length=20)
    avatar_url: Optional[str] = None
    
    # Behavior
    system_prompt: str = Field(..., min_length=10, max_length=50000)
    welcome_message: Optional[str] = None
    suggested_prompts: List[str] = Field(default=[])
    
    # Model settings
    model: str = Field(default="claude-sonnet-4-20250514")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    
    # Capabilities
    enabled_tools: List[str] = Field(default=[])
    knowledge_store_ids: List[str] = Field(default=[])
    
    # Visibility
    is_public: bool = False
    is_discoverable: bool = False


class AssistantResponse(BaseModel):
    """Assistant info"""
    id: str
    owner_id: str
    name: str
    slug: str
    tagline: Optional[str]
    description: Optional[str]
    
    icon: str
    color: str
    avatar_url: Optional[str]
    
    system_prompt: str
    welcome_message: Optional[str]
    suggested_prompts: List[str]
    
    model: str
    temperature: float
    max_tokens: int
    
    enabled_tools: List[str]
    knowledge_store_ids: List[str] = []
    
    is_public: bool
    is_discoverable: bool
    is_featured: bool
    
    conversation_count: int
    message_count: int
    average_rating: float
    
    version: int
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    # Computed
    owner_username: Optional[str] = None
    
    class Config:
        from_attributes = True


class AssistantUpdate(BaseModel):
    """Update assistant settings"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    tagline: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    
    icon: Optional[str] = Field(None, max_length=100)
    color: Optional[str] = Field(None, max_length=20)
    avatar_url: Optional[str] = None
    
    system_prompt: Optional[str] = Field(None, min_length=10, max_length=50000)
    welcome_message: Optional[str] = None
    suggested_prompts: Optional[List[str]] = None
    
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    
    enabled_tools: Optional[List[str]] = None
    knowledge_store_ids: Optional[List[str]] = None
    
    is_public: Optional[bool] = None
    is_discoverable: Optional[bool] = None


class AssistantPublicInfo(BaseModel):
    """Public info for assistant discovery"""
    id: str
    name: str
    slug: str
    tagline: Optional[str]
    description: Optional[str]
    icon: str
    color: str
    avatar_url: Optional[str]
    welcome_message: Optional[str]
    suggested_prompts: List[str]
    owner_username: str
    conversation_count: int
    average_rating: float
    is_featured: bool
    
    class Config:
        from_attributes = True


class ConversationRating(BaseModel):
    """Rate a conversation with an assistant"""
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None


# === Helper Functions ===

def generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a name"""
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    return slug


async def get_unique_slug(db: AsyncSession, base_slug: str, exclude_id: Optional[str] = None) -> str:
    """Generate a unique slug by appending numbers if needed"""
    slug = base_slug
    counter = 1
    
    while True:
        query = select(CustomAssistant).where(CustomAssistant.slug == slug)
        if exclude_id:
            query = query.where(CustomAssistant.id != exclude_id)
        
        result = await db.execute(query)
        if not result.scalar_one_or_none():
            return slug
        
        slug = f"{base_slug}-{counter}"
        counter += 1


async def get_assistant_with_access(
    assistant_id: str,
    user: User,
    db: AsyncSession,
    require_owner: bool = False,
) -> CustomAssistant:
    """Get an assistant and verify access"""
    result = await db.execute(
        select(CustomAssistant)
        .where(CustomAssistant.id == assistant_id)
        .options(selectinload(CustomAssistant.knowledge_stores))
    )
    assistant = result.scalar_one_or_none()
    
    if not assistant:
        # Try by slug
        result = await db.execute(
            select(CustomAssistant)
            .where(CustomAssistant.slug == assistant_id)
            .options(selectinload(CustomAssistant.knowledge_stores))
        )
        assistant = result.scalar_one_or_none()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant not found"
        )
    
    if require_owner and assistant.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't own this assistant"
        )
    
    # Check access for non-public assistants
    if not assistant.is_public and assistant.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this assistant"
        )
    
    return assistant


# === Endpoints: Assistants ===

@router.post("", response_model=AssistantResponse)
async def create_assistant(
    request: AssistantCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new custom assistant"""
    # Admins bypass assistant limits
    if not current_user.is_admin:
        # Check assistant limit
        result = await db.execute(
            select(func.count(CustomAssistant.id)).where(
                CustomAssistant.owner_id == current_user.id
            )
        )
        assistant_count = result.scalar()
        
        limits = {"free": 3, "pro": 20, "enterprise": 100}
        tier_limit = limits.get(current_user.tier.value, 3)
        
        if assistant_count >= tier_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assistant limit reached ({tier_limit} for {current_user.tier.value} tier)"
            )
    
    # Generate unique slug
    base_slug = generate_slug(request.name)
    slug = await get_unique_slug(db, base_slug)
    
    # Verify knowledge store access
    if request.knowledge_store_ids:
        for store_id in request.knowledge_store_ids:
            result = await db.execute(
                select(KnowledgeStore).where(
                    KnowledgeStore.id == store_id,
                    or_(
                        KnowledgeStore.owner_id == current_user.id,
                        KnowledgeStore.is_public == True
                    )
                )
            )
            if not result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Knowledge store not found or inaccessible: {store_id}"
                )
    
    # Create assistant
    assistant = CustomAssistant(
        owner_id=current_user.id,
        name=request.name,
        slug=slug,
        tagline=request.tagline,
        description=request.description,
        icon=request.icon,
        color=request.color,
        avatar_url=request.avatar_url,
        system_prompt=request.system_prompt,
        welcome_message=request.welcome_message,
        suggested_prompts=request.suggested_prompts,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        enabled_tools=request.enabled_tools,
        is_public=request.is_public,
        is_discoverable=request.is_discoverable,
    )
    
    db.add(assistant)
    await db.flush()
    
    # Link knowledge stores using association table directly (avoids lazy load issue)
    if request.knowledge_store_ids:
        for store_id in request.knowledge_store_ids:
            await db.execute(
                assistant_knowledge_stores.insert().values(
                    assistant_id=assistant.id,
                    knowledge_store_id=store_id
                )
            )
    
    await db.commit()
    await db.refresh(assistant)
    
    # Now we can safely load the relationship
    await db.refresh(assistant, ["knowledge_stores"])
    
    response = AssistantResponse.model_validate(assistant)
    response.owner_username = current_user.username
    response.knowledge_store_ids = [ks.id for ks in assistant.knowledge_stores]
    
    return response


@router.get("", response_model=List[AssistantResponse])
async def list_my_assistants(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all assistants owned by the current user"""
    result = await db.execute(
        select(CustomAssistant)
        .where(CustomAssistant.owner_id == current_user.id)
        .options(selectinload(CustomAssistant.knowledge_stores))
        .order_by(CustomAssistant.updated_at.desc())
    )
    assistants = result.scalars().all()
    
    responses = []
    for assistant in assistants:
        response = AssistantResponse.model_validate(assistant)
        response.owner_username = current_user.username
        response.knowledge_store_ids = [ks.id for ks in assistant.knowledge_stores]
        responses.append(response)
    
    return responses


@router.get("/explore", response_model=List[AssistantPublicInfo])
async def explore_assistants(
    search: Optional[str] = None,
    featured_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    Explore public assistants (no auth required).
    Returns discoverable assistants sorted by featured status and popularity.
    """
    query = (
        select(CustomAssistant, User)
        .join(User, CustomAssistant.owner_id == User.id)
        .where(
            CustomAssistant.is_public == True,
            CustomAssistant.is_discoverable == True
        )
    )
    
    if featured_only:
        query = query.where(CustomAssistant.is_featured == True)
    
    if search:
        query = query.where(
            or_(
                CustomAssistant.name.ilike(f"%{search}%"),
                CustomAssistant.tagline.ilike(f"%{search}%"),
                CustomAssistant.description.ilike(f"%{search}%")
            )
        )
    
    query = query.order_by(
        CustomAssistant.is_featured.desc(),
        CustomAssistant.conversation_count.desc()
    ).offset(offset).limit(limit)
    
    result = await db.execute(query)
    
    responses = []
    for assistant, owner in result:
        response = AssistantPublicInfo(
            id=assistant.id,
            name=assistant.name,
            slug=assistant.slug,
            tagline=assistant.tagline,
            description=assistant.description,
            icon=assistant.icon,
            color=assistant.color,
            avatar_url=assistant.avatar_url,
            welcome_message=assistant.welcome_message,
            suggested_prompts=assistant.suggested_prompts,
            owner_username=owner.username,
            conversation_count=assistant.conversation_count,
            average_rating=assistant.average_rating,
            is_featured=assistant.is_featured,
        )
        responses.append(response)
    
    return responses


@router.get("/discover", response_model=List[AssistantPublicInfo])
async def discover_assistants(
    search: Optional[str] = None,
    featured_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Discover public assistants"""
    query = (
        select(CustomAssistant, User)
        .join(User, CustomAssistant.owner_id == User.id)
        .where(
            CustomAssistant.is_public == True,
            CustomAssistant.is_discoverable == True
        )
    )
    
    if featured_only:
        query = query.where(CustomAssistant.is_featured == True)
    
    if search:
        query = query.where(
            or_(
                CustomAssistant.name.ilike(f"%{search}%"),
                CustomAssistant.tagline.ilike(f"%{search}%"),
                CustomAssistant.description.ilike(f"%{search}%")
            )
        )
    
    query = query.order_by(
        CustomAssistant.is_featured.desc(),
        CustomAssistant.conversation_count.desc()
    ).offset(offset).limit(limit)
    
    result = await db.execute(query)
    
    responses = []
    for assistant, owner in result:
        response = AssistantPublicInfo(
            id=assistant.id,
            name=assistant.name,
            slug=assistant.slug,
            tagline=assistant.tagline,
            description=assistant.description,
            icon=assistant.icon,
            color=assistant.color,
            avatar_url=assistant.avatar_url,
            welcome_message=assistant.welcome_message,
            suggested_prompts=assistant.suggested_prompts,
            owner_username=owner.username,
            conversation_count=assistant.conversation_count,
            average_rating=assistant.average_rating,
            is_featured=assistant.is_featured,
        )
        responses.append(response)
    
    return responses


@router.get("/subscribed", response_model=List[AssistantPublicInfo])
async def list_subscribed_assistants(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all assistants you're subscribed to (including your own)"""
    from sqlalchemy import text, or_
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Read preferences with raw SQL
    result = await db.execute(
        text("SELECT preferences FROM users WHERE id = :user_id"),
        {"user_id": current_user.id}
    )
    row = result.fetchone()
    
    logger.debug(f"list_subscribed: Raw preferences for user {current_user.id}: {row[0] if row else 'NOT FOUND'}")
    
    prefs = {}
    if row and row[0]:
        prefs_raw = row[0]
        if isinstance(prefs_raw, str):
            prefs = json.loads(prefs_raw)
        else:
            prefs = dict(prefs_raw)
    
    subscribed_ids = prefs.get("subscribed_assistants", [])
    logger.debug(f"list_subscribed: Subscribed IDs: {subscribed_ids}")
    
    # Build query: either subscribed OR owned by current user
    query = (
        select(CustomAssistant, User)
        .join(User, CustomAssistant.owner_id == User.id)
    )
    
    if subscribed_ids:
        # Include subscribed assistants AND owned assistants
        query = query.where(
            or_(
                CustomAssistant.id.in_(subscribed_ids),
                CustomAssistant.owner_id == current_user.id
            )
        )
    else:
        # Only owned assistants
        query = query.where(CustomAssistant.owner_id == current_user.id)
    
    query = query.order_by(CustomAssistant.name)
    result = await db.execute(query)
    
    responses = []
    seen_ids = set()  # Avoid duplicates
    for assistant, owner in result:
        if assistant.id in seen_ids:
            continue
        seen_ids.add(assistant.id)
        
        responses.append(AssistantPublicInfo(
            id=assistant.id,
            name=assistant.name,
            slug=assistant.slug,
            tagline=assistant.tagline,
            description=assistant.description,
            icon=assistant.icon,
            color=assistant.color,
            avatar_url=assistant.avatar_url,
            welcome_message=assistant.welcome_message,
            suggested_prompts=assistant.suggested_prompts or [],
            owner_username=owner.username,
            conversation_count=assistant.conversation_count,
            average_rating=assistant.average_rating,
            is_featured=assistant.is_featured,
        ))
    
    return responses


@router.get("/subscribed/stats")
async def get_subscription_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics for subscribed assistants (including owned)"""
    from sqlalchemy import text, func as sql_func, or_
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Read preferences with raw SQL
    result = await db.execute(
        text("SELECT preferences FROM users WHERE id = :user_id"),
        {"user_id": current_user.id}
    )
    row = result.fetchone()
    
    prefs = {}
    if row and row[0]:
        prefs_raw = row[0]
        if isinstance(prefs_raw, str):
            prefs = json.loads(prefs_raw)
        else:
            prefs = dict(prefs_raw)
    
    subscribed_ids = prefs.get("subscribed_assistants", [])
    
    # Get assistants: either subscribed OR owned by current user
    query = (
        select(CustomAssistant, User)
        .join(User, CustomAssistant.owner_id == User.id)
    )
    
    if subscribed_ids:
        query = query.where(
            or_(
                CustomAssistant.id.in_(subscribed_ids),
                CustomAssistant.owner_id == current_user.id
            )
        )
    else:
        query = query.where(CustomAssistant.owner_id == current_user.id)
    
    assistants_result = await db.execute(query)
    
    # Get assistant details with usage stats
    stats = []
    seen_ids = set()
    for assistant, owner in assistants_result:
        if assistant.id in seen_ids:
            continue
        seen_ids.add(assistant.id)
        
        # Get user's conversation count with this assistant
        conv_result = await db.execute(
            select(sql_func.count(AssistantConversation.id))
            .where(
                AssistantConversation.assistant_id == assistant.id,
                AssistantConversation.user_id == current_user.id
            )
        )
        conversation_count = conv_result.scalar() or 0
        
        # Get user's message count from chats with this assistant
        message_result = await db.execute(
            text("""
                SELECT COUNT(m.id), COALESCE(SUM(m.input_tokens), 0), COALESCE(SUM(m.output_tokens), 0)
                FROM messages m
                JOIN assistant_conversations ac ON m.chat_id = ac.chat_id
                WHERE ac.assistant_id = :assistant_id AND ac.user_id = :user_id
            """),
            {"assistant_id": assistant.id, "user_id": current_user.id}
        )
        msg_row = message_result.first()
        message_count = msg_row[0] if msg_row else 0
        input_tokens = msg_row[1] if msg_row else 0
        output_tokens = msg_row[2] if msg_row else 0
        
        stats.append({
            "id": assistant.id,
            "name": assistant.name,
            "slug": assistant.slug,
            "icon": assistant.icon,
            "color": assistant.color,
            "tagline": assistant.tagline,
            "owner_username": owner.username,
            "conversation_count": conversation_count,
            "message_count": message_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        })
    
    return stats


@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific assistant by ID or slug"""
    assistant = await get_assistant_with_access(assistant_id, current_user, db)
    
    result = await db.execute(
        select(User).where(User.id == assistant.owner_id)
    )
    owner = result.scalar_one()
    
    response = AssistantResponse.model_validate(assistant)
    response.owner_username = owner.username
    response.knowledge_store_ids = [ks.id for ks in assistant.knowledge_stores]
    
    return response


@router.patch("/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(
    assistant_id: str,
    update_data: AssistantUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an assistant (owner only)"""
    assistant = await get_assistant_with_access(
        assistant_id, current_user, db, require_owner=True
    )
    
    update_dict = update_data.model_dump(exclude_unset=True)
    
    # Handle slug update if name changes
    if "name" in update_dict:
        base_slug = generate_slug(update_dict["name"])
        assistant.slug = await get_unique_slug(db, base_slug, assistant.id)
    
    # Handle knowledge store updates
    if "knowledge_store_ids" in update_dict:
        store_ids = update_dict.pop("knowledge_store_ids")
        
        # Validate all stores first
        for store_id in store_ids:
            result = await db.execute(
                select(KnowledgeStore).where(
                    KnowledgeStore.id == store_id,
                    or_(
                        KnowledgeStore.owner_id == current_user.id,
                        KnowledgeStore.is_public == True
                    )
                )
            )
            store = result.scalar_one_or_none()
            if not store:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Knowledge store not found or inaccessible: {store_id}"
                )
        
        # Clear existing using direct delete on association table
        await db.execute(
            assistant_knowledge_stores.delete().where(
                assistant_knowledge_stores.c.assistant_id == assistant.id
            )
        )
        
        # Add new using direct insert on association table
        for store_id in store_ids:
            await db.execute(
                assistant_knowledge_stores.insert().values(
                    assistant_id=assistant.id,
                    knowledge_store_id=store_id
                )
            )
    
    # Update version on significant changes
    if any(k in update_dict for k in ["system_prompt", "enabled_tools", "model"]):
        assistant.version += 1
    
    for field, value in update_dict.items():
        setattr(assistant, field, value)
    
    await db.commit()
    await db.refresh(assistant)
    
    # Load knowledge stores relationship
    await db.refresh(assistant, ["knowledge_stores"])
    
    response = AssistantResponse.model_validate(assistant)
    response.owner_username = current_user.username
    response.knowledge_store_ids = [ks.id for ks in assistant.knowledge_stores]
    
    return response


@router.delete("/{assistant_id}")
async def delete_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an assistant (owner only)"""
    assistant = await get_assistant_with_access(
        assistant_id, current_user, db, require_owner=True
    )
    
    await db.delete(assistant)
    await db.commit()
    
    return {"message": "Assistant deleted", "id": assistant_id}


@router.post("/{assistant_id}/publish")
async def publish_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Publish an assistant (make it public and discoverable)"""
    assistant = await get_assistant_with_access(
        assistant_id, current_user, db, require_owner=True
    )
    
    assistant.is_public = True
    assistant.is_discoverable = True
    assistant.published_at = datetime.now(timezone.utc)
    
    await db.commit()
    
    return {
        "message": "Assistant published",
        "id": assistant.id,
        "slug": assistant.slug,
        "url": f"/assistants/{assistant.slug}",
    }


@router.post("/{assistant_id}/unpublish")
async def unpublish_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Unpublish an assistant (make it private)"""
    assistant = await get_assistant_with_access(
        assistant_id, current_user, db, require_owner=True
    )
    
    assistant.is_public = False
    assistant.is_discoverable = False
    
    await db.commit()
    
    return {"message": "Assistant unpublished", "id": assistant.id}


@router.post("/{assistant_id}/duplicate", response_model=AssistantResponse)
async def duplicate_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Duplicate an assistant.
    
    If duplicating someone else's public assistant, creates a copy owned by you.
    """
    original = await get_assistant_with_access(assistant_id, current_user, db)
    
    # Generate new slug
    base_slug = generate_slug(f"{original.name} copy")
    slug = await get_unique_slug(db, base_slug)
    
    # Create copy
    new_assistant = CustomAssistant(
        owner_id=current_user.id,
        name=f"{original.name} (Copy)",
        slug=slug,
        tagline=original.tagline,
        description=original.description,
        icon=original.icon,
        color=original.color,
        avatar_url=original.avatar_url,
        system_prompt=original.system_prompt,
        welcome_message=original.welcome_message,
        suggested_prompts=original.suggested_prompts.copy() if original.suggested_prompts else [],
        model=original.model,
        temperature=original.temperature,
        max_tokens=original.max_tokens,
        enabled_tools=original.enabled_tools.copy() if original.enabled_tools else [],
        is_public=False,
        is_discoverable=False,
    )
    
    db.add(new_assistant)
    await db.flush()
    
    # Load original's knowledge stores
    await db.refresh(original, ["knowledge_stores"])
    
    # Copy knowledge store links (only ones user has access to) using direct insert
    for store in original.knowledge_stores:
        if store.owner_id == current_user.id or store.is_public:
            await db.execute(
                assistant_knowledge_stores.insert().values(
                    assistant_id=new_assistant.id,
                    knowledge_store_id=store.id
                )
            )
    
    await db.commit()
    await db.refresh(new_assistant)
    await db.refresh(new_assistant, ["knowledge_stores"])
    
    response = AssistantResponse.model_validate(new_assistant)
    response.owner_username = current_user.username
    response.knowledge_store_ids = [ks.id for ks in new_assistant.knowledge_stores]
    
    return response


# === Endpoints: Conversations ===

@router.post("/{assistant_id}/start")
async def start_conversation(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start a new conversation with an assistant.
    Creates a chat pre-configured with the assistant's settings.
    """
    assistant = await get_assistant_with_access(assistant_id, current_user, db)
    
    # Create chat with assistant's settings
    chat = Chat(
        owner_id=current_user.id,
        title=f"Chat with {assistant.name}",
        model=assistant.model,
        system_prompt=assistant.system_prompt,
    )
    
    db.add(chat)
    await db.flush()
    
    # Track conversation
    conversation = AssistantConversation(
        assistant_id=assistant.id,
        chat_id=chat.id,
        user_id=current_user.id,
    )
    
    db.add(conversation)
    
    # Update assistant stats
    assistant.conversation_count += 1
    
    await db.commit()
    
    return {
        "chat_id": chat.id,
        "assistant_id": assistant.id,
        "assistant_name": assistant.name,
        "welcome_message": assistant.welcome_message,
        "suggested_prompts": assistant.suggested_prompts,
        "knowledge_stores": [
            {"id": ks.id, "name": ks.name}
            for ks in assistant.knowledge_stores
        ],
    }


@router.post("/{assistant_id}/conversations/{chat_id}/rate")
async def rate_conversation(
    assistant_id: str,
    chat_id: str,
    rating_data: ConversationRating,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rate a conversation with an assistant"""
    result = await db.execute(
        select(AssistantConversation).where(
            AssistantConversation.assistant_id == assistant_id,
            AssistantConversation.chat_id == chat_id,
            AssistantConversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Update conversation rating
    old_rating = conversation.rating
    conversation.rating = rating_data.rating
    conversation.feedback = rating_data.feedback
    
    # Update assistant rating stats
    result = await db.execute(
        select(CustomAssistant).where(CustomAssistant.id == assistant_id)
    )
    assistant = result.scalar_one()
    
    if old_rating:
        # Update existing rating
        assistant.rating_sum = assistant.rating_sum - old_rating + rating_data.rating
    else:
        # New rating
        assistant.rating_sum += rating_data.rating
        assistant.rating_count += 1
    
    await db.commit()
    
    return {
        "message": "Rating saved",
        "rating": rating_data.rating,
        "assistant_average": assistant.average_rating,
    }


@router.get("/{assistant_id}/conversations", response_model=List[dict])
async def list_my_conversations(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List my conversations with a specific assistant"""
    assistant = await get_assistant_with_access(assistant_id, current_user, db)
    
    result = await db.execute(
        select(AssistantConversation, Chat)
        .join(Chat, AssistantConversation.chat_id == Chat.id)
        .where(
            AssistantConversation.assistant_id == assistant_id,
            AssistantConversation.user_id == current_user.id
        )
        .order_by(Chat.updated_at.desc())
    )
    
    conversations = []
    for conv, chat in result:
        conversations.append({
            "conversation_id": conv.id,
            "chat_id": chat.id,
            "title": chat.title,
            "rating": conv.rating,
            "created_at": conv.created_at,
            "updated_at": chat.updated_at,
        })
    
    return conversations


# === Endpoints: Subscriptions ===

@router.post("/{assistant_id}/subscribe")
async def subscribe_to_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Subscribe to an assistant (adds to your model list)"""
    from sqlalchemy import text
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Verify assistant exists and is accessible
    assistant = await get_assistant_with_access(assistant_id, current_user, db)
    
    # Read current preferences directly with raw SQL
    result = await db.execute(
        text("SELECT preferences FROM users WHERE id = :user_id"),
        {"user_id": current_user.id}
    )
    row = result.fetchone()
    current_prefs_raw = row[0] if row else None
    
    logger.debug(f"Raw preferences from DB: {current_prefs_raw}")
    
    # Parse preferences
    if current_prefs_raw:
        if isinstance(current_prefs_raw, str):
            prefs = json.loads(current_prefs_raw)
        else:
            prefs = dict(current_prefs_raw)
    else:
        prefs = {}
    
    subscribed = list(prefs.get("subscribed_assistants", []))
    
    logger.debug(f"User {current_user.id} subscribing to {assistant_id}")
    logger.debug(f"Current subscriptions: {subscribed}")
    
    if assistant_id in subscribed:
        return {"message": "Already subscribed", "subscribed": True}
    
    # Add subscription
    subscribed.append(assistant_id)
    prefs["subscribed_assistants"] = subscribed
    
    # Serialize to JSON string for SQLite
    prefs_json = json.dumps(prefs)
    logger.debug(f"New prefs JSON: {prefs_json}")
    
    # Use raw SQL UPDATE
    await db.execute(
        text("UPDATE users SET preferences = :prefs WHERE id = :user_id"),
        {"prefs": prefs_json, "user_id": current_user.id}
    )
    await db.commit()
    
    # Verify with raw SQL
    result = await db.execute(
        text("SELECT preferences FROM users WHERE id = :user_id"),
        {"user_id": current_user.id}
    )
    row = result.fetchone()
    logger.debug(f"After commit, raw preferences: {row[0] if row else 'NOT FOUND'}")
    
    return {
        "message": f"Subscribed to {assistant.name}",
        "subscribed": True,
        "assistant_id": assistant_id,
        "assistant_name": assistant.name,
    }


@router.delete("/{assistant_id}/subscribe")
async def unsubscribe_from_assistant(
    assistant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Unsubscribe from an assistant (removes from your model list)"""
    from sqlalchemy import text
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Read current preferences directly with raw SQL
    result = await db.execute(
        text("SELECT preferences FROM users WHERE id = :user_id"),
        {"user_id": current_user.id}
    )
    row = result.fetchone()
    current_prefs_raw = row[0] if row else None
    
    # Parse preferences
    if current_prefs_raw:
        if isinstance(current_prefs_raw, str):
            prefs = json.loads(current_prefs_raw)
        else:
            prefs = dict(current_prefs_raw)
    else:
        prefs = {}
    
    subscribed = list(prefs.get("subscribed_assistants", []))
    
    if assistant_id not in subscribed:
        return {"message": "Not subscribed", "subscribed": False}
    
    # Remove subscription
    subscribed.remove(assistant_id)
    prefs["subscribed_assistants"] = subscribed
    
    # Serialize to JSON string for SQLite
    prefs_json = json.dumps(prefs)
    
    # Use raw SQL UPDATE
    await db.execute(
        text("UPDATE users SET preferences = :prefs WHERE id = :user_id"),
        {"prefs": prefs_json, "user_id": current_user.id}
    )
    await db.commit()
    
    logger.debug(f"User {current_user.id} unsubscribed from {assistant_id}")
    
    return {
        "message": "Unsubscribed",
        "subscribed": False,
        "assistant_id": assistant_id,
    }
