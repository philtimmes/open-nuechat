"""
Chat API routes
"""
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, or_, and_
from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.api.schemas import (
    ChatCreate, ChatUpdate, ChatResponse, ChatListResponse,
    MessageCreate, MessageResponse, MessageEdit, ChatInvite, ClientMessageCreate,
    CodeSummaryCreate, CodeSummaryResponse, FileChange, SignatureWarning
)
from app.models.models import User, Chat, Message, MessageRole, ContentType, ChatParticipant
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.billing import BillingService
from app.tools.registry import tool_registry
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chats"])

# Directory where generated images are stored
GENERATED_IMAGES_DIR = Path("/app/uploads/generated")


async def delete_chat_images(db: AsyncSession, chat_id: str) -> int:
    """
    Delete all generated images associated with a chat.
    
    Returns the number of images deleted.
    """
    deleted_count = 0
    
    # Find all messages in this chat that have generated images
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id)
    )
    messages = result.scalars().all()
    
    for message in messages:
        if message.message_metadata and isinstance(message.message_metadata, dict):
            generated_image = message.message_metadata.get("generated_image")
            if generated_image:
                job_id = generated_image.get("job_id")
                if job_id:
                    image_path = GENERATED_IMAGES_DIR / f"{job_id}.png"
                    if image_path.exists():
                        try:
                            image_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted image: {image_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete image {image_path}: {e}")
    
    return deleted_count


async def delete_user_images(db: AsyncSession, user_id: str) -> int:
    """
    Delete all generated images for all chats owned by a user.
    
    Returns the number of images deleted.
    """
    deleted_count = 0
    
    # Find all chats owned by this user
    result = await db.execute(
        select(Chat.id).where(Chat.owner_id == user_id)
    )
    chat_ids = [row[0] for row in result.all()]
    
    for chat_id in chat_ids:
        deleted_count += await delete_chat_images(db, chat_id)
    
    return deleted_count


async def delete_message_images(db: AsyncSession, message_ids: set) -> int:
    """
    Delete generated images for specific messages.
    
    Returns the number of images deleted.
    """
    deleted_count = 0
    
    # Find messages with generated images
    result = await db.execute(
        select(Message).where(Message.id.in_(message_ids))
    )
    messages = result.scalars().all()
    
    for message in messages:
        if message.message_metadata and isinstance(message.message_metadata, dict):
            generated_image = message.message_metadata.get("generated_image")
            if generated_image:
                job_id = generated_image.get("job_id")
                if job_id:
                    image_path = GENERATED_IMAGES_DIR / f"{job_id}.png"
                    if image_path.exists():
                        try:
                            image_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted image: {image_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete image {image_path}: {e}")
    
    return deleted_count


@router.get("", response_model=ChatListResponse)
async def list_chats(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    sort_by: Optional[str] = Query("modified", description="Sort by: modified, created, alphabetical, source"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's chats with pagination. Returns group counts for sidebar accordions.
    
    For grouped views (source, period), frontend should use /chats/group/{type}/{name} endpoint
    to fetch chats for each accordion independently.
    """
    base_filter = Chat.owner_id == user.id
    
    # Get group counts for sidebar display (only when not searching)
    group_counts = None
    if not search:
        group_counts = await _get_group_counts(db, user.id, sort_by or "modified")
    
    # For alphabetical, return paginated list (no grouping)
    if sort_by == "alphabetical":
        if search:
            message_search = (
                select(Message.chat_id)
                .join(Chat, Message.chat_id == Chat.id)
                .where(Chat.owner_id == user.id)
                .where(Message.content.ilike(f"%{search}%"))
                .distinct()
            )
            message_result = await db.execute(message_search)
            matching_chat_ids = [row[0] for row in message_result.all()]
            
            search_filter = or_(
                Chat.title.ilike(f"%{search}%"),
                Chat.id.in_(matching_chat_ids) if matching_chat_ids else False
            )
            query = select(Chat).where(base_filter, search_filter)
            count_query = select(func.count(Chat.id)).where(base_filter, search_filter)
        else:
            query = select(Chat).where(base_filter)
            count_query = select(func.count(Chat.id)).where(base_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        query = query.order_by(Chat.title.asc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        chats = result.scalars().all()
        
        return ChatListResponse(
            chats=[ChatResponse.model_validate(c) for c in chats],
            total=total,
            page=page,
            page_size=page_size,
            group_counts=group_counts,
        )
    
    # For grouped views (source, period), just return counts
    # Frontend fetches each group via /chats/group/{type}/{name}
    # NC-0.8.0.27: When searching, return flat results across all groups
    if search:
        message_search = (
            select(Message.chat_id)
            .join(Chat, Message.chat_id == Chat.id)
            .where(Chat.owner_id == user.id)
            .where(Message.content.ilike(f"%{search}%"))
            .distinct()
        )
        message_result = await db.execute(message_search)
        matching_chat_ids = [row[0] for row in message_result.all()]
        
        search_filter = or_(
            Chat.title.ilike(f"%{search}%"),
            Chat.id.in_(matching_chat_ids) if matching_chat_ids else False
        )
        query = select(Chat).where(base_filter, search_filter).order_by(Chat.updated_at.desc())
        count_query = select(func.count(Chat.id)).where(base_filter, search_filter)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(query)
        chats = result.scalars().all()
        
        return ChatListResponse(
            chats=[ChatResponse.model_validate(c) for c in chats],
            total=total,
            page=page,
            page_size=page_size,
            group_counts={"Search Results": total} if total else {},
        )
    
    total_result = await db.execute(select(func.count(Chat.id)).where(base_filter))
    total = total_result.scalar()
    
    return ChatListResponse(
        chats=[],  # Empty - frontend fetches per-group
        total=total,
        page=page,
        page_size=page_size,
        group_counts=group_counts,
    )


async def _get_group_counts(db: AsyncSession, user_id: str, sort_by: str) -> dict:
    """Get counts of chats per group for sidebar display."""
    from sqlalchemy import case, literal
    
    if sort_by == "source":
        # Count by source field
        source_case = case(
            (Chat.source == 'chatgpt', literal('ChatGPT')),
            (Chat.source == 'grok', literal('Grok')),
            (Chat.source == 'claude', literal('Claude')),
            else_=literal('Native')
        )
        query = (
            select(source_case.label('group_name'), func.count(Chat.id).label('count'))
            .where(Chat.owner_id == user_id)
            .group_by(source_case)
        )
        result = await db.execute(query)
        return {row.group_name: row.count for row in result.all()}
    
    elif sort_by == "alphabetical":
        # Single group - just return total
        query = select(func.count(Chat.id)).where(Chat.owner_id == user_id)
        result = await db.execute(query)
        total = result.scalar() or 0
        return {"All Chats": total}
    
    else:
        # Date-based grouping (modified or created)
        # Need to fetch dates and categorize in Python since SQLite date functions are limited
        date_field = Chat.updated_at if sort_by == "modified" else Chat.created_at
        query = select(date_field).where(Chat.owner_id == user_id)
        result = await db.execute(query)
        raw_dates = [row[0] for row in result.all()]
        
        # Use naive local datetime for comparison (SQLite stores naive datetimes as text)
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seven_days_ago = today_start - timedelta(days=7)
        thirty_days_ago = today_start - timedelta(days=30)
        
        counts = {"Today": 0, "Last 7 Days": 0, "Last 30 Days": 0, "Older": 0}
        
        for raw_dt in raw_dates:
            if raw_dt is None:
                counts["Older"] += 1
                continue
            
            # Parse string to datetime if needed (SQLite returns text)
            if isinstance(raw_dt, str):
                try:
                    dt = datetime.fromisoformat(raw_dt.replace(' ', 'T'))
                except ValueError:
                    counts["Older"] += 1
                    continue
            else:
                dt = raw_dt
                # Strip timezone if present (compare as naive)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
            
            if dt >= today_start:
                counts["Today"] += 1
            elif dt >= seven_days_ago:
                counts["Last 7 Days"] += 1
            elif dt >= thirty_days_ago:
                counts["Last 30 Days"] += 1
            else:
                counts["Older"] += 1
        
        return counts


@router.post("", response_model=ChatResponse)
async def create_chat(
    chat_data: ChatCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat"""
    
    chat = Chat(
        owner_id=user.id,
        title=chat_data.title or "New Chat",
        model=chat_data.model or settings.LLM_MODEL,
        system_prompt=chat_data.system_prompt,
        is_shared=chat_data.is_shared,
    )
    db.add(chat)
    
    # Add owner as participant if shared chat
    if chat_data.is_shared:
        participant = ChatParticipant(
            chat_id=chat.id,
            user_id=user.id,
            role="owner",
        )
        db.add(participant)
    
    await db.commit()
    await db.refresh(chat)
    
    return chat


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific chat"""
    
    chat = await _get_user_chat(db, user, chat_id)
    return chat


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    updates: ChatUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a chat"""
    from app.models.models import CustomAssistant, AssistantConversation
    
    chat = await _get_user_chat(db, user, chat_id)
    
    if updates.title is not None:
        chat.title = updates.title
    if updates.model is not None:
        new_model = updates.model
        
        # Handle "gpt:assistant_id" format - resolve to actual model and link conversation
        if new_model.startswith("gpt:"):
            assistant_id = new_model[4:]
            
            # Look up the assistant
            result = await db.execute(
                select(CustomAssistant).where(CustomAssistant.id == assistant_id)
            )
            assistant = result.scalar_one_or_none()
            
            if assistant:
                # Use the assistant's actual model
                chat.model = assistant.model
                chat.system_prompt = assistant.system_prompt
                chat.assistant_id = assistant.id
                chat.assistant_name = assistant.name
                
                # Copy mode and tools from assistant
                if assistant.mode_id:
                    chat.mode_id = assistant.mode_id
                    # Get mode's active_tools
                    from app.models import AssistantMode
                    mode_result = await db.execute(
                        select(AssistantMode).where(AssistantMode.id == assistant.mode_id)
                    )
                    mode = mode_result.scalar_one_or_none()
                    if mode:
                        chat.active_tools = mode.active_tools
                
                # Check if already linked to this assistant
                existing_conv = await db.execute(
                    select(AssistantConversation).where(
                        AssistantConversation.chat_id == chat_id
                    )
                )
                existing = existing_conv.scalar_one_or_none()
                
                if not existing:
                    # Create assistant conversation link
                    conversation = AssistantConversation(
                        assistant_id=assistant.id,
                        chat_id=chat.id,
                        user_id=user.id,
                    )
                    db.add(conversation)
                    assistant.conversation_count += 1
                elif existing.assistant_id != assistant_id:
                    # Switching to a different assistant
                    existing.assistant_id = assistant_id
            else:
                # Assistant not found, store the raw model value
                chat.model = new_model
        else:
            # Switching to a regular model - clear assistant association
            chat.model = new_model
            chat.assistant_id = None
            chat.assistant_name = None
            
            # Remove assistant conversation link if exists
            existing_conv = await db.execute(
                select(AssistantConversation).where(
                    AssistantConversation.chat_id == chat_id
                )
            )
            existing = existing_conv.scalar_one_or_none()
            if existing:
                await db.delete(existing)
            
    if updates.system_prompt is not None:
        chat.system_prompt = updates.system_prompt
    
    # Handle token limits
    if updates.max_input_tokens is not None:
        chat.max_input_tokens = updates.max_input_tokens if updates.max_input_tokens > 0 else None
    if updates.max_output_tokens is not None:
        chat.max_output_tokens = updates.max_output_tokens if updates.max_output_tokens > 0 else None
    
    chat.updated_at = datetime.now(timezone.utc)
    await db.commit()
    
    return chat


@router.patch("/{chat_id}/selected-version")
async def update_selected_version(
    chat_id: str,
    parent_id: str,  # The parent message (branch point)
    child_id: str,   # The selected child message
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update which branch is selected at a branch point in the conversation tree"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Initialize if null
    if chat.selected_versions is None:
        chat.selected_versions = {}
    
    # Update the selected version (make a copy to trigger SQLAlchemy change detection)
    new_versions = dict(chat.selected_versions)
    new_versions[parent_id] = child_id
    chat.selected_versions = new_versions
    
    # Don't update updated_at - this is just UI state
    await db.commit()
    
    return {"status": "ok", "parent_id": parent_id, "child_id": child_id}


class ShareChatRequest(BaseModel):
    """Request body for sharing a chat"""
    anonymous: bool = False  # If True, hide owner name in shared view


@router.post("/{chat_id}/share")
async def share_chat(
    chat_id: str,
    request: ShareChatRequest = ShareChatRequest(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a public share link for a chat"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Generate share_id if not exists
    if not chat.share_id:
        import uuid
        chat.share_id = str(uuid.uuid4())[:8]  # Short ID for URLs
    
    # Update anonymous setting
    chat.share_anonymous = request.anonymous
    await db.commit()
    
    return {"share_id": chat.share_id, "anonymous": chat.share_anonymous}


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat and all its messages, generated images, and knowledge index entries"""
    from app.models.models import Document
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Delete associated generated images first
    deleted_images = await delete_chat_images(db, chat_id)
    if deleted_images > 0:
        logger.debug(f"Deleted {deleted_images} generated images for chat {chat_id}")
    
    # Remove from chat knowledge index if indexed
    if chat.is_knowledge_indexed and user.chat_knowledge_store_id:
        try:
            # Find the document representing this chat in the knowledge store
            result = await db.execute(
                select(Document).where(
                    Document.knowledge_store_id == user.chat_knowledge_store_id,
                    Document.file_path == f"chat://{chat_id}"
                )
            )
            doc = result.scalar_one_or_none()
            
            if doc:
                rag_service = RAGService()
                await rag_service.delete_document(db, doc.id)
                logger.info(f"Removed chat {chat_id} from knowledge index")
        except Exception as e:
            logger.warning(f"Failed to remove chat from knowledge index: {e}")
            # Don't fail the deletion if index removal fails
    
    await db.delete(chat)
    await db.commit()
    
    return {"status": "deleted", "chat_id": chat_id, "images_deleted": deleted_images}


@router.delete("")
async def delete_all_chats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all user's chats, their generated images, and chat knowledge index"""
    from app.models.models import Document, DocumentChunk, KnowledgeStore
    
    # Delete all generated images first
    deleted_images = await delete_user_images(db, user.id)
    if deleted_images > 0:
        logger.debug(f"Deleted {deleted_images} generated images for user {user.id}")
    
    # Clear the entire chat knowledge store if exists
    if user.chat_knowledge_store_id:
        try:
            # Delete all documents and chunks in the chat knowledge store
            result = await db.execute(
                select(Document.id).where(
                    Document.knowledge_store_id == user.chat_knowledge_store_id
                )
            )
            doc_ids = [row[0] for row in result.all()]
            
            if doc_ids:
                # Delete chunks for all documents
                await db.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids))
                )
                # Delete documents
                await db.execute(
                    delete(Document).where(Document.knowledge_store_id == user.chat_knowledge_store_id)
                )
            
            # Delete the knowledge store itself
            await db.execute(
                delete(KnowledgeStore).where(KnowledgeStore.id == user.chat_knowledge_store_id)
            )
            
            # Clear user's chat knowledge settings
            user.chat_knowledge_store_id = None
            user.all_chats_knowledge_enabled = False
            user.chat_knowledge_status = "idle"
            
            logger.info(f"Deleted chat knowledge store for user {user.id}")
        except Exception as e:
            logger.warning(f"Failed to delete chat knowledge store: {e}")
            # Don't fail the deletion if knowledge store removal fails
    
    await db.execute(delete(Chat).where(Chat.owner_id == user.id))
    await db.commit()
    
    return {"status": "deleted", "message": "All chats deleted", "images_deleted": deleted_images}


@router.get("/group/{group_type}/{group_name}")
async def get_chats_by_group(
    group_type: str,  # "source" or "period"
    group_name: str,  # e.g., "chatgpt", "native" or "Today", "This Week"
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    date_field: str = Query("updated_at", description="For period type: 'updated_at' (modified) or 'created_at'"),
    search: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get chats for a specific group (source or time period) with pagination."""
    base_filter = Chat.owner_id == user.id
    
    if group_type == "source":
        # Source-based filtering
        source_map = {"Native": "native", "ChatGPT": "chatgpt", "Grok": "grok", "Claude": "claude"}
        source_val = source_map.get(group_name, group_name.lower())
        
        if source_val == "native":
            group_filter = or_(Chat.source == 'native', Chat.source.is_(None))
        else:
            group_filter = Chat.source == source_val
        
        query = (
            select(Chat)
            .where(base_filter, group_filter)
            .order_by(Chat.updated_at.desc())
        )
        count_query = select(func.count(Chat.id)).where(base_filter, group_filter)
        
    elif group_type == "period":
        # Time period filtering - use specified date field
        db_date_field = Chat.created_at if date_field == "created_at" else Chat.updated_at
        
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seven_days_ago = today_start - timedelta(days=7)
        thirty_days_ago = today_start - timedelta(days=30)
        
        if group_name == "Today":
            group_filter = db_date_field >= today_start
        elif group_name == "Last 7 Days":
            group_filter = and_(db_date_field >= seven_days_ago, db_date_field < today_start)
        elif group_name == "Last 30 Days":
            group_filter = and_(db_date_field >= thirty_days_ago, db_date_field < seven_days_ago)
        else:  # Older
            group_filter = db_date_field < thirty_days_ago
        
        query = (
            select(Chat)
            .where(base_filter, group_filter)
            .order_by(db_date_field.desc())
        )
        count_query = select(func.count(Chat.id)).where(base_filter, group_filter)
    else:
        raise HTTPException(status_code=400, detail="Invalid group_type. Use 'source' or 'period'")
    
    # NC-0.8.0.27: Apply search filter if provided
    if search:
        search_title_filter = Chat.title.ilike(f"%{search}%")
        # Also search message content
        message_search = (
            select(Message.chat_id)
            .join(Chat, Message.chat_id == Chat.id)
            .where(Chat.owner_id == user.id)
            .where(Message.content.ilike(f"%{search}%"))
            .distinct()
        )
        msg_result = await db.execute(message_search)
        matching_ids = [row[0] for row in msg_result.all()]
        combined_filter = or_(
            search_title_filter,
            Chat.id.in_(matching_ids) if matching_ids else False
        )
        query = query.where(combined_filter)
        count_query = count_query.where(combined_filter)
    
    # Get total count for this group
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    chats = result.scalars().all()
    
    return {
        "chats": [ChatResponse.model_validate(c) for c in chats],
        "total": total,
        "page": page,
        "page_size": page_size,
        "group_name": group_name,
    }


@router.delete("/group/{group_type}/{group_name}")
async def delete_chats_by_group(
    group_type: str,  # "source" or "period"
    group_name: str,  # e.g., "chatgpt", "native" or "Today", "This Week"
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all chats in a specific group (source or time period)."""
    base_filter = Chat.owner_id == user.id
    
    if group_type == "source":
        # Source-based filtering
        source_map = {"Native": "native", "ChatGPT": "chatgpt", "Grok": "grok", "Claude": "claude"}
        source_val = source_map.get(group_name, group_name.lower())
        
        if source_val == "native":
            group_filter = or_(Chat.source == 'native', Chat.source.is_(None))
        else:
            group_filter = Chat.source == source_val
            
    elif group_type == "period":
        # Time period filtering
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seven_days_ago = today_start - timedelta(days=7)
        thirty_days_ago = today_start - timedelta(days=30)
        
        if group_name == "Today":
            group_filter = Chat.updated_at >= today_start
        elif group_name == "Last 7 Days":
            group_filter = and_(Chat.updated_at >= seven_days_ago, Chat.updated_at < today_start)
        elif group_name == "Last 30 Days":
            group_filter = and_(Chat.updated_at >= thirty_days_ago, Chat.updated_at < seven_days_ago)
        else:  # Older
            group_filter = Chat.updated_at < thirty_days_ago
    else:
        raise HTTPException(status_code=400, detail="Invalid group_type. Use 'source' or 'period'")
    
    # Get chat IDs to delete (for image cleanup)
    result = await db.execute(
        select(Chat.id).where(base_filter, group_filter)
    )
    chat_ids = [row[0] for row in result.all()]
    
    if not chat_ids:
        return {"status": "deleted", "count": 0, "group": group_name}
    
    # Delete generated images for these chats
    deleted_images = 0
    for chat_id in chat_ids:
        try:
            count = await delete_chat_images(db, chat_id)
            deleted_images += count
        except Exception as e:
            logger.warning(f"Failed to delete images for chat {chat_id}: {e}")
    
    # Delete the chats
    await db.execute(delete(Chat).where(Chat.id.in_(chat_ids)))
    await db.commit()
    
    logger.info(f"User {user.id} deleted {len(chat_ids)} chats from group {group_type}/{group_name}")
    
    return {
        "status": "deleted",
        "count": len(chat_ids),
        "group": group_name,
        "images_deleted": deleted_images,
    }


# ============ Messages ============

@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
async def list_messages(
    chat_id: str,
    limit: int = Query(100, ge=1, le=500),
    before: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a chat"""
    
    await _get_user_chat(db, user, chat_id)
    
    query = select(Message).where(
        Message.chat_id == chat_id,
    )
    
    if before:
        query = query.where(Message.id < before)
    
    query = query.order_by(Message.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[LIST_MESSAGES] chat_id={chat_id}, count={len(messages)}")
    for msg in messages:
        content_preview = (msg.content[:50] + '...') if msg.content and len(msg.content) > 50 else msg.content
        logger.info(f"[LIST_MESSAGES]   msg_id={msg.id[:8]}..., role={msg.role}, parent={msg.parent_id[:8] if msg.parent_id else 'root'}, content={content_preview}")
    
    # Return in chronological order
    return [MessageResponse.model_validate(m) for m in reversed(messages)]


@router.post("/{chat_id}/messages", response_model=MessageResponse)
async def send_message(
    chat_id: str,
    message_data: MessageCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a message and get AI response (non-streaming)"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Check usage limits
    billing = BillingService()
    limit_check = await billing.check_usage_limit(db, user, billing.estimate_tokens(message_data.content))
    
    if not limit_check["can_proceed"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Token limit exceeded. Please upgrade your plan.",
        )
    
    # Process attachments
    attachments = []
    if message_data.attachments:
        for att in message_data.attachments:
            attachments.append({
                "type": att.type,
                "data": att.data,
                "url": att.url,
                "name": att.name,
                "mime_type": att.mime_type,
            })
    
    # Create user message
    user_message = Message(
        chat_id=chat_id,
        sender_id=user.id,
        role=MessageRole.USER,
        content=message_data.content,
        content_type=ContentType.TEXT,
        attachments=attachments if attachments else None,
    )
    db.add(user_message)
    await db.flush()
    
    # Get RAG context - follows same logic as websocket.py:
    # 1. Global knowledge stores are always searched
    # 2. Assistant (Custom GPT) knowledge stores are searched when using that assistant
    # 3. User's unitemized documents are searched only if enable_rag is true
    from app.api.routes.admin import get_system_setting
    from app.models.models import CustomAssistant
    from app.models.assistant import AssistantConversation
    from sqlalchemy.orm import selectinload
    
    system_prompt = chat.system_prompt or await get_system_setting(db, "default_system_prompt")
    rag_service = RAGService()
    
    # 1. Search global knowledge stores (always, independent of enable_rag)
    try:
        global_results, global_store_names = await rag_service.search_global_stores(db, message_data.content, chat_id=chat_id)
        
        if global_results:
            global_context_parts = []
            for result in global_results:
                doc_name = result.get("document_name", "Unknown")
                chunk_content = result.get("content", "")
                score = result.get("similarity", 0)
                global_context_parts.append(
                    f"[Source: {doc_name} | Confidence: {score:.0%}]\n{chunk_content}"
                )
            
            global_context = "\n\n---\n\n".join(global_context_parts)
            global_stores_list = ", ".join(global_store_names)
            
            knowledge_addendum = f"""

## AUTHORITATIVE KNOWLEDGE BASE

<trusted_knowledge source="{global_stores_list}">
IMPORTANT: The following information comes from the organization's verified global knowledge base. This content is DEFINITIVE and TRUSTED - treat it as the authoritative source of truth for the topics it covers. When this knowledge conflicts with your general training, defer to this information.

{global_context}
</trusted_knowledge>

When answering questions related to the above topics, you MUST use this authoritative information as your primary source. Cite it naturally in your responses when relevant."""

            system_prompt = f"{system_prompt}{knowledge_addendum}"
            logger.info(f"[GLOBAL_RAG] Injected {len(global_results)} results from: {global_store_names}")
    except Exception as e:
        logger.warning(f"[GLOBAL_RAG] Failed to search global stores: {e}")
    
    # 2. Check if chat is associated with an assistant (Custom GPT)
    assistant_result = await db.execute(
        select(AssistantConversation).where(AssistantConversation.chat_id == chat_id)
    )
    assistant_conv = assistant_result.scalar_one_or_none()
    enable_rag = message_data.enable_rag
    
    if assistant_conv:
        # Get the assistant's knowledge stores
        assistant_result = await db.execute(
            select(CustomAssistant)
            .where(CustomAssistant.id == assistant_conv.assistant_id)
            .options(selectinload(CustomAssistant.knowledge_stores))
        )
        assistant = assistant_result.scalar_one_or_none()
        
        if assistant and assistant.knowledge_stores:
            assistant_ks_ids = [str(ks.id) for ks in assistant.knowledge_stores]
            # Search assistant's knowledge stores
            context = await rag_service.get_knowledge_store_context(
                db=db,
                user=user,
                query=message_data.content,
                knowledge_store_ids=assistant_ks_ids,
                bypass_access_check=True,  # Allow access through assistant
                chat_id=chat_id,  # For context-aware query enhancement
            )
            if context:
                rag_prompt = await get_system_setting(db, "rag_context_prompt")
                system_prompt = f"{system_prompt}\n\n{rag_prompt}\n{context}"
            logger.debug(f"[ASSISTANT_RAG] Searched assistant KBs: {assistant_ks_ids}")
    elif enable_rag:
        # 3. Search user's unitemized documents (only if no assistant and RAG enabled)
        context = await rag_service.get_context_for_query(
            db=db,
            user=user,
            query=message_data.content,
            document_ids=message_data.document_ids,
            chat_id=chat_id,  # For context-aware query enhancement
        )
        if context:
            system_prompt = f"{system_prompt}\n\nRelevant context from documents:\n{context}"
    
    # Temporarily update system prompt with RAG context
    original_prompt = chat.system_prompt
    chat.system_prompt = system_prompt
    
    # Get tools if enabled
    tools = None
    tool_handlers = None
    
    if message_data.enable_tools:
        tools = tool_registry.get_tool_definitions()
        tool_handlers = {
            name: lambda args, n=name: tool_registry.execute(n, args, {"db": db, "user": user})
            for name in tool_registry._handlers.keys()
        }
    
    # Get AI response (using database settings)
    llm = await LLMService.from_database(db)
    assistant_message = await llm.create_message(
        db=db,
        user=user,
        chat=chat,
        user_message=message_data.content,
        attachments=attachments,
        tools=tools,
        tool_handlers=tool_handlers,
    )
    
    # Restore original prompt
    chat.system_prompt = original_prompt
    
    # Note: Title generation is handled by WebSocket route (websocket.py)
    # The WebSocket path uses LLM-based title generation which produces better results
    # than simple truncation. REST API created chats will get titles when first
    # message is sent via WebSocket.
    
    await db.commit()
    
    return MessageResponse.model_validate(assistant_message)


# ============ Shared Chat (Client-to-Client) ============

@router.post("/{chat_id}/invite")
async def invite_to_chat(
    chat_id: str,
    invite: ChatInvite,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Invite a user to a shared chat"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    if not chat.is_shared:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat is not a shared chat",
        )
    
    # Check if user is owner
    if chat.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only chat owner can invite users",
        )
    
    # Check if invited user exists
    from app.services.auth import AuthService
    invited_user = await AuthService.get_user_by_id(db, invite.user_id)
    if not invited_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check if already a participant
    result = await db.execute(
        select(ChatParticipant).where(
            ChatParticipant.chat_id == chat_id,
            ChatParticipant.user_id == invite.user_id,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a participant",
        )
    
    # Add participant
    participant = ChatParticipant(
        chat_id=chat_id,
        user_id=invite.user_id,
        role=invite.role,
    )
    db.add(participant)
    await db.commit()
    
    return {"status": "invited", "user_id": invite.user_id}


@router.patch("/{chat_id}/messages/{message_id}", response_model=MessageResponse)
async def edit_message(
    chat_id: str,
    message_id: str,
    edit_data: MessageEdit,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Edit a message - creates a new branch from the message's parent.
    The original message and its children remain intact.
    Returns the new edited message (which becomes a sibling to the original).
    
    For user messages, the frontend should follow up with a WebSocket
    regenerate_message call to generate the AI response.
    """
    chat = await _get_user_chat(db, user, chat_id)
    
    # Get the original message
    result = await db.execute(
        select(Message).where(Message.id == message_id, Message.chat_id == chat_id)
    )
    original_message = result.scalar_one_or_none()
    
    if not original_message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Create a new message as a sibling (same parent_id)
    new_message = Message(
        chat_id=chat_id,
        sender_id=user.id,
        parent_id=original_message.parent_id,  # Same parent = sibling branch
        role=original_message.role,
        content=edit_data.content,
        content_type=original_message.content_type,
        attachments=original_message.attachments,
    )
    db.add(new_message)
    await db.flush()  # Get the ID without committing
    
    # Update chat's selected_versions to point to this new branch
    selected = dict(chat.selected_versions or {})
    if original_message.parent_id:
        selected[original_message.parent_id] = new_message.id
    chat.selected_versions = selected
    chat.updated_at = datetime.now(timezone.utc)
    
    await db.commit()
    await db.refresh(new_message)
    
    return MessageResponse.model_validate(new_message)


class MessageArtifactsUpdate(BaseModel):
    """Update artifacts on a message"""
    artifacts: List[Dict]


@router.put("/{chat_id}/messages/{message_id}/artifacts", response_model=MessageResponse)
async def update_message_artifacts(
    chat_id: str,
    message_id: str,
    data: MessageArtifactsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update the artifacts array on a message.
    Used to persist extracted code artifacts with timestamps after stream_end.
    """
    chat = await _get_user_chat(db, user, chat_id)
    
    # Get the message
    result = await db.execute(
        select(Message).where(Message.id == message_id, Message.chat_id == chat_id)
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Update artifacts
    message.artifacts = data.artifacts
    await db.commit()
    await db.refresh(message)
    
    return MessageResponse.model_validate(message)


@router.delete("/{chat_id}/messages/{message_id}")
async def delete_message(
    chat_id: str,
    message_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a message and all its descendants.
    Sibling branches (other children of the same parent) are preserved.
    Updates selected_versions if the deleted message was selected.
    """
    chat = await _get_user_chat(db, user, chat_id)
    
    # Get the message to delete
    result = await db.execute(
        select(Message).where(Message.id == message_id, Message.chat_id == chat_id)
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    parent_id = message.parent_id
    
    # Collect all descendant message IDs using iterative BFS
    ids_to_delete = set([message_id])
    queue = [message_id]
    
    while queue:
        current_id = queue.pop(0)
        result = await db.execute(
            select(Message.id).where(Message.parent_id == current_id)
        )
        child_ids = [row[0] for row in result.fetchall()]
        for child_id in child_ids:
            if child_id not in ids_to_delete:
                ids_to_delete.add(child_id)
                queue.append(child_id)
    
    # Update selected_versions to remove any selections pointing to deleted messages
    selected = dict(chat.selected_versions or {})
    changed = False
    
    # Remove selections that point to deleted messages
    keys_to_remove = []
    for parent_key, child_id in list(selected.items()):
        if child_id in ids_to_delete:
            # This selection is invalid now
            # Check if there are other siblings to select
            result = await db.execute(
                select(Message).where(
                    Message.parent_id == parent_key,
                    Message.id.not_in(ids_to_delete)
                ).order_by(Message.created_at.desc())
            )
            sibling = result.scalars().first()
            
            if sibling:
                # Select the most recent remaining sibling
                selected[parent_key] = sibling.id
            else:
                # No siblings left, remove the selection
                keys_to_remove.append(parent_key)
            changed = True
    
    for key in keys_to_remove:
        del selected[key]
    
    # Also remove any selections where the parent is being deleted
    for deleted_id in ids_to_delete:
        if deleted_id in selected:
            del selected[deleted_id]
            changed = True
    
    if changed:
        chat.selected_versions = selected
    
    # Delete generated images for these messages first
    deleted_images = await delete_message_images(db, ids_to_delete)
    if deleted_images > 0:
        logger.debug(f"Deleted {deleted_images} generated images for message branch {message_id}")
    
    # Delete all messages in the branch
    await db.execute(
        delete(Message).where(Message.id.in_(ids_to_delete))
    )
    
    chat.updated_at = datetime.now(timezone.utc)
    await db.commit()
    
    return {
        "status": "deleted",
        "message_id": message_id,
        "total_deleted": len(ids_to_delete),
        "images_deleted": deleted_images,
    }


@router.post("/{chat_id}/client-message", response_model=MessageResponse)
async def send_client_message(
    chat_id: str,
    message_data: ClientMessageCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a client-to-client message (no AI response)"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    if not chat.is_shared:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat is not a shared chat",
        )
    
    # Process attachments
    attachments = []
    if message_data.attachments:
        for att in message_data.attachments:
            attachments.append({
                "type": att.type,
                "data": att.data,
                "url": att.url,
                "name": att.name,
                "mime_type": att.mime_type,
            })
    
    # Create message
    message = Message(
        chat_id=chat_id,
        sender_id=user.id,
        role=MessageRole.USER,
        content=message_data.content,
        content_type=ContentType.TEXT,
        attachments=attachments if attachments else None,
    )
    db.add(message)
    
    chat.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(message)
    
    # TODO: Broadcast via WebSocket to other participants
    
    return MessageResponse.model_validate(message)


# ============ Helpers ============

async def _get_user_chat(db: AsyncSession, user: User, chat_id: str) -> Chat:
    """Get a chat ensuring user has access"""
    
    # Check if owner
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    )
    chat = result.scalar_one_or_none()
    
    if chat:
        return chat
    
    # Check if participant in shared chat
    result = await db.execute(
        select(Chat)
        .join(ChatParticipant, Chat.id == ChatParticipant.chat_id)
        .where(Chat.id == chat_id, ChatParticipant.user_id == user.id)
    )
    chat = result.scalar_one_or_none()
    
    if chat:
        return chat
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chat not found",
    )


# ============ Zip Upload ============

from fastapi import UploadFile, File
from app.services.zip_processor import ZipProcessor, ZipSecurityError, format_signature_summary, format_llm_manifest, format_file_content_for_llm
from app.services.validators import (
    validate_file_extension, validate_file_size, is_dangerous_file,
    ALLOWED_ARCHIVE_EXTENSIONS, MAX_ZIP_SIZE, FileValidationError
)
from app.models.models import UploadedFile, UploadedArchive
import logging
from datetime import datetime, timezone
import json
import os

zip_logger = logging.getLogger(__name__)


async def get_zip_manifest_from_db(db: AsyncSession, chat_id: str) -> str | None:
    """Get the LLM manifest for a chat's uploaded zip from database"""
    result = await db.execute(
        select(UploadedArchive)
        .where(UploadedArchive.chat_id == chat_id)
        .order_by(UploadedArchive.created_at.desc())
        .limit(1)
    )
    archive = result.scalar_one_or_none()
    return archive.llm_manifest if archive else None


async def get_uploaded_files_for_chat(db: AsyncSession, chat_id: str, include_agent_files: bool = False) -> list[dict]:
    """
    Get all uploaded files for a chat as artifact-format dicts.
    
    Args:
        include_agent_files: If False (default), excludes {AgentNNNN}.md files
    """
    from app.services.agent_memory import is_agent_file
    
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.chat_id == chat_id)
        .order_by(UploadedFile.filepath)
    )
    files = result.scalars().all()
    
    artifacts = []
    for f in files:
        # Skip agent memory files unless explicitly requested
        if not include_agent_files and is_agent_file(f.filepath):
            continue
            
        artifacts.append({
            "id": f.id,
            "title": f.filename,
            "filename": f.filepath,
            "content": f.content or "",
            "language": f.language or "text",
            "type": "code",
            "source": "upload",
            "size": f.size or len(f.content or ""),  # Include file size
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "signatures": json.loads(f.signatures) if f.signatures else [],
        })
    
    return artifacts


@router.get("/{chat_id}/uploaded-files")
async def get_chat_uploaded_files(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all uploaded files for a chat.
    Used to restore artifacts after page refresh.
    """
    await _get_user_chat(db, user, chat_id)
    
    artifacts = await get_uploaded_files_for_chat(db, chat_id)
    
    # Also get archive metadata (most recent if multiple)
    result = await db.execute(
        select(UploadedArchive)
        .where(UploadedArchive.chat_id == chat_id)
        .order_by(UploadedArchive.created_at.desc())
        .limit(1)
    )
    archive = result.scalar_one_or_none()
    
    archive_info = None
    if archive:
        archive_info = {
            "filename": archive.filename,
            "total_files": archive.total_files,
            "total_size": archive.total_size,
            "languages": json.loads(archive.languages) if archive.languages else {},
            "llm_manifest": archive.llm_manifest,
            "summary": archive.summary,  # Human-readable summary with signatures
        }
    
    return {
        "artifacts": artifacts,
        "archive": archive_info,
    }


@router.post("/{chat_id}/uploaded-files")
async def save_uploaded_file(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    filename: str = Body(...),
    content: str = Body(...),
    language: str = Body(None),
    signatures: list = Body(default=[]),
):
    """
    Save an individual uploaded file (not from zip) to the database.
    Used to persist file uploads across page refreshes.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[SAVE_UPLOAD] Saving file {filename} for chat {chat_id}, content_len={len(content)}")
    
    await _get_user_chat(db, user, chat_id)
    
    ext = os.path.splitext(filename)[1].lower() if '.' in filename else ''
    
    # Check if file already exists for this chat
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.chat_id == chat_id)
        .where(UploadedFile.filepath == filename)
        .order_by(UploadedFile.created_at.desc())
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        # Update existing file
        existing.content = content
        existing.language = language
        existing.size = len(content)
        existing.signatures = json.dumps(signatures) if signatures else None
        logger.info(f"[SAVE_UPLOAD] Updated existing file {filename}")
        
        # Clean up any duplicate files (keep only the one we're updating)
        dup_result = await db.execute(
            select(UploadedFile)
            .where(UploadedFile.chat_id == chat_id)
            .where(UploadedFile.filepath == filename)
            .where(UploadedFile.id != existing.id)
        )
        duplicates = dup_result.scalars().all()
        for dup in duplicates:
            await db.delete(dup)
            logger.info(f"[SAVE_UPLOAD] Cleaned up duplicate file {filename}")
    else:
        # Create new file
        uploaded_file = UploadedFile(
            chat_id=chat_id,
            archive_name=None,  # Not from archive
            filepath=filename,
            filename=os.path.basename(filename),
            extension=ext,
            language=language,
            size=len(content),
            is_binary=False,
            content=content,
            signatures=json.dumps(signatures) if signatures else None,
        )
        db.add(uploaded_file)
        logger.info(f"[SAVE_UPLOAD] Created new file {filename}")
    
    await db.commit()
    
    return {"success": True, "filename": filename}


@router.put("/{chat_id}/revert-file")
async def revert_file(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    filename: str = Body(...),
    content: str = Body(...),
):
    """
    NC-0.8.0.12: Revert a file to a previous version.
    Updates session storage and DB with the reverted content.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    await _get_user_chat(db, user, chat_id)
    
    # Update DB
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.chat_id == chat_id)
        .where(UploadedFile.filepath == filename)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        existing.content = content
        existing.size = len(content)
        await db.commit()
        logger.info(f"[REVERT_FILE] Updated DB: {filename}")
    
    # Update session storage
    from app.tools.registry import store_session_file
    store_session_file(chat_id, filename, content)
    logger.info(f"[REVERT_FILE] Updated session: {filename}")
    
    return {"success": True, "filename": filename}


@router.post("/{chat_id}/upload-zip")
async def upload_zip(
    chat_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a zip file and extract its contents as artifacts.
    Files are persisted in database for durability across page refreshes.
    Returns file list, signature index, artifacts, and LLM context manifest.
    """
    
    # Verify chat ownership
    await _get_user_chat(db, user, chat_id)
    
    # Validate file using shared validators
    try:
        validate_file_extension(file.filename or "", ALLOWED_ARCHIVE_EXTENSIONS, "archive")
    except FileValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    
    # Check for dangerous file patterns
    if is_dangerous_file(file.filename or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type not allowed for security reasons"
        )
    
    # Read file content
    zip_data = await file.read()
    
    # Validate file size using shared validators
    try:
        validate_file_size(len(zip_data), MAX_ZIP_SIZE, "archive")
    except FileValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    
    try:
        # Process the zip file (includes security validation)
        processor = ZipProcessor()
        manifest = processor.process(zip_data)
        
        # Convert to artifacts format
        artifacts = processor.to_artifacts(manifest)
        
        # Generate summaries
        summary = format_signature_summary(manifest)
        upload_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        llm_manifest = format_llm_manifest(manifest, file.filename, upload_timestamp)
        
        # Delete any existing files for this chat (replace previous upload)
        await db.execute(
            UploadedFile.__table__.delete().where(UploadedFile.chat_id == chat_id)
        )
        await db.execute(
            UploadedArchive.__table__.delete().where(UploadedArchive.chat_id == chat_id)
        )
        
        # Store archive metadata
        archive = UploadedArchive(
            chat_id=chat_id,
            filename=file.filename,
            total_files=manifest.total_files,
            total_size=manifest.total_size,
            languages=json.dumps(manifest.languages),
            llm_manifest=llm_manifest,
            summary=summary,  # Human-readable summary with signatures
        )
        db.add(archive)
        
        # Store each file
        stored_count = 0
        for f in manifest.files:
            if f.content is not None:  # Skip binary files
                ext = os.path.splitext(f.path)[1].lower() if '.' in f.path else ''
                uploaded_file = UploadedFile(
                    chat_id=chat_id,
                    archive_name=file.filename,
                    filepath=f.path,
                    filename=os.path.basename(f.path),
                    extension=ext,
                    language=f.language,
                    size=f.size,
                    is_binary=False,
                    content=f.content,
                    signatures=json.dumps(manifest.signature_index.get(f.path, [])),
                )
                db.add(uploaded_file)
                stored_count += 1
        
        await db.commit()
        
        zip_logger.debug(f"Processed zip '{file.filename}': {manifest.total_files} files, {manifest.total_size} bytes, {stored_count} stored in DB")
        
        # NC-0.8.0.12: Extract associations and call graph at upload time
        from app.services.zip_processor import extract_associations, extract_call_graph
        associations = extract_associations(manifest)
        call_graph = extract_call_graph(manifest)
        
        return {
            "filename": file.filename,
            "total_files": manifest.total_files,
            "total_size": manifest.total_size,
            "languages": manifest.languages,
            "file_tree": manifest.file_tree,
            "signature_index": manifest.signature_index,
            "associations": associations,
            "call_graph": call_graph,
            "artifacts": artifacts,
            "summary": summary,
            "llm_manifest": llm_manifest,
        }
    
    except ZipSecurityError as e:
        zip_logger.warning(f"Security violation in zip file: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security check failed: {str(e)}",
        )
        
    except Exception as e:
        zip_logger.error(f"Error processing zip file: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process zip file: {str(e)}",
        )


# NC-0.8.0.27: Artifact file management endpoints

@router.delete("/{chat_id}/artifact-file")
async def delete_artifact_file(
    chat_id: str,
    path: str = Query(..., description="Relative path within the sandbox"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a file or empty folder from the chat's artifact sandbox."""
    import shutil
    await _get_user_chat(db, user, chat_id)
    
    from app.tools.registry import get_session_sandbox
    sandbox = get_session_sandbox(chat_id)
    full_path = os.path.realpath(os.path.join(sandbox, path.lstrip("/")))
    
    # Security: must be within sandbox
    if not full_path.startswith(os.path.realpath(sandbox) + os.sep):
        raise HTTPException(status_code=400, detail="Path escapes sandbox")
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
    else:
        os.remove(full_path)
    
    return {"ok": True, "deleted": path}


@router.post("/{chat_id}/artifact-copy")
async def copy_artifact_file(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    body: dict = None,
):
    """Copy a file within the chat's artifact sandbox."""
    import shutil
    await _get_user_chat(db, user, chat_id)
    
    if not body:
        raise HTTPException(status_code=400, detail="Body required with src and dst")
    
    src = body.get("src", "").lstrip("/")
    dst = body.get("dst", "").lstrip("/")
    if not src or not dst:
        raise HTTPException(status_code=400, detail="src and dst required")
    
    from app.tools.registry import get_session_sandbox
    sandbox = get_session_sandbox(chat_id)
    real_sandbox = os.path.realpath(sandbox)
    
    full_src = os.path.realpath(os.path.join(sandbox, src))
    full_dst = os.path.realpath(os.path.join(sandbox, dst))
    
    if not full_src.startswith(real_sandbox + os.sep) or not full_dst.startswith(real_sandbox + os.sep):
        raise HTTPException(status_code=400, detail="Path escapes sandbox")
    
    if not os.path.exists(full_src):
        raise HTTPException(status_code=404, detail="Source file not found")
    
    os.makedirs(os.path.dirname(full_dst), exist_ok=True)
    
    if os.path.isdir(full_src):
        shutil.copytree(full_src, full_dst)
    else:
        shutil.copy2(full_src, full_dst)
    
    return {"ok": True, "src": src, "dst": dst}


@router.post("/{chat_id}/artifact-mkdir")
async def create_artifact_folder(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    body: dict = None,
):
    """Create a new folder in the chat's artifact sandbox."""
    await _get_user_chat(db, user, chat_id)
    
    if not body or not body.get("path"):
        raise HTTPException(status_code=400, detail="path required")
    
    folder_path = body["path"].lstrip("/")
    
    from app.tools.registry import get_session_sandbox
    sandbox = get_session_sandbox(chat_id)
    full_path = os.path.realpath(os.path.join(sandbox, folder_path))
    
    if not full_path.startswith(os.path.realpath(sandbox) + os.sep):
        raise HTTPException(status_code=400, detail="Path escapes sandbox")
    
    os.makedirs(full_path, exist_ok=True)
    
    return {"ok": True, "path": folder_path}


@router.get("/{chat_id}/zip-file")
async def get_zip_file(
    chat_id: str,
    path: str,
    offset: int = 0,  # NC-0.8.0.6: Support chunked retrieval
    length: int = 20000,  # Default 20KB chunks
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve a specific file's content from the uploaded zip.
    Used for lazy-loading file contents when LLM requests them.
    
    NC-0.8.0.6: Now supports offset and length for chunked retrieval.
    """
    # Verify chat ownership
    await _get_user_chat(db, user, chat_id)
    
    # Try exact path match first
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.chat_id == chat_id,
            UploadedFile.filepath == path
        )
        .order_by(UploadedFile.created_at.desc())
        .limit(1)
    )
    uploaded_file = result.scalar_one_or_none()
    
    # Try partial path match if exact not found
    if not uploaded_file:
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.chat_id == chat_id)
        )
        all_files = result.scalars().all()
        for f in all_files:
            if f.filepath.endswith(path) or path.endswith(f.filepath):
                uploaded_file = f
                break
    
    if not uploaded_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {path}",
        )
    
    full_content = uploaded_file.content or ""
    total_size = len(full_content)
    
    # NC-0.8.0.6: Apply offset and length for chunked retrieval
    if offset >= total_size:
        return {
            "path": uploaded_file.filepath,
            "content": "",
            "formatted": f"[ERROR: Offset {offset} exceeds file size {total_size}]",
            "offset": offset,
            "total_size": total_size,
            "more_available": False,
        }
    
    # Extract chunk
    end_offset = min(offset + length, total_size)
    chunk = full_content[offset:end_offset]
    
    # Try to break at line boundary if not at end
    if end_offset < total_size:
        last_newline = chunk.rfind('\n')
        if last_newline > length // 2:
            chunk = chunk[:last_newline + 1]
            end_offset = offset + len(chunk)
    
    # Format for LLM context with offset info
    formatted = format_file_content_for_llm(uploaded_file.filepath, chunk, "user_upload")
    
    # Add continuation hint if more content
    if end_offset < total_size:
        remaining = total_size - end_offset
        formatted += f"\n\n[... {remaining:,} more chars available ...]\n"
        formatted += f"[Use <request_file path=\"{uploaded_file.filepath}\" offset=\"{end_offset}\"/> to continue]"
    
    return {
        "path": uploaded_file.filepath,
        "content": chunk,
        "formatted": formatted,
        "offset": offset,
        "end_offset": end_offset,
        "total_size": total_size,
        "more_available": end_offset < total_size,
    }


# ============ Code Summary Endpoints ============

@router.get("/{chat_id}/summary", response_model=CodeSummaryResponse)
async def get_code_summary(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get code summary for a chat"""
    query = select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    result = await db.execute(query)
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if not chat.code_summary:
        raise HTTPException(status_code=404, detail="No code summary found for this chat")
    
    summary = chat.code_summary
    return CodeSummaryResponse(
        id=summary.get("id", f"summary_{chat_id}"),
        chat_id=chat_id,
        files=[FileChange(**f) for f in summary.get("files", [])],
        warnings=[SignatureWarning(**w) for w in summary.get("warnings", [])],
        last_updated=summary.get("last_updated", datetime.now(timezone.utc).isoformat()),
        auto_generated=summary.get("auto_generated", True),
    )


@router.put("/{chat_id}/summary", response_model=CodeSummaryResponse)
async def update_code_summary(
    chat_id: str,
    summary_data: CodeSummaryCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update code summary for a chat"""
    query = select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    result = await db.execute(query)
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Build summary dict
    summary_dict = {
        "id": f"summary_{chat_id}",
        "chat_id": chat_id,
        "files": [f.model_dump() for f in summary_data.files],
        "warnings": [w.model_dump() for w in summary_data.warnings],
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "auto_generated": summary_data.auto_generated,
    }
    
    chat.code_summary = summary_dict
    await db.commit()
    
    return CodeSummaryResponse(
        id=summary_dict["id"],
        chat_id=chat_id,
        files=summary_data.files,
        warnings=summary_data.warnings,
        last_updated=summary_dict["last_updated"],
        auto_generated=summary_data.auto_generated,
    )


@router.delete("/{chat_id}/summary")
async def delete_code_summary(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete code summary for a chat"""
    query = select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    result = await db.execute(query)
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat.code_summary = None
    await db.commit()
    
    return {"message": "Code summary deleted"}


# =============================================================================
# CHAT IMPORT
# =============================================================================

class ImportedMessage(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] = None

class ImportedChat(BaseModel):
    title: str
    messages: List[ImportedMessage]
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    created_at: Optional[datetime] = None

class ImportResult(BaseModel):
    success: bool
    chat_id: Optional[str] = None
    title: str
    message_count: int
    source: str
    error: Optional[str] = None

class ImportResponse(BaseModel):
    total_imported: int
    total_failed: int
    results: List[ImportResult]


def _normalize_timestamp(ts) -> datetime | None:
    """Normalize timestamp to datetime, handling both seconds and milliseconds."""
    if ts is None:
        return None
    try:
        # Convert to float if string
        if isinstance(ts, str):
            ts = float(ts)
        
        # If timestamp is too large, it's likely in milliseconds
        # Unix timestamp for year 3000 is about 32503680000 (11 digits)
        # Milliseconds would be 13+ digits
        if ts > 32503680000:  # Likely milliseconds
            ts = ts / 1000
        
        # Sanity check: should be between 1970 and 2100
        if ts < 0 or ts > 4102444800:  # 4102444800 = Jan 1, 2100
            return None
            
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def parse_chatgpt_export(data: dict) -> List[ImportedChat]:
    """Parse ChatGPT export format (conversations.json)"""
    chats = []
    
    # ChatGPT exports as a list of conversations
    conversations = data if isinstance(data, list) else data.get("conversations", [data])
    
    for conv in conversations:
        title = conv.get("title", "Imported Chat")
        messages = []
        created_at = None
        
        # ChatGPT format has a mapping of message IDs to message objects
        mapping = conv.get("mapping", {})
        
        # Use iterative approach instead of recursion to handle deep trees
        # Find root nodes (nodes with no parent)
        root_id = None
        for node_id, node in mapping.items():
            if node.get("parent") is None:
                root_id = node_id
                break
        
        if root_id:
            # Iterative tree traversal using a stack
            stack = [root_id]
            visited = set()
            
            while stack:
                node_id = stack.pop()
                
                if node_id in visited or node_id not in mapping:
                    continue
                visited.add(node_id)
                
                node = mapping[node_id]
                message = node.get("message")
                
                if message:
                    author_role = message.get("author", {}).get("role", "")
                    content_parts = message.get("content", {}).get("parts", [])
                    
                    if author_role in ["user", "assistant"] and content_parts:
                        content = "\n".join(str(p) for p in content_parts if isinstance(p, str))
                        if content.strip():
                            create_time = message.get("create_time")
                            msg_time = _normalize_timestamp(create_time)
                            
                            messages.append(ImportedMessage(
                                role=author_role,
                                content=content,
                                created_at=msg_time
                            ))
                
                # Add children to stack (in reverse order to maintain order)
                children = node.get("children", [])
                stack.extend(reversed(children))
        
        # Sort messages by created_at to ensure correct order
        messages.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc))
        
        # Get conversation create time
        create_time = conv.get("create_time")
        created_at = _normalize_timestamp(create_time)
        
        if messages:
            chats.append(ImportedChat(
                title=title,
                messages=messages,
                created_at=created_at
            ))
    
    return chats


def parse_grok_export(data: dict) -> List[ImportedChat]:
    """Parse Grok/X.AI export format
    
    Grok export structure:
    {
      "conversations": [
        {
          "conversation": { "title": "...", "create_time": "...", ... },
          "responses": [
            {
              "response": {
                "message": "content",
                "sender": "human" | "assistant",
                "create_time": { "$date": { "$numberLong": "1764622426262" } },
                "model": "grok-3",
                ...
              },
              "share_link": null
            }
          ]
        }
      ]
    }
    """
    chats = []
    
    # Get conversations array
    conversations = data.get("conversations", [])
    if not conversations and isinstance(data, list):
        conversations = data
    
    for conv_wrapper in conversations:
        # Handle the nested structure: { "conversation": {...}, "responses": [...] }
        conv_metadata = conv_wrapper.get("conversation", conv_wrapper)
        responses = conv_wrapper.get("responses", [])
        
        # If no responses array, try to get messages from the conversation directly
        if not responses:
            responses = conv_metadata.get("messages", conv_metadata.get("responses", []))
        
        title = conv_metadata.get("title", conv_metadata.get("name", "Imported from Grok"))
        messages = []
        
        # Get conversation-level timestamp
        conv_time = None
        conv_create_time = conv_metadata.get("create_time")
        if conv_create_time:
            try:
                if isinstance(conv_create_time, str):
                    conv_time = datetime.fromisoformat(conv_create_time.replace("Z", "+00:00"))
            except:
                pass
        
        for resp_wrapper in responses:
            # Handle nested response: { "response": {...}, "share_link": null }
            resp = resp_wrapper.get("response", resp_wrapper)
            
            # Get sender/role
            role = resp.get("sender", resp.get("role", "")).lower()
            
            # Get message content
            content = resp.get("message", resp.get("content", resp.get("text", "")))
            
            # Map roles to standard roles
            if role in ["human", "user"]:
                role = "user"
            elif role in ["grok", "assistant", "ai", "bot"]:
                role = "assistant"
            else:
                continue
            
            if content and content.strip():
                # Parse timestamp - Grok uses MongoDB-style dates
                msg_time = None
                create_time = resp.get("create_time")
                
                if create_time:
                    if isinstance(create_time, dict):
                        # MongoDB format: { "$date": { "$numberLong": "1764622426262" } }
                        date_obj = create_time.get("$date", {})
                        if isinstance(date_obj, dict):
                            timestamp_ms = date_obj.get("$numberLong")
                            if timestamp_ms:
                                try:
                                    msg_time = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
                                except:
                                    pass
                        elif isinstance(date_obj, (int, float)):
                            try:
                                msg_time = datetime.fromtimestamp(date_obj / 1000, tz=timezone.utc)
                            except:
                                pass
                    elif isinstance(create_time, (int, float)):
                        # Unix timestamp (seconds or milliseconds)
                        try:
                            if create_time > 1e12:  # Milliseconds
                                msg_time = datetime.fromtimestamp(create_time / 1000, tz=timezone.utc)
                            else:  # Seconds
                                msg_time = datetime.fromtimestamp(create_time, tz=timezone.utc)
                        except:
                            pass
                    elif isinstance(create_time, str):
                        try:
                            msg_time = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                        except:
                            pass
                
                messages.append(ImportedMessage(
                    role=role,
                    content=content,
                    created_at=msg_time
                ))
        
        if messages:
            chats.append(ImportedChat(
                title=title,
                messages=messages,
                created_at=conv_time
            ))
    
    return chats


def parse_claude_export(data: dict) -> List[ImportedChat]:
    """Parse Claude/Anthropic export format
    
    Claude export structure:
    [
      {
        "uuid": "...",
        "name": "Conversation Title",
        "created_at": "2025-11-22T08:16:20.193491Z",
        "updated_at": "2025-11-22T08:29:57.964146Z",
        "chat_messages": [
          {
            "uuid": "...",
            "text": "Quick access text",
            "content": [
              {"type": "text", "text": "..."},
              {"type": "thinking", "thinking": "..."},
              {"type": "tool_use", "name": "...", "input": {...}},
              {"type": "tool_result", "name": "...", "output": "..."}
            ],
            "sender": "human" | "assistant",
            "created_at": "2025-11-22T08:16:21.114548Z"
          }
        ]
      }
    ]
    """
    chats = []
    
    # Claude exports as a list of conversations
    conversations = data if isinstance(data, list) else [data]
    
    for conv in conversations:
        # Claude uses "name" for title
        title = conv.get("name", conv.get("title", "Imported from Claude"))
        messages = []
        
        # Claude uses "chat_messages" array
        msg_list = conv.get("chat_messages", conv.get("messages", []))
        
        # Get conversation created_at
        conv_created = None
        if conv.get("created_at"):
            try:
                conv_created = datetime.fromisoformat(conv["created_at"].replace("Z", "+00:00"))
            except:
                pass
        
        for msg in msg_list:
            # Claude uses "sender" not "role"
            sender = msg.get("sender", msg.get("role", "")).lower()
            
            # Map sender to role
            if sender in ["human", "user"]:
                role = "user"
            elif sender in ["assistant", "claude"]:
                role = "assistant"
            else:
                continue
            
            # Extract text content
            # Claude provides a "text" field for convenience, or we can extract from content blocks
            content = ""
            
            # First try the convenience "text" field
            if msg.get("text"):
                content = msg["text"]
            else:
                # Extract from content blocks
                content_blocks = msg.get("content", [])
                if isinstance(content_blocks, list):
                    text_parts = []
                    for block in content_blocks:
                        if isinstance(block, dict):
                            block_type = block.get("type", "")
                            if block_type == "text":
                                text_parts.append(block.get("text", ""))
                            elif block_type == "thinking":
                                # Optionally include thinking - skip for now as it's internal
                                pass
                            elif block_type == "tool_use":
                                # Include tool usage info
                                tool_name = block.get("name", "unknown")
                                text_parts.append(f"[Used tool: {tool_name}]")
                            elif block_type == "tool_result":
                                # Include tool result summary
                                tool_name = block.get("name", "unknown")
                                output = block.get("output", block.get("content", ""))
                                if isinstance(output, str) and len(output) > 500:
                                    output = output[:500] + "..."
                                text_parts.append(f"[Tool {tool_name} result: {output}]")
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)
                elif isinstance(content_blocks, str):
                    content = content_blocks
            
            if content and content.strip():
                # Parse timestamp
                timestamp = msg.get("created_at", msg.get("timestamp"))
                msg_time = None
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            msg_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except:
                        pass
                
                messages.append(ImportedMessage(
                    role=role,
                    content=content,
                    created_at=msg_time
                ))
        
        if messages:
            chats.append(ImportedChat(
                title=title,
                messages=messages,
                created_at=conv_created
            ))
    
    return chats


def parse_generic_export(data: dict) -> List[ImportedChat]:
    """Parse generic chat export format - tries to be flexible"""
    chats = []
    
    # Try to find conversations array
    conversations = data
    if isinstance(data, dict):
        for key in ["conversations", "chats", "data", "items", "history"]:
            if key in data and isinstance(data[key], list):
                conversations = data[key]
                break
        if conversations == data:
            conversations = [data]
    
    if not isinstance(conversations, list):
        conversations = [conversations]
    
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
            
        # Find title
        title = None
        for key in ["title", "name", "subject", "topic"]:
            if key in conv:
                title = conv[key]
                break
        title = title or "Imported Chat"
        
        # Find messages
        messages = []
        msg_list = None
        for key in ["messages", "conversation", "chat", "history", "turns"]:
            if key in conv and isinstance(conv[key], list):
                msg_list = conv[key]
                break
        
        if not msg_list:
            continue
        
        for msg in msg_list:
            if not isinstance(msg, dict):
                continue
            
            # Find role
            role = None
            for key in ["role", "sender", "author", "from", "type"]:
                if key in msg:
                    role = str(msg[key]).lower()
                    break
            
            # Normalize role
            if role in ["human", "user", "you", "me"]:
                role = "user"
            elif role in ["assistant", "ai", "bot", "gpt", "claude", "grok", "model", "system_response"]:
                role = "assistant"
            else:
                continue
            
            # Find content
            content = None
            for key in ["content", "text", "message", "body", "value"]:
                if key in msg:
                    content = msg[key]
                    break
            
            if isinstance(content, list):
                content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
            
            if content and str(content).strip():
                messages.append(ImportedMessage(
                    role=role,
                    content=str(content)
                ))
        
        if messages:
            chats.append(ImportedChat(
                title=title,
                messages=messages
            ))
    
    return chats


def detect_and_parse_export(data: dict) -> tuple[List[ImportedChat], str]:
    """Detect export format and parse accordingly"""
    
    # Check for Claude format FIRST (list with chat_messages and sender fields)
    # Claude exports as an array of conversations with chat_messages
    if isinstance(data, list) and data:
        first_conv = data[0]
        if isinstance(first_conv, dict):
            # Claude has "chat_messages" with "sender" field
            if "chat_messages" in first_conv:
                return parse_claude_export(data), "Claude"
            # Also check for name + uuid pattern (Claude specific)
            if "name" in first_conv and "uuid" in first_conv and "chat_messages" in first_conv:
                return parse_claude_export(data), "Claude"
    
    # Check for ChatGPT format (has mapping with message tree structure)
    if isinstance(data, list) and data and "mapping" in data[0]:
        return parse_chatgpt_export(data), "ChatGPT"
    if isinstance(data, dict) and "mapping" in data:
        return parse_chatgpt_export(data), "ChatGPT"
    
    # Check for conversations array with mapping
    if isinstance(data, dict) and "conversations" in data:
        if data["conversations"] and isinstance(data["conversations"], list):
            first_conv = data["conversations"][0]
            if isinstance(first_conv, dict):
                if "mapping" in first_conv:
                    return parse_chatgpt_export(data), "ChatGPT"
                # Check for Grok's specific nested structure: { "conversation": {...}, "responses": [...] }
                if "conversation" in first_conv and "responses" in first_conv:
                    return parse_grok_export(data), "Grok"
    
    # Check for Grok format (has specific grok indicators or xai)
    if isinstance(data, dict):
        data_str = str(data).lower()[:5000]
        if "grok" in data_str or "xai_user_id" in data_str or '"sender": "human"' in str(data)[:5000]:
            return parse_grok_export(data), "Grok"
    
    # Check for Claude format (fallback - dict with claude/anthropic keywords)
    if isinstance(data, dict):
        data_str = str(data).lower()[:1000]
        if "claude" in data_str or "anthropic" in data_str:
            return parse_claude_export(data), "Claude"
    
    # Also check list format for Claude keywords
    if isinstance(data, list) and data:
        data_str = str(data[0]).lower()[:1000] if data else ""
        if "chat_messages" in data_str or '"sender"' in data_str:
            return parse_claude_export(data), "Claude"
    
    # Fall back to generic parser
    return parse_generic_export(data), "Generic"


@router.post("/import", response_model=ImportResponse)
async def import_chats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
):
    """
    Import chats from other AI platforms.
    
    Supported formats:
    - ChatGPT (conversations.json export)
    - Grok/X.AI exports
    - Claude/Anthropic exports
    - Generic JSON chat formats
    
    The importer will auto-detect the format and parse accordingly.
    """
    import json
    import uuid
    
    # Read file content
    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )
    
    # Parse the export
    try:
        parsed_chats, source = detect_and_parse_export(data)
    except Exception as e:
        logger.error(f"Failed to parse export: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse export: {str(e)}"
        )
    
    if not parsed_chats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid chats found in the export file"
        )
    
    # Sort chats from newest to oldest based on created_at or first message time
    def get_chat_time(chat):
        if chat.created_at:
            return chat.created_at
        # Fall back to first message time if available
        if chat.messages and chat.messages[0].created_at:
            return chat.messages[0].created_at
        return datetime.min.replace(tzinfo=timezone.utc)
    
    parsed_chats.sort(key=get_chat_time, reverse=True)
    
    results = []
    total_imported = 0
    total_failed = 0
    
    # Process in batches for better performance
    BATCH_SIZE = 50
    batch_count = 0
    
    for parsed_chat in parsed_chats:
        try:
            # Use original title - source is tracked in the source field
            title = parsed_chat.title
            
            # Determine timestamps from messages
            # created_at = FIRST message timestamp (when user started the chat)
            # updated_at = LAST message timestamp (when user last interacted)
            first_message_time = None
            last_message_time = None
            
            if parsed_chat.messages:
                for msg in parsed_chat.messages:
                    if msg.created_at:
                        if first_message_time is None or msg.created_at < first_message_time:
                            first_message_time = msg.created_at
                        if last_message_time is None or msg.created_at > last_message_time:
                            last_message_time = msg.created_at
            
            # Use message timestamps, falling back to parsed chat times, then now
            chat_created = first_message_time or parsed_chat.created_at or datetime.now(timezone.utc)
            chat_updated = last_message_time or chat_created
            
            # Create the chat
            chat_id = str(uuid.uuid4())
            chat = Chat(
                id=chat_id,
                owner_id=user.id,
                title=title[:200],  # Truncate long titles
                model=parsed_chat.model or settings.LLM_MODEL,
                system_prompt=parsed_chat.system_prompt,
                source=source.lower(),  # Store source provider (chatgpt, grok, claude, native)
                created_at=chat_created,
                updated_at=chat_updated,
            )
            db.add(chat)
            
            # Create messages
            parent_id = None
            for i, msg in enumerate(parsed_chat.messages):
                msg_id = str(uuid.uuid4())
                message = Message(
                    id=msg_id,
                    chat_id=chat_id,
                    sender_id=user.id if msg.role == "user" else None,
                    role=MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT,
                    content=msg.content,
                    content_type=ContentType.TEXT,
                    parent_id=parent_id,
                    created_at=msg.created_at or datetime.now(timezone.utc),
                )
                db.add(message)
                parent_id = msg_id
            
            results.append(ImportResult(
                success=True,
                chat_id=chat_id,
                title=parsed_chat.title,
                message_count=len(parsed_chat.messages),
                source=source
            ))
            total_imported += 1
            batch_count += 1
            
            # Commit in batches for better performance
            if batch_count >= BATCH_SIZE:
                await db.commit()
                batch_count = 0
                logger.info(f"[IMPORT] Progress: {total_imported} chats imported...")
            
        except Exception as e:
            logger.error(f"Failed to import chat '{parsed_chat.title}': {e}")
            await db.rollback()
            batch_count = 0  # Reset batch counter after rollback
            results.append(ImportResult(
                success=False,
                title=parsed_chat.title,
                message_count=len(parsed_chat.messages),
                source=source,
                error=str(e)
            ))
            total_failed += 1
    
    # Commit any remaining chats
    if batch_count > 0:
        await db.commit()
    
    logger.info(f"[IMPORT] User {user.id} imported {total_imported} chats from {source}, {total_failed} failed")
    
    return ImportResponse(
        total_imported=total_imported,
        total_failed=total_failed,
        results=results
    )


# =============================================================================
# CHAT TOOLS ENDPOINTS (NC-0.8.0.0)
# =============================================================================

class ChatToolsResponse(BaseModel):
    mode_id: Optional[str] = None
    active_tools: Optional[List[str]] = None


class UpdateToolsRequest(BaseModel):
    active_tools: List[str]


class UpdateModeRequest(BaseModel):
    mode_id: str


@router.get("/{chat_id}/tools", response_model=ChatToolsResponse)
async def get_chat_tools(
    chat_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get active tools for a chat."""
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return ChatToolsResponse(
        mode_id=chat.mode_id,
        active_tools=chat.active_tools or []
    )


@router.put("/{chat_id}/tools", response_model=ChatToolsResponse)
async def update_chat_tools(
    chat_id: str,
    request: UpdateToolsRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update active tools for a chat (user overrides)."""
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat.active_tools = request.active_tools
    await db.commit()
    
    return ChatToolsResponse(
        mode_id=chat.mode_id,
        active_tools=chat.active_tools or []
    )


@router.put("/{chat_id}/mode", response_model=ChatToolsResponse)
async def update_chat_mode(
    chat_id: str,
    request: UpdateModeRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update assistant mode for a chat."""
    from app.models import AssistantMode
    
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.owner_id == user.id)
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Verify mode exists
    mode_result = await db.execute(
        select(AssistantMode).where(AssistantMode.id == request.mode_id)
    )
    mode = mode_result.scalar_one_or_none()
    
    if not mode:
        raise HTTPException(status_code=404, detail="Mode not found")
    
    chat.mode_id = request.mode_id
    chat.active_tools = mode.active_tools  # Reset to mode defaults
    await db.commit()
    
    return ChatToolsResponse(
        mode_id=chat.mode_id,
        active_tools=chat.active_tools or []
    )
