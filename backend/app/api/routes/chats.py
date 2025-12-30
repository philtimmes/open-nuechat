"""
Chat API routes
"""
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, or_
from typing import Optional, List, Dict
from datetime import datetime, timezone
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
    page_size: int = Query(20, ge=1, le=1000),
    search: Optional[str] = None,
    date_group: Optional[str] = Query(None, regex="^(Today|This Week|Last 30 Days|Older)$"),
    sort_by: str = Query("modified", regex="^(modified|created)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's chats with pagination. Search queries both title and message content."""
    from datetime import datetime, timedelta
    
    base_filter = Chat.owner_id == user.id
    filters = [base_filter]
    
    # Add date group filter if specified
    if date_group:
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        days_ago_7 = today_start - timedelta(days=6)
        days_ago_30 = today_start - timedelta(days=29)
        
        date_field = Chat.created_at if sort_by == "created" else Chat.updated_at
        
        logger.info(f"[LIST_CHATS] date_group={date_group}, sort_by={sort_by}, today_start={today_start}")
        
        if date_group == "Today":
            filters.append(date_field >= today_start)
        elif date_group == "This Week":
            filters.append(date_field >= days_ago_7)
            filters.append(date_field < today_start)
        elif date_group == "Last 30 Days":
            filters.append(date_field >= days_ago_30)
            filters.append(date_field < days_ago_7)
        elif date_group == "Older":
            filters.append(date_field < days_ago_30)
        
        # Debug: count how many match
        debug_query = select(func.count(Chat.id)).where(*filters)
        debug_result = await db.execute(debug_query)
        debug_count = debug_result.scalar()
        logger.info(f"[LIST_CHATS] Filter matches {debug_count} chats")
    
    if search:
        # Search in both title and message content
        # Find chat IDs that have matching messages
        message_search = (
            select(Message.chat_id)
            .join(Chat, Message.chat_id == Chat.id)
            .where(Chat.owner_id == user.id)
            .where(Message.content.ilike(f"%{search}%"))
            .distinct()
        )
        message_result = await db.execute(message_search)
        matching_chat_ids = [row[0] for row in message_result.all()]
        
        # Combine: title matches OR has matching messages
        search_filter = or_(
            Chat.title.ilike(f"%{search}%"),
            Chat.id.in_(matching_chat_ids) if matching_chat_ids else False
        )
        filters.append(search_filter)
    
    query = select(Chat).where(*filters)
    count_query = select(func.count(Chat.id)).where(*filters)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get paginated results - use appropriate sort field
    order_field = Chat.created_at if sort_by == "created" else Chat.updated_at
    query = query.order_by(order_field.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    chats = result.scalars().all()
    
    return ChatListResponse(
        chats=[ChatResponse.model_validate(c) for c in chats],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/counts")
async def get_chat_counts(
    sort_by: str = Query("modified", regex="^(modified|created|source)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get chat counts grouped by date category or source for sidebar display."""
    from datetime import datetime, timedelta
    from sqlalchemy import case
    
    # Use naive UTC for SQLite compatibility
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    days_ago_7 = today_start - timedelta(days=6)  # 6 days ago (This Week = 1-6 days)
    days_ago_30 = today_start - timedelta(days=29)  # 29 days ago (Last 30 = 7-29 days)
    
    logger.debug(f"[COUNTS] now={now}, today_start={today_start}, days_ago_7={days_ago_7}")
    
    base_filter = Chat.owner_id == user.id
    
    if sort_by == "source":
        # Group by title prefix (ChatGPT:, Grok:, Claude:, or Local)
        source_case = case(
            (Chat.title.like("ChatGPT:%"), "ChatGPT"),
            (Chat.title.like("Grok:%"), "Grok"),
            (Chat.title.like("Claude:%"), "Claude"),
            else_="Local"
        )
        
        query = (
            select(source_case.label("group_name"), func.count(Chat.id).label("count"))
            .where(base_filter)
            .group_by(source_case)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        # Ensure all groups exist
        counts = {"Local": 0, "ChatGPT": 0, "Grok": 0, "Claude": 0}
        for row in rows:
            counts[row.group_name] = row.count
        
        return {
            "sort_by": sort_by,
            "groups": ["Local", "ChatGPT", "Grok", "Claude"],
            "counts": counts,
            "total": sum(counts.values())
        }
    
    else:
        # Group by date category (Today, This Week, Last 30 Days, Older)
        date_field = Chat.created_at if sort_by == "created" else Chat.updated_at
        
        date_case = case(
            (date_field >= today_start, "Today"),
            (date_field >= days_ago_7, "This Week"),
            (date_field >= days_ago_30, "Last 30 Days"),
            else_="Older"
        )
        
        query = (
            select(date_case.label("group_name"), func.count(Chat.id).label("count"))
            .where(base_filter)
            .group_by(date_case)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        # Ensure all groups exist
        counts = {"Today": 0, "This Week": 0, "Last 30 Days": 0, "Older": 0}
        for row in rows:
            counts[row.group_name] = row.count
        
        return {
            "sort_by": sort_by,
            "groups": ["Today", "This Week", "Last 30 Days", "Older"],
            "counts": counts,
            "total": sum(counts.values())
        }


@router.post("", response_model=ChatResponse)
async def create_chat(
    chat_data: ChatCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat"""
    from app.models.models import CustomAssistant
    
    model_to_use = chat_data.model or settings.LLM_MODEL
    
    # Validate custom assistant if using gpt: prefix
    if model_to_use and model_to_use.startswith("gpt:"):
        assistant_id = model_to_use[4:]
        result = await db.execute(
            select(CustomAssistant).where(CustomAssistant.assistant_id == assistant_id)
        )
        assistant = result.scalar_one_or_none()
        if not assistant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Custom assistant '{assistant_id}' not found"
            )
        # Use the assistant's actual model
        model_to_use = assistant.model
    elif model_to_use:
        # For regular models, just log a warning if validation fails - don't block chat creation
        try:
            llm_service = LLMService()
            available_models = await llm_service.list_models()
            model_ids = [m.get("id") for m in available_models if "id" in m]
            
            if model_ids and model_to_use not in model_ids:
                logger.warning(f"Model '{model_to_use}' not in available models list. Available: {model_ids[:5]}...")
        except Exception as e:
            logger.warning(f"Could not validate model: {e}")
    
    chat = Chat(
        owner_id=user.id,
        title=chat_data.title or "New Chat",
        model=model_to_use,
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
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Custom assistant '{assistant_id}' not found"
                )
        else:
            # Log warning if model doesn't exist, but allow the change
            # (The actual API will fail later if the model is truly invalid)
            try:
                llm_service = LLMService()
                available_models = await llm_service.list_models()
                model_ids = [m.get("id") for m in available_models if "id" in m]
                
                if model_ids and new_model not in model_ids:
                    logger.warning(f"Model '{new_model}' not in available models list: {model_ids[:5]}...")
            except Exception as e:
                logger.warning(f"Could not validate model: {e}")
            
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


@router.delete("/group/{date_group}")
async def delete_chats_by_group(
    date_group: str,
    sort_by: str = Query("modified", regex="^(modified|created)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all chats in a specific date group (Today, This Week, Last 30 Days, Older)"""
    from datetime import datetime, timedelta
    
    if date_group not in ['Today', 'This Week', 'Last 30 Days', 'Older']:
        raise HTTPException(status_code=400, detail="Invalid date group")
    
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    days_ago_7 = today_start - timedelta(days=6)
    days_ago_30 = today_start - timedelta(days=29)
    
    date_field = Chat.created_at if sort_by == "created" else Chat.updated_at
    
    # Build filter for the date group
    filters = [Chat.owner_id == user.id]
    
    if date_group == "Today":
        filters.append(date_field >= today_start)
    elif date_group == "This Week":
        filters.append(date_field >= days_ago_7)
        filters.append(date_field < today_start)
    elif date_group == "Last 30 Days":
        filters.append(date_field >= days_ago_30)
        filters.append(date_field < days_ago_7)
    elif date_group == "Older":
        filters.append(date_field < days_ago_30)
    
    # Get chat IDs to delete
    result = await db.execute(select(Chat.id).where(*filters))
    chat_ids = [row[0] for row in result.all()]
    
    if not chat_ids:
        return {"status": "deleted", "count": 0, "message": f"No chats in {date_group}"}
    
    # Delete associated data
    # 1. Delete messages
    await db.execute(delete(Message).where(Message.chat_id.in_(chat_ids)))
    
    # 2. Delete uploaded files
    from app.models.models import UploadedFile
    await db.execute(delete(UploadedFile).where(UploadedFile.chat_id.in_(chat_ids)))
    
    # 3. Delete chats
    await db.execute(delete(Chat).where(Chat.id.in_(chat_ids)))
    
    await db.commit()
    
    logger.info(f"Deleted {len(chat_ids)} chats in group '{date_group}' for user {user.id}")
    
    return {"status": "deleted", "count": len(chat_ids), "message": f"Deleted {len(chat_ids)} chats in {date_group}"}


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
    
    query = select(Message).where(Message.chat_id == chat_id)
    
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
        global_results, global_store_names = await rag_service.search_global_stores(db, message_data.content)
        
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
            sibling = result.scalar_one_or_none()
            
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
        
        return {
            "filename": file.filename,
            "total_files": manifest.total_files,
            "total_size": manifest.total_size,
            "languages": manifest.languages,
            "file_tree": manifest.file_tree,
            "signature_index": manifest.signature_index,
            "artifacts": artifacts,
            "summary": summary,  # Human-readable summary
            "llm_manifest": llm_manifest,  # LLM context injection format
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


@router.get("/{chat_id}/zip-file")
async def get_zip_file(
    chat_id: str,
    path: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve a specific file's content from the uploaded zip.
    Used for lazy-loading file contents when LLM requests them.
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
    
    # Format for LLM context
    formatted = format_file_content_for_llm(uploaded_file.filepath, uploaded_file.content, "user_upload")
    
    return {
        "path": uploaded_file.filepath,
        "content": uploaded_file.content,
        "formatted": formatted,  # Pre-formatted for LLM injection
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
    # Full chat object for frontend to add to store
    chat: Optional[dict] = None

class ImportResponse(BaseModel):
    total_imported: int
    total_failed: int
    results: List[ImportResult]
    # All successfully imported chats for direct store update
    imported_chats: List[dict] = []


def parse_chatgpt_export(data: dict) -> List[ImportedChat]:
    """Parse ChatGPT export format (conversations.json)
    
    ChatGPT export structure:
    {
        "title": "...",
        "create_time": 1234567890.123,
        "mapping": {
            "node-id": {
                "message": {
                    "author": {"role": "user"|"assistant"|"system"},
                    "content": {"content_type": "text", "parts": ["..."]},
                    "create_time": 1234567890.123,
                    "metadata": {
                        "is_visually_hidden_from_conversation": true,
                        "model_slug": "gpt-4"
                    }
                },
                "parent": "parent-node-id",
                "children": ["child-node-id"]
            }
        },
        "current_node": "latest-node-id"
    }
    """
    chats = []
    
    # ChatGPT exports as a list of conversations
    conversations = data if isinstance(data, list) else data.get("conversations", [data])
    
    for conv in conversations:
        title = conv.get("title", "Imported Chat")
        messages = []
        created_at = None
        model = None
        
        # ChatGPT format has a mapping of message IDs to message objects
        mapping = conv.get("mapping", {})
        
        # Find the root and traverse the tree
        def extract_messages(node_id: str):
            nonlocal model
            if not node_id or node_id not in mapping:
                return
            
            node = mapping[node_id]
            message = node.get("message")
            
            if message:
                author_role = message.get("author", {}).get("role", "")
                content_obj = message.get("content", {})
                content_parts = content_obj.get("parts", [])
                metadata = message.get("metadata", {})
                
                # Skip hidden system messages (ChatGPT internal)
                if metadata.get("is_visually_hidden_from_conversation"):
                    # Still follow children
                    for child_id in node.get("children", []):
                        extract_messages(child_id)
                    return
                
                # Skip system role messages entirely
                if author_role == "system":
                    for child_id in node.get("children", []):
                        extract_messages(child_id)
                    return
                
                # Capture model from assistant messages
                if author_role == "assistant" and not model:
                    model = metadata.get("model_slug")
                
                if author_role in ["user", "assistant"] and content_parts:
                    # Extract text content, skip non-string parts (images, etc)
                    content = "\n".join(str(p) for p in content_parts if isinstance(p, str) and p.strip())
                    
                    if content.strip():
                        create_time = message.get("create_time")
                        msg_time = None
                        if create_time:
                            try:
                                msg_time = datetime.fromtimestamp(create_time, tz=timezone.utc)
                            except:
                                pass
                        
                        messages.append(ImportedMessage(
                            role=author_role,
                            content=content,
                            created_at=msg_time
                        ))
            
            # Follow children
            for child_id in node.get("children", []):
                extract_messages(child_id)
        
        # Find root nodes (nodes with no parent or parent is null)
        for node_id, node in mapping.items():
            if node.get("parent") is None:
                extract_messages(node_id)
                break  # Only follow one tree path
        
        # Get conversation create time
        create_time = conv.get("create_time")
        if create_time:
            try:
                created_at = datetime.fromtimestamp(create_time, tz=timezone.utc)
            except:
                pass
        
        if messages:
            # Add ChatGPT: prefix to title
            prefixed_title = f"ChatGPT: {title}" if not title.startswith("ChatGPT:") else title
            chats.append(ImportedChat(
                title=prefixed_title,
                messages=messages,
                created_at=created_at,
                model=model  # Captured from assistant messages
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
            # Add Grok: prefix to title
            prefixed_title = f"Grok: {title}" if not title.startswith("Grok:") else title
            chats.append(ImportedChat(
                title=prefixed_title,
                messages=messages,
                created_at=conv_time
            ))
    
    return chats


def parse_claude_export(data: dict) -> List[ImportedChat]:
    """Parse Claude/Anthropic export format including Claude.ai exports"""
    chats = []
    
    conversations = data if isinstance(data, list) else [data]
    
    for conv in conversations:
        title = conv.get("title", conv.get("name", "Imported from Claude"))
        messages = []
        
        # Get chat creation time
        chat_created = None
        if conv.get("created_at"):
            try:
                chat_created = datetime.fromisoformat(conv["created_at"].replace("Z", "+00:00"))
            except:
                pass
        
        msg_list = conv.get("messages", conv.get("chat_messages", []))
        
        for msg in msg_list:
            role = msg.get("role", msg.get("sender", "")).lower()
            
            # Get content - handle multiple formats
            content = msg.get("content", msg.get("text", ""))
            
            # Handle Claude.ai format where content is array of content blocks
            if isinstance(content, list):
                content_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        
                        # Text content
                        if block_type == "text" or "text" in block:
                            text = block.get("text", "")
                            if text:
                                content_parts.append(text)
                        
                        # Thinking blocks - include with markers
                        elif block_type == "thinking":
                            thinking = block.get("thinking", "")
                            if thinking:
                                content_parts.append(f"<thinking>\n{thinking}\n</thinking>")
                        
                        # Tool use blocks
                        elif block_type == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            tool_msg = block.get("message", "")
                            if tool_msg:
                                content_parts.append(f"[Tool: {tool_name}] {tool_msg}")
                        
                        # Tool result blocks
                        elif block_type == "tool_result":
                            tool_name = block.get("name", "")
                            result_msg = block.get("message", "")
                            if result_msg:
                                content_parts.append(f"[Result: {tool_name}] {result_msg}")
                        
                        # Knowledge/citation blocks
                        elif block_type == "knowledge":
                            kb_title = block.get("title", "")
                            kb_text = block.get("text", "")
                            if kb_title or kb_text:
                                content_parts.append(f"[Knowledge: {kb_title}]\n{kb_text[:500]}...")
                        
                        # Fallback for unknown block types
                        else:
                            if "text" in block:
                                content_parts.append(block["text"])
                    else:
                        content_parts.append(str(block))
                
                content = "\n\n".join(content_parts)
            
            # Also check for 'text' field directly (some Claude exports)
            if not content and msg.get("text"):
                content = msg.get("text")
            
            if role in ["human", "user"]:
                role = "user"
            elif role in ["assistant", "claude"]:
                role = "assistant"
            else:
                continue
            
            if content and content.strip():
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
                title=f"Claude: {title}" if not title.startswith("Claude:") else title,
                messages=messages,
                created_at=chat_created
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
    
    # Check for Claude format - Claude.ai exports have specific structure
    if isinstance(data, list) and data:
        first_item = data[0]
        if isinstance(first_item, dict):
            # Claude.ai export has uuid, name, chat_messages, account fields
            if "chat_messages" in first_item or ("uuid" in first_item and "account" in first_item):
                return parse_claude_export(data), "Claude"
    
    if isinstance(data, dict):
        data_str = str(data).lower()[:1000]
        if "claude" in data_str or "anthropic" in data_str:
            return parse_claude_export(data), "Claude"
        # Check for single Claude.ai chat
        if "chat_messages" in data or ("uuid" in data and "account" in data):
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
    
    for parsed_chat in parsed_chats:
        try:
            # Create the chat
            chat_id = str(uuid.uuid4())
            # Use original chat date for both created_at and updated_at
            # This preserves the original timeline when viewing chat history
            original_date = parsed_chat.created_at or datetime.now(timezone.utc)
            
            # Get latest message date for updated_at (if available)
            latest_msg_date = original_date
            if parsed_chat.messages:
                for msg in parsed_chat.messages:
                    if msg.created_at and msg.created_at > latest_msg_date:
                        latest_msg_date = msg.created_at
            
            chat = Chat(
                id=chat_id,
                owner_id=user.id,
                title=parsed_chat.title[:200],  # Truncate long titles
                model=parsed_chat.model or settings.LLM_MODEL,
                system_prompt=parsed_chat.system_prompt,
                created_at=original_date,
                updated_at=latest_msg_date,  # Use latest message date or original date
            )
            db.add(chat)
            await db.flush()
            
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
            
            await db.commit()
            
            # Create chat dict for frontend
            chat_dict = {
                "id": chat_id,
                "title": parsed_chat.title[:200],
                "model": parsed_chat.model or settings.LLM_MODEL,
                "created_at": original_date.isoformat(),
                "updated_at": latest_msg_date.isoformat(),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
            
            results.append(ImportResult(
                success=True,
                chat_id=chat_id,
                title=parsed_chat.title,
                message_count=len(parsed_chat.messages),
                source=source,
                chat=chat_dict
            ))
            total_imported += 1
            
        except Exception as e:
            logger.error(f"Failed to import chat '{parsed_chat.title}': {e}")
            await db.rollback()
            results.append(ImportResult(
                success=False,
                title=parsed_chat.title,
                message_count=len(parsed_chat.messages),
                source=source,
                error=str(e)
            ))
            total_failed += 1
    
    logger.info(f"[IMPORT] User {user.id} imported {total_imported} chats from {source}, {total_failed} failed")
    
    # Collect all successfully imported chats
    imported_chats = [r.chat for r in results if r.success and r.chat]
    
    return ImportResponse(
        total_imported=total_imported,
        total_failed=total_failed,
        results=results,
        imported_chats=imported_chats
    )
