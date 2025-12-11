"""
Chat API routes
"""
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import Optional, List
from datetime import datetime, timezone

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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's chats with pagination"""
    
    query = select(Chat).where(Chat.owner_id == user.id)
    
    if search:
        query = query.where(Chat.title.ilike(f"%{search}%"))
    
    # Get total count
    count_query = select(func.count(Chat.id)).where(Chat.owner_id == user.id)
    if search:
        count_query = count_query.where(Chat.title.ilike(f"%{search}%"))
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get paginated results
    query = query.order_by(Chat.updated_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    chats = result.scalars().all()
    
    return ChatListResponse(
        chats=[ChatResponse.model_validate(c) for c in chats],
        total=total,
        page=page,
        page_size=page_size,
    )


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
            chat.model = new_model
            
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


@router.post("/{chat_id}/share")
async def share_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a public share link for a chat"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Generate share_id if not exists
    if not chat.share_id:
        import uuid
        chat.share_id = str(uuid.uuid4())[:8]  # Short ID for URLs
        await db.commit()
    
    return {"share_id": chat.share_id}


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat and all its messages and generated images"""
    
    chat = await _get_user_chat(db, user, chat_id)
    
    # Delete associated generated images first
    deleted_images = await delete_chat_images(db, chat_id)
    if deleted_images > 0:
        logger.debug(f"Deleted {deleted_images} generated images for chat {chat_id}")
    
    await db.delete(chat)
    await db.commit()
    
    return {"status": "deleted", "chat_id": chat_id, "images_deleted": deleted_images}


@router.delete("")
async def delete_all_chats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all user's chats and their generated images"""
    
    # Delete all generated images first
    deleted_images = await delete_user_images(db, user.id)
    if deleted_images > 0:
        logger.debug(f"Deleted {deleted_images} generated images for user {user.id}")
    
    await db.execute(delete(Chat).where(Chat.owner_id == user.id))
    await db.commit()
    
    return {"status": "deleted", "message": "All chats deleted", "images_deleted": deleted_images}


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
    
    # Get RAG context if enabled
    system_prompt = chat.system_prompt or "You are a helpful AI assistant."
    
    if message_data.enable_rag:
        rag_service = RAGService()
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
    
    # Auto-generate title for new chats
    if chat.title == "New Chat":
        chat.title = message_data.content[:50] + ("..." if len(message_data.content) > 50 else "")
    
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
        select(UploadedArchive).where(UploadedArchive.chat_id == chat_id)
    )
    archive = result.scalar_one_or_none()
    return archive.llm_manifest if archive else None


async def get_uploaded_files_for_chat(db: AsyncSession, chat_id: str) -> list[dict]:
    """Get all uploaded files for a chat as artifact-format dicts"""
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.chat_id == chat_id)
        .order_by(UploadedFile.filepath)
    )
    files = result.scalars().all()
    
    artifacts = []
    for f in files:
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
    
    # Also get archive metadata
    result = await db.execute(
        select(UploadedArchive).where(UploadedArchive.chat_id == chat_id)
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
        select(UploadedFile).where(
            UploadedFile.chat_id == chat_id,
            UploadedFile.filepath == path
        )
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
