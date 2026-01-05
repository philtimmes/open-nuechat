"""
Knowledge Store Management Routes

Allows users to create, manage, and share knowledge stores for RAG.
"""

from typing import List, Optional
from datetime import datetime, timezone, timedelta
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, or_, and_
import secrets
import os
from pathlib import Path

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.services.rate_limiter import rate_limiter, RateLimitExceeded
from app.services.validators import (
    validate_file_size, is_dangerous_file, MAX_DOCUMENT_SIZE, FileValidationError
)
from app.models.models import (
    User, KnowledgeStore, KnowledgeStoreShare, 
    Document, DocumentChunk, SharePermission
)
from app.services.rag import RAGService, DocumentProcessor
from app.services.document_queue import get_document_queue, DocumentTask
from app.core.config import settings

router = APIRouter()


# === Schemas ===

class KnowledgeStoreCreate(BaseModel):
    """Request to create a new knowledge store"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    icon: str = Field(default="📚", max_length=100)
    color: str = Field(default="#6366f1", max_length=20)
    is_public: bool = False
    is_discoverable: bool = False
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class KnowledgeStoreResponse(BaseModel):
    """Knowledge store info"""
    id: str
    owner_id: str
    name: str
    description: Optional[str]
    icon: str
    color: str
    is_public: bool
    is_discoverable: bool
    is_global: bool = False  # Auto-searched on every query (admin only)
    global_min_score: float = 0.7  # Minimum relevance score for global results
    global_max_results: int = 3  # Max results from global search
    document_count: int
    total_chunks: int
    total_size_bytes: int
    embedding_model: str
    created_at: datetime
    updated_at: datetime
    
    # Computed fields
    owner_username: Optional[str] = None
    permission: Optional[str] = None  # For shared stores
    
    class Config:
        from_attributes = True


class KnowledgeStoreUpdate(BaseModel):
    """Update knowledge store settings"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    icon: Optional[str] = Field(None, max_length=100)
    color: Optional[str] = Field(None, max_length=20)
    is_public: Optional[bool] = None
    is_discoverable: Optional[bool] = None
    # Global store settings (admin only - checked in endpoint)
    is_global: Optional[bool] = None
    global_min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    global_max_results: Optional[int] = Field(None, ge=1, le=20)


class ShareCreate(BaseModel):
    """Request to share a knowledge store"""
    user_email: Optional[str] = None  # Share with specific user
    create_link: bool = False  # Generate a shareable link
    permission: str = Field(default="view")  # view, edit, admin
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)
    max_uses: Optional[int] = Field(default=None, ge=1, le=1000)
    message: Optional[str] = None


class ShareResponse(BaseModel):
    """Share info"""
    id: str
    knowledge_store_id: str
    shared_with_user_id: Optional[str]
    shared_with_email: Optional[str] = None
    share_token: Optional[str]
    share_url: Optional[str] = None
    permission: str
    expires_at: Optional[datetime]
    max_uses: Optional[int]
    use_count: int
    created_at: datetime
    accepted_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class DocumentInfo(BaseModel):
    """Document info within a knowledge store"""
    id: str
    name: str
    description: Optional[str]
    file_type: str
    file_size: int
    is_processed: bool
    chunk_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# === Helper Functions ===

async def get_store_with_access(
    store_id: str,
    user: User,
    db: AsyncSession,
    required_permission: SharePermission = SharePermission.VIEW,
) -> tuple[KnowledgeStore, Optional[SharePermission]]:
    """
    Get a knowledge store and verify user has access.
    Returns (store, permission) tuple.
    """
    # Check if user owns the store
    result = await db.execute(
        select(KnowledgeStore).where(KnowledgeStore.id == store_id)
    )
    store = result.scalar_one_or_none()
    
    if not store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge store not found"
        )
    
    # Owner has full access
    if store.owner_id == user.id:
        return store, SharePermission.ADMIN
    
    # Check if store is public
    if store.is_public and required_permission == SharePermission.VIEW:
        return store, SharePermission.VIEW
    
    # Check for explicit share
    result = await db.execute(
        select(KnowledgeStoreShare).where(
            KnowledgeStoreShare.knowledge_store_id == store_id,
            KnowledgeStoreShare.shared_with_user_id == user.id
        )
    )
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this knowledge store"
        )
    
    # Check if share has expired
    if share.expires_at and share.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your access to this knowledge store has expired"
        )
    
    # Check permission level
    permission_levels = {
        SharePermission.VIEW: 0,
        SharePermission.EDIT: 1,
        SharePermission.ADMIN: 2,
    }
    
    if permission_levels[share.permission] < permission_levels[required_permission]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {required_permission.value}"
        )
    
    return store, share.permission


# === Endpoints: Knowledge Stores ===

@router.post("", response_model=KnowledgeStoreResponse)
async def create_knowledge_store(
    request_data: KnowledgeStoreCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge store"""
    # Rate limit knowledge store creation
    try:
        await rate_limiter.check_rate_limit("knowledge_store_creation", str(current_user.id))
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {e.retry_after} seconds.",
            headers={"Retry-After": str(e.retry_after)}
        )
    
    # Admins bypass store limits
    if not current_user.is_admin:
        # Get dynamic limits from system settings
        from app.models.models import SystemSetting
        
        async def get_limit_setting(key: str, default: int) -> int:
            result = await db.execute(
                select(SystemSetting).where(SystemSetting.key == key)
            )
            setting = result.scalar_one_or_none()
            return int(setting.value) if setting else default
        
        limits = {
            "free": await get_limit_setting("max_knowledge_stores_free", 3),
            "pro": await get_limit_setting("max_knowledge_stores_pro", 20),
            "enterprise": await get_limit_setting("max_knowledge_stores_enterprise", 100),
        }
        tier_limit = limits.get(current_user.tier.value, limits["free"])
        
        # Check store limit per user
        result = await db.execute(
            select(func.count(KnowledgeStore.id)).where(
                KnowledgeStore.owner_id == current_user.id
            )
        )
        store_count = result.scalar()
        
        if store_count >= tier_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge store limit reached ({tier_limit} for {current_user.tier.value} tier)"
            )
    
    store = KnowledgeStore(
        owner_id=current_user.id,
        name=request_data.name,
        description=request_data.description,
        icon=request_data.icon,
        color=request_data.color,
        is_public=request_data.is_public,
        is_discoverable=request_data.is_discoverable,
        chunk_size=request_data.chunk_size,
        chunk_overlap=request_data.chunk_overlap,
    )
    
    db.add(store)
    await db.commit()
    await db.refresh(store)
    
    response = KnowledgeStoreResponse.model_validate(store)
    response.owner_username = current_user.username
    response.permission = "admin"
    
    return response


@router.get("", response_model=List[KnowledgeStoreResponse])
async def list_knowledge_stores(
    include_shared: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all knowledge stores the user owns or has access to"""
    import logging
    logger = logging.getLogger(__name__)
    
    stores = []
    
    try:
        # Get owned stores
        logger.debug(f"Fetching knowledge stores for user {current_user.id}")
        result = await db.execute(
            select(KnowledgeStore)
            .where(KnowledgeStore.owner_id == current_user.id)
            .order_by(KnowledgeStore.updated_at.desc())
        )
        owned = result.scalars().all()
        logger.debug(f"Found {len(owned)} owned stores")
        
        for store in owned:
            try:
                response = KnowledgeStoreResponse.model_validate(store)
                response.owner_username = current_user.username
                response.permission = "admin"
                stores.append(response)
            except Exception as e:
                logger.error(f"Error validating store {store.id}: {e}")
        
        # Get shared stores
        if include_shared:
            result = await db.execute(
                select(KnowledgeStoreShare, KnowledgeStore, User)
                .join(KnowledgeStore, KnowledgeStoreShare.knowledge_store_id == KnowledgeStore.id)
                .join(User, KnowledgeStore.owner_id == User.id)
                .where(
                    KnowledgeStoreShare.shared_with_user_id == current_user.id,
                    or_(
                        KnowledgeStoreShare.expires_at.is_(None),
                        KnowledgeStoreShare.expires_at > datetime.now(timezone.utc)
                    )
                )
            )
            
            for share, store, owner in result:
                try:
                    response = KnowledgeStoreResponse.model_validate(store)
                    response.owner_username = owner.username
                    response.permission = share.permission.value
                    stores.append(response)
                except Exception as e:
                    logger.error(f"Error validating shared store {store.id}: {e}")
        
        logger.debug(f"Returning {len(stores)} total stores")
        return stores
    except Exception as e:
        logger.error(f"Error in list_knowledge_stores: {e}")
        raise


@router.get("/discover", response_model=List[KnowledgeStoreResponse])
async def discover_knowledge_stores(
    search: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Discover public knowledge stores"""
    query = (
        select(KnowledgeStore, User)
        .join(User, KnowledgeStore.owner_id == User.id)
        .where(
            KnowledgeStore.is_public == True,
            KnowledgeStore.is_discoverable == True
        )
    )
    
    if search:
        query = query.where(
            or_(
                KnowledgeStore.name.ilike(f"%{search}%"),
                KnowledgeStore.description.ilike(f"%{search}%")
            )
        )
    
    query = query.order_by(KnowledgeStore.document_count.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    stores = []
    
    for store, owner in result:
        response = KnowledgeStoreResponse.model_validate(store)
        response.owner_username = owner.username
        response.permission = "view"
        stores.append(response)
    
    return stores


@router.get("/{store_id}", response_model=KnowledgeStoreResponse)
async def get_knowledge_store(
    store_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific knowledge store"""
    store, permission = await get_store_with_access(store_id, current_user, db)
    
    # Get owner info
    result = await db.execute(
        select(User).where(User.id == store.owner_id)
    )
    owner = result.scalar_one()
    
    response = KnowledgeStoreResponse.model_validate(store)
    response.owner_username = owner.username
    response.permission = permission.value
    
    return response


@router.patch("/{store_id}", response_model=KnowledgeStoreResponse)
async def update_knowledge_store(
    store_id: str,
    update_data: KnowledgeStoreUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a knowledge store (requires edit permission, global settings require admin)"""
    store, permission = await get_store_with_access(
        store_id, current_user, db, SharePermission.EDIT
    )
    
    update_dict = update_data.model_dump(exclude_unset=True)
    
    # Global settings require admin
    global_fields = {"is_global", "global_min_score", "global_max_results"}
    if any(field in update_dict for field in global_fields):
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can modify global knowledge store settings"
            )
    
    for field, value in update_dict.items():
        setattr(store, field, value)
    
    await db.commit()
    await db.refresh(store)
    
    result = await db.execute(
        select(User).where(User.id == store.owner_id)
    )
    owner = result.scalar_one()
    
    response = KnowledgeStoreResponse.model_validate(store)
    response.owner_username = owner.username
    response.permission = permission.value
    
    return response


@router.delete("/{store_id}")
async def delete_knowledge_store(
    store_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge store (owner only)"""
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.ADMIN
    )
    
    # Only owner can delete
    if store.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can delete a knowledge store"
        )
    
    await db.delete(store)
    await db.commit()
    
    return {"message": "Knowledge store deleted", "id": store_id}


# === Endpoints: Documents ===

@router.get("/{store_id}/documents", response_model=List[DocumentInfo])
async def list_store_documents(
    store_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge store"""
    store, _ = await get_store_with_access(store_id, current_user, db)
    
    result = await db.execute(
        select(Document)
        .where(Document.knowledge_store_id == store_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    
    return [DocumentInfo.model_validate(d) for d in documents]


@router.post("/{store_id}/documents")
async def add_document_to_store(
    store_id: str,
    request: Request,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document to a knowledge store.
    Supported formats: PDF, TXT, MD, code files, and more.
    """
    from app.models.models import SystemSetting
    
    # Note: No rate limiting for knowledge base uploads - users need to bulk upload files
    
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.EDIT
    )
    
    # Get dynamic settings
    async def get_setting_int(key: str, default: int) -> int:
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        return int(setting.value) if setting else default
    
    max_upload_size_mb = await get_setting_int("max_upload_size_mb", settings.MAX_UPLOAD_SIZE_MB)
    max_kb_size_mb = await get_setting_int("max_knowledge_store_size_mb", 500)
    
    # Security: Check for dangerous file patterns
    if is_dangerous_file(file.filename or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type not allowed for security reasons"
        )
    
    # Get file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    
    # Map MIME types to simple type names
    mime_to_type = {
        "application/pdf": "pdf",
        "text/plain": "txt",
        "text/markdown": "md",
        "text/html": "html",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/json": "json",
        "text/csv": "csv",
        "text/javascript": "js",
        "application/javascript": "js",
        "text/x-python": "py",
    }
    
    # Programming file extensions (treated as text)
    code_extensions = {
        ".py", ".pyi", ".pyw",  # Python
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",  # JS/TS
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx",  # C/C++
        ".java",  # Java
        ".rs",  # Rust
        ".go",  # Go
        ".rb",  # Ruby
        ".html", ".htm", ".css", ".scss", ".sass", ".less",  # Web
        ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg",  # Config
        ".sh", ".bash", ".zsh",  # Shell
        ".sql", ".r", ".swift", ".kt", ".scala", ".php",
        ".lua", ".pl", ".pm", ".ex", ".exs", ".erl",
        ".hs", ".ml", ".fs", ".clj", ".lisp", ".el",
        ".vim", ".dockerfile", ".makefile", ".cmake",
        ".md", ".txt", ".json", ".csv",  # Documents
    }
    
    content_type = file.content_type or "application/octet-stream"
    
    # Determine file type - prefer extension for code files
    if file_ext in code_extensions:
        file_type = file_ext[1:]  # Remove the dot
    elif content_type in mime_to_type:
        file_type = mime_to_type[content_type]
    elif content_type.startswith("text/"):
        file_type = file_ext[1:] if file_ext else "txt"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type} ({file_ext}). Upload code files, documents, or text files."
        )
    
    # Save file
    file_content = await file.read()
    file_size = len(file_content)
    
    # Check individual file size limit
    max_size = max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {max_upload_size_mb}MB"
        )
    
    # Check knowledge store total size limit (admins bypass this)
    if not current_user.is_admin:
        current_store_size = store.total_size_bytes or 0
        max_kb_size_bytes = max_kb_size_mb * 1024 * 1024
        if current_store_size + file_size > max_kb_size_bytes:
            current_mb = current_store_size / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge store size limit reached. Current: {current_mb:.1f}MB, Max: {max_kb_size_mb}MB"
            )
    
    # Create upload directory
    upload_dir = f"uploads/knowledge_stores/{store_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Sanitize filename to prevent path traversal attacks
    safe_filename = os.path.basename(file.filename or "upload")
    # Remove any remaining path separators
    safe_filename = safe_filename.replace("/", "_").replace("\\", "_")
    # Prevent .. sequences
    safe_filename = safe_filename.replace("..", "_")
    
    file_path = f"{upload_dir}/{safe_filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Create document record
    document = Document(
        owner_id=current_user.id,
        knowledge_store_id=store_id,
        name=file.filename,
        description=description,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
    )
    
    db.add(document)
    await db.commit()
    await db.refresh(document)
    
    # Queue document for processing (survives restarts)
    queue = get_document_queue()
    task = DocumentTask(
        task_id=str(uuid.uuid4()),
        document_id=document.id,
        user_id=current_user.id,
        knowledge_store_id=store_id,
        file_path=file_path,
        file_type=file_type,
    )
    queue.add_task(task)
    
    return DocumentInfo.model_validate(document)


@router.delete("/{store_id}/documents/{document_id}")
async def remove_document_from_store(
    store_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a document from a knowledge store"""
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.EDIT
    )
    
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.knowledge_store_id == store_id
        )
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Update store stats
    store.document_count = max(0, store.document_count - 1)
    store.total_chunks = max(0, store.total_chunks - document.chunk_count)
    store.total_size_bytes = max(0, store.total_size_bytes - document.file_size)
    
    # Delete file
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete document and chunks via RAGService (handles FAISS update)
    from app.services.rag import RAGService
    rag_service = RAGService()
    await rag_service.delete_document(db, document_id)
    
    await db.commit()
    
    return {"message": "Document removed", "id": document_id}


# === Endpoints: Sharing ===

@router.post("/{store_id}/share", response_model=ShareResponse)
async def share_knowledge_store(
    store_id: str,
    request: ShareCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Share a knowledge store with another user or generate a share link.
    
    Either provide `user_email` to share with a specific user,
    or set `create_link=true` to generate a shareable link.
    """
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.ADMIN
    )
    
    # Only owner can share
    if store.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can share this knowledge store"
        )
    
    # Validate permission
    try:
        permission = SharePermission(request.permission)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission: {request.permission}"
        )
    
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
    
    share = KnowledgeStoreShare(
        knowledge_store_id=store_id,
        permission=permission,
        expires_at=expires_at,
        max_uses=request.max_uses,
        shared_by_user_id=current_user.id,
        message=request.message,
    )
    
    if request.user_email:
        # Share with specific user
        result = await db.execute(
            select(User).where(User.email == request.user_email)
        )
        target_user = result.scalar_one_or_none()
        
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if target_user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot share with yourself"
            )
        
        # Check if already shared
        result = await db.execute(
            select(KnowledgeStoreShare).where(
                KnowledgeStoreShare.knowledge_store_id == store_id,
                KnowledgeStoreShare.shared_with_user_id == target_user.id
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Already shared with this user"
            )
        
        share.shared_with_user_id = target_user.id
        share.accepted_at = datetime.now(timezone.utc)  # Auto-accept for direct shares
        
    elif request.create_link:
        # Generate share link
        share.share_token = secrets.token_urlsafe(32)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide user_email or set create_link=true"
        )
    
    db.add(share)
    await db.commit()
    await db.refresh(share)
    
    response = ShareResponse.model_validate(share)
    if share.share_token:
        response.share_url = f"/api/knowledge-stores/join/{share.share_token}"
    if request.user_email:
        response.shared_with_email = request.user_email
    
    return response


@router.get("/{store_id}/shares", response_model=List[ShareResponse])
async def list_store_shares(
    store_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all shares for a knowledge store (owner only)"""
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.ADMIN
    )
    
    if store.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can view shares"
        )
    
    result = await db.execute(
        select(KnowledgeStoreShare, User)
        .outerjoin(User, KnowledgeStoreShare.shared_with_user_id == User.id)
        .where(KnowledgeStoreShare.knowledge_store_id == store_id)
        .order_by(KnowledgeStoreShare.created_at.desc())
    )
    
    shares = []
    for share, user in result:
        response = ShareResponse.model_validate(share)
        if user:
            response.shared_with_email = user.email
        if share.share_token:
            response.share_url = f"/api/knowledge-stores/join/{share.share_token}"
        shares.append(response)
    
    return shares


@router.delete("/{store_id}/shares/{share_id}")
async def revoke_share(
    store_id: str,
    share_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a share"""
    store, _ = await get_store_with_access(
        store_id, current_user, db, SharePermission.ADMIN
    )
    
    if store.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can revoke shares"
        )
    
    result = await db.execute(
        select(KnowledgeStoreShare).where(
            KnowledgeStoreShare.id == share_id,
            KnowledgeStoreShare.knowledge_store_id == store_id
        )
    )
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share not found"
        )
    
    await db.delete(share)
    await db.commit()
    
    return {"message": "Share revoked", "id": share_id}


@router.post("/join/{share_token}")
async def join_via_share_link(
    share_token: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Join a knowledge store via share link"""
    result = await db.execute(
        select(KnowledgeStoreShare, KnowledgeStore)
        .join(KnowledgeStore, KnowledgeStoreShare.knowledge_store_id == KnowledgeStore.id)
        .where(KnowledgeStoreShare.share_token == share_token)
    )
    row = result.one_or_none()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid share link"
        )
    
    share, store = row
    
    # Check expiration
    if share.expires_at and share.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Share link has expired"
        )
    
    # Check max uses
    if share.max_uses and share.use_count >= share.max_uses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Share link has reached maximum uses"
        )
    
    # Check if already a member
    if store.owner_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You own this knowledge store"
        )
    
    result = await db.execute(
        select(KnowledgeStoreShare).where(
            KnowledgeStoreShare.knowledge_store_id == store.id,
            KnowledgeStoreShare.shared_with_user_id == current_user.id
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have access to this knowledge store"
        )
    
    # Create new share for this user
    new_share = KnowledgeStoreShare(
        knowledge_store_id=store.id,
        shared_with_user_id=current_user.id,
        permission=share.permission,
        expires_at=share.expires_at,
        shared_by_user_id=share.shared_by_user_id,
        accepted_at=datetime.now(timezone.utc),
    )
    
    db.add(new_share)
    share.use_count += 1
    await db.commit()
    
    return {
        "message": "Successfully joined knowledge store",
        "store_id": store.id,
        "store_name": store.name,
        "permission": share.permission.value,
    }


# === Admin Endpoints: Global Knowledge Stores ===

@router.get("/admin/global", response_model=List[KnowledgeStoreResponse])
async def list_global_stores(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all global knowledge stores (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    result = await db.execute(
        select(KnowledgeStore)
        .where(KnowledgeStore.is_global == True)
        .order_by(KnowledgeStore.name)
    )
    stores = result.scalars().all()
    
    responses = []
    for store in stores:
        owner_result = await db.execute(
            select(User).where(User.id == store.owner_id)
        )
        owner = owner_result.scalar_one_or_none()
        
        response = KnowledgeStoreResponse.model_validate(store)
        response.owner_username = owner.username if owner else "Unknown"
        responses.append(response)
    
    return responses


@router.post("/admin/global/{store_id}")
async def set_store_global(
    store_id: str,
    is_global: bool = True,
    min_score: float = 0.7,
    max_results: int = 3,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Set a knowledge store as global (admin only)
    
    When a store is marked as global:
    - It's automatically searched on every chat query for all users
    - It's also made public so users can see it exists
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    result = await db.execute(
        select(KnowledgeStore).where(KnowledgeStore.id == store_id)
    )
    store = result.scalar_one_or_none()
    
    if not store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge store not found"
        )
    
    store.is_global = is_global
    store.global_min_score = min_score
    store.global_max_results = max_results
    
    # Global stores should also be public so users can see they exist
    # (they're being searched on their behalf, so they should know about it)
    if is_global:
        store.is_public = True
        store.is_discoverable = True
    
    await db.commit()
    
    action = "enabled" if is_global else "disabled"
    return {
        "message": f"Global store {action}",
        "store_id": store_id,
        "store_name": store.name,
        "is_global": is_global,
        "is_public": store.is_public,
        "global_min_score": min_score,
        "global_max_results": max_results,
    }


@router.get("/admin/all", response_model=List[KnowledgeStoreResponse])
async def list_all_stores_admin(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all knowledge stores in the system (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    result = await db.execute(
        select(KnowledgeStore)
        .order_by(KnowledgeStore.is_global.desc(), KnowledgeStore.name)
    )
    stores = result.scalars().all()
    
    # Batch fetch owners
    owner_ids = list(set(s.owner_id for s in stores))
    owner_result = await db.execute(
        select(User).where(User.id.in_(owner_ids))
    )
    owners = {str(u.id): u.username for u in owner_result.scalars().all()}
    
    responses = []
    for store in stores:
        response = KnowledgeStoreResponse.model_validate(store)
        response.owner_username = owners.get(str(store.owner_id), "Unknown")
        responses.append(response)
    
    return responses
