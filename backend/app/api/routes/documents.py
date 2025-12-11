"""
Documents and RAG API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import os
import uuid
import aiofiles
from pathlib import Path

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.api.schemas import DocumentResponse, DocumentSearch, SearchResult
from app.models.models import User, Document
from app.services.rag import RAGService, DocumentProcessor
from app.services.document_queue import get_document_queue, DocumentTask
from app.core.config import settings


router = APIRouter(tags=["Documents"])


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    processed_only: bool = False,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's documents"""
    
    query = select(Document).where(Document.owner_id == user.id)
    
    if processed_only:
        query = query.where(Document.is_processed == True)
    
    query = query.order_by(Document.created_at.desc())
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return [DocumentResponse.model_validate(d) for d in documents]


@router.post("", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document for RAG"""
    
    # Check file type - allow by MIME type OR by extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    mime_allowed = file.content_type in settings.ALLOWED_FILE_TYPES
    ext_allowed = file_ext in settings.ALLOWED_FILE_EXTENSIONS
    
    if not mime_allowed and not ext_allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed extensions: {', '.join(settings.ALLOWED_FILE_EXTENSIONS[:20])}...",
        )
    
    # Check file size
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )
    
    # Save file with UUID name (prevents path traversal)
    file_id = str(uuid.uuid4())
    # Safely extract extension - only allow alphanumeric extensions
    raw_ext = Path(file.filename).suffix if file.filename else ""
    file_ext = raw_ext if raw_ext.replace(".", "").isalnum() else ""
    file_path = os.path.join(settings.UPLOAD_DIR, user.id, f"{file_id}{file_ext}")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)
    
    # Create document record
    document = Document(
        owner_id=user.id,
        name=file.filename,
        description=description,
        file_path=file_path,
        file_type=file.content_type,
        file_size=file_size,
    )
    db.add(document)
    await db.flush()
    
    # Queue document for processing (survives restarts)
    queue = get_document_queue()
    task = DocumentTask(
        task_id=str(uuid.uuid4()),
        document_id=document.id,
        user_id=user.id,
        knowledge_store_id=None,
        file_path=file_path,
        file_type=file.content_type,
    )
    queue.add_task(task)
    
    await db.commit()
    await db.refresh(document)
    
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific document"""
    
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document"""
    
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Delete file from disk
    if document.file_path and os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete from database (RAGService handles document, chunks, and FAISS update)
    rag_service = RAGService()
    await rag_service.delete_document(db, document_id)
    
    await db.commit()
    
    return {"status": "deleted", "document_id": document_id}


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reprocess a document (regenerate embeddings)"""
    
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    if not os.path.exists(document.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document file not found",
        )
    
    # Reset document status and queue for reprocessing
    document.is_processed = False
    document.chunk_count = 0
    await db.commit()
    
    # Queue for reprocessing
    queue = get_document_queue()
    task = DocumentTask(
        task_id=str(uuid.uuid4()),
        document_id=document.id,
        user_id=user.id,
        knowledge_store_id=document.knowledge_store_id,
        file_path=document.file_path,
        file_type=document.file_type,
    )
    queue.add_task(task)
    
    await db.refresh(document)
    
    return DocumentResponse.model_validate(document)


@router.post("/search", response_model=List[SearchResult])
async def search_documents(
    search: DocumentSearch,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search across documents using semantic search"""
    
    rag_service = RAGService()
    results = await rag_service.search(
        db=db,
        user=user,
        query=search.query,
        document_ids=search.document_ids,
        top_k=search.top_k,
    )
    
    return [
        SearchResult(
            document_id=r["document_id"],
            document_name=r["document_name"],
            content=r["content"],
            similarity=r["similarity"],
            chunk_index=r["chunk_index"],
        )
        for r in results
    ]
