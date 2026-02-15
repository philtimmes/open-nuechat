"""
User Settings API routes

Includes:
- Chat Knowledge Base settings
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import Optional

from app.db.database import get_db, sync_session_maker
from app.api.dependencies import get_current_user
from app.models.models import User, Chat, Message, KnowledgeStore, Document, DocumentChunk
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["User Settings"])

# Thread pool for background indexing
_indexing_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat_knowledge_")


class ChatKnowledgeStatus(BaseModel):
    """Response model for chat knowledge status"""
    enabled: bool
    status: str  # idle, processing, completed
    indexed_count: int
    total_count: int
    last_indexed: Optional[str] = None


class ChatKnowledgeToggle(BaseModel):
    """Request to toggle chat knowledge"""
    enabled: bool


@router.get("/chat-knowledge", response_model=ChatKnowledgeStatus)
async def get_chat_knowledge_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current chat knowledge status"""
    
    # Count indexed and total chats
    indexed_result = await db.execute(
        select(Chat).where(
            Chat.owner_id == user.id,
            Chat.is_knowledge_indexed == True
        )
    )
    indexed_count = len(indexed_result.scalars().all())
    
    total_result = await db.execute(
        select(Chat).where(Chat.owner_id == user.id)
    )
    total_count = len(total_result.scalars().all())
    
    return ChatKnowledgeStatus(
        enabled=user.all_chats_knowledge_enabled or False,
        status=user.chat_knowledge_status or "idle",
        indexed_count=indexed_count,
        total_count=total_count,
        last_indexed=user.chat_knowledge_last_indexed.isoformat() if user.chat_knowledge_last_indexed else None
    )


@router.post("/chat-knowledge", response_model=ChatKnowledgeStatus)
async def toggle_chat_knowledge(
    request: ChatKnowledgeToggle,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Toggle chat knowledge and start/stop indexing"""
    
    if request.enabled:
        # Enable and start indexing
        user.all_chats_knowledge_enabled = True
        user.chat_knowledge_status = "processing"
        await db.commit()
        
        # Start background indexing in thread pool (using sync DB)
        _indexing_executor.submit(index_user_chats_sync, user.id)
        
        logger.info(f"[CHAT_KNOWLEDGE] Started indexing for user {user.id}")
    else:
        # Disable - clear indexed data
        user.all_chats_knowledge_enabled = False
        user.chat_knowledge_status = "idle"
        
        # Clear indexed flags on all chats
        await db.execute(
            update(Chat)
            .where(Chat.owner_id == user.id)
            .values(is_knowledge_indexed=False)
        )
        
        # Delete the chat knowledge store if it exists
        if user.chat_knowledge_store_id:
            store_result = await db.execute(
                select(KnowledgeStore).where(KnowledgeStore.id == user.chat_knowledge_store_id)
            )
            store = store_result.scalar_one_or_none()
            if store:
                await db.delete(store)
            user.chat_knowledge_store_id = None
        
        await db.commit()
        logger.info(f"[CHAT_KNOWLEDGE] Disabled for user {user.id}")
    
    # Return current status
    indexed_result = await db.execute(
        select(Chat).where(
            Chat.owner_id == user.id,
            Chat.is_knowledge_indexed == True
        )
    )
    indexed_count = len(indexed_result.scalars().all())
    
    total_result = await db.execute(
        select(Chat).where(Chat.owner_id == user.id)
    )
    total_count = len(total_result.scalars().all())
    
    return ChatKnowledgeStatus(
        enabled=user.all_chats_knowledge_enabled,
        status=user.chat_knowledge_status,
        indexed_count=indexed_count,
        total_count=total_count,
        last_indexed=user.chat_knowledge_last_indexed.isoformat() if user.chat_knowledge_last_indexed else None
    )


@router.get("/export-data")
async def export_user_data(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Export all user data as a downloadable zip file.
    
    Includes:
    - All chats with messages (JSON)
    - Knowledge stores metadata (JSON)
    - Document metadata (JSON)
    """
    import json
    import zipfile
    import io
    from fastapi.responses import StreamingResponse
    
    logger.info(f"[EXPORT] Starting data export for user {user.id}")
    
    export_data = {
        "export_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
    }
    
    # NC-0.8.0.13: Check if tool call logs should be included
    from app.services.settings_service import SettingsService
    include_tool_logs = await SettingsService.get_bool(db, "store_tool_call_log")
    
    # Export all chats with messages
    chats_result = await db.execute(
        select(Chat).where(Chat.owner_id == user.id).order_by(Chat.created_at.desc())
    )
    chats = chats_result.scalars().all()
    
    exported_chats = []
    for chat in chats:
        # Get all messages for this chat
        messages_result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat.id)
            .order_by(Message.created_at)
        )
        messages = messages_result.scalars().all()
        
        chat_data = {
            "id": chat.id,
            "title": chat.title,
            "created_at": chat.created_at.isoformat() if chat.created_at else None,
            "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
            "system_prompt": chat.system_prompt,
            "model": chat.model,
            "source": chat.source,
            "assistant_id": chat.assistant_id,
            "assistant_name": chat.assistant_name,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role.value if msg.role else None,
                    "content": msg.content,
                    "content_type": msg.content_type.value if msg.content_type else None,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "parent_id": msg.parent_id,
                    "attachments": msg.attachments,
                    **({"tool_call_log": (msg.message_metadata or {}).get("tool_call_log")}
                       if include_tool_logs and isinstance(msg.message_metadata, dict)
                       and msg.message_metadata.get("tool_call_log") else {}),
                }
                for msg in messages
            ]
        }
        exported_chats.append(chat_data)
    
    export_data["chats"] = exported_chats
    logger.info(f"[EXPORT] Exported {len(exported_chats)} chats")
    
    # Export knowledge stores
    ks_result = await db.execute(
        select(KnowledgeStore).where(KnowledgeStore.owner_id == user.id)
    )
    knowledge_stores = ks_result.scalars().all()
    
    exported_ks = []
    for ks in knowledge_stores:
        # Get documents in this knowledge store
        docs_result = await db.execute(
            select(Document).where(Document.knowledge_store_id == ks.id)
        )
        docs = docs_result.scalars().all()
        
        ks_data = {
            "id": ks.id,
            "name": ks.name,
            "description": ks.description,
            "is_public": ks.is_public,
            "is_global": ks.is_global,
            "created_at": ks.created_at.isoformat() if ks.created_at else None,
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "description": doc.description,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "chunk_count": doc.chunk_count,
                    "is_processed": doc.is_processed,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                }
                for doc in docs
            ]
        }
        exported_ks.append(ks_data)
    
    export_data["knowledge_stores"] = exported_ks
    logger.info(f"[EXPORT] Exported {len(exported_ks)} knowledge stores")
    
    # Export user's own documents (not in a knowledge store)
    own_docs_result = await db.execute(
        select(Document).where(
            Document.owner_id == user.id,
            Document.knowledge_store_id == None
        )
    )
    own_docs = own_docs_result.scalars().all()
    
    export_data["documents"] = [
        {
            "id": doc.id,
            "name": doc.name,
            "description": doc.description,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "is_processed": doc.is_processed,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in own_docs
    ]
    logger.info(f"[EXPORT] Exported {len(own_docs)} standalone documents")
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Main export file
        zf.writestr(
            'export.json',
            json.dumps(export_data, indent=2, ensure_ascii=False)
        )
        
        # Also create separate files for easier access
        zf.writestr(
            'chats.json',
            json.dumps({"chats": exported_chats}, indent=2, ensure_ascii=False)
        )
        
        if exported_ks:
            zf.writestr(
                'knowledge_stores.json',
                json.dumps({"knowledge_stores": exported_ks}, indent=2, ensure_ascii=False)
            )
        
        # Add a readme
        readme = f"""NueChat Data Export
==================

Exported: {export_data['exported_at']}
User: {user.email}

Contents:
- export.json: Complete export with all data
- chats.json: All chat conversations ({len(exported_chats)} chats)
- knowledge_stores.json: Knowledge bases and documents ({len(exported_ks)} stores)

This export contains:
- {len(exported_chats)} chats with {sum(len(c['messages']) for c in exported_chats)} total messages
- {len(exported_ks)} knowledge stores
- {sum(len(ks['documents']) for ks in exported_ks)} documents in knowledge stores
- {len(own_docs)} standalone documents

Note: Document files (PDFs, etc.) are not included in this export.
Only metadata and text content is exported.
"""
        zf.writestr('README.txt', readme)
    
    zip_buffer.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nuechat_export_{timestamp}.zip"
    
    logger.info(f"[EXPORT] Export complete for user {user.id}: {filename}")
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


def index_user_chats_sync(user_id: str):
    """Synchronous background task to index all user chats into knowledge base"""
    import uuid
    import numpy as np
    
    db = sync_session_maker()
    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.error(f"[CHAT_KNOWLEDGE] User {user_id} not found")
            return
        
        # Create or get knowledge store for user's chats
        store_id = user.chat_knowledge_store_id
        if not store_id:
            store_id = str(uuid.uuid4())
            store = KnowledgeStore(
                id=store_id,
                owner_id=user_id,
                name="My Chat History",
                description="Automatically indexed knowledge from all your conversations",
                is_public=False,
            )
            db.add(store)
            user.chat_knowledge_store_id = store_id
            db.commit()
            logger.info(f"[CHAT_KNOWLEDGE] Created knowledge store {store_id} for user {user_id}")
        else:
            # Verify store exists
            store = db.query(KnowledgeStore).filter(KnowledgeStore.id == store_id).first()
            if not store:
                # Recreate if missing
                store = KnowledgeStore(
                    id=store_id,
                    owner_id=user_id,
                    name="My Chat History",
                    description="Automatically indexed knowledge from all your conversations",
                    is_public=False,
                )
                db.add(store)
                db.commit()
        
        # Get all unindexed chats, ordered by newest first
        chats = db.query(Chat).filter(
            Chat.owner_id == user_id,
            Chat.is_knowledge_indexed == False
        ).order_by(Chat.updated_at.desc()).all()
        
        logger.info(f"[CHAT_KNOWLEDGE] Found {len(chats)} chats to index for user {user_id}")
        
        rag_service = RAGService()
        all_embeddings = []
        all_chunk_ids = []
        
        for chat in chats:
            # Check if user disabled while processing
            db.refresh(user)
            if not user.all_chats_knowledge_enabled:
                logger.info(f"[CHAT_KNOWLEDGE] Indexing cancelled for user {user_id}")
                return
            
            try:
                # Get all messages for this chat
                messages = db.query(Message).filter(
                    Message.chat_id == chat.id
                ).order_by(Message.created_at).all()
                
                if not messages:
                    chat.is_knowledge_indexed = True
                    db.commit()
                    continue
                
                # Build conversation text
                conversation_parts = []
                conversation_parts.append(f"# {chat.title}")
                conversation_parts.append(f"Date: {chat.created_at.strftime('%Y-%m-%d')}")
                conversation_parts.append("")
                
                for msg in messages:
                    role_label = "User" if msg.role.value == "user" else "Assistant"
                    content = msg.content or ""
                    
                    # Include attachment content (PDFs, docs, etc.)
                    if msg.attachments:
                        for att in msg.attachments:
                            att_content = att.get("content", "")
                            att_name = att.get("filename", att.get("name", "attachment"))
                            if att_content and len(att_content) > 100:
                                # Add attachment content with header
                                content += f"\n\n[Attached: {att_name}]\n{att_content}"
                    
                    if len(content) > 50000:  # Increased limit for attachments
                        content = content[:50000] + "... [truncated]"
                    conversation_parts.append(f"**{role_label}:** {content}")
                    conversation_parts.append("")
                
                full_text = "\n".join(conversation_parts)
                
                # Create a Document to represent this chat
                doc_id = str(uuid.uuid4())
                doc = Document(
                    id=doc_id,
                    owner_id=user_id,
                    knowledge_store_id=store_id,
                    name=f"Chat: {chat.title}",
                    description=f"Indexed chat from {chat.created_at.strftime('%Y-%m-%d')}",
                    file_path=f"chat://{chat.id}",  # Virtual path
                    file_type="chat/history",
                    file_size=len(full_text),
                    is_processed=True,
                )
                db.add(doc)
                
                # Chunk the conversation using RAG service
                chunks = rag_service.chunk_text(full_text)
                
                # Create embeddings and store chunks
                for chunk_data in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_text = chunk_data["content"]
                    chunk_index = chunk_data["index"]
                    
                    # Generate embedding
                    embedding = rag_service.embed_text(chunk_text)
                    if embedding is None:
                        continue
                    
                    # Store chunk in database - linked to Document, not KnowledgeStore
                    chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        embedding=embedding.tobytes(),  # Store as binary
                        chunk_metadata={
                            "chat_id": chat.id,
                            "chat_title": chat.title,
                            "source": "chat_history"
                        }
                    )
                    db.add(chunk)
                    
                    # Collect for FAISS index
                    all_embeddings.append(embedding)
                    all_chunk_ids.append(chunk_id)
                
                # Update document chunk count
                doc.chunk_count = len(chunks)
                
                # Mark chat as indexed
                chat.is_knowledge_indexed = True
                db.commit()
                
                logger.info(f"[CHAT_KNOWLEDGE] Indexed chat {chat.id} ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"[CHAT_KNOWLEDGE] Failed to index chat {chat.id}: {e}")
                db.rollback()
                continue
        
        # Build FAISS index for the knowledge store
        if all_embeddings:
            try:
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                index_manager = rag_service.get_index_manager()
                index_manager.build_index(store_id, embeddings_array, all_chunk_ids)
                logger.info(f"[CHAT_KNOWLEDGE] Built FAISS index with {len(all_chunk_ids)} chunks")
            except Exception as e:
                logger.error(f"[CHAT_KNOWLEDGE] Failed to build FAISS index: {e}")
        
        # Update user status
        db.refresh(user)
        if user.all_chats_knowledge_enabled:
            user.chat_knowledge_status = "completed"
            user.chat_knowledge_last_indexed = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"[CHAT_KNOWLEDGE] Completed indexing for user {user_id}")
            
    except Exception as e:
        logger.error(f"[CHAT_KNOWLEDGE] Error indexing chats for user {user_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            db.rollback()
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.chat_knowledge_status = "idle"
                db.commit()
        except:
            pass
    finally:
        db.close()


def index_single_chat_sync(user_id: str, chat_id: str):
    """
    Index a single chat into the user's chat knowledge base.
    Called after a new message is added to keep the KB up to date.
    """
    import uuid
    import numpy as np
    
    db = sync_session_maker()
    try:
        # Get user and check if chat knowledge is enabled
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.all_chats_knowledge_enabled:
            return
        
        store_id = user.chat_knowledge_store_id
        if not store_id:
            logger.debug(f"[CHAT_KNOWLEDGE] No knowledge store for user {user_id}, skipping single chat index")
            return
        
        # Get the chat
        chat = db.query(Chat).filter(Chat.id == chat_id, Chat.owner_id == user_id).first()
        if not chat:
            logger.debug(f"[CHAT_KNOWLEDGE] Chat {chat_id} not found for user {user_id}")
            return
        
        # Check if already indexed - if so, we need to re-index (update)
        # Delete existing document for this chat
        existing_doc = db.query(Document).filter(
            Document.knowledge_store_id == store_id,
            Document.file_path == f"chat://{chat_id}"
        ).first()
        
        if existing_doc:
            db.delete(existing_doc)  # Cascades to chunks
            db.commit()
            logger.debug(f"[CHAT_KNOWLEDGE] Removed old index for chat {chat_id}")
        
        # Get all messages for this chat
        messages = db.query(Message).filter(
            Message.chat_id == chat_id
        ).order_by(Message.created_at).all()
        
        if not messages:
            chat.is_knowledge_indexed = True
            db.commit()
            return
        
        # Build conversation text
        conversation_parts = []
        conversation_parts.append(f"# {chat.title}")
        conversation_parts.append(f"Date: {chat.created_at.strftime('%Y-%m-%d')}")
        conversation_parts.append("")
        
        for msg in messages:
            role_label = "User" if msg.role.value == "user" else "Assistant"
            content = msg.content or ""
            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"
            conversation_parts.append(f"**{role_label}:** {content}")
            conversation_parts.append("")
        
        full_text = "\n".join(conversation_parts)
        
        # Create a Document to represent this chat
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            owner_id=user_id,
            knowledge_store_id=store_id,
            name=f"Chat: {chat.title}",
            description=f"Indexed chat from {chat.created_at.strftime('%Y-%m-%d')}",
            file_path=f"chat://{chat_id}",  # Virtual path
            file_type="chat/history",
            file_size=len(full_text),
            is_processed=True,
        )
        db.add(doc)
        
        rag_service = RAGService()
        
        # Chunk the conversation using RAG service
        chunks = rag_service.chunk_text(full_text)
        
        new_embeddings = []
        new_chunk_ids = []
        
        # Create embeddings and store chunks
        for chunk_data in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_text = chunk_data["content"]
            chunk_index = chunk_data["index"]
            
            # Generate embedding
            embedding = rag_service.embed_text(chunk_text)
            if embedding is None:
                continue
            
            # Store chunk in database
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                embedding=embedding.tobytes(),
                chunk_metadata={
                    "chat_id": chat_id,
                    "chat_title": chat.title,
                    "source": "chat_history"
                }
            )
            db.add(chunk)
            
            new_embeddings.append(embedding)
            new_chunk_ids.append(chunk_id)
        
        # Update document chunk count
        doc.chunk_count = len(chunks)
        
        # Mark chat as indexed
        chat.is_knowledge_indexed = True
        db.commit()
        
        # Rebuild FAISS index with all chunks from this store
        # Load all embeddings from the knowledge store
        all_docs = db.query(Document).filter(Document.knowledge_store_id == store_id).all()
        all_doc_ids = [d.id for d in all_docs]
        
        all_chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id.in_(all_doc_ids)
        ).all()
        
        if all_chunks:
            all_embeddings = []
            all_chunk_ids = []
            for chunk in all_chunks:
                if chunk.embedding:
                    # Convert binary back to numpy array
                    emb = np.frombuffer(chunk.embedding, dtype=np.float32)
                    all_embeddings.append(emb)
                    all_chunk_ids.append(chunk.id)
            
            if all_embeddings:
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                index_manager = rag_service.get_index_manager()
                index_manager.build_index(store_id, embeddings_array, all_chunk_ids)
                logger.info(f"[CHAT_KNOWLEDGE] Rebuilt FAISS index with {len(all_chunk_ids)} total chunks after adding chat {chat_id}")
        
        logger.info(f"[CHAT_KNOWLEDGE] Indexed single chat {chat_id} ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"[CHAT_KNOWLEDGE] Error indexing single chat {chat_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            db.rollback()
        except:
            pass
    finally:
        db.close()
    
