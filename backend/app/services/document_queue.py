"""
Persistent document processing queue.

Documents are queued for processing and persist across restarts.
A background worker processes pending documents asynchronously.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
from filelock import FileLock

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.models.models import Document, KnowledgeStore, SystemSetting

logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentTask:
    """A document processing task."""
    task_id: str
    document_id: str
    user_id: str
    knowledge_store_id: Optional[str]
    file_path: str
    file_type: str
    status: str = ProcessingStatus.PENDING.value
    created_at: str = ""
    error: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTask":
        return cls(**data)


class DocumentQueueService:
    """
    Persistent document processing queue.
    
    Queue state is persisted to disk so it survives container restarts.
    A background worker processes documents asynchronously.
    """
    
    _instance: Optional["DocumentQueueService"] = None
    
    def __init__(self, queue_file: str = "/app/data/document_queue.json"):
        self.queue_file = Path(queue_file)
        self.lock_file = Path(f"{queue_file}.lock")
        self.tasks: Dict[str, DocumentTask] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._debug_enabled = False  # Cached debug flag, updated by worker
        self._load_queue()
    
    @classmethod
    def get_instance(cls, queue_file: str = "/app/data/document_queue.json") -> "DocumentQueueService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(queue_file)
        return cls._instance
    
    async def _check_debug_enabled(self) -> bool:
        """Check if debug_document_queue setting is enabled."""
        try:
            async with async_session_maker() as db:
                result = await db.execute(
                    select(SystemSetting).where(SystemSetting.key == "debug_document_queue")
                )
                setting = result.scalar_one_or_none()
                self._debug_enabled = setting and setting.value == "true"
                return self._debug_enabled
        except Exception:
            return self._debug_enabled
    
    def _load_queue(self) -> None:
        """Load queue from disk."""
        try:
            # Ensure directory exists
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.queue_file.exists():
                with FileLock(self.lock_file):
                    with open(self.queue_file, "r") as f:
                        data = json.load(f)
                        self.tasks = {
                            k: DocumentTask.from_dict(v) 
                            for k, v in data.items()
                        }
                logger.info(f"Loaded {len(self.tasks)} tasks from queue file")
                
                # Count pending tasks
                pending = sum(1 for t in self.tasks.values() if t.status == ProcessingStatus.PENDING.value)
                processing = sum(1 for t in self.tasks.values() if t.status == ProcessingStatus.PROCESSING.value)
                if pending > 0 or processing > 0:
                    logger.info(f"Queue has {pending} pending, {processing} processing tasks")
                    # Reset processing tasks to pending (they were interrupted)
                    for task in self.tasks.values():
                        if task.status == ProcessingStatus.PROCESSING.value:
                            task.status = ProcessingStatus.PENDING.value
                            task.retry_count += 1
                    self._save_queue()
        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
            self.tasks = {}
    
    def _save_queue(self) -> None:
        """Save queue to disk."""
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.lock_file):
                with open(self.queue_file, "w") as f:
                    json.dump(
                        {k: v.to_dict() for k, v in self.tasks.items()},
                        f,
                        indent=2
                    )
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
    
    def add_task(self, task: DocumentTask) -> None:
        """Add a task to the queue."""
        self.tasks[task.task_id] = task
        self._save_queue()
        if self._debug_enabled:
            logger.info(f"[DOC_QUEUE] Added task {task.task_id} for document {task.document_id}")
    
    def remove_task(self, task_id: str) -> None:
        """Remove a completed task from the queue."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_queue()
            if self._debug_enabled:
                logger.info(f"[DOC_QUEUE] Removed task {task_id}")
    
    def get_pending_tasks(self) -> List[DocumentTask]:
        """Get all pending tasks."""
        return [
            t for t in self.tasks.values() 
            if t.status == ProcessingStatus.PENDING.value
        ]
    
    def get_task(self, task_id: str) -> Optional[DocumentTask]:
        """Get a specific task."""
        return self.tasks.get(task_id)
    
    def update_task_status(
        self, 
        task_id: str, 
        status: ProcessingStatus, 
        error: Optional[str] = None
    ) -> None:
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status.value
            if error:
                self.tasks[task_id].error = error
            self._save_queue()
    
    async def process_task(self, task: DocumentTask) -> bool:
        """
        Process a single document task.
        Returns True if successful, False otherwise.
        """
        from app.services.rag import RAGService, DocumentProcessor
        
        if self._debug_enabled:
            logger.info(f"[DOC_QUEUE] Processing task {task.task_id} - document {task.document_id}")
        
        self.update_task_status(task.task_id, ProcessingStatus.PROCESSING)
        
        try:
            async with async_session_maker() as db:
                # Get the document
                result = await db.execute(
                    select(Document).where(Document.id == task.document_id)
                )
                document = result.scalar_one_or_none()
                
                if not document:
                    if self._debug_enabled:
                        logger.info(f"[DOC_QUEUE] Document {task.document_id} not found, removing task")
                    self.remove_task(task.task_id)
                    return False
                
                # Check if already processed
                if document.is_processed:
                    if self._debug_enabled:
                        logger.info(f"[DOC_QUEUE] Document {task.document_id} already processed, skipping")
                    self.remove_task(task.task_id)
                    return True
                
                # Determine MIME type for text extraction
                type_to_mime = {
                    "pdf": "application/pdf",
                    "application/pdf": "application/pdf",
                    "txt": "text/plain",
                    "text/plain": "text/plain",
                    "md": "text/markdown",
                    "text/markdown": "text/markdown",
                    "json": "application/json",
                    "application/json": "application/json",
                    "csv": "text/csv",
                    "text/csv": "text/csv",
                }
                mime_type = type_to_mime.get(task.file_type, task.file_type)
                
                # Skip image files - they cannot be embedded and may crash the model
                image_types = {'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 
                              'image/bmp', 'image/svg+xml', 'png', 'jpg', 'jpeg', 'gif', 'webp'}
                if task.file_type in image_types or mime_type.startswith('image/'):
                    logger.info(f"[DOC_QUEUE] Skipping image file {task.document_id} (type={task.file_type})")
                    document.is_processed = True
                    document.chunk_count = 0
                    await db.commit()
                    self.remove_task(task.task_id)
                    return True
                
                # Extract text content
                if self._debug_enabled:
                    logger.info(f"[DOC_QUEUE] Extracting text from {task.file_path}")
                text_content = await DocumentProcessor.extract_text(task.file_path, mime_type)
                
                if not text_content or not text_content.strip():
                    # Mark as processed but with no chunks
                    document.is_processed = True
                    document.chunk_count = 0
                    await db.commit()
                    if self._debug_enabled:
                        logger.info(f"[DOC_QUEUE] Document {task.document_id} has no text content")
                    self.remove_task(task.task_id)
                    return True
                
                # Get knowledge store settings if applicable
                store = None
                if task.knowledge_store_id:
                    store_result = await db.execute(
                        select(KnowledgeStore).where(KnowledgeStore.id == task.knowledge_store_id)
                    )
                    store = store_result.scalar_one_or_none()
                
                # Process document
                rag_service = RAGService()
                
                # Use store-specific chunk settings if available
                if store:
                    if store.chunk_size:
                        rag_service.chunk_size = store.chunk_size
                    if store.chunk_overlap:
                        rag_service.chunk_overlap = store.chunk_overlap
                
                if self._debug_enabled:
                    logger.info(f"[DOC_QUEUE] Creating embeddings for document {task.document_id}")
                chunk_count = await rag_service.process_document(
                    db=db,
                    document=document,
                    text_content=text_content,
                )
                
                # Update store stats if applicable
                if store:
                    store.document_count += 1
                    store.total_chunks += chunk_count
                    store.total_size_bytes += document.file_size
                
                await db.commit()
                
                if self._debug_enabled:
                    logger.info(f"[DOC_QUEUE] Completed processing document {task.document_id} ({chunk_count} chunks)")
                self.remove_task(task.task_id)
                return True
                
        except Exception as e:
            if self._debug_enabled:
                logger.error(f"[DOC_QUEUE] Error processing task {task.task_id}: {e}")
            
            # Update task with error
            task.retry_count += 1
            max_retries = 3
            
            if task.retry_count >= max_retries:
                self.update_task_status(task.task_id, ProcessingStatus.FAILED, str(e))
                if self._debug_enabled:
                    logger.error(f"[DOC_QUEUE] Task {task.task_id} failed after {max_retries} retries")
                
                # Mark document as failed
                try:
                    async with async_session_maker() as db:
                        await db.execute(
                            update(Document)
                            .where(Document.id == task.document_id)
                            .values(is_processed=False)
                        )
                        await db.commit()
                except Exception:
                    pass
            else:
                self.update_task_status(task.task_id, ProcessingStatus.PENDING, str(e))
                if self._debug_enabled:
                    logger.info(f"[DOC_QUEUE] Task {task.task_id} will retry (attempt {task.retry_count}/{max_retries})")
            
            return False
    
    async def worker(self) -> None:
        """Background worker that processes pending documents."""
        # Check debug setting at start
        await self._check_debug_enabled()
        if self._debug_enabled:
            logger.info("[DOC_QUEUE] Worker started")
        
        check_counter = 0
        while self._running:
            try:
                # Refresh debug setting every 12 iterations (~1 minute at 5s sleep)
                check_counter += 1
                if check_counter >= 12:
                    await self._check_debug_enabled()
                    check_counter = 0
                
                pending = self.get_pending_tasks()
                
                if pending:
                    # Process one task at a time
                    task = pending[0]
                    await self.process_task(task)
                    # Small delay between tasks to yield to other coroutines
                    await asyncio.sleep(1)
                else:
                    # No pending tasks
                    if self._debug_enabled:
                        logger.info("[DOC_QUEUE] Empty Document Queue")
                    # Wait before checking again
                    await asyncio.sleep(5)
                    
            except asyncio.CancelledError:
                if self._debug_enabled:
                    logger.info("[DOC_QUEUE] Worker cancelled")
                break
            except Exception as e:
                if self._debug_enabled:
                    logger.error(f"[DOC_QUEUE] Worker error: {e}")
                await asyncio.sleep(10)
        
        if self._debug_enabled:
            logger.info("[DOC_QUEUE] Worker stopped")
    
    def start_worker(self) -> None:
        """Start the background worker."""
        if self._worker_task is not None:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self.worker())
        # Note: Initial debug log happens in worker() after checking setting
    
    def stop_worker(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None
        # Note: Stopped log happens in worker() based on debug setting
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status summary."""
        by_status = {}
        for task in self.tasks.values():
            by_status[task.status] = by_status.get(task.status, 0) + 1
        
        return {
            "total": len(self.tasks),
            "by_status": by_status,
            "worker_running": self._running,
        }


def get_document_queue() -> DocumentQueueService:
    """Get the document queue singleton."""
    return DocumentQueueService.get_instance()
