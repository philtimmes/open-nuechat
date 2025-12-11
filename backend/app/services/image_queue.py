"""
Image Generation Queue Service
Manages a queue of image generation tasks with async polling and frontend notification
"""
import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Awaitable
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Constants
MAX_QUEUE_SIZE = 10
POLL_INTERVAL = 2.0  # seconds between status polls
IMAGES_DIR = Path("/app/uploads/generated")


class TaskStatus(Enum):
    PENDING = "pending"       # Waiting in queue
    SUBMITTED = "submitted"   # Sent to image service
    PROCESSING = "processing" # Being processed by image service
    COMPLETED = "completed"   # Done, image saved
    FAILED = "failed"         # Generation failed
    CANCELLED = "cancelled"   # User cancelled


@dataclass
class ImageTask:
    """Represents an image generation task"""
    id: str
    prompt: str
    user_id: str
    chat_id: str
    message_id: str  # The assistant message ID to update
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    job_id: Optional[str] = None  # Job ID from image service
    
    # Results
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Callback for frontend notification
    notify_callback: Optional[Callable[[], Awaitable[None]]] = field(default=None, repr=False)


class ImageQueueService:
    """
    Manages image generation queue with polling and frontend notification.
    
    Usage:
        queue = ImageQueueService()
        await queue.start()
        
        task_id = await queue.add_task(
            prompt="A sunset",
            user_id="...",
            chat_id="...",
            message_id="...",
            notify_callback=async_notify_function
        )
        
        # Task will be processed, image saved, and callback invoked
    """
    
    def __init__(self):
        self.tasks: OrderedDict[str, ImageTask] = OrderedDict()
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._client = None
        
        # Ensure images directory exists
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Lazy-init image gen client"""
        if self._client is None:
            from app.services.image_gen import get_image_gen_client
            self._client = get_image_gen_client()
        return self._client
    
    async def start(self):
        """Start the queue worker"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Image queue service started")
    
    async def stop(self):
        """Stop the queue worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Image queue service stopped")
    
    @property
    def queue_size(self) -> int:
        """Current number of tasks in queue"""
        return self.queue.qsize()
    
    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity"""
        return self.queue.qsize() >= MAX_QUEUE_SIZE
    
    def get_task(self, task_id: str) -> Optional[ImageTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_user_tasks(self, user_id: str) -> list[ImageTask]:
        """Get all tasks for a user"""
        return [t for t in self.tasks.values() if t.user_id == user_id]
    
    def get_pending_count(self) -> int:
        """Count of pending/processing tasks"""
        return sum(1 for t in self.tasks.values() 
                   if t.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED, TaskStatus.PROCESSING))
    
    async def add_task(
        self,
        prompt: str,
        user_id: str,
        chat_id: str,
        message_id: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        notify_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        """
        Add a new image generation task to the queue.
        
        Returns:
            Task ID
            
        Raises:
            RuntimeError: If queue is full
        """
        if self.is_full:
            raise RuntimeError(f"Image queue is full (max {MAX_QUEUE_SIZE} tasks)")
        
        task_id = str(uuid.uuid4())
        task = ImageTask(
            id=task_id,
            prompt=prompt,
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            width=width,
            height=height,
            seed=seed,
            notify_callback=notify_callback,
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        
        logger.info(f"Added image task {task_id} to queue (position: {self.queue_size})")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def _worker(self):
        """Main worker loop - processes tasks from queue"""
        logger.info("Image queue worker started")
        
        while self._running:
            try:
                # Wait for a task (with timeout to allow checking _running)
                try:
                    task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task:
                    logger.warning(f"Task {task_id} not found, skipping")
                    continue
                
                if task.status == TaskStatus.CANCELLED:
                    logger.info(f"Task {task_id} was cancelled, skipping")
                    continue
                
                # Process the task
                await self._process_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in queue worker: {e}")
                await asyncio.sleep(1)
        
        logger.info("Image queue worker stopped")
    
    async def _process_task(self, task: ImageTask):
        """Process a single image generation task"""
        logger.info(f"Processing task {task.id}: {task.prompt[:50]}...")
        
        task.status = TaskStatus.SUBMITTED
        task.started_at = datetime.now(timezone.utc)
        
        try:
            # Submit async generation request
            job_id = await self._submit_generation(task)
            if not job_id:
                task.status = TaskStatus.FAILED
                task.error = "Failed to submit generation request"
                return
            
            task.job_id = job_id
            task.status = TaskStatus.PROCESSING
            logger.info(f"Task {task.id} submitted, job_id={job_id}")
            
            # Poll for completion
            result = await self._poll_until_complete(task)
            
            if result and result.get("success"):
                # Save image to disk
                await self._save_image(task, result)
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                task.generation_time = result.get("generation_time")
                logger.info(f"Task {task.id} completed: {task.image_url}")
            else:
                task.status = TaskStatus.FAILED
                task.error = result.get("error", "Unknown error") if result else "No result"
                logger.error(f"Task {task.id} failed: {task.error}")
            
            # Notify frontend
            if task.notify_callback:
                try:
                    await task.notify_callback()
                except Exception as e:
                    logger.error(f"Failed to notify frontend for task {task.id}: {e}")
            
        except Exception as e:
            logger.exception(f"Error processing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
    
    async def _submit_generation(self, task: ImageTask) -> Optional[str]:
        """Submit generation request to image service (async endpoint)"""
        try:
            from app.services.image_gen import extract_image_prompt
            
            clean_prompt = extract_image_prompt(task.prompt)
            payload = {
                "prompt": clean_prompt,
                "original_prompt": task.prompt,
                "width": task.width,
                "height": task.height,
            }
            if task.seed is not None:
                payload["seed"] = task.seed
            
            # Use async endpoint
            response = await self.client.client.post("/generate/async", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("job_id")
            
        except Exception as e:
            logger.error(f"Failed to submit generation for task {task.id}: {e}")
            return None
    
    async def _poll_until_complete(self, task: ImageTask) -> Optional[Dict[str, Any]]:
        """Poll job status until complete or failed"""
        if not task.job_id:
            return None
        
        max_polls = 300  # 10 minutes at 2s intervals
        polls = 0
        
        while polls < max_polls:
            try:
                response = await self.client.client.get(f"/job/{task.job_id}")
                response.raise_for_status()
                
                status_data = response.json()
                status = status_data.get("status")
                
                logger.debug(f"Task {task.id} poll {polls}: status={status}")
                
                if status == "completed":
                    # Fetch the result
                    result_response = await self.client.client.get(f"/job/{task.job_id}/result")
                    result_response.raise_for_status()
                    return result_response.json()
                
                elif status == "failed":
                    return {"success": False, "error": status_data.get("error", "Generation failed")}
                
                elif status in ("pending", "processing"):
                    await asyncio.sleep(POLL_INTERVAL)
                    polls += 1
                else:
                    logger.warning(f"Unknown job status: {status}")
                    await asyncio.sleep(POLL_INTERVAL)
                    polls += 1
                    
            except Exception as e:
                logger.error(f"Error polling task {task.id}: {e}")
                await asyncio.sleep(POLL_INTERVAL)
                polls += 1
        
        return {"success": False, "error": "Polling timeout"}
    
    async def _save_image(self, task: ImageTask, result: Dict[str, Any]):
        """Save image to disk and set URL"""
        image_base64 = result.get("image_base64")
        if not image_base64:
            raise ValueError("No image data in result")
        
        # Use job_id or task_id for filename
        filename = f"{task.job_id or task.id}.png"
        image_path = IMAGES_DIR / filename
        
        # Decode and save
        image_bytes = base64.b64decode(image_base64)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        task.image_path = str(image_path)
        task.image_url = f"/api/images/generated/{filename}"
        
        # Store additional result data
        task.seed = result.get("seed", task.seed)
        task.width = result.get("width", task.width)
        task.height = result.get("height", task.height)
        
        logger.info(f"Saved image for task {task.id} to {image_path}")
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result for frontend"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.id,
            "status": task.status.value,
            "job_id": task.job_id,
            "prompt": task.prompt,
            "width": task.width,
            "height": task.height,
            "seed": task.seed,
            "image_url": task.image_url,
            "error": task.error,
            "generation_time": task.generation_time,
            "queue_position": self._get_queue_position(task_id),
        }
    
    def _get_queue_position(self, task_id: str) -> Optional[int]:
        """Get task's position in queue (1-based)"""
        position = 1
        for tid, task in self.tasks.items():
            if task.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED, TaskStatus.PROCESSING):
                if tid == task_id:
                    return position
                position += 1
        return None
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove completed/failed tasks older than max_age_hours"""
        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=max_age_hours)
        
        to_remove = []
        for task_id, task in self.tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(task_id)
                elif task.created_at < cutoff:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old image tasks")


# Singleton instance
_queue_service: Optional[ImageQueueService] = None


def get_image_queue() -> ImageQueueService:
    """Get or create the image queue service singleton"""
    global _queue_service
    if _queue_service is None:
        _queue_service = ImageQueueService()
    return _queue_service


async def ensure_queue_started():
    """Ensure the queue service is running"""
    queue = get_image_queue()
    await queue.start()
    return queue
