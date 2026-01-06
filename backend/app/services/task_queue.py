"""
Agentic Task Queue Service

Manages a FIFO queue of tasks that can be added by LLM or User.
Tasks are processed automatically after each LLM response.

Key Design Decisions:
1. NO separate verification LLM call - wasteful and unreliable
2. Tasks auto-advance when LLM calls `complete_task` tool
3. LLM can work on multiple things per turn naturally
4. "auto_continue" flag lets tasks chain without user input
5. Queue visible in system prompt so LLM knows what's pending
6. Frontend gets real-time updates via WebSocket

Task Structure:
{
    "id": "uuid",
    "description": "Short task description",
    "instructions": "Detailed instructions (up to 512 tokens)",
    "status": "queued|in_progress|completed|failed|paused",
    "auto_continue": true,  # Auto-start next task when this completes
    "priority": 0,  # Higher = more urgent (0 is normal)
    "created_at": "ISO datetime",
    "completed_at": "ISO datetime or null",
    "result_summary": "Brief summary of what was done",
    "source": "user|llm"
}
"""
import logging
import json
import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.models import Chat, Message, UploadedFile
from app.models.base import MessageRole, ContentType
from app.services.agent_memory import estimate_tokens, AGENT_FILE_PREFIX, AGENT_FILE_SUFFIX

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"  # User paused the queue


class TaskSource(str, Enum):
    USER = "user"
    LLM = "llm"


@dataclass
class AgentTask:
    """A single task in the queue"""
    id: str
    description: str  # Short description for display
    instructions: str  # Detailed instructions (up to 512 tokens)
    status: TaskStatus = TaskStatus.QUEUED
    auto_continue: bool = True  # Auto-start next task when this completes
    priority: int = 0  # Higher = more urgent
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    result_summary: Optional[str] = None  # Brief summary when completed
    source: TaskSource = TaskSource.USER
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "instructions": self.instructions,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "auto_continue": self.auto_continue,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
            "source": self.source.value if isinstance(self.source, TaskSource) else self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTask":
        return cls(
            id=data["id"],
            description=data["description"],
            instructions=data["instructions"],
            status=TaskStatus(data.get("status", "queued")),
            auto_continue=data.get("auto_continue", True),
            priority=data.get("priority", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=data.get("completed_at"),
            result_summary=data.get("result_summary"),
            source=TaskSource(data.get("source", "user")),
        )


# Task log file prefix - stored similarly to Agent files
TASK_LOG_PREFIX = "{TaskLog"
TASK_LOG_SUFFIX = "}.md"

# Max tokens for instructions
MAX_INSTRUCTION_TOKENS = 512


class TaskQueueService:
    """
    Manages the agentic task queue for a chat session.
    
    The queue is stored in the Chat's metadata and persisted to the database.
    Task execution logs overflow to TaskLog*.md files.
    
    Usage:
        queue = TaskQueueService(db, chat_id)
        await queue.add_task("Write tests", "Create unit tests for the auth module")
        
        # LLM sees tasks in system prompt
        # When LLM finishes a task, it calls complete_task tool
        # Queue auto-advances if auto_continue=True
    """
    
    def __init__(self, db: AsyncSession, chat_id: str):
        self.db = db
        self.chat_id = chat_id
        self._queue: List[AgentTask] = []
        self._current_task: Optional[AgentTask] = None
        self._completed_tasks: List[AgentTask] = []  # History of completed tasks
        self._task_log: List[str] = []  # Human-readable log entries
        self._paused: bool = False  # Queue paused by user
        self._loaded = False
    
    async def _load_queue(self) -> None:
        """Load queue state from chat metadata"""
        if self._loaded:
            return
        
        result = await self.db.execute(
            select(Chat).where(Chat.id == self.chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if chat and chat.chat_metadata:
            queue_data = chat.chat_metadata.get("task_queue", {})
            
            # Load queue
            queue_list = queue_data.get("queue", [])
            self._queue = [AgentTask.from_dict(t) for t in queue_list]
            
            # Load current task
            current = queue_data.get("current_task")
            self._current_task = AgentTask.from_dict(current) if current else None
            
            # Load completed tasks (recent history)
            completed_list = queue_data.get("completed", [])
            self._completed_tasks = [AgentTask.from_dict(t) for t in completed_list]
            
            # Load state
            self._paused = queue_data.get("paused", False)
            self._task_log = queue_data.get("log", [])
        
        self._loaded = True
        logger.debug(f"[TASK_QUEUE] Loaded queue for chat {self.chat_id}: {len(self._queue)} queued, current={self._current_task is not None}, paused={self._paused}")
    
    async def _save_queue(self) -> None:
        """Save queue state to chat metadata"""
        result = await self.db.execute(
            select(Chat).where(Chat.id == self.chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            logger.error(f"[TASK_QUEUE] Chat {self.chat_id} not found")
            return
        
        # Get existing metadata or create new
        metadata = chat.chat_metadata or {}
        
        # Update task queue data
        metadata["task_queue"] = {
            "queue": [t.to_dict() for t in self._queue],
            "current_task": self._current_task.to_dict() if self._current_task else None,
            "completed": [t.to_dict() for t in self._completed_tasks[-20:]],  # Keep last 20 completed
            "paused": self._paused,
            "log": self._task_log[-100:],  # Keep last 100 log entries
        }
        
        chat.chat_metadata = metadata
        await self.db.commit()
        
        logger.debug(f"[TASK_QUEUE] Saved queue for chat {self.chat_id}")
    
    def _truncate_instructions(self, instructions: str) -> str:
        """Truncate instructions to max token limit"""
        tokens = estimate_tokens(instructions)
        if tokens <= MAX_INSTRUCTION_TOKENS:
            return instructions
        
        # Rough truncation - 4 chars per token
        max_chars = MAX_INSTRUCTION_TOKENS * 4
        return instructions[:max_chars] + "..."
    
    def _log(self, message: str) -> None:
        """Add to human-readable task log"""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._task_log.append(entry)
        logger.info(f"[TASK_QUEUE] {message}")
    
    def _sort_queue(self) -> None:
        """Sort queue by priority (higher first), then by creation time (older first)"""
        self._queue.sort(key=lambda t: (-t.priority, t.created_at))
    
    async def add_task(
        self,
        description: str,
        instructions: str,
        source: TaskSource = TaskSource.USER,
        auto_continue: bool = True,
        priority: int = 0,
    ) -> AgentTask:
        """
        Add a new task to the queue.
        
        Args:
            description: Short task description
            instructions: Detailed instructions (truncated to 512 tokens)
            source: Whether task came from user or LLM
            auto_continue: Auto-start next task when this completes
            priority: Higher = more urgent (0 is normal)
            
        Returns:
            The created task
        """
        await self._load_queue()
        
        task = AgentTask(
            id=str(uuid.uuid4()),
            description=description,
            instructions=self._truncate_instructions(instructions),
            source=source,
            auto_continue=auto_continue,
            priority=priority,
        )
        
        self._queue.append(task)
        self._sort_queue()
        self._log(f"Task added: \"{description}\" (priority: {priority}, auto_continue: {auto_continue})")
        
        # Auto-start if no current task and not paused
        if not self._current_task and not self._paused:
            await self._start_next_task()
        
        await self._save_queue()
        return task
    
    async def add_tasks_batch(
        self,
        tasks: List[Dict[str, Any]],
        source: TaskSource = TaskSource.LLM,
    ) -> List[AgentTask]:
        """
        Add multiple tasks at once (typically from LLM planning).
        
        Args:
            tasks: List of task dicts with "description", "instructions", 
                   optional "priority" and "auto_continue"
            source: Source of the tasks
            
        Returns:
            List of created tasks
        """
        await self._load_queue()
        
        created = []
        for task_data in tasks:
            task = AgentTask(
                id=str(uuid.uuid4()),
                description=task_data.get("description", "Unnamed task"),
                instructions=self._truncate_instructions(task_data.get("instructions", "")),
                source=source,
                auto_continue=task_data.get("auto_continue", True),
                priority=task_data.get("priority", 0),
            )
            self._queue.append(task)
            created.append(task)
        
        self._sort_queue()
        self._log(f"Batch added: {len(created)} tasks")
        
        # Auto-start if no current task and not paused
        if not self._current_task and not self._paused and created:
            await self._start_next_task()
        
        await self._save_queue()
        return created
    
    async def _start_next_task(self) -> Optional[AgentTask]:
        """Internal: Start the next task from queue"""
        if not self._queue:
            return None
        
        task = self._queue.pop(0)
        task.status = TaskStatus.IN_PROGRESS
        self._current_task = task
        self._log(f"Started: \"{task.description}\"")
        return task
    
    async def get_next_task(self) -> Optional[AgentTask]:
        """
        Get the next task from the queue.
        Used externally to check what's next without starting it.
        """
        await self._load_queue()
        
        if not self._queue:
            return None
        
        return self._queue[0]  # Peek, don't pop
    
    async def get_current_task(self) -> Optional[AgentTask]:
        """Get the currently executing task"""
        await self._load_queue()
        return self._current_task
    
    async def complete_task(self, result_summary: Optional[str] = None) -> Tuple[Optional[AgentTask], Optional[AgentTask]]:
        """
        Mark the current task as completed.
        If auto_continue, starts the next task.
        
        Args:
            result_summary: Brief summary of what was accomplished
            
        Returns:
            Tuple of (completed_task, next_task or None)
        """
        await self._load_queue()
        
        if not self._current_task:
            return None, None
        
        # Complete current task
        task = self._current_task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result_summary = result_summary
        
        self._completed_tasks.append(task)
        self._log(f"Completed: \"{task.description}\"" + (f" - {result_summary}" if result_summary else ""))
        
        self._current_task = None
        
        # Auto-continue to next task if enabled and not paused
        next_task = None
        if task.auto_continue and not self._paused and self._queue:
            next_task = await self._start_next_task()
        
        await self._save_queue()
        return task, next_task
    
    async def fail_task(self, reason: str = "") -> Optional[AgentTask]:
        """
        Mark the current task as failed.
        Does NOT auto-retry - user/LLM decides what to do.
        """
        await self._load_queue()
        
        if not self._current_task:
            return None
        
        task = self._current_task
        task.status = TaskStatus.FAILED
        task.result_summary = f"Failed: {reason}" if reason else "Failed"
        
        self._completed_tasks.append(task)
        self._log(f"Failed: \"{task.description}\"" + (f" - {reason}" if reason else ""))
        
        self._current_task = None
        await self._save_queue()
        
        return task
    
    async def skip_task(self) -> Tuple[Optional[AgentTask], Optional[AgentTask]]:
        """
        Skip the current task and move to next.
        
        Returns:
            Tuple of (skipped_task, next_task or None)
        """
        await self._load_queue()
        
        if not self._current_task:
            return None, None
        
        task = self._current_task
        task.status = TaskStatus.COMPLETED
        task.result_summary = "Skipped"
        
        self._completed_tasks.append(task)
        self._log(f"Skipped: \"{task.description}\"")
        
        self._current_task = None
        
        # Start next if not paused
        next_task = None
        if not self._paused and self._queue:
            next_task = await self._start_next_task()
        
        await self._save_queue()
        return task, next_task
    
    async def pause_queue(self) -> None:
        """Pause task queue execution (user control)"""
        await self._load_queue()
        self._paused = True
        self._log("Queue paused")
        await self._save_queue()
    
    async def resume_queue(self) -> Optional[AgentTask]:
        """
        Resume task queue execution.
        Returns the task that was started (if any).
        """
        await self._load_queue()
        self._paused = False
        self._log("Queue resumed")
        
        # Start next task if none in progress
        next_task = None
        if not self._current_task and self._queue:
            next_task = await self._start_next_task()
        
        await self._save_queue()
        return next_task
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        await self._load_queue()
        
        return {
            "queue_length": len(self._queue),
            "current_task": self._current_task.to_dict() if self._current_task else None,
            "queued_tasks": [t.to_dict() for t in self._queue],
            "completed_count": len(self._completed_tasks),
            "recent_completed": [t.to_dict() for t in self._completed_tasks[-5:]],
            "has_pending": len(self._queue) > 0 or self._current_task is not None,
            "paused": self._paused,
        }
    
    async def clear_queue(self) -> int:
        """Clear all queued tasks. Returns number cleared."""
        await self._load_queue()
        
        count = len(self._queue)
        if self._current_task:
            count += 1
        
        self._queue = []
        self._current_task = None
        self._log(f"Queue cleared: {count} tasks removed")
        
        await self._save_queue()
        return count
    
    async def get_task_log(self) -> List[str]:
        """Get human-readable task log"""
        await self._load_queue()
        return self._task_log.copy()
    
    def get_system_prompt_addition(self, include_capability_notice: bool = True) -> str:
        """
        Generate system prompt addition describing pending tasks.
        Call after _load_queue().
        
        Args:
            include_capability_notice: If True, always include a brief note about
                                       task queue capability even when empty
        """
        has_tasks = bool(self._queue or self._current_task or self._completed_tasks)
        
        # If no tasks and no capability notice needed, return empty
        if not has_tasks and not include_capability_notice:
            return ""
        
        parts = []
        parts.append("\n\n--- AGENTIC TASK QUEUE ---")
        
        if self._paused:
            parts.append("\n⏸️ **QUEUE PAUSED** - User has paused task execution.")
            parts.append("Tasks are shown for reference but do not auto-continue.")
        
        if self._current_task:
            parts.append(f"\n🔄 **CURRENT TASK**: {self._current_task.description}")
            if self._current_task.instructions:
                parts.append(f"\nInstructions:\n{self._current_task.instructions}")
            parts.append("\n\nWhen you complete this task, call the `complete_task` tool with a brief summary.")
            parts.append("If you cannot complete it, call `fail_task` with the reason.")
        
        if self._queue:
            parts.append(f"\n\n📋 **QUEUED** ({len(self._queue)} tasks):")
            for i, task in enumerate(self._queue[:5], 1):
                priority_marker = "⚡" if task.priority > 0 else ""
                parts.append(f"  {i}. {priority_marker}{task.description}")
            if len(self._queue) > 5:
                parts.append(f"  ... and {len(self._queue) - 5} more")
        
        if self._completed_tasks:
            recent = self._completed_tasks[-3:]
            parts.append(f"\n\n✅ **RECENTLY COMPLETED** ({len(self._completed_tasks)} total):")
            for task in reversed(recent):
                summary = f" - {task.result_summary}" if task.result_summary else ""
                parts.append(f"  • {task.description}{summary}")
        
        # Always include capability notice if requested and no current tasks
        if not has_tasks and include_capability_notice:
            parts.append("\n📋 **TASK QUEUE AVAILABLE**")
            parts.append("For complex multi-step work, you have task queue tools available:")
            parts.append("• add_task / add_tasks_batch - Queue tasks with descriptions and instructions")
            parts.append("• complete_task / fail_task / skip_task - Mark task progress")
            parts.append("• pause_task_queue / resume_task_queue - Control execution")
            parts.append("Tasks will appear here with instructions when queued.")
        
        parts.append("\n--- END TASK QUEUE ---")
        
        return "\n".join(parts)
    
    async def archive_log_overflow(self) -> Optional[str]:
        """
        Archive task log to TaskLog*.md file if it's getting large.
        Returns filename if archived, None otherwise.
        """
        await self._load_queue()
        
        if len(self._task_log) < 50:
            return None
        
        # Get next file number
        result = await self.db.execute(
            select(UploadedFile.filepath)
            .where(UploadedFile.chat_id == self.chat_id)
            .where(UploadedFile.filepath.like(f"{TASK_LOG_PREFIX}%"))
        )
        existing = result.scalars().all()
        
        max_num = 0
        import re
        for fp in existing:
            match = re.search(r'\{TaskLog(\d+)\}\.md', fp)
            if match:
                max_num = max(max_num, int(match.group(1)))
        
        filename = f"{TASK_LOG_PREFIX}{max_num + 1:04d}{TASK_LOG_SUFFIX}"
        
        # Build content
        content = f"""# Task Queue Log Archive
**Chat ID**: {self.chat_id}
**Archived**: {datetime.now(timezone.utc).isoformat()}
**Entries**: {len(self._task_log)}

---

"""
        content += "\n".join(self._task_log)
        
        # Store file
        log_file = UploadedFile(
            chat_id=self.chat_id,
            archive_name=None,
            filepath=filename,
            filename=filename,
            extension=".md",
            language="markdown",
            size=len(content),
            is_binary=False,
            content=content,
            signatures=None,
        )
        self.db.add(log_file)
        
        # Clear log (keep last 10 entries)
        self._task_log = self._task_log[-10:]
        
        await self._save_queue()
        await self.db.commit()
        
        logger.info(f"[TASK_QUEUE] Archived log to {filename}")
        return filename


# Helper function for parsing tasks from LLM responses (used by tools)
def parse_tasks_from_llm_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse task definitions from LLM response.
    
    Supports formats:
    - JSON array: [{"description": "...", "instructions": "...", "priority": 0}, ...]
    - XML-style: <task description="..." instructions="..."/>
    - Markdown list with TASK: prefix
    """
    import re
    tasks = []
    
    # Try JSON array first (most reliable)
    try:
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "description" in item:
                        tasks.append({
                            "description": item.get("description", ""),
                            "instructions": item.get("instructions", ""),
                            "priority": item.get("priority", 0),
                            "auto_continue": item.get("auto_continue", True),
                        })
    except json.JSONDecodeError:
        pass
    
    if tasks:
        return tasks
    
    # Try XML-style tags
    xml_pattern = r'<task\s+description="([^"]+)"(?:\s+instructions="([^"]*)")?(?:\s+priority="(\d+)")?[^/]*/>'
    for match in re.finditer(xml_pattern, response, re.IGNORECASE | re.DOTALL):
        tasks.append({
            "description": match.group(1),
            "instructions": match.group(2) or "",
            "priority": int(match.group(3)) if match.group(3) else 0,
        })
    
    if tasks:
        return tasks
    
    # Try markdown list format
    lines = response.split('\n')
    current_task = {}
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith("task:") or line.startswith("- "):
            if current_task.get("description"):
                tasks.append(current_task)
            desc = line[5:].strip() if line.lower().startswith("task:") else line[2:].strip()
            current_task = {"description": desc, "instructions": "", "priority": 0}
        elif line.lower().startswith("instructions:") and current_task.get("description"):
            current_task["instructions"] = line[13:].strip()
    
    if current_task.get("description"):
        tasks.append(current_task)
    
    return tasks


# Export
__all__ = [
    'TaskQueueService',
    'AgentTask',
    'TaskStatus',
    'TaskSource',
    'parse_tasks_from_llm_response',
]
