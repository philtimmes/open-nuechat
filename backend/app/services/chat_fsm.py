"""
Chat Session Finite State Machine

Manages chat processing states to ensure:
1. Database transactions only occur in IDLE state
2. Intermediate state is kept in memory during processing
3. Frontend receives clear "ready" signals
4. File request continuations are handled atomically
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Awaitable
from uuid import uuid4

logger = logging.getLogger(__name__)


class ChatState(Enum):
    """Chat processing states"""
    IDLE = auto()           # Ready for new input
    STREAMING = auto()      # LLM is generating response
    AWAITING_FILES = auto() # Waiting for file content from frontend
    COMMITTING = auto()     # Persisting to database
    ERROR = auto()          # Error state, needs recovery


@dataclass
class PendingMessage:
    """Message data held in memory until commit"""
    id: str
    chat_id: str
    role: str  # 'user' or 'assistant'
    content: str
    parent_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "parent_id": self.parent_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "tool_calls": self.tool_calls,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChatSession:
    """
    State machine for a single chat processing session.
    
    Holds all intermediate state in memory until processing completes,
    then commits everything atomically.
    """
    chat_id: str
    user_id: str
    state: ChatState = ChatState.IDLE
    
    # Pending messages (not yet committed)
    pending_messages: List[PendingMessage] = field(default_factory=list)
    
    # Current streaming state
    current_message_id: Optional[str] = None
    streaming_content: str = ""
    
    # File request tracking
    pending_file_requests: List[str] = field(default_factory=list)
    file_contents: Dict[str, str] = field(default_factory=dict)
    
    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Error state
    error: Optional[str] = None
    
    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def can_accept_input(self) -> bool:
        """Check if session can accept new user input"""
        return self.state == ChatState.IDLE
    
    def can_accept_file_content(self) -> bool:
        """Check if session is waiting for file content"""
        return self.state == ChatState.AWAITING_FILES
    
    async def start_streaming(self, user_content: str, parent_id: Optional[str] = None) -> str:
        """
        Start a new streaming session.
        Creates pending user and assistant messages.
        Returns the assistant message ID.
        """
        async with self._lock:
            if self.state != ChatState.IDLE:
                raise ValueError(f"Cannot start streaming in state {self.state}")
            
            self.state = ChatState.STREAMING
            self.streaming_content = ""
            self.pending_file_requests = []
            self.file_contents = {}
            
            # Create pending user message
            user_msg_id = str(uuid4())
            user_msg = PendingMessage(
                id=user_msg_id,
                chat_id=self.chat_id,
                role="user",
                content=user_content,
                parent_id=parent_id,
            )
            self.pending_messages.append(user_msg)
            
            # Create pending assistant message
            assistant_msg_id = str(uuid4())
            self.current_message_id = assistant_msg_id
            assistant_msg = PendingMessage(
                id=assistant_msg_id,
                chat_id=self.chat_id,
                role="assistant",
                content="",  # Will be filled during streaming
                parent_id=user_msg_id,
            )
            self.pending_messages.append(assistant_msg)
            
            logger.info(f"[FSM] Chat {self.chat_id}: IDLE -> STREAMING, msg={assistant_msg_id}")
            return assistant_msg_id
    
    async def append_content(self, content: str):
        """Append content to the current streaming message"""
        async with self._lock:
            if self.state != ChatState.STREAMING:
                logger.warning(f"[FSM] Ignoring content in state {self.state}")
                return
            
            self.streaming_content += content
            
            # Update the pending assistant message
            if self.pending_messages and self.pending_messages[-1].role == "assistant":
                self.pending_messages[-1].content = self.streaming_content
    
    async def complete_streaming(
        self,
        input_tokens: int,
        output_tokens: int,
        file_requests: Optional[List[str]] = None,
    ) -> bool:
        """
        Complete the streaming phase.
        
        Returns True if ready to commit (no file requests).
        Returns False if waiting for file content.
        """
        async with self._lock:
            if self.state != ChatState.STREAMING:
                logger.warning(f"[FSM] Cannot complete streaming in state {self.state}")
                return False
            
            # Update token counts
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # Update the pending assistant message
            if self.pending_messages and self.pending_messages[-1].role == "assistant":
                msg = self.pending_messages[-1]
                msg.content = self.streaming_content
                msg.input_tokens = input_tokens
                msg.output_tokens = output_tokens
            
            # Check for file requests
            if file_requests:
                self.pending_file_requests = file_requests
                self.state = ChatState.AWAITING_FILES
                logger.info(f"[FSM] Chat {self.chat_id}: STREAMING -> AWAITING_FILES, files={file_requests}")
                return False
            else:
                self.state = ChatState.COMMITTING
                logger.info(f"[FSM] Chat {self.chat_id}: STREAMING -> COMMITTING")
                return True
    
    async def add_file_content(self, path: str, content: str) -> bool:
        """
        Add file content for a pending request.
        
        Returns True when all files have been received.
        """
        async with self._lock:
            if self.state != ChatState.AWAITING_FILES:
                logger.warning(f"[FSM] Cannot add file content in state {self.state}")
                return False
            
            self.file_contents[path] = content
            
            # Check if all files received
            received = set(self.file_contents.keys())
            requested = set(self.pending_file_requests)
            
            if received >= requested:
                logger.info(f"[FSM] Chat {self.chat_id}: All files received, continuing")
                return True
            
            return False
    
    async def continue_with_files(self) -> str:
        """
        Continue streaming with the received file content.
        Returns the combined file content to send to LLM.
        """
        async with self._lock:
            if self.state != ChatState.AWAITING_FILES:
                raise ValueError(f"Cannot continue with files in state {self.state}")
            
            # Combine all file contents
            combined = []
            for path in self.pending_file_requests:
                content = self.file_contents.get(path, f"[ERROR: File {path} not found]")
                combined.append(f"=== {path} ===\n{content}")
            
            # Create a file content user message (for history)
            file_msg_id = str(uuid4())
            last_assistant_id = self.current_message_id
            
            file_msg = PendingMessage(
                id=file_msg_id,
                chat_id=self.chat_id,
                role="user",
                content="\n\n".join(combined),
                parent_id=last_assistant_id,
            )
            self.pending_messages.append(file_msg)
            
            # Create new assistant message for the continuation
            new_assistant_id = str(uuid4())
            self.current_message_id = new_assistant_id
            
            new_assistant_msg = PendingMessage(
                id=new_assistant_id,
                chat_id=self.chat_id,
                role="assistant",
                content="",
                parent_id=file_msg_id,
            )
            self.pending_messages.append(new_assistant_msg)
            
            # Reset for new streaming
            self.streaming_content = ""
            self.pending_file_requests = []
            self.file_contents = {}
            self.state = ChatState.STREAMING
            
            logger.info(f"[FSM] Chat {self.chat_id}: AWAITING_FILES -> STREAMING (continuation)")
            
            return "\n\n".join(combined)
    
    async def set_error(self, error: str):
        """Set error state"""
        async with self._lock:
            self.error = error
            self.state = ChatState.ERROR
            logger.error(f"[FSM] Chat {self.chat_id}: -> ERROR: {error}")
    
    async def get_pending_for_commit(self) -> List[PendingMessage]:
        """Get all pending messages for database commit"""
        async with self._lock:
            if self.state != ChatState.COMMITTING:
                raise ValueError(f"Cannot get pending messages in state {self.state}")
            return list(self.pending_messages)
    
    async def commit_complete(self):
        """Mark commit as complete, transition to IDLE"""
        async with self._lock:
            self.pending_messages = []
            self.current_message_id = None
            self.streaming_content = ""
            self.state = ChatState.IDLE
            logger.info(f"[FSM] Chat {self.chat_id}: COMMITTING -> IDLE")
    
    async def reset(self):
        """Reset to IDLE state (for error recovery)"""
        async with self._lock:
            self.pending_messages = []
            self.current_message_id = None
            self.streaming_content = ""
            self.pending_file_requests = []
            self.file_contents = {}
            self.error = None
            self.state = ChatState.IDLE
            logger.info(f"[FSM] Chat {self.chat_id}: -> IDLE (reset)")


class ChatSessionManager:
    """
    Manages ChatSession instances for all active chats.
    Thread-safe singleton.
    """
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._session_lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls) -> "ChatSessionManager":
        """Get or create the singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ChatSessionManager()
        return cls._instance
    
    async def get_session(self, chat_id: str, user_id: str) -> ChatSession:
        """Get or create a session for a chat"""
        async with self._session_lock:
            if chat_id not in self._sessions:
                self._sessions[chat_id] = ChatSession(chat_id=chat_id, user_id=user_id)
            return self._sessions[chat_id]
    
    async def remove_session(self, chat_id: str):
        """Remove a session (e.g., when chat is deleted)"""
        async with self._session_lock:
            if chat_id in self._sessions:
                del self._sessions[chat_id]
    
    async def cleanup_idle_sessions(self, max_age_seconds: int = 3600):
        """Clean up sessions that have been idle too long"""
        # TODO: Implement periodic cleanup
        pass
