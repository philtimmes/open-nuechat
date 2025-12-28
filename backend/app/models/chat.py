"""
Chat and message models

Contains:
- Chat: Conversation container
- Message: Individual messages with branching support
- ChatParticipant: Multi-user chat support
"""
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, 
    ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid, MessageRole, ContentType


class Chat(Base):
    """
    A conversation between a user and the AI.
    
    Features:
    - Configurable model and system prompt
    - Token tracking per conversation
    - Branch selection for conversation trees
    - Public sharing via share_id
    - Code context from uploaded zip files
    """
    __tablename__ = "chats"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), default="New Chat")
    model = Column(String(100), default="claude-sonnet-4-20250514")
    system_prompt = Column(Text, nullable=True)
    
    # Custom GPT / Assistant association (no FK constraint to avoid circular dependency)
    assistant_id = Column(String(36), nullable=True)
    assistant_name = Column(String(255), nullable=True)  # Denormalized for display
    
    # Chat type
    is_shared = Column(Boolean, default=False)  # For client-to-client chat
    
    # Public sharing
    share_id = Column(String(36), nullable=True, unique=True, index=True)
    share_anonymous = Column(Boolean, default=False)  # If True, hide owner name in shared view
    
    # Token tracking
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    
    # Branch selection - tracks which child is selected at each branch point
    # Format: { "parent_message_id": "selected_child_id" }
    selected_versions = Column(JSON, default=dict)
    
    # Code summary - tracks files and signatures for LLM code generation
    # Format: { "files": [...], "warnings": [...], "last_updated": "...", "auto_generated": true }
    code_summary = Column(JSON, nullable=True)
    
    # Knowledge base indexing
    is_knowledge_indexed = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at")
    participants = relationship("ChatParticipant", back_populates="chat", cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="chat", cascade="all, delete-orphan")
    uploaded_archive = relationship("UploadedArchive", back_populates="chat", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_chat_owner_updated", "owner_id", "updated_at"),
    )
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this chat"""
        return self.total_input_tokens + self.total_output_tokens


class Message(Base):
    """
    A single message in a conversation.
    
    Supports:
    - Role-based messages (user, assistant, system, tool)
    - Conversation branching via parent_id
    - Multimodal attachments
    - Tool calls and results
    - Token tracking per message
    """
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    parent_id = Column(String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=True)
    content_type = Column(Enum(ContentType), default=ContentType.TEXT)
    
    # For multimodal content
    attachments = Column(JSON, default=list)  # [{type, url, name, size, mime_type}]
    
    # Extracted code artifacts with timestamps
    # [{id, title, type, language, content, filename, created_at}]
    artifacts = Column(JSON, nullable=True)
    
    # For tool calls
    tool_calls = Column(JSON, nullable=True)  # [{id, name, arguments}]
    tool_call_id = Column(String(100), nullable=True)  # For tool results
    
    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    
    # Timing metrics (milliseconds)
    time_to_first_token = Column(Integer, nullable=True, default=None)  # Time from request to first token
    time_to_complete = Column(Integer, nullable=True, default=None)  # Time from request to completion
    
    # Metadata
    model = Column(String(100), nullable=True)
    message_metadata = Column(JSON, default=dict)
    
    # Status
    is_streaming = Column(Boolean, default=False)
    is_error = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    
    chat = relationship("Chat", back_populates="messages")
    sender = relationship("User")
    parent = relationship("Message", remote_side=[id], backref="children")
    
    __table_args__ = (
        Index("idx_message_chat_created", "chat_id", "created_at"),
        Index("idx_message_parent", "parent_id"),
    )
    
    @property
    def total_tokens(self) -> int:
        """Total tokens for this message"""
        return self.input_tokens + self.output_tokens
    
    @property
    def has_siblings(self) -> bool:
        """Check if this message has sibling branches"""
        return len(self.children) > 1 if hasattr(self, 'children') else False


class ChatParticipant(Base):
    """
    Participant in a multi-user chat.
    
    Supports different roles (owner, admin, member) for
    permission-based access control.
    """
    __tablename__ = "chat_participants"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), default="member")  # owner, admin, member
    joined_at = Column(DateTime, default=func.now())
    last_read_at = Column(DateTime, nullable=True)
    
    chat = relationship("Chat", back_populates="participants")
    user = relationship("User")
    
    __table_args__ = (
        Index("idx_participant_chat_user", "chat_id", "user_id", unique=True),
    )
