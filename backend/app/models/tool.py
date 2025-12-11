"""
Tool integration models

Contains:
- Tool: External tool definitions (MCP/OpenAPI)
- ToolUsage: Invocation tracking for citations
"""
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, 
    ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid, ToolType


class Tool(Base):
    """
    External tool integration (MCP servers or OpenAPI endpoints).
    
    Features:
    - MCP or OpenAPI protocol support
    - Encrypted API key storage
    - Schema caching for discovery
    - Access control (public/private, enabled/disabled)
    """
    __tablename__ = "tools"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    tool_type = Column(Enum(ToolType), nullable=False)
    
    # Connection details
    url = Column(String(500), nullable=False)  # MCP server URL or OpenAPI spec URL
    api_key_encrypted = Column(Text, nullable=True)  # Encrypted API key
    
    # Access control
    is_public = Column(Boolean, default=False)  # Visible to all users
    is_enabled = Column(Boolean, default=True)  # Can be used
    
    # Tool metadata (cached from MCP/OpenAPI discovery)
    schema_cache = Column(JSON, nullable=True)  # Cached tool schemas/operations
    last_schema_fetch = Column(DateTime, nullable=True)
    
    # For OpenAPI: specific operations to expose (null = all)
    enabled_operations = Column(JSON, nullable=True)  # List of operation IDs
    
    # Additional config
    config = Column(JSON, nullable=True)  # Headers, auth type, etc.
    
    # Ownership
    created_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    creator = relationship("User")
    
    __table_args__ = (
        Index("idx_tool_type", "tool_type"),
        Index("idx_tool_public", "is_public", "is_enabled"),
    )


class ToolUsage(Base):
    """
    Track tool invocations for citations and debugging.
    
    Records:
    - Tool and operation called
    - Input parameters
    - Success/failure status
    - Timing information
    """
    __tablename__ = "tool_usage"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    tool_id = Column(String(36), ForeignKey("tools.id", ondelete="SET NULL"), nullable=True)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Invocation details
    tool_name = Column(String(100), nullable=False)  # Preserved even if tool deleted
    operation = Column(String(100), nullable=True)  # For OpenAPI: operationId
    input_params = Column(JSON, nullable=True)
    
    # Result
    success = Column(Boolean, default=True)
    result_summary = Column(Text, nullable=True)  # Brief summary for citation
    result_url = Column(String(500), nullable=True)  # If tool returned a URL
    error_message = Column(Text, nullable=True)
    
    # Timing
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    tool = relationship("Tool")
    message = relationship("Message")
    chat = relationship("Chat")
    user = relationship("User")
    
    __table_args__ = (
        Index("idx_tool_usage_chat", "chat_id"),
        Index("idx_tool_usage_message", "message_id"),
    )
