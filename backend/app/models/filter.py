"""
Chat filter models for message processing pipelines

Contains:
- ChatFilter: Configurable message processing rules
"""
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, 
    ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid, FilterType, FilterPriority


class ChatFilter(Base):
    """
    Database-backed filter configurations.
    
    Filters can process messages at different points in the pipeline:
    - TO_LLM: Before user messages go to the LLM
    - FROM_LLM: Before LLM responses go to the user
    - TO_TOOLS: Before messages are sent to tools
    - FROM_TOOLS: Before tool results are processed
    
    Filter modes:
    - pattern: Regex or word list matching
    - code: Custom Python code (admin only, sandboxed)
    - llm: LLM-based processing
    
    Actions on match:
    - modify: Transform the content
    - block: Reject with error message
    - log: Log and pass through
    - passthrough: No modification
    """
    __tablename__ = "chat_filters"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Basic info
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    filter_type = Column(Enum(FilterType), nullable=False)
    
    # Filter behavior
    priority = Column(Enum(FilterPriority), default=FilterPriority.MEDIUM)
    enabled = Column(Boolean, default=True)
    
    # Filter code/configuration
    filter_mode = Column(String(50), default="pattern")  # pattern, code, llm
    
    # Pattern mode: regex or word matching
    pattern = Column(Text, nullable=True)  # Regex pattern
    replacement = Column(Text, nullable=True)  # Replacement text (for regex)
    word_list = Column(JSON, nullable=True)  # List of words to match/block
    case_sensitive = Column(Boolean, default=False)
    
    # Action: what to do when matched
    action = Column(String(50), default="modify")  # modify, block, log, passthrough
    block_message = Column(Text, nullable=True)  # Message shown when blocked
    
    # Code mode: custom Python code (admin only, sandboxed)
    code = Column(Text, nullable=True)  # Python code for custom logic
    
    # LLM mode: use LLM to process/filter content
    llm_prompt = Column(Text, nullable=True)  # Prompt for LLM-based filtering
    
    # Configuration
    config = Column(JSON, nullable=True)  # Additional configuration
    
    # Ownership
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_global = Column(Boolean, default=False)  # Available to all users
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    owner = relationship("User")
    
    __table_args__ = (
        Index("idx_filter_type", "filter_type"),
        Index("idx_filter_enabled", "enabled"),
        Index("idx_filter_global", "is_global"),
    )
    
    @property
    def priority_value(self) -> int:
        """Get numeric priority value for sorting"""
        priority_map = {
            FilterPriority.HIGHEST: 0,
            FilterPriority.HIGH: 25,
            FilterPriority.MEDIUM: 50,
            FilterPriority.LOW: 75,
            FilterPriority.LEAST: 100,
        }
        return priority_map.get(self.priority, 50)
