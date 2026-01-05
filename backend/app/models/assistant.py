"""
Custom assistant (GPT) models

Contains:
- CustomAssistant: User-created AI assistants
- AssistantConversation: Usage tracking and ratings
- AssistantCategory: Categories for organizing assistants
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, JSON, Index, Table
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid


# Association table for Assistant <-> KnowledgeStore
assistant_knowledge_stores = Table(
    "assistant_knowledge_stores",
    Base.metadata,
    Column("assistant_id", String(36), ForeignKey("custom_assistants.id", ondelete="CASCADE")),
    Column("knowledge_store_id", String(36), ForeignKey("knowledge_stores.id", ondelete="CASCADE")),
)


class AssistantCategory(Base):
    """
    Categories for organizing Custom Assistants/GPTs.
    Admin-managed, used for filtering in the marketplace.
    """
    __tablename__ = "assistant_categories"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    value = Column(String(50), nullable=False, unique=True)  # URL-friendly slug
    label = Column(String(100), nullable=False)  # Display name
    icon = Column(String(50), default="📁")  # Emoji or icon
    description = Column(String(255), nullable=True)
    sort_order = Column(Integer, default=0)  # For custom ordering
    is_active = Column(Boolean, default=True)  # Can be disabled without deleting
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_category_sort", "sort_order", "label"),
    )


class CustomAssistant(Base):
    """
    Custom AI assistant with specific objectives, knowledge, and tools.
    Similar to OpenAI's GPTs.
    
    Features:
    - Customizable system prompt and behavior
    - Attached knowledge bases for RAG
    - Model and temperature configuration
    - Public marketplace with ratings
    - Usage statistics
    """
    __tablename__ = "custom_assistants"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Identity
    name = Column(String(100), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)  # URL-friendly identifier
    tagline = Column(String(255), nullable=True)  # Short description
    description = Column(Text, nullable=True)  # Full description (markdown)
    
    # Appearance
    avatar_url = Column(String(500), nullable=True)
    icon = Column(String(100), default="🤖")
    color = Column(String(20), default="#6366f1")
    
    # Behavior
    system_prompt = Column(Text, nullable=False)  # The core instructions
    welcome_message = Column(Text, nullable=True)  # Initial message to show users
    suggested_prompts = Column(JSON, default=list)  # Example prompts to show
    
    # Model settings
    model = Column(String(100), default="claude-sonnet-4-20250514")
    temperature = Column(Float, default=1.0)
    max_tokens = Column(Integer, default=4096)
    
    # Capabilities
    enabled_tools = Column(JSON, default=list)  # ["web_search", "code_execution", etc.]
    custom_tools = Column(JSON, default=list)  # Custom tool definitions
    
    # Visibility
    is_public = Column(Boolean, default=False)
    is_discoverable = Column(Boolean, default=False)  # Shows in public directory
    is_featured = Column(Boolean, default=False)  # Admin-featured
    category = Column(String(50), default="general")  # Category for filtering
    
    # Usage stats
    conversation_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    rating_sum = Column(Integer, default=0)
    rating_count = Column(Integer, default=0)
    
    # Versioning
    version = Column(Integer, default=1)
    published_at = Column(DateTime, nullable=True)  # When made public
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="custom_assistants")
    knowledge_stores = relationship(
        "KnowledgeStore",
        secondary=assistant_knowledge_stores,
        backref="assistants"
    )
    
    __table_args__ = (
        Index("idx_assistant_owner", "owner_id"),
        Index("idx_assistant_slug", "slug"),
        Index("idx_assistant_public", "is_public", "is_discoverable"),
    )
    
    @property
    def average_rating(self) -> float:
        """Calculate average rating from sum and count"""
        if self.rating_count == 0:
            return 0.0
        return self.rating_sum / self.rating_count
    
    @property
    def subscriber_count(self) -> int:
        """Get number of subscribers (placeholder for future implementation)"""
        return self.conversation_count


class AssistantConversation(Base):
    """
    Track conversations with custom assistants.
    
    Links a chat to an assistant for:
    - Usage statistics
    - User feedback/ratings
    """
    __tablename__ = "assistant_conversations"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    assistant_id = Column(String(36), ForeignKey("custom_assistants.id", ondelete="CASCADE"), nullable=False)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # User feedback
    rating = Column(Integer, nullable=True)  # 1-5
    feedback = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    assistant = relationship("CustomAssistant")
    chat = relationship("Chat")
    user = relationship("User")
    
    __table_args__ = (
        Index("idx_assistant_conv", "assistant_id", "user_id"),
    )
