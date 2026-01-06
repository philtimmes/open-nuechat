"""
Open-NueChat Database Models

This package contains all SQLAlchemy ORM models organized by domain.

Module Structure:
- base.py: Base class, UUID generator, and all enums
- user.py: User, OAuthAccount, APIKey
- chat.py: Chat, Message, ChatParticipant
- document.py: Document, DocumentChunk, KnowledgeStore, KnowledgeStoreShare
- assistant.py: CustomAssistant, AssistantConversation, assistant_knowledge_stores
- assistant_mode.py: AssistantMode (tool presets)
- billing.py: TokenUsage
- tool.py: Tool, ToolUsage
- filter.py: ChatFilter
- upload.py: UploadedFile, UploadedArchive
- settings.py: SystemSetting, Theme
- llm_provider.py: LLMProvider (multi-model support)

All models are re-exported here for backward compatibility.
Usage:
    from app.models.models import User, Chat, Message  # Legacy
    from app.models import User, Chat, Message         # Preferred
"""

# Base utilities and enums
from .base import (
    Base,
    generate_uuid,
    UserTier,
    MessageRole,
    ContentType,
    SharePermission,
    APIKeyScope,
    ToolType,
    FilterType,
    FilterPriority,
)

# User-related models
from .user import User, OAuthAccount, APIKey

# Chat and message models
from .chat import Chat, Message, ChatParticipant

# Document and knowledge store models
from .document import Document, DocumentChunk, KnowledgeStore, KnowledgeStoreShare

# Assistant models
from .assistant import CustomAssistant, AssistantConversation, AssistantCategory, assistant_knowledge_stores
from .assistant_mode import AssistantMode, DEFAULT_ASSISTANT_MODES

# Billing models
from .billing import TokenUsage

# Tool models
from .tool import Tool, ToolUsage

# Filter models
from .filter import ChatFilter
from .filter_chain import FilterChain

# Upload models
from .upload import UploadedFile, UploadedArchive

# Settings models
from .settings import SystemSetting, Theme

# LLM Provider models
from .llm_provider import LLMProvider

# Export all for "from app.models import *"
__all__ = [
    # Base
    "Base",
    "generate_uuid",
    # Enums
    "UserTier",
    "MessageRole",
    "ContentType",
    "SharePermission",
    "APIKeyScope",
    "ToolType",
    "FilterType",
    "FilterPriority",
    # User
    "User",
    "OAuthAccount",
    "APIKey",
    # Chat
    "Chat",
    "Message",
    "ChatParticipant",
    # Document
    "Document",
    "DocumentChunk",
    "KnowledgeStore",
    "KnowledgeStoreShare",
    # Assistant
    "CustomAssistant",
    "AssistantConversation",
    "AssistantCategory",
    "assistant_knowledge_stores",
    # Assistant Mode
    "AssistantMode",
    "DEFAULT_ASSISTANT_MODES",
    # Billing
    "TokenUsage",
    # Tool
    "Tool",
    "ToolUsage",
    # Filter
    "ChatFilter",
    "FilterChain",
    # Upload
    "UploadedFile",
    "UploadedArchive",
    # Settings
    "SystemSetting",
    "Theme",
    # LLM Provider
    "LLMProvider",
]
