"""
Assistant Mode Model

Defines presets for tool activation and LLM behavior.
Modes control which tools are active and advertised to the LLM.
Modes also serve as categories for Custom GPTs in the marketplace.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from .base import Base

# Emoji mapping for modes (derived from name, not stored in DB)
MODE_EMOJIS = {
    "General": "🤖",
    "Creative Writing": "✍️",
    "Coding": "💻",
    "Deep Research": "🔬",
    "Legal": "⚖️",
    "Data Analysis": "📊",
    "Image Generation": "🎨",
}


class AssistantMode(Base):
    """
    An assistant mode preset.
    
    Modes define which tools are active and advertised, allowing
    quick switching between different assistant behaviors like
    "Creative Writing", "Coding", "Deep Research", etc.
    
    Modes also serve as categories for Custom GPTs - when a user
    creates a GPT and selects a "category", they're selecting a mode.
    """
    __tablename__ = "assistant_modes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    icon = Column(String(500), nullable=True)  # Path to SVG or icon URL
    
    # Tool configuration
    # active_tools: Tools the user can use (shown in toolbar)
    # Example: ["web_search", "artifacts", "image_gen"]
    active_tools = Column(JSON, nullable=False, default=list)
    
    # advertised_tools: Tools mentioned in system prompt to LLM
    # Subset of active_tools that LLM is told about
    # Example: ["web_search"]
    advertised_tools = Column(JSON, nullable=False, default=list)
    
    # Optional: Link to a filter chain for this mode
    filter_chain_id = Column(String(36), ForeignKey("filter_chains.id"), nullable=True)
    
    # Ordering in dropdown
    sort_order = Column(Integer, default=0, nullable=False)
    
    # State
    enabled = Column(Boolean, default=True, nullable=False)
    is_global = Column(Boolean, default=True, nullable=False)  # Available to all users
    
    # Metadata
    created_by = Column(String(36), nullable=True)  # Admin user ID
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    @property
    def emoji(self) -> str:
        """Get emoji for this mode (derived from name)."""
        return MODE_EMOJIS.get(self.name, "🤖")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API/cache."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "emoji": self.emoji,
            "active_tools": self.active_tools or [],
            "advertised_tools": self.advertised_tools or [],
            "filter_chain_id": self.filter_chain_id,
            "sort_order": self.sort_order,
            "enabled": self.enabled,
            "is_global": self.is_global,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Default modes to seed on first run
DEFAULT_ASSISTANT_MODES = [
    {
        "name": "General",
        "description": "Balanced tool availability for everyday tasks",
        "active_tools": ["web_search", "artifacts", "kb_search"],
        "advertised_tools": [],
        "sort_order": 0,
    },
    {
        "name": "Creative Writing",
        "description": "Focused writing without tool distractions",
        "active_tools": [],
        "advertised_tools": [],
        "sort_order": 1,
    },
    {
        "name": "Coding",
        "description": "Web search, chat history, and full development tools",
        "active_tools": ["web_search", "user_chats_kb", "artifacts", "file_ops", "code_exec"],
        "advertised_tools": ["web_search", "artifacts", "file_ops"],
        "sort_order": 2,
    },
    {
        "name": "Deep Research",
        "description": "Comprehensive research with web, knowledge bases, and citations",
        "active_tools": ["web_search", "kb_search", "local_rag", "user_chats_kb", "citations", "artifacts"],
        "advertised_tools": ["web_search", "kb_search", "citations"],
        "sort_order": 3,
    },
    {
        "name": "Legal",
        "description": "Legal research with all knowledge bases, no web search",
        "active_tools": ["kb_search", "local_rag", "user_chats_kb", "file_ops", "artifacts"],
        "advertised_tools": ["kb_search", "file_ops"],
        "sort_order": 4,
    },
    {
        "name": "Data Analysis",
        "description": "Code execution and file tools for data processing",
        "active_tools": ["code_exec", "file_ops", "artifacts", "local_rag"],
        "advertised_tools": ["code_exec", "file_ops"],
        "sort_order": 5,
    },
    {
        "name": "Image Generation",
        "description": "Focused on creating and discussing images",
        "active_tools": ["image_gen", "artifacts"],
        "advertised_tools": ["image_gen"],
        "sort_order": 6,
    },
]
