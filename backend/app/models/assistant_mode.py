"""
Assistant Mode Model

Defines presets for tool activation and LLM behavior.
Modes control which tools are active and advertised to the LLM.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from .base import Base


class AssistantMode(Base):
    """
    An assistant mode preset.
    
    Modes define which tools are active and advertised, allowing
    quick switching between different assistant behaviors like
    "Creative Writing", "Coding", "Deep Research", etc.
    """
    __tablename__ = "assistant_modes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    icon = Column(String(500), nullable=True)  # Path to SVG
    
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API/cache."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
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
        "icon": None,
        "active_tools": ["web_search", "artifacts"],
        "advertised_tools": [],
        "sort_order": 0,
    },
    {
        "name": "Creative Writing",
        "description": "Focused writing without tool distractions",
        "icon": None,
        "active_tools": [],
        "advertised_tools": [],
        "sort_order": 1,
    },
    {
        "name": "Coding",
        "description": "Full artifact and code tools for development",
        "icon": None,
        "active_tools": ["artifacts", "file_ops", "code_exec"],
        "advertised_tools": ["artifacts", "file_ops"],
        "sort_order": 2,
    },
    {
        "name": "Deep Research",
        "description": "Web search and knowledge base tools for research",
        "icon": None,
        "active_tools": ["web_search", "kb_search", "citations"],
        "advertised_tools": ["web_search", "kb_search"],
        "sort_order": 3,
    },
]
