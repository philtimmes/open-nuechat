"""
Base model utilities and enums for Open-NueChat

This module contains:
- SQLAlchemy Base class
- UUID generation utility
- All enum types used across models
"""
from sqlalchemy.orm import declarative_base
import enum
import uuid

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a new UUID string for model primary keys"""
    return str(uuid.uuid4())


# =============================================================================
# Enums
# =============================================================================

class UserTier(str, enum.Enum):
    """User subscription tier levels"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class MessageRole(str, enum.Enum):
    """Role of a message sender in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, enum.Enum):
    """Type of content in a message"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class SharePermission(str, enum.Enum):
    """Permission levels for shared resources"""
    VIEW = "view"           # Can view and use in chats
    EDIT = "edit"           # Can add/remove documents
    ADMIN = "admin"         # Can manage sharing


class APIKeyScope(str, enum.Enum):
    """Scopes for API key permissions"""
    # Legacy scopes (for internal API)
    CHAT = "chat"               # Create messages, stream responses
    KNOWLEDGE = "knowledge"     # Access knowledge stores
    ASSISTANTS = "assistants"   # Use custom assistants
    BILLING = "billing"         # View usage/billing
    FULL = "full"               # All permissions
    
    # OpenAI-compatible API scopes (v1/)
    MODELS = "models"           # GET /v1/models
    COMPLETIONS = "completions" # POST /v1/chat/completions
    IMAGES = "images"           # POST /v1/images/generations
    EMBEDDINGS = "embeddings"   # POST /v1/embeddings


class ToolType(str, enum.Enum):
    """Types of external tools"""
    MCP = "mcp"           # Model Context Protocol server
    OPENAPI = "openapi"   # OpenAPI/Swagger specification


class FilterType(str, enum.Enum):
    """Types of message filters"""
    TO_LLM = "to_llm"           # User input -> LLM (ToLLMFromChat)
    FROM_LLM = "from_llm"       # LLM output -> User (FromLLMToChat)
    TO_TOOLS = "to_tools"       # Message -> Tool input
    FROM_TOOLS = "from_tools"   # Tool output -> Processing


class FilterPriority(str, enum.Enum):
    """Priority levels for filter execution"""
    HIGHEST = "highest"  # 0
    HIGH = "high"        # 25
    MEDIUM = "medium"    # 50
    LOW = "low"          # 75
    LEAST = "least"      # 100
