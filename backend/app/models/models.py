"""
Database models for Open-NueChat
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, Enum, JSON, LargeBinary, Index, Table
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum
import uuid

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class UserTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, enum.Enum):
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
    CHAT = "chat"               # Create messages, stream responses
    KNOWLEDGE = "knowledge"     # Access knowledge stores
    ASSISTANTS = "assistants"   # Use custom assistants
    BILLING = "billing"         # View usage/billing
    FULL = "full"               # All permissions


class ToolType(str, enum.Enum):
    """Types of external tools"""
    MCP = "mcp"           # Model Context Protocol server
    OPENAPI = "openapi"   # OpenAPI/Swagger specification


# Association table for Assistant <-> KnowledgeStore
assistant_knowledge_stores = Table(
    "assistant_knowledge_stores",
    Base.metadata,
    Column("assistant_id", String(36), ForeignKey("custom_assistants.id", ondelete="CASCADE")),
    Column("knowledge_store_id", String(36), ForeignKey("knowledge_stores.id", ondelete="CASCADE")),
)


class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)  # Null for OAuth users
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    # Billing
    tier = Column(Enum(UserTier), default=UserTier.FREE)
    stripe_customer_id = Column(String(255), nullable=True)
    
    # Token usage
    tokens_used_this_month = Column(Integer, default=0)
    tokens_limit = Column(Integer, default=100_000)
    
    # Preferences
    theme = Column(String(50), default="dark")
    preferences = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="owner", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    token_usage = relationship("TokenUsage", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    knowledge_stores = relationship("KnowledgeStore", back_populates="owner", cascade="all, delete-orphan")
    custom_assistants = relationship("CustomAssistant", back_populates="owner", cascade="all, delete-orphan")


class OAuthAccount(Base):
    __tablename__ = "oauth_accounts"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)  # google, github, etc.
    provider_user_id = Column(String(255), nullable=False)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="oauth_accounts")
    
    __table_args__ = (
        Index("idx_oauth_provider_user", "provider", "provider_user_id", unique=True),
    )


class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), default="New Chat")
    model = Column(String(100), default="claude-sonnet-4-20250514")
    system_prompt = Column(Text, nullable=True)
    
    # Chat type
    is_shared = Column(Boolean, default=False)  # For client-to-client chat
    
    # Public sharing
    share_id = Column(String(36), nullable=True, unique=True, index=True)
    
    # Token tracking
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    
    # Branch selection - tracks which child is selected at each branch point
    # Format: { "parent_message_id": "selected_child_id" }
    # When a message has multiple children (siblings), this determines which branch to show
    selected_versions = Column(JSON, default=dict)
    
    # Code summary - tracks files and signatures for LLM code generation
    # Format: { "files": [...], "warnings": [...], "last_updated": "...", "auto_generated": true }
    code_summary = Column(JSON, nullable=True)
    
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


class ChatParticipant(Base):
    """For client-to-client chat support"""
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


class UploadedFile(Base):
    """
    Files uploaded via zip archives, persisted server-side.
    Linked to chat for automatic cleanup on chat deletion.
    """
    __tablename__ = "uploaded_files"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    
    # Archive info
    archive_name = Column(String(255), nullable=True)  # Original zip filename
    
    # File info
    filepath = Column(String(1000), nullable=False)  # Path within archive
    filename = Column(String(255), nullable=False)  # Just the filename
    extension = Column(String(50), nullable=True)
    language = Column(String(50), nullable=True)  # Detected language
    size = Column(Integer, default=0)
    is_binary = Column(Boolean, default=False)
    
    # Content (null for binary files)
    content = Column(Text, nullable=True)
    
    # Code analysis
    signatures = Column(JSON, nullable=True)  # [{name, kind, line, signature}]
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    chat = relationship("Chat", back_populates="uploaded_files")
    
    __table_args__ = (
        Index("idx_uploaded_file_chat", "chat_id"),
        Index("idx_uploaded_file_path", "chat_id", "filepath"),
    )


class UploadedArchive(Base):
    """
    Metadata about uploaded zip archives.
    Stores the LLM manifest for injection into prompts.
    """
    __tablename__ = "uploaded_archives"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    filename = Column(String(255), nullable=False)
    total_files = Column(Integer, default=0)
    total_size = Column(Integer, default=0)
    languages = Column(JSON, nullable=True)  # {language: count}
    
    # Pre-formatted manifest for LLM injection
    llm_manifest = Column(Text, nullable=True)
    
    # Human-readable summary with file tree and signatures
    summary = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    chat = relationship("Chat", back_populates="uploaded_archive", uselist=False)
    
    __table_args__ = (
        Index("idx_uploaded_archive_chat", "chat_id"),
    )


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    parent_id = Column(String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)  # For conversation branching
    
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=True)
    content_type = Column(Enum(ContentType), default=ContentType.TEXT)
    
    # For multimodal content
    attachments = Column(JSON, default=list)  # [{type, url, name, size, mime_type}]
    
    # For tool calls
    tool_calls = Column(JSON, nullable=True)  # [{id, name, arguments}]
    tool_call_id = Column(String(100), nullable=True)  # For tool results
    
    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    
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


class Document(Base):
    """Documents for RAG"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    knowledge_store_id = Column(String(36), ForeignKey("knowledge_stores.id", ondelete="SET NULL"), nullable=True)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    chunk_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    owner = relationship("User", back_populates="documents")
    knowledge_store = relationship("KnowledgeStore", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Vector chunks for RAG"""
    __tablename__ = "document_chunks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Store embedding as binary (will use sqlite-vec for similarity search)
    embedding = Column(LargeBinary, nullable=True)
    
    # Metadata for context
    chunk_metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=func.now())
    
    document = relationship("Document", back_populates="chunks")
    
    __table_args__ = (
        Index("idx_chunk_document", "document_id", "chunk_index"),
    )


class TokenUsage(Base):
    """Detailed token usage tracking for billing"""
    __tablename__ = "token_usage"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="SET NULL"), nullable=True)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    
    model = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    
    # Cost calculation
    input_cost = Column(Float, default=0.0)
    output_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    
    # For monthly aggregation
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    
    user = relationship("User", back_populates="token_usage")
    
    __table_args__ = (
        Index("idx_usage_user_month", "user_id", "year", "month"),
    )


class Theme(Base):
    """User-created themes"""
    __tablename__ = "themes"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    creator_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)
    is_system = Column(Boolean, default=False)  # Built-in themes
    
    # Theme colors and styles
    colors = Column(JSON, nullable=False)  # {primary, secondary, background, text, accent, etc.}
    fonts = Column(JSON, default=dict)  # {heading, body, code}
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    creator = relationship("User")


# =============================================================================
# API Keys
# =============================================================================

class APIKey(Base):
    """User-generated API keys for programmatic access"""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(100), nullable=False)  # User-friendly name
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification (e.g., "nxs_abc1")
    key_hash = Column(String(255), nullable=False)  # Hashed full key
    
    # Permissions
    scopes = Column(JSON, default=list)  # List of APIKeyScope values
    
    # Rate limiting
    rate_limit = Column(Integer, default=100)  # Requests per minute
    
    # Restrictions
    allowed_ips = Column(JSON, default=list)  # Empty = allow all
    allowed_assistants = Column(JSON, default=list)  # Empty = allow all
    allowed_knowledge_stores = Column(JSON, default=list)  # Empty = allow all
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    last_used_ip = Column(String(45), nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Expiration
    expires_at = Column(DateTime, nullable=True)  # Null = never expires
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index("idx_api_key_prefix", "key_prefix"),
        Index("idx_api_key_user", "user_id"),
    )


# =============================================================================
# Knowledge Stores
# =============================================================================

class KnowledgeStore(Base):
    """
    Collection of documents for RAG.
    Can be personal or shared with other users.
    """
    __tablename__ = "knowledge_stores"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(100), default="📚")  # Emoji or icon name
    color = Column(String(20), default="#6366f1")  # Hex color for UI
    
    # Visibility
    is_public = Column(Boolean, default=False)  # Anyone can view/use
    is_discoverable = Column(Boolean, default=False)  # Shows in public directory
    
    # Stats
    document_count = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    
    # Settings
    embedding_model = Column(String(100), default="all-MiniLM-L6-v2")
    chunk_size = Column(Integer, default=500)
    chunk_overlap = Column(Integer, default=50)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="knowledge_stores")
    documents = relationship("Document", back_populates="knowledge_store")
    shares = relationship("KnowledgeStoreShare", back_populates="knowledge_store", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_knowledge_store_owner", "owner_id"),
        Index("idx_knowledge_store_public", "is_public", "is_discoverable"),
    )


class KnowledgeStoreShare(Base):
    """Sharing permissions for knowledge stores"""
    __tablename__ = "knowledge_store_shares"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    knowledge_store_id = Column(String(36), ForeignKey("knowledge_stores.id", ondelete="CASCADE"), nullable=False)
    
    # Share with user OR generate a share link
    shared_with_user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    share_token = Column(String(64), nullable=True, unique=True)  # For link sharing
    
    permission = Column(Enum(SharePermission), default=SharePermission.VIEW)
    
    # Optional restrictions
    expires_at = Column(DateTime, nullable=True)
    max_uses = Column(Integer, nullable=True)  # For share links
    use_count = Column(Integer, default=0)
    
    # Metadata
    shared_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    message = Column(Text, nullable=True)  # Optional message to recipient
    
    created_at = Column(DateTime, default=func.now())
    accepted_at = Column(DateTime, nullable=True)  # When user accepted the share
    
    # Relationships
    knowledge_store = relationship("KnowledgeStore", back_populates="shares")
    shared_with_user = relationship("User", foreign_keys=[shared_with_user_id])
    shared_by_user = relationship("User", foreign_keys=[shared_by_user_id])
    
    __table_args__ = (
        Index("idx_share_store_user", "knowledge_store_id", "shared_with_user_id"),
        Index("idx_share_token", "share_token"),
    )


# =============================================================================
# Custom Assistants (GPTs)
# =============================================================================

class CustomAssistant(Base):
    """
    Custom AI assistants with specific objectives, knowledge, and tools.
    Similar to OpenAI's GPTs.
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
        if self.rating_count == 0:
            return 0.0
        return self.rating_sum / self.rating_count


class AssistantConversation(Base):
    """Track conversations with custom assistants"""
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


class SystemSetting(Base):
    """Key-value store for system settings configurable by admins"""
    __tablename__ = "system_settings"
    
    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Tool(Base):
    """External tools (MCP servers or OpenAPI endpoints)"""
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
    """Track tool invocations for citations and debugging"""
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


class ChatFilter(Base):
    """
    Database-backed filter configurations.
    
    Filters can process messages at different points in the pipeline:
    - TO_LLM: Before user messages go to the LLM
    - FROM_LLM: Before LLM responses go to the user
    - TO_TOOLS: Before messages are sent to tools
    - FROM_TOOLS: Before tool results are processed
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
    # For simple filters: regex patterns, word lists, etc.
    # For complex filters: Python code executed in sandbox
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
