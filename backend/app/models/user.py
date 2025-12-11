"""
User-related models

Contains:
- User: Core user account
- OAuthAccount: OAuth provider connections
- APIKey: Programmatic access tokens
"""
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, 
    ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid, UserTier, APIKeyScope


class User(Base):
    """
    Core user account model.
    
    Supports both password-based and OAuth authentication.
    Tracks billing tier, token usage, and preferences.
    """
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
    
    # Relationships - defined with string references for lazy loading
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="owner", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    token_usage = relationship("TokenUsage", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    knowledge_stores = relationship("KnowledgeStore", back_populates="owner", cascade="all, delete-orphan")
    custom_assistants = relationship("CustomAssistant", back_populates="owner", cascade="all, delete-orphan")
    
    @property
    def is_unlimited(self) -> bool:
        """Check if user has unlimited tokens (admin bypass)"""
        return self.is_admin


class OAuthAccount(Base):
    """
    OAuth provider account linked to a user.
    
    Stores tokens and provider-specific user ID for reconnection.
    """
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


class APIKey(Base):
    """
    User-generated API key for programmatic access.
    
    Features:
    - Scoped permissions
    - IP restrictions
    - Usage tracking
    - Optional expiration
    """
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(100), nullable=False)  # User-friendly name
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification
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
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if this key has a specific scope"""
        if APIKeyScope.FULL.value in self.scopes:
            return True
        return scope.value in self.scopes
