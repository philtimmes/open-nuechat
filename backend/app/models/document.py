"""
Document and knowledge store models for RAG

Contains:
- Document: Uploaded document metadata
- DocumentChunk: Vector chunks for similarity search
- KnowledgeStore: Collection of documents
- KnowledgeStoreShare: Sharing permissions
"""
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, Float,
    ForeignKey, Enum, JSON, LargeBinary, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid, SharePermission


class Document(Base):
    """
    Document uploaded for RAG processing.
    
    Documents go through an async processing pipeline:
    1. Upload → is_processed=False
    2. Text extraction
    3. Chunking
    4. Embedding generation
    5. is_processed=True, chunk_count set
    
    NC-0.8.0.1.1: Document-level keyword filtering for Global KBs
    - Each doc can require specific keywords in user query
    - Supports "exact phrases" and comma-separated keywords
    - Match mode: all, any, or mixed (phrases=all, keywords=any)
    """
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
    
    # NC-0.8.0.1.1: Document-level keyword filtering for Global KBs
    # When enabled, document chunks only returned if query contains keywords
    require_keywords_enabled = Column(Boolean, default=False)
    required_keywords = Column(Text, nullable=True)  # Comma-separated, "phrases in quotes"
    keyword_match_mode = Column(String(10), default='any')  # 'any', 'all', 'mixed'
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    owner = relationship("User", back_populates="documents")
    knowledge_store = relationship("KnowledgeStore", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """
    Vector chunk for RAG similarity search.
    
    Stores:
    - Text content
    - Binary embedding (sentence-transformers)
    - Chunk position and metadata
    """
    __tablename__ = "document_chunks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Store embedding as binary (FAISS used for similarity search)
    embedding = Column(LargeBinary, nullable=True)
    
    # Metadata for context
    chunk_metadata = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=func.now())
    
    document = relationship("Document", back_populates="chunks")
    
    __table_args__ = (
        Index("idx_chunk_document", "document_id", "chunk_index"),
    )


class KnowledgeStore(Base):
    """
    Collection of documents for RAG.
    
    Features:
    - Personal or shared ownership
    - Configurable embedding settings
    - Usage statistics
    - Public/discoverable visibility options
    - Global stores auto-searched on every query
    - Required keywords filter for relevance (NC-0.8.0.1)
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
    is_global = Column(Boolean, default=False)  # Auto-searched on every query (admin only)
    
    # Global store settings
    global_min_score = Column(Float, default=0.7)  # Minimum relevance score to include results
    global_max_results = Column(Integer, default=3)  # Max results to include from global search
    
    # Required keywords filter (NC-0.8.0.1)
    # When enabled, global KB only activates if query contains at least one keyword
    require_keywords_enabled = Column(Boolean, default=False)
    required_keywords = Column(JSON, nullable=True)  # List of keywords/phrases: ["pricing", "cost", "subscription"]
    
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
        Index("idx_knowledge_store_global", "is_global"),
    )


class KnowledgeStoreShare(Base):
    """
    Sharing permissions for knowledge stores.
    
    Supports:
    - Direct user sharing
    - Link-based sharing with tokens
    - Permission levels (view, edit, admin)
    - Optional expiration and usage limits
    """
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
