"""
File upload models for code context

Contains:
- UploadedFile: Individual files from zip archives
- UploadedArchive: Metadata about uploaded archives
"""
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, 
    ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid


class UploadedFile(Base):
    """
    Files uploaded via zip archives, persisted server-side.
    
    Stores:
    - File metadata (path, size, type)
    - Content (for text files under size limit)
    - Code signatures for LLM context
    
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
    
    @property
    def signature_count(self) -> int:
        """Count of code signatures in this file"""
        return len(self.signatures) if self.signatures else 0


class UploadedArchive(Base):
    """
    Metadata about uploaded zip archives.
    
    Stores:
    - Archive statistics (file count, total size)
    - Language breakdown
    - Pre-formatted LLM manifest for prompt injection
    - Human-readable summary with file tree
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
    
    @property
    def primary_language(self) -> str | None:
        """Get the most common language in the archive"""
        if not self.languages:
            return None
        return max(self.languages.items(), key=lambda x: x[1])[0]
