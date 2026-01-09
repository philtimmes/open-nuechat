"""
Knowledge Graph for RAG Temporal Validity

Tracks relationships between document chunks to enable:
- Temporal awareness: Detecting when information is outdated
- Supersession chains: Knowing when new info replaces old
- Contradiction detection: Flagging conflicting information
- Entity linking: Connecting chunks about the same topic/entity

NC-0.8.0.2: Knowledge Graph Filtering
"""
from sqlalchemy import Column, String, Text, DateTime, Float, Enum, ForeignKey, Index, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.models.base import Base, generate_uuid


class EdgeType(enum.Enum):
    """Types of relationships between chunks"""
    SUPERSEDES = "supersedes"      # New chunk replaces old chunk (same topic, newer info)
    UPDATES = "updates"            # New chunk modifies/amends old chunk
    CONTRADICTS = "contradicts"    # New chunk conflicts with old chunk
    CONFIRMS = "confirms"          # New chunk validates old chunk is still accurate
    RELATED_TO = "related_to"      # Semantic similarity (same topic cluster)
    REFERENCES = "references"      # Explicit citation/reference
    PART_OF = "part_of"            # Hierarchical: chunk is part of larger context


class KnowledgeEdge(Base):
    """
    Directed edge in the knowledge graph.
    
    source_chunk --[edge_type]--> target_chunk
    
    For temporal relationships:
    - SUPERSEDES: source is newer, target is outdated
    - UPDATES: source modifies target
    - CONTRADICTS: source and target have conflicting info
    """
    __tablename__ = "knowledge_edges"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Source and target chunks
    source_chunk_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    target_chunk_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    
    # Relationship type
    edge_type = Column(String(20), nullable=False)  # EdgeType value
    
    # Confidence score (0-1) - how confident are we in this relationship
    confidence = Column(Float, default=1.0)
    
    # Entity/topic this relationship is about (for filtering)
    # e.g., "president_of_usa", "twitter_api_pricing", "florida_statute_790.06"
    entity = Column(String(255), nullable=True)
    
    # Reason/evidence for this edge (LLM explanation)
    reason = Column(Text, nullable=True)
    
    # Auto-generated vs manually created
    is_auto = Column(Boolean, default=True)
    
    # When was this relationship detected
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    source_chunk = relationship("DocumentChunk", foreign_keys=[source_chunk_id])
    target_chunk = relationship("DocumentChunk", foreign_keys=[target_chunk_id])
    
    __table_args__ = (
        Index("idx_edge_source", "source_chunk_id"),
        Index("idx_edge_target", "target_chunk_id"),
        Index("idx_edge_type", "edge_type"),
        Index("idx_edge_entity", "entity"),
        # Unique constraint: only one edge of each type between same chunks
        Index("idx_edge_unique", "source_chunk_id", "target_chunk_id", "edge_type", unique=True),
    )


class ChunkValidity(Base):
    """
    Cached validity status for chunks.
    
    Updated when edges change. Provides fast lookup during search
    without traversing the graph every time.
    """
    __tablename__ = "chunk_validity"
    
    chunk_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True)
    
    # Is this chunk considered currently valid?
    is_valid = Column(Boolean, default=True)
    
    # If superseded, by which chunk?
    superseded_by_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    
    # Validity score (1.0 = fully valid, 0.0 = completely outdated)
    validity_score = Column(Float, default=1.0)
    
    # When was validity last computed
    computed_at = Column(DateTime, default=func.now())
    
    # Reason for current validity status
    reason = Column(Text, nullable=True)
    
    # Relationships
    chunk = relationship("DocumentChunk", foreign_keys=[chunk_id])
    superseded_by = relationship("DocumentChunk", foreign_keys=[superseded_by_id])
    
    __table_args__ = (
        Index("idx_validity_valid", "is_valid"),
        Index("idx_validity_score", "validity_score"),
    )


class EntityMention(Base):
    """
    Tracks which entities are mentioned in which chunks.
    
    Enables finding all chunks about a specific entity
    and detecting when new info about an entity arrives.
    """
    __tablename__ = "entity_mentions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    chunk_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    
    # Normalized entity identifier
    # e.g., "person:joe_biden", "company:twitter", "law:fl_790.06"
    entity_id = Column(String(255), nullable=False)
    
    # Entity type for filtering
    entity_type = Column(String(50), nullable=True)  # person, company, law, product, etc.
    
    # Original mention text
    mention_text = Column(String(500), nullable=True)
    
    # Confidence that this entity is actually mentioned
    confidence = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    chunk = relationship("DocumentChunk")
    
    __table_args__ = (
        Index("idx_mention_chunk", "chunk_id"),
        Index("idx_mention_entity", "entity_id"),
        Index("idx_mention_type", "entity_type"),
    )
