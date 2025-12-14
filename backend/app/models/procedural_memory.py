"""
Procedural Memory Models

Skills are learned from successful interactions and stored as reusable neural modules.
They contain:
- Action sequences (the steps to accomplish a task)
- Contextual embeddings (for similarity-based retrieval)
- Metadata (success rate, usage count, etc.)

This is stored in a separate database for isolation and scalability.
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, LargeBinary, Index, create_engine
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

# Separate base for procedural memory database
ProceduralBase = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Skill(ProceduralBase):
    """
    A learned skill extracted from successful interactions.
    
    Skills are reusable procedures that can be retrieved and applied
    to similar future situations.
    """
    __tablename__ = "skills"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), nullable=False, index=True)
    
    # Skill identification
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)  # e.g., "code_generation", "analysis", "writing"
    
    # The actual skill content
    trigger_pattern = Column(Text, nullable=False)  # What kind of request triggers this skill
    action_sequence = Column(JSON, nullable=False)  # Steps/actions to accomplish the task
    example_input = Column(Text, nullable=True)  # Representative input that triggers this skill
    example_output = Column(Text, nullable=True)  # Representative successful output
    
    # Embedding for similarity search
    embedding = Column(LargeBinary, nullable=True)  # Vector embedding of trigger_pattern
    
    # Performance metrics
    success_count = Column(Integer, default=1)
    failure_count = Column(Integer, default=0)
    usage_count = Column(Integer, default=1)
    avg_quality_score = Column(Float, default=1.0)  # 0-1 score based on feedback
    
    # Metadata
    source_chat_id = Column(String(36), nullable=True)  # Chat where skill was learned
    source_message_id = Column(String(36), nullable=True)
    model_used = Column(String(100), nullable=True)  # Model that generated the successful response
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_used_at = Column(DateTime, nullable=True)
    
    # Soft delete
    is_active = Column(Boolean, default=True)
    
    # Relationships
    executions = relationship("SkillExecution", back_populates="skill", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_skill_user_category", "user_id", "category"),
        Index("idx_skill_active", "is_active"),
    )


class SkillExecution(ProceduralBase):
    """
    Record of a skill being executed/applied.
    
    Tracks when skills are used and their outcomes for
    continuous improvement.
    """
    __tablename__ = "skill_executions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    skill_id = Column(String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), nullable=False)
    
    # Execution context
    chat_id = Column(String(36), nullable=True)
    input_text = Column(Text, nullable=False)  # The query that triggered skill retrieval
    similarity_score = Column(Float, nullable=True)  # How similar was the input to the skill trigger
    
    # Outcome
    was_successful = Column(Boolean, nullable=True)  # Based on user feedback or heuristics
    quality_score = Column(Float, nullable=True)  # 0-1 quality rating
    execution_time_ms = Column(Integer, nullable=True)
    
    # Feedback
    user_feedback = Column(String(50), nullable=True)  # "helpful", "not_helpful", "wrong"
    feedback_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    skill = relationship("Skill", back_populates="executions")
    
    __table_args__ = (
        Index("idx_execution_skill", "skill_id"),
        Index("idx_execution_user", "user_id"),
    )


class SkillComposition(ProceduralBase):
    """
    Tracks how skills can be composed together.
    
    Some complex tasks may require combining multiple skills
    in sequence or parallel.
    """
    __tablename__ = "skill_compositions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Parent skill (the composed skill)
    parent_skill_id = Column(String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False)
    
    # Child skill (component)
    child_skill_id = Column(String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False)
    
    # Composition order
    sequence_order = Column(Integer, default=0)
    
    # How to connect them
    connection_type = Column(String(50), default="sequential")  # "sequential", "parallel", "conditional"
    condition = Column(Text, nullable=True)  # For conditional connections
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index("idx_composition_parent", "parent_skill_id"),
    )


class LearningEvent(ProceduralBase):
    """
    Records when new skills are learned or existing ones updated.
    
    Useful for understanding the agent's learning trajectory
    and for debugging.
    """
    __tablename__ = "learning_events"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), nullable=False)
    
    event_type = Column(String(50), nullable=False)  # "skill_created", "skill_updated", "skill_merged", "skill_deprecated"
    skill_id = Column(String(36), nullable=True)
    
    # What triggered the learning
    trigger_source = Column(String(50), nullable=False)  # "thumbs_up", "successful_completion", "explicit_save"
    source_chat_id = Column(String(36), nullable=True)
    
    # Details
    details = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index("idx_learning_user", "user_id"),
        Index("idx_learning_type", "event_type"),
    )
