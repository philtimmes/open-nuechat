"""
Agent Flow models for visual workflow builder
"""
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from .base import Base, generate_uuid


class AgentFlow(Base):
    """User-created agent workflow"""
    __tablename__ = "agent_flows"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False, default="New Agent Flow")
    description = Column(Text, nullable=True)
    
    # Flow definition as JSON
    # Contains: nodes[], connections[]
    definition = Column(JSON, nullable=False, default=dict)
    
    # Flow settings
    is_active = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)  # Can be shared/used by others
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    owner = relationship("User", backref="agent_flows")
    
    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "description": self.description,
            "definition": self.definition,
            "is_active": self.is_active,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
