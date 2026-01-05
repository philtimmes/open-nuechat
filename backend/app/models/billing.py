"""
Billing and token tracking models

Contains:
- TokenUsage: Detailed token usage tracking per message
"""
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, 
    ForeignKey, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid


class TokenUsage(Base):
    """
    Detailed token usage tracking for billing.
    
    Records:
    - Per-message token counts
    - Model used
    - Cost calculation
    - Monthly aggregation indexes
    """
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
    
    @property
    def total_tokens(self) -> int:
        """Total tokens for this usage record"""
        return self.input_tokens + self.output_tokens
