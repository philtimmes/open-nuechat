"""
Billing and token tracking models

Contains:
- TokenUsage: Detailed token usage tracking per message
- Subscription: User subscription management
- PaymentMethod: Stored payment methods
- Transaction: Payment transaction history
"""
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean,
    ForeignKey, Index, Enum as SQLEnum, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from .base import Base, generate_uuid


class PaymentProvider(str, enum.Enum):
    """Supported payment providers"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    GOOGLE_PAY = "google_pay"


class PaymentStatus(str, enum.Enum):
    """Payment transaction status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class SubscriptionStatus(str, enum.Enum):
    """Subscription status"""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    TRIALING = "trialing"


class Subscription(Base):
    """
    User subscription tracking.
    
    Manages:
    - Subscription tier and status
    - Billing cycle dates
    - Payment provider references
    """
    __tablename__ = "subscriptions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Subscription details
    tier = Column(String(20), nullable=False, default="free")
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)
    
    # Billing cycle
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    
    # Payment provider references
    provider = Column(SQLEnum(PaymentProvider), nullable=True)
    provider_subscription_id = Column(String(255), nullable=True)  # Stripe sub_xxx, PayPal I-xxx
    provider_customer_id = Column(String(255), nullable=True)  # Stripe cus_xxx, PayPal payer ID
    
    # Cancellation
    cancel_at_period_end = Column(Boolean, default=False)
    cancelled_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="subscription")
    
    __table_args__ = (
        Index("idx_subscription_user", "user_id"),
        Index("idx_subscription_provider", "provider", "provider_subscription_id"),
    )


class PaymentMethod(Base):
    """
    Stored payment methods for users.
    
    Stores:
    - Card details (tokenized)
    - PayPal accounts
    - Google Pay tokens
    """
    __tablename__ = "payment_methods"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Provider info
    provider = Column(SQLEnum(PaymentProvider), nullable=False)
    provider_payment_method_id = Column(String(255), nullable=False)  # pm_xxx, etc.
    
    # Display info (safe to store)
    type = Column(String(20), nullable=False)  # card, paypal, google_pay
    last_four = Column(String(4), nullable=True)  # Last 4 digits of card
    brand = Column(String(20), nullable=True)  # visa, mastercard, amex, etc.
    exp_month = Column(Integer, nullable=True)
    exp_year = Column(Integer, nullable=True)
    email = Column(String(255), nullable=True)  # PayPal email
    
    # Status
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="payment_methods")
    
    __table_args__ = (
        Index("idx_payment_method_user", "user_id"),
    )


class Transaction(Base):
    """
    Payment transaction history.
    
    Records:
    - All payment attempts
    - Subscription payments
    - One-time purchases
    - Refunds
    """
    __tablename__ = "transactions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    subscription_id = Column(String(36), ForeignKey("subscriptions.id", ondelete="SET NULL"), nullable=True)
    
    # Transaction details
    type = Column(String(20), nullable=False)  # subscription, one_time, refund
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    
    # Provider info
    provider = Column(SQLEnum(PaymentProvider), nullable=False)
    provider_transaction_id = Column(String(255), nullable=True)  # pi_xxx, PAYID-xxx
    provider_payment_method_id = Column(String(255), nullable=True)
    
    # Metadata
    description = Column(String(500), nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string for extra data
    
    # Error handling
    error_code = Column(String(50), nullable=True)
    error_message = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    
    __table_args__ = (
        Index("idx_transaction_user", "user_id"),
        Index("idx_transaction_provider", "provider", "provider_transaction_id"),
        Index("idx_transaction_status", "status"),
    )


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
