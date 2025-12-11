"""
System settings and theme models

Contains:
- SystemSetting: Key-value store for admin settings
- Theme: User-created UI themes
"""
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, 
    ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base, generate_uuid


class SystemSetting(Base):
    """
    Key-value store for system settings configurable by admins.
    
    Common settings:
    - token_refill_interval_hours: Hours between token resets
    - last_token_reset_timestamp: ISO datetime of last reset
    - debug_token_resets: Enable verbose token reset logging
    - debug_document_queue: Enable document queue logging
    - system_prompt: Default system prompt for new chats
    - input_token_price: Cost per input token
    - output_token_price: Cost per output token
    """
    __tablename__ = "system_settings"
    
    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    @classmethod
    def get_int(cls, value: str, default: int = 0) -> int:
        """Parse setting value as integer"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def get_float(cls, value: str, default: float = 0.0) -> float:
        """Parse setting value as float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def get_bool(cls, value: str, default: bool = False) -> bool:
        """Parse setting value as boolean"""
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')


class Theme(Base):
    """
    User-created UI themes.
    
    Stores:
    - Color schemes
    - Font configurations
    - Visibility settings (public/system)
    """
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
    
    @property
    def primary_color(self) -> str:
        """Get primary theme color"""
        return self.colors.get('primary', '#6366f1')
    
    @property
    def is_dark(self) -> bool:
        """Check if this is a dark theme based on background color"""
        bg = self.colors.get('background', '#ffffff')
        # Simple heuristic: dark themes have low luminosity backgrounds
        if bg.startswith('#'):
            r = int(bg[1:3], 16)
            g = int(bg[3:5], 16)
            b = int(bg[5:7], 16)
            luminosity = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return luminosity < 0.5
        return False
