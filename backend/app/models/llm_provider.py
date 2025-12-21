"""
LLM Provider Model

Supports multiple LLM providers with routing capabilities:
- Multiple providers (different APIs, models)
- Multimodal flag for vision/video support
- Hybrid routing: smart text model + vision model for image understanding
"""
from sqlalchemy import Column, String, Boolean, Integer, Text, DateTime, func
from app.models.base import Base


class LLMProvider(Base):
    """
    LLM Provider configuration.
    
    Supports hybrid routing where:
    - Primary model handles text/reasoning
    - Vision model handles image understanding, provides descriptions to primary
    """
    __tablename__ = "llm_providers"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)  # Display name
    
    # Connection settings
    base_url = Column(String(500), nullable=False)
    api_key = Column(String(500), nullable=True)  # Can be empty for local models
    model_id = Column(String(200), nullable=False)  # Actual model identifier
    
    # Capabilities
    is_multimodal = Column(Boolean, default=False)  # Supports images/vision
    supports_tools = Column(Boolean, default=True)  # Function calling
    supports_streaming = Column(Boolean, default=True)
    
    # Routing flags
    is_default = Column(Boolean, default=False)  # Primary text model
    is_vision_default = Column(Boolean, default=False)  # Used for image understanding
    is_enabled = Column(Boolean, default=True)
    
    # Model settings
    timeout = Column(Integer, default=300)
    max_tokens = Column(Integer, default=8192)
    context_size = Column(Integer, default=128000)  # Context window size
    temperature = Column(String(10), default="0.7")  # Stored as string for precision
    
    # Optional description prompt for vision model
    # If set, used when this model describes images for non-MM models
    vision_prompt = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key[:8] + "..." if self.api_key and len(self.api_key) > 8 else self.api_key,
            "model_id": self.model_id,
            "is_multimodal": self.is_multimodal,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "is_default": self.is_default,
            "is_vision_default": self.is_vision_default,
            "is_enabled": self.is_enabled,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "context_size": self.context_size,
            "temperature": self.temperature,
            "vision_prompt": self.vision_prompt,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def to_dict_full(self):
        """Full dict including full API key (for internal use only)"""
        d = self.to_dict()
        d["api_key"] = self.api_key
        return d


# Default vision prompt for image description
DEFAULT_VISION_PROMPT = """Describe this image in comprehensive detail for another AI that cannot see images.

User's context: {user_message}

Include:
- Main subjects and their appearance
- Actions or activities happening
- Setting/environment/background
- Any text visible in the image
- Colors, lighting, composition
- Relevant details that might help answer the user's question

Be thorough but concise. Focus on factual description."""
