"""
Vision Routing Service

Handles hybrid model flows:
- Routes images to vision-capable models
- Gets descriptions for non-multimodal primary models
- Caches descriptions in message metadata
- Transparent to user - appears as single model
"""
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.llm_provider import LLMProvider, DEFAULT_VISION_PROMPT

logger = logging.getLogger(__name__)


class VisionRouter:
    """
    Routes vision requests to appropriate models.
    
    Flow for hybrid setup (non-MM primary + MM vision):
    1. Detect images in current turn attachments
    2. Send to vision model with user context
    3. Get description
    4. Replace image with description for primary model
    5. Cache description in message metadata
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._primary_provider: Optional[LLMProvider] = None
        self._vision_provider: Optional[LLMProvider] = None
        self._loaded = False
    
    async def load_providers(self):
        """Load default and vision providers from database."""
        if self._loaded:
            return
        
        # Get default provider
        result = await self.db.execute(
            select(LLMProvider)
            .where(LLMProvider.is_default == True)
            .where(LLMProvider.is_enabled == True)
            .limit(1)
        )
        self._primary_provider = result.scalar_one_or_none()
        
        # Get vision provider
        result = await self.db.execute(
            select(LLMProvider)
            .where(LLMProvider.is_vision_default == True)
            .where(LLMProvider.is_enabled == True)
            .limit(1)
        )
        self._vision_provider = result.scalar_one_or_none()
        
        # If no vision provider set but primary is multimodal, use it
        if not self._vision_provider and self._primary_provider and self._primary_provider.is_multimodal:
            self._vision_provider = self._primary_provider
        
        # If no providers at all, check legacy multimodal setting
        if not self._primary_provider:
            try:
                from app.services.settings_service import SettingsService
                from app.core.settings_keys import SK
                legacy_multimodal = await SettingsService.get_bool(self.db, SK.LLM_MULTIMODAL)
                if legacy_multimodal:
                    # Create a virtual provider to indicate MM capability
                    self._legacy_multimodal = True
                    logger.debug("[VISION_ROUTER] Using legacy multimodal setting")
                else:
                    self._legacy_multimodal = False
            except Exception:
                self._legacy_multimodal = False
        else:
            self._legacy_multimodal = False
        
        self._loaded = True
        
        logger.debug(
            f"[VISION_ROUTER] Loaded providers: "
            f"primary={self._primary_provider.name if self._primary_provider else 'None'}, "
            f"vision={self._vision_provider.name if self._vision_provider else 'None'}, "
            f"legacy_mm={getattr(self, '_legacy_multimodal', False)}"
        )
    
    @property
    def primary_provider(self) -> Optional[LLMProvider]:
        return self._primary_provider
    
    @property
    def vision_provider(self) -> Optional[LLMProvider]:
        return self._vision_provider
    
    @property
    def has_vision_capability(self) -> bool:
        """Check if there's any way to process images."""
        # Legacy multimodal setting
        if getattr(self, '_legacy_multimodal', False):
            return True
        # Primary provider is multimodal
        if self._primary_provider and self._primary_provider.is_multimodal:
            return True
        # Separate vision provider available
        if self._vision_provider:
            return True
        return False
    
    def needs_vision_routing(self, attachments: List[Dict]) -> bool:
        """
        Check if we need to route images through vision model.
        
        Returns True if:
        - There are image attachments
        - Primary model is NOT multimodal (and not legacy MM)
        - Vision model IS available
        """
        if not attachments:
            return False
        
        has_images = any(att.get("type") == "image" for att in attachments)
        if not has_images:
            return False
        
        # Check legacy multimodal - if enabled, images go directly to model
        if getattr(self, '_legacy_multimodal', False):
            return False
        
        # No primary provider - route through vision if available
        if not self._primary_provider:
            return self._vision_provider is not None
        
        # If primary is multimodal, no routing needed
        if self._primary_provider.is_multimodal:
            return False
        
        # Primary is NOT MM - need vision provider for routing
        return self._vision_provider is not None
    
    def should_strip_images(self, attachments: List[Dict]) -> bool:
        """
        Check if we should strip images (no MM capability).
        
        Returns True if:
        - There are images
        - Primary model is NOT multimodal (or no provider configured)
        - No vision model available
        - Legacy multimodal setting is False
        """
        if not attachments:
            return False
        
        has_images = any(att.get("type") == "image" for att in attachments)
        if not has_images:
            return False
        
        # Check legacy multimodal setting first
        if getattr(self, '_legacy_multimodal', False):
            return False  # Legacy MM enabled, don't strip
        
        # No primary provider - check if vision provider exists
        if not self._primary_provider:
            # No primary and no vision = must strip
            return self._vision_provider is None
        
        # Primary is MM - no stripping needed
        if self._primary_provider.is_multimodal:
            return False
        
        # Primary is NOT MM - check if vision provider exists
        return self._vision_provider is None
    
    async def describe_images(
        self,
        images: List[Dict],
        user_message: str,
    ) -> List[Dict]:
        """
        Get descriptions for images using vision model.
        
        Args:
            images: List of image attachments with type='image', data=base64
            user_message: User's text for context
            
        Returns:
            List of dicts with {index, description}
        """
        if not self._vision_provider:
            return []
        
        import httpx
        from openai import AsyncOpenAI
        
        descriptions = []
        vision_prompt = self._vision_provider.vision_prompt or DEFAULT_VISION_PROMPT
        
        for i, img in enumerate(images):
            if img.get("type") != "image":
                continue
            
            image_data = img.get("data", "")
            mime_type = img.get("mime_type", "image/jpeg")
            
            if not image_data:
                descriptions.append({
                    "index": i,
                    "description": "[Image data not available]",
                })
                continue
            
            # Build vision request
            prompt = vision_prompt.format(user_message=user_message[:500])
            
            try:
                import httpx
                client = AsyncOpenAI(
                    base_url=self._vision_provider.base_url,
                    api_key=self._vision_provider.api_key or "not-needed",
                    timeout=httpx.Timeout(
                        connect=30.0,
                        read=float(self._vision_provider.timeout),
                        write=30.0,
                        pool=10.0,
                    ),
                )
                
                response = await client.chat.completions.create(
                    model=self._vision_provider.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}",
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3,  # Lower temp for factual description
                )
                
                description = response.choices[0].message.content
                descriptions.append({
                    "index": i,
                    "description": description,
                    "model": self._vision_provider.name,
                })
                
                logger.info(f"[VISION_ROUTER] Got description for image {i} ({len(description)} chars)")
                
            except Exception as e:
                logger.error(f"[VISION_ROUTER] Failed to describe image {i}: {e}")
                descriptions.append({
                    "index": i,
                    "description": f"[Failed to analyze image: {str(e)[:100]}]",
                    "error": str(e),
                })
        
        return descriptions
    
    async def process_attachments_for_routing(
        self,
        attachments: List[Dict],
        user_message: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process attachments for hybrid routing.
        
        If routing needed:
        - Send images to vision model
        - Replace images with descriptions
        - Return modified attachments
        
        Args:
            attachments: Original attachments
            user_message: User's text message
            
        Returns:
            Tuple of (modified_attachments, image_descriptions)
        """
        await self.load_providers()
        
        if not attachments:
            return attachments, []
        
        # Debug: Log current state
        has_images = any(att.get("type") == "image" for att in attachments)
        logger.info(
            f"[VISION_ROUTER] Processing: has_images={has_images}, "
            f"primary={self._primary_provider.name if self._primary_provider else 'None'}, "
            f"primary_mm={self._primary_provider.is_multimodal if self._primary_provider else 'N/A'}, "
            f"vision={self._vision_provider.name if self._vision_provider else 'None'}, "
            f"legacy_mm={getattr(self, '_legacy_multimodal', False)}"
        )
        
        # Check if we need to strip images (no MM at all)
        should_strip = self.should_strip_images(attachments)
        logger.info(f"[VISION_ROUTER] should_strip_images={should_strip}")
        
        if should_strip:
            logger.warning("[VISION_ROUTER] No multimodal capability - stripping images")
            filtered = [a for a in attachments if a.get("type") != "image"]
            stripped_count = len(attachments) - len(filtered)
            
            if stripped_count > 0:
                logger.info(f"[VISION_ROUTER] Stripped {stripped_count} image(s) - no vision model configured")
            
            # Return filtered attachments with empty descriptions (no note to LLM)
            return filtered, []
        
        # Check if routing is needed
        if not self.needs_vision_routing(attachments):
            return attachments, []
        
        # Extract images for processing
        images = [a for a in attachments if a.get("type") == "image"]
        other_attachments = [a for a in attachments if a.get("type") != "image"]
        
        if not images:
            return attachments, []
        
        logger.info(f"[VISION_ROUTER] Processing {len(images)} images through vision model")
        
        # Get descriptions
        descriptions = await self.describe_images(images, user_message)
        
        # Build modified attachments with descriptions instead of images
        modified = other_attachments.copy()
        
        for desc in descriptions:
            if desc.get("error"):
                # Keep error note
                modified.append({
                    "type": "image_description",
                    "description": desc["description"],
                    "error": True,
                })
            else:
                # Add description as context
                modified.append({
                    "type": "image_description",
                    "description": desc["description"],
                    "source_model": desc.get("model", "vision"),
                })
        
        return modified, descriptions


async def get_vision_router(db: AsyncSession) -> VisionRouter:
    """Get a configured vision router."""
    router = VisionRouter(db)
    await router.load_providers()
    return router


async def get_active_providers(db: AsyncSession) -> Tuple[Optional[LLMProvider], Optional[LLMProvider]]:
    """
    Get the active primary and vision providers.
    
    Returns:
        Tuple of (primary_provider, vision_provider)
    """
    # Get default provider
    result = await db.execute(
        select(LLMProvider)
        .where(LLMProvider.is_default == True)
        .where(LLMProvider.is_enabled == True)
        .limit(1)
    )
    primary = result.scalar_one_or_none()
    
    # Get vision provider
    result = await db.execute(
        select(LLMProvider)
        .where(LLMProvider.is_vision_default == True)
        .where(LLMProvider.is_enabled == True)
        .limit(1)
    )
    vision = result.scalar_one_or_none()
    
    # If no vision provider but primary is MM, use it
    if not vision and primary and primary.is_multimodal:
        vision = primary
    
    return primary, vision


def format_image_descriptions_for_llm(descriptions: List[Dict]) -> str:
    """Format image descriptions for injection into LLM context."""
    if not descriptions:
        return ""
    
    parts = []
    for i, desc in enumerate(descriptions, 1):
        if desc.get("error"):
            parts.append(f"[Image {i}: {desc['description']}]")
        elif desc.get("description"):
            parts.append(f"[Image {i} Description: {desc['description']}]")
    
    return "\n\n".join(parts)


# Export
__all__ = [
    'VisionRouter',
    'get_vision_router',
    'get_active_providers',
    'format_image_descriptions_for_llm',
]
