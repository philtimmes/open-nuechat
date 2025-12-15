"""
OpenAI-compatible /v1/images/generations endpoint

Generates images using the configured image generation service.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models import User, APIKey
from app.api.routes.api_keys import get_api_key_user, require_scope
from app.services.image_gen import (
    get_image_gen_client,
    extract_size_from_text,
    ASPECT_RATIO_SIZES,
)
from app.services.billing import BillingService

logger = logging.getLogger(__name__)

router = APIRouter()


# === Schemas (OpenAI-compatible) ===

class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request"""
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: Optional[str] = Field(default="z-image-turbo", description="Model to use")
    n: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images")
    size: Optional[str] = Field(
        default="1024x1024",
        description="Image size (e.g., '1024x1024', '1280x720', '720x1280')"
    )
    quality: Optional[str] = Field(default="standard", description="Quality: standard or hd")
    response_format: Optional[str] = Field(
        default="b64_json",
        description="Response format: 'url' or 'b64_json'"
    )
    style: Optional[str] = Field(default="natural", description="Style: vivid or natural")
    user: Optional[str] = Field(default=None, description="User identifier for abuse tracking")


class ImageData(BaseModel):
    """Generated image data"""
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """OpenAI-compatible image generation response"""
    created: int
    data: List[ImageData]


# === Helper Functions ===

def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string like '1024x1024' into (width, height)."""
    try:
        parts = size_str.lower().split("x")
        if len(parts) == 2:
            width = int(parts[0])
            height = int(parts[1])
            # Clamp to reasonable limits
            width = max(256, min(2048, width))
            height = max(256, min(2048, height))
            # Round to multiple of 64
            width = (width // 64) * 64
            height = (height // 64) * 64
            return width, height
    except (ValueError, IndexError):
        pass
    
    # Default to square
    return 1024, 1024


# === Endpoints ===

@router.post("/images/generations", response_model=ImageGenerationResponse)
async def create_image(
    request: ImageGenerationRequest,
    auth: tuple[User, APIKey] = Depends(require_scope("images")),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate images from a text prompt.
    
    Compatible with OpenAI's /v1/images/generations endpoint.
    
    **Supported sizes:**
    - Square: 512x512, 768x768, 1024x1024, 1080x1080
    - Landscape: 1280x720, 1920x1080, 1024x768, 1200x800
    - Portrait: 720x1280, 1080x1920, 768x1024, 800x1200
    
    **Note:** Currently only `b64_json` response format is fully supported.
    URL format will return a placeholder that expires quickly.
    """
    user, api_key = auth
    
    # Parse size
    width, height = parse_size(request.size or "1024x1024")
    
    # Get image generation client
    client = get_image_gen_client()
    
    # Check if service is available
    if not await client.is_available():
        raise HTTPException(
            status_code=503,
            detail="Image generation service is not available"
        )
    
    # Generate images (loop for n > 1)
    images: List[ImageData] = []
    
    for i in range(request.n or 1):
        try:
            result = await client.generate_image(
                prompt=request.prompt,
                width=width,
                height=height,
                seed=None,  # Random seed
                user_id=user.id,
                chat_id=None,
            )
            
            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Image generation failed")
                )
            
            image_data = ImageData(revised_prompt=request.prompt)
            
            if request.response_format == "b64_json":
                image_data.b64_json = result.get("image_base64")
            else:
                # For URL format, we'd need to store the image and return a URL
                # For now, return base64 anyway with a warning
                image_data.b64_json = result.get("image_base64")
                logger.warning("URL response format requested but returning b64_json")
            
            images.append(image_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Bill for image generation (flat rate per image)
    billing = BillingService()
    # Image generation costs ~1000 tokens equivalent per image
    await billing.record_usage(
        db=db,
        user_id=user.id,
        input_tokens=len(request.prompt.split()),
        output_tokens=1000 * len(images),  # Flat rate per image
        model="image-gen",
        source="v1_api",
    )
    
    return ImageGenerationResponse(
        created=int(datetime.now(timezone.utc).timestamp()),
        data=images,
    )


@router.get("/images/models")
async def list_image_models(
    auth: tuple[User, APIKey] = Depends(require_scope("images")),
):
    """
    List available image generation models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "z-image-turbo",
                "object": "model",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "owned_by": "system",
                "description": "Fast image generation model (Z-Image-Turbo)",
            }
        ]
    }
