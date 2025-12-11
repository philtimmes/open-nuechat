"""
Image routes for serving generated images
"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["Images"])

# Directory where generated images are stored
GENERATED_IMAGES_DIR = Path("/app/uploads/generated")


@router.get("/generated/{filename}")
async def get_generated_image(filename: str):
    """
    Serve a generated image by filename.
    
    Images are stored when generated and served here for persistence across page reloads.
    """
    # Security: only allow .png files and prevent directory traversal
    if not filename.endswith(".png") or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    image_path = GENERATED_IMAGES_DIR / filename
    
    if not image_path.exists():
        logger.warning(f"Generated image not found: {filename}")
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000",  # Cache for 1 year (immutable content)
        }
    )


@router.delete("/generated/{filename}")
async def delete_generated_image(filename: str):
    """
    Delete a generated image.
    """
    # Security checks
    if not filename.endswith(".png") or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    image_path = GENERATED_IMAGES_DIR / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        image_path.unlink()
        logger.debug(f"Deleted generated image: {filename}")
        return {"success": True, "message": f"Image {filename} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete image")
