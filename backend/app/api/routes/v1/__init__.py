"""
OpenAI-Compatible API v1 Routes

Provides drop-in replacement endpoints for OpenAI API:
- GET  /v1/models
- POST /v1/chat/completions
- POST /v1/images/generations
- POST /v1/embeddings
"""
from fastapi import APIRouter

from .models import router as models_router
from .completions import router as completions_router
from .images import router as images_router
from .embeddings import router as embeddings_router

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible API"])

router.include_router(models_router)
router.include_router(completions_router)
router.include_router(images_router)
router.include_router(embeddings_router)

# API Scopes for access control
API_SCOPES = {
    "models:list": "List available models",
    "chat:completions": "Create chat completions",
    "images:generate": "Generate images",
    "embeddings:create": "Create embeddings",
}

__all__ = ["router", "API_SCOPES"]
