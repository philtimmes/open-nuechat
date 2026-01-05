"""
OpenAI-compatible /v1/models endpoint

Lists available models including:
- Base LLM models from the configured backend
- Custom GPTs/Assistants the user has access to
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.db.database import get_db
from app.models import User, APIKey, CustomAssistant
from app.api.routes.api_keys import get_api_key_user, require_scope
from app.services.settings_service import SettingsService

logger = logging.getLogger(__name__)

router = APIRouter()


# === Schemas (OpenAI-compatible) ===

class ModelObject(BaseModel):
    """OpenAI-compatible model object"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[dict] = []
    root: Optional[str] = None
    parent: Optional[str] = None
    # Extension fields for NueChat
    description: Optional[str] = None
    context_length: Optional[int] = None
    is_custom_gpt: bool = False


class ModelsListResponse(BaseModel):
    """OpenAI-compatible models list response"""
    object: str = "list"
    data: List[ModelObject]


# === Endpoints ===

@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    auth: tuple[User, APIKey] = Depends(require_scope("models")),
    db: AsyncSession = Depends(get_db),
):
    """
    List available models.
    
    Returns base models from the LLM backend plus any Custom GPTs
    the user has created or subscribed to.
    
    Model IDs:
    - Base models: Use the model name directly (e.g., "llama3.2", "gpt-4")
    - Custom GPTs: Prefixed with "gpt:" (e.g., "gpt:my-assistant-id")
    """
    user, api_key = auth
    models: List[ModelObject] = []
    
    # 1. Add base models from settings
    try:
        default_model = await SettingsService.get(db, "llm_model")
        if not default_model:
            default_model = "llama3.2"
        models.append(ModelObject(
            id=default_model,
            created=int(datetime.now(timezone.utc).timestamp()),
            owned_by="system",
            description="Default LLM model",
        ))
    except Exception as e:
        logger.warning(f"Could not get default model: {e}")
    
    # 2. Get user's custom assistants
    result = await db.execute(
        select(CustomAssistant).where(
            CustomAssistant.owner_id == user.id,
        )
    )
    user_assistants = result.scalars().all()
    
    for assistant in user_assistants:
        models.append(ModelObject(
            id=f"gpt:{assistant.id}",
            created=int(assistant.created_at.timestamp()) if assistant.created_at else int(datetime.now(timezone.utc).timestamp()),
            owned_by=user.email or user.id,
            description=assistant.description,
            is_custom_gpt=True,
            root=assistant.model,  # The underlying model
        ))
    
    # 3. Get subscribed assistants (public ones user subscribed to)
    try:
        # Parse subscribed IDs from user preferences
        import json
        prefs = json.loads(user.preferences or "{}")
        subscribed_ids = prefs.get("subscribed_assistants", [])
        
        if subscribed_ids:
            result = await db.execute(
                select(CustomAssistant).where(
                    CustomAssistant.id.in_(subscribed_ids),
                    CustomAssistant.is_public == True,
                )
            )
            subscribed = result.scalars().all()
            
            for assistant in subscribed:
                # Avoid duplicates
                if f"gpt:{assistant.id}" not in [m.id for m in models]:
                    models.append(ModelObject(
                        id=f"gpt:{assistant.id}",
                        created=int(assistant.created_at.timestamp()) if assistant.created_at else int(datetime.now(timezone.utc).timestamp()),
                        owned_by="marketplace",
                        description=assistant.description,
                        is_custom_gpt=True,
                        root=assistant.model,
                    ))
    except Exception as e:
        logger.warning(f"Could not load subscribed assistants: {e}")
    
    # 4. Check API key restrictions
    if api_key.allowed_assistants:
        # Filter to only allowed assistants
        allowed_ids = set(api_key.allowed_assistants)
        models = [
            m for m in models 
            if not m.is_custom_gpt or m.id.replace("gpt:", "") in allowed_ids
        ]
    
    return ModelsListResponse(data=models)


@router.get("/models/{model_id}", response_model=ModelObject)
async def get_model(
    model_id: str,
    auth: tuple[User, APIKey] = Depends(require_scope("models")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get details of a specific model.
    """
    user, api_key = auth
    
    # Check if it's a custom GPT
    if model_id.startswith("gpt:"):
        assistant_id = model_id[4:]  # Remove "gpt:" prefix
        
        # Check API key restrictions
        if api_key.allowed_assistants and assistant_id not in api_key.allowed_assistants:
            raise HTTPException(status_code=403, detail="Access to this model is not allowed")
        
        result = await db.execute(
            select(CustomAssistant).where(
                CustomAssistant.id == assistant_id,
            )
        )
        assistant = result.scalar_one_or_none()
        
        if not assistant:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check access
        is_owner = assistant.owner_id == user.id
        is_public = assistant.is_public
        
        # Check subscription
        import json
        prefs = json.loads(user.preferences or "{}")
        subscribed_ids = prefs.get("subscribed_assistants", [])
        is_subscribed = assistant.id in subscribed_ids
        
        if not (is_owner or is_public or is_subscribed):
            raise HTTPException(status_code=403, detail="Access denied to this model")
        
        return ModelObject(
            id=model_id,
            created=int(assistant.created_at.timestamp()) if assistant.created_at else int(datetime.now(timezone.utc).timestamp()),
            owned_by=user.email if is_owner else "marketplace",
            description=assistant.description,
            is_custom_gpt=True,
            root=assistant.model,
        )
    
    # Base model - just return basic info
    return ModelObject(
        id=model_id,
        created=int(datetime.now(timezone.utc).timestamp()),
        owned_by="system",
        description=f"Base model: {model_id}",
    )
