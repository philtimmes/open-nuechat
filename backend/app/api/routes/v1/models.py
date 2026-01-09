"""
OpenAI-compatible /v1/models endpoint

Lists available models including:
- Base LLM models from the configured backend
- Custom GPTs/Assistants the user has access to
"""
import logging
import re
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


def assistant_name_to_model_id(name: str) -> str:
    """
    Convert assistant name to a clean model ID.
    
    - Replace spaces with underscores
    - Remove any characters that aren't alphanumeric, dash, or underscore
    - Lowercase for consistency
    
    Example: "My Cool Assistant!" -> "my_cool_assistant"
    """
    # Replace spaces with underscores
    clean = name.replace(" ", "_")
    # Remove any non-alphanumeric characters except - and _
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', clean)
    # Lowercase
    clean = clean.lower()
    # Remove consecutive underscores/dashes
    clean = re.sub(r'[-_]+', '_', clean)
    # Strip leading/trailing underscores
    clean = clean.strip('_-')
    return clean


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
    
    Model IDs for Custom GPTs use cleaned assistant names:
    - Spaces replaced with underscores
    - Non-alphanumeric characters (except - and _) removed
    - Lowercased
    
    Example: "My Research Assistant!" -> "my_research_assistant"
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
        model_id = assistant_name_to_model_id(assistant.name)
        models.append(ModelObject(
            id=model_id,
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
        prefs = user.preferences or {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
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
                model_id = assistant_name_to_model_id(assistant.name)
                # Avoid duplicates
                if model_id not in [m.id for m in models]:
                    models.append(ModelObject(
                        id=model_id,
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
        # Filter to only allowed assistants - check by ID
        allowed_ids = set(api_key.allowed_assistants)
        # We need to re-query to check IDs since we're now using names
        result = await db.execute(
            select(CustomAssistant).where(
                CustomAssistant.id.in_(allowed_ids)
            )
        )
        allowed_assistants = result.scalars().all()
        allowed_names = {assistant_name_to_model_id(a.name) for a in allowed_assistants}
        
        models = [
            m for m in models 
            if not m.is_custom_gpt or m.id in allowed_names
        ]
    
    return ModelsListResponse(data=models)


@router.get("/models/{model_id:path}", response_model=ModelObject)
async def get_model(
    model_id: str,
    auth: tuple[User, APIKey] = Depends(require_scope("models")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get details of a specific model.
    
    Model ID can be:
    - A base model name (e.g., "llama3.2", "gpt-4")
    - A cleaned assistant name (e.g., "my_research_assistant")
    - Legacy format with gpt: prefix still supported
    """
    user, api_key = auth
    
    # Check if it's a legacy gpt: format (still supported for backward compatibility)
    if model_id.startswith("gpt:"):
        assistant_id = model_id[4:]  # Remove "gpt:" prefix
        
        result = await db.execute(
            select(CustomAssistant).where(
                CustomAssistant.id == assistant_id,
            )
        )
        assistant = result.scalar_one_or_none()
    else:
        # Try to find by cleaned name
        # We need to check all assistants the user has access to
        result = await db.execute(
            select(CustomAssistant).where(
                or_(
                    CustomAssistant.owner_id == user.id,
                    CustomAssistant.is_public == True,
                )
            )
        )
        all_assistants = result.scalars().all()
        
        # Find assistant matching the model_id
        assistant = None
        for a in all_assistants:
            if assistant_name_to_model_id(a.name) == model_id:
                assistant = a
                break
    
    if not assistant:
        # Not a custom GPT, assume base model
        return ModelObject(
            id=model_id,
            created=int(datetime.now(timezone.utc).timestamp()),
            owned_by="system",
            description=f"Base model: {model_id}",
        )
    
    # Check API key restrictions
    if api_key.allowed_assistants and assistant.id not in api_key.allowed_assistants:
        raise HTTPException(status_code=403, detail="Access to this model is not allowed")
    
    # Check access
    is_owner = assistant.owner_id == user.id
    is_public = assistant.is_public
    
    # Check subscription
    import json
    prefs = user.preferences or {}
    if isinstance(prefs, str):
        prefs = json.loads(prefs)
    subscribed_ids = prefs.get("subscribed_assistants", [])
    is_subscribed = assistant.id in subscribed_ids
    
    if not (is_owner or is_public or is_subscribed):
        raise HTTPException(status_code=403, detail="Access denied to this model")
    
    return ModelObject(
        id=assistant_name_to_model_id(assistant.name),
        created=int(assistant.created_at.timestamp()) if assistant.created_at else int(datetime.now(timezone.utc).timestamp()),
        owned_by=user.email if is_owner else "marketplace",
        description=assistant.description,
        is_custom_gpt=True,
        root=assistant.model,
    )
