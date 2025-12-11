"""
Filter Management API Routes

Allows runtime configuration and monitoring of the filter system.

Filter Types:
- OverrideToLLM: Filters for user input going TO the LLM
- OverrideFromLLM: Filters for LLM output coming FROM the LLM

Priority Levels (executed in order):
- HIGHEST (0): Runs first
- HIGH (25)
- MEDIUM (50): Default
- LOW (75)
- LEAST (100): Runs last
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_admin_user
from app.models.models import User
from app.filters import (
    get_filter_registry,
    get_chat_filters,
    setup_default_filters,
    setup_minimal_filters,
    setup_strict_filters,
    FilterContext,
    Priority,
)

router = APIRouter()


# === Schemas ===

class FilterInfo(BaseModel):
    """Information about a registered filter."""
    name: str
    priority: str
    priority_value: int
    enabled: bool
    type: str


class RegistryStatusResponse(BaseModel):
    """Status of the global filter registry."""
    default_to_llm_count: int
    default_from_llm_count: int
    active_chat_managers: int
    chat_ids: List[str]


class ChatFiltersResponse(BaseModel):
    """Status of filters for a specific chat."""
    chat_id: str
    to_llm_filters: List[FilterInfo]
    from_llm_filters: List[FilterInfo]


class FilterConfigUpdate(BaseModel):
    """Request to update filter configuration."""
    name: str
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class FilterPresetRequest(BaseModel):
    """Request to apply a filter preset."""
    preset: str  # "default", "minimal", "strict"


class TestFilterRequest(BaseModel):
    """Request to test filters."""
    content: str
    direction: str = "to_llm"  # "to_llm" or "from_llm"
    chat_id: str = "test"


# === Endpoints ===

@router.get("/registry/status", response_model=RegistryStatusResponse)
async def get_registry_status(
    current_user: User = Depends(get_current_user),
):
    """
    Get the status of the global filter registry.
    
    Shows the number of default filters and active chat managers.
    """
    registry = get_filter_registry()
    status = registry.get_status()
    
    return RegistryStatusResponse(**status)


@router.get("/chat/{chat_id}", response_model=ChatFiltersResponse)
async def get_chat_filter_status(
    chat_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Get the filter status for a specific chat.
    
    Returns all ToLLM and FromLLM filters with their priorities.
    """
    manager = get_chat_filters(chat_id)
    status = manager.get_status()
    
    return ChatFiltersResponse(
        chat_id=status["chat_id"],
        to_llm_filters=[FilterInfo(**f) for f in status["to_llm_filters"]],
        from_llm_filters=[FilterInfo(**f) for f in status["from_llm_filters"]],
    )


@router.post("/chat/{chat_id}/enable/{filter_name}")
async def enable_chat_filter(
    chat_id: str,
    filter_name: str,
    direction: str = "both",
    current_user: User = Depends(get_admin_user),
):
    """
    Enable a filter for a specific chat.
    
    Args:
        chat_id: The chat ID
        filter_name: Name of the filter to enable
        direction: "to_llm", "from_llm", or "both"
    """
    manager = get_chat_filters(chat_id)
    enabled_count = 0
    
    if direction in ("to_llm", "both"):
        if manager.to_llm_chain.enable(filter_name):
            enabled_count += 1
    
    if direction in ("from_llm", "both"):
        if manager.from_llm_chain.enable(filter_name):
            enabled_count += 1
    
    if enabled_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter '{filter_name}' not found"
        )
    
    return {"message": f"Filter '{filter_name}' enabled", "count": enabled_count}


@router.post("/chat/{chat_id}/disable/{filter_name}")
async def disable_chat_filter(
    chat_id: str,
    filter_name: str,
    direction: str = "both",
    current_user: User = Depends(get_admin_user),
):
    """
    Disable a filter for a specific chat.
    
    Args:
        chat_id: The chat ID
        filter_name: Name of the filter to disable
        direction: "to_llm", "from_llm", or "both"
    """
    manager = get_chat_filters(chat_id)
    disabled_count = 0
    
    if direction in ("to_llm", "both"):
        if manager.to_llm_chain.disable(filter_name):
            disabled_count += 1
    
    if direction in ("from_llm", "both"):
        if manager.from_llm_chain.disable(filter_name):
            disabled_count += 1
    
    if disabled_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter '{filter_name}' not found"
        )
    
    return {"message": f"Filter '{filter_name}' disabled", "count": disabled_count}


@router.post("/chat/{chat_id}/configure")
async def configure_chat_filter(
    chat_id: str,
    update: FilterConfigUpdate,
    direction: str = "both",
    current_user: User = Depends(get_admin_user),
):
    """
    Update configuration for a filter in a specific chat.
    
    Args:
        chat_id: The chat ID
        update: Configuration update containing filter name and new settings
        direction: "to_llm", "from_llm", or "both"
    """
    manager = get_chat_filters(chat_id)
    updated_count = 0
    
    def update_filter(chain, name: str, update: FilterConfigUpdate):
        f = chain.get(name)
        if f:
            if update.enabled is not None:
                f.enabled = update.enabled
            if update.config:
                f.configure(**update.config)
            return True
        return False
    
    if direction in ("to_llm", "both"):
        if update_filter(manager.to_llm_chain, update.name, update):
            updated_count += 1
    
    if direction in ("from_llm", "both"):
        if update_filter(manager.from_llm_chain, update.name, update):
            updated_count += 1
    
    if updated_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter '{update.name}' not found"
        )
    
    return {"message": f"Filter '{update.name}' configured", "count": updated_count}


@router.post("/preset")
async def apply_preset(
    request: FilterPresetRequest,
    current_user: User = Depends(get_admin_user),
):
    """
    Apply a filter preset configuration.
    
    This resets the global registry and applies new default filters.
    All existing chat filter managers will be cleared.
    
    Available presets:
    - "default": Standard security with moderate settings
    - "minimal": Only essential security filters for low latency
    - "strict": All filters with aggressive settings for high security
    """
    preset_map = {
        "default": setup_default_filters,
        "minimal": setup_minimal_filters,
        "strict": setup_strict_filters,
    }
    
    setup_func = preset_map.get(request.preset)
    if not setup_func:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown preset '{request.preset}'. Available: {list(preset_map.keys())}"
        )
    
    setup_func()
    
    registry = get_filter_registry()
    status_info = registry.get_status()
    
    return {
        "message": f"Applied '{request.preset}' preset",
        "default_to_llm_count": status_info["default_to_llm_count"],
        "default_from_llm_count": status_info["default_from_llm_count"],
    }


@router.get("/chat/{chat_id}/filter/{filter_name}")
async def get_filter_details(
    chat_id: str,
    filter_name: str,
    direction: str = "to_llm",
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed information about a specific filter in a chat.
    """
    manager = get_chat_filters(chat_id)
    
    chain = manager.to_llm_chain if direction == "to_llm" else manager.from_llm_chain
    f = chain.get(filter_name)
    
    if not f:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter '{filter_name}' not found in {direction} chain"
        )
    
    return {
        "name": f.name,
        "type": f.__class__.__name__,
        "priority": f.priority.name,
        "priority_value": f.priority.value,
        "enabled": f.enabled,
        "config": f._config,
        "docstring": f.__class__.__doc__,
    }


@router.post("/test")
async def test_filters(
    request: TestFilterRequest,
    current_user: User = Depends(get_admin_user),
):
    """
    Test the filter chain with sample content.
    
    Useful for debugging and validating filter configurations.
    
    Args:
        request: Test request containing content and direction
    """
    manager = get_chat_filters(request.chat_id)
    
    context = FilterContext(
        user_id=str(current_user.id),
        chat_id=request.chat_id,
    )
    
    if request.direction == "to_llm":
        result = await manager.process_to_llm(request.content, context)
    else:
        result = await manager.process_from_llm(request.content, context)
    
    return {
        "original": request.content,
        "filtered": result.content,
        "modified": result.modified,
        "blocked": result.blocked,
        "block_reason": result.block_reason,
        "metadata": result.metadata,
        "context_metadata": context.metadata,
    }


@router.delete("/chat/{chat_id}")
async def remove_chat_filters(
    chat_id: str,
    current_user: User = Depends(get_admin_user),
):
    """
    Remove the filter manager for a specific chat.
    
    The next request for this chat will create a new manager with default filters.
    """
    registry = get_filter_registry()
    removed = registry.remove_chat_manager(chat_id)
    
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No filter manager found for chat '{chat_id}'"
        )
    
    return {"message": f"Filter manager for chat '{chat_id}' removed"}


@router.delete("/registry/reset")
async def reset_registry(
    current_user: User = Depends(get_admin_user),
):
    """
    Reset the entire filter registry.
    
    Warning: This removes all default filters and chat managers.
    You should call /preset after this to restore filters.
    """
    registry = get_filter_registry()
    registry.reset()
    
    return {"message": "Filter registry reset"}


@router.get("/priorities")
async def list_priorities(
    current_user: User = Depends(get_current_user),
):
    """
    List all available priority levels.
    
    Filters are executed in priority order (lowest value first).
    """
    return {
        "priorities": [
            {"name": p.name, "value": p.value}
            for p in Priority
        ],
        "description": "Filters execute from HIGHEST (0) to LEAST (100)"
    }
