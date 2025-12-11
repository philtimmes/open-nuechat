"""
Tools API routes - MCP and OpenAPI tool management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, Integer
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timezone

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.models.models import User, Tool, ToolUsage, ToolType
from app.services.tool_service import (
    tool_service, encrypt_api_key, decrypt_api_key,
    MCPClient, OpenAPIClient
)

router = APIRouter(prefix="/tools", tags=["tools"])


# === Schemas ===

class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    tool_type: str = Field(..., pattern="^(mcp|openapi)$")
    url: str = Field(..., min_length=1)
    api_key: Optional[str] = None
    is_public: bool = False
    config: Optional[Dict[str, Any]] = None
    enabled_operations: Optional[List[str]] = None  # For OpenAPI


class ToolUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    url: Optional[str] = None
    api_key: Optional[str] = None  # Send empty string to clear
    is_public: Optional[bool] = None
    is_enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    enabled_operations: Optional[List[str]] = None


class ToolResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    tool_type: str
    url: str
    has_api_key: bool
    is_public: bool
    is_enabled: bool
    config: Optional[Dict[str, Any]]
    enabled_operations: Optional[List[str]]
    schema_cache: Optional[List[Dict[str, Any]]]
    last_schema_fetch: Optional[datetime]
    created_by: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ToolExecuteRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = {}


class ToolUsageResponse(BaseModel):
    id: str
    tool_id: Optional[str]
    tool_name: str
    operation: Optional[str]
    success: bool
    result_summary: Optional[str]
    result_url: Optional[str]
    error_message: Optional[str]
    duration_ms: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


# === Helper ===

def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin access"""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def tool_to_response(tool: Tool) -> ToolResponse:
    return ToolResponse(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        tool_type=tool.tool_type.value,
        url=tool.url,
        has_api_key=bool(tool.api_key_encrypted),
        is_public=tool.is_public,
        is_enabled=tool.is_enabled,
        config=tool.config,
        enabled_operations=tool.enabled_operations,
        schema_cache=tool.schema_cache,
        last_schema_fetch=tool.last_schema_fetch,
        created_by=tool.created_by,
        created_at=tool.created_at,
    )


# === Admin Routes ===

@router.get("", response_model=List[ToolResponse])
async def list_tools(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all tools (admin sees all, users see public only)"""
    query = select(Tool)
    
    if not user.is_admin:
        query = query.where(Tool.is_public == True, Tool.is_enabled == True)
    
    query = query.order_by(Tool.created_at.desc())
    result = await db.execute(query)
    tools = result.scalars().all()
    
    return [tool_to_response(t) for t in tools]


@router.post("", response_model=ToolResponse)
async def create_tool(
    data: ToolCreate,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new tool (admin only)"""
    
    # Validate URL by attempting to fetch schema
    api_key = data.api_key
    try:
        if data.tool_type == "mcp":
            client = MCPClient(data.url, api_key, data.config)
            schema = await client.discover_tools()
        else:
            client = OpenAPIClient(data.url, api_key, data.config)
            schema = await client.discover_tools()
            
            # Filter if specific operations requested
            if data.enabled_operations:
                schema = [s for s in schema if s['name'] in data.enabled_operations]
    except Exception as e:
        import traceback
        error_detail = str(e)
        # Include more context for debugging
        if "406" in error_detail:
            error_detail += " (HTTP 406 Not Acceptable - server may not support the requested content type)"
        elif "404" in error_detail:
            error_detail += " (HTTP 404 - endpoint not found, check URL)"
        elif "401" in error_detail or "403" in error_detail:
            error_detail += " (Authentication error - check API key)"
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to connect to tool: {error_detail}"
        )
    
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tools discovered at this endpoint. Check if the URL is correct and the server is running."
        )
    
    tool = Tool(
        name=data.name,
        description=data.description,
        tool_type=ToolType(data.tool_type),
        url=data.url,
        api_key_encrypted=encrypt_api_key(api_key) if api_key else None,
        is_public=data.is_public,
        config=data.config,
        enabled_operations=data.enabled_operations,
        schema_cache=schema,
        last_schema_fetch=datetime.now(timezone.utc),
        created_by=user.id,
    )
    
    db.add(tool)
    await db.commit()
    await db.refresh(tool)
    
    return tool_to_response(tool)


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a tool by ID"""
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    if not user.is_admin and not tool.is_public:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return tool_to_response(tool)


@router.patch("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    data: ToolUpdate,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update a tool (admin only)"""
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    if data.name is not None:
        tool.name = data.name
    if data.description is not None:
        tool.description = data.description
    if data.url is not None:
        tool.url = data.url
        # Clear cache when URL changes
        tool.schema_cache = None
        tool.last_schema_fetch = None
    if data.api_key is not None:
        if data.api_key == "":
            tool.api_key_encrypted = None
        else:
            tool.api_key_encrypted = encrypt_api_key(data.api_key)
    if data.is_public is not None:
        tool.is_public = data.is_public
    if data.is_enabled is not None:
        tool.is_enabled = data.is_enabled
    if data.config is not None:
        tool.config = data.config
    if data.enabled_operations is not None:
        tool.enabled_operations = data.enabled_operations
    
    await db.commit()
    await db.refresh(tool)
    
    return tool_to_response(tool)


@router.delete("/{tool_id}")
async def delete_tool(
    tool_id: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a tool (admin only)"""
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    await db.delete(tool)
    await db.commit()
    
    return {"message": "Tool deleted", "id": tool_id}


class ToolProbeRequest(BaseModel):
    """Request to probe a tool URL"""
    url: str
    tool_type: str = Field(..., pattern="^(mcp|openapi)$")
    api_key: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.post("/probe")
async def probe_tool_url(
    data: ToolProbeRequest,
    user: User = Depends(require_admin),
):
    """
    Probe a tool URL to check connectivity and discover tools.
    Use this to test before creating a tool.
    """
    try:
        if data.tool_type == "mcp":
            client = MCPClient(data.url, data.api_key, data.config)
            schema = await client.discover_tools()
        else:
            client = OpenAPIClient(data.url, data.api_key, data.config)
            schema = await client.discover_tools()
        
        return {
            "success": True,
            "tool_count": len(schema),
            "tools": [
                {
                    "name": t.get("name"),
                    "description": t.get("description", "")[:100]
                }
                for t in schema[:20]  # Limit to first 20
            ],
            "message": f"Successfully discovered {len(schema)} tool(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to connect: {str(e)}"
        }


@router.post("/{tool_id}/refresh")
async def refresh_tool_schema(
    tool_id: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Refresh a tool's schema cache (admin only)"""
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    try:
        schema = await tool_service.refresh_tool_schema(db, tool)
        return {
            "message": "Schema refreshed",
            "tool_count": len(schema),
            "tools": [s.get('name') for s in schema]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to refresh schema: {str(e)}"
        )


@router.post("/{tool_id}/test")
async def test_tool(
    tool_id: str,
    data: ToolExecuteRequest,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test execute a tool (admin only)"""
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Execute without recording usage
    api_key = decrypt_api_key(tool.api_key_encrypted) if tool.api_key_encrypted else None
    
    try:
        if tool.tool_type == ToolType.MCP:
            client = MCPClient(tool.url, api_key, tool.config)
            success, result_data, result_url = await client.execute_tool(data.tool_name, data.params)
        else:
            client = OpenAPIClient(tool.url, api_key, tool.config)
            success, result_data, result_url = await client.execute_operation(data.tool_name, data.params)
        
        return {
            "success": success,
            "result": result_data,
            "result_url": result_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# === Available Tools for LLM ===

@router.get("/available/schemas")
async def get_available_tool_schemas(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all available tool schemas for the current user.
    This is what the LLM uses to know what tools are available.
    """
    tools = await tool_service.get_available_tools(db, user.id, user.is_admin)
    
    # Format for LLM consumption
    formatted = []
    for tool in tools:
        formatted.append({
            "tool_id": tool.get('_tool_id'),
            "source_name": tool.get('_tool_name'),
            "source_type": tool.get('_tool_type'),
            "name": tool.get('name'),
            "description": tool.get('description', ''),
            "parameters": tool.get('parameters', []),
        })
    
    return {"tools": formatted}


# === Tool Usage / Citations ===

@router.get("/usage/chat/{chat_id}", response_model=List[ToolUsageResponse])
async def get_chat_tool_usage(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all tool usages for a chat (for citations)"""
    # Verify user has access to chat
    from app.models.models import Chat
    chat_result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = chat_result.scalar_one_or_none()
    
    if not chat or chat.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    usages = await tool_service.get_chat_tool_usage(db, chat_id)
    return [ToolUsageResponse.model_validate(u) for u in usages]


@router.get("/usage/message/{message_id}", response_model=List[ToolUsageResponse])
async def get_message_tool_usage(
    message_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get tool usages for a specific message"""
    usages = await tool_service.get_tool_usage_for_message(db, message_id)
    return [ToolUsageResponse.model_validate(u) for u in usages]


# === Tool Usage Stats (Admin) ===

class ToolUsageStats(BaseModel):
    """Statistics for tool usage"""
    tool_id: str
    tool_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_duration_ms: int
    avg_duration_ms: float
    last_used: Optional[datetime]
    unique_users: int


class AllToolsUsageStats(BaseModel):
    """Aggregated stats for all tools"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    tools: List[ToolUsageStats]


@router.get("/usage/stats", response_model=AllToolsUsageStats)
async def get_tool_usage_stats(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics for all tools (admin only)"""
    from sqlalchemy import func, distinct
    
    # Get per-tool stats
    query = (
        select(
            ToolUsage.tool_id,
            ToolUsage.tool_name,
            func.count(ToolUsage.id).label('total_calls'),
            func.sum(func.cast(ToolUsage.success, Integer)).label('successful_calls'),
            func.sum(ToolUsage.duration_ms).label('total_duration_ms'),
            func.max(ToolUsage.created_at).label('last_used'),
            func.count(distinct(ToolUsage.user_id)).label('unique_users'),
        )
        .group_by(ToolUsage.tool_id, ToolUsage.tool_name)
        .order_by(func.count(ToolUsage.id).desc())
    )
    
    result = await db.execute(query)
    rows = result.all()
    
    tools_stats = []
    total_calls = 0
    total_success = 0
    total_failed = 0
    
    for row in rows:
        successful = row.successful_calls or 0
        total = row.total_calls or 0
        failed = total - successful
        total_duration = row.total_duration_ms or 0
        avg_duration = total_duration / total if total > 0 else 0
        
        tools_stats.append(ToolUsageStats(
            tool_id=row.tool_id or 'unknown',
            tool_name=row.tool_name,
            total_calls=total,
            successful_calls=successful,
            failed_calls=failed,
            total_duration_ms=total_duration,
            avg_duration_ms=round(avg_duration, 2),
            last_used=row.last_used,
            unique_users=row.unique_users or 0,
        ))
        
        total_calls += total
        total_success += successful
        total_failed += failed
    
    return AllToolsUsageStats(
        total_calls=total_calls,
        successful_calls=total_success,
        failed_calls=total_failed,
        tools=tools_stats,
    )


@router.delete("/usage/stats")
async def reset_all_tool_usage(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reset usage statistics for all tools (admin only)"""
    from sqlalchemy import delete
    
    result = await db.execute(delete(ToolUsage))
    await db.commit()
    
    return {
        "message": "All tool usage statistics reset",
        "deleted_count": result.rowcount,
    }


@router.delete("/usage/stats/{tool_id}")
async def reset_tool_usage(
    tool_id: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reset usage statistics for a specific tool (admin only)"""
    from sqlalchemy import delete
    
    # Check tool exists
    result = await db.execute(select(Tool).where(Tool.id == tool_id))
    tool = result.scalar_one_or_none()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Delete usage records for this tool
    result = await db.execute(
        delete(ToolUsage).where(ToolUsage.tool_id == tool_id)
    )
    await db.commit()
    
    return {
        "message": f"Usage statistics reset for {tool.name}",
        "tool_id": tool_id,
        "deleted_count": result.rowcount,
    }
