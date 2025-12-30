"""
Agent Flows API routes
CRUD operations for user-created visual agent workflows
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional, List, Any
import logging

from app.db.database import get_db
from app.models.models import User, AgentFlow
from app.api.routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-flows", tags=["agent-flows"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class FlowNode(BaseModel):
    id: str
    type: str
    label: str
    position: dict
    config: Optional[dict] = None


class FlowConnection(BaseModel):
    id: str
    fromNodeId: str
    toNodeId: str


class FlowDefinition(BaseModel):
    nodes: List[FlowNode] = []
    connections: List[FlowConnection] = []


class AgentFlowCreate(BaseModel):
    name: str = "New Agent Flow"
    description: Optional[str] = None
    definition: Optional[dict] = None


class AgentFlowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[dict] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None


class AgentFlowResponse(BaseModel):
    id: str
    owner_id: str
    name: str
    description: Optional[str]
    definition: dict
    is_active: bool
    is_public: bool
    created_at: Optional[str]
    updated_at: Optional[str]
    
    class Config:
        from_attributes = True


# ============================================================================
# API Routes
# ============================================================================

@router.get("", response_model=List[AgentFlowResponse])
async def list_flows(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all agent flows for the current user"""
    result = await db.execute(
        select(AgentFlow)
        .where(AgentFlow.owner_id == user.id)
        .order_by(AgentFlow.updated_at.desc())
    )
    flows = result.scalars().all()
    
    return [AgentFlowResponse(
        id=f.id,
        owner_id=f.owner_id,
        name=f.name,
        description=f.description,
        definition=f.definition or {"nodes": [], "connections": []},
        is_active=f.is_active,
        is_public=f.is_public,
        created_at=f.created_at.isoformat() if f.created_at else None,
        updated_at=f.updated_at.isoformat() if f.updated_at else None,
    ) for f in flows]


@router.post("", response_model=AgentFlowResponse)
async def create_flow(
    data: AgentFlowCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new agent flow"""
    flow = AgentFlow(
        owner_id=user.id,
        name=data.name,
        description=data.description,
        definition=data.definition or {"nodes": [], "connections": []},
    )
    db.add(flow)
    await db.commit()
    await db.refresh(flow)
    
    logger.info(f"Created agent flow '{flow.name}' for user {user.id}")
    
    return AgentFlowResponse(
        id=flow.id,
        owner_id=flow.owner_id,
        name=flow.name,
        description=flow.description,
        definition=flow.definition,
        is_active=flow.is_active,
        is_public=flow.is_public,
        created_at=flow.created_at.isoformat() if flow.created_at else None,
        updated_at=flow.updated_at.isoformat() if flow.updated_at else None,
    )


@router.get("/{flow_id}", response_model=AgentFlowResponse)
async def get_flow(
    flow_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific agent flow"""
    result = await db.execute(
        select(AgentFlow).where(
            AgentFlow.id == flow_id,
            AgentFlow.owner_id == user.id
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    return AgentFlowResponse(
        id=flow.id,
        owner_id=flow.owner_id,
        name=flow.name,
        description=flow.description,
        definition=flow.definition or {"nodes": [], "connections": []},
        is_active=flow.is_active,
        is_public=flow.is_public,
        created_at=flow.created_at.isoformat() if flow.created_at else None,
        updated_at=flow.updated_at.isoformat() if flow.updated_at else None,
    )


@router.put("/{flow_id}", response_model=AgentFlowResponse)
async def update_flow(
    flow_id: str,
    data: AgentFlowUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an agent flow"""
    result = await db.execute(
        select(AgentFlow).where(
            AgentFlow.id == flow_id,
            AgentFlow.owner_id == user.id
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    # Update fields
    if data.name is not None:
        flow.name = data.name
    if data.description is not None:
        flow.description = data.description
    if data.definition is not None:
        flow.definition = data.definition
    if data.is_active is not None:
        flow.is_active = data.is_active
    if data.is_public is not None:
        flow.is_public = data.is_public
    
    await db.commit()
    await db.refresh(flow)
    
    logger.info(f"Updated agent flow '{flow.name}' for user {user.id}")
    
    return AgentFlowResponse(
        id=flow.id,
        owner_id=flow.owner_id,
        name=flow.name,
        description=flow.description,
        definition=flow.definition or {"nodes": [], "connections": []},
        is_active=flow.is_active,
        is_public=flow.is_public,
        created_at=flow.created_at.isoformat() if flow.created_at else None,
        updated_at=flow.updated_at.isoformat() if flow.updated_at else None,
    )


@router.delete("/{flow_id}")
async def delete_flow(
    flow_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an agent flow"""
    result = await db.execute(
        select(AgentFlow).where(
            AgentFlow.id == flow_id,
            AgentFlow.owner_id == user.id
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    await db.delete(flow)
    await db.commit()
    
    logger.info(f"Deleted agent flow '{flow.name}' for user {user.id}")
    
    return {"status": "deleted", "id": flow_id}


@router.post("/{flow_id}/duplicate", response_model=AgentFlowResponse)
async def duplicate_flow(
    flow_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Duplicate an agent flow"""
    result = await db.execute(
        select(AgentFlow).where(
            AgentFlow.id == flow_id,
            AgentFlow.owner_id == user.id
        )
    )
    original = result.scalar_one_or_none()
    
    if not original:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    # Create copy
    copy = AgentFlow(
        owner_id=user.id,
        name=f"{original.name} (Copy)",
        description=original.description,
        definition=original.definition,
        is_active=False,
        is_public=False,
    )
    db.add(copy)
    await db.commit()
    await db.refresh(copy)
    
    logger.info(f"Duplicated agent flow '{original.name}' as '{copy.name}' for user {user.id}")
    
    return AgentFlowResponse(
        id=copy.id,
        owner_id=copy.owner_id,
        name=copy.name,
        description=copy.description,
        definition=copy.definition or {"nodes": [], "connections": []},
        is_active=copy.is_active,
        is_public=copy.is_public,
        created_at=copy.created_at.isoformat() if copy.created_at else None,
        updated_at=copy.updated_at.isoformat() if copy.updated_at else None,
    )
