"""
Procedural Memory API Routes

Endpoints for managing learned skills and viewing learning statistics.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.services.procedural_memory import ProceduralMemoryService
from app.models.models import User

router = APIRouter(prefix="/procedural-memory", tags=["Procedural Memory"])


# ==================== SCHEMAS ====================

class SkillResponse(BaseModel):
    id: str
    name: str
    category: Optional[str]
    description: Optional[str]
    usage_count: int
    success_rate: float
    avg_quality_score: float
    created_at: Optional[str]
    last_used_at: Optional[str]


class SkillDetailResponse(SkillResponse):
    trigger_pattern: str
    example_input: Optional[str]
    example_output: Optional[str]
    source_chat_id: Optional[str]


class LearningStatsResponse(BaseModel):
    total_skills: int
    total_usage: int
    avg_quality: float
    skills_by_category: dict
    learning_events_last_7_days: int


class FeedbackRequest(BaseModel):
    execution_id: str
    was_successful: bool
    feedback: Optional[str] = None
    feedback_text: Optional[str] = None


class LearnSkillRequest(BaseModel):
    input_text: str
    output_text: str
    category: Optional[str] = None
    name: Optional[str] = None
    chat_id: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.get("/skills", response_model=List[SkillResponse])
async def get_skills(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum skills to return"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all learned skills for the current user."""
    skills = await ProceduralMemoryService.get_user_skills(
        user_id=str(user.id),
        category=category,
        limit=limit,
    )
    return skills


@router.get("/skills/{skill_id}", response_model=SkillDetailResponse)
async def get_skill_detail(
    skill_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get detailed information about a specific skill."""
    skills = await ProceduralMemoryService.get_user_skills(
        user_id=str(user.id),
        limit=1000,
    )
    
    # Find the specific skill
    for skill in skills:
        if skill["id"] == skill_id:
            # Get full details from DB
            async with await ProceduralMemoryService.get_session() as session:
                from app.models.procedural_memory import Skill
                from sqlalchemy import select
                result = await session.execute(
                    select(Skill).where(Skill.id == skill_id).where(Skill.user_id == str(user.id))
                )
                skill_obj = result.scalar_one_or_none()
                
                if skill_obj:
                    return {
                        "id": skill_obj.id,
                        "name": skill_obj.name,
                        "category": skill_obj.category,
                        "description": skill_obj.description,
                        "usage_count": skill_obj.usage_count,
                        "success_rate": skill_obj.success_count / max(skill_obj.usage_count, 1),
                        "avg_quality_score": skill_obj.avg_quality_score,
                        "created_at": skill_obj.created_at.isoformat() if skill_obj.created_at else None,
                        "last_used_at": skill_obj.last_used_at.isoformat() if skill_obj.last_used_at else None,
                        "trigger_pattern": skill_obj.trigger_pattern,
                        "example_input": skill_obj.example_input,
                        "example_output": skill_obj.example_output,
                        "source_chat_id": skill_obj.source_chat_id,
                    }
    
    raise HTTPException(status_code=404, detail="Skill not found")


@router.delete("/skills/{skill_id}")
async def delete_skill(
    skill_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete (deactivate) a learned skill."""
    success = await ProceduralMemoryService.delete_skill(
        skill_id=skill_id,
        user_id=str(user.id),
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Skill not found or already deleted")
    
    return {"status": "deleted", "skill_id": skill_id}


@router.get("/stats", response_model=LearningStatsResponse)
async def get_learning_stats(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get learning statistics for the current user."""
    stats = await ProceduralMemoryService.get_learning_stats(str(user.id))
    return stats


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Submit feedback on a skill execution."""
    await ProceduralMemoryService.record_feedback(
        execution_id=feedback.execution_id,
        was_successful=feedback.was_successful,
        feedback=feedback.feedback,
        feedback_text=feedback.feedback_text,
    )
    return {"status": "recorded"}


@router.post("/learn", response_model=SkillResponse)
async def learn_skill_manually(
    request: LearnSkillRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Manually learn a skill from a successful interaction."""
    skill = await ProceduralMemoryService.learn_skill_from_interaction(
        user_id=str(user.id),
        input_text=request.input_text,
        output_text=request.output_text,
        chat_id=request.chat_id,
        category=request.category,
        quality_score=1.0,  # Manual learning is assumed high quality
    )
    
    if not skill:
        raise HTTPException(status_code=500, detail="Failed to learn skill")
    
    return {
        "id": skill.id,
        "name": skill.name,
        "category": skill.category,
        "description": skill.description,
        "usage_count": skill.usage_count,
        "success_rate": skill.success_count / max(skill.usage_count, 1),
        "avg_quality_score": skill.avg_quality_score,
        "created_at": skill.created_at.isoformat() if skill.created_at else None,
        "last_used_at": skill.last_used_at.isoformat() if skill.last_used_at else None,
    }


@router.get("/search")
async def search_skills(
    query: str = Query(..., min_length=3, description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Search for relevant skills based on a query."""
    skills = await ProceduralMemoryService.retrieve_relevant_skills(
        user_id=str(user.id),
        query=query,
        top_k=top_k,
        category=category,
    )
    return {"skills": skills, "query": query}


@router.get("/categories")
async def get_skill_categories(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all skill categories for the user."""
    stats = await ProceduralMemoryService.get_learning_stats(str(user.id))
    return {
        "categories": list(stats.get("skills_by_category", {}).keys()),
        "counts": stats.get("skills_by_category", {}),
    }
