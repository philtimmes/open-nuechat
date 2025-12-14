"""
Procedural Memory Service

Implements a skill learning and retrieval system inspired by human procedural memory.
Skills are learned from successful interactions and retrieved for similar future tasks.

Key Features:
- Skill extraction from successful completions
- Embedding-based similarity search for skill retrieval
- Usage tracking and performance metrics
- Skill composition for complex tasks
- Automatic skill updates based on feedback

This service uses a separate database from the main application.
"""
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, select, update, func, and_, or_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from app.models.procedural_memory import (
    ProceduralBase, Skill, SkillExecution, SkillComposition, LearningEvent
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations (embeddings)
_executor = ThreadPoolExecutor(max_workers=2)

# Separate database URL for procedural memory
def _get_procedural_db_url() -> str:
    """Get the procedural memory database URL."""
    if settings.PROCEDURAL_DATABASE_URL:
        return settings.PROCEDURAL_DATABASE_URL
    
    # Derive from main DATABASE_URL
    main_url = settings.DATABASE_URL
    if "sqlite" in main_url:
        # For SQLite, use a separate file
        return main_url.replace("nuechat.db", "procedural_memory.db")
    else:
        # For PostgreSQL/MySQL, use a different database name
        return main_url.replace("/nuechat", "/procedural_memory")

PROCEDURAL_DB_URL = _get_procedural_db_url()

# Convert to async URL
if PROCEDURAL_DB_URL.startswith("postgresql://"):
    ASYNC_PROCEDURAL_DB_URL = PROCEDURAL_DB_URL.replace("postgresql://", "postgresql+asyncpg://")
elif PROCEDURAL_DB_URL.startswith("sqlite://"):
    ASYNC_PROCEDURAL_DB_URL = PROCEDURAL_DB_URL.replace("sqlite://", "sqlite+aiosqlite://")
elif "sqlite+aiosqlite" in PROCEDURAL_DB_URL:
    ASYNC_PROCEDURAL_DB_URL = PROCEDURAL_DB_URL
else:
    ASYNC_PROCEDURAL_DB_URL = PROCEDURAL_DB_URL


class ProceduralMemoryService:
    """
    Service for managing procedural memory - learning and retrieving skills.
    
    Skills are extracted from successful interactions and stored with embeddings
    for similarity-based retrieval. When a new query comes in, relevant skills
    are retrieved to augment the LLM context.
    """
    
    # Configuration
    SIMILARITY_THRESHOLD = 0.7  # Minimum similarity to consider a skill relevant
    MAX_SKILLS_PER_QUERY = 3  # Maximum skills to retrieve per query
    MIN_SUCCESS_RATE = 0.5  # Minimum success rate to use a skill
    SKILL_DECAY_DAYS = 90  # Skills not used in this many days get lower priority
    
    _engine = None
    _session_factory = None
    _embedding_model = None
    _initialized = False
    
    @classmethod
    async def initialize(cls):
        """Initialize the procedural memory database and embedding model."""
        if cls._initialized:
            return
        
        try:
            # Create async engine for procedural memory database
            cls._engine = create_async_engine(
                ASYNC_PROCEDURAL_DB_URL,
                echo=False,
                pool_pre_ping=True,
            )
            
            cls._session_factory = async_sessionmaker(
                cls._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Create tables
            async with cls._engine.begin() as conn:
                await conn.run_sync(ProceduralBase.metadata.create_all)
            
            cls._initialized = True
            logger.info("Procedural Memory Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Procedural Memory Service: {e}")
            raise
    
    @classmethod
    def get_embedding_model(cls):
        """Lazy load the embedding model."""
        if cls._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model for procedural memory")
            except ImportError:
                logger.warning("sentence-transformers not available, skill retrieval will use keyword matching")
        return cls._embedding_model
    
    @classmethod
    def embed_text(cls, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        model = cls.get_embedding_model()
        if model is None:
            return None
        
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    @classmethod
    async def get_session(cls) -> AsyncSession:
        """Get a database session."""
        if not cls._initialized:
            await cls.initialize()
        return cls._session_factory()
    
    # ==================== SKILL LEARNING ====================
    
    @classmethod
    async def learn_skill_from_interaction(
        cls,
        user_id: str,
        input_text: str,
        output_text: str,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        model_used: Optional[str] = None,
        category: Optional[str] = None,
        quality_score: float = 1.0,
    ) -> Optional[Skill]:
        """
        Learn a new skill from a successful interaction.
        
        This extracts the pattern from the interaction and stores it
        as a reusable skill.
        """
        async with await cls.get_session() as db:
            try:
                # Check if a similar skill already exists
                existing_skill = await cls._find_similar_skill(db, user_id, input_text)
                
                if existing_skill:
                    # Update existing skill with new data
                    await cls._update_skill(db, existing_skill, output_text, quality_score)
                    return existing_skill
                
                # Extract skill metadata
                skill_name = await cls._generate_skill_name(input_text)
                trigger_pattern = cls._extract_trigger_pattern(input_text)
                action_sequence = cls._extract_action_sequence(input_text, output_text)
                
                # Generate embedding
                embedding = await asyncio.get_event_loop().run_in_executor(
                    _executor, cls.embed_text, trigger_pattern
                )
                
                # Create new skill
                skill = Skill(
                    user_id=user_id,
                    name=skill_name,
                    description=f"Learned from successful interaction",
                    category=category or cls._infer_category(input_text),
                    trigger_pattern=trigger_pattern,
                    action_sequence=action_sequence,
                    example_input=input_text[:2000],  # Truncate if too long
                    example_output=output_text[:2000],
                    embedding=embedding.tobytes() if embedding is not None else None,
                    source_chat_id=chat_id,
                    source_message_id=message_id,
                    model_used=model_used,
                    avg_quality_score=quality_score,
                )
                
                db.add(skill)
                
                # Record learning event
                event = LearningEvent(
                    user_id=user_id,
                    event_type="skill_created",
                    skill_id=skill.id,
                    trigger_source="successful_completion",
                    source_chat_id=chat_id,
                    details={
                        "skill_name": skill_name,
                        "category": skill.category,
                        "model_used": model_used,
                    }
                )
                db.add(event)
                
                await db.commit()
                await db.refresh(skill)
                
                logger.info(f"Learned new skill: {skill_name} for user {user_id}")
                return skill
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to learn skill: {e}")
                return None
    
    @classmethod
    async def _find_similar_skill(
        cls,
        db: AsyncSession,
        user_id: str,
        input_text: str,
        threshold: float = 0.85,
    ) -> Optional[Skill]:
        """Find an existing skill that's very similar to the input."""
        # Generate embedding for input
        embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, cls.embed_text, input_text
        )
        
        if embedding is None:
            return None
        
        # Get user's skills
        result = await db.execute(
            select(Skill)
            .where(Skill.user_id == user_id)
            .where(Skill.is_active == True)
            .where(Skill.embedding.isnot(None))
        )
        skills = result.scalars().all()
        
        # Find most similar
        best_skill = None
        best_similarity = 0
        
        for skill in skills:
            skill_embedding = np.frombuffer(skill.embedding, dtype=np.float32)
            similarity = float(np.dot(embedding, skill_embedding))
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_skill = skill
        
        return best_skill
    
    @classmethod
    async def _update_skill(
        cls,
        db: AsyncSession,
        skill: Skill,
        new_output: str,
        quality_score: float,
    ):
        """Update an existing skill with new data."""
        # Update metrics using exponential moving average
        alpha = 0.3  # Weight for new data
        skill.avg_quality_score = (1 - alpha) * skill.avg_quality_score + alpha * quality_score
        skill.usage_count += 1
        skill.success_count += 1 if quality_score >= 0.5 else 0
        skill.updated_at = datetime.utcnow()
        
        # Record learning event
        event = LearningEvent(
            user_id=skill.user_id,
            event_type="skill_updated",
            skill_id=skill.id,
            trigger_source="successful_completion",
            details={"new_quality_score": quality_score}
        )
        db.add(event)
        
        await db.commit()
    
    @classmethod
    async def _generate_skill_name(cls, input_text: str) -> str:
        """Generate a descriptive name for the skill."""
        # Simple heuristic - extract key words
        words = input_text.lower().split()[:10]
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                      'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'me',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'please'}
        
        key_words = [w for w in words if w not in stop_words and len(w) > 2][:5]
        
        if key_words:
            return "_".join(key_words)
        return f"skill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    @classmethod
    def _extract_trigger_pattern(cls, input_text: str) -> str:
        """Extract the trigger pattern from input text."""
        # For now, use the input as-is (truncated)
        # In a more sophisticated system, this would extract
        # the semantic "intent" of the request
        return input_text[:500]
    
    @classmethod
    def _extract_action_sequence(cls, input_text: str, output_text: str) -> Dict[str, Any]:
        """Extract the action sequence from the interaction."""
        return {
            "type": "direct_response",
            "input_pattern": input_text[:200],
            "response_template": output_text[:500],
            "steps": [
                {"action": "understand_request", "description": "Parse and understand the user request"},
                {"action": "generate_response", "description": "Generate appropriate response"},
            ]
        }
    
    @classmethod
    def _infer_category(cls, input_text: str) -> str:
        """Infer the category of the skill from input text."""
        text_lower = input_text.lower()
        
        if any(kw in text_lower for kw in ['code', 'function', 'class', 'program', 'script', 'debug', 'error', 'bug']):
            return "code_generation"
        elif any(kw in text_lower for kw in ['write', 'essay', 'article', 'blog', 'story', 'poem']):
            return "writing"
        elif any(kw in text_lower for kw in ['analyze', 'analysis', 'compare', 'evaluate', 'assess']):
            return "analysis"
        elif any(kw in text_lower for kw in ['explain', 'what is', 'how does', 'why', 'describe']):
            return "explanation"
        elif any(kw in text_lower for kw in ['summarize', 'summary', 'tldr', 'brief']):
            return "summarization"
        elif any(kw in text_lower for kw in ['translate', 'translation']):
            return "translation"
        elif any(kw in text_lower for kw in ['math', 'calculate', 'equation', 'formula']):
            return "mathematics"
        else:
            return "general"
    
    # ==================== SKILL RETRIEVAL ====================
    
    @classmethod
    async def retrieve_relevant_skills(
        cls,
        user_id: str,
        query: str,
        top_k: int = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve skills relevant to the given query.
        
        Returns skills sorted by relevance with metadata for
        use in the LLM prompt.
        """
        top_k = top_k or cls.MAX_SKILLS_PER_QUERY
        
        async with await cls.get_session() as db:
            try:
                # Generate query embedding
                query_embedding = await asyncio.get_event_loop().run_in_executor(
                    _executor, cls.embed_text, query
                )
                
                # Build query
                skill_query = (
                    select(Skill)
                    .where(Skill.user_id == user_id)
                    .where(Skill.is_active == True)
                )
                
                if category:
                    skill_query = skill_query.where(Skill.category == category)
                
                result = await db.execute(skill_query)
                skills = result.scalars().all()
                
                if not skills:
                    return []
                
                # Score and rank skills
                scored_skills = []
                
                for skill in skills:
                    score = await cls._score_skill(skill, query, query_embedding)
                    
                    if score >= cls.SIMILARITY_THRESHOLD:
                        scored_skills.append({
                            "skill": skill,
                            "relevance_score": score,
                        })
                
                # Sort by score
                scored_skills.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                # Take top k
                top_skills = scored_skills[:top_k]
                
                # Format for return
                result = []
                for item in top_skills:
                    skill = item["skill"]
                    result.append({
                        "id": skill.id,
                        "name": skill.name,
                        "category": skill.category,
                        "trigger_pattern": skill.trigger_pattern,
                        "action_sequence": skill.action_sequence,
                        "example_input": skill.example_input,
                        "example_output": skill.example_output,
                        "relevance_score": item["relevance_score"],
                        "success_rate": skill.success_count / max(skill.usage_count, 1),
                        "usage_count": skill.usage_count,
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to retrieve skills: {e}")
                return []
    
    @classmethod
    async def _score_skill(
        cls,
        skill: Skill,
        query: str,
        query_embedding: Optional[np.ndarray],
    ) -> float:
        """Score a skill's relevance to a query."""
        score = 0.0
        
        # Embedding similarity (primary factor)
        if query_embedding is not None and skill.embedding:
            skill_embedding = np.frombuffer(skill.embedding, dtype=np.float32)
            embedding_sim = float(np.dot(query_embedding, skill_embedding))
            score += embedding_sim * 0.6  # 60% weight
        
        # Keyword overlap (secondary factor)
        query_words = set(query.lower().split())
        trigger_words = set(skill.trigger_pattern.lower().split())
        overlap = len(query_words & trigger_words) / max(len(query_words | trigger_words), 1)
        score += overlap * 0.2  # 20% weight
        
        # Performance metrics (tertiary factor)
        success_rate = skill.success_count / max(skill.usage_count, 1)
        score += success_rate * 0.1  # 10% weight
        
        # Recency bonus
        if skill.last_used_at:
            days_since_use = (datetime.utcnow() - skill.last_used_at).days
            recency_factor = max(0, 1 - days_since_use / cls.SKILL_DECAY_DAYS)
            score += recency_factor * 0.1  # 10% weight
        else:
            score += 0.05  # Small bonus for never-used skills
        
        return score
    
    # ==================== SKILL EXECUTION ====================
    
    @classmethod
    async def record_skill_execution(
        cls,
        skill_id: str,
        user_id: str,
        input_text: str,
        chat_id: Optional[str] = None,
        similarity_score: Optional[float] = None,
        was_successful: Optional[bool] = None,
        quality_score: Optional[float] = None,
        execution_time_ms: Optional[int] = None,
    ) -> Optional[SkillExecution]:
        """Record that a skill was executed/applied."""
        async with await cls.get_session() as db:
            try:
                # Create execution record
                execution = SkillExecution(
                    skill_id=skill_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    input_text=input_text[:2000],
                    similarity_score=similarity_score,
                    was_successful=was_successful,
                    quality_score=quality_score,
                    execution_time_ms=execution_time_ms,
                )
                db.add(execution)
                
                # Update skill metrics
                await db.execute(
                    update(Skill)
                    .where(Skill.id == skill_id)
                    .values(
                        usage_count=Skill.usage_count + 1,
                        last_used_at=datetime.utcnow(),
                    )
                )
                
                await db.commit()
                return execution
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to record skill execution: {e}")
                return None
    
    @classmethod
    async def record_feedback(
        cls,
        execution_id: str,
        was_successful: bool,
        feedback: Optional[str] = None,
        feedback_text: Optional[str] = None,
    ):
        """Record user feedback on a skill execution."""
        async with await cls.get_session() as db:
            try:
                # Update execution
                result = await db.execute(
                    select(SkillExecution).where(SkillExecution.id == execution_id)
                )
                execution = result.scalar_one_or_none()
                
                if not execution:
                    return
                
                execution.was_successful = was_successful
                execution.user_feedback = feedback
                execution.feedback_text = feedback_text
                
                # Update skill metrics based on feedback
                quality_delta = 0.1 if was_successful else -0.1
                
                await db.execute(
                    update(Skill)
                    .where(Skill.id == execution.skill_id)
                    .values(
                        success_count=Skill.success_count + (1 if was_successful else 0),
                        failure_count=Skill.failure_count + (0 if was_successful else 1),
                    )
                )
                
                await db.commit()
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to record feedback: {e}")
    
    # ==================== CONTEXT GENERATION ====================
    
    @classmethod
    async def get_skill_context_for_prompt(
        cls,
        user_id: str,
        query: str,
        max_skills: int = 3,
    ) -> str:
        """
        Generate skill context to inject into the LLM prompt.
        
        This is the main integration point with the LLM flow.
        Returns formatted text describing relevant skills.
        """
        skills = await cls.retrieve_relevant_skills(user_id, query, top_k=max_skills)
        
        if not skills:
            return ""
        
        context_parts = ["[PROCEDURAL MEMORY - Relevant learned skills]"]
        
        for i, skill in enumerate(skills, 1):
            context_parts.append(f"\n--- Skill {i}: {skill['name']} (category: {skill['category']}) ---")
            context_parts.append(f"Pattern: {skill['trigger_pattern'][:200]}")
            
            if skill.get('example_output'):
                context_parts.append(f"Previous successful response approach: {skill['example_output'][:300]}...")
            
            context_parts.append(f"Success rate: {skill['success_rate']:.0%} ({skill['usage_count']} uses)")
        
        context_parts.append("\n[END PROCEDURAL MEMORY]")
        
        return "\n".join(context_parts)
    
    # ==================== ADMIN FUNCTIONS ====================
    
    @classmethod
    async def get_user_skills(
        cls,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get all skills for a user."""
        async with await cls.get_session() as db:
            query = (
                select(Skill)
                .where(Skill.user_id == user_id)
                .where(Skill.is_active == True)
                .order_by(Skill.usage_count.desc())
                .limit(limit)
            )
            
            if category:
                query = query.where(Skill.category == category)
            
            result = await db.execute(query)
            skills = result.scalars().all()
            
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "category": s.category,
                    "description": s.description,
                    "usage_count": s.usage_count,
                    "success_rate": s.success_count / max(s.usage_count, 1),
                    "avg_quality_score": s.avg_quality_score,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "last_used_at": s.last_used_at.isoformat() if s.last_used_at else None,
                }
                for s in skills
            ]
    
    @classmethod
    async def delete_skill(cls, skill_id: str, user_id: str) -> bool:
        """Soft delete a skill."""
        async with await cls.get_session() as db:
            try:
                await db.execute(
                    update(Skill)
                    .where(Skill.id == skill_id)
                    .where(Skill.user_id == user_id)
                    .values(is_active=False)
                )
                
                # Record deletion
                event = LearningEvent(
                    user_id=user_id,
                    event_type="skill_deprecated",
                    skill_id=skill_id,
                    trigger_source="user_action",
                )
                db.add(event)
                
                await db.commit()
                return True
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to delete skill: {e}")
                return False
    
    @classmethod
    async def get_learning_stats(cls, user_id: str) -> Dict[str, Any]:
        """Get learning statistics for a user."""
        async with await cls.get_session() as db:
            # Count skills by category
            result = await db.execute(
                select(Skill.category, func.count(Skill.id))
                .where(Skill.user_id == user_id)
                .where(Skill.is_active == True)
                .group_by(Skill.category)
            )
            category_counts = dict(result.all())
            
            # Total stats
            total_result = await db.execute(
                select(
                    func.count(Skill.id),
                    func.sum(Skill.usage_count),
                    func.avg(Skill.avg_quality_score),
                )
                .where(Skill.user_id == user_id)
                .where(Skill.is_active == True)
            )
            total_stats = total_result.first()
            
            # Recent learning events
            events_result = await db.execute(
                select(func.count(LearningEvent.id))
                .where(LearningEvent.user_id == user_id)
                .where(LearningEvent.created_at >= datetime.utcnow() - timedelta(days=7))
            )
            recent_events = events_result.scalar() or 0
            
            return {
                "total_skills": total_stats[0] or 0,
                "total_usage": total_stats[1] or 0,
                "avg_quality": round(total_stats[2] or 0, 2),
                "skills_by_category": category_counts,
                "learning_events_last_7_days": recent_events,
            }


# Convenience function for integration
async def get_procedural_context(user_id: str, query: str) -> str:
    """Get procedural memory context for a query."""
    return await ProceduralMemoryService.get_skill_context_for_prompt(user_id, query)
