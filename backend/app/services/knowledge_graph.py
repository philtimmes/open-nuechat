"""
Knowledge Graph Service for RAG Temporal Validity

Provides:
1. Edge creation during document ingestion (LLM-based relationship detection)
2. Validity scoring and caching
3. Post-FAISS filtering based on temporal validity
4. Entity extraction and linking

NC-0.8.0.2: Knowledge Graph Filtering

Flow:
1. FAISS returns top N candidates based on semantic similarity
2. Knowledge Graph filters/reranks based on:
   - Temporal validity (is this chunk superseded?)
   - Entity relevance (does query mention specific entities?)
   - Relationship chains (follow SUPERSEDES edges to find current info)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, and_, or_, func
import json

from app.models.knowledge_graph import KnowledgeEdge, ChunkValidity, EntityMention, EdgeType
from app.models.document import DocumentChunk, Document
from app.core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Manages the knowledge graph for temporal validity filtering.
    """
    
    # Entities that are inherently time-sensitive
    TIME_SENSITIVE_ENTITY_TYPES = {
        "current_event", "news", "price", "status", "position",
        "president", "ceo", "leader", "policy", "law_current"
    }
    
    # Query patterns that indicate time-sensitive intent
    TIME_SENSITIVE_PATTERNS = [
        "current", "latest", "recent", "now", "today",
        "this week", "this month", "this year",
        "what is", "who is",  # Present tense questions
        "news", "update", "happening",
    ]
    
    def __init__(self):
        self.llm_service = None  # Lazy load to avoid circular imports
    
    def _get_llm_service(self):
        """Lazy load LLM service"""
        if self.llm_service is None:
            from app.services.llm import LLMService
            self.llm_service = LLMService()
        return self.llm_service
    
    # =========================================================================
    # QUERY INTENT CLASSIFICATION
    # =========================================================================
    
    def classify_temporal_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify whether a query has time-sensitive intent.
        
        Returns:
            {
                "is_time_sensitive": bool,
                "intent_type": "current" | "historical" | "time_bounded",
                "time_range": {"start": datetime, "end": datetime} | None,
                "confidence": float
            }
        """
        query_lower = query.lower()
        
        # Check for time-sensitive patterns
        has_time_pattern = any(p in query_lower for p in self.TIME_SENSITIVE_PATTERNS)
        
        # Check for explicit time bounds
        time_range = self._extract_time_range(query_lower)
        
        # Check for historical indicators
        historical_patterns = ["history of", "when did", "in the past", "originally", "used to"]
        is_historical = any(p in query_lower for p in historical_patterns)
        
        if time_range:
            return {
                "is_time_sensitive": True,
                "intent_type": "time_bounded",
                "time_range": time_range,
                "confidence": 0.9
            }
        elif has_time_pattern and not is_historical:
            return {
                "is_time_sensitive": True,
                "intent_type": "current",
                "time_range": None,
                "confidence": 0.8
            }
        elif is_historical:
            return {
                "is_time_sensitive": False,
                "intent_type": "historical",
                "time_range": None,
                "confidence": 0.7
            }
        else:
            return {
                "is_time_sensitive": False,
                "intent_type": "general",
                "time_range": None,
                "confidence": 0.5
            }
    
    def _extract_time_range(self, query: str) -> Optional[Dict[str, datetime]]:
        """Extract explicit time bounds from query"""
        now = datetime.now()
        
        # Simple pattern matching (could be enhanced with NLP)
        if "last week" in query:
            return {"start": now - timedelta(days=7), "end": now}
        elif "last month" in query:
            return {"start": now - timedelta(days=30), "end": now}
        elif "last year" in query:
            return {"start": now - timedelta(days=365), "end": now}
        elif "yesterday" in query:
            return {"start": now - timedelta(days=1), "end": now}
        elif "today" in query:
            return {"start": now.replace(hour=0, minute=0, second=0), "end": now}
        
        # Could add regex for "in 2024", "since January", etc.
        return None
    
    # =========================================================================
    # POST-FAISS FILTERING
    # =========================================================================
    
    async def filter_by_validity(
        self,
        db: AsyncSession,
        chunk_results: List[Dict[str, Any]],
        query: str,
        temporal_intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter FAISS results based on temporal validity.
        
        Args:
            chunk_results: Results from FAISS search [{chunk_id, similarity, ...}, ...]
            query: Original user query
            temporal_intent: Pre-classified intent (or will classify if None)
        
        Returns:
            Filtered and reranked results with validity info
        """
        if not chunk_results:
            return []
        
        # Classify intent if not provided
        if temporal_intent is None:
            temporal_intent = self.classify_temporal_intent(query)
        
        chunk_ids = [r["chunk_id"] for r in chunk_results]
        
        # Get validity status for all chunks
        validity_map = await self._get_validity_map(db, chunk_ids)
        
        filtered_results = []
        superseded_chunks = []
        
        for result in chunk_results:
            chunk_id = result["chunk_id"]
            validity = validity_map.get(chunk_id)
            
            # Add validity info to result
            result["validity"] = {
                "is_valid": validity.is_valid if validity else True,
                "validity_score": validity.validity_score if validity else 1.0,
                "superseded_by": validity.superseded_by_id if validity else None,
                "reason": validity.reason if validity else None,
            }
            
            # For time-sensitive queries, filter out invalid chunks
            if temporal_intent["is_time_sensitive"]:
                if validity and not validity.is_valid:
                    # Track superseded chunks - we might want to follow the chain
                    superseded_chunks.append((result, validity.superseded_by_id))
                    continue
                
                # Boost score based on validity
                if validity:
                    result["similarity"] *= validity.validity_score
            
            filtered_results.append(result)
        
        # For superseded chunks, try to find and include the superseding chunk
        if temporal_intent["is_time_sensitive"] and superseded_chunks:
            replacement_ids = [s[1] for s in superseded_chunks if s[1]]
            if replacement_ids:
                # Check if replacements are already in results
                existing_ids = {r["chunk_id"] for r in filtered_results}
                missing_replacements = [rid for rid in replacement_ids if rid not in existing_ids]
                
                if missing_replacements:
                    # Fetch and add replacement chunks
                    replacements = await self._fetch_replacement_chunks(db, missing_replacements)
                    for repl in replacements:
                        # Give replacement chunks a boost
                        repl["similarity"] = 0.85  # High relevance as it's the current version
                        repl["is_replacement"] = True
                        filtered_results.append(repl)
        
        # Sort by adjusted similarity
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return filtered_results
    
    async def _get_validity_map(
        self,
        db: AsyncSession,
        chunk_ids: List[str]
    ) -> Dict[str, ChunkValidity]:
        """Get validity status for multiple chunks"""
        result = await db.execute(
            select(ChunkValidity).where(ChunkValidity.chunk_id.in_(chunk_ids))
        )
        validities = result.scalars().all()
        return {v.chunk_id: v for v in validities}
    
    async def _fetch_replacement_chunks(
        self,
        db: AsyncSession,
        chunk_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch chunk details for replacement chunks"""
        result = await db.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
        rows = result.all()
        
        results = []
        for chunk, document in rows:
            results.append({
                "chunk_id": str(chunk.id),
                "content": chunk.content,
                "document_id": str(document.id),
                "document_name": document.name,
                "similarity": 0.0,  # Will be set by caller
            })
        return results
    
    # =========================================================================
    # EDGE CREATION (During Ingestion)
    # =========================================================================
    
    async def analyze_new_chunk(
        self,
        db: AsyncSession,
        new_chunk: DocumentChunk,
        existing_chunks: List[DocumentChunk],
        use_llm: bool = True,
    ) -> List[KnowledgeEdge]:
        """
        Analyze a new chunk against existing chunks to detect relationships.
        
        Called during document ingestion to build the knowledge graph.
        """
        if not existing_chunks:
            return []
        
        edges = []
        
        if use_llm:
            # Use LLM for sophisticated relationship detection
            edges = await self._llm_relationship_detection(db, new_chunk, existing_chunks)
        else:
            # Fallback to heuristic detection
            edges = await self._heuristic_relationship_detection(db, new_chunk, existing_chunks)
        
        # Save edges
        for edge in edges:
            db.add(edge)
        
        # Update validity cache for affected chunks
        await self._update_validity_cache(db, edges)
        
        return edges
    
    async def _llm_relationship_detection(
        self,
        db: AsyncSession,
        new_chunk: DocumentChunk,
        existing_chunks: List[DocumentChunk],
    ) -> List[KnowledgeEdge]:
        """Use LLM to detect relationships between chunks"""
        edges = []
        
        # Batch chunks for efficiency (compare against top N most similar)
        # In practice, you'd first do a quick similarity filter
        chunks_to_compare = existing_chunks[:20]  # Limit for performance
        
        for existing in chunks_to_compare:
            relationship = await self._detect_relationship_llm(new_chunk, existing)
            
            if relationship:
                edge = KnowledgeEdge(
                    source_chunk_id=str(new_chunk.id),
                    target_chunk_id=str(existing.id),
                    edge_type=relationship["type"],
                    confidence=relationship["confidence"],
                    entity=relationship.get("entity"),
                    reason=relationship.get("reason"),
                    is_auto=True,
                )
                edges.append(edge)
        
        return edges
    
    async def _detect_relationship_llm(
        self,
        new_chunk: DocumentChunk,
        existing_chunk: DocumentChunk,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to detect if new_chunk has a temporal relationship with existing_chunk.
        """
        llm = self._get_llm_service()
        
        prompt = f"""Analyze these two text chunks and determine if there is a temporal/validity relationship between them.

CHUNK A (NEWER):
{new_chunk.content[:1000]}

CHUNK B (OLDER):
{existing_chunk.content[:1000]}

Determine if Chunk A:
1. SUPERSEDES Chunk B (same topic but Chunk A has newer/updated information that replaces B)
2. UPDATES Chunk B (Chunk A modifies or amends information in B)
3. CONTRADICTS Chunk B (Chunk A conflicts with information in B)
4. CONFIRMS Chunk B (Chunk A validates that B is still accurate)
5. NONE - No temporal relationship

Respond in JSON format:
{{
    "relationship": "SUPERSEDES" | "UPDATES" | "CONTRADICTS" | "CONFIRMS" | "NONE",
    "confidence": 0.0-1.0,
    "entity": "topic or entity this relationship is about (if any)",
    "reason": "brief explanation"
}}

Only output the JSON, nothing else."""

        try:
            response = await llm.generate_simple(prompt, max_tokens=200)
            result = json.loads(response)
            
            if result.get("relationship") == "NONE":
                return None
            
            return {
                "type": result["relationship"].lower(),
                "confidence": result.get("confidence", 0.7),
                "entity": result.get("entity"),
                "reason": result.get("reason"),
            }
        except Exception as e:
            logger.warning(f"LLM relationship detection failed: {e}")
            return None
    
    async def _heuristic_relationship_detection(
        self,
        db: AsyncSession,
        new_chunk: DocumentChunk,
        existing_chunks: List[DocumentChunk],
    ) -> List[KnowledgeEdge]:
        """Fallback heuristic relationship detection without LLM"""
        edges = []
        
        new_content_lower = new_chunk.content.lower()
        
        # Look for update/supersession indicators
        update_indicators = [
            "update:", "updated", "new version", "replaces",
            "as of", "effective", "amendment", "revised"
        ]
        
        has_update_indicator = any(ind in new_content_lower for ind in update_indicators)
        
        if has_update_indicator:
            # Find similar existing chunks (by simple word overlap)
            for existing in existing_chunks:
                similarity = self._word_overlap_similarity(
                    new_chunk.content, existing.content
                )
                
                if similarity > 0.5:  # Significant overlap
                    edge = KnowledgeEdge(
                        source_chunk_id=str(new_chunk.id),
                        target_chunk_id=str(existing.id),
                        edge_type="supersedes",
                        confidence=similarity * 0.7,  # Lower confidence for heuristic
                        reason="Heuristic: update indicators + content overlap",
                        is_auto=True,
                    )
                    edges.append(edge)
        
        return edges
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    # =========================================================================
    # VALIDITY CACHE MANAGEMENT
    # =========================================================================
    
    async def _update_validity_cache(
        self,
        db: AsyncSession,
        new_edges: List[KnowledgeEdge],
    ) -> None:
        """Update validity cache based on new edges"""
        for edge in new_edges:
            if edge.edge_type in ["supersedes", "contradicts"]:
                # Mark target chunk as potentially invalid
                await self._mark_chunk_superseded(
                    db,
                    chunk_id=edge.target_chunk_id,
                    superseded_by=edge.source_chunk_id,
                    reason=edge.reason,
                    confidence=edge.confidence,
                )
    
    async def _mark_chunk_superseded(
        self,
        db: AsyncSession,
        chunk_id: str,
        superseded_by: str,
        reason: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        """Mark a chunk as superseded by another"""
        existing = await db.execute(
            select(ChunkValidity).where(ChunkValidity.chunk_id == chunk_id)
        )
        validity = existing.scalar_one_or_none()
        
        if validity:
            validity.is_valid = False
            validity.superseded_by_id = superseded_by
            validity.validity_score = max(0, 1 - confidence)
            validity.reason = reason
            validity.computed_at = datetime.now()
        else:
            validity = ChunkValidity(
                chunk_id=chunk_id,
                is_valid=False,
                superseded_by_id=superseded_by,
                validity_score=max(0, 1 - confidence),
                reason=reason,
            )
            db.add(validity)
    
    async def recompute_validity(
        self,
        db: AsyncSession,
        chunk_id: str,
    ) -> ChunkValidity:
        """Recompute validity for a chunk by traversing the graph"""
        # Find all edges where this chunk is the target
        result = await db.execute(
            select(KnowledgeEdge)
            .where(KnowledgeEdge.target_chunk_id == chunk_id)
            .where(KnowledgeEdge.edge_type.in_(["supersedes", "contradicts"]))
        )
        invalidating_edges = result.scalars().all()
        
        if not invalidating_edges:
            # No invalidating edges - chunk is valid
            validity = ChunkValidity(
                chunk_id=chunk_id,
                is_valid=True,
                validity_score=1.0,
                computed_at=datetime.now(),
            )
        else:
            # Find the most confident invalidation
            best_edge = max(invalidating_edges, key=lambda e: e.confidence)
            validity = ChunkValidity(
                chunk_id=chunk_id,
                is_valid=False,
                superseded_by_id=best_edge.source_chunk_id,
                validity_score=max(0, 1 - best_edge.confidence),
                reason=best_edge.reason,
                computed_at=datetime.now(),
            )
        
        # Upsert
        existing = await db.execute(
            select(ChunkValidity).where(ChunkValidity.chunk_id == chunk_id)
        )
        existing_validity = existing.scalar_one_or_none()
        
        if existing_validity:
            existing_validity.is_valid = validity.is_valid
            existing_validity.superseded_by_id = validity.superseded_by_id
            existing_validity.validity_score = validity.validity_score
            existing_validity.reason = validity.reason
            existing_validity.computed_at = validity.computed_at
            return existing_validity
        else:
            db.add(validity)
            return validity
    
    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================
    
    async def extract_entities(
        self,
        db: AsyncSession,
        chunk: DocumentChunk,
        use_llm: bool = True,
    ) -> List[EntityMention]:
        """Extract entities mentioned in a chunk"""
        if use_llm:
            return await self._extract_entities_llm(db, chunk)
        else:
            return await self._extract_entities_heuristic(db, chunk)
    
    async def _extract_entities_llm(
        self,
        db: AsyncSession,
        chunk: DocumentChunk,
    ) -> List[EntityMention]:
        """Use LLM to extract entities"""
        llm = self._get_llm_service()
        
        prompt = f"""Extract named entities from this text. Focus on entities that might change over time (people in positions, companies, laws, prices, etc.).

TEXT:
{chunk.content[:1500]}

Respond in JSON format:
{{
    "entities": [
        {{
            "id": "normalized_id (e.g., person:joe_biden, company:twitter, law:fl_790.06)",
            "type": "person|company|law|product|position|price|event",
            "mention": "exact text mentioning this entity"
        }}
    ]
}}

Only output the JSON, nothing else."""

        try:
            response = await llm.generate_simple(prompt, max_tokens=500)
            result = json.loads(response)
            
            mentions = []
            for entity in result.get("entities", []):
                mention = EntityMention(
                    chunk_id=str(chunk.id),
                    entity_id=entity["id"],
                    entity_type=entity.get("type"),
                    mention_text=entity.get("mention"),
                    confidence=0.8,
                )
                mentions.append(mention)
                db.add(mention)
            
            return mentions
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []
    
    async def _extract_entities_heuristic(
        self,
        db: AsyncSession,
        chunk: DocumentChunk,
    ) -> List[EntityMention]:
        """Simple heuristic entity extraction"""
        # Very basic - in production you'd use spaCy or similar
        mentions = []
        content = chunk.content
        
        # Look for common patterns
        import re
        
        # Statute patterns (e.g., "790.06", "Section 790.06")
        statute_pattern = r'\b(\d{3}\.\d{2,3})\b'
        for match in re.finditer(statute_pattern, content):
            mention = EntityMention(
                chunk_id=str(chunk.id),
                entity_id=f"statute:{match.group(1)}",
                entity_type="law",
                mention_text=match.group(0),
                confidence=0.7,
            )
            mentions.append(mention)
            db.add(mention)
        
        return mentions
    
    # =========================================================================
    # GRAPH QUERIES
    # =========================================================================
    
    async def get_supersession_chain(
        self,
        db: AsyncSession,
        chunk_id: str,
        max_depth: int = 5,
    ) -> List[str]:
        """
        Follow SUPERSEDES edges to find the current version of information.
        
        Returns chain of chunk IDs from oldest to newest.
        """
        chain = [chunk_id]
        current_id = chunk_id
        
        for _ in range(max_depth):
            result = await db.execute(
                select(KnowledgeEdge)
                .where(KnowledgeEdge.target_chunk_id == current_id)
                .where(KnowledgeEdge.edge_type == "supersedes")
                .order_by(KnowledgeEdge.confidence.desc())
            )
            edge = result.scalar_one_or_none()
            
            if not edge:
                break
            
            current_id = edge.source_chunk_id
            chain.append(current_id)
        
        return chain
    
    async def find_related_chunks(
        self,
        db: AsyncSession,
        chunk_id: str,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find chunks related to the given chunk.
        
        Returns: [(related_chunk_id, edge_type, confidence), ...]
        """
        query = select(KnowledgeEdge).where(
            or_(
                KnowledgeEdge.source_chunk_id == chunk_id,
                KnowledgeEdge.target_chunk_id == chunk_id,
            )
        )
        
        if edge_types:
            query = query.where(KnowledgeEdge.edge_type.in_(edge_types))
        
        result = await db.execute(query)
        edges = result.scalars().all()
        
        related = []
        for edge in edges:
            other_id = edge.target_chunk_id if edge.source_chunk_id == chunk_id else edge.source_chunk_id
            related.append((other_id, edge.edge_type, edge.confidence))
        
        return related


# Global service instance
_knowledge_graph_service: Optional[KnowledgeGraphService] = None


def get_knowledge_graph_service() -> KnowledgeGraphService:
    """Get or create the knowledge graph service singleton"""
    global _knowledge_graph_service
    if _knowledge_graph_service is None:
        _knowledge_graph_service = KnowledgeGraphService()
    return _knowledge_graph_service
