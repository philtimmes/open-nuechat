"""
Local Chat RAG Service

Creates an in-memory FAISS index for the current chat's content:
- Uploaded file attachments (PDFs, docs, text)
- AgentNNNN.md overflow files
- Provides context injection when main context overflows

This is SEPARATE from the global RAG system - it's per-chat, ephemeral,
and used specifically for context compression/overflow scenarios.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import faiss

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MIN_SCORE = 0.5  # More lenient than global RAG
CHARS_PER_TOKEN = 4


@dataclass
class LocalChunk:
    """A chunk of content from local chat sources"""
    id: str
    source: str  # filename or "attachment:{name}"
    content: str
    embedding: Optional[np.ndarray] = None


class LocalChatRAG:
    """
    Per-chat FAISS index for local content retrieval.
    
    Indexes:
    - File attachments from messages
    - AgentNNNN.md overflow files
    - Uploaded files (zip contents, etc.)
    
    Used for context injection when chat exceeds token limits.
    """
    
    def __init__(self, chat_id: str, min_score: float = DEFAULT_MIN_SCORE):
        self.chat_id = chat_id
        self.min_score = min_score
        self.chunks: List[LocalChunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 default
        self._model = None
    
    def _get_model(self):
        """Get embedding model (lazy load)"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    model_kwargs={"low_cpu_mem_usage": False},
                )
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"[LOCAL_RAG] Failed to load model: {e}")
                return None
        return self._model
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if c]
    
    def add_content(self, source: str, content: str):
        """Add content to the index"""
        if not content or len(content) < 50:
            return
        
        # Chunk the content
        text_chunks = self._chunk_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = LocalChunk(
                id=f"{source}:{i}",
                source=source,
                content=chunk_text,
            )
            self.chunks.append(chunk)
        
        logger.debug(f"[LOCAL_RAG] Added {len(text_chunks)} chunks from {source}")
    
    def add_attachments(self, messages: List[Dict]):
        """Index attachments from messages"""
        for msg in messages:
            attachments = msg.get("attachments", [])
            if not attachments:
                continue
            
            for att in attachments:
                content = att.get("content", "")
                name = att.get("filename", att.get("name", "attachment"))
                if content and len(content) > 100:
                    self.add_content(f"attachment:{name}", content)
    
    def add_agent_files(self, agent_files: List[Dict]):
        """Index AgentNNNN.md files"""
        for af in agent_files:
            content = af.get("content", "")
            filename = af.get("filename", "agent")
            if content:
                self.add_content(f"agent:{filename}", content)
    
    def add_uploaded_files(self, uploaded_files: List):
        """Index UploadedFile objects"""
        for uf in uploaded_files:
            content = getattr(uf, "content", None)
            filename = getattr(uf, "filename", "file")
            if content and not filename.startswith("{Agent"):  # Skip agent files, handled separately
                self.add_content(f"file:{filename}", content)
    
    def build_index(self) -> bool:
        """Build FAISS index from chunks"""
        if not self.chunks:
            logger.debug("[LOCAL_RAG] No chunks to index")
            return False
        
        model = self._get_model()
        if model is None:
            return False
        
        try:
            # Generate embeddings
            texts = [c.content for c in self.chunks]
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            
            # Store embeddings in chunks
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = embeddings[i]
            
            # Build FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings.astype(np.float32))
            
            logger.info(f"[LOCAL_RAG] Built index with {len(self.chunks)} chunks for chat {self.chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"[LOCAL_RAG] Failed to build index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index or not self.chunks:
            return []
        
        model = self._get_model()
        if model is None:
            return []
        
        try:
            # Embed query
            query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype(np.float32), min(top_k, len(self.chunks)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < self.min_score:
                    continue
                
                chunk = self.chunks[idx]
                results.append({
                    "source": chunk.source,
                    "content": chunk.content,
                    "score": float(score),
                    "id": chunk.id,
                })
            
            logger.debug(f"[LOCAL_RAG] Found {len(results)} results above threshold {self.min_score}")
            return results
            
        except Exception as e:
            logger.error(f"[LOCAL_RAG] Search failed: {e}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 4000,
        top_k: int = 10,
    ) -> Optional[str]:
        """
        Get relevant context for a query, formatted for injection.
        
        Used when main context overflows - searches local FAISS
        and returns relevant content.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return None
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * CHARS_PER_TOKEN
        
        for r in results:
            content = r["content"]
            source = r["source"]
            score = r["score"]
            
            if total_chars + len(content) > max_chars:
                # Truncate this chunk
                remaining = max_chars - total_chars
                if remaining > 200:
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[From {source} | Relevance: {score:.0%}]\n{content}")
            total_chars += len(content)
        
        if not context_parts:
            return None
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""
<local_context>
The following is relevant content from this chat's files and history:

{context}
</local_context>
"""


# Cache of active LocalChatRAG instances
_chat_rag_cache: Dict[str, LocalChatRAG] = {}


def get_local_chat_rag(chat_id: str, min_score: float = DEFAULT_MIN_SCORE) -> LocalChatRAG:
    """Get or create LocalChatRAG for a chat"""
    if chat_id not in _chat_rag_cache:
        _chat_rag_cache[chat_id] = LocalChatRAG(chat_id, min_score)
    return _chat_rag_cache[chat_id]


def clear_local_chat_rag(chat_id: str):
    """Clear cached LocalChatRAG for a chat"""
    if chat_id in _chat_rag_cache:
        del _chat_rag_cache[chat_id]


async def build_local_chat_index(
    db,
    chat_id: str,
    messages: List[Dict],
    min_score: float = DEFAULT_MIN_SCORE,
) -> Optional[LocalChatRAG]:
    """
    Build a local FAISS index for a chat.
    
    Indexes:
    1. Message attachments (PDFs, docs)
    2. AgentNNNN.md overflow files
    3. Uploaded files (zip contents)
    
    Returns LocalChatRAG instance or None if no content.
    """
    from app.models.models import UploadedFile
    from app.services.agent_memory import AGENT_FILE_PREFIX
    
    rag = get_local_chat_rag(chat_id, min_score)
    
    # 1. Index message attachments
    rag.add_attachments(messages)
    
    # 2. Get uploaded files including agent files
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.chat_id == chat_id)
        )
        uploaded_files = result.scalars().all()
        
        agent_files = []
        regular_files = []
        
        for uf in uploaded_files:
            if uf.filename.startswith(AGENT_FILE_PREFIX):
                agent_files.append({"filename": uf.filename, "content": uf.content})
            else:
                regular_files.append(uf)
        
        # Index agent files
        if agent_files:
            rag.add_agent_files(agent_files)
        
        # Index regular uploaded files
        if regular_files:
            rag.add_uploaded_files(regular_files)
            
    except Exception as e:
        logger.warning(f"[LOCAL_RAG] Failed to load uploaded files: {e}")
    
    # Build the index
    if rag.build_index():
        return rag
    
    return None


async def get_local_context_for_overflow(
    db,
    chat_id: str,
    query: str,
    messages: List[Dict],
    max_tokens: int = 4000,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Optional[str]:
    """
    Get relevant local context when main context overflows.
    
    This is called during context compression to inject relevant
    content from local FAISS index.
    """
    # Build index if not exists
    rag = _chat_rag_cache.get(chat_id)
    if not rag or not rag.index:
        rag = await build_local_chat_index(db, chat_id, messages, min_score)
    
    if not rag:
        return None
    
    return rag.get_context_for_query(query, max_tokens=max_tokens)
